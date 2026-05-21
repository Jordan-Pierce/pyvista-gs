import sys
import os
import argparse
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
import vtk
import OpenGL.GL as gl

from PyQt5.QtWidgets import QApplication, QMainWindow, QDockWidget, QAction, QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QEvent, QTimer
from PyQt5.QtGui import QFont, QSurfaceFormat

# ── Must be set BEFORE QApplication ──────────────────────────────────────── #
def _configure_surface_format():
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(0)           
    fmt.setSwapInterval(0)      
    QSurfaceFormat.setDefaultFormat(fmt)

_configure_surface_format()
# ──────────────────────────────────────────────────────────────────────────── #

import util_gau
from renderer_ogl import OpenGLRenderer
from control_panel import ControlPanel
from main import STYLESHEET, _build_dark_palette


class VTKCameraAdapter:
    """Bridges PyVista's vtkCamera and your util.Camera format."""
    def __init__(self, vtk_cam, width, height):
        self.vtk_cam = vtk_cam
        self.w = width
        self.h = height
        self.position = np.array(vtk_cam.GetPosition(), dtype=np.float32)

    def _vtk_to_numpy(self, vtk_matrix):
        m = np.zeros((4, 4), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                m[i, j] = vtk_matrix.GetElement(i, j)
        return m

    def get_view_matrix(self):
        mat = self.vtk_cam.GetModelViewTransformObject().GetMatrix()
        return self._vtk_to_numpy(mat)

    def get_project_matrix(self):
        aspect = self.w / self.h if self.h != 0 else 1.0
        mat = self.vtk_cam.GetProjectionTransformMatrix(aspect, -1, 1)
        return self._vtk_to_numpy(mat)

    def get_htanfovxy_focal(self):
        fovy = np.radians(self.vtk_cam.GetViewAngle())
        htany = np.tan(fovy / 2.0)
        htanx = htany / self.h * self.w
        focal = self.h / (2.0 * htany)
        return [htanx, htany, focal]

class MainWindow(QMainWindow):
    # ── Signals expected by ControlPanel ── #
    sig_fps_changed = pyqtSignal(float)
    sig_gau_count_changed = pyqtSignal(int)
    sig_status_message = pyqtSignal(str)
    sig_loading_changed = pyqtSignal(bool)
    sig_focal_changed = pyqtSignal(float, float, float)

    def __init__(self, hidpi: bool = False):
        super().__init__()
        self.setWindowTitle("Gaussian Splatting Viewer (PyVista Edition)")
        self.resize(1440, 860)

        # UI State matching GaussianWidget expectations
        self._scale_modifier = 1.0
        self._render_mode = 7 # Default SH:0~3
        self._auto_sort = True
        self._reduce_updates = True

        # PyVista Setup
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#13161f")
        self.setCentralWidget(self.plotter)

        # Gaussian State
        self.ogl_renderer = None
        self.gaussians = None
        self.dummy_mesh = None

        # Hook VTK Render
        self.plotter.renderer.AddObserver(vtk.vtkCommand.EndEvent, self.on_render_end)
        
        # Use Qt's Event Filter to catch double clicks reliably
        self.plotter.interactor.installEventFilter(self)

        # Control Panel setup
        self._panel = ControlPanel(self, self)
        dock = QDockWidget("Controls", self)
        dock.setWidget(self._panel)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        # Menu & Status
        menu = self.menuBar()
        view_menu = menu.addMenu("View")
        toggle_panel = QAction("Show Control Panel", self, checkable=True, checked=True)
        toggle_panel.triggered.connect(dock.setVisible)
        view_menu.addAction(toggle_panel)

        self._status_bar = self.statusBar()
        self._status_lbl = QLabel("Ready - Please open a .ply file")
        self._status_bar.addPermanentWidget(self._status_lbl)
        
        self.sig_status_message.connect(self._status_lbl.setText)

        if hidpi:
            QApplication.instance().setFont(QFont(QApplication.font().family(), 14))


    # ─── Mouse Interaction (Qt Event Filter) ──────────────────────────── #

    def eventFilter(self, source, event):
        # Intercept double clicks specifically on the VTK interactor widget
        if source is self.plotter.interactor and event.type() == QEvent.MouseButtonDblClick:
            if event.button() == Qt.LeftButton:
                self._handle_double_click(event.pos())
                return True # Consume event so pyvistaqt doesn't override it
        return super().eventFilter(source, event)
    
    def _handle_double_click(self, pos):
        if self.dummy_mesh is None:
            self.sig_status_message.emit("Please load a .ply file first.")
            return

        # Convert Qt's top-left coordinates to VTK's bottom-left coordinates
        x = pos.x()
        y = self.plotter.interactor.height() - pos.y()
        
        # Raycast into the scene
        picker = vtk.vtkPointPicker()
        picker.SetTolerance(0.015) # ~1.5% of the screen width/height, makes it easy to hit
        picker.Pick(x, y, 0, self.plotter.renderer)
        
        # Check if we hit the invisible dummy mesh
        if picker.GetActor() is not None:
            pick_pos = picker.GetPickPosition()
            self.set_focal_point(pick_pos[0], pick_pos[1], pick_pos[2])
            
            # Emit signals to update the UI spinners and status text
            self.sig_focal_changed.emit(pick_pos[0], pick_pos[1], pick_pos[2])
            self.sig_status_message.emit(f"Focal point set to ({pick_pos[0]:.2f}, {pick_pos[1]:.2f}, {pick_pos[2]:.2f})")
        else:
            self.sig_status_message.emit("Missed Gaussians. Try clicking closer to the object.")


    # ─── Control Panel Interface (Exact match for main.py) ────────────── #

    def gaussian_count(self):
        return len(self.gaussians) if self.gaussians else 0

    @property
    def scale_modifier(self): return self._scale_modifier

    def set_scale_modifier(self, modifier: float):
        self._scale_modifier = modifier
        if self.ogl_renderer:
            self.ogl_renderer.set_scale_modifier(modifier)
            self.plotter.update()

    @property
    def render_mode(self): return self._render_mode

    def set_render_mode(self, mod: int):
        self._render_mode = mod
        if self.ogl_renderer:
            self.ogl_renderer.set_render_mod(mod)
            self.plotter.update()

    @property
    def auto_sort(self): return self._auto_sort

    @auto_sort.setter
    def auto_sort(self, val: bool):
        self._auto_sort = val

    def set_reduce_updates(self, val: bool):
        self._reduce_updates = val
        if self.ogl_renderer:
            self.ogl_renderer.reduce_updates = val

    def backend_names(self):
        return ["OpenGL (PyVista Bridge)"]

    def current_backend_idx(self):
        return 0

    def set_backend(self, idx: int):
        pass # Stubbed to prevent crashes; backend fixed to PyVista

    def fovy_deg(self):
        return self.plotter.camera.view_angle

    def set_fovy_deg(self, deg: float):
        self.plotter.camera.view_angle = deg
        self.plotter.update()

    def focal_point(self):
        return self.plotter.camera.focal_point

    def set_focal_point(self, x: float, y: float, z: float):
        self.plotter.camera.focal_point = (x, y, z)
        self.plotter.update()

    def fit_camera_to_gaussians(self):
        self.plotter.reset_camera()

    def reset_camera(self):
        self.plotter.reset_camera()

    def flip_ground(self):
        cam = self.plotter.camera
        cam.up = (cam.up[0], -cam.up[1], -cam.up[2])
        self.plotter.update()

    def sort_gaussians(self):
        self.plotter.update() # Triggers the render hook which sorts

    def save_image(self):
        import time
        filename = f"viewport_{int(time.time())}.png"
        self.plotter.screenshot(filename)
        self.sig_status_message.emit(f"Saved {filename}")

    def load_ply(self, path: str):
        if not os.path.exists(path):
            self.sig_status_message.emit(f"File not found: {path}")
            return

        self.sig_loading_changed.emit(True)
        self.sig_status_message.emit(f"Loading {os.path.basename(path)}...")
        QApplication.processEvents() # Force UI refresh

        try:
            self.gaussians = util_gau.load_ply(path)
            
            # Center it
            centroid = np.mean(self.gaussians.xyz, axis=0)
            self.gaussians.xyz -= centroid

            # Update Dummy Mesh for PyVista interactions
            if self.dummy_mesh is None:
                self.dummy_mesh = pv.PolyData(self.gaussians.xyz)
                # Ensure the dummy points have mathematical "size" so they are easy to pick
                self.plotter.add_mesh(self.dummy_mesh, opacity=0.0, pickable=True, point_size=5.0)
            else:
                self.dummy_mesh.points = self.gaussians.xyz

            # Update OpenGL Renderer if initialized
            if self.ogl_renderer:
                self.ogl_renderer.update_gaussian_data(self.gaussians)

            self.plotter.reset_camera()
            self.sig_gau_count_changed.emit(self.gaussian_count())
            self.sig_status_message.emit("Ready")
            self.plotter.update()
        except Exception as e:
            self.sig_status_message.emit(f"Error loading PLY: {e}")
        finally:
            self.sig_loading_changed.emit(False)


    # ─── VTK/OpenGL Render Hook ───────────────────────────────────────── #

    def on_render_end(self, caller, event):
        if self.gaussians is None:
            return

        window = caller.GetRenderWindow()
        w, h = window.GetSize()
        
        # Initialize Gaussian renderer on first valid frame
        if self.ogl_renderer is None:
            self.ogl_renderer = OpenGLRenderer(w, h)
            self.ogl_renderer.update_gaussian_data(self.gaussians)
            self.ogl_renderer.set_scale_modifier(self._scale_modifier)
            self.ogl_renderer.set_render_mod(self._render_mode)
            self.ogl_renderer.reduce_updates = self._reduce_updates
        
        self.ogl_renderer.set_render_reso(w, h)
        
        vtk_cam = caller.GetActiveCamera()
        cam_adapter = VTKCameraAdapter(vtk_cam, w, h)
        
        # Determine if sorting is needed
        if self._auto_sort:
            self.ogl_renderer.sort_and_update(cam_adapter)
        
        # --- PRESERVE VTK STATE ---
        last_prog = gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)
        last_vao = gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING)
        last_blend = gl.glGetBooleanv(gl.GL_BLEND)
        last_depth_mask = gl.glGetBooleanv(gl.GL_DEPTH_WRITEMASK)
        
        # --- CONFIGURE 3DGS STATE ---
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDepthMask(gl.GL_FALSE) 
        
        # --- DRAW GAUSSIANS ---
        self.ogl_renderer.update_camera_pose(cam_adapter)
        self.ogl_renderer.update_camera_intrin(cam_adapter)
        self.ogl_renderer.draw()
        
        # --- RESTORE VTK STATE ---
        if last_depth_mask: gl.glDepthMask(gl.GL_TRUE)
        else: gl.glDepthMask(gl.GL_FALSE)
            
        if not last_blend: gl.glDisable(gl.GL_BLEND)
            
        gl.glUseProgram(last_prog)
        gl.glBindVertexArray(last_vao)
        
def main():
    parser = argparse.ArgumentParser(description="Gaussian Viewer — PyVista Edition")
    parser.add_argument("--hidpi", action="store_true", help="Enable HiDPI font scaling")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_build_dark_palette())
    app.setStyleSheet(STYLESHEET)

    win = MainWindow(hidpi=args.hidpi)
    win.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()