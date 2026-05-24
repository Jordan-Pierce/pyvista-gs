import sys
import os
import argparse
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
import vtk

from PyQt5.QtWidgets import QApplication, QMainWindow, QDockWidget, QAction, QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QEvent
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
from gaussian_actor import GaussianActor, VTKCameraAdapter
from control_panel import ControlPanel
from main import STYLESHEET, _build_dark_palette


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

        # PyVista Setup
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#13161f")
        self.setCentralWidget(self.plotter)

        # 3DGS Actor Reference
        self.gs_actor: GaussianActor | None = None

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

        # Tools Menu for Cropping
        tools_menu = menu.addMenu("Tools")
        self.action_crop = QAction("Interactive Crop Box", self, checkable=True)
        self.action_crop.triggered.connect(self._toggle_crop)
        tools_menu.addAction(self.action_crop)

        self._status_bar = self.statusBar()
        self._status_lbl = QLabel("Ready - Please open a .ply file")
        self._status_bar.addPermanentWidget(self._status_lbl)
        
        self.sig_status_message.connect(self._status_lbl.setText)

        if hidpi:
            QApplication.instance().setFont(QFont(QApplication.font().family(), 14))

    # ─── Tool Interactions ────────────────────────────────────────────── #

    def _toggle_crop(self, state: bool):
        if not self.gs_actor:
            self.action_crop.setChecked(False)
            self.sig_status_message.emit("Please load a model first.")
            return

        if state:
            self.plotter.add_box_widget(
                callback=self._on_crop_box_modified,
                bounds=self.gs_actor._original_mesh.bounds,
                color="white",
                outline_translation=False,
                pass_widget=False
            )
            self.sig_status_message.emit("Cropping enabled. Drag the box faces to slice.")
        else:
            self.plotter.clear_box_widgets()
            self.gs_actor.mesh = self.gs_actor._original_mesh # Restore original mesh
            self.plotter.update()
            self.sig_gau_count_changed.emit(self.gaussian_count())
            self.sig_status_message.emit("Cropping disabled. Restored original volume.")

    def _on_crop_box_modified(self, bounds):
        if self.gs_actor:
            self.gs_actor.apply_crop_box(bounds)
            self.plotter.update()
            self.sig_gau_count_changed.emit(self.gaussian_count())

    # ─── Core Loading ─────────────────────────────────────────────────── #

    def load_ply(self, path: str):
        if not os.path.exists(path):
            self.sig_status_message.emit(f"File not found: {path}")
            return

        self.sig_loading_changed.emit(True)
        self.sig_status_message.emit(f"Loading {os.path.basename(path)}...")
        QApplication.processEvents()

        try:
            # NEW: Explicitly free VRAM from the old model before loading a new one
            if self.gs_actor:
                self.gs_actor.cleanup()
                self.gs_actor = None
                self.plotter.clear_box_widgets()
                self.action_crop.setChecked(False)

            raw_gaussians = util_gau.load_ply(path)
            
            centroid = np.mean(raw_gaussians.xyz, axis=0)
            raw_gaussians.xyz -= centroid

            self.plotter.clear()
            
            self.gs_actor = GaussianActor(raw_gaussians)
            self.gs_actor.bind_to_plotter(self.plotter)

            self.plotter.reset_camera()
            self.sig_gau_count_changed.emit(self.gaussian_count())
            self.sig_status_message.emit("Ready")
            self.plotter.update()
        except Exception as e:
            self.sig_status_message.emit(f"Error loading PLY: {e}")
        finally:
            self.sig_loading_changed.emit(False)


    # ─── Mouse Interaction (Qt Event Filter) ──────────────────────────── #

    def eventFilter(self, source, event):
        if source is self.plotter.interactor and event.type() == QEvent.MouseButtonDblClick:
            if event.button() == Qt.LeftButton:
                self._handle_double_click(event.pos())
                return True
        return super().eventFilter(source, event)
    
    def _handle_double_click(self, pos):
        if not self.gs_actor or self.gs_actor.point_count == 0:
            self.sig_status_message.emit("Please load a .ply file first.")
            return

        x = pos.x()
        y = self.plotter.interactor.height() - pos.y()
        w, h = self.plotter.window_size

        # Convert 2D screen click to 3D world ray
        renderer = self.plotter.renderer
        renderer.SetDisplayPoint(x, y, 0.0)
        renderer.DisplayToWorld()
        ray_origin = np.array(renderer.GetWorldPoint()[:3], dtype=np.float32)

        renderer.SetDisplayPoint(x, y, 1.0)
        renderer.DisplayToWorld()
        ray_target = np.array(renderer.GetWorldPoint()[:3], dtype=np.float32)
        
        # Calculate normalized ray direction
        ray_dir = ray_target - ray_origin
        norm = np.linalg.norm(ray_dir)
        if norm == 0:
            return
        ray_dir /= norm

        fovy_rad = np.radians(self.plotter.camera.view_angle)
        
        # Query the actor for a mathematical hit
        hit_pos = self.gs_actor.pick_gaussian(ray_origin, ray_dir, fovy_rad, h)

        if hit_pos is not None:
            self.set_focal_point(float(hit_pos[0]), float(hit_pos[1]), float(hit_pos[2]))
            self.sig_focal_changed.emit(hit_pos[0], hit_pos[1], hit_pos[2])
            self.sig_status_message.emit(f"Focal point set to ({hit_pos[0]:.2f}, {hit_pos[1]:.2f}, {hit_pos[2]:.2f})")
        else:
            self.sig_status_message.emit("Missed Gaussians. Try clicking closer to the object.")


    # ─── Control Panel Interface ──────────────────────────────────────── #

    def gaussian_count(self):
        return self.gs_actor.point_count if self.gs_actor else 0

    @property
    def scale_modifier(self): 
        return self.gs_actor.scale_modifier if self.gs_actor else 1.0

    def set_scale_modifier(self, modifier: float):
        if self.gs_actor:
            self.gs_actor.scale_modifier = modifier
            self.plotter.update()

    @property
    def render_mode(self): 
        return self.gs_actor.render_mode if self.gs_actor else 7

    def set_render_mode(self, mod: int):
        if self.gs_actor:
            self.gs_actor.render_mode = mod
            self.plotter.update()

    @property
    def auto_sort(self): 
        return self.gs_actor.auto_sort if self.gs_actor else False

    @auto_sort.setter
    def auto_sort(self, val: bool):
        if self.gs_actor:
            self.gs_actor.auto_sort = val

    def set_reduce_updates(self, val: bool):
        if self.gs_actor:
            self.gs_actor.reduce_updates = val

    def backend_names(self):
        return ["OpenGL (PyVista Bridge)"]

    def current_backend_idx(self):
        return 0

    def set_backend(self, idx: int):
        pass 

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
        if self.gs_actor:
            w, h = self.plotter.window_size
            cam_adapter = VTKCameraAdapter(self.plotter.camera, w, h)
            self.gs_actor.sort_gaussians(cam_adapter)
            self.plotter.update()

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
        QApplication.processEvents()

        try:
            raw_gaussians = util_gau.load_ply(path)
            
            # Center the data
            centroid = np.mean(raw_gaussians.xyz, axis=0)
            raw_gaussians.xyz -= centroid

            # Clear existing scene
            self.plotter.clear()
            
            # Create our new modular actor and bind it
            self.gs_actor = GaussianActor(raw_gaussians)
            self.gs_actor.bind_to_plotter(self.plotter)

            self.plotter.reset_camera()
            self.sig_gau_count_changed.emit(self.gaussian_count())
            self.sig_status_message.emit("Ready")
            self.plotter.update()
        except Exception as e:
            self.sig_status_message.emit(f"Error loading PLY: {e}")
        finally:
            self.sig_loading_changed.emit(False)


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