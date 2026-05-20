"""
gaussian_widget.py
PyVista-backed widget hosting the Gaussian Splatting renderer via OpenGL injection.
"""
from __future__ import annotations

import time
import numpy as np

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore    import Qt, QTimer, QThread, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtGui     import QPainter, QColor, QFont, QPen

import pyvista as pv
from pyvistaqt import QtInteractor
import vtk
import OpenGL.GL as gl

import util_gau
from renderer_ogl import OpenGLRenderer


# ─────────────────────────────────────────────────────────────────────────── #
#  Camera Adapter                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class VTKCameraAdapter:
    """Bridges PyVista's vtkCamera to your existing OpenGLRenderer expectations."""
    def __init__(self, vtk_cam, width, height):
        self.vtk_cam = vtk_cam
        self.w = max(width, 1)
        self.h = max(height, 1)
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
        aspect = self.w / self.h
        mat = self.vtk_cam.GetProjectionTransformMatrix(aspect, -1, 1)
        return self._vtk_to_numpy(mat)

    def get_htanfovxy_focal(self):
        fovy = np.radians(self.vtk_cam.GetViewAngle())
        htany = np.tan(fovy / 2.0)
        htanx = htany / self.h * self.w
        focal = self.h / (2.0 * htany)
        return [htanx, htany, focal]


# ─────────────────────────────────────────────────────────────────────────── #
#  Background worker & Loading Overlay (Unchanged from your code)             #
# ─────────────────────────────────────────────────────────────────────────── #

class _PlyLoaderWorker(QObject):
    finished = pyqtSignal(object)
    errored  = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self._path = path

    @pyqtSlot()
    def run(self):
        try:
            self.finished.emit(util_gau.load_ply(self._path))
        except Exception as exc:
            self.errored.emit(str(exc))

class LoadingOverlay(QWidget):
    _FRAMES = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._filename = ""
        self._tick = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._step)
        self._timer.setInterval(80)

    def start(self, filename: str = ""):
        self._filename = filename
        self._tick = 0
        self.resize(self.parent().size())
        self.raise_()
        self.show()
        self._timer.start()

    def stop(self):
        self._timer.stop()
        self.hide()

    def _step(self):
        self._tick += 1
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(8, 10, 18, 200))
        cx, cy, R = self.width() // 2, self.height() // 2, 44
        p.setPen(QPen(QColor(40, 80, 180), 2))
        p.drawEllipse(cx - R, cy - R - 24, R*2, R*2)
        p.setPen(QPen(QColor(80, 140, 255), 3))
        p.drawArc(cx - R, cy - R - 24, R*2, R*2, (self._tick * 40) * 16, 260 * 16)
        p.setFont(QFont("Segoe UI Symbol", 20))
        p.setPen(QColor(140, 180, 255))
        p.drawText(cx - 14, cy - 24 + R + 10, self._FRAMES[self._tick % len(self._FRAMES)])
        p.setFont(QFont("JetBrains Mono", 12, QFont.Bold))
        p.setPen(QColor(180, 200, 255))
        p.drawText(cx - 80, cy + 46, 160, 24, Qt.AlignCenter, "Loading…")
        p.end()


# ─────────────────────────────────────────────────────────────────────────── #
#  GaussianWidget (Now inheriting from QWidget, embedding PyVista)            #
# ─────────────────────────────────────────────────────────────────────────── #

class GaussianWidget(QWidget):

    sig_fps_changed       = pyqtSignal(float)
    sig_gau_count_changed = pyqtSignal(int)
    sig_status_message    = pyqtSignal(str)
    sig_loading_changed   = pyqtSignal(bool)
    sig_focal_changed     = pyqtSignal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 1. Setup Layout & PyVista Interactor
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#080a12") # Match dark theme
        layout.addWidget(self.plotter.interactor)

        # 2. Rendering State
        self._renderer: OpenGLRenderer | None = None
        self._gaussians: util_gau.GaussianData | None = None
        self.scale_modifier = 1.0
        self.render_mode    = 7
        self.auto_sort      = False
        self.reduce_updates = True

        # 3. Utilities
        self._is_loading = False
        self._loader_thread: QThread | None = None
        self._overlay = LoadingOverlay(self)
        self._overlay.hide()
        
        # FPS Tracking
        self._frame_times: list[float] = []
        self._last_frame_t = time.perf_counter()

        # 4. Attach to PyVista's rendering pipeline
        self.plotter.renderer.AddObserver(vtk.vtkCommand.EndEvent, self._on_render_end)
        self.plotter.interactor.mouseDoubleClickEvent = self._on_double_click
        
        # Auto-sort timer (if camera moves and auto-sort is on)
        self._sort_timer = QTimer(self)
        self._sort_timer.timeout.connect(self._maybe_auto_sort)
        self._sort_timer.start(80)

    # ── PyVista Render Hook ────────────────────────────────────────────── #

    def _on_render_end(self, caller, event):
        if not self._gaussians:
            return

        window = caller.GetRenderWindow()
        w, h = window.GetSize()
        if w == 0 or h == 0:
            return

        # Initialize raw PyOpenGL context inside PyVista's active window
        if self._renderer is None:
            self._renderer = OpenGLRenderer(w, h)
            self._renderer.update_gaussian_data(self._gaussians)
            self._renderer.set_scale_modifier(self.scale_modifier)
            self._renderer.set_render_mod(self.render_mode - 4)

        self._renderer.set_render_reso(w, h)

        # Wrap the PyVista Camera
        vtk_cam = caller.GetActiveCamera()
        cam_adapter = VTKCameraAdapter(vtk_cam, w, h)

        # Draw frame
        self._draw_gaussians(cam_adapter)
        
        # Calculate FPS
        now = time.perf_counter()
        self._frame_times.append(now - self._last_frame_t)
        self._last_frame_t = now
        if len(self._frame_times) > 60:
            self._frame_times.pop(0)
        avg = sum(self._frame_times) / len(self._frame_times)
        self.sig_fps_changed.emit(1.0 / avg if avg > 0 else 0.0)

    def _draw_gaussians(self, cam_adapter):
        """Executes the raw PyOpenGL pipeline while preserving VTK's state."""
        # --- PRESERVE VTK STATE ---
        last_prog = gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)
        last_vao = gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING)
        last_blend = gl.glGetBoolean(gl.GL_BLEND)
        last_depth_mask = gl.glGetBoolean(gl.GL_DEPTH_WRITEMASK)
        
        # --- CONFIGURE OUR 3DGS STATE ---
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDepthMask(gl.GL_FALSE) 
        
        # --- DRAW GAUSSIANS ---
        self._renderer.update_camera_pose(cam_adapter)
        self._renderer.update_camera_intrin(cam_adapter)
        self._renderer.draw()
        
        # --- RESTORE VTK STATE ---
        if last_depth_mask: gl.glDepthMask(gl.GL_TRUE)
        else: gl.glDepthMask(gl.GL_FALSE)
            
        if not last_blend: gl.glDisable(gl.GL_BLEND)
            
        gl.glUseProgram(last_prog)
        gl.glBindVertexArray(last_vao)


    # ── Public API (Called by ControlPanel) ────────────────────────────── #

    def load_ply(self, path: str):
        if self._is_loading: return
        self._is_loading = True
        self.sig_loading_changed.emit(True)
        self._overlay.start(path)

        worker = _PlyLoaderWorker(path)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_ply_loaded)
        worker.errored.connect(self._on_ply_error)
        worker.finished.connect(thread.quit)
        worker.errored.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        self._loader_thread = thread
        worker.setParent(thread)
        thread.start()

    def sort_gaussians(self):
        if self._renderer and self._gaussians and not self._is_loading:
            vtk_cam = self.plotter.camera
            w, h = self.plotter.window_size
            cam_adapter = VTKCameraAdapter(vtk_cam, w, h)
            self._renderer.sort_and_update(cam_adapter)
            self.plotter.render()

    def fit_camera_to_gaussians(self):
        self.plotter.reset_camera()
        self.sig_focal_changed.emit(*self.focal_point())

    def reset_camera(self):
        self.plotter.camera_position = 'xy'
        self.plotter.reset_camera()
        self.sig_focal_changed.emit(*self.focal_point())

    def set_scale_modifier(self, val: float):
        self.scale_modifier = val
        if self._renderer:
            self._renderer.set_scale_modifier(val)
            self.plotter.render()

    def set_render_mode(self, mode: int):
        self.render_mode = mode
        if self._renderer:
            self._renderer.set_render_mod(mode - 4)
            self.plotter.render()

    def set_fovy_deg(self, deg: float):
        self.plotter.camera.view_angle = deg
        self.plotter.render()

    def fovy_deg(self) -> float:
        return self.plotter.camera.view_angle

    def set_reduce_updates(self, val: bool):
        self.reduce_updates = val

    def set_focal_point(self, x: float, y: float, z: float):
        self.plotter.camera.focal_point = (x, y, z)
        self.plotter.render()
        self.sig_focal_changed.emit(x, y, z)

    def focal_point(self) -> tuple[float, float, float]:
        return self.plotter.camera.focal_point

    def set_backend(self, idx: int):
        self.sig_status_message.emit("CUDA renderer currently disabled in PyVista mode.")

    def flip_ground(self):
        up = np.array(self.plotter.camera.up)
        self.plotter.camera.up = tuple(-up)
        self.plotter.render()

    def save_image(self) -> str:
        out = "save.png"
        self.plotter.screenshot(out)
        self.sig_status_message.emit(f"Image saved → {out}")
        return out

    def gaussian_count(self) -> int:
        return len(self._gaussians) if self._gaussians else 0

    def backend_names(self) -> list[str]:
        return ["OpenGL (PyVista Injection)"]

    def current_backend_idx(self) -> int:
        return 0

    def is_loading(self) -> bool:
        return self._is_loading

    # ── Private slots ──────────────────────────────────────────────────── #

    @pyqtSlot(object)
    def _on_ply_loaded(self, data: util_gau.GaussianData):
        self._gaussians = data
        
        # Center the data so it revolves perfectly around the camera origin
        centroid = np.mean(self._gaussians.xyz, axis=0)
        self._gaussians.xyz -= centroid

        # Force a re-initialization of the renderer on the next frame
        self._renderer = None 

        # Add invisible mesh to guide PyVista's camera and clipping planes
        self.plotter.clear()
        dummy_pc = pv.PolyData(self._gaussians.xyz)
        self.plotter.add_mesh(dummy_pc, opacity=0.0)
        
        self.plotter.reset_camera()
        self.sig_focal_changed.emit(*self.focal_point())
        self.sort_gaussians()

        self._is_loading = False
        self._overlay.stop()
        self.sig_loading_changed.emit(False)
        self.sig_gau_count_changed.emit(len(self._gaussians))
        self.sig_status_message.emit(f"Loaded {len(self._gaussians):,} Gaussians")

    @pyqtSlot(str)
    def _on_ply_error(self, msg: str):
        self._is_loading = False
        self._overlay.stop()
        self.sig_loading_changed.emit(False)
        self.sig_status_message.emit(f"Error loading PLY: {msg}")

    def _maybe_auto_sort(self):
        if self.auto_sort and self._renderer and not self._is_loading:
            self.sort_gaussians()

    def _on_double_click(self, event):
        if not self._gaussians or self._is_loading:
            return

        x, y = event.x(), event.y()
        w, h = self.plotter.window_size
        if w == 0 or h == 0:
            return

        renderer = self.plotter.renderer
        renderer.SetDisplayPoint(x, h - y, 0.0)
        renderer.DisplayToWorld()
        ray_origin = np.array(renderer.GetWorldPoint()[:3], dtype=np.float32)

        renderer.SetDisplayPoint(x, h - y, 1.0)
        renderer.DisplayToWorld()
        ray_target = np.array(renderer.GetWorldPoint()[:3], dtype=np.float32)
        ray_dir = ray_target - ray_origin
        norm = np.linalg.norm(ray_dir)
        if norm == 0:
            return
        ray_dir /= norm

        vecs = self._gaussians.xyz - ray_origin
        t = np.sum(vecs * ray_dir, axis=1)

        front_mask = t > 0
        if not np.any(front_mask):
            return

        front_xyz = self._gaussians.xyz[front_mask]
        front_vecs = front_xyz - ray_origin
        front_t = t[front_mask]

        proj = front_t[:, None] * ray_dir
        dists = np.linalg.norm(front_vecs - proj, axis=1)
        angles = dists / front_t

        fovy_rad = np.radians(self.plotter.camera.view_angle)
        tolerance = 5.0 * (fovy_rad / h)

        hit_mask = angles < tolerance
        if np.any(hit_mask):
            hit_indices = np.where(hit_mask)[0]
            best_idx = hit_indices[np.argmin(front_t[hit_mask])]
        else:
            best_idx = int(np.argmin(angles))

        best_pt = front_xyz[best_idx]
        self.set_focal_point(float(best_pt[0]), float(best_pt[1]), float(best_pt[2]))