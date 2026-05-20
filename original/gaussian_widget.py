"""
gaussian_widget.py
QOpenGLWidget hosting the Gaussian Splatting renderer.

New in this version:
  - Threaded PLY loading (QThread) so the UI never freezes
  - Camera auto-fit on load (centroid + 95th-percentile bounding sphere)
  - reset_camera() / fit_camera_to_gaussians() public methods
  - sig_loading_changed(bool) signal drives the loading overlay
  - Mouse/keyboard input blocked while loading
  - F = fit to scene, R = reset camera keyboard shortcuts
"""
from __future__ import annotations

import time
import math
import numpy as np

from PyQt5.QtWidgets import QOpenGLWidget, QWidget
from PyQt5.QtCore    import Qt, QTimer, QThread, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtGui     import QPainter, QColor, QFont, QPen

import OpenGL.GL as gl

import util as util
import util_gau as util_gau
from renderer_ogl import OpenGLRenderer, GaussianRenderBase


# ─────────────────────────────────────────────────────────────────────────── #
#  Background worker                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

class _PlyLoaderWorker(QObject):
    finished = pyqtSignal(object)   # util_gau.GaussianData
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


# ─────────────────────────────────────────────────────────────────────────── #
#  Loading overlay                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

class LoadingOverlay(QWidget):
    _FRAMES = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._filename = ""
        self._tick     = 0
        self._timer    = QTimer(self)
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

        # Dim backdrop
        p.fillRect(self.rect(), QColor(8, 10, 18, 200))

        cx = self.width()  // 2
        cy = self.height() // 2
        R  = 44

        # Spinning arc
        p.setPen(QPen(QColor(40, 80, 180), 2))
        p.drawEllipse(cx - R, cy - R - 24, R*2, R*2)
        p.setPen(QPen(QColor(80, 140, 255), 3))
        p.drawArc(cx - R, cy - R - 24, R*2, R*2,
                  (self._tick * 40) * 16, 260 * 16)

        # Braille spinner glyph
        p.setFont(QFont("Segoe UI Symbol", 20))
        p.setPen(QColor(140, 180, 255))
        p.drawText(cx - 14, cy - 24 + R + 10, self._FRAMES[self._tick % len(self._FRAMES)])

        # "Loading" text
        f = QFont("JetBrains Mono", 12, QFont.Bold)
        p.setFont(f)
        p.setPen(QColor(180, 200, 255))
        p.drawText(cx - 80, cy + 46, 160, 24, Qt.AlignCenter, "Loading…")

        # Filename
        if self._filename:
            name = self._filename
            if len(name) > 50:
                name = "…" + name[-47:]
            p.setFont(QFont("JetBrains Mono", 9))
            p.setPen(QColor(80, 110, 170))
            p.drawText(cx - 220, cy + 72, 440, 20, Qt.AlignCenter, name)

        p.end()


# ─────────────────────────────────────────────────────────────────────────── #
#  GaussianWidget                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class GaussianWidget(QOpenGLWidget):

    sig_fps_changed       = pyqtSignal(float)
    sig_gau_count_changed = pyqtSignal(int)
    sig_status_message    = pyqtSignal(str)
    sig_loading_changed   = pyqtSignal(bool)   # True = started, False = done

    def __init__(self, parent=None):
        super().__init__(parent)

        self.camera = util.Camera(720, 1280)
        self._default_cam = self._snapshot_camera()

        self._renderer:      GaussianRenderBase | None = None
        self._renderer_list: list                      = []
        self._renderer_idx   = 0

        self._gaussians: util_gau.GaussianData | None = None

        self.scale_modifier = 1.0
        self.render_mode    = 7
        self.auto_sort      = False
        self.reduce_updates = True

        self._is_loading       = False
        self._loader_thread: QThread | None = None

        self._frame_times: list[float] = []
        self._last_frame_t = time.perf_counter()

        self._overlay = LoadingOverlay(self)
        self._overlay.hide()

        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self.update)
        self._render_timer.start(16)

        self._sort_timer = QTimer(self)
        self._sort_timer.timeout.connect(self._maybe_auto_sort)
        self._sort_timer.start(80)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)

    # ── GL lifecycle ───────────────────────────────────────────────────── #

    def initializeGL(self):
        w, h = self.width(), self.height()

        ogl = OpenGLRenderer(w, h)
        self._renderer_list.append(ogl)

        try:
            from renderer_cuda import CUDARenderer
            self._renderer_list.append(CUDARenderer(w, h))
            self._renderer_idx = 1
            self.sig_status_message.emit("CUDA renderer active")
        except Exception as exc:
            self._renderer_idx = 0
            self.sig_status_message.emit(f"OpenGL renderer active (no CUDA: {exc})")

        self._renderer = self._renderer_list[self._renderer_idx]
        self._gaussians = util_gau.naive_gaussian()
        self._push_renderer_state()

    def resizeGL(self, w: int, h: int):
        if self._renderer is None:
            return
        self.camera.update_resolution(h, w)
        self._renderer.set_render_reso(w, h)
        self._renderer.update_camera_intrin(self.camera)
        self.camera.is_intrin_dirty = False
        self._overlay.resize(w, h)

    def paintGL(self):
        if self._renderer is None:
            return
        if self.camera.is_pose_dirty:
            self._renderer.update_camera_pose(self.camera)
            self.camera.is_pose_dirty = False
        if self.camera.is_intrin_dirty:
            self._renderer.update_camera_intrin(self.camera)
            self.camera.is_intrin_dirty = False

        gl.glClearColor(0.08, 0.08, 0.10, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        self._renderer.draw()

        now = time.perf_counter()
        self._frame_times.append(now - self._last_frame_t)
        self._last_frame_t = now
        if len(self._frame_times) > 60:
            self._frame_times.pop(0)
        avg = sum(self._frame_times) / len(self._frame_times)
        self.sig_fps_changed.emit(1.0 / avg if avg > 0 else 0.0)

    # ── Public API ─────────────────────────────────────────────────────── #

    def load_ply(self, path: str):
        """Kick off a background load; UI stays alive the whole time."""
        if self._is_loading:
            return
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
        # Keep worker alive as long as thread lives
        worker.setParent(thread)
        thread.start()

    def sort_gaussians(self):
        if self._renderer and self._gaussians and not self._is_loading:
            self.makeCurrent()
            self._renderer.sort_and_update(self.camera)
            self.doneCurrent()

    def fit_camera_to_gaussians(self):
        """Frame the loaded splat — centroid + 95th-pct bounding radius."""
        if not self._gaussians:
            return
        xyz      = self._gaussians.xyz
        centroid = xyz.mean(axis=0).astype(np.float32)
        dists    = np.linalg.norm(xyz - centroid, axis=1)
        radius   = float(max(np.percentile(dists, 95), 0.5))

        cam     = self.camera
        cur_dir = cam.target - cam.position
        n       = np.linalg.norm(cur_dir)
        cur_dir = (cur_dir / n if n > 1e-6 else
                   np.array([0., 0., -1.], dtype=np.float32))

        cam.target      = centroid
        cam.position    = (centroid - cur_dir * radius * 2.5).astype(np.float32)
        cam.target_dist = radius * 2.5
        cam.is_pose_dirty = True

    def reset_camera(self):
        """Restore the original default camera pose."""
        c  = self.camera
        s  = self._default_cam
        c.position    = s["position"].copy()
        c.target      = s["target"].copy()
        c.up          = s["up"].copy()
        c.yaw         = s["yaw"]
        c.pitch       = s["pitch"]
        c.fovy        = s["fovy"]
        c.target_dist = s["target_dist"]
        c.is_pose_dirty   = True
        c.is_intrin_dirty = True

    def set_scale_modifier(self, val: float):
        self.scale_modifier = val
        if self._renderer:
            self.makeCurrent()
            self._renderer.set_scale_modifier(val)
            self.doneCurrent()

    def set_render_mode(self, mode: int):
        self.render_mode = mode
        if self._renderer:
            self.makeCurrent()
            self._renderer.set_render_mod(mode - 4)
            self.doneCurrent()

    def set_fovy_deg(self, deg: float):
        self.camera.fovy        = math.radians(deg)
        self.camera.is_intrin_dirty = True

    def fovy_deg(self) -> float:
        return math.degrees(self.camera.fovy)

    def set_reduce_updates(self, val: bool):
        self.reduce_updates = val
        if self._renderer:
            self._renderer.reduce_updates = val

    def set_backend(self, idx: int):
        if idx >= len(self._renderer_list):
            return
        self._renderer_idx = idx
        self.makeCurrent()
        self._renderer = self._renderer_list[idx]
        self._push_renderer_state()
        self.doneCurrent()

    def flip_ground(self):
        self.camera.flip_ground()

    def save_image(self) -> str:
        out  = "save.png"
        qimg = self.grabFramebuffer()
        qimg.save(out)
        self.sig_status_message.emit(f"Image saved → {out}")
        return out

    def gaussian_count(self) -> int:
        return len(self._gaussians) if self._gaussians else 0

    def backend_names(self) -> list[str]:
        names = ["OpenGL"]
        if len(self._renderer_list) > 1:
            names.append("CUDA")
        return names

    def current_backend_idx(self) -> int:
        return self._renderer_idx

    def is_loading(self) -> bool:
        return self._is_loading

    # ── Input events ───────────────────────────────────────────────────── #

    def mousePressEvent(self, event):
        if self._is_loading:
            return
        self.camera.is_leftmouse_pressed  = (event.button() == Qt.LeftButton)
        self.camera.is_rightmouse_pressed = (event.button() == Qt.RightButton)
        self.camera.first_mouse = True
        self.camera.last_x = event.x()
        self.camera.last_y = event.y()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.camera.is_leftmouse_pressed = False
        if event.button() == Qt.RightButton:
            self.camera.is_rightmouse_pressed = False

    def mouseMoveEvent(self, event):
        if not self._is_loading:
            self.camera.process_mouse(event.x(), event.y())

    def wheelEvent(self, event):
        if not self._is_loading:
            self.camera.process_wheel(0, event.angleDelta().y() / 120.0)

    def keyPressEvent(self, event):
        if self._is_loading:
            return
        k = event.key()
        if   k == Qt.Key_Q: self.camera.process_roll_key(1)
        elif k == Qt.Key_E: self.camera.process_roll_key(-1)
        elif k == Qt.Key_F: self.fit_camera_to_gaussians()
        elif k == Qt.Key_R: self.reset_camera()
        else: super().keyPressEvent(event)

    # ── Private slots ──────────────────────────────────────────────────── #

    @pyqtSlot(object)
    def _on_ply_loaded(self, data: util_gau.GaussianData):
        self._gaussians = data
        self.makeCurrent()
        self._renderer.update_gaussian_data(self._gaussians)
        self._renderer.sort_and_update(self.camera)
        self.doneCurrent()

        self.fit_camera_to_gaussians()

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

    # ── Helpers ────────────────────────────────────────────────────────── #

    def _push_renderer_state(self):
        if not self._renderer or not self._gaussians:
            return
        r = self._renderer
        r.update_gaussian_data(self._gaussians)
        r.sort_and_update(self.camera)
        r.set_scale_modifier(self.scale_modifier)
        r.set_render_mod(self.render_mode - 4)
        r.update_camera_pose(self.camera)
        r.update_camera_intrin(self.camera)
        r.set_render_reso(self.camera.w, self.camera.h)
        r.reduce_updates = self.reduce_updates
        self.camera.is_pose_dirty   = False
        self.camera.is_intrin_dirty = False
        self.sig_gau_count_changed.emit(len(self._gaussians))

    def _maybe_auto_sort(self):
        if self.auto_sort and self._renderer and self._gaussians and not self._is_loading:
            self.makeCurrent()
            self._renderer.sort_and_update(self.camera)
            self.doneCurrent()

    def _snapshot_camera(self) -> dict:
        c = self.camera
        return dict(
            position    = c.position.copy(),
            target      = c.target.copy(),
            up          = c.up.copy(),
            yaw         = c.yaw,
            pitch       = c.pitch,
            fovy        = c.fovy,
            target_dist = c.target_dist,
        )
