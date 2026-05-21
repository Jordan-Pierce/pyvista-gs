"""
gs_widget.py
PyVistaQt central widget for the Gaussian Splatting viewer.

Exposes the same Qt API as the original gaussian_widget.GaussianWidget so
gs_control_panel.py works as a drop-in replacement for control_panel.py.

Signals
  sig_status_message(str)   – status bar text
  sig_fps_changed(float)    – current FPS
  sig_gau_count_changed(int)– gaussian count after load
  sig_loading_changed(bool) – True while loading

Methods (mirrors GaussianWidget)
  load_ply(path)
  gaussian_count() -> int
  fovy_deg() -> float
  set_fovy_deg(deg)
  fit_camera_to_gaussians()   # alias for center_view(reset_orientation=False)
  reset_camera()
  flip_ground()
  set_render_mode(mode)       # 0-7, matching RENDER_MODES combo indices
  set_scale_modifier(v)
  set_opacity(v)
  set_bbox_visible(b)
  set_reduce_updates(v)       # no-op – PyVista has no equivalent
  sort_gaussians()
  backend_names() -> list[str]
  current_backend_idx() -> int
  save_image(path=None)
  auto_sort    bool attribute
  render_mode  int  attribute
  scale_modifier float attribute

Keyboard shortcuts (registered via PyVista key bindings)
  Q / E  – roll camera left / right
  F      – fit camera to scene
  R      – reset camera
"""
from __future__ import annotations

import math
import os
import time
from typing import Optional

import numpy as np
import pyvista as pv
import vtk
from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QFileDialog, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

from native_actor_gs import (
    apply_3dgs_shaders,
    load_3dgs_as_polydata,
    set_render_mode_on_actor,
    sort_splats_by_depth,
)


# ---------------------------------------------------------------------------
#  Loading overlay  (mirrors LoadingOverlay in gaussian_widget.py)
# ---------------------------------------------------------------------------

class _LoadingOverlay(QWidget):
    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._filename = ""
        self._tick = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._step)
        self._timer.setInterval(80)

    def start(self, filename: str = "") -> None:
        self._filename = filename
        self._tick = 0
        self.resize(self.parent().size())
        self.raise_()
        self.show()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    def _step(self) -> None:
        self._tick += 1
        self.update()

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(8, 10, 18, 200))

        cx, cy = self.width() // 2, self.height() // 2
        R = 44

        p.setPen(QPen(QColor(40, 80, 180), 2))
        p.drawEllipse(cx - R, cy - R - 24, R * 2, R * 2)
        p.setPen(QPen(QColor(80, 140, 255), 3))
        p.drawArc(cx - R, cy - R - 24, R * 2, R * 2,
                  (self._tick * 40) * 16, 260 * 16)

        p.setFont(QFont("Segoe UI Symbol", 20))
        p.setPen(QColor(140, 180, 255))
        p.drawText(cx - 14, cy - 24 + R + 10,
                   self._FRAMES[self._tick % len(self._FRAMES)])

        p.setFont(QFont("JetBrains Mono", 12, QFont.Bold))
        p.setPen(QColor(180, 200, 255))
        p.drawText(cx - 80, cy + 46, 160, 24, Qt.AlignCenter, "Loading…")

        if self._filename:
            name = os.path.basename(self._filename)
            if len(name) > 50:
                name = "…" + name[-47:]
            p.setFont(QFont("JetBrains Mono", 9))
            p.setPen(QColor(80, 110, 170))
            p.drawText(cx - 220, cy + 72, 440, 20, Qt.AlignCenter, name)

        p.end()


# ---------------------------------------------------------------------------
#  Background PLY loader
# ---------------------------------------------------------------------------

class _PlyLoader(QObject):
    finished = pyqtSignal(object, str)   # (mesh | None, error_msg)

    def __init__(self, path: str):
        super().__init__()
        self._path = path

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.finished.emit(load_3dgs_as_polydata(self._path), "")
        except Exception as exc:
            self.finished.emit(None, str(exc))


# ---------------------------------------------------------------------------
#  GaussianSplatWidget
# ---------------------------------------------------------------------------

class GaussianSplatWidget(QWidget):
    """Central widget: PyVistaQt interactor + 3DGS GLSL shaders."""

    sig_status_message    = pyqtSignal(str)
    sig_fps_changed       = pyqtSignal(float)
    sig_gau_count_changed = pyqtSignal(int)
    sig_loading_changed   = pyqtSignal(bool)

    _DBLCLICK_MAX_DT  = 0.35   # seconds
    _DBLCLICK_MAX_DXY = 5      # pixels

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._plotter = QtInteractor(self)
        self._plotter.set_background("#13161f")
        layout.addWidget(self._plotter.interactor)

        # State
        self._mesh: Optional[pv.PolyData] = None
        self._splat_actor = None
        self._bbox_actor  = None

        self._bbox_visible   = True
        self._opacity        = 0.99
        self._scale_modifier = 1.0
        self.render_mode     = 7
        self.auto_sort       = False
        self.reduce_updates  = True

        self._original_scales: Optional[np.ndarray] = None
        self._load_thread: Optional[QThread]    = None
        self._load_worker: Optional[_PlyLoader] = None

        # Loading overlay
        self._overlay = _LoadingOverlay(self)
        self._overlay.hide()

        # FPS tracking
        self._frame_count = 0
        self._fps_t0      = time.perf_counter()
        self._fps_timer   = QTimer(self)
        self._fps_timer.setInterval(500)
        self._fps_timer.timeout.connect(self._tick_fps)
        self._fps_timer.start()
        self._plotter.iren.add_observer("RenderEvent", self._on_render)

        # Resort on camera stop
        self._plotter.iren.add_observer(
            "EndInteractionEvent", lambda *_: self._resort())

        # Auto-sort timer (80 ms, same period as original)
        self._sort_timer = QTimer(self)
        self._sort_timer.setInterval(80)
        self._sort_timer.timeout.connect(self._maybe_auto_sort)
        self._sort_timer.start()

        # Double-click focal-point picking
        self._picker       = vtk.vtkPointPicker()
        self._picker.SetTolerance(0.01)
        self._last_click_t = 0.0
        self._click_xy     = (0, 0)
        self._plotter.iren.add_observer("LeftButtonPressEvent", self._on_left_press)

        # Keyboard shortcuts  (Q/E roll, F fit, R reset)
        self._plotter.add_key_event("q", lambda: self._roll_camera(5.0))
        self._plotter.add_key_event("e", lambda: self._roll_camera(-5.0))
        self._plotter.add_key_event("f", self.fit_camera_to_gaussians)
        self._plotter.add_key_event("r", self.reset_camera)

        self.sig_status_message.emit("Ready — open a .ply file to begin.")

    # -----------------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------------

    def gaussian_count(self) -> int:
        return 0 if self._mesh is None else int(self._mesh.n_points)

    # -- Loading ------------------------------------------------------------

    def load_ply(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            self.sig_status_message.emit(f"File not found: {path}")
            return
        if self._load_thread is not None and self._load_thread.isRunning():
            self.sig_status_message.emit("A load is already in progress…")
            return

        self.sig_loading_changed.emit(True)
        self.sig_status_message.emit(f"Loading {os.path.basename(path)}…")
        self._overlay.start(path)

        self._load_thread = QThread(self)
        self._load_worker = _PlyLoader(path)
        self._load_worker.moveToThread(self._load_thread)
        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.finished.connect(self._on_load_finished)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.finished.connect(self._load_worker.deleteLater)
        self._load_thread.finished.connect(self._load_thread.deleteLater)
        self._load_thread.start()

    # -- Camera -------------------------------------------------------------

    def fovy_deg(self) -> float:
        """Return vertical FOV in degrees."""
        return float(self._plotter.camera.view_angle)

    def set_fovy_deg(self, deg: float) -> None:
        self._plotter.camera.view_angle = float(deg)
        self._plotter.render()

    def fit_camera_to_gaussians(self) -> None:
        """Frame the scene (mirrors original fit_camera_to_gaussians)."""
        self.center_view(reset_orientation=False)

    def center_view(self, reset_orientation: bool = True) -> None:
        if self._mesh is None:
            return
        cx, cy, cz = (float(c) for c in self._mesh.center)
        xmin, xmax, ymin, ymax, zmin, zmax = self._mesh.bounds
        radius = max(math.sqrt(
            (0.5*(xmax-xmin))**2 + (0.5*(ymax-ymin))**2 + (0.5*(zmax-zmin))**2
        ), 1e-3)

        cam = self._plotter.camera
        if reset_orientation:
            view_dir = [0.0, -1.0, 0.0]
            cam.up   = (0.0, 0.0, 1.0)
        else:
            pos = cam.position
            foc = cam.focal_point
            view_dir = [pos[i] - foc[i] for i in range(3)]
            norm = math.sqrt(sum(c*c for c in view_dir))
            if norm < 1e-6:
                view_dir = [0.0, -1.0, 0.0]; cam.up = (0.0, 0.0, 1.0); norm = 1.0
            view_dir = [c / norm for c in view_dir]

        distance = (radius / max(math.sin(math.radians(cam.view_angle * 0.5)), 1e-3)) * 1.15
        cam.focal_point = (cx, cy, cz)
        cam.position = (cx + view_dir[0]*distance,
                        cy + view_dir[1]*distance,
                        cz + view_dir[2]*distance)
        self._plotter.renderer.ResetCameraClippingRange()
        self._resort()
        self._plotter.render()

    def reset_camera(self) -> None:
        if self._mesh is not None:
            self.center_view(reset_orientation=False)
        else:
            self._plotter.reset_camera()
        self._resort()
        self._plotter.render()

    def flip_ground(self) -> None:
        """Flip the camera up-vector (mirrors original flip_ground)."""
        vtk_cam = self._plotter.renderer.GetActiveCamera()
        ux, uy, uz = vtk_cam.GetViewUp()
        vtk_cam.SetViewUp(-ux, -uy, -uz)
        self._resort()
        self._plotter.render()

    # -- Rendering ----------------------------------------------------------

    def set_opacity(self, v: float) -> None:
        self._opacity = float(np.clip(v, 0.0, 1.0))
        if self._splat_actor is not None:
            self._splat_actor.GetProperty().SetOpacity(self._opacity)
            self._plotter.render()

    def set_scale_modifier(self, v: float) -> None:
        self._scale_modifier = max(float(v), 1e-4)
        if self._mesh is None or self._original_scales is None:
            return
        scaled = (self._original_scales * self._scale_modifier).astype(np.float32)
        self._mesh.point_data["gs_scales"] = scaled
        arr = self._mesh.GetPointData().GetArray("gs_scales")
        if arr is not None:
            arr.Modified()
        self._plotter.render()

    @property
    def scale_modifier(self) -> float:
        return self._scale_modifier

    @scale_modifier.setter
    def scale_modifier(self, v: float) -> None:
        self.set_scale_modifier(v)

    def set_render_mode(self, mode: int) -> None:
        """Switch shading/SH mode (0-7, matching RENDER_MODES combo indices)."""
        self.render_mode = int(mode)
        if self._splat_actor is not None:
            set_render_mode_on_actor(self._splat_actor, self.render_mode)
            self._plotter.render()

    def set_bbox_visible(self, visible: bool) -> None:
        self._bbox_visible = bool(visible)
        if self._bbox_actor is not None:
            self._bbox_actor.SetVisibility(self._bbox_visible)
            self._plotter.render()

    def set_reduce_updates(self, val: bool) -> None:
        """No-op: PyVista/VTK has no equivalent VSync/reduce control."""
        self.reduce_updates = bool(val)

    # -- Sorting ------------------------------------------------------------

    def sort_gaussians(self) -> None:
        self._resort()
        self._plotter.render()

    # -- Backend (single backend) -------------------------------------------

    def backend_names(self) -> list[str]:
        return ["PyVista / VTK"]

    def current_backend_idx(self) -> int:
        return 0

    # -- Export -------------------------------------------------------------

    def save_image(self, path: Optional[str] = None) -> None:
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Viewport Image",
                os.path.expanduser("~/gaussian_splat.png"),
                "PNG image (*.png);;JPEG image (*.jpg *.jpeg)",
            )
            if not path:
                return
        try:
            self._plotter.screenshot(path)
            self.sig_status_message.emit(f"Saved → {path}")
        except Exception as exc:
            self.sig_status_message.emit(f"Save failed: {exc}")

    # -----------------------------------------------------------------------
    #  Private slots
    # -----------------------------------------------------------------------

    @pyqtSlot(object, str)
    def _on_load_finished(self, mesh, error: str) -> None:
        self._overlay.stop()
        self.sig_loading_changed.emit(False)

        if mesh is None:
            self.sig_status_message.emit(f"Load failed: {error}")
            return

        if self._splat_actor is not None:
            self._plotter.remove_actor(self._splat_actor, render=False)
        if self._bbox_actor is not None:
            self._plotter.remove_actor(self._bbox_actor, render=False)

        self._mesh = mesh
        self._original_scales = np.asarray(
            mesh.point_data["gs_scales"], dtype=np.float32).copy()
        if abs(self._scale_modifier - 1.0) > 1e-6:
            self.set_scale_modifier(self._scale_modifier)

        self._splat_actor = self._plotter.add_mesh(
            self._mesh, style="points", render_points_as_spheres=False,
            lighting=False, show_scalar_bar=False, rgb=True, reset_camera=False)
        self._splat_actor.GetProperty().SetOpacity(self._opacity)
        apply_3dgs_shaders(self._splat_actor, self._mesh, self.render_mode)

        box = pv.Box(bounds=self._mesh.bounds)
        self._bbox_actor = self._plotter.add_mesh(
            box, color="cyan", style="wireframe", line_width=2, reset_camera=False)
        self._bbox_actor.SetVisibility(self._bbox_visible)

        self.center_view(reset_orientation=True)
        self._resort()
        self._plotter.render()

        n = self.gaussian_count()
        self.sig_gau_count_changed.emit(n)
        self.sig_status_message.emit(f"Loaded {n:,} Gaussians.")

    # -- Camera helpers -----------------------------------------------------

    def _roll_camera(self, deg: float) -> None:
        self._plotter.renderer.GetActiveCamera().Roll(deg)
        self._resort()
        self._plotter.render()

    def _set_focal_point(self, world_xyz) -> None:
        cam     = self._plotter.camera
        old_pos = cam.position
        old_foc = cam.focal_point
        vd      = [old_foc[i] - old_pos[i] for i in range(3)]
        cam.focal_point = tuple(float(c) for c in world_xyz)
        cam.position    = tuple(float(world_xyz[i] - vd[i]) for i in range(3))
        self._plotter.renderer.ResetCameraClippingRange()
        self._resort()
        self._plotter.render()

    def _get_vtk_iren(self):
        iren = self._plotter.iren
        for attr in ("interactor", "_iren", "_Iren"):
            inner = getattr(iren, attr, None)
            if inner is not None and hasattr(inner, "GetEventPosition"):
                return inner
        rw = self._plotter.render_window
        if rw is not None:
            vi = rw.GetInteractor()
            if vi is not None:
                return vi
        return iren

    def _on_left_press(self, *_args) -> None:
        if self._mesh is None:
            return
        x, y = self._get_vtk_iren().GetEventPosition()

        now = time.perf_counter()
        dt  = now - self._last_click_t
        dx  = abs(x - self._click_xy[0])
        dy  = abs(y - self._click_xy[1])
        self._last_click_t = now
        self._click_xy     = (x, y)

        if dt > self._DBLCLICK_MAX_DT or dx > self._DBLCLICK_MAX_DXY \
                or dy > self._DBLCLICK_MAX_DXY:
            return

        self._last_click_t = 0.0
        renderer = self._plotter.renderer
        hit = self._picker.Pick(x, y, 0, renderer)
        pos = self._picker.GetPickPosition()
        pid = self._picker.GetPointId()

        if not hit or pid < 0:
            self.sig_status_message.emit("No splat under cursor.")
            return

        self._set_focal_point(pos)
        self.sig_status_message.emit(
            f"Focal point: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    # -- Sort / FPS ---------------------------------------------------------

    def _resort(self) -> None:
        if self._mesh is None:
            return
        try:
            sort_splats_by_depth(self._plotter, self._mesh)
        except Exception:
            pass

    def _maybe_auto_sort(self) -> None:
        if self.auto_sort and self._mesh is not None:
            self._resort()
            self._plotter.render()

    def _on_render(self, *_args) -> None:
        self._frame_count += 1

    def _tick_fps(self) -> None:
        now = time.perf_counter()
        dt  = now - self._fps_t0
        if dt <= 0:
            return
        fps               = self._frame_count / dt
        self._frame_count = 0
        self._fps_t0      = now
        self.sig_fps_changed.emit(fps)

    # -----------------------------------------------------------------------
    #  Qt lifecycle
    # -----------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._overlay.isVisible():
            self._overlay.resize(self.size())

    def closeEvent(self, event) -> None:
        try:
            self._plotter.close()
        finally:
            super().closeEvent(event)
