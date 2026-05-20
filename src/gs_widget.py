"""
gs_widget.py
PyVistaQt-based central widget that renders a 3D Gaussian Splat scene using
the loader and custom GLSL shaders defined in `native_actor_gs.py`.

Exposes a small Qt API consumed by `gs_control_panel.py`:

  Signals
    sig_status_message(str)       - status messages for the status bar
    sig_fps_changed(float)        - current FPS (drives the status bar readout)
    sig_gau_count_changed(int)    - gaussian count after a successful load
    sig_loading_changed(bool)     - toggles whilst a PLY is loading

  Methods
    load_ply(path: str)
    set_opacity(v: float)         - global opacity multiplier (0..1)
    set_scale_modifier(v: float)  - multiplies the per-splat scales
    set_bbox_visible(b: bool)
    reset_camera()
    center_view()                 - frame the splat in the viewport
    save_image(path)              - save a PNG of the current viewport
    gaussian_count() -> int
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
from PyQt5.QtWidgets import QFileDialog, QWidget, QVBoxLayout
from pyvistaqt import QtInteractor

# Re-use the loader + shader injection from the standalone script
from native_actor_gs import (
    load_3dgs_as_polydata,
    apply_3dgs_shaders,
    sort_splats_by_depth,
)


# --------------------------------------------------------------------------- #
#  Background loader thread                                                   #
# --------------------------------------------------------------------------- #

class _PlyLoader(QObject):
    """Loads a PLY file off the GUI thread."""
    finished = pyqtSignal(object, str)   # (mesh or None, error_msg)

    def __init__(self, path: str):
        super().__init__()
        self._path = path

    @pyqtSlot()
    def run(self):
        try:
            mesh = load_3dgs_as_polydata(self._path)
            self.finished.emit(mesh, "")
        except Exception as exc:  # pragma: no cover
            self.finished.emit(None, str(exc))


# --------------------------------------------------------------------------- #
#  GaussianSplatWidget                                                        #
# --------------------------------------------------------------------------- #

class GaussianSplatWidget(QWidget):
    """Central widget hosting a PyVistaQt QtInteractor with custom GS shaders."""

    sig_status_message    = pyqtSignal(str)
    sig_fps_changed       = pyqtSignal(float)
    sig_gau_count_changed = pyqtSignal(int)
    sig_loading_changed   = pyqtSignal(bool)

    # Double-click detection thresholds (manual because VTK+Qt drops
    # LeftButtonDoubleClickEvent).
    _DBLCLICK_MAX_DT = 0.35   # seconds between two presses
    _DBLCLICK_MAX_DXY = 5     # pixels of jitter allowed

    def __init__(self, parent=None):
        super().__init__(parent)

        # ---- Layout: QtInteractor fills the widget --------------------- #
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plotter = QtInteractor(self)
        self._plotter.set_background("#13161f")
        layout.addWidget(self._plotter.interactor)

        # ---- State ----------------------------------------------------- #
        self._mesh: Optional[pv.PolyData] = None
        self._splat_actor = None
        self._bbox_actor = None
        self._bbox_visible = True
        self._opacity = 0.99
        self._scale_modifier = 1.0
        self._original_scales: Optional[np.ndarray] = None

        # ---- FPS tracking --------------------------------------------- #
        self._frame_count = 0
        self._fps_t0 = time.perf_counter()
        self._fps_timer = QTimer(self)
        self._fps_timer.setInterval(500)
        self._fps_timer.timeout.connect(self._tick_fps)
        self._fps_timer.start()

        # Count a frame on every render
        self._plotter.iren.add_observer("RenderEvent", self._on_render)

        # Re-sort splats when the camera stops moving
        self._plotter.iren.add_observer(
            "EndInteractionEvent", lambda *_: self._resort()
        )

        # Double-left-click detection: VTK on Qt does not reliably emit
        # LeftButtonDoubleClickEvent (Qt swallows the second click), so we
        # detect it ourselves by timing consecutive LeftButtonPressEvents.
        # vtkPointPicker is the right choice: splats render as GL points
        # (style='points'), with no cells for vtkCellPicker to hit.
        self._picker = vtk.vtkPointPicker()
        self._picker.SetTolerance(0.01)
        self._last_click_t = 0.0
        self._click_xy = (0, 0)
        self._plotter.iren.add_observer(
            "LeftButtonPressEvent", self._on_left_press
        )

        # ---- Background load plumbing --------------------------------- #
        self._load_thread: Optional[QThread] = None
        self._load_worker: Optional[_PlyLoader] = None

        self.sig_status_message.emit("Ready - open a .ply file to begin.")

    # ---- Public API --------------------------------------------------- #

    def gaussian_count(self) -> int:
        return 0 if self._mesh is None else int(self._mesh.n_points)

    def load_ply(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            self.sig_status_message.emit(f"File not found: {path}")
            return

        if self._load_thread is not None and self._load_thread.isRunning():
            self.sig_status_message.emit("A load is already in progress...")
            return

        self.sig_loading_changed.emit(True)
        self.sig_status_message.emit(f"Loading {os.path.basename(path)}...")

        self._load_thread = QThread(self)
        self._load_worker = _PlyLoader(path)
        self._load_worker.moveToThread(self._load_thread)
        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.finished.connect(self._on_load_finished)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.finished.connect(self._load_worker.deleteLater)
        self._load_thread.finished.connect(self._load_thread.deleteLater)
        self._load_thread.start()

    def set_opacity(self, v: float) -> None:
        self._opacity = float(np.clip(v, 0.0, 1.0))
        if self._splat_actor is not None:
            self._splat_actor.GetProperty().SetOpacity(self._opacity)
            self._plotter.render()

    def set_scale_modifier(self, v: float) -> None:
        """Multiply per-splat scales by `v` relative to the loaded values."""
        self._scale_modifier = max(float(v), 1e-4)
        if self._mesh is None or self._original_scales is None:
            return
        scaled = (self._original_scales * self._scale_modifier).astype(np.float32)
        self._mesh.point_data['gs_scales'] = scaled
        vtk_arr = self._mesh.GetPointData().GetArray('gs_scales')
        if vtk_arr is not None:
            vtk_arr.Modified()
        self._plotter.render()

    def set_bbox_visible(self, visible: bool) -> None:
        self._bbox_visible = bool(visible)
        if self._bbox_actor is not None:
            self._bbox_actor.SetVisibility(self._bbox_visible)
            self._plotter.render()

    def reset_camera(self) -> None:
        if self._mesh is not None:
            self.center_view(reset_orientation=False)
        else:
            self._plotter.reset_camera()
        self._resort()
        self._plotter.render()

    def center_view(self, reset_orientation: bool = True) -> None:
        """Center the loaded splat in the viewport.

        Computes the bounding-sphere radius from the AABB, then places the
        camera at a distance derived from the current FOV so the whole splat
        fits with a small margin. Focal point is the mesh centroid.
        """
        if self._mesh is None:
            return
        cx, cy, cz = (float(c) for c in self._mesh.center)
        xmin, xmax, ymin, ymax, zmin, zmax = self._mesh.bounds
        rx = 0.5 * (xmax - xmin)
        ry = 0.5 * (ymax - ymin)
        rz = 0.5 * (zmax - zmin)
        radius = max(math.sqrt(rx * rx + ry * ry + rz * rz), 1e-3)

        cam = self._plotter.camera

        if reset_orientation:
            view_dir = [0.0, -1.0, 0.0]
            cam.up = (0.0, 0.0, 1.0)
        else:
            pos = cam.position
            foc = cam.focal_point
            view_dir = [pos[i] - foc[i] for i in range(3)]
            norm = math.sqrt(sum(c * c for c in view_dir))
            if norm < 1e-6:
                view_dir = [0.0, -1.0, 0.0]
                cam.up = (0.0, 0.0, 1.0)
                norm = 1.0
            view_dir = [c / norm for c in view_dir]

        half_fov_rad = math.radians(cam.view_angle * 0.5)
        margin = 1.15
        distance = (radius / max(math.sin(half_fov_rad), 1e-3)) * margin

        cam.focal_point = (cx, cy, cz)
        cam.position = (
            cx + view_dir[0] * distance,
            cy + view_dir[1] * distance,
            cz + view_dir[2] * distance,
        )
        self._plotter.renderer.ResetCameraClippingRange()
        self._plotter.render()

    def save_image(self, path: Optional[str] = None) -> None:
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Viewport Image",
                os.path.expanduser("~/gaussian_splat.png"),
                "PNG image (*.png);;JPEG image (*.jpg *.jpeg)"
            )
            if not path:
                return
        try:
            self._plotter.screenshot(path)
            self.sig_status_message.emit(f"Saved {path}")
        except Exception as exc:
            self.sig_status_message.emit(f"Save failed: {exc}")

    # ---- Internals ---------------------------------------------------- #

    @pyqtSlot(object, str)
    def _on_load_finished(self, mesh, error: str) -> None:
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
            mesh.point_data['gs_scales'], dtype=np.float32
        ).copy()
        if abs(self._scale_modifier - 1.0) > 1e-6:
            self.set_scale_modifier(self._scale_modifier)

        self._splat_actor = self._plotter.add_mesh(
            self._mesh,
            style='points',
            render_points_as_spheres=False,
            lighting=False,
            show_scalar_bar=False,
            rgb=True,
            reset_camera=False,
        )
        self._splat_actor.GetProperty().SetOpacity(self._opacity)

        apply_3dgs_shaders(self._splat_actor, self._mesh)

        box = pv.Box(bounds=self._mesh.bounds)
        self._bbox_actor = self._plotter.add_mesh(
            box, color="cyan", style="wireframe", line_width=2,
            reset_camera=False,
        )
        self._bbox_actor.SetVisibility(self._bbox_visible)

        self.center_view(reset_orientation=True)

        self._resort()
        self._plotter.render()

        n = self.gaussian_count()
        self.sig_gau_count_changed.emit(n)
        self.sig_status_message.emit(f"Loaded {n:,} gaussians.")

    def _set_focal_point(self, world_xyz) -> None:
        """Re-aim the camera at `world_xyz`, preserving view direction & distance."""
        cam = self._plotter.camera
        old_pos = cam.position
        old_foc = cam.focal_point
        view_dir = [old_foc[i] - old_pos[i] for i in range(3)]
        new_pos = [world_xyz[i] - view_dir[i] for i in range(3)]
        cam.focal_point = tuple(float(c) for c in world_xyz)
        cam.position = tuple(float(c) for c in new_pos)
        self._plotter.renderer.ResetCameraClippingRange()
        self._resort()
        self._plotter.render()

    def _get_vtk_iren(self):
        """Return the underlying vtkRenderWindowInteractor.

        `self._plotter.iren` is PyVistaQt's `RenderWindowInteractor` wrapper,
        which doesn't expose `GetEventPosition` directly. The real VTK
        interactor sits at one of these attribute paths depending on the
        PyVistaQt version, so we probe them in order.
        """
        iren = self._plotter.iren
        # PyVistaQt >= 0.10 keeps the QVTKRenderWindowInteractor on .interactor
        for attr in ('interactor', '_iren', '_Iren'):
            inner = getattr(iren, attr, None)
            if inner is not None and hasattr(inner, 'GetEventPosition'):
                return inner
        # Last resort: ask the render window for its interactor.
        rw = self._plotter.render_window
        if rw is not None:
            vi = rw.GetInteractor()
            if vi is not None:
                return vi
        # Give up and return the wrapper; the caller will raise a clear error.
        return iren

    def _on_left_press(self, *_args) -> None:
        """Synthesise double-click detection from LeftButtonPressEvent."""
        if self._mesh is None:
            return

        vtk_iren = self._get_vtk_iren()
        x, y = vtk_iren.GetEventPosition()

        now = time.perf_counter()
        dt = now - self._last_click_t
        dx = abs(x - self._click_xy[0])
        dy = abs(y - self._click_xy[1])
        self._last_click_t = now
        self._click_xy = (x, y)

        if dt > self._DBLCLICK_MAX_DT or dx > self._DBLCLICK_MAX_DXY \
                or dy > self._DBLCLICK_MAX_DXY:
            return  # single click; let normal orbit/pan proceed

        # Reset timer so triple-clicks do not re-fire.
        self._last_click_t = 0.0

        renderer = self._plotter.renderer
        hit = self._picker.Pick(x, y, 0, renderer)
        pos = self._picker.GetPickPosition()
        point_id = self._picker.GetPointId()

        if not hit or point_id < 0:
            self.sig_status_message.emit("No splat under cursor.")
            return

        self._set_focal_point(pos)
        self.sig_status_message.emit(
            f"Focal point: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
        )

    def _resort(self) -> None:
        if self._mesh is None:
            return
        try:
            sort_splats_by_depth(self._plotter, self._mesh)
        except Exception:
            pass

    def _on_render(self, *_args) -> None:
        self._frame_count += 1

    def _tick_fps(self) -> None:
        now = time.perf_counter()
        dt = now - self._fps_t0
        if dt <= 0:
            return
        fps = self._frame_count / dt
        self._frame_count = 0
        self._fps_t0 = now
        self.sig_fps_changed.emit(fps)

    # ---- Qt lifecycle ------------------------------------------------- #

    def closeEvent(self, event):  # pragma: no cover - GUI-only
        try:
            self._plotter.close()
        finally:
            super().closeEvent(event)
