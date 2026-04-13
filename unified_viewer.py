"""
unified_viewer.py
Single-window viewer that holds meshes, camera frustums, axes, and a
3D Gaussian Splat in the same PyVista scene.

All scene decorations (frustums, axes, point clouds) are permanent vtkActors.
The "subject" — mesh or 3DGS splat — is toggled with toggle_representation().

Since 3DGS is a first-class vtkActor (via VTKGaussianRenderer):
  • Left-click sets focal point on a Gaussian point, not just empty space
  • Rotate / pan / zoom / zoom-to-selection all work as normal
  • Camera-ray queries and PyVista pick operations reach the splat
  • Depth ordering between splat and opaque geometry is correct
  • The splat respects every PyVista camera manipulation: it is the same scene

Usage example
-------------
    from unified_viewer import UnifiedViewer
    import numpy as np
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    win = UnifiedViewer()

    # Add a mesh
    win.add_mesh("my_model.ply", color="lightgrey", opacity=0.6)

    # Add camera frustums (4×4 camera-to-world matrices)
    for c2w in my_camera_poses:
        win.add_camera_frustum(c2w, scale=0.15, color="cyan")

    # Load 3DGS splat asynchronously
    win.load_splat("point_cloud.ply")

    win.show()
    sys.exit(app.exec_())
"""
from __future__ import annotations

import os
import math
import sys
import numpy as np
from pathlib import Path
from typing import Sequence


_ROOT_DIR = Path(__file__).resolve().parent
_ROOT_DIR_STR = str(_ROOT_DIR)
if _ROOT_DIR_STR not in sys.path:
    sys.path.insert(0, _ROOT_DIR_STR)

import vtk
import pyvista as pv
from pyvistaqt import QtInteractor

from PyQt5.QtWidgets import (
    QMainWindow, QDockWidget, QToolBar, QAction, QFileDialog,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox, QCheckBox, QPushButton, QGroupBox,
    QScrollArea, QFrame, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QFont

from src import util
from src import util_gau
from src.renderer_vtk import VTKGaussianRenderer

# ── GLSL render-mode labels (kept in sync with original control_panel.py) ─── #
RENDER_MODES = [
    "Gaussian Ball",       # render_mod = -4
    "Flat Ball",           # render_mod = -3
    "Billboard",           # render_mod = -2
    "Depth",               # render_mod = -1
    "SH:0",                # render_mod =  0
    "SH:0~1",              # render_mod =  1
    "SH:0~2",              # render_mod =  2
    "SH:0~3 (default)",    # render_mod =  3
]
_RENDER_MODE_OFFSET = 4   # combo index 0 → render_mod -4


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_frustum(c2w: np.ndarray, scale: float = 0.3) -> pv.PolyData:
    """
    Wire-frame camera frustum from a 4×4 camera-to-world matrix.
    The frustum is a pyramid: apex = camera centre, base = image plane.
    Aspect ratio is fixed at 4:3; adjust local_corners if needed.
    """
    c2w = np.asarray(c2w, dtype=np.float32)
    cam_center    = c2w[:3, 3]
    local_corners = np.array([
        [-1.0, -0.75, 2.0],
        [ 1.0, -0.75, 2.0],
        [ 1.0,  0.75, 2.0],
        [-1.0,  0.75, 2.0],
    ], dtype=np.float32) * scale
    world_corners = (c2w[:3, :3] @ local_corners.T).T + cam_center
    points = np.vstack([cam_center, world_corners])  # 5 points: [0]=apex, [1..4]=base

    # Lines in PyVista format: [n_pts, p0, p1, n_pts, p0, p1, ...]
    lines = np.array([
        2, 0, 1,   2, 0, 2,   2, 0, 3,   2, 0, 4,   # apex → each corner
        2, 1, 2,   2, 2, 3,   2, 3, 4,   2, 4, 1,   # base rectangle
    ], dtype=np.int32)
    mesh = pv.PolyData()
    mesh.points = points
    mesh.lines  = lines
    return mesh


# ── Background PLY loader ─────────────────────────────────────────────────── #

class _PlyLoaderWorker(QObject):
    finished = pyqtSignal(object)  # util_gau.GaussianData
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


# ══════════════════════════════════════════════════════════════════════════════
# Side-panel (self-contained; talks to UnifiedViewer via signals/methods)
# ══════════════════════════════════════════════════════════════════════════════

class _SidePanel(QWidget):
    """Compact sidebar wired to a UnifiedViewer."""

    def __init__(self, viewer: "UnifiedViewer", parent=None):
        super().__init__(parent)
        self._viewer = viewer
        self.setFixedWidth(300)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        root = QVBoxLayout(content)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Scene section ─────────────────────────────────────────────── #
        scene_g = self._group("Scene")
        self._open_ply_btn  = QPushButton("Open Splat (.ply)…")
        self._open_mesh_btn = QPushButton("Open Mesh…")
        self._toggle_btn    = QPushButton("Switch to Mesh")
        self._toggle_btn.setCheckable(False)
        for b in [self._open_ply_btn, self._open_mesh_btn, self._toggle_btn]:
            scene_g.layout().addWidget(b)
        root.addWidget(scene_g)

        # ── Camera section ────────────────────────────────────────────── #
        cam_g = self._group("Camera")
        self._fit_btn   = QPushButton("⌖  Fit to Scene")
        self._reset_btn = QPushButton("⟳  Reset Camera")
        for b in [self._fit_btn, self._reset_btn]:
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            cam_g.layout().addWidget(b)
        root.addWidget(cam_g)

        # ── Splat rendering ───────────────────────────────────────────── #
        rend_g = self._group("Splat Rendering")

        # Scale modifier
        scale_row = QVBoxLayout()
        sh = QHBoxLayout()
        sh.addWidget(QLabel("Scale modifier"))
        self._scale_lbl = QLabel("1.00")
        self._scale_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        sh.addWidget(self._scale_lbl)
        scale_row.addLayout(sh)
        self._scale_slider = QSlider(Qt.Horizontal)
        self._scale_slider.setRange(1, 200)   # /20 → 0.05 … 10.0
        self._scale_slider.setValue(20)
        scale_row.addWidget(self._scale_slider)
        rend_g.layout().addLayout(scale_row)

        # Shading mode
        sh2 = QHBoxLayout()
        sh2.addWidget(QLabel("Shading"))
        self._shading_combo = QComboBox()
        self._shading_combo.addItems(RENDER_MODES)
        self._shading_combo.setCurrentIndex(7)  # SH:0~3
        sh2.addWidget(self._shading_combo, 1)
        rend_g.layout().addLayout(sh2)

        # Auto-sort
        self._auto_sort_chk = QCheckBox("Auto-sort Gaussians")
        self._auto_sort_chk.setChecked(True)
        rend_g.layout().addWidget(self._auto_sort_chk)

        root.addWidget(rend_g)

        # ── Status ────────────────────────────────────────────────────── #
        status_g = self._group("Status")
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setWordWrap(True)
        status_g.layout().addWidget(self._status_lbl)
        root.addWidget(status_g)

        # ── Help ──────────────────────────────────────────────────────── #
        help_g = self._group("Mouse / Keys")
        tips = [
            ("Left drag",  "Orbit"),
            ("Right drag", "Pan"),
            ("Scroll",     "Zoom"),
            ("T",          "Toggle splat / mesh"),
            ("R",          "Reset camera"),
            ("F",          "Fit scene"),
        ]
        for key, desc in tips:
            row = QHBoxLayout()
            k = QLabel(key); k.setFixedWidth(72); k.setFont(QFont("Monospace", 9))
            d = QLabel(desc)
            row.addWidget(k); row.addWidget(d); row.addStretch()
            help_g.layout().addLayout(row)
        root.addWidget(help_g)

        root.addStretch()

        # ── Wire ──────────────────────────────────────────────────────── #
        self._open_ply_btn.clicked.connect(viewer._on_open_ply)
        self._open_mesh_btn.clicked.connect(viewer._on_open_mesh)
        self._toggle_btn.clicked.connect(viewer.toggle_representation)
        self._fit_btn.clicked.connect(viewer.fit_scene)
        self._reset_btn.clicked.connect(viewer.reset_camera)
        self._scale_slider.valueChanged.connect(self._on_scale)
        self._shading_combo.currentIndexChanged.connect(self._on_shading)
        self._auto_sort_chk.toggled.connect(self._on_auto_sort)

        viewer.sig_status.connect(self._status_lbl.setText)
        viewer.sig_representation_changed.connect(self._on_repr_changed)

    # ── Slots ─────────────────────────────────────────────────────────── #

    def _on_scale(self, step: int):
        val = step / 20.0
        self._scale_lbl.setText(f"{val:.2f}")
        gs = self._viewer.gs_renderer
        if gs:
            gs.set_scale_modifier(val)
            self._viewer.plotter.render()

    def _on_shading(self, idx: int):
        gs = self._viewer.gs_renderer
        if gs:
            gs.set_render_mod(idx - _RENDER_MODE_OFFSET)
            self._viewer.plotter.render()

    def _on_auto_sort(self, checked: bool):
        gs = self._viewer.gs_renderer
        if gs:
            gs._auto_sort = checked

    def _on_repr_changed(self, showing_splat: bool):
        self._toggle_btn.setText(
            "Switch to Mesh" if showing_splat else "Switch to Splat"
        )

    @staticmethod
    def _group(title: str) -> QGroupBox:
        g = QGroupBox(title)
        g.setLayout(QVBoxLayout())
        g.layout().setContentsMargins(8, 14, 8, 10)
        g.layout().setSpacing(6)
        return g


# ══════════════════════════════════════════════════════════════════════════════
# Unified viewer
# ══════════════════════════════════════════════════════════════════════════════

class UnifiedViewer(QMainWindow):
    """
    One window.  One scene.  Meshes, camera frustums, and 3DGS all together.

    The 3DGS splat is a *first-class vtkActor*, so everything that works on
    a mesh works on the splat too: focal-point click, camera-orbit, distance
    measurement, picking, etc.

    Persistent scene actors (frustums, axes, reference geometry) remain visible
    regardless of which representation is active.  Only the "subject" — the
    mesh(es) or the splat — is toggled.
    """

    sig_status                = pyqtSignal(str)
    sig_representation_changed = pyqtSignal(bool)   # True = splat active

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Unified 3DGS + Mesh Viewer")
        self.resize(1440, 900)

        # ── Central PyVista widget ─────────────────────────────────────── #
        self._plotter = QtInteractor(self)
        self._plotter.set_background(color=(0.08, 0.08, 0.10))
        ren_win = getattr(self._plotter, "ren_win", None)
        if ren_win is not None and hasattr(ren_win, "SetUseSRGBColorSpace"):
            ren_win.SetUseSRGBColorSpace(False)
        self.setCentralWidget(self._plotter)

        self._default_cam_pos = np.array([0.0, 0.0, 3.0], dtype=np.float32)
        self._default_cam_focal = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._default_cam_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        self.set_camera_pose(self._default_cam_pos, self._default_cam_focal, self._default_cam_up)

        # ── Actor registries ───────────────────────────────────────────── #
        self._mesh_actors:    list[vtk.vtkActor] = []   # toggled with splat
        self._frustum_actors: list[vtk.vtkActor] = []   # always visible
        self._overlay_actors: list[vtk.vtkActor] = []   # always visible (axes, etc.)

        # ── Representation state ───────────────────────────────────────── #
        self._show_splat = False   # hidden until data is loaded
        self._show_mesh  = True

        # ── 3DGS renderer (lazy: created after first GL context render) ── #
        self._gs_renderer: VTKGaussianRenderer | None = None
        self._gs_init_done = False
        self._plotter.add_on_render_callback(self._lazy_init_gs)

        # ── Background loader ──────────────────────────────────────────── #
        self._loader_thread: QThread | None = None
        self._is_loading = False

        # ── Side panel (dock) ──────────────────────────────────────────── #
        self._panel = _SidePanel(self)
        dock = QDockWidget("Controls", self)
        dock.setWidget(self._panel)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        # ── Status bar ─────────────────────────────────────────────────── #
        self._status_lbl = QLabel("Ready")
        self.statusBar().addWidget(self._status_lbl)
        self.sig_status.connect(self._status_lbl.setText)

        # ── Keyboard shortcuts ─────────────────────────────────────────── #
        self._plotter.add_key_event("t", self.toggle_representation)
        self._plotter.add_key_event("r", self.reset_camera)
        self._plotter.add_key_event("f", self.fit_scene)

        # Trigger first render so the GL context is ready immediately
        self._plotter.render()

    # ── Lazy GL init ───────────────────────────────────────────────────────── #

    def _lazy_init_gs(self, *_args):
        if not self._gs_init_done:
            self._gs_renderer = VTKGaussianRenderer(
                self._plotter.renderer, auto_sort=True
            )
            self._gs_renderer.actor.SetVisibility(False)
            self._gs_init_done = True

    # ══════════════════════════════════════════════════════════════════════════
    # Public scene-building API
    # ══════════════════════════════════════════════════════════════════════════

    def add_mesh(self, mesh_or_path, persistent: bool = False, **kwargs) -> vtk.vtkActor:
        """
        Add a mesh to the scene.

        Parameters
        ----------
        mesh_or_path : str | pyvista mesh
            File path or any PyVista / VTK mesh object.
        persistent : bool
            If True the mesh is always visible and not affected by
            toggle_representation().  Use for reference geometry, ground
            planes, bounding boxes, etc.
        **kwargs
            Forwarded to plotter.add_mesh().

        Returns
        -------
        vtkActor
            The actor so you can manipulate it later.
        """
        if isinstance(mesh_or_path, str):
            mesh = pv.read(mesh_or_path)
        else:
            mesh = mesh_or_path

        actor = self._plotter.add_mesh(mesh, **kwargs)
        if persistent:
            self._overlay_actors.append(actor)
        else:
            self._mesh_actors.append(actor)
            actor.SetVisibility(self._show_mesh)
        self._plotter.render()
        return actor

    def add_camera_frustum(
        self,
        c2w: np.ndarray,
        scale: float   = 0.3,
        color: str     = "yellow",
        line_width: float = 1.5,
    ) -> vtk.vtkActor:
        """
        Add a camera frustum.  Always visible — not toggled.

        Parameters
        ----------
        c2w : (4, 4) array
            Camera-to-world transform matrix.
        scale : float
            Controls frustum size in world units.
        """
        mesh  = _make_frustum(np.asarray(c2w, np.float32), scale)
        actor = self._plotter.add_mesh(
            mesh, color=color, style="wireframe",
            line_width=line_width, render_lines_as_tubes=False,
        )
        self._frustum_actors.append(actor)
        self._plotter.render()
        return actor

    def add_camera_frustums(
        self,
        c2w_list: Sequence[np.ndarray],
        scale: float = 0.15,
        color: str   = "cyan",
    ):
        """Add multiple camera frustums at once."""
        for c2w in c2w_list:
            self.add_camera_frustum(c2w, scale=scale, color=color)

    def load_splat(self, path: str):
        """
        Asynchronously load a .ply 3DGS file.

        The viewer stays responsive while loading.  On completion the splat
        actor is made visible if show_splat is True (or it was previously
        loaded).
        """
        if self._is_loading:
            return
        if not self._gs_init_done:
            self._plotter.render()  # ensure lazy init ran

        self._is_loading = True
        self.sig_status.emit(f"Loading {os.path.basename(path)}…")
        self._panel._open_ply_btn.setEnabled(False)

        worker = _PlyLoaderWorker(path)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_splat_loaded)
        worker.errored.connect(self._on_splat_error)
        worker.finished.connect(thread.quit)
        worker.errored.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        worker.setParent(thread)
        self._loader_thread = thread
        thread.start()

    def add_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        length: float = 5.0,
        color: str    = "white",
        radius: float = 0.005,
    ) -> vtk.vtkActor:
        """
        Draw a ray (e.g. a camera viewing ray) into the scene.
        Always visible — not affected by toggle.
        """
        origin    = np.asarray(origin,    dtype=np.float32)
        direction = np.asarray(direction, dtype=np.float32)
        direction = direction / np.linalg.norm(direction)
        end       = origin + direction * length
        line      = pv.Line(origin, end)
        actor     = self._plotter.add_mesh(
            line, color=color, line_width=2,
            render_lines_as_tubes=True, tube_radius=radius,
        )
        self._overlay_actors.append(actor)
        self._plotter.render()
        return actor

    # ══════════════════════════════════════════════════════════════════════════
    # Representation toggle
    # ══════════════════════════════════════════════════════════════════════════

    def toggle_representation(self):
        """
        Swap between mesh and 3DGS splat.
        Frustums, rays, and persistent actors are always visible.
        """
        self._show_splat = not self._show_splat
        self._show_mesh  = not self._show_mesh

        if self._gs_renderer:
            self._gs_renderer.actor.SetVisibility(self._show_splat)
        for a in self._mesh_actors:
            a.SetVisibility(self._show_mesh)

        self._plotter.render()
        self.sig_representation_changed.emit(self._show_splat)
        label = "Splat" if self._show_splat else "Mesh"
        self.sig_status.emit(f"Showing: {label}")

    def set_splat_visible(self, visible: bool):
        self._show_splat = visible
        if self._gs_renderer:
            self._gs_renderer.actor.SetVisibility(visible)
        self._plotter.render()

    def set_mesh_visible(self, visible: bool):
        self._show_mesh = visible
        for a in self._mesh_actors:
            a.SetVisibility(visible)
        self._plotter.render()

    # ══════════════════════════════════════════════════════════════════════════
    # Camera helpers
    # ══════════════════════════════════════════════════════════════════════════

    def fit_scene(self):
        if self._show_splat and self._gs_renderer and self._gs_renderer._gaussians is not None:
            self._frame_splat_like_original(self._gs_renderer._gaussians)
        else:
            self._plotter.reset_camera()
            self._plotter.render()

    def reset_camera(self):
        self.set_camera_pose(self._default_cam_pos, self._default_cam_focal, self._default_cam_up)

    def set_camera_pose(self, position, focal_point, view_up=(0, -1, 0)):
        cam = self._plotter.renderer.GetActiveCamera()
        cam.SetPosition(*position)
        cam.SetFocalPoint(*focal_point)
        cam.SetViewUp(*view_up)
        self._plotter.reset_camera_clipping_range()
        self._plotter.render()

    # ══════════════════════════════════════════════════════════════════════════
    # Properties
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def plotter(self) -> QtInteractor:
        return self._plotter

    @property
    def gs_renderer(self) -> VTKGaussianRenderer | None:
        return self._gs_renderer

    # ══════════════════════════════════════════════════════════════════════════
    # Internal slots
    # ══════════════════════════════════════════════════════════════════════════

    @pyqtSlot(object)
    def _on_splat_loaded(self, gaus: util_gau.GaussianData):
        self._gs_renderer.update_gaussian_data(gaus)
        # Show splat, hide mesh on first load
        self._show_splat = True
        self._show_mesh  = False
        self._gs_renderer.actor.SetVisibility(True)
        for a in self._mesh_actors:
            a.SetVisibility(False)

        self._plotter.render()
        self._frame_splat_like_original(gaus)

        self._is_loading = False
        self._panel._open_ply_btn.setEnabled(True)
        self.sig_status.emit(f"Loaded {len(gaus):,} Gaussians")
        self.sig_representation_changed.emit(True)

    @pyqtSlot(str)
    def _on_splat_error(self, msg: str):
        self._is_loading = False
        self._panel._open_ply_btn.setEnabled(True)
        self.sig_status.emit(f"Error loading PLY: {msg}")

    def _on_open_ply(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Gaussian Splat",
            os.path.expanduser("~"),
            "PLY files (*.ply)",
        )
        if path:
            self.load_splat(path)

    def _on_open_mesh(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Mesh",
            os.path.expanduser("~"),
            "Mesh files (*.ply *.obj *.stl *.vtk *.vtp *.glb *.gltf)",
        )
        if path:
            self.add_mesh(path, color="lightgrey", opacity=0.85)

    def _frame_splat_like_original(self, gaus: util_gau.GaussianData):
        xyz = gaus.xyz
        centroid = xyz.mean(axis=0).astype(np.float32)
        dists = np.linalg.norm(xyz - centroid, axis=1)
        radius = float(max(np.percentile(dists, 95), 0.5))

        cam = self._plotter.renderer.GetActiveCamera()
        position = np.array(cam.GetPosition(), dtype=np.float32)
        focal_point = np.array(cam.GetFocalPoint(), dtype=np.float32)
        direction = focal_point - position
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm
        else:
            direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        self.set_camera_pose(centroid - direction * radius * 2.5, centroid, self._default_cam_up)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    win = UnifiedViewer()

    # ── Demo: add a reference sphere ──────────────────────────────────── #
    sphere = pv.Sphere(radius=0.5)
    win.add_mesh(sphere, color="steelblue", opacity=0.4, persistent=True)

    # ── Demo: add a few fake camera frustums (identity-ish poses) ──────── #
    for i, angle in enumerate(np.linspace(0, np.pi * 1.5, 6)):
        c2w = np.eye(4, dtype=np.float32)
        c2w[0, 3] = 2.0 * np.cos(angle)
        c2w[2, 3] = 2.0 * np.sin(angle)
        win.add_camera_frustum(c2w, scale=0.2, color="yellow")

    win.show()
    sys.exit(app.exec_())
