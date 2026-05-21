"""
gs_control_panel.py
Sidebar control panel for the PyVistaQt Gaussian Splatting viewer.

Mirrors control_panel.py feature-for-feature:
  - Stats: FPS, Gaussian count, status label
  - Scene: Open .ply, bounding-box toggle
  - Camera: FOV slider, Fit to Scene, Reset Camera, Flip Ground
  - Rendering: Scale modifier, Opacity, Shading/render-mode combo
  - Gaussian Sorting: Sort Now, Auto-sort toggle
  - Export: Save viewport image
  - Help: keyboard & mouse shortcuts

All controls use the same QSS object names (primaryBtn, monoLabel, …) as
control_panel.py so the main-window stylesheet applies without changes.
"""
from __future__ import annotations

import os
import time

from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from native_actor_gs import RENDER_MODES


# ─────────────────────────────────────────────────────────────────────────── #
#  Small reusable helpers (verbatim from control_panel.py)                    #
# ─────────────────────────────────────────────────────────────────────────── #

def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setObjectName("separator")
    return f


def _group(title: str) -> QGroupBox:
    g = QGroupBox(title)
    g.setLayout(QVBoxLayout())
    g.layout().setContentsMargins(8, 14, 8, 10)
    g.layout().setSpacing(6)
    return g


class _ValueSlider(QWidget):
    """Label + slider + live value readout with optional reset button."""

    def __init__(self, label: str, lo: float, hi: float, value: float,
                 decimals: int = 1, unit: str = "",
                 show_reset: bool = False, parent=None):
        super().__init__(parent)
        self._lo, self._hi = lo, hi
        self._dec       = decimals
        self._unit      = unit
        self._steps     = 1000
        self._callbacks: list = []
        self._reset_val = value

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        # Row 1: label + value readout
        top = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setObjectName("sliderLabel")
        self._val_lbl = QLabel(self._fmt(value))
        self._val_lbl.setObjectName("valueLabel")
        self._val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._val_lbl.setMinimumWidth(52)
        top.addWidget(lbl)
        top.addStretch()
        top.addWidget(self._val_lbl)
        root.addLayout(top)

        # Row 2: slider (+ optional reset)
        row = QHBoxLayout()
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, self._steps)
        self._slider.setValue(self._to_step(value))
        self._slider.valueChanged.connect(self._on_change)
        row.addWidget(self._slider)

        if show_reset:
            btn = QPushButton("↺")
            btn.setObjectName("resetBtn")
            btn.setFixedWidth(28)
            btn.setToolTip("Reset to default")
            btn.clicked.connect(self._on_reset)
            row.addWidget(btn)

        root.addLayout(row)

    # Public
    def on_change(self, fn):    self._callbacks.append(fn)
    def value(self) -> float:   return self._to_float(self._slider.value())

    def set_value(self, v: float):
        self._slider.blockSignals(True)
        self._slider.setValue(self._to_step(v))
        self._val_lbl.setText(self._fmt(v))
        self._slider.blockSignals(False)

    def setEnabled(self, enabled: bool):
        super().setEnabled(enabled)
        self._slider.setEnabled(enabled)

    # Private
    def _fmt(self, v):      return f"{v:.{self._dec}f}{self._unit}"
    def _to_step(self, v):  return int((v - self._lo) / (self._hi - self._lo) * self._steps)
    def _to_float(self, s): return self._lo + s / self._steps * (self._hi - self._lo)

    def _on_change(self, step):
        v = self._to_float(step)
        self._val_lbl.setText(self._fmt(v))
        for fn in self._callbacks:
            fn(v)

    def _on_reset(self):
        self.set_value(self._reset_val)
        for fn in self._callbacks:
            fn(self._reset_val)


# ─────────────────────────────────────────────────────────────────────────── #
#  GSControlPanel                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class GSControlPanel(QWidget):
    """Sidebar panel for GaussianSplatWidget. Wrap in a QDockWidget."""

    def __init__(self, splat_widget, parent=None):
        super().__init__(parent)
        self._gw = splat_widget
        self._load_start_t: float | None = None

        self.setObjectName("controlPanel")
        self.setFixedWidth(320)

        # ── Scroll area wraps all content ─────────────────────────────── #
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

        # ── Stats ─────────────────────────────────────────────────────── #
        stats = _group("Stats")
        self._fps_lbl   = QLabel("FPS: —")
        self._count_lbl = QLabel("Gaussians: —")
        self._status_lbl = QLabel("Initialising…")
        self._status_lbl.setWordWrap(True)
        for w in (self._fps_lbl, self._count_lbl):
            w.setObjectName("monoLabel")
            stats.layout().addWidget(w)
        self._status_lbl.setObjectName("statusLabel")
        stats.layout().addWidget(self._status_lbl)
        root.addWidget(stats)

        # ── Scene ─────────────────────────────────────────────────────── #
        scene = _group("Scene")
        self._open_btn = QPushButton("Open .ply File…")
        self._open_btn.setObjectName("primaryBtn")
        self._open_btn.clicked.connect(self._on_open_ply)
        scene.layout().addWidget(self._open_btn)

        self._bbox_chk = QCheckBox("Show bounding box")
        self._bbox_chk.setChecked(True)
        self._bbox_chk.toggled.connect(self._gw.set_bbox_visible)
        scene.layout().addWidget(self._bbox_chk)
        root.addWidget(scene)

        # ── Camera ────────────────────────────────────────────────────── #
        cam = _group("Camera")

        self._fov_slider = _ValueSlider(
            "Field of View", 5.0, 170.0,
            self._gw.fovy_deg(),
            decimals=1, unit="°", show_reset=True,
        )
        self._fov_slider.on_change(self._gw.set_fovy_deg)
        cam.layout().addWidget(self._fov_slider)

        cam_btns = QHBoxLayout()
        self._fit_btn   = QPushButton("⌖  Fit to Scene")
        self._reset_btn = QPushButton("⟳  Reset Camera")
        for b in (self._fit_btn, self._reset_btn):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._fit_btn.clicked.connect(self._gw.fit_camera_to_gaussians)
        self._reset_btn.clicked.connect(self._gw.reset_camera)
        cam_btns.addWidget(self._fit_btn)
        cam_btns.addWidget(self._reset_btn)
        cam.layout().addLayout(cam_btns)

        self._flip_btn = QPushButton("↕  Flip Ground")
        self._flip_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._flip_btn.clicked.connect(self._gw.flip_ground)
        cam.layout().addWidget(self._flip_btn)

        root.addWidget(cam)

        # ── Rendering ─────────────────────────────────────────────────── #
        rend = _group("Rendering")

        self._scale_slider = _ValueSlider(
            "Scale Modifier", 0.1, 10.0,
            self._gw.scale_modifier,
            decimals=2, show_reset=True,
        )
        self._scale_slider.on_change(self._gw.set_scale_modifier)
        rend.layout().addWidget(self._scale_slider)

        self._opacity_slider = _ValueSlider(
            "Opacity", 0.0, 1.0, 0.99,
            decimals=2, show_reset=True,
        )
        self._opacity_slider.on_change(self._gw.set_opacity)
        rend.layout().addWidget(self._opacity_slider)

        shading_row = QHBoxLayout()
        shading_lbl = QLabel("Shading")
        shading_lbl.setObjectName("sliderLabel")
        self._shading_combo = QComboBox()
        self._shading_combo.addItems(RENDER_MODES)
        self._shading_combo.setCurrentIndex(self._gw.render_mode)
        self._shading_combo.currentIndexChanged.connect(self._gw.set_render_mode)
        shading_row.addWidget(shading_lbl)
        shading_row.addWidget(self._shading_combo, 1)
        rend.layout().addLayout(shading_row)

        root.addWidget(rend)

        # ── Gaussian Sorting ──────────────────────────────────────────── #
        sort_grp = _group("Gaussian Sorting")
        sort_row = QHBoxLayout()
        self._sort_btn = QPushButton("Sort Now")
        self._sort_btn.clicked.connect(self._gw.sort_gaussians)
        self._auto_sort_chk = QCheckBox("Auto-sort")
        self._auto_sort_chk.setToolTip("Re-sort every ~80 ms as the camera moves")
        self._auto_sort_chk.toggled.connect(self._on_auto_sort)
        sort_row.addWidget(self._sort_btn)
        sort_row.addWidget(self._auto_sort_chk)
        sort_grp.layout().addLayout(sort_row)
        root.addWidget(sort_grp)

        # ── Export ────────────────────────────────────────────────────── #
        export = _group("Export")
        self._save_btn = QPushButton("Save Viewport Image")
        self._save_btn.clicked.connect(lambda: self._gw.save_image(None))
        export.layout().addWidget(self._save_btn)
        root.addWidget(export)

        # ── Help ──────────────────────────────────────────────────────── #
        help_g = _group("Keyboard & Mouse")
        shortcuts = [
            ("Left drag",   "Orbit camera"),
            ("Middle drag", "Pan camera"),
            ("Scroll",      "Zoom in / out"),
            ("Q / E",       "Roll camera"),
            ("F",           "Fit scene to view"),
            ("R",           "Reset camera"),
            ("Dbl-click",   "Set focal point"),
        ]
        grid_w = QWidget()
        grid = QVBoxLayout(grid_w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(3)
        for key, desc in shortcuts:
            row = QHBoxLayout()
            k = QLabel(key)
            k.setObjectName("kbdKey")
            k.setFixedWidth(80)
            d = QLabel(desc)
            d.setObjectName("helpText")
            row.addWidget(k)
            row.addWidget(d)
            row.addStretch()
            grid.addLayout(row)
        help_g.layout().addWidget(grid_w)
        root.addWidget(help_g)

        root.addStretch()

        # ── Wire up GaussianSplatWidget signals ───────────────────────── #
        self._gw.sig_fps_changed.connect(self._on_fps)
        self._gw.sig_gau_count_changed.connect(self._on_count)
        self._gw.sig_status_message.connect(self._on_status)
        self._gw.sig_loading_changed.connect(self._on_loading)

        # Controls disabled while a PLY is loading
        self._interactive: list[QWidget] = [
            self._open_btn, self._bbox_chk,
            self._fov_slider, self._scale_slider, self._opacity_slider,
            self._shading_combo,
            self._fit_btn, self._reset_btn, self._flip_btn,
            self._sort_btn, self._auto_sort_chk, self._save_btn,
        ]

        # Loading elapsed-time ticker
        self._load_timer = QTimer(self)
        self._load_timer.setInterval(250)
        self._load_timer.timeout.connect(self._update_load_elapsed)

        # Sync FOV once the interactor is ready
        QTimer.singleShot(400, self._refresh_fov)

    # ── Slots ──────────────────────────────────────────────────────────── #

    @pyqtSlot(float)
    def _on_fps(self, fps: float):
        self._fps_lbl.setText(f"FPS: {fps:.1f}")

    @pyqtSlot(int)
    def _on_count(self, n: int):
        self._count_lbl.setText(f"Gaussians: {n:,}")

    @pyqtSlot(str)
    def _on_status(self, msg: str):
        self._status_lbl.setText(msg)

    @pyqtSlot(bool)
    def _on_loading(self, loading: bool):
        for w in self._interactive:
            w.setEnabled(not loading)
        if loading:
            self._load_start_t = time.perf_counter()
            self._load_timer.start()
        else:
            self._load_timer.stop()
            self._load_start_t = None
            # Sync FOV slider after a new file loads
            QTimer.singleShot(100, self._refresh_fov)

    def _update_load_elapsed(self):
        if self._load_start_t is not None:
            elapsed = time.perf_counter() - self._load_start_t
            self._status_lbl.setText(f"Loading… {elapsed:.1f}s")

    def _on_open_ply(self):
        start_dir = os.path.dirname(os.path.abspath(__file__))
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Gaussian Splatting PLY",
            start_dir,
            "PLY files (*.ply)",
        )
        if path:
            self._gw.load_ply(path)

    def _on_auto_sort(self, checked: bool):
        self._gw.auto_sort = checked

    def _refresh_fov(self):
        """Sync the FOV slider to the widget's current camera angle."""
        try:
            self._fov_slider.set_value(self._gw.fovy_deg())
        except Exception:
            pass
