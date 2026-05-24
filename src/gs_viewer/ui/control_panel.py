"""
control_panel.py
Sidebar control panel for the Gaussian Splatting viewer.

Changes vs v1:
  - Wrapped in QScrollArea so it works on small screens
  - FOV shown and edited in degrees (not radians)
  - "Fit to Scene" and "Reset Camera" buttons
  - All interactive controls disabled while a PLY is loading
  - Loading status shows elapsed seconds
  - Keyboard shortcuts listed in the help section
"""
from __future__ import annotations

import os
import time

from PyQt5.QtWidgets import (
    QWidget, QScrollArea, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QComboBox, QCheckBox,
    QFileDialog, QFrame, QGroupBox, QSizePolicy,
    QDoubleSpinBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
 


def _group(title: str) -> QGroupBox:
    g = QGroupBox(title)
    g.setLayout(QVBoxLayout())
    g.layout().setContentsMargins(8, 14, 8, 10)
    g.layout().setSpacing(6)
    return g


class _ValueSlider(QWidget):
    """Label + slider + live value readout, optional reset button."""

    def __init__(self, label: str, lo: float, hi: float, value: float,
                 decimals: int = 1, unit: str = "",
                 show_reset: bool = False, parent=None):
        super().__init__(parent)
        self._lo, self._hi = lo, hi
        self._dec   = decimals
        self._unit  = unit
        self._steps = 1000
        self._callbacks: list = []
        self._reset_val = value

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        # Row 1: label + value
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
    def on_change(self, fn): self._callbacks.append(fn)
    def value(self) -> float: return self._to_float(self._slider.value())

    def set_value(self, v: float):
        self._slider.blockSignals(True)
        self._slider.setValue(self._to_step(v))
        self._val_lbl.setText(self._fmt(v))
        self._slider.blockSignals(False)

    def setEnabled(self, enabled: bool):
        super().setEnabled(enabled)
        self._slider.setEnabled(enabled)

    # Private
    def _fmt(self, v):
        return f"{v:.{self._dec}f}{self._unit}"

    def _to_step(self, v):
        return int((v - self._lo) / (self._hi - self._lo) * self._steps)

    def _to_float(self, s):
        return self._lo + s / self._steps * (self._hi - self._lo)

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
#  ControlPanel                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

RENDER_MODES = [
    "Gaussian Ball", "Flat Ball", "Billboard",
    "Depth", "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)",
]


class ControlPanel(QWidget):
    """
    Sidebar panel. Wrap it in a QDockWidget in your main window.
    Pass the GaussianWidget as `gaussian_widget`.
    """

    def __init__(self, gaussian_widget, parent=None):
        super().__init__(parent)
        self._gw = gaussian_widget
        self._load_start_t: float | None = None

        self.setObjectName("controlPanel")
        self.setFixedWidth(400)

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
        self._fps_lbl    = QLabel("FPS: —")
        self._count_lbl  = QLabel("Gaussians: —")
        self._status_lbl = QLabel("Initialising…")
        self._status_lbl.setWordWrap(True)
        for w in [self._fps_lbl, self._count_lbl]:
            w.setObjectName("monoLabel")
            stats.layout().addWidget(w)
        self._status_lbl.setObjectName("statusLabel")
        stats.layout().addWidget(self._status_lbl)
        root.addWidget(stats)

        # ── Backend ───────────────────────────────────────────────────── #
        backend = _group("Renderer Backend")
        self._backend_combo = QComboBox()
        self._backend_combo.addItems(["OpenGL"])
        self._backend_combo.currentIndexChanged.connect(self._gw.set_backend)
        self._reduce_chk = QCheckBox("Reduce updates")
        self._reduce_chk.setChecked(True)
        self._reduce_chk.setToolTip("Skip re-renders when nothing has changed")
        self._reduce_chk.toggled.connect(self._gw.set_reduce_updates)
        backend.layout().addWidget(self._backend_combo)
        backend.layout().addWidget(self._reduce_chk)
        root.addWidget(backend)

        # ── Scene ─────────────────────────────────────────────────────── #
        scene = _group("Scene")
        self._open_btn = QPushButton("Open .ply File…")
        self._open_btn.setObjectName("primaryBtn")
        self._open_btn.clicked.connect(self._on_open_ply)
        scene.layout().addWidget(self._open_btn)
        root.addWidget(scene)

        # ── Camera ────────────────────────────────────────────────────── #
        cam = _group("Camera")

        self._fov_slider = _ValueSlider(
            "Field of View", 5.0, 170.0,
            self._gw.fovy_deg(),
            decimals=1, unit="°", show_reset=True
        )
        self._fov_slider.on_change(self._gw.set_fovy_deg)

        focal_row = QHBoxLayout()
        focal_lbl = QLabel("Focal Pt (X,Y,Z)")
        focal_lbl.setObjectName("sliderLabel")
        focal_row.addWidget(focal_lbl)

        self._focal_spinners: list[QDoubleSpinBox] = []
        current_focal = self._gw.focal_point()
        for i in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(-1000.0, 1000.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
            spin.setValue(float(current_focal[i]))
            spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
            spin.setStyleSheet("background: #1e2130; border: 1px solid #2e3348; color: #c8d0ea; padding: 2px;")
            spin.valueChanged.connect(self._on_focal_changed)
            self._focal_spinners.append(spin)
            focal_row.addWidget(spin)

        cam_btns = QHBoxLayout()
        self._fit_btn   = QPushButton("⌖  Fit to Scene")
        self._reset_btn = QPushButton("⟳  Reset Camera")
        self._flip_btn  = QPushButton("↕  Flip Ground")
        for b in [self._fit_btn, self._reset_btn, self._flip_btn]:
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._fit_btn.clicked.connect(self._gw.fit_camera_to_gaussians)
        self._reset_btn.clicked.connect(self._gw.reset_camera)
        self._flip_btn.clicked.connect(self._gw.flip_ground)
        cam_btns.addWidget(self._fit_btn)
        cam_btns.addWidget(self._reset_btn)

        cam.layout().addWidget(self._fov_slider)
        cam.layout().addLayout(focal_row)
        cam.layout().addLayout(cam_btns)
        cam.layout().addWidget(self._flip_btn)
        root.addWidget(cam)

        # ── Rendering ─────────────────────────────────────────────────── #
        rend = _group("Rendering")
        self._scale_slider = _ValueSlider(
            "Scale Modifier", 0.1, 10.0,
            self._gw.scale_modifier,
            decimals=2, show_reset=True
        )
        self._scale_slider.on_change(self._gw.set_scale_modifier)

        shading_row = QHBoxLayout()
        shading_row.addWidget(QLabel("Shading"))
        self._shading_combo = QComboBox()
        self._shading_combo.addItems(RENDER_MODES)
        self._shading_combo.setCurrentIndex(self._gw.render_mode)
        self._shading_combo.currentIndexChanged.connect(self._gw.set_render_mode)
        shading_row.addWidget(self._shading_combo, 1)

        rend.layout().addWidget(self._scale_slider)
        rend.layout().addLayout(shading_row)
        root.addWidget(rend)

        # ── Sorting ───────────────────────────────────────────────────── #
        sort = _group("Gaussian Sorting")
        sort_row = QHBoxLayout()
        self._sort_btn = QPushButton("Sort Now")
        self._sort_btn.clicked.connect(self._gw.sort_gaussians)
        self._auto_sort_chk = QCheckBox("Auto-sort")
        self._auto_sort_chk.setToolTip("Re-sort every ~80 ms as camera moves")
        self._auto_sort_chk.toggled.connect(self._on_auto_sort)
        sort_row.addWidget(self._sort_btn)
        sort_row.addWidget(self._auto_sort_chk)
        sort.layout().addLayout(sort_row)
        root.addWidget(sort)

        # ── Export ────────────────────────────────────────────────────── #
        export = _group("Export")
        self._save_btn = QPushButton("Save Viewport Image")
        self._save_btn.clicked.connect(self._on_save)
        export.layout().addWidget(self._save_btn)
        root.addWidget(export)

        # ── Help ──────────────────────────────────────────────────────── #
        help_g = _group("Keyboard & Mouse")
        shortcuts = [
            ("Left drag",   "Orbit camera"),
            ("Right drag",  "Pan camera"),
            ("Scroll",      "Zoom in / out"),
            ("Q / E",       "Roll camera"),
            ("F",           "Fit scene to view"),
            ("R",           "Reset camera"),
        ]
        grid_w = QWidget()
        grid = QVBoxLayout(grid_w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(3)
        for key, desc in shortcuts:
            row = QHBoxLayout()
            k = QLabel(key)
            k.setObjectName("kbdKey")
            k.setFixedWidth(72)
            d = QLabel(desc)
            d.setObjectName("helpText")
            row.addWidget(k)
            row.addWidget(d)
            row.addStretch()
            grid.addLayout(row)
        help_g.layout().addWidget(grid_w)
        root.addWidget(help_g)

        root.addStretch()

        # ── Wire up GaussianWidget signals ────────────────────────────── #
        self._gw.sig_fps_changed.connect(self._on_fps)
        self._gw.sig_gau_count_changed.connect(self._on_count)
        self._gw.sig_status_message.connect(self._on_status)
        self._gw.sig_loading_changed.connect(self._on_loading)
        self._gw.sig_focal_changed.connect(self._on_focal_updated_from_view)

        # Collect all interactive controls (disabled during load)
        self._interactive: list[QWidget] = [
            self._open_btn, self._backend_combo,
            self._fov_slider, self._scale_slider, self._shading_combo,
            *self._focal_spinners,
            self._fit_btn, self._reset_btn, self._flip_btn,
            self._sort_btn, self._auto_sort_chk, self._save_btn,
        ]

        # Populate backend combo after GL initialises
        QTimer.singleShot(600, self._refresh_backend_combo)

        # Loading elapsed-time ticker
        self._load_timer = QTimer(self)
        self._load_timer.setInterval(250)
        self._load_timer.timeout.connect(self._update_load_elapsed)

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

    def _update_load_elapsed(self):
        if self._load_start_t is not None:
            elapsed = time.perf_counter() - self._load_start_t
            self._status_lbl.setText(f"Loading… {elapsed:.1f}s")

    @pyqtSlot(float, float, float)
    def _on_focal_updated_from_view(self, x: float, y: float, z: float):
        coords = [x, y, z]
        for i, spin in enumerate(self._focal_spinners):
            spin.blockSignals(True)
            spin.setValue(coords[i])
            spin.blockSignals(False)

    def _on_focal_changed(self, *_):
        x = self._focal_spinners[0].value()
        y = self._focal_spinners[1].value()
        z = self._focal_spinners[2].value()
        self._gw.set_focal_point(x, y, z)

    def _on_open_ply(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Gaussian Splatting PLY",
            os.path.expanduser("~"),
            "PLY files (*.ply)"
        )
        if path:
            self._gw.load_ply(path)

    def _on_auto_sort(self, checked: bool):
        self._gw.auto_sort = checked

    def _on_save(self):
        self._gw.save_image()

    def _refresh_backend_combo(self):
        names = self._gw.backend_names()
        self._backend_combo.blockSignals(True)
        self._backend_combo.clear()
        self._backend_combo.addItems(names)
        self._backend_combo.setCurrentIndex(self._gw.current_backend_idx())
        self._backend_combo.blockSignals(False)
        self._on_count(self._gw.gaussian_count())
