"""
gs_control_panel.py
Sidebar control panel for the PyVistaQt Gaussian Splatting viewer.

Styled to match `control_panel.py`: uses the same `_group`, `_ValueSlider`,
and object names (`primaryBtn`, `monoLabel`, `statusLabel`, `kbdKey`, …) so
the stylesheet defined in `main.py` applies without changes.

Controls exposed:
  - PLY file loader
  - Opacity & scale-modifier sliders (with reset)
  - Bounding-box toggle
  - Reset camera + Save viewport image
  - Stats (FPS, gaussian count, status)
"""
from __future__ import annotations

import os
import time

from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)


# ─────────────────────────────────────────────────────────────────────────── #
#  Small reusable widgets (mirrored from control_panel.py)                    #
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
    """Label + slider + live value readout, optional reset button."""

    def __init__(self, label: str, lo: float, hi: float, value: float,
                 decimals: int = 2, unit: str = "",
                 show_reset: bool = False, parent=None):
        super().__init__(parent)
        self._lo, self._hi = lo, hi
        self._dec = decimals
        self._unit = unit
        self._steps = 1000
        self._callbacks: list = []
        self._reset_val = value

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

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
#  GSControlPanel                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class GSControlPanel(QWidget):
    """Sidebar panel for the GaussianSplatWidget. Wrap in a QDockWidget."""

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
        self._fps_lbl    = QLabel("FPS: —")
        self._count_lbl  = QLabel("Gaussians: —")
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

        # ── Rendering ─────────────────────────────────────────────────── #
        rend = _group("Rendering")
        self._opacity_slider = _ValueSlider(
            "Opacity", 0.0, 1.0, 0.99,
            decimals=2, show_reset=True,
        )
        self._opacity_slider.on_change(self._gw.set_opacity)

        self._scale_slider = _ValueSlider(
            "Scale Modifier", 0.1, 10.0, 1.0,
            decimals=2, show_reset=True,
        )
        self._scale_slider.on_change(self._gw.set_scale_modifier)

        rend.layout().addWidget(self._opacity_slider)
        rend.layout().addWidget(self._scale_slider)
        root.addWidget(rend)

        # ── Camera ────────────────────────────────────────────────────── #
        cam = _group("Camera")
        self._reset_btn = QPushButton("⟳  Reset Camera")
        self._reset_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._reset_btn.clicked.connect(self._gw.reset_camera)
        cam.layout().addWidget(self._reset_btn)
        root.addWidget(cam)

        # ── Export ────────────────────────────────────────────────────── #
        export = _group("Export")
        self._save_btn = QPushButton("Save Viewport Image")
        self._save_btn.clicked.connect(lambda: self._gw.save_image(None))
        export.layout().addWidget(self._save_btn)
        root.addWidget(export)

        # ── Help ──────────────────────────────────────────────────────── #
        help_g = _group("Mouse Controls")
        shortcuts = [
            ("Left drag",   "Orbit camera"),
            ("Middle drag", "Pan camera"),
            ("Scroll",      "Zoom in / out"),
            ("Dbl-click",   "Set focal point"),
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
            k.setFixedWidth(78)
            d = QLabel(desc)
            d.setObjectName("helpText")
            row.addWidget(k)
            row.addWidget(d)
            row.addStretch()
            grid.addLayout(row)
        help_g.layout().addWidget(grid_w)
        root.addWidget(help_g)

        root.addStretch()

        # ── Wire up widget signals ────────────────────────────────────── #
        self._gw.sig_fps_changed.connect(self._on_fps)
        self._gw.sig_gau_count_changed.connect(self._on_count)
        self._gw.sig_status_message.connect(self._on_status)
        self._gw.sig_loading_changed.connect(self._on_loading)

        # Controls disabled while a PLY is loading
        self._interactive: list[QWidget] = [
            self._open_btn, self._bbox_chk,
            self._opacity_slider, self._scale_slider,
            self._reset_btn, self._save_btn,
        ]

        # Loading elapsed-time ticker
        self._load_timer = QTimer(self)
        self._load_timer.setInterval(250)
        self._load_timer.timeout.connect(self._update_load_elapsed)

    # ── Slots ─────────────────────────────────────────────────────────── #

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

    def _on_open_ply(self):
        start_dir = os.path.dirname(os.path.abspath(__file__))
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Gaussian Splatting PLY",
            start_dir,
            "PLY files (*.ply)"
        )
        if path:
            self._gw.load_ply(path)