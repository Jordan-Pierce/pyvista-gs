from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from pyvistaqt import QtInteractor

from PyQt5.QtWidgets import QApplication, QMainWindow, QDockWidget, QAction, QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QEvent
from PyQt5.QtGui import QFont, QSurfaceFormat, QPalette, QColor

from . import data as util_gau
from .actor import GaussianActor, VTKCameraAdapter
from .ui.control_panel import ControlPanel


# Must be set before QApplication.
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


def _build_dark_palette() -> QPalette:
    pal = QPalette()
    base = QColor("#0f1117")
    alt_base = QColor("#181b24")
    window = QColor("#13161f")
    text = QColor("#d4d8e8")
    bright = QColor("#ffffff")
    mid = QColor("#2a2d3a")
    highlight = QColor("#3d7aed")
    disabled = QColor("#555870")

    pal.setColor(QPalette.Window, window)
    pal.setColor(QPalette.WindowText, text)
    pal.setColor(QPalette.Base, base)
    pal.setColor(QPalette.AlternateBase, alt_base)
    pal.setColor(QPalette.ToolTipBase, base)
    pal.setColor(QPalette.ToolTipText, text)
    pal.setColor(QPalette.Text, text)
    pal.setColor(QPalette.Button, mid)
    pal.setColor(QPalette.ButtonText, text)
    pal.setColor(QPalette.BrightText, bright)
    pal.setColor(QPalette.Link, highlight)
    pal.setColor(QPalette.Highlight, highlight)
    pal.setColor(QPalette.HighlightedText, bright)
    pal.setColor(QPalette.Disabled, QPalette.Text, disabled)
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, disabled)
    return pal


STYLESHEET = """
QMainWindow, QWidget {
    background-color: #13161f;
    color: #d4d8e8;
    font-family: "JetBrains Mono", "Cascadia Code", "Fira Code", monospace;
    font-size: 12px;
}

QDockWidget {
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}
QDockWidget::title {
    background: #1e2130;
    padding: 6px 10px;
    font-weight: 600;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7b8ec8;
    border-bottom: 1px solid #2a2d3a;
}

QGroupBox {
    border: 1px solid #2a2d3a;
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 6px;
    background: #0f1117;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: 2px;
    color: #5a7ec8;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

QPushButton {
    background-color: #1e2130;
    color: #c8d0ea;
    border: 1px solid #2e3348;
    border-radius: 5px;
    padding: 5px 12px;
    font-size: 12px;
    min-height: 26px;
}
QPushButton:hover {
    background-color: #252a3d;
    border-color: #3d5aad;
    color: #e8ecff;
}
QPushButton:pressed {
    background-color: #1a1f30;
    border-color: #3d7aed;
}

QPushButton#primaryBtn {
    background-color: #1e3a6e;
    border-color: #3d7aed;
    color: #e0eaff;
    font-weight: 600;
}
QPushButton#primaryBtn:hover {
    background-color: #2347a0;
    border-color: #5090ff;
}

QPushButton#resetBtn {
    background-color: #1e2130;
    border: 1px solid #2e3348;
    border-radius: 4px;
    color: #7b8ec8;
    padding: 2px 4px;
    font-size: 13px;
    min-height: 20px;
}
QPushButton#resetBtn:hover {
    color: #aac0ff;
    border-color: #3d5aad;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #252a3d;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #3d7aed;
    border: none;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #5090ff;
}
QSlider::sub-page:horizontal {
    background: #2e4a90;
    border-radius: 2px;
}

QComboBox {
    background-color: #1e2130;
    border: 1px solid #2e3348;
    border-radius: 5px;
    padding: 4px 8px;
    color: #c8d0ea;
    min-height: 24px;
}
QComboBox::drop-down {
    border: none;
    width: 22px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #7b8ec8;
    width: 0;
    height: 0;
    margin-right: 6px;
}
QComboBox QAbstractItemView {
    background-color: #1a1d28;
    border: 1px solid #3d5aad;
    selection-background-color: #2347a0;
    color: #d4d8e8;
    outline: none;
}

QCheckBox {
    spacing: 8px;
    color: #b0bada;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #3a3f58;
    border-radius: 3px;
    background: #1a1d28;
}
QCheckBox::indicator:checked {
    background-color: #3d7aed;
    border-color: #3d7aed;
}
QCheckBox::indicator:hover {
    border-color: #5090ff;
}

QLabel#monoLabel {
    font-family: "JetBrains Mono", monospace;
    font-size: 12px;
    color: #7b9aed;
    padding: 1px 0;
}
QLabel#sliderLabel {
    font-size: 11px;
    color: #8892b0;
    letter-spacing: 0.04em;
}
QLabel#valueLabel {
    font-family: "JetBrains Mono", monospace;
    font-size: 12px;
    color: #aac0ff;
}
QLabel#statusLabel {
    font-size: 10px;
    color: #556080;
    font-style: italic;
}
QLabel#helpText {
    color: #6070a0;
    font-size: 11px;
}
QLabel#kbdKey {
    font-family: "JetBrains Mono", monospace;
    font-size: 10px;
    color: #aac0ff;
    background: #1a1f32;
    border: 1px solid #2e3a5a;
    border-radius: 3px;
    padding: 1px 5px;
}

QScrollBar:vertical {
    background: #0f1117;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #2a2d3a;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #3d5aad;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QStatusBar {
    background: #0c0e15;
    color: #4a5470;
    font-size: 11px;
    border-top: 1px solid #1e2130;
}
QStatusBar QLabel {
    color: #4a5470;
}

QMenuBar {
    background: #0c0e15;
    color: #8892b0;
    border-bottom: 1px solid #1e2130;
    font-size: 12px;
}
QMenuBar::item:selected {
    background: #1e2130;
    color: #d4d8e8;
}
QMenu {
    background: #13161f;
    border: 1px solid #2a2d3a;
    color: #d4d8e8;
}
QMenu::item:selected {
    background: #1e3a6e;
    color: #e0eaff;
}

QFrame#separator {
    color: #1e2130;
    max-height: 1px;
}

QScrollArea {
    border: none;
    background: transparent;
}
"""


class MainWindow(QMainWindow):
    sig_fps_changed = pyqtSignal(float)
    sig_gau_count_changed = pyqtSignal(int)
    sig_status_message = pyqtSignal(str)
    sig_loading_changed = pyqtSignal(bool)
    sig_focal_changed = pyqtSignal(float, float, float)

    def __init__(self, hidpi: bool = False):
        super().__init__()
        self.setWindowTitle("Gaussian Splatting Viewer (PyVista Edition)")
        self.resize(1440, 860)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#13161f")
        self.setCentralWidget(self.plotter)

        self.gs_actor: GaussianActor | None = None

        self.plotter.interactor.installEventFilter(self)

        self._panel = ControlPanel(self, self)
        dock = QDockWidget("Controls", self)
        dock.setWidget(self._panel)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        menu = self.menuBar()
        view_menu = menu.addMenu("View")
        toggle_panel = QAction("Show Control Panel", self, checkable=True, checked=True)
        toggle_panel.triggered.connect(dock.setVisible)
        view_menu.addAction(toggle_panel)

        tools_menu = menu.addMenu("Tools")
        self.action_crop = QAction("Crop Preview", self, checkable=True)
        self.action_crop.triggered.connect(self._toggle_crop)
        tools_menu.addAction(self.action_crop)
        self.action_apply_crop = QAction("Apply Crop", self)
        self.action_apply_crop.triggered.connect(self.apply_crop_box)
        tools_menu.addAction(self.action_apply_crop)

        self._status_bar = self.statusBar()
        self._status_lbl = QLabel("Ready - Please open a .ply file")
        self._status_bar.addPermanentWidget(self._status_lbl)
        self.sig_status_message.connect(self._status_lbl.setText)

        if hidpi:
            QApplication.instance().setFont(QFont(QApplication.font().family(), 14))

    def _toggle_crop(self, state: bool):
        self.set_crop_box_enabled(state)

    def _sync_crop_controls(self, enabled: bool):
        self.action_crop.blockSignals(True)
        self.action_crop.setChecked(enabled)
        self.action_crop.blockSignals(False)

        crop_chk = getattr(self._panel, "_crop_chk", None)
        if crop_chk is not None:
            crop_chk.blockSignals(True)
            crop_chk.setChecked(enabled)
            crop_chk.blockSignals(False)

    def set_crop_box_enabled(self, state: bool):
        if not self.gs_actor:
            self._sync_crop_controls(False)
            self.sig_status_message.emit("Please load a model first.")
            return

        if state:
            self.plotter.add_box_widget(
                callback=self._on_crop_box_modified,
                bounds=self.gs_actor._original_mesh.bounds,
                color="white",
                outline_translation=False,
                pass_widget=False,
            )
            self.gs_actor.set_crop_bounds(self.gs_actor._original_mesh.bounds)
            self._sync_crop_controls(True)
            self.sig_status_message.emit("Crop preview enabled. Drag the box to preview clipping.")
        else:
            self.plotter.clear_box_widgets()
            self.gs_actor.clear_crop_box()
            self.plotter.update()
            self._sync_crop_controls(False)
            self.sig_status_message.emit("Crop preview disabled.")

    def _on_crop_box_modified(self, bounds):
        if self.gs_actor:
            self.gs_actor.set_crop_bounds(bounds)
            self.plotter.update()

    def apply_crop_box(self):
        if not self.gs_actor:
            return

        self.gs_actor.apply_crop_box()
        self.plotter.update()
        self.sig_gau_count_changed.emit(self.gaussian_count())
        self.sig_status_message.emit(f"Applied crop. Remaining splats: {self.gaussian_count():,}")

    def load_ply(self, path: str):
        if not os.path.exists(path):
            self.sig_status_message.emit(f"File not found: {path}")
            return

        self.sig_loading_changed.emit(True)
        self.sig_status_message.emit(f"Loading {os.path.basename(path)}...")
        QApplication.processEvents()

        try:
            if self.gs_actor:
                self.gs_actor.clear_crop_box()
                self.plotter.clear_box_widgets()
                self._sync_crop_controls(False)
                self.gs_actor.cleanup()
                self.gs_actor = None

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
        _, h = self.plotter.window_size

        renderer = self.plotter.renderer
        renderer.SetDisplayPoint(x, y, 0.0)
        renderer.DisplayToWorld()
        ray_origin = np.array(renderer.GetWorldPoint()[:3], dtype=np.float32)

        renderer.SetDisplayPoint(x, y, 1.0)
        renderer.DisplayToWorld()
        ray_target = np.array(renderer.GetWorldPoint()[:3], dtype=np.float32)

        ray_dir = ray_target - ray_origin
        norm = np.linalg.norm(ray_dir)
        if norm == 0:
            return
        ray_dir /= norm

        fovy_rad = np.radians(self.plotter.camera.view_angle)

        hit_pos = self.gs_actor.pick_gaussian(ray_origin, ray_dir, fovy_rad, h)

        if hit_pos is not None:
            self.set_focal_point(float(hit_pos[0]), float(hit_pos[1]), float(hit_pos[2]))
            self.sig_focal_changed.emit(hit_pos[0], hit_pos[1], hit_pos[2])
            self.sig_status_message.emit(f"Focal point set to ({hit_pos[0]:.2f}, {hit_pos[1]:.2f}, {hit_pos[2]:.2f})")
        else:
            self.sig_status_message.emit("Missed Gaussians. Try clicking closer to the object.")

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

    def set_backend(self, _idx: int):
        del _idx
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


def main():
    parser = argparse.ArgumentParser(description="Gaussian Splatting Viewer - PyVista Edition")
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
