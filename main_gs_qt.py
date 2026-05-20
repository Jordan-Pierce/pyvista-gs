"""
main_gs_qt.py
PyQt5 entry point for the native 3D Gaussian Splatting viewer (PyVista/VTK).

Mirrors the structure of `main.py`:
  - Dark palette + matching stylesheet
  - QMainWindow with a docked control panel on the left
  - Status bar showing FPS, gaussian count, and engine string
  - "View" menu with a panel toggle
  - HiDPI font scaling via --hidpi

Usage:
    python main_gs_qt.py [--hidpi]

A file dialog prompts for a .ply file on launch.
"""
from __future__ import annotations

import argparse
import os
import sys

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import (
    QAction, QApplication, QDockWidget, QFileDialog, QLabel,
    QMainWindow,
)

from gs_widget import GaussianSplatWidget
from gs_control_panel import GSControlPanel


# ─────────────────────────────────────────────────────────────────────────── #
#  Dark palette (verbatim from main.py)                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def _build_dark_palette() -> QPalette:
    pal = QPalette()
    base      = QColor("#0f1117")
    alt_base  = QColor("#181b24")
    window    = QColor("#13161f")
    text      = QColor("#d4d8e8")
    bright    = QColor("#ffffff")
    mid       = QColor("#2a2d3a")
    highlight = QColor("#3d7aed")
    disabled  = QColor("#555870")

    pal.setColor(QPalette.Window,          window)
    pal.setColor(QPalette.WindowText,      text)
    pal.setColor(QPalette.Base,            base)
    pal.setColor(QPalette.AlternateBase,   alt_base)
    pal.setColor(QPalette.ToolTipBase,     base)
    pal.setColor(QPalette.ToolTipText,     text)
    pal.setColor(QPalette.Text,            text)
    pal.setColor(QPalette.Button,          mid)
    pal.setColor(QPalette.ButtonText,      text)
    pal.setColor(QPalette.BrightText,      bright)
    pal.setColor(QPalette.Link,            highlight)
    pal.setColor(QPalette.Highlight,       highlight)
    pal.setColor(QPalette.HighlightedText, bright)
    pal.setColor(QPalette.Disabled, QPalette.Text,       disabled)
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, disabled)
    return pal


# ─────────────────────────────────────────────────────────────────────────── #
#  QSS stylesheet (verbatim from main.py)                                     #
# ─────────────────────────────────────────────────────────────────────────── #

STYLESHEET = """
/* ── Root ─────────────────────────────────────────────────── */
QMainWindow, QWidget {
    background-color: #13161f;
    color: #d4d8e8;
    font-family: "JetBrains Mono", "Cascadia Code", "Fira Code", monospace;
    font-size: 12px;
}

/* ── Dock ──────────────────────────────────────────────────── */
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

/* ── GroupBox ───────────────────────────────────────────────── */
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

/* ── Buttons ────────────────────────────────────────────────── */
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

/* ── Sliders ────────────────────────────────────────────────── */
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

/* ── CheckBox ───────────────────────────────────────────────── */
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

/* ── Labels ─────────────────────────────────────────────────── */
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

/* ── ScrollBar ──────────────────────────────────────────────── */
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

/* ── Status bar ─────────────────────────────────────────────── */
QStatusBar {
    background: #0c0e15;
    color: #4a5470;
    font-size: 11px;
    border-top: 1px solid #1e2130;
}
QStatusBar QLabel {
    color: #4a5470;
}

/* ── Menu bar ───────────────────────────────────────────────── */
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

/* ── Separator ──────────────────────────────────────────────── */
QFrame#separator {
    color: #1e2130;
    max-height: 1px;
}

/* ── Control panel scroll area ──────────────────────────────── */
QScrollArea {
    border: none;
    background: transparent;
}
"""


# ─────────────────────────────────────────────────────────────────────────── #
#  Main Window                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

class MainWindow(QMainWindow):
    def __init__(self, hidpi: bool = False, ply_path: str | None = None):
        super().__init__()
        self.setWindowTitle("Gaussian Splatting Viewer — Native (PyVista)")
        self.resize(1440, 860)

        # ── Central PyVistaQt widget ──────────────────────────────────── #
        self._gw = GaussianSplatWidget()
        self.setCentralWidget(self._gw)

        # ── Control panel dock ────────────────────────────────────────── #
        self._panel = GSControlPanel(self._gw)
        dock = QDockWidget("Controls", self)
        dock.setWidget(self._panel)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        # ── Menu ──────────────────────────────────────────────────────── #
        menu = self.menuBar()

        file_menu = menu.addMenu("File")
        open_act = QAction("Open .ply…", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._prompt_for_ply)
        file_menu.addAction(open_act)

        save_act = QAction("Save Viewport Image…", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(lambda: self._gw.save_image(None))
        file_menu.addAction(save_act)

        file_menu.addSeparator()
        quit_act = QAction("Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        view_menu = menu.addMenu("View")
        toggle_panel = QAction("Show Control Panel", self, checkable=True, checked=True)
        toggle_panel.triggered.connect(dock.setVisible)
        view_menu.addAction(toggle_panel)

        # ── Status bar ────────────────────────────────────────────────── #
        self._status_bar = self.statusBar()
        self._status_lbl = QLabel("Ready")
        self._status_bar.addPermanentWidget(self._status_lbl)
        self._gw.sig_status_message.connect(self._status_lbl.setText)
        self._gw.sig_fps_changed.connect(
            lambda fps: self._status_bar.showMessage(
                f"  {fps:.1f} fps  |  "
                f"{self._gw.gaussian_count():,} Gaussians  |  "
                f"PyVista / VTK"
            )
        )

        if hidpi:
            QApplication.instance().setFont(
                QFont(QApplication.font().family(), 14)
            )

        # ── Initial PLY load ──────────────────────────────────────────── #
        # Delay slightly so the GL context is created before the load fires.
        if ply_path:
            QTimer.singleShot(150, lambda: self._gw.load_ply(ply_path))
        else:
            QTimer.singleShot(200, self._prompt_for_ply)

    # ── Helpers ───────────────────────────────────────────────────────── #

    def _prompt_for_ply(self):
        start_dir = os.path.dirname(os.path.abspath(__file__))
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Gaussian Splatting PLY",
            start_dir,
            "PLY files (*.ply)"
        )
        if path:
            self._gw.load_ply(path)


# ─────────────────────────────────────────────────────────────────────────── #
#  Entry point                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(
        description="Native Gaussian Splatting Viewer — PyQt5 + PyVistaQt"
    )
    parser.add_argument("--hidpi", action="store_true",
                        help="Enable HiDPI font scaling")
    parser.add_argument("--ply", default=None,
                        help="Optional .ply file to load on startup "
                             "(skips the file picker)")
    args = parser.parse_args()

    # Add repo root to path so native_actor_gs et al are importable
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    os.chdir(here)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_build_dark_palette())
    app.setStyleSheet(STYLESHEET)

    win = MainWindow(hidpi=args.hidpi, ply_path=args.ply)
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
