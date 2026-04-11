# Gaussian Splatting Viewer — PyQt5 Edition

Rewrite of the original GLFW/ImGui viewer as a self-contained **PyQt5 widget**,
ready to be embedded in any larger Qt application.

![teaser.png](./assets/teaser.png)

## Install

```bash
pip install PyQt5 PyOpenGL PyOpenGL_accelerate numpy imageio plyfile PyGLM
# for CUDA renderer:
# pip install torch  (CUDA build)
```

## Run

```bash
python main.py
python main.py --hidpi    # 1.5× font scale on HiDPI displays
```

## Embedding in another application

`GaussianWidget` is a plain `QOpenGLWidget` subclass — drop it anywhere:

```python
from gaussian_widget import GaussianWidget
from control_panel   import ControlPanel

# In your own QMainWindow / QDialog / QSplitter:
viewer = GaussianWidget()
panel  = ControlPanel(viewer)

# Load a scene programmatically:
viewer.load_ply("/path/to/point_cloud.ply")
```

Signals emitted by `GaussianWidget`:
- `sig_fps_changed(float)`        — current frames-per-second
- `sig_gau_count_changed(int)`    — number of loaded Gaussians
- `sig_status_message(str)`       — human-readable status string

## Key architecture decisions

| Problem | Solution |
|---|---|
| GL context timing | All GL construction lives in `initializeGL()`, never `__init__` |
| GL calls from UI thread | `makeCurrent()` / `doneCurrent()` guards around every out-of-paintGL call |
| Render loop | `QTimer(interval=16)` → `update()` ≈ 60 fps; `reduce_updates` flag passes through to renderer |
| CUDA context | `QSurfaceFormat` set to OpenGL 4.3 Core **before** `QApplication` is constructed |
| Mouse tracking | `camera.first_mouse = True` reset on each `mousePressEvent` to prevent jump |
| Auto-sort | Second `QTimer(interval=80ms)` calls `sort_and_update` when enabled |
