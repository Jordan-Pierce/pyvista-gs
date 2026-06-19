# pyvista-gs

PyVista-backed 3D Gaussian Splatting viewer packaged as an installable Python
module. `GaussianActor` is the main reusable integration point: it can be
embedded in other PyVista or Qt applications and behaves like a first-class VTK
actor from the outside, with full support for picking, bounding-box queries, and
scene composition alongside meshes and point clouds.

Originally forked from
[limacv/GaussianSplattingViewer](https://github.com/limacv/GaussianSplattingViewer).

![teaser.png](./assets/teaser.png)

## Architecture

pyvista-gs uses a **hybrid VTK + ModernGL** rendering architecture. A single
`GaussianActor` owns two subsystems that work together each frame:

```
GaussianActor
├── VTK Proxy Actor (scene participation)
│   ├── Real vtkActor added to the VTK renderer
│   ├── Geometry shader emits correctly-sized billboard quads
│   ├── Fragment shader discards all pixels (invisible)
│   └── Provides bounds, picking, and depth ordering to VTK
│
├── ModernGL Renderer (visual output)
│   ├── Draws splats with correct Spherical Harmonics evaluation
│   ├── Renders via EndEvent callback after VTK's own pass
│   └── Supports view-dependent color (SH bands 0–3)
│
└── Shared sort: depth-sorted once per frame, indices fed to both
```

**Why two renderers?** VTK's shader replacement pipeline (geometry shader) can
emit correctly-sized quads for scene participation, but its SH color evaluation
produces blown-out results. ModernGL's vertex shader handles SH correctly, but
VTK can't see actors drawn outside its pipeline. The hybrid gives us both:
correct visuals *and* full VTK scene citizenship.

### VTK Proxy

The proxy is a `vtkActor` backed by a `vtkPolyData` with one vertex per
Gaussian centre. A geometry shader reads per-Gaussian attributes (position,
rotation, scale, opacity) from SSBOs and emits billboard quads sized by the
3D→2D covariance projection — the same math used for visual rendering. The
fragment shader unconditionally `discard`s every pixel, so the proxy is
invisible during normal rendering. During VTK's hardware pick pass, VTK
replaces the fragment shader with its own, making the quads hittable.

### ModernGL Renderer

After VTK finishes its render pass, an `EndEvent` callback hands control to the
ModernGL renderer. It draws instanced quads with full Spherical Harmonics
evaluation (bands 0–3), producing view-dependent color. Blending is enabled and
depth writes are disabled so splats composite correctly over VTK's framebuffer.

## Install

```bash
pip install -e .
# if/when published to PyPI:
# pip install pyvista-gs
```

## Run

```bash
pyvista-gs
python -m pyvista_gs
python -m pyvista_gs --hidpi    # 1.5x font scale on HiDPI displays
```

## Usage

Use `GaussianActor` to add 3D Gaussian splats to an existing PyVista
application:

```python
from pyvista_gs import GaussianActor, load_ply

gaussians = load_ply("/path/to/point_cloud.ply")
actor = GaussianActor(gaussians)
actor.bind_to_plotter(plotter)
```

### Features

- **Crop preview** — `set_crop_bounds(...)` applies a shader-side crop without
  mutating the data. Call `apply_crop_box()` to permanently remove splats
  outside the box.
- **Transform** — `transform(matrix)` applies a 4×4 homogeneous transform to
  positions, rotations, and scales.
- **Floater removal** — `remove_floaters(min_opacity, max_scale)` culls noisy
  splats that are too transparent or too large.
- **Tinting** — `tint_gaussians(indices, color_rgb)` modifies the DC spherical
  harmonic coefficients to recolor selected splats.
- **Picking** — `pick_gaussian(ray_origin, ray_dir, ...)` performs CPU
  ray-casting against Gaussian centres.

### Embedding

If you want the full standalone window, import `MainWindow`. If you want the
reusable Qt sidebar in your own app, import `ControlPanel`.

### GPU sorting

Optional CUDA sorting backends (`torch` or `cupy`) are detected at runtime and
used automatically when installed. Falls back to NumPy CPU sorting otherwise.

## Contributing: help us remove the hybrid workaround

The hybrid architecture exists because we haven't been able to get Spherical
Harmonics evaluation working correctly inside VTK's geometry shader. The SH
math itself is straightforward, but running it in the geometry shader (which
processes one primitive at a time with limited I/O) produces blown-out,
incorrect colors compared to the identical math in a standard vertex shader.

If the SH evaluation worked correctly in VTK's shader replacement pipeline, we
could drop the ModernGL renderer entirely and have Gaussian splats as a true
single-pass VTK actor — simpler code, one fewer dependency, and better
integration with VTK's depth buffer and compositing.

**PRs are welcome** that fix the SH color evaluation in the VTK geometry shader
so it matches the output of
[gau_vert.glsl](src/pyvista_gs/shaders/gau_vert.glsl).

A self-contained reproduction script is included at
[examples/vtk_native_sh_broken.py](examples/vtk_native_sh_broken.py). It
renders splats as a single-pass VTK actor with the broken SH evaluation — run
it alongside the main viewer on the same PLY file to compare:

```bash
# Broken single-pass VTK rendering (blown-out colors)
python examples/vtk_native_sh_broken.py path/to/splat.ply

# Working hybrid rendering (correct colors)
python -m pyvista_gs path/to/splat.ply
```
