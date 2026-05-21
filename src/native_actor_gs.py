"""
native_actor_gs.py  –  3DGS PLY loader + VTK/PyVista GLSL shader injection.

Public API
----------
  RENDER_MODES                  list[str]  – 8 mode labels (matches original)
  load_3dgs_as_polydata(path)   -> pv.PolyData
  apply_3dgs_shaders(actor, mesh, render_mode=7)
  set_render_mode_on_actor(actor, render_mode)
  sort_splats_by_depth(plotter, mesh)

Render modes
------------
  0  Gaussian Ball   – Gaussian falloff, full-SH colour
  1  Flat Ball       – Hard-disk cutoff, full-SH colour
  2  Billboard       – alias for 0
  3  Depth           – rainbow depth colour
  4  SH:0            – DC only
  5  SH:0~1          – DC + degree 1
  6  SH:0~2          – DC + degrees 1-2
  7  SH:0~3 default  – DC + degrees 1-2 (degree-3 exceeds 16-attr GL limit)

SH loading fix
--------------
The original native_actor_gs incorrectly used f_rest_0,1,2 as the RGB vector
for SH1 m=-1, but those are all R-channel for different basis functions.
Correct layout for a degree-3 PLY (n_per_ch=15 per colour channel):
  R coeff k  ->  f_rest_{k}
  G coeff k  ->  f_rest_{n_per_ch + k}
  B coeff k  ->  f_rest_{2*n_per_ch + k}
"""
from __future__ import annotations

import numpy as np
import pyvista as pv
import vtk
from plyfile import PlyData


RENDER_MODES = [
    "Gaussian Ball", "Flat Ball", "Billboard",
    "Depth", "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)",
]


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

def load_3dgs_as_polydata(ply_path: str) -> pv.PolyData:
    """Load a 3DGS .ply and return a PyVista PolyData with splat attributes."""
    print(f"Loading PLY: {ply_path}")
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]

    # Positions
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)
    mesh = pv.PolyData(xyz)

    # Scales (exp-activate)
    scales = np.exp(
        np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1)
    ).astype(np.float32)
    mesh.point_data["gs_scales"] = scales

    # Quaternions – PLY stores (w,x,y,z) in rot_0..rot_3; normalise
    quats = np.stack(
        [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=-1
    ).astype(np.float64)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    mesh.point_data["gs_quats"] = quats.astype(np.float32)

    # Opacity (sigmoid)
    opacity = (1.0 / (1.0 + np.exp(-vertex["opacity"].astype(np.float64)))).astype(np.float32)
    mesh.point_data["gs_opacity"] = opacity

    # SH degree-0 (DC)
    sh_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1).astype(np.float32)
    mesh.point_data["gs_sh_dc"] = sh_dc

    # Higher-degree SH
    # PLY layout: first n_per_ch f_rest_ values = all non-DC SH for R,
    # next n_per_ch for G, then B.  Within each channel: l=1 m=-1,0,+1;
    # l=2 m=-2,-1,0,+1,+2; l=3 …
    rest_props = sorted(
        [p.name for p in vertex.properties if p.name.startswith("f_rest_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    n_per_ch = len(rest_props) // 3

    def _sh_rgb(k: int) -> np.ndarray:
        return np.stack(
            [vertex[rest_props[k]],
             vertex[rest_props[n_per_ch + k]],
             vertex[rest_props[2 * n_per_ch + k]]],
            axis=-1,
        ).astype(np.float32)

    if n_per_ch >= 3:
        mesh.point_data["gs_sh1_0"] = _sh_rgb(0)   # l=1, m=-1
        mesh.point_data["gs_sh1_1"] = _sh_rgb(1)   # l=1, m=0
        mesh.point_data["gs_sh1_2"] = _sh_rgb(2)   # l=1, m=+1

    if n_per_ch >= 8:
        mesh.point_data["gs_sh2_0"] = _sh_rgb(3)   # l=2, m=-2
        mesh.point_data["gs_sh2_1"] = _sh_rgb(4)   # l=2, m=-1
        mesh.point_data["gs_sh2_2"] = _sh_rgb(5)   # l=2, m=0
        mesh.point_data["gs_sh2_3"] = _sh_rgb(6)   # l=2, m=+1
        mesh.point_data["gs_sh2_4"] = _sh_rgb(7)   # l=2, m=+2

    centroid = mesh.points.mean(axis=0)
    mesh.points -= centroid

    print(f"  {mesh.n_points:,} splats, {n_per_ch} SH coeffs/channel loaded.")
    return mesh


# ---------------------------------------------------------------------------
#  GLSL strings
# ---------------------------------------------------------------------------

_VERT_DEC = """
//VTK::PositionVC::Dec

uniform mat4 MCVCMatrix;

in vec3  gs_scales;
in vec4  gs_quats;
in float gs_opacity;
in vec3  gs_sh_dc;
in vec3  gs_sh1_0;
in vec3  gs_sh1_1;
in vec3  gs_sh1_2;
in vec3  gs_sh2_0;
in vec3  gs_sh2_1;
in vec3  gs_sh2_2;
in vec3  gs_sh2_3;
in vec3  gs_sh2_4;

out vec3  v_conic;
out vec3  v_color;
out float v_opacity;
out float v_pointSize;
out float v_depth;
"""

_VERT_IMPL = """
//VTK::PositionVC::Impl

gl_Position = MCDCMatrix * vec4(vertexMC.xyz, 1.0);

float qw = gs_quats.x, qx = gs_quats.y, qy = gs_quats.z, qz = gs_quats.w;
mat3 R = mat3(
    1.0 - 2.0*(qy*qy + qz*qz),  2.0*(qx*qy + qw*qz),        2.0*(qx*qz - qw*qy),
    2.0*(qx*qy - qw*qz),         1.0 - 2.0*(qx*qx + qz*qz),  2.0*(qy*qz + qw*qx),
    2.0*(qx*qz + qw*qy),         2.0*(qy*qz - qw*qx),         1.0 - 2.0*(qx*qx + qy*qy)
);

mat3 S  = mat3(gs_scales.x, 0.0, 0.0, 0.0, gs_scales.y, 0.0, 0.0, 0.0, gs_scales.z);
mat3 RS = R * S;
mat3 Sigma = RS * transpose(RS);

vec4  pos_cam = MCVCMatrix * vec4(vertexMC.xyz, 1.0);
float depth   = max(-pos_cam.z, 1e-4);
v_depth = depth;

float focal = 800.0;
float k     = (focal * focal) / (depth * depth);
float cxx   = Sigma[0][0] * k + 0.3;
float cyy   = Sigma[1][1] * k + 0.3;
float cxy   = Sigma[0][1] * k;

float det     = cxx * cyy - cxy * cxy;
float inv_det = 1.0 / max(det, 1e-7);
v_conic = vec3(cyy * inv_det, -cxy * inv_det, cxx * inv_det);

float mid    = 0.5 * (cxx + cyy);
float rad    = length(vec2(0.5 * (cxx - cyy), cxy));
float lam1   = mid + rad;
gl_PointSize = clamp(ceil(3.0 * sqrt(lam1)) * 2.0, 2.0, 1024.0);
v_pointSize  = gl_PointSize;

vec3 camPosMC = (inverse(MCVCMatrix) * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
vec3 dir      = normalize(vertexMC.xyz - camPosMC);

float SH_C0 = 0.28209479;
float SH_C1 = 0.48860251;
vec3 color = gs_sh_dc * SH_C0;

if (render_mode != 3 && render_mode != 4) {
    color += -SH_C1 * dir.y * gs_sh1_0;
    color +=  SH_C1 * dir.z * gs_sh1_1;
    color += -SH_C1 * dir.x * gs_sh1_2;
}

if (render_mode <= 2 || render_mode >= 6) {
    float c20 =  1.0925484, c21 = -1.0925484, c22 = 0.3153916;
    float c23 = -1.0925484, c24 =  0.5462742;
    float xx = dir.x*dir.x, yy = dir.y*dir.y, zz = dir.z*dir.z;
    float xy = dir.x*dir.y, xz = dir.x*dir.z, yz = dir.y*dir.z;
    color += c20 * xy                  * gs_sh2_0;
    color += c21 * yz                  * gs_sh2_1;
    color += c22 * (2.0*zz - xx - yy) * gs_sh2_2;
    color += c23 * xz                  * gs_sh2_3;
    color += c24 * (xx - yy)           * gs_sh2_4;
}

if (render_mode == 3) {
    float t  = clamp(depth / 10.0, 0.0, 1.0);
    color = vec3(t, 1.0 - abs(2.0*t - 1.0), 1.0 - t);
}

v_color   = clamp(color + 0.5, 0.0, 1.0);
v_opacity = gs_opacity;
"""

_FRAG_DEC = """
//VTK::Color::Dec
in vec3  v_conic;
in vec3  v_color;
in float v_opacity;
in float v_pointSize;
in float v_depth;
"""

_FRAG_IMPL = """
//VTK::Color::Impl

vec2  d = (gl_PointCoord - 0.5) * v_pointSize;
float final_alpha;

if (render_mode == 1) {
    float r2     = dot(d, d);
    float radius = v_pointSize * 0.5;
    if (r2 > radius * radius) discard;
    final_alpha = v_opacity;
} else {
    float power = -0.5 * (v_conic.x * d.x*d.x + v_conic.z * d.y*d.y)
                  - v_conic.y * d.x * d.y;
    if (power > 0.0) discard;
    final_alpha = v_opacity * exp(power);
}

if (final_alpha < (1.0 / 255.0)) discard;

ambientColor = v_color;
diffuseColor = v_color;
opacity      = final_alpha;
"""


# ---------------------------------------------------------------------------
#  Shader injection
# ---------------------------------------------------------------------------

def apply_3dgs_shaders(actor, mesh: pv.PolyData, render_mode: int = 7) -> None:
    """Inject 3DGS GLSL and wire vertex-attribute mappings.  Call once after
    adding the mesh to the plotter; use set_render_mode_on_actor to update."""
    has_sh1 = "gs_sh1_0" in mesh.point_data
    has_sh2 = "gs_sh2_0" in mesh.point_data

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)

    def _map(name: str):
        mapper.MapDataArrayToVertexAttribute(
            name, name, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)

    _map("gs_scales"); _map("gs_quats"); _map("gs_opacity"); _map("gs_sh_dc")
    if has_sh1:
        _map("gs_sh1_0"); _map("gs_sh1_1"); _map("gs_sh1_2")
    if has_sh2:
        _map("gs_sh2_0"); _map("gs_sh2_1"); _map("gs_sh2_2")
        _map("gs_sh2_3"); _map("gs_sh2_4")

    if hasattr(mapper, "SetUseProgramPointSize"):
        mapper.SetUseProgramPointSize(True)

    actor.SetMapper(mapper)

    sp = actor.GetShaderProperty()
    sp.AddVertexShaderReplacement(  "//VTK::PositionVC::Dec",  True, _VERT_DEC,  False)
    sp.AddVertexShaderReplacement(  "//VTK::PositionVC::Impl", True, _VERT_IMPL, False)
    sp.AddFragmentShaderReplacement("//VTK::Color::Dec",        True, _FRAG_DEC,  False)
    sp.AddFragmentShaderReplacement("//VTK::Color::Impl",       True, _FRAG_IMPL, False)

    set_render_mode_on_actor(actor, render_mode)


def set_render_mode_on_actor(actor, render_mode: int) -> None:
    """Update the render_mode uniform on an already-shaded actor."""
    sp = actor.GetShaderProperty()
    sp.GetVertexCustomUniforms().SetUniformi("render_mode", int(render_mode))
    sp.GetFragmentCustomUniforms().SetUniformi("render_mode", int(render_mode))


# ---------------------------------------------------------------------------
#  Depth sorting
# ---------------------------------------------------------------------------

def sort_splats_by_depth(plotter, mesh: pv.PolyData) -> None:
    """Re-order vertices back-to-front along the current view vector."""
    cam      = plotter.camera
    view_dir = np.array(cam.focal_point, dtype=np.float64) - np.array(cam.position, dtype=np.float64)
    depths         = mesh.points.astype(np.float64) @ view_dir
    sorted_indices = np.argsort(depths)[::-1].astype(np.int64)
    n = len(sorted_indices)
    verts = np.empty((n, 2), dtype=np.int64)
    verts[:, 0] = 1
    verts[:, 1] = sorted_indices
    mesh.verts = verts.ravel()


# ---------------------------------------------------------------------------
#  Standalone entry point
# ---------------------------------------------------------------------------

def main():
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "splat.ply"
    mesh = load_3dgs_as_polydata(path)

    pl = pv.Plotter()
    pl.set_background("#13161f")

    actor = pl.add_mesh(mesh, style="points", render_points_as_spheres=False,
                        lighting=False, show_scalar_bar=False, rgb=True)
    actor.GetProperty().SetOpacity(0.99)
    apply_3dgs_shaders(actor, mesh, render_mode=7)

    pl.add_mesh(pv.Box(bounds=mesh.bounds), color="cyan", style="wireframe", line_width=2)
    pl.iren.add_observer("EndInteractionEvent",
                         lambda *_: (sort_splats_by_depth(pl, mesh), pl.render()))
    sort_splats_by_depth(pl, mesh)
    pl.show()


if __name__ == "__main__":
    main()
