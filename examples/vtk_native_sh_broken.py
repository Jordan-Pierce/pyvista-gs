"""
Minimal reproduction of broken SH evaluation in VTK's geometry shader.

This script renders Gaussian splats as a **single-pass VTK actor** using full
shader replacement on vtkOpenGLPolyDataMapper.  The geometry shader performs
covariance projection (quad sizing) AND Spherical Harmonics color evaluation.

The SH math is identical to the working vertex shader in gau_vert.glsl, but
when executed in the geometry shader the colors come out blown-out / incorrect.
We have not been able to determine whether this is a precision issue, a
matrix-convention mismatch in the geometry stage, or something else entirely.

If you can fix this so the output matches pyvista-gs's ModernGL renderer, the
hybrid architecture can be replaced with a single-pass VTK actor.  See the
README's "Contributing" section.

Usage:
    python examples/vtk_native_sh_broken.py path/to/point_cloud.ply

Requirements:
    pip install pyvista pyvistaqt numpy PyOpenGL PyGLM
"""
from __future__ import annotations

import sys
import os

import numpy as np
import vtk
import pyvista as pv
from pyvistaqt import QtInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.util.misc import calldata_type
from vtkmodules.vtkCommonCore import VTK_OBJECT
from OpenGL import GL as gl
import glm

# ---------------------------------------------------------------------------
# Add src/ to path so we can reuse the data loader and sort function
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pyvista_gs.data import GaussianData, load_ply  # noqa: E402
from pyvista_gs.renderer import _sort_gaussian       # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════════
# Inline VTK shaders — the geometry shader contains the broken SH evaluation
# ═══════════════════════════════════════════════════════════════════════════════

VERT_SHADER = """\
#version 430 core

layout(location = 0) in vec4 vertexMC;

uniform mat4 view_matrix;
uniform mat4 projection_matrix;

void main()
{
    gl_Position = projection_matrix * view_matrix * vertexMC;
}
"""

# This geometry shader is the problematic one.  The SH evaluation (search for
# "Spherical Harmonics" below) produces blown-out colors.  Everything else
# (covariance, quad emission, opacity) works correctly.
GEOM_SHADER = """\
#version 430 core

#define SH_C0  0.28209479177387814f
#define SH_C1  0.4886025119029199f

#define SH_C2_0  1.0925484305920792f
#define SH_C2_1 -1.0925484305920792f
#define SH_C2_2  0.31539156525252005f
#define SH_C2_3 -1.0925484305920792f
#define SH_C2_4  0.5462742152960396f

#define SH_C3_0 -0.5900435899266435f
#define SH_C3_1  2.890611442640554f
#define SH_C3_2 -0.4570457994644658f
#define SH_C3_3  0.3731763325901154f
#define SH_C3_4 -0.4570457994644658f
#define SH_C3_5  1.445305721320277f
#define SH_C3_6 -0.5900435899266435f

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

layout(std430, binding = 0) buffer gaussian_data  { float g_data[]; };
layout(std430, binding = 1) buffer gaussian_order { int   gi[];     };

uniform mat4  view_matrix;
uniform mat4  projection_matrix;
uniform vec3  hfovxy_focal;
uniform vec3  cam_pos;
uniform int   sh_dim;
uniform float scale_modifier;
uniform int   render_mod;

out vec3  frag_color;
out float frag_alpha;
out vec3  frag_conic;
out vec2  frag_coordxy;

#define POS_IDX     0
#define ROT_IDX     3
#define SCALE_IDX   7
#define OPACITY_IDX 10
#define SH_IDX      11

vec3 get_vec3(int offset)
{
    return vec3(g_data[offset], g_data[offset + 1], g_data[offset + 2]);
}
vec4 get_vec4(int offset)
{
    return vec4(g_data[offset], g_data[offset + 1], g_data[offset + 2], g_data[offset + 3]);
}

mat3 computeCov3D(vec3 scale, vec4 q)
{
    mat3 S = mat3(0.f);
    S[0][0] = scale.x;
    S[1][1] = scale.y;
    S[2][2] = scale.z;

    float r = q.x, x = q.y, y = q.z, z = q.w;
    mat3 R = mat3(
        1.f - 2.f*(y*y + z*z),   2.f*(x*y - r*z),         2.f*(x*z + r*y),
            2.f*(x*y + r*z), 1.f - 2.f*(x*x + z*z),   2.f*(y*z - r*x),
            2.f*(x*z - r*y),       2.f*(y*z + r*x), 1.f - 2.f*(x*x + y*y)
    );
    mat3 M = S * R;
    return transpose(M) * M;
}

vec3 computeCov2D(vec4 mean_view,
                  float focal_x, float focal_y,
                  float tan_fovx, float tan_fovy,
                  mat3 cov3D, mat4 viewmatrix)
{
    vec4 t = mean_view;
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    t.x = clamp(t.x / t.z, -limx, limx) * t.z;
    t.y = clamp(t.y / t.z, -limy, limy) * t.z;

    mat3 J = mat3(
        focal_x / t.z, 0.f, -(focal_x * t.x) / (t.z * t.z),
        0.f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0.f, 0.f, 0.f
    );
    mat3 W   = transpose(mat3(viewmatrix));
    mat3 T   = W * J;
    mat3 cov = transpose(T) * transpose(cov3D) * T;
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;
    return vec3(cov[0][0], cov[0][1], cov[1][1]);
}

void main()
{
    int boxid     = gi[gl_PrimitiveIDIn];
    int total_dim = 3 + 4 + 3 + 1 + sh_dim;
    int start     = boxid * total_dim;

    vec4 g_pos        = vec4(get_vec3(start + POS_IDX), 1.f);
    vec4 g_pos_view   = view_matrix * g_pos;
    vec4 g_pos_clip   = projection_matrix * g_pos_view;
    g_pos_clip.xyz   /= g_pos_clip.w;
    g_pos_clip.w      = 1.f;

    if (any(greaterThan(abs(g_pos_clip.xyz), vec3(1.3f))))
        return;

    vec4  g_rot     = get_vec4(start + ROT_IDX);
    vec3  g_scale   = get_vec3(start + SCALE_IDX);
    float g_opacity = g_data[start + OPACITY_IDX];

    mat3 cov3d  = computeCov3D(g_scale * scale_modifier, g_rot);
    vec2 wh     = 2.f * hfovxy_focal.xy * hfovxy_focal.z;
    vec3 cov2d  = computeCov2D(g_pos_view,
                               hfovxy_focal.z, hfovxy_focal.z,
                               hfovxy_focal.x, hfovxy_focal.y,
                               cov3d, view_matrix);

    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f) return;
    float det_inv = 1.f / det;
    vec3 conic = vec3(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);

    vec2 quadwh_scr = vec2(3.f * sqrt(cov2d.x), 3.f * sqrt(cov2d.z));
    vec2 quadwh_ndc = quadwh_scr / wh * 2.f;

    // ── Spherical Harmonics color evaluation ──────────────────────────
    // THIS IS THE BROKEN PART.  The math is identical to gau_vert.glsl
    // but produces blown-out / incorrect colors in the geometry shader.
    vec3 color;
    if (render_mod == -1) {
        float depth = -g_pos_view.z;
        depth = (depth < 0.05f) ? 1.f : depth;
        depth = 1.f / depth;
        color = vec3(depth, depth, depth);
    } else {
        int  sh_start = start + SH_IDX;
        vec3 dir      = normalize(g_pos.xyz - cam_pos);
        color         = SH_C0 * get_vec3(sh_start);

        if (sh_dim > 3 && render_mod >= 1) {
            float x = dir.x, y = dir.y, z = dir.z;
            color += -SH_C1 * y * get_vec3(sh_start + 3)
                   +  SH_C1 * z * get_vec3(sh_start + 6)
                   -  SH_C1 * x * get_vec3(sh_start + 9);

            if (sh_dim > 12 && render_mod >= 2) {
                float xx = x*x, yy = y*y, zz = z*z;
                float xy = x*y, yz = y*z, xz = x*z;
                color +=
                    SH_C2_0 * xy          * get_vec3(sh_start + 12) +
                    SH_C2_1 * yz          * get_vec3(sh_start + 15) +
                    SH_C2_2 * (2*zz-xx-yy)* get_vec3(sh_start + 18) +
                    SH_C2_3 * xz          * get_vec3(sh_start + 21) +
                    SH_C2_4 * (xx-yy)     * get_vec3(sh_start + 24);

                if (sh_dim > 27 && render_mod >= 3) {
                    color +=
                        SH_C3_0 * y * (3*xx - yy)      * get_vec3(sh_start + 27) +
                        SH_C3_1 * xy * z                * get_vec3(sh_start + 30) +
                        SH_C3_2 * y * (4*zz - xx - yy) * get_vec3(sh_start + 33) +
                        SH_C3_3 * z * (2*zz - 3*xx - 3*yy) * get_vec3(sh_start + 36) +
                        SH_C3_4 * x * (4*zz - xx - yy) * get_vec3(sh_start + 39) +
                        SH_C3_5 * z * (xx - yy)         * get_vec3(sh_start + 42) +
                        SH_C3_6 * x * (xx - 3*yy)       * get_vec3(sh_start + 45);
                }
            }
        }
        color += 0.5f;
    }

    // ── Emit billboard quad ───────────────────────────────────────────
    vec2 corners_ndc[4] = vec2[4](
        vec2(-1.f, -1.f), vec2( 1.f, -1.f),
        vec2(-1.f,  1.f), vec2( 1.f,  1.f)
    );

    for (int i = 0; i < 4; i++) {
        vec4 pos  = g_pos_clip;
        pos.xy   += corners_ndc[i] * quadwh_ndc;
        gl_Position  = pos;
        frag_color   = color;
        frag_alpha   = g_opacity;
        frag_conic   = conic;
        frag_coordxy = corners_ndc[i] * quadwh_scr;
        EmitVertex();
    }
    EndPrimitive();
}
"""

FRAG_SHADER = """\
#version 430 core

in vec3  frag_color;
in float frag_alpha;
in vec3  frag_conic;
in vec2  frag_coordxy;

uniform int render_mod;

out vec4 FragColor;

void main()
{
    if (render_mod == -2) {
        FragColor = vec4(frag_color, 1.f);
        return;
    }

    float power = -0.5f * (frag_conic.x * frag_coordxy.x * frag_coordxy.x
                         + frag_conic.z * frag_coordxy.y * frag_coordxy.y)
                - frag_conic.y * frag_coordxy.x * frag_coordxy.y;

    if (power > 0.f)
        discard;

    float opacity = min(0.99f, frag_alpha * exp(power));
    if (opacity < 1.f / 255.f)
        discard;

    FragColor = vec4(frag_color, opacity);

    if (render_mod == -3) {
        FragColor.a = (FragColor.a > 0.22f) ? 1.f : 0.f;
    } else if (render_mod == -4) {
        FragColor.a   = (FragColor.a > 0.22f) ? 1.f : 0.f;
        FragColor.rgb = FragColor.rgb * exp(power);
    }
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# VTK-native Gaussian renderer (single-pass, broken SH)
# ═══════════════════════════════════════════════════════════════════════════════

class VTKNativeGaussianRenderer:
    """Single-pass VTK actor with full shader replacement.

    This is the approach we *want* to work — one vtkActor that both renders
    correct splats and participates in VTK's scene graph.  Currently the SH
    colors are wrong.
    """

    def __init__(self, vtk_renderer: vtk.vtkRenderer):
        self._vtk_renderer = vtk_renderer
        self._gaussians: GaussianData | None = None

        self._scale_modifier = 1.0
        self._render_mod = 3
        self._sh_dim = 3

        self._data_ssbo: int | None = None
        self._index_ssbo: int | None = None
        self._ssbo_ready = False

        self._pending_data: np.ndarray | None = None
        self._pending_index: np.ndarray | None = None
        self._data_dirty = False
        self._index_dirty = False
        self._sort_needed = False

        self._poly = vtk.vtkPolyData()
        self._pts = vtk.vtkPoints()
        self._pts.SetDataTypeToDouble()
        self._poly.SetPoints(self._pts)

        self._mapper = vtk.vtkOpenGLPolyDataMapper()
        self._mapper.SetInputData(self._poly)

        @calldata_type(VTK_OBJECT)
        def _on_shader(caller, event, calldata):
            self._on_update_shader(caller, event, calldata)

        self._mapper.AddObserver("UpdateShaderEvent", _on_shader)

        self._actor = vtk.vtkActor()
        self._actor.SetMapper(self._mapper)
        self._actor.ForceTranslucentOn()
        self._actor.GetProperty().SetOpacity(1.0)
        self._actor.GetProperty().SetPointSize(1)

        sp = self._actor.GetShaderProperty()
        sp.SetVertexShaderCode(VERT_SHADER)
        sp.SetGeometryShaderCode(GEOM_SHADER)
        sp.SetFragmentShaderCode(FRAG_SHADER)

        vtk_renderer.AddActor(self._actor)
        vtk_renderer.GetActiveCamera().AddObserver(
            vtk.vtkCommand.ModifiedEvent,
            lambda *_: setattr(self, '_sort_needed', True),
        )

    @property
    def actor(self) -> vtk.vtkActor:
        return self._actor

    def load(self, gaus: GaussianData):
        self._gaussians = gaus
        self._sh_dim = gaus.sh_dim
        n = len(gaus)

        self._pts.SetData(numpy_to_vtk(gaus.xyz.astype(np.float64), deep=True))
        cells = np.empty(2 * n, dtype=np.int64)
        cells[0::2] = 1
        cells[1::2] = np.arange(n)
        verts = vtk.vtkCellArray()
        verts.SetCells(n, numpy_to_vtk(cells, deep=True, array_type=vtk.VTK_ID_TYPE))
        self._poly.SetVerts(verts)
        self._poly.Modified()

        self._pending_data = gaus.flat().astype(np.float32)
        self._pending_index = np.arange(n, dtype=np.int32)
        self._data_dirty = True
        self._index_dirty = True
        self._sort_needed = True

    def _on_update_shader(self, _caller, _event, calldata):
        program = calldata
        if program is None:
            return

        while gl.glGetError() != gl.GL_NO_ERROR:
            pass

        if not self._ssbo_ready:
            ids = gl.glGenBuffers(2)
            self._data_ssbo = int(ids[0])
            self._index_ssbo = int(ids[1])
            self._ssbo_ready = True

        if self._sort_needed and self._gaussians is not None:
            vtk_cam = self._vtk_renderer.GetActiveCamera()
            pos = np.array(vtk_cam.GetPosition(), dtype=np.float32)
            focal = np.array(vtk_cam.GetFocalPoint(), dtype=np.float32)
            up = np.array(vtk_cam.GetViewUp(), dtype=np.float32)
            view_mat = np.array(glm.lookAt(
                glm.vec3(*pos.tolist()),
                glm.vec3(*focal.tolist()),
                glm.vec3(*up.tolist()),
            ), dtype=np.float32)
            idx = _sort_gaussian(self._gaussians, view_mat)
            self._pending_index = idx.flatten().astype(np.int32)
            self._index_dirty = True
            self._sort_needed = False

        if self._data_dirty and self._pending_data is not None:
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self._data_ssbo)
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,
                            self._pending_data.nbytes,
                            self._pending_data, gl.GL_DYNAMIC_DRAW)
            self._data_dirty = False

        if self._index_dirty and self._pending_index is not None:
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self._index_ssbo)
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,
                            self._pending_index.nbytes,
                            self._pending_index, gl.GL_DYNAMIC_DRAW)
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
            self._index_dirty = False

        if self._ssbo_ready:
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, self._data_ssbo)
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1, self._index_ssbo)

        vtk_cam = self._vtk_renderer.GetActiveCamera()
        size = self._vtk_renderer.GetSize()
        w, h = max(size[0], 1), max(size[1], 1)

        pos = np.array(vtk_cam.GetPosition(), dtype=np.float32)
        focal_pt = np.array(vtk_cam.GetFocalPoint(), dtype=np.float32)
        up = np.array(vtk_cam.GetViewUp(), dtype=np.float32)
        fovy_deg = vtk_cam.GetViewAngle()

        fovy_rad = np.radians(fovy_deg)
        aspect = w / h
        htany = np.tan(fovy_rad / 2.0)
        htanx = htany * aspect
        focal_len = h / (2.0 * htany)

        view_mat = np.array(glm.lookAt(
            glm.vec3(*pos.tolist()),
            glm.vec3(*focal_pt.tolist()),
            glm.vec3(*up.tolist()),
        ), dtype=np.float32)

        proj_mat = np.array(glm.perspective(
            fovy_rad, float(aspect), 0.01, 100.0,
        ), dtype=np.float32)

        pid = program.GetHandle()
        self._set_mat4(pid, "view_matrix", view_mat)
        self._set_mat4(pid, "projection_matrix", proj_mat)
        self._set_v3(pid, "cam_pos", pos)
        self._set_v3(pid, "hfovxy_focal",
                     np.array([htanx, htany, focal_len], np.float32))
        self._set_1f(pid, "scale_modifier", self._scale_modifier)
        self._set_1i(pid, "sh_dim", self._sh_dim)
        self._set_1i(pid, "render_mod", self._render_mod)

    @staticmethod
    def _set_mat4(pid, name, mat):
        loc = gl.glGetUniformLocation(pid, name)
        if loc >= 0:
            gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE,
                                  mat.T.flatten().astype(np.float32))

    @staticmethod
    def _set_v3(pid, name, v):
        loc = gl.glGetUniformLocation(pid, name)
        if loc >= 0:
            gl.glUniform3f(loc, float(v[0]), float(v[1]), float(v[2]))

    @staticmethod
    def _set_1f(pid, name, v):
        loc = gl.glGetUniformLocation(pid, name)
        if loc >= 0:
            gl.glUniform1f(loc, float(v))

    @staticmethod
    def _set_1i(pid, name, v):
        loc = gl.glGetUniformLocation(pid, name)
        if loc >= 0:
            gl.glUniform1i(loc, int(v))


# ═══════════════════════════════════════════════════════════════════════════════
# Minimal viewer
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/vtk_native_sh_broken.py <path/to/point_cloud.ply>")
        sys.exit(1)

    ply_path = sys.argv[1]
    if not os.path.isfile(ply_path):
        print(f"File not found: {ply_path}")
        sys.exit(1)

    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)

    plotter = pv.Plotter()
    plotter.set_background("black")

    gaussians = load_ply(ply_path)
    gaussians.xyz -= gaussians.xyz.mean(axis=0)

    renderer = VTKNativeGaussianRenderer(plotter.renderer)
    renderer.load(gaussians)

    plotter.reset_camera()
    plotter.show(title="VTK-native Gaussian Splatting (broken SH)")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
