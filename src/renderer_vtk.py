"""
renderer_vtk.py
VTK-based 3D Gaussian Splatting renderer.

Creates a first-class vtkActor so 3DGS participates fully in PyVista's scene:
  - Focal-point click lands on a Gaussian point  →  correct scene pivot
  - Depth-sorted correctly against opaque meshes and frustum actors
  - Toggle visibility with actor.SetVisibility() like any other actor
  - Camera-ray / pick operations work natively

Architecture
------------
  One vtkPolyData point per Gaussian (centres as xyz) gives VTK the correct
  bounding box and pick geometry.  Two raw OpenGL SSBOs hold the flat Gaussian
  parameter array and the sorted index array.  A geometry shader (full shader
  replacement on vtkOpenGLPolyDataMapper) expands each GL_POINTS primitive into
  a screen-space billboard quad, performing all covariance / SH evaluation.
  Per-frame uniforms and SSBO binds are injected via the UpdateShaderEvent
  callback so they are always in sync with VTK's active camera.

SSBO uploads happen lazily inside the callback where the GL context is
guaranteed to be current — no makeCurrent() / doneCurrent() gymnastics needed.
"""
from __future__ import annotations

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
from OpenGL import GL as gl
import glm

from . import util
from . import util_gau
from .renderer_ogl import GaussianRenderBase, _sort_gaussian
from vtkmodules.util.misc import calldata_type
from vtkmodules.vtkCommonCore import VTK_OBJECT

# ── Shader sources ─────────────────────────────────────────────────────────── #


def _read(name: str) -> str:
    path = util.resource_path("shaders", name)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


# ══════════════════════════════════════════════════════════════════════════════
class VTKGaussianRenderer(GaussianRenderBase):
    """
    GaussianRenderBase backed by a vtkOpenGLPolyDataMapper.

    Parameters
    ----------
    vtk_renderer : vtkRenderer
        Renderer from your pyvistaqt.QtInteractor / BackgroundPlotter.
        The Gaussian actor is added automatically.
    auto_sort : bool
        If True, re-sorts Gaussians every time VTK's camera fires
        ModifiedEvent.  Adds ~1–5 ms per frame for large splats; disable
        and call sort_and_update() manually if performance matters.
    """

    def __init__(self, vtk_renderer: vtk.vtkRenderer, auto_sort: bool = True):
        super().__init__()

        self._vtk_renderer   = vtk_renderer
        self._gaussians: util_gau.GaussianData | None = None
        self._auto_sort      = auto_sort

        # ── Gaussian render state ──────────────────────────────────────── #
        self._scale_modifier = 1.0
        self._render_mod     = 3      # default: full SH (SH:0~3)
        self._sh_dim         = 3

        # ── OpenGL SSBO handles (created lazily inside UpdateShaderEvent) ─ #
        self._data_ssbo:  int | None = None
        self._index_ssbo: int | None = None
        self._ssbo_ready             = False

        # Dirty flags — actual uploads happen inside UpdateShaderEvent
        self._pending_data:  np.ndarray | None = None
        self._pending_index: np.ndarray | None = None
        self._data_dirty   = False
        self._index_dirty  = False
        self._sort_needed  = False

        # ── vtkPolyData — one point per Gaussian ──────────────────────── #
        self._poly = vtk.vtkPolyData()
        self._pts  = vtk.vtkPoints()
        self._pts.SetDataTypeToDouble()
        self._poly.SetPoints(self._pts)

        # ── Mapper with full shader replacement ───────────────────────── #
        self._mapper = vtk.vtkOpenGLPolyDataMapper()
        self._mapper.SetInputData(self._poly)

        # Per-frame callback: uniform sync + SSBO bind/upload
        @calldata_type(VTK_OBJECT)
        def _update_shader_callback(caller, event, calldata):
            self._on_update_shader(caller, event, calldata)

        self._update_shader_callback = _update_shader_callback
        self._mapper.AddObserver("UpdateShaderEvent", self._update_shader_callback)

        # ── Actor ─────────────────────────────────────────────────────── #
        self._actor = vtk.vtkActor()
        self._actor.SetMapper(self._mapper)
        # ForceTranslucentOn ensures VTK uses its alpha-blending render pass:
        #   depth test ON (read-only), depth write OFF, src_alpha blending.
        # This is exactly correct for back-to-front Gaussian compositing.
        self._actor.ForceTranslucentOn()
        self._actor.GetProperty().SetOpacity(1.0)
        self._actor.GetProperty().SetPointSize(1)
        shader_property = self._actor.GetShaderProperty()
        shader_property.SetVertexShaderCode(_read("gau_vert_vtk.glsl"))
        shader_property.SetGeometryShaderCode(_read("gau_geom_vtk.glsl"))
        shader_property.SetFragmentShaderCode(_read("gau_frag_vtk.glsl"))

        vtk_renderer.AddActor(self._actor)

        # Camera observer for auto-sort
        vtk_renderer.GetActiveCamera().AddObserver(
            vtk.vtkCommand.ModifiedEvent, self._on_camera_modified
        )

    # ── Public property ────────────────────────────────────────────────────── #

    @property
    def actor(self) -> vtk.vtkActor:
        """The VTK actor; use actor.SetVisibility(bool) to show/hide."""
        return self._actor

    # ── GaussianRenderBase interface ───────────────────────────────────────── #

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        self._gaussians = gaus
        self._sh_dim    = gaus.sh_dim

        # ── Update vtkPolyData so VTK has correct bounds & pick targets ── #
        n = len(gaus)
        pts_vtk = numpy_to_vtk(gaus.xyz.astype(np.float64), deep=True)
        self._pts.SetData(pts_vtk)

        # Build vertex cells [1, 0, 1, 1, 1, 2, ...] (legacy cell format)
        cells_np = np.empty(2 * n, dtype=np.int64)
        cells_np[0::2] = 1                   # each cell is 1 point
        cells_np[1::2] = np.arange(n)        # point index
        cell_vtk = numpy_to_vtk(cells_np, deep=True, array_type=vtk.VTK_ID_TYPE)
        verts = vtk.vtkCellArray()
        verts.SetCells(n, cell_vtk)
        self._poly.SetVerts(verts)
        self._poly.Modified()

        # ── Queue SSBO uploads (actual GL call deferred to callback) ───── #
        self._pending_data  = gaus.flat().astype(np.float32)
        self._pending_index = np.arange(n, dtype=np.int32)
        self._data_dirty    = True
        self._index_dirty   = True
        self._sort_needed   = True   # sort immediately on next frame

    def sort_and_update(self, camera: util.Camera | None = None):
        """
        Re-sort Gaussians back-to-front.

        If *camera* is a util.Camera the sort happens synchronously now.
        If *camera* is None the sort is deferred to the next render frame
        using VTK's active camera (safe to call from any thread).
        """
        if self._gaussians is None:
            return
        if camera is not None:
            idx = _sort_gaussian(self._gaussians, camera.get_view_matrix())
            self._pending_index = idx.flatten().astype(np.int32)
            self._index_dirty   = True
        else:
            self._sort_needed = True

    def set_scale_modifier(self, modifier: float):
        self._scale_modifier = float(modifier)

    def set_render_mod(self, mod: int):
        self._render_mod = int(mod)

    def set_render_reso(self, w: int, h: int):
        pass  # VTK owns the viewport / render window

    def update_camera_pose(self, camera: util.Camera):
        pass  # handled inside _on_update_shader

    def update_camera_intrin(self, camera: util.Camera):
        pass  # handled inside _on_update_shader

    def draw(self):
        pass  # VTK drives the render loop

    def update_vsync(self):
        pass

    # ── Internal callbacks ─────────────────────────────────────────────────── #

    def _on_camera_modified(self, _caller, _event):
        if self._auto_sort:
            self._sort_needed = True

    def _on_update_shader(self, _caller, _event, calldata):
        """
        Fires every frame, inside VTK's render loop, with the GL context
        guaranteed current.  Three jobs:
          1. Lazy SSBO creation / data uploads
          2. Auto-sort when camera has moved
          3. Set all custom uniforms and re-bind SSBOs
        """
        program = calldata
        if program is None:
            return

        # ── 1. Lazy SSBO creation ─────────────────────────────────────── #
        if not self._ssbo_ready:
            ids = gl.glGenBuffers(2)
            self._data_ssbo  = int(ids[0])
            self._index_ssbo = int(ids[1])
            self._ssbo_ready = True

        # ── 2. Auto-sort ─────────────────────────────────────────────── #
        if self._sort_needed and self._gaussians is not None:
            vtk_cam = self._vtk_renderer.GetActiveCamera()
            view_mat = self._view_matrix_from_vtk(vtk_cam)
            idx = _sort_gaussian(self._gaussians, view_mat)
            self._pending_index = idx.flatten().astype(np.int32)
            self._index_dirty   = True
            self._sort_needed   = False

        # ── 3a. Upload Gaussian data SSBO ─────────────────────────────── #
        if self._data_dirty and self._pending_data is not None:
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self._data_ssbo)
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,
                            self._pending_data.nbytes,
                            self._pending_data,
                            gl.GL_DYNAMIC_DRAW)
            self._data_dirty = False

        # ── 3b. Upload sort-index SSBO ───────────────────────────────── #
        if self._index_dirty and self._pending_index is not None:
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self._index_ssbo)
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,
                            self._pending_index.nbytes,
                            self._pending_index,
                            gl.GL_DYNAMIC_DRAW)
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
            self._index_dirty = False

        # ── 4. Bind SSBOs to the expected binding points ──────────────── #
        if self._ssbo_ready:
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, self._data_ssbo)
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1, self._index_ssbo)

        # ── 5. Set uniforms ───────────────────────────────────────────── #
        vtk_cam  = self._vtk_renderer.GetActiveCamera()
        size     = self._vtk_renderer.GetSize()
        w, h     = max(size[0], 1), max(size[1], 1)

        pos      = np.array(vtk_cam.GetPosition(),   dtype=np.float32)
        focal_pt = np.array(vtk_cam.GetFocalPoint(),  dtype=np.float32)
        up       = np.array(vtk_cam.GetViewUp(),      dtype=np.float32)
        fovy_deg = vtk_cam.GetViewAngle()

        fovy_rad  = np.radians(fovy_deg)
        aspect    = w / h
        htany     = np.tan(fovy_rad / 2.0)
        htanx     = htany * aspect
        focal_len = h / (2.0 * htany)

        view_mat = np.array(glm.lookAt(
            glm.vec3(*pos.tolist()),
            glm.vec3(*focal_pt.tolist()),
            glm.vec3(*up.tolist())
        ), dtype=np.float32)

        proj_mat = np.array(glm.perspective(
            fovy_rad, float(aspect), 0.01, 100.0
        ), dtype=np.float32)

        pid = program.GetHandle()
        self._set_mat4(pid, "view_matrix",       view_mat)
        self._set_mat4(pid, "projection_matrix", proj_mat)
        self._set_v3  (pid, "cam_pos",           pos)
        self._set_v3  (pid, "hfovxy_focal",      np.array([htanx, htany, focal_len], np.float32))
        self._set_1f  (pid, "scale_modifier",    self._scale_modifier)
        self._set_1i  (pid, "sh_dim",            self._sh_dim)
        self._set_1i  (pid, "render_mod",        self._render_mod)

    # ── Uniform helpers ────────────────────────────────────────────────────── #

    @staticmethod
    def _set_mat4(pid: int, name: str, mat: np.ndarray):
        loc = gl.glGetUniformLocation(pid, name)
        if loc >= 0:
            # glm mat4 is column-major; numpy is row-major — transpose before upload
            gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE,
                                  mat.T.flatten().astype(np.float32))

    @staticmethod
    def _set_v3(pid: int, name: str, v: np.ndarray):
        loc = gl.glGetUniformLocation(pid, name)
        if loc >= 0:
            gl.glUniform3f(loc, float(v[0]), float(v[1]), float(v[2]))

    @staticmethod
    def _set_1f(pid: int, name: str, v: float):
        loc = gl.glGetUniformLocation(pid, name)
        if loc >= 0:
            gl.glUniform1f(loc, float(v))

    @staticmethod
    def _set_1i(pid: int, name: str, v: int):
        loc = gl.glGetUniformLocation(pid, name)
        if loc >= 0:
            gl.glUniform1i(loc, int(v))

    @staticmethod
    def _view_matrix_from_vtk(vtk_cam) -> np.ndarray:
        """Build a glm-style numpy view matrix from a vtkCamera."""
        pos   = np.array(vtk_cam.GetPosition(),   dtype=np.float32)
        focal = np.array(vtk_cam.GetFocalPoint(),  dtype=np.float32)
        up    = np.array(vtk_cam.GetViewUp(),      dtype=np.float32)
        return np.array(glm.lookAt(
            glm.vec3(*pos.tolist()),
            glm.vec3(*focal.tolist()),
            glm.vec3(*up.tolist())
        ), dtype=np.float32)
