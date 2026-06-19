from __future__ import annotations

import os

import numpy as np
import vtk
from OpenGL import GL as gl
import glm
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.util.misc import calldata_type
from vtkmodules.vtkCommonCore import VTK_OBJECT

from . import data as util_gau

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROXY_VERT = os.path.join(MODULE_DIR, 'shaders', 'proxy_vert.glsl')
_PROXY_GEOM = os.path.join(MODULE_DIR, 'shaders', 'proxy_geom.glsl')
_PROXY_FRAG = os.path.join(MODULE_DIR, 'shaders', 'proxy_frag.glsl')


def _read_shader(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as fh:
        return fh.read()


class VTKProxyActor:
    """
    Invisible VTK actor that gives Gaussians first-class scene participation.

    The proxy emits correctly-sized billboard quads via a geometry shader so
    VTK's pick pass can hit them, but the fragment shader discards all pixels
    during normal rendering.
    """

    def __init__(self):
        self._vtk_renderer: vtk.vtkRenderer | None = None

        self._poly = vtk.vtkPolyData()
        self._pts = vtk.vtkPoints()
        self._pts.SetDataTypeToDouble()
        self._poly.SetPoints(self._pts)

        self._mapper = vtk.vtkOpenGLPolyDataMapper()
        self._mapper.SetInputData(self._poly)

        self._actor = vtk.vtkActor()
        self._actor.SetMapper(self._mapper)
        self._actor.ForceTranslucentOn()
        self._actor.GetProperty().SetOpacity(1.0)
        self._actor.GetProperty().SetPointSize(1)

        shader_prop = self._actor.GetShaderProperty()
        shader_prop.SetVertexShaderCode(_read_shader(_PROXY_VERT))
        shader_prop.SetGeometryShaderCode(_read_shader(_PROXY_GEOM))
        shader_prop.SetFragmentShaderCode(_read_shader(_PROXY_FRAG))

        self._data_ssbo: int | None = None
        self._index_ssbo: int | None = None
        self._ssbo_ready = False

        self._pending_data: np.ndarray | None = None
        self._pending_index: np.ndarray | None = None
        self._data_dirty = False
        self._index_dirty = False

        self._sh_dim = 3
        self._scale_modifier = 1.0

        @calldata_type(VTK_OBJECT)
        def _update_shader_cb(caller, event, calldata):
            self._on_update_shader(caller, event, calldata)

        self._update_shader_cb = _update_shader_cb

    @property
    def actor(self) -> vtk.vtkActor:
        return self._actor

    @property
    def mapper(self) -> vtk.vtkOpenGLPolyDataMapper:
        return self._mapper

    def attach_to_renderer(self, vtk_renderer: vtk.vtkRenderer):
        self._vtk_renderer = vtk_renderer
        vtk_renderer.AddActor(self._actor)
        self._mapper.AddObserver("UpdateShaderEvent", self._update_shader_cb)

    def update_data(self, gaus: util_gau.GaussianData):
        self._sh_dim = gaus.sh_dim
        n = len(gaus)

        pts_vtk = numpy_to_vtk(gaus.xyz.astype(np.float64), deep=True)
        self._pts.SetData(pts_vtk)

        cells_np = np.empty(2 * n, dtype=np.int64)
        cells_np[0::2] = 1
        cells_np[1::2] = np.arange(n)
        cell_vtk = numpy_to_vtk(cells_np, deep=True, array_type=vtk.VTK_ID_TYPE)
        verts = vtk.vtkCellArray()
        verts.SetCells(n, cell_vtk)
        self._poly.SetVerts(verts)
        self._poly.Modified()

        self._pending_data = gaus.flat().astype(np.float32)
        self._data_dirty = True

        self._pending_index = np.arange(n, dtype=np.int32)
        self._index_dirty = True

    def update_sort_indices(self, indices: np.ndarray):
        self._pending_index = np.asarray(indices, dtype=np.int32).ravel()
        self._index_dirty = True

    def set_scale_modifier(self, modifier: float):
        self._scale_modifier = float(modifier)

    def _on_update_shader(self, _caller, _event, calldata):
        program = calldata
        if program is None:
            return

        # Drain any stale GL errors left by VTK's pipeline so PyOpenGL's
        # error checker doesn't attribute them to our calls.
        while gl.glGetError() != gl.GL_NO_ERROR:
            pass

        if not self._ssbo_ready:
            ids = gl.glGenBuffers(2)
            self._data_ssbo = int(ids[0])
            self._index_ssbo = int(ids[1])
            self._ssbo_ready = True

        if self._data_dirty and self._pending_data is not None:
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self._data_ssbo)
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,
                            self._pending_data.nbytes,
                            self._pending_data,
                            gl.GL_DYNAMIC_DRAW)
            self._data_dirty = False

        if self._index_dirty and self._pending_index is not None:
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self._index_ssbo)
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,
                            self._pending_index.nbytes,
                            self._pending_index,
                            gl.GL_DYNAMIC_DRAW)
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
            self._index_dirty = False

        if self._ssbo_ready:
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, self._data_ssbo)
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1, self._index_ssbo)

        if self._vtk_renderer is None:
            return

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
        self._set_v3(pid, "hfovxy_focal", np.array([htanx, htany, focal_len], np.float32))
        self._set_1f(pid, "scale_modifier", self._scale_modifier)
        self._set_1i(pid, "sh_dim", self._sh_dim)

    def cleanup(self):
        if self._vtk_renderer is not None and self._actor is not None:
            self._vtk_renderer.RemoveActor(self._actor)

        try:
            buffers = []
            if self._data_ssbo is not None:
                buffers.append(self._data_ssbo)
            if self._index_ssbo is not None:
                buffers.append(self._index_ssbo)
            if buffers:
                gl.glDeleteBuffers(len(buffers), buffers)
        except Exception:
            pass

        self._data_ssbo = None
        self._index_ssbo = None
        self._ssbo_ready = False
        self._vtk_renderer = None

    @staticmethod
    def _set_mat4(pid: int, name: str, mat: np.ndarray):
        loc = gl.glGetUniformLocation(pid, name)
        if loc >= 0:
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
