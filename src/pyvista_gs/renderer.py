from __future__ import annotations

import importlib.util
import os

import numpy as np
import moderngl

from . import data as util_gau
from . import utils as util

try:
    from OpenGL.raw.WGL.EXT.swap_control import wglSwapIntervalEXT
except Exception:
    wglSwapIntervalEXT = None


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
VS_PATH = os.path.join(MODULE_DIR, 'shaders', 'gau_vert.glsl')
FS_PATH = os.path.join(MODULE_DIR, 'shaders', 'gau_frag.glsl')

_sort_buffer_xyz = None
_sort_buffer_gausid = None


def _sort_gaussian_cpu(gaus, view_mat):
    xyz = np.asarray(gaus.xyz)
    view_mat = np.asarray(view_mat)

    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]

    index = np.argsort(depth)
    index = index.astype(np.int32).reshape(-1, 1)
    return index


def _sort_gaussian_cupy(gaus, view_mat):
    import cupy as cp  # type: ignore[reportMissingImports]

    global _sort_buffer_gausid, _sort_buffer_xyz
    if _sort_buffer_gausid != id(gaus):
        _sort_buffer_xyz = cp.asarray(gaus.xyz)
        _sort_buffer_gausid = id(gaus)

    xyz = _sort_buffer_xyz
    view_mat = cp.asarray(view_mat)

    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]

    index = cp.argsort(depth)
    index = index.astype(cp.int32).reshape(-1, 1)

    index = cp.asnumpy(index)
    return index


def _sort_gaussian_torch(gaus, view_mat):
    global _sort_buffer_gausid, _sort_buffer_xyz
    if _sort_buffer_gausid != id(gaus):
        _sort_buffer_xyz = torch.tensor(gaus.xyz).cuda()
        _sort_buffer_gausid = id(gaus)

    xyz = _sort_buffer_xyz
    view_mat = torch.tensor(view_mat).cuda()
    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]
    index = torch.argsort(depth)
    index = index.type(torch.int32).reshape(-1, 1).cpu().numpy()
    return index


_sort_gaussian = None
try:
    import torch  # type: ignore[reportMissingImports]
    if not torch.cuda.is_available():
        raise ImportError
    print("Detect torch cuda installed, will use torch as sorting backend")
    _sort_gaussian = _sort_gaussian_torch
except ImportError:
    if importlib.util.find_spec("cupy") is not None:
        print("Detect cupy installed, will use cupy as sorting backend")
        _sort_gaussian = _sort_gaussian_cupy
    else:
        _sort_gaussian = _sort_gaussian_cpu


class GaussianRenderBase:
    def __init__(self):
        self.gaussians = None
        self._reduce_updates = True

    @property
    def reduce_updates(self):
        return self._reduce_updates

    @reduce_updates.setter
    def reduce_updates(self, val):
        self._reduce_updates = val
        self.update_vsync()

    def update_vsync(self):
        print("VSync is not supported")

    def update_gaussian_data(self, _gaus: util_gau.GaussianData):
        del _gaus
        raise NotImplementedError()

    def sort_and_update(self):
        raise NotImplementedError()

    def set_scale_modifier(self, _modifier: float):
        del _modifier
        raise NotImplementedError()

    def set_render_mod(self, _mod: int):
        del _mod
        raise NotImplementedError()

    def update_camera_pose(self, _camera: util.Camera):
        del _camera
        raise NotImplementedError()

    def update_camera_intrin(self, _camera: util.Camera):
        del _camera
        raise NotImplementedError()

    def draw(self):
        raise NotImplementedError()

    def set_render_reso(self, _w, _h):
        del _w, _h
        raise NotImplementedError()


class ModernGLRenderer(GaussianRenderBase):
    def __init__(self, w, h):
        super().__init__()
        self.ctx = moderngl.create_context(require=430)
        self.ctx.viewport = (0, 0, w, h)

        self._crop_bounds: np.ndarray | None = None

        with open(VS_PATH, 'r', encoding='utf-8') as f:
            vs_source = f.read()
        with open(FS_PATH, 'r', encoding='utf-8') as f:
            fs_source = f.read()

        self.program = self.ctx.program(vertex_shader=vs_source, fragment_shader=fs_source)

        self.quad_v = np.array([
            -1, 1,
            1, 1,
            1, -1,
            -1, -1,
        ], dtype=np.float32).reshape(4, 2)
        self.quad_f = np.array([
            0, 1, 2,
            0, 2, 3,
        ], dtype=np.int32).reshape(2, 3)

        self.vbo = self.ctx.buffer(self.quad_v.tobytes())
        self.ibo = self.ctx.buffer(self.quad_f.tobytes())
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, '2f', 'position')], self.ibo)

        self.gau_buffer = None
        self.index_buffer = None

        self.update_vsync()

    def update_vsync(self):
        if wglSwapIntervalEXT is not None:
            wglSwapIntervalEXT(1 if self.reduce_updates else 0)
        else:
            print("VSync is not supported")

    def set_crop_bounds(self, bounds):
        if bounds is None:
            self._crop_bounds = None
            return
        crop_bounds = np.asarray(bounds, dtype=np.float32).ravel()
        if crop_bounds.size != 6:
            raise ValueError("Crop bounds must contain 6 values: xmin, xmax, ymin, ymax, zmin, zmax")
        self._crop_bounds = crop_bounds

    def clear_crop_bounds(self):
        self._crop_bounds = None

    def _upload_crop_uniforms(self):
        enabled = 1 if self._crop_bounds is not None else 0
        if 'crop_enabled' in self.program:
            self.program['crop_enabled'].value = enabled
        if self._crop_bounds is not None:
            if 'crop_min' in self.program:
                self.program['crop_min'].value = tuple(self._crop_bounds[[0, 2, 4]])
            if 'crop_max' in self.program:
                self.program['crop_max'].value = tuple(self._crop_bounds[[1, 3, 5]])

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        self.gaussians = gaus
        gaussian_data = gaus.flat()

        if self.gau_buffer is None or self.gau_buffer.size < gaussian_data.nbytes:
            if self.gau_buffer:
                self.gau_buffer.release()
            self.gau_buffer = self.ctx.buffer(gaussian_data.tobytes())
        else:
            self.gau_buffer.write(gaussian_data.tobytes())

        self.gau_buffer.bind_to_storage_buffer(binding=0)

        if 'sh_dim' in self.program:
            self.program['sh_dim'].value = gaus.sh_dim

    def sort_and_update(self, camera: util.Camera):
        index = _sort_gaussian(self.gaussians, camera.get_view_matrix())
        if self.index_buffer is None or self.index_buffer.size < index.nbytes:
            if self.index_buffer:
                self.index_buffer.release()
            self.index_buffer = self.ctx.buffer(index.tobytes())
        else:
            self.index_buffer.write(index.tobytes())

        self.index_buffer.bind_to_storage_buffer(binding=1)

    def set_scale_modifier(self, modifier):
        if 'scale_modifier' in self.program:
            self.program['scale_modifier'].value = float(modifier)

    def set_render_mod(self, mod: int):
        if 'render_mod' in self.program:
            self.program['render_mod'].value = int(mod)

    def set_render_reso(self, w, h):
        self.ctx.viewport = (0, 0, w, h)

    def set_model_matrix(self, matrix):
        if matrix is None:
            matrix = np.eye(4, dtype=np.float32)
        if 'model_matrix' in self.program:
            self.program['model_matrix'].write(matrix.astype('f4').T.tobytes())

    def update_camera_pose(self, camera: util.Camera):
        view_mat = camera.get_view_matrix()
        if 'view_matrix' in self.program:
            self.program['view_matrix'].write(view_mat.astype('f4').T.tobytes())
        if 'cam_pos' in self.program:
            self.program['cam_pos'].value = tuple(camera.position)

    def update_camera_intrin(self, camera: util.Camera):
        proj_mat = camera.get_project_matrix()
        if 'projection_matrix' in self.program:
            self.program['projection_matrix'].write(proj_mat.astype('f4').T.tobytes())
        if 'hfovxy_focal' in self.program:
            self.program['hfovxy_focal'].value = tuple(camera.get_htanfovxy_focal())

    def draw(self):
        self._upload_crop_uniforms()
        num_gau = len(self.gaussians)
        self.vao.render(moderngl.TRIANGLES, instances=num_gau)

    def cleanup(self):
        try:
            if self.gau_buffer:
                self.gau_buffer.release()
                self.gau_buffer = None
            if self.index_buffer:
                self.index_buffer.release()
                self.index_buffer = None
            if self.vao:
                self.vao.release()
                self.vao = None
            if self.vbo:
                self.vbo.release()
                self.vbo = None
            if self.ibo:
                self.ibo.release()
                self.ibo = None
            if self.program:
                self.program.release()
                self.program = None
        except Exception as e:
            print(f"Warning: ModernGL cleanup failed: {e}")
