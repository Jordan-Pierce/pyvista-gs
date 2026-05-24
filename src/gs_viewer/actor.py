from __future__ import annotations

import numpy as np
import pyvista as pv
import vtk
import OpenGL.GL as gl

from . import data as util_gau
from .renderer import OpenGLRenderer


class VTKCameraAdapter:
    """Bridges PyVista's vtkCamera and the util.Camera interface."""

    def __init__(self, vtk_cam, width, height):
        self.vtk_cam = vtk_cam
        self.w = max(width, 1)
        self.h = max(height, 1)
        self.position = np.array(vtk_cam.GetPosition(), dtype=np.float32)

    def _vtk_to_numpy(self, vtk_matrix):
        m = np.zeros((4, 4), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                m[i, j] = vtk_matrix.GetElement(i, j)
        return m

    def get_view_matrix(self):
        mat = self.vtk_cam.GetModelViewTransformObject().GetMatrix()
        return self._vtk_to_numpy(mat)

    def get_project_matrix(self):
        aspect = self.w / self.h if self.h != 0 else 1.0
        mat = self.vtk_cam.GetProjectionTransformMatrix(aspect, -1, 1)
        return self._vtk_to_numpy(mat)

    def get_htanfovxy_focal(self):
        fovy = np.radians(self.vtk_cam.GetViewAngle())
        htany = np.tan(fovy / 2.0)
        htanx = htany / self.h * self.w
        focal = self.h / (2.0 * htany)
        return [htanx, htany, focal]


class GaussianActor:
    """A PyVista actor that proxies the Gaussian splatting renderer."""

    def __init__(self, gaussian_data: util_gau.GaussianData):
        self.renderer: OpenGLRenderer | None = None
        self._sync_needed = True
        self._last_mtime = 0

        self._last_view_matrix = None
        self._sort_tolerance = 1e-4

        self.scale_modifier = 1.0
        self.render_mode = 7
        self.auto_sort = True
        self.reduce_updates = True

        self._mesh = pv.PolyData(gaussian_data.xyz)
        self._mesh.point_data['rot'] = gaussian_data.rot
        self._mesh.point_data['scale'] = gaussian_data.scale
        self._mesh.point_data['opacity'] = gaussian_data.opacity
        self._mesh.point_data['sh'] = gaussian_data.sh

        self._original_mesh = self._mesh.copy()

        self.mapper = pv.DataSetMapper(self._mesh)
        self.actor = pv.Actor(mapper=self.mapper)
        self.actor.prop.opacity = 0.0
        self.actor.prop.point_size = 5.0

    def cleanup(self):
        """Release the OpenGL renderer resources."""
        if self.renderer:
            self.renderer.cleanup()
            self.renderer = None

    def apply_crop_box(self, bounds):
        """Clip the original mesh to the given bounds and trigger an update."""
        self.mesh = self._original_mesh.clip_box(bounds, invert=False)

    @property
    def mesh(self) -> pv.PolyData:
        return self._mesh

    @mesh.setter
    def mesh(self, new_mesh: pv.PolyData):
        self._mesh = new_mesh
        self.mapper.dataset = self._mesh
        self._sync_needed = True

    @property
    def point_count(self) -> int:
        return self._mesh.n_points if self._mesh else 0

    def bind_to_plotter(self, plotter: pv.Plotter):
        plotter.add_actor(self.actor, pickable=True)
        plotter.renderer.AddObserver(vtk.vtkCommand.EndEvent, self._on_render_end)

    def _sync_to_renderer(self):
        if self._mesh.n_points == 0:
            return

        opacity_array = np.array(self._mesh.point_data['opacity'])
        if opacity_array.ndim == 1:
            opacity_array = opacity_array.reshape(-1, 1)

        rebuilt_gaussians = util_gau.GaussianData(
            xyz=np.array(self._mesh.points),
            rot=np.array(self._mesh.point_data['rot']),
            scale=np.array(self._mesh.point_data['scale']),
            opacity=opacity_array,
            sh=np.array(self._mesh.point_data['sh']),
        )

        if self.renderer:
            self.renderer.update_gaussian_data(rebuilt_gaussians)

        self._last_mtime = self._mesh.GetMTime()
        self._sync_needed = False
        self._last_view_matrix = None

    def pick_gaussian(self, ray_origin: np.ndarray, ray_dir: np.ndarray, fovy_rad: float, window_height: int) -> np.ndarray | None:
        if self.point_count == 0:
            return None

        xyz = np.array(self._mesh.points)
        vecs = xyz - ray_origin
        t = np.sum(vecs * ray_dir, axis=1)

        front_mask = t > 0
        if not np.any(front_mask):
            return None

        front_xyz = xyz[front_mask]
        front_vecs = front_xyz - ray_origin
        front_t = t[front_mask]

        proj = front_t[:, None] * ray_dir
        dists = np.linalg.norm(front_vecs - proj, axis=1)
        angles = dists / front_t

        tolerance = 5.0 * (fovy_rad / window_height)
        hit_mask = angles < tolerance

        if np.any(hit_mask):
            hit_indices = np.where(hit_mask)[0]
            best_idx = hit_indices[np.argmin(front_t[hit_mask])]
            return front_xyz[best_idx]

        return None

    def sort_gaussians(self, cam_adapter):
        if not self.renderer or self.point_count == 0:
            return

        view_mat = cam_adapter.get_view_matrix()

        if self._last_view_matrix is not None:
            if np.allclose(view_mat, self._last_view_matrix, atol=self._sort_tolerance):
                return

        self.renderer.sort_and_update(cam_adapter)
        self._last_view_matrix = view_mat.copy()

    def _on_render_end(self, caller, _event):
        del _event
        if self.point_count == 0:
            return

        if self._sync_needed or self._mesh.GetMTime() > self._last_mtime:
            self._sync_to_renderer()

        window = caller.GetRenderWindow()
        w, h = window.GetSize()
        if w == 0 or h == 0:
            return

        if self.renderer is None:
            self.renderer = OpenGLRenderer(w, h)
            self._sync_to_renderer()

        self.renderer.set_scale_modifier(self.scale_modifier)
        self.renderer.set_render_mod(self.render_mode - 4)
        self.renderer.reduce_updates = self.reduce_updates
        self.renderer.set_render_reso(w, h)

        vtk_cam = caller.GetActiveCamera()
        cam_adapter = VTKCameraAdapter(vtk_cam, w, h)

        if self.auto_sort:
            self.sort_gaussians(cam_adapter)

        last_prog = gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)
        last_vao = gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING)
        last_blend = gl.glGetBoolean(gl.GL_BLEND)
        last_depth_mask = gl.glGetBoolean(gl.GL_DEPTH_WRITEMASK)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDepthMask(gl.GL_FALSE)

        self.renderer.update_camera_pose(cam_adapter)
        self.renderer.update_camera_intrin(cam_adapter)
        self.renderer.draw()

        if last_depth_mask:
            gl.glDepthMask(gl.GL_TRUE)
        else:
            gl.glDepthMask(gl.GL_FALSE)

        if not last_blend:
            gl.glDisable(gl.GL_BLEND)

        gl.glUseProgram(last_prog)
        gl.glBindVertexArray(last_vao)
