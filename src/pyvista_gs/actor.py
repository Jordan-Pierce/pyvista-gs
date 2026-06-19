from __future__ import annotations

import numpy as np
import pyvista as pv
import vtk
from . import data as util_gau
from .renderer import ModernGLRenderer


def _rotation_matrix_to_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
    trace = float(np.trace(rotation_matrix))

    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / scale
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / scale
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / scale
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        scale = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2.0
        w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale
        z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        scale = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2.0
        w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / scale
        x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2.0
        w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / scale
        x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale
        y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale
        z = 0.25 * scale

    quaternion = np.array([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quaternion / norm


def _multiply_quaternions_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64).reshape(1, 4)
    right = np.asarray(right, dtype=np.float64).reshape(-1, 4)

    w1, x1, y1, z1 = left[0]
    w2 = right[:, 0]
    x2 = right[:, 1]
    y2 = right[:, 2]
    z2 = right[:, 3]

    return np.column_stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _coerce_crop_bounds(bounds) -> np.ndarray:
    """Convert a PyVista crop widget payload into xmin/xmax/ymin/ymax/zmin/zmax."""
    if bounds is None:
        raise ValueError("Crop bounds cannot be None")

    if hasattr(bounds, "GetBounds"):
        bounds = bounds.GetBounds()
    elif hasattr(bounds, "bounds"):
        bounds = bounds.bounds

    if hasattr(bounds, "points"):
        points = np.asarray(bounds.points, dtype=np.float64)
        if points.ndim == 2 and points.shape[1] == 3:
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            return np.array([mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]], dtype=np.float64)

    bounds_array = np.asarray(bounds, dtype=np.float64)

    if bounds_array.size == 6:
        return bounds_array.reshape(6)

    if bounds_array.ndim == 2 and bounds_array.shape == (3, 2):
        return np.array([
            bounds_array[0, 0], bounds_array[0, 1],
            bounds_array[1, 0], bounds_array[1, 1],
            bounds_array[2, 0], bounds_array[2, 1],
        ], dtype=np.float64)

    if bounds_array.ndim == 2 and bounds_array.shape[1] == 3:
        mins = bounds_array.min(axis=0)
        maxs = bounds_array.max(axis=0)
        return np.array([mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]], dtype=np.float64)

    raise ValueError("Crop bounds must resolve to 6 values")


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
        self.renderer: ModernGLRenderer | None = None
        self._sync_needed = True
        self._last_mtime = 0

        self._last_view_matrix = None
        self._sort_tolerance = 1e-4

        self.scale_modifier = 1.0
        self.render_mode = 7
        self.auto_sort = True
        self.reduce_updates = True
        self._crop_bounds: np.ndarray | None = None

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

    def set_crop_bounds(self, bounds: np.ndarray):
        """Enable a shader-side crop preview without mutating the mesh."""
        crop_bounds = _coerce_crop_bounds(bounds)
        self._crop_bounds = crop_bounds
        if self.renderer:
            self.renderer.set_crop_bounds(crop_bounds)

    def clear_crop_box(self):
        """Disable the crop preview."""
        self._crop_bounds = None
        if self.renderer:
            self.renderer.clear_crop_bounds()

    def transform(self, matrix: np.ndarray):
        """
        Apply a 4x4 homogeneous transform to the splat positions, rotations, and scales.

        PyVista updates point positions, but 3DGS also stores per-point quaternion and
        scale data. We keep those attributes in sync so the actor remains mathematically
        consistent after being moved, rotated, or non-uniformly scaled by callers.
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("GaussianActor.transform expects a 4x4 matrix")

        self._mesh.transform(matrix, inplace=True)
        self._original_mesh.transform(matrix, inplace=True)

        applied_scales = np.linalg.norm(matrix[:3, :3], axis=1)
        safe_scales = np.where(applied_scales == 0.0, 1.0, applied_scales)
        rotation_matrix = matrix[:3, :3] / safe_scales[:, None]
        applied_quaternion = _rotation_matrix_to_wxyz(rotation_matrix)

        current_rots = np.asarray(self._mesh.point_data['rot'], dtype=np.float64)
        new_rots = _multiply_quaternions_wxyz(applied_quaternion, current_rots)
        new_rots_norm = np.linalg.norm(new_rots, axis=1, keepdims=True)
        new_rots_norm = np.where(new_rots_norm == 0.0, 1.0, new_rots_norm)
        new_rots = (new_rots / new_rots_norm).astype(np.float32)

        current_scales = np.asarray(self._mesh.point_data['scale'], dtype=np.float64)
        new_scales = (current_scales * applied_scales).astype(np.float32)

        self._mesh.point_data['rot'] = new_rots
        self._original_mesh.point_data['rot'] = new_rots.copy()
        self._mesh.point_data['scale'] = new_scales
        self._original_mesh.point_data['scale'] = new_scales.copy()

        self._sync_needed = True
        self._last_view_matrix = None

    def remove_floaters(self, min_opacity: float = 0.05, max_scale: float = 1.0):
        """
        Cull noisy splats that are too transparent or too large.

        The selection is applied to both the live mesh and the original backup mesh so
        cropping and reset operations continue to see the same cleaned dataset.
        """
        if self.point_count == 0:
            return

        opacities = np.asarray(self._mesh.point_data['opacity']).ravel()
        scales = np.asarray(self._mesh.point_data['scale'])
        max_scales_per_splat = np.max(scales, axis=1)

        valid_mask = (opacities >= min_opacity) & (max_scales_per_splat <= max_scale)
        if not np.any(valid_mask):
            print("Cull aborted: Parameters are too aggressive and would delete the entire model.")
            return

        points_removed = valid_mask.size - int(np.sum(valid_mask))
        if points_removed == 0:
            return

        culled_mesh = self._mesh.extract_points(valid_mask)
        self.mesh = culled_mesh
        self._original_mesh = culled_mesh.copy()
        self._sync_needed = True
        self._last_view_matrix = None

        print(f"Culled {points_removed:,} floaters. Remaining splats: {self.point_count:,}")

    def tint_gaussians(self, indices: np.ndarray, color_rgb: tuple[int, int, int], blend_factor: float = 0.6):
        """
        Tint selected splats by modifying the DC spherical harmonic coefficients.

        `indices` may be an integer index array or a boolean mask. Only the first three
        SH coefficients are modified so the view-dependent bands remain untouched.
        """
        if self.point_count == 0:
            return

        selection = np.asarray(indices)
        if selection.size == 0:
            return

        if selection.dtype == bool:
            selection = selection.ravel()
            if selection.size != self.point_count:
                raise ValueError("Boolean mask length must match the number of splats")
            selection = np.flatnonzero(selection)
        else:
            selection = selection.astype(np.intp, copy=False).ravel()

        if selection.size == 0:
            return

        blend_factor = float(np.clip(blend_factor, 0.0, 1.0))
        target_rgb = np.clip(np.asarray(color_rgb, dtype=np.float64) / 255.0, 0.0, 1.0)

        sh_c0 = 0.28209479177387814
        target_sh_dc = (target_rgb - 0.5) / sh_c0

        current_sh = np.asarray(self._mesh.point_data['sh'], dtype=np.float64)
        current_dc = current_sh[selection, 0:3]
        new_dc = current_dc * (1.0 - blend_factor) + target_sh_dc * blend_factor
        current_sh[selection, 0:3] = new_dc

        tinted_sh = current_sh.astype(np.float32)
        self._mesh.point_data['sh'] = tinted_sh
        self._original_mesh.point_data['sh'] = tinted_sh.copy()
        self._sync_needed = True
        self._last_view_matrix = None

    def apply_crop_box(self, bounds: np.ndarray | None = None):
        """
        Commit the current crop preview by deleting points outside the crop bounds.

        If bounds are provided they become the active preview bounds before the commit.
        """
        if bounds is not None:
            self.set_crop_bounds(bounds)

        if self._crop_bounds is None or self.point_count == 0:
            return

        crop_bounds = np.asarray(self._crop_bounds, dtype=np.float64)
        points = np.asarray(self._mesh.points, dtype=np.float64)
        valid_mask = (
            (points[:, 0] >= crop_bounds[0]) & (points[:, 0] <= crop_bounds[1]) &
            (points[:, 1] >= crop_bounds[2]) & (points[:, 1] <= crop_bounds[3]) &
            (points[:, 2] >= crop_bounds[4]) & (points[:, 2] <= crop_bounds[5])
        )

        if not np.any(valid_mask):
            print("Crop aborted: The current crop bounds would delete the entire model.")
            return

        points_removed = valid_mask.size - int(np.sum(valid_mask))
        if points_removed == 0:
            return

        culled_mesh = self._mesh.extract_points(valid_mask)
        self.mesh = culled_mesh
        self._original_mesh = culled_mesh.copy()
        self._sync_needed = True
        self._last_view_matrix = None

        print(f"Applied crop: removed {points_removed:,} splats. Remaining splats: {self.point_count:,}")

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

    # Expose common PyVista actor properties so users can manipulate the
    # GaussianActor the same way they would a standard PyVista actor.
    @property
    def position(self):
        return self.actor.position

    @position.setter
    def position(self, pos: tuple[float, float, float]):
        self.actor.position = pos
        try:
            self.actor.Modified()
        except Exception:
            pass

    @property
    def scale(self):
        return self.actor.scale

    @scale.setter
    def scale(self, scale_factor: tuple[float, float, float]):
        self.actor.scale = scale_factor
        try:
            self.actor.Modified()
        except Exception:
            pass

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

        # If the PyVista actor has a transform (position/scale/orientation),
        # apply it so picking is performed in world space.
        try:
            vtk_mat = self.actor.GetMatrix()
            model_mat = np.zeros((4, 4), dtype=np.float64)
            for i in range(4):
                for j in range(4):
                    model_mat[i, j] = vtk_mat.GetElement(i, j)
            xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
            world_xyz = (model_mat @ xyz_h.T).T[:, :3]
        except Exception:
            world_xyz = xyz

        vecs = world_xyz - ray_origin
        t = np.sum(vecs * ray_dir, axis=1)

        front_mask = t > 0
        if not np.any(front_mask):
            return None

        front_xyz = world_xyz[front_mask]
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

        # Honor the VTK actor visibility flag so external code can hide/show splats
        if not self.actor.GetVisibility():
            return

        if self._sync_needed or self._mesh.GetMTime() > self._last_mtime:
            self._sync_to_renderer()

        window = caller.GetRenderWindow()
        w, h = window.GetSize()
        if w == 0 or h == 0:
            return

        if self.renderer is None:
            self.renderer = ModernGLRenderer(w, h)
            self._sync_to_renderer()

        self.renderer.set_crop_bounds(self._crop_bounds)
        self.renderer.set_scale_modifier(self.scale_modifier)
        self.renderer.set_render_mod(self.render_mode - 4)
        self.renderer.reduce_updates = self.reduce_updates
        self.renderer.set_render_reso(w, h)

        vtk_cam = caller.GetActiveCamera()
        cam_adapter = VTKCameraAdapter(vtk_cam, w, h)

        # Extract the current spatial transform of the PyVista actor and pass
        # it to the renderer so actor-level transforms are honored in the shader.
        try:
            vtk_matrix = self.actor.GetMatrix()
            model_matrix = np.zeros((4, 4), dtype=np.float32)
            for i in range(4):
                for j in range(4):
                    model_matrix[i, j] = vtk_matrix.GetElement(i, j)
        except Exception:
            model_matrix = np.eye(4, dtype=np.float32)

        self.renderer.set_model_matrix(model_matrix)

        if self.auto_sort:
            self.sort_gaussians(cam_adapter)

        self.renderer.ctx.enable(self.renderer.ctx.BLEND)
        self.renderer.ctx.blend_func = (self.renderer.ctx.SRC_ALPHA, self.renderer.ctx.ONE_MINUS_SRC_ALPHA)
        self.renderer.ctx.depth_mask = False

        self.renderer.update_camera_pose(cam_adapter)
        self.renderer.update_camera_intrin(cam_adapter)
        self.renderer.draw()

        self.renderer.ctx.depth_mask = True
        self.renderer.ctx.disable(self.renderer.ctx.BLEND)
