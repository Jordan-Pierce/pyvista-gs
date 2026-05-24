import numpy as np
import pyvista as pv
import vtk
import OpenGL.GL as gl

import util_gau
from renderer_ogl import OpenGLRenderer


class VTKCameraAdapter:
    """Bridges PyVista's vtkCamera and your util.Camera format."""
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
    """
    A 'Trojan Horse' PyVista object for 3D Gaussian Splatting.
    To PyVista, this is an invisible point cloud. To OpenGL, it's a 3DGS volume.
    """
    def __init__(self, gaussian_data: util_gau.GaussianData):
        self.renderer: OpenGLRenderer | None = None
        self._sync_needed = True
        self._last_mtime = 0
        
        # 3DGS Configuration
        self.scale_modifier = 1.0
        self.render_mode = 7 
        self.auto_sort = True
        self.reduce_updates = True

        # Map 3DGS data to a PyVista PolyData object
        self._mesh = pv.PolyData(gaussian_data.xyz)
        self._mesh.point_data['rot'] = gaussian_data.rot
        self._mesh.point_data['scale'] = gaussian_data.scale
        self._mesh.point_data['opacity'] = gaussian_data.opacity
        self._mesh.point_data['sh'] = gaussian_data.sh
        
        # Create the invisible VTK anchor
        self.mapper = pv.DataSetMapper(self._mesh)
        self.actor = pv.Actor(mapper=self.mapper)
        self.actor.prop.opacity = 0.0     # Hide VTK rendering
        self.actor.prop.point_size = 5.0  # Keep mathematical size for raycasting

    @property
    def mesh(self) -> pv.PolyData:
        return self._mesh

    @mesh.setter
    def mesh(self, new_mesh: pv.PolyData):
        """
        Allows users to replace the mesh (e.g., gs_actor.mesh = gs_actor.mesh.clip('y')).
        The actor will automatically detect this and rebuild the OpenGL data.
        """
        self._mesh = new_mesh
        self.mapper.dataset = self._mesh
        self._sync_needed = True

    @property
    def point_count(self) -> int:
        return self._mesh.n_points if self._mesh else 0

    def bind_to_plotter(self, plotter: pv.Plotter):
        """Adds the actor to the PyVista scene and hooks the OpenGL render pipeline."""
        plotter.add_actor(self.actor, pickable=True)
        plotter.renderer.AddObserver(vtk.vtkCommand.EndEvent, self._on_render_end)

    def _sync_to_renderer(self):
        """Extracts surviving VTK arrays and pushes them to the custom OpenGL renderer."""
        if self._mesh.n_points == 0:
            return

        # Safely extract opacity and ensure it maintains an (N, 1) shape. 
        # VTK flattens single-component arrays to 1D (N,).
        opacity_array = np.array(self._mesh.point_data['opacity'])
        if opacity_array.ndim == 1:
            opacity_array = opacity_array.reshape(-1, 1)

        rebuilt_gaussians = util_gau.GaussianData(
            xyz=np.array(self._mesh.points),
            rot=np.array(self._mesh.point_data['rot']),
            scale=np.array(self._mesh.point_data['scale']),
            opacity=opacity_array,
            sh=np.array(self._mesh.point_data['sh'])
        )
        
        if self.renderer:
            self.renderer.update_gaussian_data(rebuilt_gaussians)

        self._last_mtime = self._mesh.GetMTime()
        self._sync_needed = False

    def sort_gaussians(self, cam_adapter):
        if self.renderer and self.point_count > 0:
            self.renderer.sort_and_update(cam_adapter)

    def pick_gaussian(self, ray_origin: np.ndarray, ray_dir: np.ndarray, fovy_rad: float, window_height: int) -> np.ndarray | None:
        """Finds the closest Gaussian intersected by the given ray using angular tolerance."""
        if self.point_count == 0:
            return None

        xyz = np.array(self._mesh.points)
        vecs = xyz - ray_origin
        t = np.sum(vecs * ray_dir, axis=1)

        # Only consider points in front of the camera
        front_mask = t > 0
        if not np.any(front_mask):
            return None

        front_xyz = xyz[front_mask]
        front_vecs = front_xyz - ray_origin
        front_t = t[front_mask]

        # Calculate angular distance to the ray
        proj = front_t[:, None] * ray_dir
        dists = np.linalg.norm(front_vecs - proj, axis=1)
        angles = dists / front_t

        # Tolerance based on FOV and window height (approx 5 pixels)
        tolerance = 5.0 * (fovy_rad / window_height)

        hit_mask = angles < tolerance
        if np.any(hit_mask):
            hit_indices = np.where(hit_mask)[0]
            # Out of all hits, find the one closest to the camera
            best_idx = hit_indices[np.argmin(front_t[hit_mask])]
            return front_xyz[best_idx]
        
        return None

    def _on_render_end(self, caller, event):
        """Injected immediately after PyVista finishes drawing standard geometry."""
        if self.point_count == 0:
            return

        # Check if mesh has been modified (translated, points deleted, etc.)
        if self._sync_needed or self._mesh.GetMTime() > self._last_mtime:
            self._sync_to_renderer()

        window = caller.GetRenderWindow()
        w, h = window.GetSize()
        if w == 0 or h == 0:
            return

        # Lazy Init Renderer
        if self.renderer is None:
            self.renderer = OpenGLRenderer(w, h)
            self._sync_to_renderer()

        # Update Settings
        self.renderer.set_scale_modifier(self.scale_modifier)
        self.renderer.set_render_mod(self.render_mode - 4)
        self.renderer.reduce_updates = self.reduce_updates
        self.renderer.set_render_reso(w, h)

        vtk_cam = caller.GetActiveCamera()
        cam_adapter = VTKCameraAdapter(vtk_cam, w, h)

        if self.auto_sort:
            self.sort_gaussians(cam_adapter)

        # --- PRESERVE VTK STATE ---
        last_prog = gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)
        last_vao = gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING)
        last_blend = gl.glGetBoolean(gl.GL_BLEND)
        last_depth_mask = gl.glGetBoolean(gl.GL_DEPTH_WRITEMASK)
        
        # --- CONFIGURE 3DGS STATE ---
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDepthMask(gl.GL_FALSE) 
        
        # --- DRAW GAUSSIANS ---
        self.renderer.update_camera_pose(cam_adapter)
        self.renderer.update_camera_intrin(cam_adapter)
        self.renderer.draw()
        
        # --- RESTORE VTK STATE ---
        if last_depth_mask: gl.glDepthMask(gl.GL_TRUE)
        else: gl.glDepthMask(gl.GL_FALSE)
            
        if not last_blend: gl.glDisable(gl.GL_BLEND)
            
        gl.glUseProgram(last_prog)
        gl.glBindVertexArray(last_vao)