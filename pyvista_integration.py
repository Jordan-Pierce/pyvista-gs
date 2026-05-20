import numpy as np
import pyvista as pv
import vtk
import OpenGL.GL as gl

# Import your existing codebase modules
import util_gau
from renderer_ogl import OpenGLRenderer


class VTKCameraAdapter:
    """
    Acts as a bridge between PyVista's vtkCamera and your util.Camera,
    providing the matrices in the exact format renderer_ogl expects.
    """
    def __init__(self, vtk_cam, width, height):
        self.vtk_cam = vtk_cam
        self.w = width
        self.h = height
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


def main():

    # This example demonstrates how to integrate your Gaussian Splat Renderer into a PyVista application.
    # 1. Load Gaussian Data using your existing loader
    ply_path = r"path/to/your/gaussians.ply"
    print(f"Loading Gaussians from {ply_path}...")
    gaussians = util_gau.load_ply(ply_path)
    print(f"Loaded {len(gaussians)} splats.")

    # --- Center the Gaussians at PyVista's (0,0,0) ---
    centroid = np.mean(gaussians.xyz, axis=0)
    gaussians.xyz -= centroid
    # -----------------------------------------------------

    print(f"Loaded {len(gaussians)} splats. Shifted by {-centroid}")

    # 2. Setup PyVista Plotter
    plotter = pv.Plotter()
    plotter.set_background("#13161f")

    # Trick PyVista into knowing the bounds of our Gaussians so the camera 
    # interaction and clipping planes (znear/zfar) work perfectly.
    # We use a point cloud of the Gaussian centers, but set opacity to 0.
    dummy_pc = pv.PolyData(gaussians.xyz)
    plotter.add_mesh(dummy_pc, opacity=0.0)

    # 3. Global state for our custom OpenGL renderer
    ogl_renderer = None

    # 4. Define the callback hook
    def on_render_end(caller, event):
        nonlocal ogl_renderer
        
        window = caller.GetRenderWindow()
        w, h = window.GetSize()
        
        # Initialize our Gaussian renderer on the first frame (Context is now active!)
        if ogl_renderer is None:
            ogl_renderer = OpenGLRenderer(w, h)
            ogl_renderer.update_gaussian_data(gaussians)
            ogl_renderer.set_scale_modifier(1.0)
            ogl_renderer.set_render_mod(3) # Default SH mode
        
        # Update resolution if window was resized
        ogl_renderer.set_render_reso(w, h)
        
        # Wrap the VTK camera
        vtk_cam = caller.GetActiveCamera()
        cam_adapter = VTKCameraAdapter(vtk_cam, w, h)
        
        # Sort the Gaussians based on the new camera view
        # (Using your existing CPU/Torch/CuPy logic)
        ogl_renderer.sort_and_update(cam_adapter)
        
        # --- PRESERVE VTK STATE ---
        last_prog = gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)
        last_vao = gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING)
        last_blend = gl.glGetBooleanv(gl.GL_BLEND)
        last_depth_mask = gl.glGetBooleanv(gl.GL_DEPTH_WRITEMASK)
        
        # --- CONFIGURE OUR 3DGS STATE ---
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        
        # Disable writing to the depth buffer for Gaussians
        gl.glDepthMask(gl.GL_FALSE) 
        
        # --- DRAW GAUSSIANS ---
        ogl_renderer.update_camera_pose(cam_adapter)
        ogl_renderer.update_camera_intrin(cam_adapter)
        ogl_renderer.draw()
        
        # --- RESTORE VTK STATE ---
        # Restore Depth Mask
        if last_depth_mask:
            gl.glDepthMask(gl.GL_TRUE)
        else:
            gl.glDepthMask(gl.GL_FALSE)
            
        # Restore Blend state
        if not last_blend:
            gl.glDisable(gl.GL_BLEND)
            
        # Restore Program and VAO
        gl.glUseProgram(last_prog)
        gl.glBindVertexArray(last_vao)


    # 5. Attach the observer to PyVista's underlying VTK Renderer
    plotter.renderer.AddObserver(vtk.vtkCommand.EndEvent, on_render_end)

    # Launch PyVista window
    plotter.show()

if __name__ == '__main__':
    main()