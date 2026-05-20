import numpy as np
import pyvista as pv
import vtk
from plyfile import PlyData

def load_3dgs_as_polydata(ply_path):
    print("Loading PLY data...")
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    # 1. Extract Positions
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    mesh = pv.PolyData(xyz)
    
    # 2. Extract Scales (exponentiate)
    scale_names = ['scale_0', 'scale_1', 'scale_2']
    scales = np.exp(np.stack([vertex[n] for n in scale_names], axis=-1))
    mesh.point_data['gs_scales'] = scales.astype(np.float32)
    
    # 3. Extract Quaternions (normalize)
    rot_names = ['rot_0', 'rot_1', 'rot_2', 'rot_3']
    quats = np.stack([vertex[n] for n in rot_names], axis=-1)
    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    mesh.point_data['gs_quats'] = quats.astype(np.float32)
    
    # 4. Extract Opacity (sigmoid)
    opacity = 1.0 / (1.0 + np.exp(-vertex['opacity']))
    mesh.point_data['gs_opacity'] = opacity.astype(np.float32)
    
    # 5. Extract Base Color (SH DC Component)
    sh_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=-1)
    mesh.point_data['gs_sh_dc'] = sh_dc.astype(np.float32)

    # 6. Extract Degree 1 Spherical Harmonics (View-Dependent Lighting)
    sh1_y = np.stack([vertex['f_rest_0'], vertex['f_rest_1'], vertex['f_rest_2']], axis=-1)
    sh1_z = np.stack([vertex['f_rest_3'], vertex['f_rest_4'], vertex['f_rest_5']], axis=-1)
    sh1_x = np.stack([vertex['f_rest_6'], vertex['f_rest_7'], vertex['f_rest_8']], axis=-1)

    mesh.point_data['gs_sh1_y'] = sh1_y.astype(np.float32)
    mesh.point_data['gs_sh1_z'] = sh1_z.astype(np.float32)
    mesh.point_data['gs_sh1_x'] = sh1_x.astype(np.float32)
    
    # Center the mesh (Optional, but recommended for viewing)
    centroid = np.mean(mesh.points, axis=0)
    mesh.points -= centroid
    
    return mesh

def apply_3dgs_shaders(actor, mesh):
    """Modifies the VTK Mapper and Shader Property to render true 3DGS ellipsoids."""
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    
    # 1. Map PyVista point_data arrays to GLSL vertex attributes
    mapper.MapDataArrayToVertexAttribute("gs_scales", "gs_scales", vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)
    mapper.MapDataArrayToVertexAttribute("gs_quats", "gs_quats", vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)
    mapper.MapDataArrayToVertexAttribute("gs_opacity", "gs_opacity", vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)
    mapper.MapDataArrayToVertexAttribute("gs_sh_dc", "gs_sh_dc", vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)
    mapper.MapDataArrayToVertexAttribute("gs_sh1_y", "gs_sh1_y", vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)
    mapper.MapDataArrayToVertexAttribute("gs_sh1_z", "gs_sh1_z", vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)
    mapper.MapDataArrayToVertexAttribute("gs_sh1_x", "gs_sh1_x", vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)
    
    if hasattr(mapper, 'SetUseProgramPointSize'):
        mapper.SetUseProgramPointSize(True)
        
    actor.SetMapper(mapper)
    shader_prop = actor.GetShaderProperty()
    
    # --- VERTEX SHADER DECLARATIONS ---
    shader_prop.AddVertexShaderReplacement(
        "//VTK::PositionVC::Dec", True,
        """
        //VTK::PositionVC::Dec
        
        // Force VTK to provide the Model-to-View matrix
        uniform mat4 MCVCMatrix; 
        
        in vec3 gs_scales;
        in vec4 gs_quats;
        in float gs_opacity;
        in vec3 gs_sh_dc;
        in vec3 gs_sh1_y;
        in vec3 gs_sh1_z;
        in vec3 gs_sh1_x;
        
        out vec3 v_conic;
        out vec3 v_color;
        out float v_opacity;
        out float v_pointSize;
        """, False
    )
    
    # --- VERTEX SHADER IMPLEMENTATION (3DGS Projection Math) ---
    shader_prop.AddVertexShaderReplacement(
        "//VTK::PositionVC::Impl", True,
        """
        //VTK::PositionVC::Impl
        
        // Standard position projection
        gl_Position = MCDCMatrix * vec4(vertexMC.xyz, 1.0);
        
        // Decode Quaternion (PLY format is usually w, x, y, z)
        float w = gs_quats.x;
        float x = gs_quats.y;
        float y = gs_quats.z;
        float z = gs_quats.w;
        
        // Compute Rotation Matrix from Quaternion (Column-major construction)
        mat3 R = mat3(
            1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z), 2.0*(x*z - w*y),
            2.0*(x*y - w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x),
            2.0*(x*z + w*y), 2.0*(y*z - w*x), 1.0 - 2.0*(x*x + y*y)
        );
        
        // Compute Scale Matrix
        mat3 S = mat3(
            gs_scales.x, 0.0, 0.0,
            0.0, gs_scales.y, 0.0,
            0.0, 0.0, gs_scales.z
        );
        
        // Calculate 3D Covariance: Sigma = R * S * S^T * R^T
        mat3 M = R * S;
        mat3 Sigma = M * transpose(M);
        
        // Estimate the camera depth to approximate perspective scaling
        vec4 pos_cam = MCVCMatrix * vec4(vertexMC.xyz, 1.0);
        float depth = max(-pos_cam.z, 0.0001); // In VTK camera space, viewing down -Z
        float focal_approx = 800.0; // Simulated focal length for projection 
        
        // Project to 2D Screen Covariance
        float cov2d_xx = (Sigma[0][0]) * (focal_approx * focal_approx) / (depth * depth);
        float cov2d_yy = (Sigma[1][1]) * (focal_approx * focal_approx) / (depth * depth);
        float cov2d_xy = (Sigma[0][1]) * (focal_approx * focal_approx) / (depth * depth);
        
        // Apply low-pass filter (adds 0.3 to variance) to prevent aliasing
        cov2d_xx += 0.3;
        cov2d_yy += 0.3;
        
        // Invert the 2D covariance to get conic parameters
        float det = (cov2d_xx * cov2d_yy - cov2d_xy * cov2d_xy);
        float inv_det = 1.0 / max(det, 0.0000001);
        v_conic = vec3(cov2d_yy * inv_det, -cov2d_xy * inv_det, cov2d_xx * inv_det);
        
        // Calculate Point Size to cover 3 standard deviations (eigenvalues of 2D cov)
        float mid = 0.5 * (cov2d_xx + cov2d_yy);
        float radius = length(vec2((cov2d_xx - cov2d_yy) * 0.5, cov2d_xy));
        float lambda1 = mid + radius; // Max eigenvalue
        float max_radius = ceil(3.0 * sqrt(lambda1));
        
        gl_PointSize = clamp(max_radius * 2.0, 2.0, 1024.0);
        v_pointSize = gl_PointSize; // Pass to fragment shader
        
        // Evaluate Degree 1 Spherical Harmonics for view-dependent lighting
        vec3 camPosMC = (inverse(MCVCMatrix) * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
        vec3 dir = normalize(vertexMC.xyz - camPosMC);

        float SH_C0 = 0.28209479;
        float SH_C1 = 0.48860251;

        vec3 color = gs_sh_dc * SH_C0;
        color = color - (SH_C1 * dir.y * gs_sh1_y)
                      + (SH_C1 * dir.z * gs_sh1_z)
                      - (SH_C1 * dir.x * gs_sh1_x);

        v_color = clamp(color + 0.5, 0.0, 1.0);
        v_opacity = gs_opacity;
        """, False
    )
    
    # --- FRAGMENT SHADER DECLARATIONS ---
    shader_prop.AddFragmentShaderReplacement(
        "//VTK::Color::Dec", True,
        """
        //VTK::Color::Dec
        in vec3 v_conic;
        in vec3 v_color;
        in float v_opacity;
        in float v_pointSize;
        """, False
    )
    
    # --- FRAGMENT SHADER IMPLEMENTATION ---
    shader_prop.AddFragmentShaderReplacement(
        "//VTK::Color::Impl", True,
        """
        //VTK::Color::Impl
        
        // Map the gl_PointCoord (0 to 1) bounding box back to exact physical pixel displacement
        vec2 d = (gl_PointCoord - 0.5) * v_pointSize; 
        
        // Evaluate the Gaussian falloff in pixel space using the inverse 2D covariance (conic)
        float power = -0.5 * (v_conic.x * d.x * d.x + v_conic.z * d.y * d.y) - v_conic.y * d.x * d.y;
        
        // Optimization: if displacement causes a positive power, drop it.
        if (power > 0.0) discard;
        
        float final_alpha = v_opacity * exp(power);
        
        // This is what functionally "clips" the square bounding box into an ellipse
        if (final_alpha < 1.0 / 255.0) discard;
        
        ambientColor = v_color;
        diffuseColor = v_color;
        opacity = final_alpha;
        """, False
    )


def sort_splats_by_depth(plotter, mesh):
    """Sorts the VTK points back-to-front based on the camera view vector."""
    cam = plotter.camera

    cam_pos = np.array(cam.position)
    focal_pt = np.array(cam.focal_point)
    view_dir = focal_pt - cam_pos

    depths = np.dot(mesh.points, view_dir)
    sorted_indices = np.argsort(depths)[::-1]

    n_points = len(sorted_indices)
    verts = np.empty((n_points, 2), dtype=np.int64)
    verts[:, 0] = 1
    verts[:, 1] = sorted_indices

    mesh.verts = verts.ravel()


def main():
    # Load data
    asset_path = "splat.ply" # Update this path to your PLY file
    splat_mesh = load_3dgs_as_polydata(asset_path) # Update path!
    
    plotter = pv.Plotter()
    plotter.set_background("#13161f")
    
    # 1. Add as points
    splat_actor = plotter.add_mesh(
        splat_mesh, 
        style='points', 
        render_points_as_spheres=False,
        lighting=False,
        show_scalar_bar=False,
        rgb=True
    )
    
    splat_actor.GetProperty().SetOpacity(0.99)
    
    # 2. Inject 3DGS Math (Pass the mesh as the second argument!)
    apply_3dgs_shaders(splat_actor, splat_mesh)
    
    # 3. Add a standard PyVista object to prove they exist in the same space
    box = pv.Box(bounds=splat_mesh.bounds)
    plotter.add_mesh(box, color="cyan", style="wireframe", line_width=2)

    def on_camera_stop_move(caller, event):
        sort_splats_by_depth(plotter, splat_mesh)
        plotter.render()

    plotter.iren.add_observer("EndInteractionEvent", on_camera_stop_move)
    sort_splats_by_depth(plotter, splat_mesh)
    
    plotter.show()

if __name__ == '__main__':
    main()