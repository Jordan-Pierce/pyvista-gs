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
    
    # Center the mesh (Optional, but recommended for viewing)
    centroid = np.mean(mesh.points, axis=0)
    mesh.points -= centroid
    
    return mesh

def apply_3dgs_shaders(actor, mesh):
    """Modifies the VTK Mapper and Shader Property to render 3DGS."""
    
    # --- NEW: Swap PyVista's mapper for a native VTK OpenGL mapper ---
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    
    # 1. Map PyVista point_data arrays to GLSL vertex attributes!
    mapper.MapDataArrayToVertexAttribute("gs_scales", "gs_scales", vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)
    mapper.MapDataArrayToVertexAttribute("gs_quats", "gs_quats", vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)
    mapper.MapDataArrayToVertexAttribute("gs_opacity", "gs_opacity", vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)
    mapper.MapDataArrayToVertexAttribute("gs_sh_dc", "gs_sh_dc", vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)
    
    # Tell VTK that the Vertex Shader is allowed to change the point size
    if hasattr(mapper, 'SetUseProgramPointSize'):
        mapper.SetUseProgramPointSize(True)
        
    # Assign our newly configured native mapper back to the PyVista actor
    actor.SetMapper(mapper)
    # -----------------------------------------------------------------
    
    shader_prop = actor.GetShaderProperty()
    
    # --- VERTEX SHADER INJECTION ---
    shader_prop.AddVertexShaderReplacement(
        "//VTK::PositionVC::Dec",
        True,
        """
        //VTK::PositionVC::Dec
        in vec3 gs_scales;
        in vec4 gs_quats;
        in float gs_opacity;
        in vec3 gs_sh_dc;
        
        out vec3 v_conic;
        out vec3 v_color;
        out float v_opacity;
        """,
        False
    )
    
    shader_prop.AddVertexShaderReplacement(
        "//VTK::PositionVC::Impl",
        True,
        """
        //VTK::PositionVC::Impl
        
        // 1. Standard position projection
        gl_Position = MCDCMatrix * vec4(vertexMC.xyz, 1.0);
        
        // 2. Simplified Splat Sizing
        float max_scale = max(gs_scales.x, max(gs_scales.y, gs_scales.z));
        
        // --- THE FIX ---
        // The 'w' component of the projection holds the perspective depth!
        float depth = gl_Position.w; 
        
        // Set point size based on scale and depth (+ 0.0001 to prevent divide-by-zero)
        gl_PointSize = clamp((max_scale * 800.0) / (depth + 0.0001), 1.0, 500.0);
        
        // Pass attributes to fragment shader (SH_C0 = 0.28209479)
        v_color = (gs_sh_dc * 0.28209479) + 0.5;
        v_opacity = gs_opacity;
        v_conic = vec3(1.0, 0.0, 1.0);
        """,
        False
    )
    
    # --- FRAGMENT SHADER INJECTION ---
    shader_prop.AddFragmentShaderReplacement(
        "//VTK::Color::Dec",
        True,
        """
        //VTK::Color::Dec
        in vec3 v_conic;
        in vec3 v_color;
        in float v_opacity;
        """,
        False
    )
    
    shader_prop.AddFragmentShaderReplacement(
        "//VTK::Color::Impl",
        True,
        """
        //VTK::Color::Impl
        
        vec2 coord = (gl_PointCoord - 0.5) * 2.0; 
        
        float power = -0.5 * (v_conic.x * coord.x * coord.x + 
                              v_conic.z * coord.y * coord.y) - 
                              v_conic.y * coord.x * coord.y;
                              
        if (power > 0.0) discard;
        float final_alpha = min(0.99, v_opacity * exp(power));
        if (final_alpha < 1.0 / 255.0) discard;
        
        ambientColor = v_color;
        diffuseColor = v_color;
        opacity = final_alpha;
        """,
        False
    )


def main():
    # Load data
    asset_path = "coral_reef.ply" # Update this path to your PLY file
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
    
    plotter.show()

if __name__ == '__main__':
    main()