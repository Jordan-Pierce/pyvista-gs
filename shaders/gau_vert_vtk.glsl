#version 430 core

// VTK binds position data at location 0.
// The geometry shader does all real work via SSBOs and gl_PrimitiveIDIn;
// this shader just projects the Gaussian centre so VTK's clipping stage
// doesn't incorrectly cull primitives before they reach the geometry shader.

layout(location = 0) in vec4 vertexMC;

uniform mat4 view_matrix;
uniform mat4 projection_matrix;

void main()
{
    gl_Position = projection_matrix * view_matrix * vertexMC;
}
