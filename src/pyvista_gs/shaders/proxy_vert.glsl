#version 430 core

layout(location = 0) in vec4 vertexMC;

uniform mat4 view_matrix;
uniform mat4 projection_matrix;

void main()
{
    gl_Position = projection_matrix * view_matrix * vertexMC;
}
