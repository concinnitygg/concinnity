#version 450

// World-space line pass - vertex shader. Mirrors
// src/directx/shaders/line_vert.hlsl and src/metal/shaders/line.metal. The CPU
// expanded each line into a camera-facing ribbon whose corners already sit off
// the line centre (gfx::lines::build_vertices), so this stage only applies the
// camera VP.

layout(location = 0) in vec3  in_pos;
layout(location = 1) in float in_edge;
layout(location = 2) in vec4  in_color;

layout(std140, set = 0, binding = 0) uniform LineViewBlock {
    mat4  vp;
    float occluded_alpha;
    float _pad0;
    float _pad1;
    float _pad2;
} view;

layout(location = 0) out float v_edge;
layout(location = 1) out vec4  v_color;

void main() {
    gl_Position = view.vp * vec4(in_pos, 1.0);
    v_edge      = in_edge;
    v_color     = in_color;
}
