#version 450

// Depth-only vertex shader for the GPU-driven shadow pass. Mirrors
// main_bindless.vert: the object id rides gl_InstanceIndex (the cull kernel
// wrote it into first_instance), indexing the per-frame GpuObjectData SSBO for
// the model matrix, so the CPU never pushes a per-draw model. The cascade index
// is a push constant (one indirect draw per cascade), selecting which
// light_vps[i] to project through.

layout(location = 0) in vec3 in_pos;
// Remaining attributes declared to preserve binding locations but not used.
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_tangent;
layout(location = 3) in vec3 in_color;
layout(location = 4) in vec2 in_uv;

layout(std140, set = 0, binding = 0) uniform ShadowGlobal {
    mat4 light_vps[4];
    vec4 cascade_splits;
} sg;

// The shadow VS only reads `model`, but the full record strides objects[oid]
// correctly.
{OBJECT_DATA}

layout(std430, set = 1, binding = 0) readonly buffer ObjectBlock {
    GpuObjectData objects[];
} obj_buf;

layout(push_constant) uniform CascadePush {
    uint cascade_idx;
} push;

void main() {
    mat4 model = obj_buf.objects[uint(gl_InstanceIndex)].model;
    gl_Position = sg.light_vps[push.cascade_idx] * model * vec4(in_pos, 1.0);
}
