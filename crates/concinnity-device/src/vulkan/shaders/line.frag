#version 450

// World-space line pass - fragment shader. Mirrors
// src/directx/shaders/line_frag.hlsl and src/metal/shaders/line.metal.
//
// Alpha does the two things a line needs: `edge` fades the outer pixel of the
// ribbon so the line antialiases without MSAA, and the vertex colour's own
// alpha carries the CPU-side distance fade.
//
// `USE_MSAA` is injected by the host (1 when the main pass uses MSAA, 0
// otherwise) so the depth sampler type matches the underlying resource.

layout(std140, set = 0, binding = 0) uniform LineViewBlock {
    mat4  vp;
    // Alpha multiplier for the part of a line behind scene geometry. 0 hides
    // occluded lines outright; a small value leaves a hint of the line showing
    // through, which is what makes it useful for orienting in a dense scene.
    float occluded_alpha;
    float _pad0;
    float _pad1;
    float _pad2;
} view;

#if USE_MSAA
layout(set = 0, binding = 1) uniform sampler2DMS scene_depth;
#else
layout(set = 0, binding = 1) uniform sampler2D scene_depth;
#endif

layout(location = 0) in float v_edge;
layout(location = 1) in vec4  v_color;

layout(location = 0) out vec4 out_color;

void main() {
    // Ribbon edge fade: `edge` runs -1..1 across the width, so one pixel of
    // its own screen-space derivative is the softest edge that still reads as
    // a solid line.
    float aa = max(fwidth(v_edge), 1e-4);
    float coverage = 1.0 - smoothstep(1.0 - aa, 1.0, abs(v_edge));

    // Manual depth test against the scene depth: this pass runs after the main
    // pass and has no depth attachment of its own. The bias keeps a line drawn
    // exactly on a surface (a ground-plane axis) from z-fighting itself into a
    // dashed mess. The pass uses the same negative-height viewport as the main
    // pass, so gl_FragCoord addresses the depth texel under this pixel.
    float scene_z = texelFetch(scene_depth, ivec2(gl_FragCoord.xy), 0).r;
    float occlusion = (gl_FragCoord.z > scene_z + 1e-6) ? view.occluded_alpha : 1.0;

    float alpha = v_color.a * coverage * occlusion;
    if (alpha <= 0.002) {
        discard;
    }
    out_color = vec4(v_color.rgb, alpha);
}
