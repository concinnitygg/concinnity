// The records the raymarched SDF volume pass binds, spliced at a shader's
// RAYMARCH_TYPES marker. The first half of the raymarch splice: this declares
// what a binding names, `raymarch_common.slang` the body that reads it.
//
// Layouts must match `render/uniforms/`: RaymarchView (208 B),
// RaymarchVolumeUniforms (176 B), RaymarchShadowCascade (16 B). The light and
// cascade blocks come from `light_types.slang` because the encoders bind the
// same two buffers the forward main pass does.

{LIGHT_TYPES}

struct RaymarchView
{
    float4x4 vp;
    float4x4 inv_vp;
    // World-space camera position; `.w` is padding.
    float4 cam_pos;
    // Render-target size in pixels.
    float2 viewport;
    // Seconds since the world started, available to the authored SDF.
    float time;
    // Mip count of the bound IBL prefilter cube. 0 means no EnvironmentMap is
    // bound and the ambient helper takes its hemispheric fallback.
    float prefilter_mip_count;
    // Rows of the rotation from world space into the environment cubemaps'
    // baked frame; identity when the sky does not turn.
    float4 sky_rot[3];
};

// The authored parameter block, 32 floats the volume carries and the SDF
// interprets. Spelled as eight float4 lanes because a constant-buffer array
// strides 16 bytes per element on every target: a `float vals[32]` would occupy
// 512 bytes, not the 128 the CPU uploads. Read a slot with `sdf_param`.
struct SdfParams { float4 vals[8]; };

float sdf_param(SdfParams p, uint i) { return p.vals[i / 4u][i % 4u]; }

// Each (float3, pad) pair is spelled as one float4 lane: Metal sizes a float3
// at 16 bytes in a constant buffer where SPIR-V and DXIL size it at 12, so a
// literal transcription puts every field after the first pair four bytes late
// on Metal alone. `centre` and `extent` are the xyz of their lane.
struct SdfVolumeUniforms
{
    float4 centre;
    float4 extent;
    // 1 / max_gradient; the cone-step scale factor in `cone_raymarch`.
    float cone_ratio;
    float max_distance;
    int max_steps;
    int receive_shadows;
    SdfParams params;
};

// Which cascade a shadow-caster draw targets. Selects `light_vps[cascade_idx]`
// for the vertex projection and the fragment's depth reprojection alike.
// The three pads are scalars rather than a `uint3`: a three-component vector is
// 16-byte aligned, which would open a second lane and double the block.
struct RaymarchShadowCascade
{
    uint cascade_idx;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// The per-point material an authored `shade` returns. The template runs PBR on
// it so a raymarched surface reads like a rasterised one. `transmitted` is an
// additive contribution added after the PBR sun and IBL terms: opaque shaders
// leave it zero, refractive ones set it to the scene-tap colour they want to
// show through, already attenuated.
struct SdfSurface
{
    float3 albedo;
    float roughness;
    float metallic;
    float3 emissive;
    float3 transmitted;
};

// The per-point sample an authored `sampleVolume` returns. The volumetric
// template integrates these front to back:
//   density    extinction coefficient at the point (>= 0); 0 is empty
//   scattering single-scatter albedo, multiplied by sun radiance
//   emission   self-emitted radiance, added regardless of lighting
struct VolumeSample
{
    float density;
    float3 scattering;
    float3 emission;
};

struct RayHit
{
    float t;
    bool hit;
    int steps;
};

// The authored surface. A volume defines the pair its mode reaches: `map` plus
// `shade` for a surface volume, `sampleVolume` for a volumetric one. The
// declarations the mode does not reach are never called, so no stub is needed.
float map(float3 p, SdfParams params, float time);
// `frag_uv` is the pixel's [0, 1] screen UV, which a refractive shader hands to
// `sampleSceneRefracted`. No resource is passed: the scene snapshot, the cubes
// and the shadow array are all reached through helpers, so an authored shader
// names no backend resource type.
SdfSurface shade(float3 p, float3 normal, SdfParams params, float time,
                 float2 frag_uv);
VolumeSample sampleVolume(float3 p, SdfParams params, float time);
