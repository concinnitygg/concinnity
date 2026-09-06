// The raymarched SDF volume pass: three families of proxy-cube draw over one
// authored distance field.
//
// A volume rasterises the back faces of its world-space bounding box and the
// fragment sphere-traces the field inside it. The world's `SdfVolume` supplies
// the field itself, spliced at SDF_BODY between the helpers that forward-declare
// it and the entry points that call it, so this source is only complete once a
// world is loaded.
//
// RAYMARCH_SURFACE, RAYMARCH_VOLUMETRIC and RAYMARCH_SHADOW select the family:
// an opaque surface writing colour and depth, a participating medium blended
// over the scene, and a depth-only caster marched from the light side.
//
// RAYMARCH_METAL / RAYMARCH_DXIL select the host's binding layout. These slots
// are private between this file and the encoders: an authored shader defines
// `map`, `shade` or `sampleVolume` and never names a binding, so unlike the main
// pass there is no published contract pinning them.

{RAYMARCH_TYPES}

#ifdef RAYMARCH_METAL

// Metal takes its slots from declaration order, not from `register()`: the b and
// t spaces fold onto one buffer index there, so a `register(t0)` beside a
// `register(b0)` reads as an overlap and the compile fills with E39001. A
// declaration no entry point reaches is dropped but leaves its slot behind, so
// every family below agrees on these numbers whether or not it binds them.
ConstantBuffer<RaymarchView> view_cb;
ConstantBuffer<SdfVolumeUniforms> vol_cb;
ConstantBuffer<LightUniforms> lights_cb;
ConstantBuffer<ShadowUniforms> shadow_cb;
ConstantBuffer<RaymarchShadowCascade> cascade_cb;

Texture2DMS<float> main_depth;
Texture2DArray<float> shadow_map;
TextureCube<float4> irradiance_cube;
TextureCube<float4> prefilter_cube;
Texture2D<float4> scene_color;

SamplerComparisonState shadow_samp;
SamplerState cube_samp;
SamplerState scene_samp;

float shadow_map_cmp(float3 uv_layer, float ref)
{
    return shadow_map.SampleCmpLevelZero(shadow_samp, uv_layer, ref);
}
float2 shadow_map_size()
{
    uint w, h, e;
    shadow_map.GetDimensions(w, h, e);
    return float2(float(w), float(h));
}
float3 irradiance_sample(float3 d) { return irradiance_cube.Sample(cube_samp, d).rgb; }
float3 prefilter_sample_lod(float3 d, float lod)
{
    return prefilter_cube.SampleLevel(cube_samp, d, lod).rgb;
}
float3 scene_sample(float2 uv) { return scene_color.SampleLevel(scene_samp, uv, 0.0).rgb; }

// Metal is the one host that binds the main pass's depth, so it is the one that
// can clip the march to the rasterised surface instead of paying for a march
// the depth test would discard. The other two return a distance past any
// `max_distance`, which makes the clip a no-op there.
#define RAYMARCH_READS_SCENE_DEPTH 1

// Slang lowers `Texture2DMS.Load` to an MSL `read` taking `int2`, which the
// Metal compiler rejects because it wants `uint2` (upstream bug 1). The
// mis-lowering only reaches a metallib, so emitting MSL text hides it; this is
// the one construct in the pass that has to be spelled per target.
float ms_depth_read(Texture2DMS<float> t, uint2 px)
{
    __target_switch
    {
    case metal: __intrinsic_asm "($0).read($1, 0).x";
    default: return t.Load(int2(px), 0);
    }
}

float rasterised_distance(float2 px, float3 cam, float4 sv_pos)
{
    float depth_ndc = ms_depth_read(main_depth, uint2(px));
    float2 ndc_xy = (sv_pos.xy / view_cb.viewport) * 2.0 - 1.0;
    // Metal clip space is y-down after the projection flip the engine applies,
    // so re-mirror Y to match the inv_vp the CPU built from the unflipped one.
    ndc_xy.y = -ndc_xy.y;
    float4 world = mul(view_cb.inv_vp, float4(ndc_xy, depth_ndc, 1.0));
    world /= max(world.w, 1e-6);
    return length(world.xyz - cam);
}

#elif defined(RAYMARCH_DXIL)

// Root parameters 0..3 are CBVs as root descriptors, 4 a four-SRV table, 5 a
// three-sampler table; the shadow family's signature is 0..3 plus b4 as 32-bit
// root constants. A `register()` without a `[[vk::binding]]` warns (E39029), so
// every declaration carries both even though only one is read per target.
[[vk::binding(0, 0)]] ConstantBuffer<RaymarchView> view_cb : register(b0);
[[vk::binding(0, 1)]] ConstantBuffer<SdfVolumeUniforms> vol_cb : register(b1);
[[vk::binding(1, 0)]] ConstantBuffer<LightUniforms> lights_cb : register(b2);
[[vk::binding(2, 0)]] ConstantBuffer<ShadowUniforms> shadow_cb : register(b3);
[[vk::binding(7, 0)]] ConstantBuffer<RaymarchShadowCascade> cascade_cb : register(b4);

[[vk::binding(3, 0)]] Texture2DArray<float> shadow_map : register(t0);
[[vk::binding(4, 0)]] TextureCube<float4> irradiance_cube : register(t1);
[[vk::binding(5, 0)]] TextureCube<float4> prefilter_cube : register(t2);
[[vk::binding(6, 0)]] Texture2D<float4> scene_color : register(t3);

[[vk::binding(8, 0)]] SamplerComparisonState shadow_samp : register(s0);
[[vk::binding(9, 0)]] SamplerState cube_samp : register(s1);
[[vk::binding(10, 0)]] SamplerState scene_samp : register(s2);

float shadow_map_cmp(float3 uv_layer, float ref)
{
    return shadow_map.SampleCmpLevelZero(shadow_samp, uv_layer, ref);
}
float2 shadow_map_size()
{
    uint w, h, e;
    shadow_map.GetDimensions(w, h, e);
    return float2(float(w), float(h));
}
float3 irradiance_sample(float3 d) { return irradiance_cube.Sample(cube_samp, d).rgb; }
float3 prefilter_sample_lod(float3 d, float lod)
{
    return prefilter_cube.SampleLevel(cube_samp, d, lod).rgb;
}
float3 scene_sample(float2 uv) { return scene_color.SampleLevel(scene_samp, uv, 0.0).rgb; }

// No scene depth is bound here: the hardware depth test against the writable
// DSV is what composites this pass against rasterised geometry.
float rasterised_distance(float2 px, float3 cam, float4 sv_pos) { return 1e30; }

#else

// Set 0 is the scene-wide raymarch set, set 1 the per-volume block, and the
// cascade index rides a push constant. The cubes and the shadow array are
// combined image samplers, which is what this host allocates.
[[vk::binding(0, 0)]] ConstantBuffer<RaymarchView> view_cb;
[[vk::binding(0, 1)]] ConstantBuffer<SdfVolumeUniforms> vol_cb;
[[vk::binding(1, 0)]] ConstantBuffer<LightUniforms> lights_cb;
[[vk::binding(2, 0)]] ConstantBuffer<ShadowUniforms> shadow_cb;
[[vk::push_constant]] ConstantBuffer<RaymarchShadowCascade> cascade_cb;

[[vk::binding(3, 0)]] Sampler2DArrayShadow shadow_map;
[[vk::binding(4, 0)]] SamplerCube<float4> irradiance_cube;
[[vk::binding(5, 0)]] SamplerCube<float4> prefilter_cube;
[[vk::binding(6, 0)]] Sampler2D<float4> scene_color;

float shadow_map_cmp(float3 uv_layer, float ref)
{
    return shadow_map.SampleCmp(uv_layer, ref);
}
float2 shadow_map_size()
{
    uint w, h, e;
    shadow_map.GetDimensions(w, h, e);
    return float2(float(w), float(h));
}
float3 irradiance_sample(float3 d) { return irradiance_cube.SampleLevel(d, 0.0).rgb; }
float3 prefilter_sample_lod(float3 d, float lod) { return prefilter_cube.SampleLevel(d, lod).rgb; }
float3 scene_sample(float2 uv) { return scene_color.SampleLevel(uv, 0.0).rgb; }

float rasterised_distance(float2 px, float3 cam, float4 sv_pos) { return 1e30; }

#endif

#define VIEW view_cb
#define VOL vol_cb
#define LIGHTS lights_cb
#define SHADOW_UNI shadow_cb

// A world direction in the environment cubemaps' own frame, so the ambient fill
// turns with the sky exactly as the main pass's does.
#define RM_SKY_DIR(d) float3(dot(VIEW.sky_rot[0].xyz, (d)), \
                             dot(VIEW.sky_rot[1].xyz, (d)), \
                             dot(VIEW.sky_rot[2].xyz, (d)))

{RAYMARCH_COMMON}

// The world's own distance field.
{SDF_BODY}

// The proxy geometry is a unit cube at +/-1 in the engine's 56-byte vertex
// layout; only position is fetched. Scaling by the volume's extent and offsetting
// by its centre lands it on the bounding box, and the encoders cull front faces
// so each pixel inside the box takes exactly one fragment whether the camera is
// outside the box or in it.
struct RaymarchVertexIn
{
    float3 pos : POSITION;
};

struct RaymarchVertexOut
{
    // A pixel shader that writes SV_DepthLessEqual without running at sample
    // frequency must declare its position input centroid; DXIL validation
    // rejects the plain one. `noperspective centroid` is the other legal
    // spelling and Slang cannot express it (bug 13), but the position is
    // already non-perspective, so `centroid` alone is what validates.
    centroid float4 sv_pos : SV_Position;
    float3 world_pos : WORLDPOS;
};

float3 proxy_world_pos(float3 pos) { return pos * VOL.extent.xyz + VOL.centre.xyz; }

#if defined(RAYMARCH_SURFACE) || defined(RAYMARCH_VOLUMETRIC)

RaymarchVertexOut raymarch_proxy(RaymarchVertexIn v)
{
    float3 wp = proxy_world_pos(v.pos);
    RaymarchVertexOut o;
    o.sv_pos = mul(VIEW.vp, float4(wp, 1.0));
    o.world_pos = wp;
    return o;
}

#endif

#ifdef RAYMARCH_SURFACE

[shader("vertex")]
RaymarchVertexOut raymarch_vertex(RaymarchVertexIn v) { return raymarch_proxy(v); }

struct RaymarchFragOut
{
    float4 color : SV_Target;
    // Writing a nearer depth keeps early-Z while letting the hit composite
    // against rasterised geometry, and feeds the raymarched surface's depth to
    // the passes downstream that sample it.
    float depth : SV_DepthLessEqual;
};

[shader("fragment")]
RaymarchFragOut raymarch_fragment(RaymarchVertexOut input)
{
    float3 cam = VIEW.cam_pos.xyz;
    float3 ray_dir = normalize(input.world_pos - cam);

    // The proxy's back faces rasterised, so `world_pos` is on the far side of
    // the box. The slab test gives entry and exit both, and handles a camera
    // inside the box uniformly.
    float3 box_min = VOL.centre.xyz - VOL.extent.xyz;
    float3 box_max = VOL.centre.xyz + VOL.extent.xyz;
    float2 box_t = rayBox(cam, ray_dir, box_min, box_max);
    if (box_t.y < max(box_t.x, 0.0)) discard;
    float t_enter = max(box_t.x, 0.001);

    float t_raster = rasterised_distance(input.sv_pos.xy, cam, input.sv_pos);
    float t_max = min(box_t.y, min(t_raster, VOL.max_distance));
    if (t_enter >= t_max) discard;

    RayHit hit = coneRaymarch(cam, ray_dir, t_enter, t_max, VIEW.time);
    if (!hit.hit) discard;

    float3 hit_pos = cam + ray_dir * hit.t;
    float3 normal = sdfNormal(hit_pos, VOL.params, VIEW.time, 0.001);
    float2 frag_uv = input.sv_pos.xy / VIEW.viewport;
    SdfSurface surf = shade(hit_pos, normal, VOL.params, VIEW.time, frag_uv);

    float3 view_dir = -ray_dir;
    float3 color = shadeAmbientIbl(surf, normal, view_dir);
    if (LIGHTS.num_dir > 0)
    {
        float shadow_factor = 1.0;
        if (VOL.receive_shadows != 0)
        {
            // `hit.t` is distance along the view ray, close enough to view-space
            // depth for cascade selection without carrying the view matrix.
            shadow_factor = sampleSunShadow(hit_pos, hit.t, input.sv_pos.xy);
        }
        color += shadePbrSun(surf, normal, view_dir, LIGHTS.dir[0], shadow_factor);
    }
    // Whatever the authored shader wants to show through, already attenuated.
    color += surf.transmitted;

    // Reprojecting through the same matrix the proxy rasterised with puts the
    // hit in the rasterised geometry's depth space exactly.
    float4 hit_clip = mul(VIEW.vp, float4(hit_pos, 1.0));
    RaymarchFragOut o;
    o.color = float4(color, 1.0);
    o.depth = hit_clip.z / max(hit_clip.w, 1e-6);
    return o;
}

#endif

#ifdef RAYMARCH_VOLUMETRIC

[shader("vertex")]
RaymarchVertexOut raymarch_volumetric_vertex(RaymarchVertexIn v) { return raymarch_proxy(v); }

// Linear march from box entry to exit accumulating Beer-Lambert transmittance
// front to back, adding in-scattered sun light and emission per slab. The result
// is alpha-blended over the scene and writes no depth.
//
// The medium is not self-shadowed and the march is not clamped to scene depth,
// so a volume is sized not to intersect geometry it should render behind.
[shader("fragment")]
float4 raymarch_volumetric_fragment(RaymarchVertexOut input) : SV_Target
{
    float3 cam = VIEW.cam_pos.xyz;
    float3 ray_dir = normalize(input.world_pos - cam);

    float3 box_min = VOL.centre.xyz - VOL.extent.xyz;
    float3 box_max = VOL.centre.xyz + VOL.extent.xyz;
    float2 box_t = rayBox(cam, ray_dir, box_min, box_max);
    if (box_t.y < max(box_t.x, 0.0)) discard;
    float t_enter = max(box_t.x, 0.001);
    float t_exit = min(box_t.y, VOL.max_distance);
    if (t_enter >= t_exit) discard;

    uint step_count = uint(max(VOL.max_steps, 1));
    float step_size = (t_exit - t_enter) / float(step_count);

    float3 sun_radiance = float3(0.0, 0.0, 0.0);
    if (LIGHTS.num_dir > 0)
    {
        sun_radiance = LIGHTS.dir[0].col.xyz * LIGHTS.dir[0].dir_i.w;
    }

    float transmittance = 1.0;
    float3 luminance = float3(0.0, 0.0, 0.0);
    for (uint i = 0u; i < step_count; ++i)
    {
        float t = t_enter + (float(i) + 0.5) * step_size;
        VolumeSample vs = sampleVolume(cam + ray_dir * t, VOL.params, VIEW.time);
        if (vs.density <= 0.0) continue;

        float step_T = exp(-vs.density * step_size);
        // Energy in-scattered inside this slab is (1 - step_T) times the
        // radiance: single-scatter from the sun, plus self-emission.
        float3 step_radiance = sun_radiance * vs.scattering + vs.emission;
        luminance += transmittance * step_radiance * (1.0 - step_T);
        transmittance *= step_T;
        if (transmittance < 0.005) break;
    }

    float alpha = 1.0 - transmittance;
    if (alpha < 0.005) discard;
    return float4(luminance, alpha);
}

#endif

#ifdef RAYMARCH_SHADOW

// The caster draws once per cascade into that cascade's slice, projected through
// the light rather than the camera. The slice's depth test composites it with
// the rasterised casters already there: the nearer occluder wins per texel.
[shader("vertex")]
RaymarchVertexOut raymarch_shadow_vertex(RaymarchVertexIn v)
{
    float3 wp = proxy_world_pos(v.pos);
    RaymarchVertexOut o;
    o.sv_pos = mul(SHADOW_UNI.light_vps[cascade_cb.cascade_idx], float4(wp, 1.0));
    o.world_pos = wp;
    return o;
}

[shader("fragment")]
float raymarch_shadow_fragment(RaymarchVertexOut input) : SV_DepthLessEqual
{
    // `dir_i.xyz` is L, surface to light, which is what `shadePbrSun` reads from
    // the same field; incoming light travels along -L, so the shadow ray does.
    float3 ray_dir = -normalize(LIGHTS.dir[0].dir_i.xyz);

    // `world_pos` is on the box face farthest from the light, the encoder having
    // culled front faces. Stepping back by the bounding-sphere diameter lets the
    // slab test pick up the true front-face entry from outside the box.
    float3 origin = input.world_pos - ray_dir * (length(VOL.extent.xyz) * 2.5);
    float3 box_min = VOL.centre.xyz - VOL.extent.xyz;
    float3 box_max = VOL.centre.xyz + VOL.extent.xyz;
    float2 box_t = rayBox(origin, ray_dir, box_min, box_max);
    if (box_t.y < max(box_t.x, 0.0)) discard;
    float t_enter = max(box_t.x, 0.001);
    float t_max = min(box_t.y, VOL.max_distance);
    if (t_enter >= t_max) discard;

    RayHit hit = coneRaymarch(origin, ray_dir, t_enter, t_max, VIEW.time);
    if (!hit.hit) discard;

    // Reprojecting through the same cascade matrix the vertex rasterised with
    // shares the rasterised casters' depth space in this slice. The march is
    // bounded by the box exit, so the hit is never behind the back face and the
    // SV_DepthLessEqual contract holds.
    float3 hit_pos = origin + ray_dir * hit.t;
    float4 hit_clip = mul(SHADOW_UNI.light_vps[cascade_cb.cascade_idx], float4(hit_pos, 1.0));
    return hit_clip.z / max(hit_clip.w, 1e-6);
}

#endif
