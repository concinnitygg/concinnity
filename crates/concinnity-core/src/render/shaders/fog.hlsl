// Volumetric fog: single source for every backend, both halves in one file.
//
// Frostbite-style. Each frame `fog_froxel_kernel` populates a screen-aligned 3D
// RGBA16F volume of (scattered_rgb, 1 - T) across the view frustum, and the
// fullscreen `fog_fragment` samples it by (screen_uv, view_z) instead of
// marching per pixel. The scatter integral is the same for every pixel inside a
// froxel column, so the per-slice work (density + CSM shadow tap +
// Henyey-Greenstein phase) amortizes across many pixels -- which is also what
// buys the per-slice sun shadowing an inline ray-march could not afford at 32
// shadow taps per pixel.
//
// Z distribution is linear from z_near to z_far (= fog.max_distance). Log-Z
// would put more samples near the camera; that is a follow-up.
//
// The two halves share `FogParams` and `FogFroxelParams`, which is why they
// move as one unit: splitting them across languages would leave two
// unguarded copies of both structs. One entry per compile (FOG_FROXEL /
// FOG_FRAGMENT) so each variant declares only the resources it binds.
//
// How the backends differ, and how the source settles it:
//
//  * The depth texture is fetched by integer coordinate on all three backends
//    and never sampled, so it is declared as a plain texture and no backend
//    binds a sampler for it. The froxel volume IS filtered, so it is a
//    texture-sampler pair and the host supplies that sampler state.
//  * USE_MSAA is a HOST difference, not a target one -- Vulkan and DirectX
//    read the multisampled main depth when the HDR target is multisampled,
//    while Metal always reads the resolved single-sample copy -- so it is a
//    define rather than a per-target branch.
//  * The fullscreen vertex is the shared `fullscreen_vertex` out of
//    fullscreen.hlsl. The pass does no culling, so its winding does not matter.
//
// Every resource takes its Metal index from the number on its `register()` (see
// concinnity-shader's `metal_bindings`), so one annotation serves D3D and Metal
// alike. The destination volume is the exception: D3D gives a UAV its own `u`
// space where Metal folds it into the texture namespace beside the cascades, so
// it takes a `CN_BACKEND_DIRECTX` branch.
//
// FogParams (176 B) and FogFroxelParams (96 B) mirror gfx::render_types; their
// vec3 + scalar pairs are spelled as float4 / uint4 because MSL sizes a
// constant-buffer float3 at 16 bytes, so a literal transcription would shift
// every following field on Metal alone.

static const uint NUM_SHADOW_CASCADES = 4u;

struct FogParams
{
    // Inverse view-projection: reconstructs world position from depth.
    float4x4 inv_vp;
    // Linear-space fog tint (RGB); alpha unused.
    float4 color;
    // xyz = world-space camera position, w = pad.
    float4 cam_pos;
    // xyz = first directional light's world-space direction (toward the light),
    // w = pad.
    float4 sun_dir;
    // xyz = that light's color pre-multiplied with its intensity, w = pad.
    float4 sun_color;
    // Base density at height_reference, per world unit.
    float density;
    // Exponential height-falloff rate; 0 = homogeneous medium.
    float height_falloff;
    // World-space Y at which density equals `density`.
    float height_reference;
    float max_distance;
    // Henyey-Greenstein anisotropy in (-0.95, 0.95).
    float phase_g;
    // Ambient (sky-side) scattering, added isotropically each slab.
    float ambient;
    // Width / height of the HDR resolve in pixels.
    float2 viewport;
    float inv_max_distance;
    float _pad3a;
    float _pad3b;
    float _pad3c;
};

struct FogFroxelParams
{
    // World -> view, for view-space depth.
    float4x4 view;
    // xyz = volume extents along screen-x / screen-y / view-z, w = pad.
    uint4 froxel_dims;
    // Camera near plane, in view units.
    float z_near;
    // Far edge of the volume: FogSettings.max_distance.
    float z_far;
    float _pad0;
    float _pad1;
};

#ifdef FOG_FROXEL

struct ShadowUniforms
{
    float4x4 light_vps[NUM_SHADOW_CASCADES];
    float4 cascade_splits;
    uint active_cascades;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[[vk::binding(0, 0)]] ConstantBuffer<FogParams> fog : register(b0);
[[vk::binding(1, 0)]] ConstantBuffer<FogFroxelParams> froxel : register(b1);
[[vk::binding(2, 0)]] ConstantBuffer<ShadowUniforms> shadow_uni : register(b2);
// Cascaded shadow depth array, sampled with depth compare for the per-slab tap.
[[vk::binding(3, 0)]] Texture2DArray<float4> shadow_map : register(t0);
[[vk::binding(5, 0)]] SamplerComparisonState shadow_samp : register(s0);
// Destination volume: camera->slice integrated (scattered, 1 - T). Write-only,
// so Metal gets access::write rather than a read_write qualifier RGBA16Float
// would need read-write texture support for. The format is declared because an
// unformatted storage image writes through a SPIR-V capability the device does
// not ask for.
#ifdef CN_BACKEND_DIRECTX
[[vk::binding(4, 0)]] [[vk::image_format("rgba16f")]]
RWTexture3D<float4> fog_volume : register(u0);
#else
// Metal folds the `u` space into the texture namespace, where the cascades
// already hold index 0.
[[vk::binding(4, 0)]] [[vk::image_format("rgba16f")]]
RWTexture3D<float4> fog_volume : register(u1);
#endif

#else

[[vk::binding(0, 0)]] ConstantBuffer<FogParams> fog : register(b0);
[[vk::binding(2, 0)]] ConstantBuffer<FogFroxelParams> froxel : register(b1);
#if USE_MSAA
[[vk::binding(1, 0)]] Texture2DMS<float> scene_depth : register(t0);
#else
[[vk::binding(1, 0)]] Texture2D<float> scene_depth : register(t0);
#endif
[[vk::binding(3, 0)]] Texture3D<float4> fog_volume : register(t1);
[[vk::binding(4, 0)]] SamplerState fog_volume_samp : register(s0);

#endif

#ifdef FOG_FROXEL

// Closed-form Henyey-Greenstein phase function. `cos_theta` is the cosine of
// the angle between the view ray and the direction toward the sun; positive `g`
// gives forward scattering.
float henyey_greenstein(float cos_theta, float g)
{
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * 3.14159265358979 * pow(max(denom, 1e-5), 1.5));
}

// Cascade-aware single-sample shadow tap. No PCF: the trilinear sample at
// fragment-shader time smooths the result. Returns 1.0 (fully lit) outside every
// cascade, matching the main shader's fall-through.
float fog_shadow_factor(float3 world_pos, float view_depth)
{
    uint cascade = NUM_SHADOW_CASCADES;
    if (view_depth < shadow_uni.cascade_splits[0]) cascade = 0u;
    else if (view_depth < shadow_uni.cascade_splits[1]) cascade = 1u;
    else if (view_depth < shadow_uni.cascade_splits[2]) cascade = 2u;
    else if (view_depth < shadow_uni.cascade_splits[3]) cascade = 3u;
    if (cascade >= shadow_uni.active_cascades)
    {
        return 1.0;
    }

    float4 light_clip = mul(shadow_uni.light_vps[cascade], float4(world_pos, 1.0));
    float3 ndc = light_clip.xyz / light_clip.w;
    // Flip Y: the shadow pass rasterizes through a negative-height viewport on
    // Vulkan and has a top-left origin on Metal / DirectX, so the sampled UV
    // mirrors Y on every backend alike.
    float2 uv = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if (any(uv < 0.0) || any(uv > 1.0) || ndc.z < 0.0 || ndc.z > 1.0)
    {
        return 1.0;
    }
    float bias = 0.0015 * (1.0 + float(cascade) * 0.7);
    float3 uv_layer = float3(uv, float(cascade));
    // Explicit LOD, not an implicit one. A compute kernel has no fragment quad
    // to derive a mip from -- neighboring threads are unrelated froxel columns
    // -- and the cascade array has one mip, so level zero is the same tap.
    return shadow_map.SampleCmpLevelZero(shadow_samp, uv_layer, ndc.z - bias);
}

// World-space position at a froxel center. `z_slice` is a floating-point slab
// index; the caller offsets it by an interleaved-gradient-noise jitter.
float3 froxel_to_world(uint x, uint y, float z_slice)
{
    float2 uv = float2(float(x) + 0.5, float(y) + 0.5)
              / float2(float(froxel.froxel_dims.x), float(froxel.froxel_dims.y));
    float2 ndc_xy = float2(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));

    // Linear-Z distribution across [z_near, z_far].
    float view_z = lerp(froxel.z_near, froxel.z_far,
                        (z_slice + 0.5) / float(froxel.froxel_dims.z));

    // Un-project a far-plane direction, then walk that ray to the requested
    // view-space z. Cheaper than inverting a per-froxel matrix and correct for
    // any perspective projection.
    float4 clip_far = float4(ndc_xy, 1.0, 1.0);
    float4 world_far = mul(fog.inv_vp, clip_far);
    world_far /= world_far.w;
    float3 ray = normalize(world_far.xyz - fog.cam_pos.xyz);

    // Projection of `ray` onto the view-forward axis. `view` is world->view and
    // positive view depth is -z, so view-forward in world space is the negated
    // third ROW -- `view[2].xyz` under the row-first matrix subscript, which is
    // the (view[0][2], view[1][2], view[2][2]) the column-indexed CPU matrix
    // spells.
    float3 view_fwd = -froxel.view[2].xyz;
    float forward = max(dot(ray, view_fwd), 1e-4);
    return fog.cam_pos.xyz + ray * (view_z / forward);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void fog_froxel_kernel(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= froxel.froxel_dims.x || tid.y >= froxel.froxel_dims.y)
    {
        return;
    }

    // Interleaved gradient noise: a per-(x, y) tile offset so neighboring
    // columns sample density and shadows at slightly different Z. Trilinear
    // filtering at sample time plus TAA smear it into smooth illumination.
    float2 tile_xy = float2(float(tid.x), float(tid.y));
    float ign = frac(52.9829189 * frac(dot(tile_xy, float2(0.06711056, 0.00583715))));

    // Ray direction at the (x, y, 0) froxel. Within a column the direction is
    // approximately constant across Z (small-FOV approximation), so the phase
    // term is evaluated once.
    float3 col_world = froxel_to_world(tid.x, tid.y, 0.0);
    float3 ray_dir = normalize(col_world - fog.cam_pos.xyz);
    float cos_theta = dot(ray_dir, normalize(fog.sun_dir.xyz));
    float phase = henyey_greenstein(cos_theta, fog.phase_g);

    // Ambient is isotropic (no phase modulation) so the medium still reads in
    // shaded regions.
    float3 sun_inscatter_unshadowed = fog.sun_color.xyz * phase * fog.color.rgb;
    float3 ambient_inscatter = fog.color.rgb * fog.ambient;

    float step_len = (froxel.z_far - froxel.z_near) / float(froxel.froxel_dims.z);

    float3 accumulated = (float3)(0.0);
    float transmittance = 1.0;

    for (uint z = 0u; z < froxel.froxel_dims.z; ++z)
    {
        // Jittered slab center. The slab integral stays exact in the
        // constant-density limit because `tau` uses the full slab width; only
        // the sample point shifts.
        float z_jittered = float(z) + ign - 0.5;
        float3 pos = froxel_to_world(tid.x, tid.y, z_jittered);

        // Exponential height falloff, matching the inline ray-march path.
        float h = pos.y - fog.height_reference;
        float local_density = fog.density * exp(-max(h, -50.0) * fog.height_falloff);

        float slab_view_z = lerp(froxel.z_near, froxel.z_far,
                                 (z_jittered + 0.5) / float(froxel.froxel_dims.z));
        float shad = fog_shadow_factor(pos, slab_view_z);

        // Per-slab Beer-Lambert plus an analytic energy-conserving in-scatter.
        float tau = local_density * step_len;
        float slab_T = exp(-tau);
        float3 inscatter = sun_inscatter_unshadowed * shad + ambient_inscatter;
        accumulated += transmittance * (1.0 - slab_T) * inscatter;
        transmittance *= slab_T;

        // Each slice carries the camera->slice integral, so a sample at any
        // depth is the right value without a second accumulation pass.
        float4 stored = float4(accumulated, 1.0 - transmittance);
        fog_volume[uint3(tid.x, tid.y, z)] = stored;

        // Once the medium is almost opaque the remaining slices keep the
        // saturated value; any sample past this point reads the same pair.
        if (transmittance < 0.005)
        {
            transmittance = 0.0;
            for (uint zz = z + 1u; zz < froxel.froxel_dims.z; ++zz)
            {
                fog_volume[uint3(tid.x, tid.y, zz)] = stored;
            }
            break;
        }
    }
}

#else

// The fragment takes the shared fullscreen varying but derives its own `uv`
// from SV_Position: the varying is the texture-space map (unflipped on Vulkan,
// which rasterizes through a negative-height viewport) while both uses here --
// the depth unprojection and the froxel volume, which the kernel fills in
// screen tile order -- want the framebuffer-relative one.
[shader("pixel")]
float4 fog_fragment(
    [[vk::location(0)]] float2 vertex_uv : TEXCOORD0,
    float4 sv_pos : SV_Position) : SV_Target
{
    int2 pixel = int2(sv_pos.xy);
    if (pixel.x < 0 || pixel.y < 0 ||
        pixel.x >= int(fog.viewport.x) || pixel.y >= int(fog.viewport.y))
    {
        discard;
    }
#if USE_MSAA
    float depth = scene_depth.Load(pixel, 0);
#else
    float depth = scene_depth.Load(int3(pixel, 0));
#endif

    float2 uv = sv_pos.xy / fog.viewport;
    float2 ndc_xy = float2(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));

    // Reconstruct view-space depth at the pixel. depth == 1.0 (skybox, or never
    // written) maps to the far edge of the volume, so the sky takes fog
    // integrated across the whole volume.
    float view_z;
    if (depth < 1.0)
    {
        float4 world = mul(fog.inv_vp, float4(ndc_xy, depth, 1.0));
        world /= world.w;
        view_z = -mul(froxel.view, float4(world.xyz, 1.0)).z;
    }
    else
    {
        view_z = froxel.z_far;
    }

    // Normalized volume W. Clamped so the skybox and anything past the volume's
    // far edge sample the fully-integrated last slice.
    float z01 = saturate((view_z - froxel.z_near) / max(froxel.z_far - froxel.z_near, 1e-4));

    // The volume already stores camera->slice integrated (scattered, 1 - T), so
    // the trilinear sample IS the output blend pair.
    return fog_volume.SampleLevel(fog_volume_samp, float3(uv, z01), 0);
}

#endif
