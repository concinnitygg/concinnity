// Water surface pass: single source for every backend.
//
// The second producer of the engine's transparent pass (`PassId::Transparent`,
// after the SSR resolve and before TAA), alongside `glass.hlsl`. Where a glass
// pane is a flat quad the vertex stage only projects, a water surface is a
// tessellated XZ grid moved by a sum of waves the host resolves from the
// world's wind and the surface's swell (render::water::WaterBody), longest
// first. Everything is measured from an anchor the host picks each frame: the
// surface's center for a summed body, the camera for a simulated ocean, whose
// grid follows the camera and is clamped to the surface's rectangle. A large
// body's wind sea comes from its simulated ocean's cascades
// (water_fft.hlsl) instead of the sum: the vertex stage adds their
// displacement, filtered to its grid cell, and the fragment their slopes,
// whose mip-filtered second moments carry what the pixel cannot resolve into
// the roughness (LEAN), and the whitecap foam its longest cascades keep where
// crests break. Summed:
//
//   * The long waves, down to two grid cells, displace the grid as Gerstner
//     waves; anything shorter would alias between vertices.
//   * The fragment stage takes the world-space normal from the analytic
//     derivatives of every wave at its own rest position, the short ones as a
//     plain height field, so a coarse grid shades as finely as a dense one.
//   * Below the shortest wave, procedural ripples down to about a centimeter,
//     whose strength follows the local wind: the shared gust field (wind.hlsl)
//     roughens patches that roll downwind in step with the grass, and leaves
//     glassy water in the lulls. With no wind only sub-millimeter ripples stay.
//   * Every wave and ripple fades out as its wavelength shrinks to a few
//     pixels, and the slope variance it carried folds into the GGX roughness
//     (LEAN / Toksvig), along with the share of the ripples too fine to draw
//     at all, so the far field turns into a glitter path instead of aliasing.
//
// It then composites, in order:
//
//   * Refraction - the pre-transparent scene snapshot at a normal-perturbed
//                  screen UV, so the seabed bends under the waves.
//   * Body       - the light the water column scatters back out: the sun or
//                  moon (through its cascaded shadow) and the sky's
//                  irradiance, off a reflectance blended shallow to deep by
//                  the column thickness the main depth gives, an exponential
//                  falloff over `depth_falloff`.
//   * Foam       - a lit white diffuse layer: a band where the seabed is just
//                  below the surface, and on a simulated ocean that asks for
//                  them the whitecaps on breaking crests and the streaks
//                  they leave, scaled by their strength and broken up
//                  by procedural lace. Whitecaps cover the reflection and the
//                  sun's glint beneath them.
//   * Reflection - see below, plus a GGX specular lobe for the scene's sun off
//                  the same wave normal.
//   * Fresnel    - Schlick with a water-vs-air F0 of 0.02, shaped by
//                  `fresnel_power`.
//
// One fragment entry per compile, selected by a define. Only the reflection
// differs between them, exactly as in glass.hlsl:
//
//   default                   - the sharp planar reflection when this surface
//                               has a slot (screen UV perturbed by the wave
//                               normal, so the mirror ripples), else the
//                               box-projected reflection-probe set, else the sky
//                               prefilter cube, else a hand-tuned sky gradient.
//   WATER_RT                  - that same planar reflection where the surface has
//                               a slot, else a per-pixel reflection ray against
//                               the scene acceleration structure (the shared
//                               RT_TRACE fragment) off the wave surface point, so
//                               `R` follows the waves; the miss falls back to the
//                               same probe / sky chain.
//   WATER_RT with RT_TEXTURED - the same trace, with reflected hits taking their
//                               albedo / normal / emissive maps from the
//                               bindless pool.
//
// The mirror outranks the trace for water because a water surface IS a plane:
// one mirrored scene render resolves it exactly, where a trace off the
// per-fragment wave normal is hypersensitive at grazing angles and drops to the
// probe / sky chain wherever it misses. A glass pane does the opposite (see
// glass.hlsl), which is why only this file reads `planar` on the RT path.
//
// The inputs every transparent producer shares come from TRANSPARENT_SCENE, and
// the planar reflection takes the slot glass.hlsl declares for it.
// CN_BACKEND_DIRECTX pins every register to the root signatures in
// `directx/transparent.rs`; the other targets carry one declaration with both a
// `[[vk::binding]]` and a `register()`, whose number IS the Metal index in the
// namespace the resource's kind implies. The bindless
// pool is a whole descriptor set on Metal, named by a
// `[[cn::metal_argument_buffer(n)]]`, because no per-resource register can name
// a set.

#ifndef USE_MSAA
#define USE_MSAA 0
#endif

{DEPTH_CONVENTION}

// Waves summed per surface: the wind sea plus the authored swell. Mirrors
// `WATER_MAX_WAVES` in `core::render`'s uniforms; the array length is part of
// the `WaterParams` layout, so it is a constant here rather than a define.
static const uint WATER_MAX_WAVES = 24u;

// Ripple octaves. Mirrors `RIPPLES` in `core::render::water::ripples`; the
// array length is part of the `WaterParams` layout too.
static const uint WATER_RIPPLES = 4u;

// Cascades one simulated ocean runs at most. Mirrors `MAX_OCEAN_CASCADES` in
// `core::render::water::ocean`.
static const uint WATER_OCEAN_MAX_CASCADES = 4u;

{PROBE_TYPES}

#ifdef WATER_RT
{RT_TYPES}
#endif

{TRANSPARENT_TYPES}

{LIGHT_TYPES}

{WIND}

// One wave of the sum, packed into two float4 lanes so the CPU `[f32; 4]` pair
// is byte-identical on every target. Matches `WaterWaveGpu` (32 B). The pinch
// is the Gerstner horizontal displacement per meter of amplitude, 0 for a wave
// that only shades. The phase is the wave's at the anchor this frame,
// which the host wraps into [0, tau), so no time ever reaches the sum.
struct WaterWave
{
    float4 dir_amp_k; // [direction.x, direction.z, amplitude, wavenumber]
    float4 motion;    // [pinch, phase, _, _]
};

// One ripple octave. Matches `WaterRippleGpu` (32 B). The offset is its noise
// coordinate at the anchor this frame, in lattice cells, which the
// host wraps into one WATER_NOISE_PERIOD.
struct WaterRipple
{
    float4 shape;  // [heading.x, heading.z, wavelength, _]
    float4 offset; // [along the heading, along the crests, _, _]
};

// Where a body samples its simulated ocean. Matches `WaterOceanGpu` (64 B).
struct WaterOcean
{
    uint4 layers;        // [cascades, first layer, texels per side, whitecapping cascades]
    float4 inv_lengths;  // 1 / each cascade's side, per meter
    float4 offsets[2];   // the anchor's place in each cascade's tile, [u, v] pairs
};

// Per-surface tunables, refreshed every frame. Layout matches `WaterParams`
// (1136 B); the vec3 fields are float4 so MSL's 16-byte constant-buffer float3
// cannot desynchronize them from the CPU struct.
struct WaterParams
{
    float4 anchor;        // world-space point everything is measured from
    float4 deep_color;    // body reflectance at full column depth, [0, 1]
    float4 shallow_color; // body reflectance at the shore, [0, 1]
    float depth_falloff;
    float foam_width;
    float foam_intensity;
    float fresnel_power;
    float roughness;
    float refraction_strength;
    uint wave_count;
    // The first `displace_count` waves move the grid; the rest shade it.
    uint displace_count;
    // The wind, as wind.hlsl unpacks it: [direction.x, direction.z, strength,
    // gustiness] and [1 / gust scale, wind response, whitecaps, _].
    float4 wind;
    float4 wind_gust;
    WaterWave waves[WATER_MAX_WAVES];
    // [strength, distortion, _, _]. `strength > 0.5` selects the sharp planar
    // reflection over the trace / probe / sky cube; `distortion` scales the
    // wave-normal perturbation of its screen-UV lookup. Set on both paths: the
    // host raises it whenever this surface holds a mirror slot and the planar
    // pass ran, and the RT entry honors it too.
    float4 planar;
    WaterRipple ripples[WATER_RIPPLES];
    // [x, z, _, _]: the ripples' domain warp coordinate at the anchor this
    // frame, in its noise cells, wrapped like the ripple offsets.
    float4 ripple_warp;
    // [min x, min z, max x, max z]: the surface relative to the anchor.
    float4 rest_bounds;
    // [inner cell, inner half-width, widening, _]: the grid's local cell size
    // is `inner + max(|x| - half, 0) * widening` along each axis.
    float4 grid;
    WaterOcean ocean;
};

// ---- Resource bindings ----

#define TRANSPARENT_PARAMS WaterParams
{TRANSPARENT_SCENE}

// The simulated oceans' cascades, one layer each: (dx, dy, dz, foam) and
// (sx, sz, sx^2, sz^2), with mips, read through a wrapping anisotropic
// sampler. Bound for every surface; a summed body has no cascades and never
// samples them.
#ifdef CN_BACKEND_DIRECTX
Texture2DArray<float4> ocean_displacement : register(t15);
Texture2DArray<float4> ocean_slopes : register(t16);
SamplerState ocean_sampler : register(s4);
#else
[[vk::binding(3, 1)]] Texture2DArray<float4> ocean_displacement : register(t9);
[[vk::binding(4, 1)]] Texture2DArray<float4> ocean_slopes : register(t10);
[[vk::binding(5, 1)]] SamplerState ocean_sampler : register(s6);
#endif

// This surface's planar reflection target, bound per surface and sampled at the
// fragment's screen UV when `planar.x > 0.5`; a surface with no planar slot
// binds a valid stand-in and never samples it. Read by both entries: the RT one
// takes the mirror over its own trace wherever a slot exists.
#ifdef CN_BACKEND_DIRECTX
Texture2D<float4> planar_reflection : register(t3);
#define planar_reflection_sampler post_samp
#else
[[vk::binding(1, 1)]] Texture2D<float4> planar_reflection : register(t3);
[[vk::binding(2, 1)]] SamplerState planar_reflection_sampler : register(s3);
#endif

// The mirror is drawn at the frame's pre-exposure; this returns it absolute.
float3 planar_sample(float2 uv)
{
    return planar_reflection.Sample(planar_reflection_sampler, uv).rgb / pre_exposure();
}

// The light the body is lit by: the main pass's cascaded shadow of the first
// directional light and the sky's irradiance cube. The Vulkan bindings are the
// forward global set's own, which the pass binds as set 2.
#ifdef CN_BACKEND_DIRECTX
ConstantBuffer<ShadowUniforms> shadow_cb : register(b3);
Texture2DArray<float> shadow_map : register(t13);
TextureCube<float4> irradiance_cube : register(t14);
SamplerComparisonState shadow_sampler : register(s3);
#else
[[vk::binding(2, 2)]] ConstantBuffer<ShadowUniforms> shadow_cb : register(b14);
[[vk::binding(3, 2)]] Texture2DArray<float> shadow_map : register(t7);
[[vk::binding(4, 2)]] TextureCube<float4> irradiance_cube : register(t8);
[[vk::binding(18, 2)]] SamplerComparisonState shadow_sampler : register(s5);
#endif

#define SHADOW_UNI shadow_cb

float shadow_map_cmp(float3 uv_layer, float ref)
{
    return shadow_map.SampleCmpLevelZero(shadow_sampler, uv_layer, ref);
}
float2 shadow_map_size()
{
    uint w, h, e;
    shadow_map.GetDimensions(w, h, e);
    return float2(float(w), float(h));
}

{SHADOW_CASCADE}

{TRANSPARENT_PROBES}

#ifdef WATER_RT
{TRANSPARENT_RT}

{RT_TRACE}
#endif

// ---- Stage interface ----

struct WaterVertexIn
{
    [[vk::location(0)]] float3 pos : POSITION;
};

struct WaterVertexOut
{
    [[vk::location(0)]] float3 world_pos : TEXCOORD0;
    // The flat-rest XZ the vertex was displaced from, relative to the anchor;
    // the fragment evaluates the wave normal there itself, so the grid's
    // density sets only the silhouette of the displacement, never the
    // shading. Anchor-relative, so its precision follows the distance from
    // the surface's center, or for an ocean from the camera, rather than from
    // the world origin.
    [[vk::location(1)]] float2 local_xz : TEXCOORD1;
    float4 position : SV_Position;
};

static const float WATER_TAU = 6.28318530718;

// A wave fades out over this span of its wavelength in pixels: drawn whole
// above the second, gone below the first.
static const float2 WATER_FADE_PIXELS = float2(3.0, 6.0);

// How much of `wavelength` survives a pixel `footprint` meters across.
float water_fade(float wavelength, float footprint)
{
    return smoothstep(WATER_FADE_PIXELS.x, WATER_FADE_PIXELS.y, wavelength / max(footprint, 1e-6));
}

// Meters one pixel spans at clip-space depth `clip_w`. The view is rigid, so
// the length of the view-projection's second row is the projection's
// vertical scale.
float water_pixel_meters(float clip_w)
{
    float p11 = length(view.vp[1].xyz);
    return 2.0 * clip_w / max(p11 * view.viewport.y, 1e-6);
}

// The phase of `w` this frame at anchor-relative rest position `xz`.
float water_phase(WaterWave w, float2 xz)
{
    return w.dir_amp_k.w * dot(w.dir_amp_k.xy, xz) + w.motion.y;
}

// The grid point at anchor-relative flat-rest XZ `local_xz` displaced by the
// long waves, each a horizontal pinch plus a vertical sinusoid, faded by a
// pixel `footprint` meters across; still relative to the anchor. Reference:
// NVIDIA, "Effective Water Simulation from Physical Models", GPU Gems 1.
float3 water_displace(float2 local_xz, float footprint)
{
    float3 p = float3(local_xz.x, 0.0, local_xz.y);
    uint count = min(params.displace_count, WATER_MAX_WAVES);
    for (uint i = 0u; i < count; i++)
    {
        WaterWave w = params.waves[i];
        float amp = w.dir_amp_k.z * water_fade(WATER_TAU / w.dir_amp_k.w, footprint);
        float s, c;
        sincos(water_phase(w, local_xz), s, c);
        p.xz += w.dir_amp_k.xy * (w.motion.x * amp * c);
        p.y += amp * s;
    }
    return p;
}

// Ripples below the shortest wave: octaves of value noise (`params.ripples`,
// shaped by core::render::water::ripples) from the shortest wave down to about
// a centimeter. Noise rather than sinusoids, so no octave can line up into a
// lattice; each is squeezed across its heading so its crests run a few lengths
// long, and is carried along that heading at its own speed by the host.
//
// A crest's length over the ripple's, inverted: how much less the noise
// varies along a crest than across it. Mirrors `RIPPLE_CREST`.
static const float WATER_RIPPLE_CREST = 0.4;
// The rms gradient of the value noise per unit of its frequency, which turns a
// slope target into a height.
static const float WATER_NOISE_SLOPE = 0.5;
// The noise lattice repeats every this many cells, so the host can carry each
// offset modulo it. Mirrors `NOISE_PERIOD`.
static const float WATER_NOISE_PERIOD = 4096.0;
// The domain warp's noise frequency, per meter. Mirrors `WARP_FREQUENCY`.
static const float WATER_WARP_FREQUENCY = 2.0;
// The ripples' summed slope variance: a floor still water keeps, and a rise
// per m/s of local wind past a gentle onset (about Cox and Munk's), so a lull
// lies glassy and a gust roughens the water.
static const float WATER_STILL_RIPPLE_VARIANCE = 2.0e-4;
static const float WATER_RIPPLE_VARIANCE_PER_SPEED = 0.0045;
static const float2 WATER_RIPPLE_ONSET = float2(0.5, 2.5);
// The share of that variance in ripples finer than any drawn, which goes
// straight into the roughness.
static const float WATER_UNRESOLVED_RIPPLE = 0.35;

// wind_lattice at `cell` wrapped into the one period centered on zero: the
// same values there, repeating outside it. Exact, since every term is an
// integer or a power of two.
float water_lattice(float2 cell)
{
    float2 half_period = (float2)(0.5 * WATER_NOISE_PERIOD);
    return wind_lattice(cell - WATER_NOISE_PERIOD * floor((cell + half_period) / WATER_NOISE_PERIOD));
}

// Value noise on the periodic lattice at `p` with its gradient: `(value,
// d/dx, d/dy)`.
float3 water_noise_grad(float2 p)
{
    float2 cell = floor(p);
    float2 f = p - cell;
    float2 s = f * f * (3.0 - 2.0 * f);
    float2 ds = 6.0 * f * (1.0 - f);
    float a = water_lattice(cell);
    float b = water_lattice(cell + float2(1.0, 0.0));
    float c = water_lattice(cell + float2(0.0, 1.0));
    float d = water_lattice(cell + float2(1.0, 1.0));
    float k = a - b - c + d;
    float n = a + (b - a) * s.x + (c - a) * s.y + k * s.x * s.y;
    return float3(n, ds * float2((b - a) + k * s.y, (c - a) + k * s.x));
}

// The ripples' slope variance under a local wind of `speed` m/s.
float water_ripple_variance(float speed)
{
    return WATER_STILL_RIPPLE_VARIANCE
           + WATER_RIPPLE_VARIANCE_PER_SPEED * speed
                 * smoothstep(WATER_RIPPLE_ONSET.x, WATER_RIPPLE_ONSET.y, speed);
}

// The texture coordinate of anchor-relative `xz` in cascade `c`'s tile: one
// tile per its side, starting at the anchor's place in it, shifted half a
// texel so texel `i` sits at `i` texels from the tile's origin on every mip.
float2 water_ocean_uv(float2 xz, uint c)
{
    float4 pair = params.ocean.offsets[c >> 1u];
    float2 offset = (c & 1u) == 0u ? pair.xy : pair.zw;
    return xz * params.ocean.inv_lengths[c] + offset + 0.5 / float(params.ocean.layers.z);
}

// The ocean grid's cell size at anchor-relative `xz`, along its coarser axis.
float water_grid_cell(float2 xz)
{
    float2 cell = params.grid.x + max(abs(xz) - params.grid.y, (float2)0.0) * params.grid.z;
    return max(cell.x, cell.y);
}

// The simulated ocean's displacement at anchor-relative rest XZ `xz`, each
// cascade read at the mip whose texels span `footprint` meters.
float3 water_ocean_displace(float2 xz, float footprint)
{
    float3 d = (float3)0.0;
    uint count = min(params.ocean.layers.x, WATER_OCEAN_MAX_CASCADES);
    float size = float(params.ocean.layers.z);
    for (uint c = 0u; c < count; c++)
    {
        float texels = footprint * params.ocean.inv_lengths[c] * size;
        float lod = max(log2(max(texels, 1e-6)), 0.0);
        float layer = float(params.ocean.layers.y + c);
        d += ocean_displacement.SampleLevel(ocean_sampler, float3(water_ocean_uv(xz, c), layer), lod).xyz;
    }
    return d;
}

// Whether this body whitecaps. Every whitecap term sits behind it, so a body
// without them shades exactly as if whitecaps did not exist.
bool water_whitecaps()
{
    return params.ocean.layers.w > 0u;
}

// The shading at one rest position: the wave normal, and the GGX alpha that
// carries the slopes too fine for the pixel.
struct WaterShading
{
    float3 normal;
    float alpha;
};

// Whitecap lace: value noise over two octaves, the coarser this many meters
// a cell, the finer a quarter of it.
static const float WATER_FOAM_CELL = 0.2;
// How soft a lace edge is, in noise units.
static const float WATER_FOAM_EDGE = 0.08;

// Value noise in [0, 1] at `p` on a lattice that repeats every `period`
// cells, `p` within one period.
float water_periodic_noise(float2 p, float period)
{
    float2 cell = floor(p);
    float2 f = p - cell;
    float2 s = f * f * (3.0 - 2.0 * f);
    float2 next = cell + 1.0;
    next = select(next >= period, (float2)0.0, next);
    float a = wind_lattice(cell);
    float b = wind_lattice(float2(next.x, cell.y));
    float c = wind_lattice(float2(cell.x, next.y));
    float d = wind_lattice(next);
    return lerp(lerp(a, b, s.x), lerp(c, d, s.x), s.y);
}

// The lace that breaks up whitecap `cover` at anchor-relative rest XZ
// `local_xz`, seen through a pixel `footprint` meters across: the noise is
// cut at `1 - cover`, so dense foam stays whole and thinning foam breaks into
// ever smaller white flecks rather than dimming, while lace too fine for the
// pixel fades back to `cover` itself. Laid on the tile of the finest
// whitecapping cascade, a whole number of cells across, so it stays fixed to
// the sea as the tile offsets wrap.
float water_foam_lace(float cover, float2 local_xz, float footprint)
{
    uint c = params.ocean.layers.w - 1u;
    float2 uv = water_ocean_uv(local_xz, c);
    uv -= floor(uv);
    float period = max(round(1.0 / (params.ocean.inv_lengths[c] * WATER_FOAM_CELL)), 1.0);
    float coarse = water_periodic_noise(uv * period, period);
    float fine = water_periodic_noise(uv * (4.0 * period), 4.0 * period);
    // Two octaves of value noise, stretched toward an even spread over
    // [0, 1] so the share above the cut is about `cover`.
    float lace = saturate(0.5 + 2.5 * (0.65 * (coarse - 0.5) + 0.35 * (fine - 0.5)));
    // The stretch clips the noise at one, so foam thinner than an edge
    // fades out rather than keeping those clipped flecks.
    float cut = smoothstep(1.0 - cover - WATER_FOAM_EDGE, 1.0 - cover + WATER_FOAM_EDGE, lace)
                * saturate(cover / WATER_FOAM_EDGE);
    return lerp(cover, cut, water_fade(WATER_FOAM_CELL, footprint));
}

// Every wave and ripple at anchor-relative rest XZ `local_xz`, faded by the
// pixel's footprint (`dx`, `dy`: the rest position's screen derivatives). The
// analytic partials against (x, z) give the normal at the displaced point;
// what the fades remove is added to the base roughness as slope variance. A
// simulated ocean's cascades add their filtered slopes, and the variance
// their mips averaged away.
WaterShading water_shading(float2 local_xz, float2 dx, float2 dy)
{
    float3 binormal = float3(1.0, 0.0, 0.0); // dP/dx
    float3 tangent = float3(0.0, 0.0, 1.0);  // dP/dz
    float removed = 0.0;

    uint cascades = min(params.ocean.layers.x, WATER_OCEAN_MAX_CASCADES);
    for (uint c = 0u; c < cascades; c++)
    {
        float inv_length = params.ocean.inv_lengths[c];
        float layer = float(params.ocean.layers.y + c);
        float4 m = ocean_slopes.SampleGrad(ocean_sampler, float3(water_ocean_uv(local_xz, c), layer),
                                           dx * inv_length, dy * inv_length);
        binormal.y += m.x;
        tangent.y += m.y;
        removed += max(m.z - m.x * m.x, 0.0) + max(m.w - m.y * m.y, 0.0);
    }

    uint count = min(params.wave_count, WATER_MAX_WAVES);
    for (uint i = 0u; i < count; i++)
    {
        WaterWave w = params.waves[i];
        float2 dir = w.dir_amp_k.xy;
        float k = w.dir_amp_k.w;
        // The footprint along the wave's travel is what blurs it.
        float foot = max(abs(dot(dir, dx)), abs(dot(dir, dy)));
        float fade = water_fade(WATER_TAU / k, foot);
        float slope = k * w.dir_amp_k.z;
        removed += (1.0 - fade * fade) * 0.5 * slope * slope;

        float wa = slope * fade;
        float q = w.motion.x;
        float s, c;
        sincos(water_phase(w, local_xz), s, c);
        binormal.x += -q * dir.x * dir.x * wa * s;
        binormal.y += dir.x * wa * c;
        binormal.z += -q * dir.x * dir.y * wa * s;
        tangent.x += -q * dir.x * dir.y * wa * s;
        tangent.y += dir.y * wa * c;
        tangent.z += -q * dir.y * dir.y * wa * s;
    }

    // The ripples ride the local wind, gusts and all, at this surface's
    // response. The gust field is the shared one, read at world XZ and the
    // view's time so it rolls downwind in step with the grass. The ripples'
    // crests bend over a slowly drifting domain warp so they never line up
    // into a pattern.
    Wind wind = wind_unpack(params.wind, params.wind_gust);
    float speed = wind_speed(wind, local_xz + params.anchor.xz, view.time) * params.wind_gust.y;
    float variance = water_ripple_variance(speed);
    float per_ripple = sqrt(variance * (1.0 - WATER_UNRESOLVED_RIPPLE) / float(WATER_RIPPLES));
    float2 warp_at = local_xz * WATER_WARP_FREQUENCY + params.ripple_warp.xy;
    float2 warped = local_xz + 0.08 * float2(water_noise_grad(warp_at).x,
                                             water_noise_grad(warp_at + 17.0).x);
    for (uint r = 0u; r < WATER_RIPPLES; r++)
    {
        WaterRipple ripple = params.ripples[r];
        float2 dir = ripple.shape.xy;
        float wavelength = ripple.shape.z;
        float2 across = float2(-dir.y, dir.x);
        float fade = water_fade(wavelength, max(length(dx), length(dy)));
        removed += (1.0 - fade * fade) * per_ripple * per_ripple;

        float2 q = float2(dot(warped, dir), dot(warped, across) * WATER_RIPPLE_CREST) / wavelength
                   + ripple.offset.xy;
        float3 n = water_noise_grad(q);
        float height = per_ripple * wavelength / WATER_NOISE_SLOPE;
        float2 slope = (n.y * dir + n.z * WATER_RIPPLE_CREST * across) * (height * fade / wavelength);
        binormal.y += slope.x;
        tangent.y += slope.y;
    }

    WaterShading o;
    o.normal = normalize(cross(tangent, binormal));
    float base = max(params.roughness, 0.02);
    base *= base;
    o.alpha = sqrt(base * base + removed + variance * WATER_UNRESOLVED_RIPPLE);
    return o;
}

// The whitecap cover at anchor-relative rest XZ `local_xz`, seen through the
// pixel footprint `dx`, `dy`: the union of the whitecapping cascades' foam,
// broken up by the lace.
float water_whitecap_cover(float2 local_xz, float2 dx, float2 dy)
{
    float clear = 1.0;
    uint foaming = min(params.ocean.layers.w, min(params.ocean.layers.x, WATER_OCEAN_MAX_CASCADES));
    for (uint c = 0u; c < foaming; c++)
    {
        float inv_length = params.ocean.inv_lengths[c];
        float layer = float(params.ocean.layers.y + c);
        float3 uv = float3(water_ocean_uv(local_xz, c), layer);
        clear *= 1.0 - saturate(ocean_displacement.SampleGrad(ocean_sampler, uv, dx * inv_length,
                                                              dy * inv_length).w);
    }
    return water_foam_lace(1.0 - clear, local_xz, max(length(dx), length(dy)));
}

[shader("vertex")]
WaterVertexOut water_vertex(WaterVertexIn v)
{
    // The grid is built around the origin in the XZ plane, so the anchor
    // rides in through the params rather than the vertex buffer, and the
    // waves are summed before it is added. A summed body's grid lies inside
    // the surface already; an ocean's follows the camera and is held to it.
    float2 local_xz = clamp(v.pos.xz, params.rest_bounds.xy, params.rest_bounds.zw);
    float3 rest = float3(local_xz.x, 0.0, local_xz.y) + params.anchor.xyz;
    float rest_w = mul(view.vp, float4(rest, 1.0)).w;
    float footprint = water_pixel_meters(rest_w);
    float3 pos = water_displace(local_xz, footprint);
    if (params.ocean.layers.x > 0u)
    {
        // Nothing finer than two cells, which the grid would alias.
        pos += water_ocean_displace(local_xz, max(footprint, 2.0 * water_grid_cell(local_xz)));
    }
    pos += params.anchor.xyz;

    WaterVertexOut o;
    o.world_pos = pos;
    o.local_xz = local_xz;
    o.position = mul(view.vp, float4(pos, 1.0));
    return o;
}

// Depth stored at this pixel by the main pass, for the manual occlusion test and
// the water-column thickness.
float water_scene_depth(int2 pixel)
{
#if USE_MSAA
    return scene_depth.Load(pixel, 0);
#else
    return scene_depth.Load(int3(pixel, 0));
#endif
}

// A screen-space pixel coordinate clamped into the attachment.
int2 water_clamp_pixel(float2 pixel_xy)
{
    int2 last = int2(max(view.viewport, (float2)(1.0))) - int2(1, 1);
    return clamp(int2(pixel_xy), int2(0, 0), last);
}

// Linear camera distance to the scene point a screen NDC position and its stored
// non-linear depth describe. Gives the water column thickness when differenced
// against the distance to the surface itself. With no surface behind the water
// the column is bottomless.
float water_scene_distance(float2 ndc_xy, float depth01)
{
    float3 world;
    if (!depth_reconstruct(view.inv_vp, ndc_xy, depth01, world))
    {
        return 1e30;
    }
    return distance(world, view.camera_pos.xyz);
}

// The radiance a white Lambertian surface facing `n` reflects: the sun's
// irradiance through `shadow` plus the sky's, both over pi. Without an
// irradiance cube the sky is the main pass's flat ambient fill.
static const float WATER_INV_PI = 0.318309886;
static const float WATER_FLAT_AMBIENT = 0.03;

float3 water_lambert(float3 n, float shadow)
{
    float3 sun = view.sun_color.rgb * (saturate(dot(n, view.sun_dir.xyz)) * shadow * WATER_INV_PI);
    float3 sky = view.ambient.x > 0.5
        ? irradiance_cube.SampleLevel(prefilter_cube_sampler, SKY_DIR(n), 0.0).rgb * WATER_INV_PI
        : (float3)(WATER_FLAT_AMBIENT);
    return sun + sky * view.ambient.y;
}

// The sun's shadow at a surface point, skipped when there is no sun to cast it.
float water_sun_shadow(float3 world_pos, float2 screen_xy)
{
    if (dot(view.sun_color.rgb, (float3)(1.0)) <= 0.0)
    {
        return 1.0;
    }
    // Clip w is the view-space depth the main pass selects its cascade by.
    float view_depth = mul(view.vp, float4(world_pos, 1.0)).w;
    return shadow_factor_cascaded(world_pos, view_depth, screen_xy);
}

// The surface at this fragment: the wave normal, the fragment's screen UV, and
// everything below the waterline -- the refracted scene, the light the column
// scatters back out, and foam where the seabed is close or a crest breaks.
struct WaterSurfacePoint
{
    float3 normal;
    // GGX alpha, the base roughness widened by the slopes the pixel filtered.
    float alpha;
    // The whitecap cover, which hides the reflection and the glint.
    float foam;
    float2 frag_uv;
    float3 below;
    float camera_distance;
};

// `dx`, `dy` are the screen derivatives of `i.local_xz`, which the entry
// takes before its occlusion `discard`: a quad with a killed lane has none.
WaterSurfacePoint water_surface(WaterVertexOut i, float2 dx, float2 dy)
{
    WaterSurfacePoint s;
    WaterShading shading = water_shading(i.local_xz, dx, dy);
    s.normal = shading.normal;
    s.alpha = shading.alpha;
    s.foam = 0.0;

    float2 vp_dim = max(view.viewport, (float2)(1.0));
    s.frag_uv = i.position.xy / vp_dim;

    // Refraction: perturb the screen lookup by the wave normal's XZ so the
    // seabed bends under the waves.
    float2 refract_uv = clamp(s.frag_uv + s.normal.xz * params.refraction_strength,
                              (float2)(0.001), (float2)(0.999));
    float3 refracted = scene_sample(refract_uv).rgb;

    // Read the main depth at the REFRACTED pixel, so the thickness matches the
    // texel just sampled: a refraction that bends into a foreground edge would
    // otherwise be tinted as if that edge were underwater.
    float scene_depth01 = water_scene_depth(water_clamp_pixel(refract_uv * vp_dim));
    float2 ndc_xy = float2(s.frag_uv.x * 2.0 - 1.0, -(s.frag_uv.y * 2.0 - 1.0));
    float scene_dist = water_scene_distance(ndc_xy, scene_depth01);
    float water_dist = distance(i.world_pos, view.camera_pos.xyz);
    s.camera_distance = water_dist;
    float water_depth = max(scene_dist - water_dist, 0.0);

    // The body as a lit single-scatter layer: light entering the flat surface
    // (the sun past its entry Fresnel, and the sky) is scattered back out by
    // the column as a Lambertian reflector of the shallow-to-deep reflectance,
    // its share growing as the column thickens over `depth_falloff` meters.
    // The seabed seen through the shallows is filtered by the shallow color.
    float shadow = water_sun_shadow(i.world_pos, i.position.xy);
    float sun_entry = 1.0 - (0.02 + 0.98 * pow(1.0 - saturate(view.sun_dir.y), 5.0));
    float3 body_light = water_lambert(float3(0.0, 1.0, 0.0), shadow * sun_entry);
    float depth_t = 1.0 - exp(-water_depth / max(params.depth_falloff, 1e-3));
    float3 scattered = lerp(params.shallow_color.rgb, params.deep_color.rgb, depth_t) * body_light;
    float3 below = lerp(refracted * params.shallow_color.rgb, scattered, depth_t);

    // Foam: a white diffuse surface on the waves in a soft band where the
    // seabed is just below the surface, which is both the shoreline and any
    // intersection line with standing geometry, then the whitecaps over what
    // that leaves.
    float foam_t = saturate(1.0 - water_depth / max(params.foam_width, 1e-3));
    float3 foam = water_lambert(s.normal, shadow);
    s.below = lerp(below, foam, foam_t * foam_t * params.foam_intensity);
    if (water_whitecaps())
    {
        s.foam = saturate(water_whitecap_cover(i.local_xz, dx, dy) * params.wind_gust.z);
        s.below = lerp(s.below, foam, s.foam);
    }
    return s;
}

// The mirror render for this surface, at the fragment's own screen UV perturbed
// by the wave normal. The planar render mirrors the scene across the surface's
// rest plane, so it lands exactly under the reflector; perturbing that lookup is
// what turns a flat mirror back into rippling water.
float3 water_planar_reflection(WaterSurfacePoint s)
{
    float2 uv = clamp(s.frag_uv + s.normal.xz * params.planar.y,
                      (float2)(0.001), (float2)(0.999));
    return planar_sample(uv);
}

// The reflection a surface with no sharp source falls back to: the local
// box-projected probe set where a probe actually covers this point, else the sky
// prefilter cube, else a hand-tuned vertical sky gradient (bluer overhead, paler
// at the horizon) so an environment-less world still reads as water rather than
// as flat tinted glass.
//
// The coverage test matters more here than anywhere else: a pool routinely
// stretches past every probe box in the world, and the probe set's own
// out-of-box fallback would hand the whole surface one foreign capture.
float3 water_environment(float3 world_pos, float3 r, float alpha)
{
    float horizon = saturate(r.y * 0.5 + 0.5);
    float3 gradient = lerp(float3(0.55, 0.62, 0.7), float3(0.25, 0.45, 0.7), horizon);
    // Blurrier water reads a coarser cube level.
    return transparent_environment(world_pos, r, saturate(sqrt(alpha)), gradient);
}

// The sun's specular lobe off this fragment's wave normal at GGX `alpha`: GGX with a
// height-correlated Smith visibility term and the water-vs-air F0, driven by the
// first directional light the view block carries. A high sun draws a compact
// disc; a low one spreads the same lobe into a path running toward the camera,
// since the wave slopes that mirror it span more of the surface. Zero when the
// world declares no directional light, whose `sun_color` is zero.
float3 water_sun_glint(float3 normal, float3 view_dir, float alpha)
{
    float n_dot_l = dot(normal, view.sun_dir.xyz);
    if (n_dot_l <= 0.0)
    {
        return (float3)(0.0);
    }
    float3 h = normalize(view.sun_dir.xyz + view_dir);
    float n_dot_v = max(dot(normal, view_dir), 1e-4);
    float n_dot_h = saturate(dot(normal, h));
    float v_dot_h = saturate(dot(view_dir, h));

    float a2 = alpha * alpha;
    float denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    float d = a2 / max(3.14159265358979 * denom * denom, 1e-6);

    float lambda_v = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    float lambda_l = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    float vis = 0.5 / max(lambda_v + lambda_l, 1e-6);

    float f = 0.02 + 0.98 * pow(1.0 - v_dot_h, 5.0);
    return view.sun_color.rgb * (d * vis * f * n_dot_l);
}

// Schlick Fresnel mix of the reflection over everything below the waterline.
// F0 = 0.02 is the water-vs-air value; `fresnel_power` shapes the falloff, so a
// low power keeps the reflection strong head-on instead of only at grazing.
// The sun's glint is added after that mix: its lobe carries its own Fresnel
// term, which is what strengthens it toward grazing angles, so weighting it by
// the surface Fresnel as well would count that twice.
// Alpha 1: water fully covers what it drew over, and the pipeline's straight
// alpha blend leaves the composited color as-is. The reactive share is the
// glint's, a sharp highlight riding the waves, or a part of the Fresnel weight,
// since the reflection and the refraction move with neither the surface's
// motion nor what they show. The Fresnel part fades out with distance: far off,
// where a wave shrinks toward a pixel, the history is what keeps the surface
// from crawling with the jitter.
static const float WATER_REACTIVE_FRESNEL = 0.5;
static const float2 WATER_REACTIVE_FADE_METERS = float2(4.0, 16.0);

TransparentOut water_resolve(WaterSurfacePoint s, float3 view_dir, float3 reflection)
{
    float n_dot_v = saturate(dot(s.normal, view_dir));
    float fresnel = 0.02 + 0.98 * pow(1.0 - n_dot_v, max(params.fresnel_power, 1e-3));
    float3 glint = water_sun_glint(s.normal, view_dir, s.alpha);
    float3 color = lerp(s.below, reflection, fresnel) + glint;
    if (water_whitecaps())
    {
        // Whitecaps lie on top of the water: what they cover neither
        // mirrors nor glints.
        color = lerp(color, s.below, s.foam);
    }
    float near = 1.0 - smoothstep(WATER_REACTIVE_FADE_METERS.x, WATER_REACTIVE_FADE_METERS.y,
                                  s.camera_distance);
    float reactive = max(reactive_luminance(glint), fresnel * WATER_REACTIVE_FRESNEL * near);
    return transparent_out(float4(color, 1.0), reactive);
}

// True where nearer opaque geometry occludes the surface. The transparent pass
// binds no depth attachment, so the test is manual; every backend rasterized
// this depth under the same viewport convention the main pass used, so the
// fragment position lines up with the stored texel.
bool water_occluded(WaterVertexOut i)
{
    return depth_in_front(water_scene_depth(water_clamp_pixel(i.position.xy)), i.position.z);
}

#ifdef WATER_RT

[shader("pixel")]
TransparentOut water_rt_fragment(WaterVertexOut i)
{
    // Derivatives first: after the discard below, a quad on the shoreline may
    // have lost a lane, and its derivatives are undefined.
    float2 dx = ddx(i.local_xz);
    float2 dy = ddy(i.local_xz);
    if (water_occluded(i))
    {
        discard;
    }
    float3 view_dir = normalize(view.camera_pos.xyz - i.world_pos);
    WaterSurfacePoint s = water_surface(i, dx, dy);

    float3 reflection;
    if (params.planar.x > 0.5)
    {
        reflection = water_planar_reflection(s);
    }
    // No mirror plane for this surface, so trace instead. The per-fragment
    // wave normal makes `R` vary across the surface, so the traced
    // reflection follows the waves rather than mirroring one flat plane.
    else
    {
        float3 r = reflect(-view_dir, s.normal);
        if (!rt_trace_reflection(i.world_pos + s.normal * 0.02, r,
                                 view.prefilter_mip_count > 0.5,
                                 view.prefilter_mip_count - 1.0, reflection))
        {
            reflection = water_environment(i.world_pos, r, s.alpha);
        }
    }
    return water_resolve(s, view_dir, reflection);
}

#else

[shader("pixel")]
TransparentOut water_fragment(WaterVertexOut i)
{
    // Derivatives first: after the discard below, a quad on the shoreline may
    // have lost a lane, and its derivatives are undefined.
    float2 dx = ddx(i.local_xz);
    float2 dy = ddy(i.local_xz);
    if (water_occluded(i))
    {
        discard;
    }
    float3 view_dir = normalize(view.camera_pos.xyz - i.world_pos);
    WaterSurfacePoint s = water_surface(i, dx, dy);

    float3 reflection;
    if (params.planar.x > 0.5)
    {
        reflection = water_planar_reflection(s);
    }
    else
    {
        reflection = water_environment(i.world_pos, reflect(-view_dir, s.normal), s.alpha);
    }
    return water_resolve(s, view_dir, reflection);
}

#endif
