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
// A planet's sea compiles under WATER_SEA: the same grid, laid out in the
// meters of the cube face under the camera and bent onto the sea's sphere in
// the vertex stage, held within the horizon. Its waves are laid out on the
// planet, each cube face carrying its own copy of the sea through a
// stereographic map (render::water::sea), so they stay put when the simulated
// frame moves; where two faces meet, their seas blend across a band. Every
// wave lookup above is shared, reading the face being sampled through the
// WATER_* hooks below, which expand to the rectangle's own expressions
// otherwise. A sea also carries the wake bodies make crossing it near the
// camera (water_wake.hlsl): its height rides on the waves, held to what the
// depth lets stand, its slopes add to theirs, and its foam lies over the
// whitecaps and the surf.
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

// Cube faces a planet sea reads. Mirrors `SEA_FACES` in
// `core::render::water::sea`.
static const uint WATER_SEA_FACES = 3u;

// One cube face's view of a planet sea. Matches `WaterSeaFaceGpu` (128 B).
struct WaterSeaFace
{
    float4 rows[3];           // simulated frame -> face [u, normal, v]; w = the anchor's direction
    float4 warp;              // [x, z, _, _]: the ripples' warp offset at the anchor on this face
    float4 ocean_offsets[2];  // the anchor's place in each cascade's tile, [u, v] pairs
    float4 ripple_offsets[2]; // each ripple octave's offset, [along, across] pairs
};

// Shore train phase-table nodes. Mirrors `SHORE_NODES` in
// `core::render::water::sea::shore`.
static const uint WATER_SHORE_NODES = 32u;
// The noise octaves the shore reads: the continents', the phase noise's, then
// the swell's.
static const uint WATER_CONTINENT_OCTAVES = 6u;
static const uint WATER_SHORE_OCTAVES = 9u;

// A planet sea's shore waves. Matches `WaterShoreGpu` (1296 B); zero for a
// sea without coasts, which `continent_octaves` 0 marks.
struct WaterShore
{
    float4 planet_rows[3];       // simulated frame -> planet axes; w = the anchor's direction
    uint4 continent_salts[2];
    float continent_frequency;   // broadest octave's lattice cells per unit of direction
    float continent_scale;       // meters inland per unit of the octaves' sum
    float continent_offset;      // meters inland at a sum of 0, negated
    uint continent_octaves;
    uint4 noise_salts;           // the phase noise's octave salts, zw unused
    float4 noise;                // [cells per unit of direction, radians, 1 / planet radius, 1 / sea radius]
    float4 coast;                // [sea level, waterline, ocean depth, shelf]
    float4 beach;                // [beach width, beach rise, table reach, peak wavelength]
    float4 bounds;               // [inland seaward of which no shore wave or fade reaches, _, _, _]
    float4 swell;                // [cells per unit of direction, share either way, _, _]
    float4 lace;                 // [anchor in the planet's axes, wrapped; the period it wraps in]
    float4 omegas;               // each train's angular frequency
    float4 amplitudes;           // each train's deep-water height
    float4 phases;               // each train's phase at the waterline this frame
    float4 cascade_wavelengths;  // the wavelength each cascade fades out at; 0 keeps it
    float4 table[2u * WATER_SHORE_NODES]; // per node: phases, then wavenumbers
    int4 lattice_cells[WATER_SHORE_OCTAVES];    // per octave, the anchor's lattice cell
    float4 lattice_places[WATER_SHORE_OCTAVES]; // and where in it the anchor lies
};

// Where a planet sea's wake lies. Matches `WaterWakeGpu` (32 B); all zero
// while none runs.
struct WaterWake
{
    float4 lookup; // [1 / window width, cell size, center x, center z]
    float4 fade;   // [clear half-width, 1 / sponge width, height per meter of depth, on]
};

// Where a planet sea lies and how its grid is laid on it. Matches
// `WaterSeaGpu` (2096 B); zero for a rectangle.
struct WaterSea
{
    float4 up;          // [radial up at the anchor, sea radius]
    float4 tangent_u;   // [grid x along the sea at the anchor, meters per grid meter]
    float4 tangent_v;   // [grid z along the sea at the anchor, drawn reach in meters]
    float4 shape;       // [1 / radius, face blend, _, _]
    float4 wind_map[2]; // rows over a point that give where its gusts are read
    WaterSeaFace faces[WATER_SEA_FACES];
    WaterShore shore;
    WaterWake wake;
};

// Per-surface tunables, refreshed every frame. Layout matches `WaterParams`
// (3232 B); the vec3 fields are float4 so MSL's 16-byte constant-buffer float3
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
    WaterSea sea;
};

// The hooks the shared wave lookups read through: on a planet sea, the face
// being sampled (`water_face`) and the gusts at the shaded point; on a
// rectangle, the body's own lanes.
#ifdef WATER_SEA
static uint water_face = 0u;
static float2 water_gust_xz = (float2)0.0;
// How much of each cascade the shallows under the point being shaded leave.
static float4 water_shore_damp = (float4)1.0;
#define WATER_OCEAN_OFFSETS params.sea.faces[water_face].ocean_offsets
#define WATER_RIPPLE_WARP params.sea.faces[water_face].warp.xy
#define WATER_RIPPLE_OFFSET(r, ripple) water_sea_ripple_offset(r)
#define WATER_GUST_AT(local_xz) water_gust_xz
#define WATER_STEP float3
#define WATER_SHADE(i, dx, dy) water_sea_shading(i, dx, dy)
#define WATER_UP(i) water_sea_up(i.rest)
#define WATER_SUN_HEIGHT(i) dot(view.sun_dir.xyz, water_sea_up(i.rest))
#define WATER_WHITECAP(i, dx, dy) water_sea_cover
#define WATER_DAMPED(c, x) ((x) * water_shore_damp[c])
#define WATER_DAMPED_SLOPES(c, m) ((m) * float4(water_shore_damp[c], water_shore_damp[c], \
                                              water_shore_damp[c] * water_shore_damp[c], \
                                              water_shore_damp[c] * water_shore_damp[c]))
#else
#define WATER_OCEAN_OFFSETS params.ocean.offsets
#define WATER_RIPPLE_WARP params.ripple_warp.xy
#define WATER_RIPPLE_OFFSET(r, ripple) ripple.offset.xy
#define WATER_GUST_AT(local_xz) local_xz + params.anchor.xz
#define WATER_STEP float2
#define WATER_SHADE(i, dx, dy) water_shading(i.local_xz, dx, dy)
#define WATER_UP(i) float3(0.0, 1.0, 0.0)
#define WATER_SUN_HEIGHT(i) view.sun_dir.y
#define WATER_WHITECAP(i, dx, dy) water_whitecap_cover(i.local_xz, dx, dy)
#define WATER_DAMPED(c, x) x
#define WATER_DAMPED_SLOPES(c, m) m
#endif

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

#ifdef WATER_SEA
// The wake around the camera (water_wake.hlsl): (height, slope x, slope z,
// foam) per cell, with mips, each cell in the texel its coordinates wrap to,
// read through the oceans' wrapping sampler. Bound for every sea; zero while
// none runs.
#ifdef CN_BACKEND_DIRECTX
Texture2D<float4> wake_field : register(t17);
#else
[[vk::binding(6, 1)]] Texture2D<float4> wake_field : register(t11);
#endif
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

#ifdef WATER_SEA
struct WaterVertexOut
{
    [[vk::location(0)]] float3 world_pos : TEXCOORD0;
    // The point on the sea at rest the vertex was displaced from, relative
    // to the anchor; the fragment finds its place on each face itself.
    [[vk::location(1)]] float3 rest : TEXCOORD1;
    // The coast under the rest point (`water_shore_field`) and the shore's
    // phase noise there (`water_shore_noise`), found per vertex: both vary
    // over hundreds of meters, far wider than the grid's cells where the
    // shore waves show.
    [[vk::location(2)]] float4 shore_field : TEXCOORD2;
    [[vk::location(3)]] float4 shore_noise : TEXCOORD3;
    // How tall the swell runs there (`water_shore_swell`).
    [[vk::location(4)]] float shore_swell : TEXCOORD4;
    float4 position : SV_Position;
};

// One face's ripple octave `r` offset at the anchor, for the face being
// sampled.
float2 water_sea_ripple_offset(uint r)
{
    float4 row = params.sea.faces[water_face].ripple_offsets[r >> 1u];
    return (r & 1u) == 0u ? row.xy : row.zw;
}
#else
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
#endif

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
    float4 pair = WATER_OCEAN_OFFSETS[c >> 1u];
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
        d += WATER_DAMPED(c, ocean_displacement.SampleLevel(ocean_sampler, float3(water_ocean_uv(xz, c), layer),
                                                            lod).xyz);
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

#ifdef WATER_SEA
// How much the wake's foam calms the short waves under it, at full foam, and
// the foam that calms them fully: churned water smooths the ripples, so a
// trail reads glassy against the sea long after its foam thins.
static const float WATER_WAKE_SLICK = 0.85;
static const float WATER_WAKE_SLICK_FOAM = 0.12;

// The cascades the slick calms: all of one whose tile is this many meters
// or less, none of one this long or longer.
static const float2 WATER_WAKE_SLICK_TILES = float2(8.0, 32.0);

// How much of the short waves a sea's wake leaves standing at the fragment
// being shaded: exactly 1 away from it.
static float water_calm = 1.0;
#endif

// Every wave and ripple at anchor-relative rest XZ `local_xz`, faded by the
// pixel's footprint (`dx`, `dy`: how far the rest position moves across one
// pixel, from `water_footprint`). The analytic partials against (x, z) give
// the normal at the displaced point; what the fades remove is added to the
// base roughness as slope variance. A simulated ocean's cascades add their
// filtered slopes, and the variance their mips averaged away.
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
        float4 m = WATER_DAMPED_SLOPES(c, ocean_slopes.SampleGrad(ocean_sampler,
                                                                  float3(water_ocean_uv(local_xz, c), layer),
                                                                  dx * inv_length, dy * inv_length));
#ifdef WATER_SEA
        float calm = 1.0 - (1.0 - water_calm) * (1.0 - smoothstep(WATER_WAKE_SLICK_TILES.x,
                                                                 WATER_WAKE_SLICK_TILES.y, 1.0 / inv_length));
        m *= float4(calm, calm, calm * calm, calm * calm);
#endif
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
    float speed = wind_speed(wind, WATER_GUST_AT(local_xz), view.time) * params.wind_gust.y;
    float variance = water_ripple_variance(speed);
#ifdef WATER_SEA
    variance *= water_calm * water_calm;
#endif
    float per_ripple = sqrt(variance * (1.0 - WATER_UNRESOLVED_RIPPLE) / float(WATER_RIPPLES));
    float2 warp_at = local_xz * WATER_WARP_FREQUENCY + WATER_RIPPLE_WARP;
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
                   + WATER_RIPPLE_OFFSET(r, ripple);
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
        clear *= 1.0 - saturate(WATER_DAMPED(c, ocean_displacement.SampleGrad(ocean_sampler, uv,
                                                                              dx * inv_length,
                                                                              dy * inv_length).w));
    }
    return water_foam_lace(1.0 - clear, local_xz, max(length(dx), length(dy)));
}

#ifdef WATER_SEA

// A rest point's place on one cube face: its face meters from the anchor's
// place there, how many face meters a meter of the sea spans, and the
// directions along the sea, in the simulated frame, the face's u and v grow
// along. Mirrors `sea_point` in render::water::sea.
struct WaterSeaPoint
{
    float2 s;
    float stretch;
    float3 tu;
    float3 tv;
};

// The anchor's direction from the planet's center in face `f`'s axes.
float3 water_sea_anchor_dir(uint f)
{
    return float3(params.sea.faces[f].rows[0].w, params.sea.faces[f].rows[1].w,
                  params.sea.faces[f].rows[2].w);
}

// `rest`, an offset in the simulated frame, in face `f`'s axes.
float3 water_sea_face_offset(uint f, float3 rest)
{
    return float3(dot(params.sea.faces[f].rows[0].xyz, rest), dot(params.sea.faces[f].rows[1].xyz, rest),
                  dot(params.sea.faces[f].rows[2].xyz, rest));
}

// A direction in face `f`'s axes, in the simulated frame.
float3 water_sea_from_face(uint f, float3 v)
{
    return v.x * params.sea.faces[f].rows[0].xyz + v.y * params.sea.faces[f].rows[1].xyz
           + v.z * params.sea.faces[f].rows[2].xyz;
}

// The face map is the stereographic projection from the opposite face's
// center, `2 radius (u, v) / (1 + normal)`, differenced against the anchor's
// in a form that stays exact for points beside it.
WaterSeaPoint water_sea_point(uint f, float3 rest)
{
    float3 a = water_sea_anchor_dir(f);
    float3 dl = water_sea_face_offset(f, rest);
    float3 d = a + dl * params.sea.shape.x;
    float denom_a = 1.0 + a.y;
    float denom_p = 1.0 + d.y;
    float k = 2.0 / (denom_p * denom_a);
    WaterSeaPoint p;
    p.s = k * (dl.xz * denom_a - a.xz * dl.y);
    p.stretch = 2.0 / denom_p;
    float3 n = normalize(d);
    float3 gu = float3(denom_p, -d.x, 0.0);
    float3 gv = float3(0.0, -d.z, denom_p);
    p.tu = water_sea_from_face(f, normalize(gu - dot(gu, n) * n));
    p.tv = water_sea_from_face(f, normalize(gv - dot(gv, n) * n));
    return p;
}

// Each face's share of a rest point: whole well inside its face, blending
// with its neighbor across a band at their edge, the squares summing to one
// so two blended seas keep the roughness of one.
float3 water_sea_weights(float3 rest)
{
    float3 w;
    for (uint f = 0u; f < WATER_SEA_FACES; f++)
    {
        float3 d = water_sea_anchor_dir(f) + water_sea_face_offset(f, rest) * params.sea.shape.x;
        float margin = d.y - max(abs(d.x), abs(d.z));
        w[f] = saturate(0.5 + margin * params.sea.shape.y);
    }
    return w * rsqrt(max(dot(w, w), 1e-12));
}

// The sea's radial up at a rest point.
float3 water_sea_up(float3 rest)
{
    return normalize(rest + params.sea.up.xyz * params.sea.up.w);
}

{PLANET_NOISE}

// The shore waves (render::water::sea::shore), read in the planet's axes.
#define WATER_SHORE params.sea.shore

// Mirrors the constants of render::water::sea::shore.
static const uint WATER_SHORE_TRAINS = 3u;
static const float WATER_BREAKER_INDEX = 0.78;
static const float WATER_SHALLOWEST = 0.1;
static const float2 WATER_GRADIENT_FADE = float2(0.15, 0.3);
static const float WATER_MAX_LEAN = 0.75;
static const float WATER_MAX_PINCH = 0.9;
static const float WATER_FOAM_TRAIL = 0.45;
static const float WATER_FOAM_LEAD = 0.5;
static const float WATER_RESIDUAL_FOAM = 0.2;
static const float WATER_FOAM_RIDE = 0.8;
static const uint WATER_PHASE_NOISE_OCTAVES = 2u;
// The phase noise's octaves weigh this much together.
static const float WATER_PHASE_NOISE_WEIGHT = 1.5;
static const float WATER_GRAVITY = 9.81;

// The turn from one noise octave's lattice to the next: planet::OCTAVE_TURN,
// row by row.
static const float3x3 WATER_OCTAVE_TURN = float3x3(0.811838377983, -0.459433173493, 0.360332634004,
                                                   0.534144405765, 0.833629154062, -0.140542475031,
                                                   -0.235813913551, 0.306567435635, 0.922175799717);

// `octaves` octaves of the planet's value noise at the unit direction `away`
// from the anchor's, the broadest `frequency` cells per unit of
// direction, octave `o` salted by `salts[o / 4][o % 4]` and read from the
// anchor's lattice cell `first + o`: xyz = the sum's gradient over
// directions, w = the sum (planet::fbm). From the third octave on, a sum sure
// to end below `keep.x` stops at the most it could still reach, and one sure
// to end above `keep.y` at the least.
float4 water_shore_octaves(float3 away, uint first, float frequency, uint octaves, uint4 salts[2],
                           float2 keep)
{
    float3x3 to_lattice = WATER_OCTAVE_TURN * frequency;
    float weight = 1.0;
    float4 sum = (float4)0.0;
    for (uint o = 0u; o < octaves; o++)
    {
        if (o >= 2u)
        {
            // The octaves left add at most twice the next weight, less the
            // last's.
            float rest = 2.0 * weight - exp2(1.0 - float(octaves));
            if (sum.w + rest < keep.x)
            {
                sum.w += rest;
                break;
            }
            if (sum.w > keep.y)
            {
                break;
            }
        }
        // The lattice point as the anchor's cell and a small step from it,
        // which f32 holds however far the cell lies from the lattice's origin.
        float3 q = WATER_SHORE.lattice_places[first + o].xyz + mul(to_lattice, away);
        float3 step = floor(q);
        float4 n = ground_noise_cell(WATER_SHORE.lattice_cells[first + o].xyz + int3(step), q - step,
                                     salts[o >> 2u][o & 3u]);
        sum.xyz += mul(n.xyz, to_lattice) * weight;
        sum.w += n.w * weight;
        to_lattice = mul(WATER_OCTAVE_TURN, to_lattice) * 2.0;
        weight *= 0.5;
    }
    return sum;
}

// A vector in the planet's axes, in the simulated frame.
float3 water_shore_to_frame(float3 v)
{
    return v.x * WATER_SHORE.planet_rows[0].xyz + v.y * WATER_SHORE.planet_rows[1].xyz
           + v.z * WATER_SHORE.planet_rows[2].xyz;
}

// How deep the water is `inland` meters inland of the coast where no hills
// stand (`CoastProfile::depth`).
float water_shore_depth(float inland)
{
    float ground;
    if (inland >= 0.0)
    {
        float t = min(inland / WATER_SHORE.beach.x, 1.0);
        ground = WATER_SHORE.beach.y * t * (2.0 - t);
    }
    else
    {
        ground = -WATER_SHORE.coast.z * (1.0 - exp(inland / WATER_SHORE.coast.w));
    }
    return max(WATER_SHORE.coast.x - ground, 0.0);
}

// Every train's phase and wavenumber `offshore` meters out (`PhaseTable::at`).
void water_shore_table(float offshore, out float3 phase, out float3 k)
{
    float reach = WATER_SHORE.beach.z;
    float u = max(offshore, 0.0);
    uint last = WATER_SHORE_NODES - 1u;
    if (u >= reach)
    {
        k = WATER_SHORE.table[2u * last + 1u].xyz;
        phase = WATER_SHORE.table[2u * last].xyz + k * (u - reach);
        return;
    }
    uint j = min(uint(float(last) * sqrt(u / reach)), last - 1u);
    float fa = float(j) / float(last);
    float fb = float(j + 1u) / float(last);
    float a = reach * fa * fa;
    float span = reach * fb * fb - a;
    float s = (u - a) / span;
    float s2 = s * s;
    float s3 = s2 * s;
    float3 p0 = WATER_SHORE.table[2u * j].xyz;
    float3 k0 = WATER_SHORE.table[2u * j + 1u].xyz;
    float3 p1 = WATER_SHORE.table[2u * j + 2u].xyz;
    float3 k1 = WATER_SHORE.table[2u * j + 3u].xyz;
    phase = (2.0 * s3 - 3.0 * s2 + 1.0) * p0 + (s3 - 2.0 * s2 + s) * span * k0
          + (-2.0 * s3 + 3.0 * s2) * p1 + (s3 - s2) * span * k1;
    k = k0 + s * (k1 - k0);
}

// Where a rest point lies on the planet, in its axes: its direction from
// the center (`dir`), and that direction less the anchor's (`away`), found
// without the anchor's own rounding, so it stays precise on any planet.
void water_shore_place(float3 rest, out float3 dir, out float3 away)
{
    float3 a = float3(WATER_SHORE.planet_rows[0].w, WATER_SHORE.planet_rows[1].w,
                      WATER_SHORE.planet_rows[2].w);
    float3 l = float3(dot(WATER_SHORE.planet_rows[0].xyz, rest),
                      dot(WATER_SHORE.planet_rows[1].xyz, rest),
                      dot(WATER_SHORE.planet_rows[2].xyz, rest))
             * WATER_SHORE.noise.w;
    // (a + l) / |a + l| - a for a unit a, with |a + l| - 1 found from its
    // square so no 1 is ever subtracted.
    float n = sqrt(1.0 + 2.0 * dot(a, l) + dot(l, l));
    away = (l - a * ((2.0 * dot(a, l) + dot(l, l)) / (n + 1.0))) / n;
    dir = a + away;
}

// The coast at unit direction `dir`, `away` from the anchor's: x = meters
// inland, yzw = how fast that grows per meter along the sea, in the planet's
// axes. Seaward of where the
// shore reaches, and on dry ground, x is only a bound on the side it lies,
// which reads the same. Zero for a sea without coasts.
float4 water_shore_field(float3 dir, float3 away)
{
    if (WATER_SHORE.continent_octaves == 0u)
    {
        return (float4)0.0;
    }
    float scale = WATER_SHORE.continent_scale;
    float offset = WATER_SHORE.continent_offset;
    float2 keep = float2(WATER_SHORE.bounds.x + offset, WATER_SHORE.coast.y + offset) / scale;
    float4 field = water_shore_octaves(away, 0u, WATER_SHORE.continent_frequency,
                                       min(WATER_SHORE.continent_octaves, WATER_CONTINENT_OCTAVES),
                                       WATER_SHORE.continent_salts, keep);
    float3 grad = field.xyz * scale;
    return float4(field.w * scale - offset, (grad - dir * dot(dir, grad)) * WATER_SHORE.noise.z);
}

// Whether the shore reaches coast `field` (dry ground included, so the
// triangles along the waterline interpolate a whole phase noise).
bool water_shore_reaches(float4 field)
{
    return WATER_SHORE.continent_octaves > 0u && field.x > WATER_SHORE.bounds.x;
}

// Where a point at coast `field` lies against the shore (`ShorePoint`), and
// how much of each cascade the depth there leaves.
struct WaterShorePoint
{
    float depth;
    float weight;
    float3 phases;
    float3 k;
    float3 landward;
    float4 damp;
};

WaterShorePoint water_shore_point(float4 field)
{
    WaterShorePoint p = (WaterShorePoint)0;
    p.damp = (float4)1.0;
    // Out where the shore stops reaching, the open sea is left as it is.
    if (!water_shore_reaches(field))
    {
        return p;
    }
    float inland = field.x;
    float gl = length(field.yzw);
    p.depth = water_shore_depth(inland);
    for (uint c = 0u; c < WATER_OCEAN_MAX_CASCADES; c++)
    {
        float l = WATER_SHORE.cascade_wavelengths[c];
        p.damp[c] = l > 0.0 ? smoothstep(0.25 * l, 0.5 * l, p.depth) : 1.0;
    }
    float offshore = WATER_SHORE.coast.y - inland;
    float3 phases;
    water_shore_table(offshore, phases, p.k);
    float peak = WATER_SHORE.beach.w;
    float reach = WATER_SHORE.beach.z;
    p.weight = p.depth > 0.0
                 ? (1.0 - smoothstep(0.25 * peak, 0.5 * peak, p.depth))
                       * (1.0 - smoothstep(0.8 * reach, reach, offshore))
                       * smoothstep(WATER_GRADIENT_FADE.x, WATER_GRADIENT_FADE.y, gl)
                 : 0.0;
    p.phases = phases / max(gl, WATER_GRADIENT_FADE.x);
    p.landward = gl > 1e-9 ? field.yzw / gl : (float3)0.0;
    return p;
}

// The phase noise at unit direction `dir`, `away` from the anchor's: w =
// radians, xyz = its gradient in radians per meter along the sea, in the
// planet's axes.
float4 water_shore_noise(float3 dir, float3 away)
{
    uint4 salts[2] = { WATER_SHORE.noise_salts, (uint4)0u };
    float4 n = water_shore_octaves(away, WATER_CONTINENT_OCTAVES, WATER_SHORE.noise.x,
                                   WATER_PHASE_NOISE_OCTAVES, salts, float2(-1.0e30, 1.0e30));
    float k = 2.0 * WATER_SHORE.noise.y / WATER_PHASE_NOISE_WEIGHT;
    float3 along = n.xyz - dir * dot(dir, n.xyz);
    return float4(along * (k * WATER_SHORE.noise.w), k * n.w - WATER_SHORE.noise.y);
}

// How tall the swell runs `away` from the anchor's direction, as a share
// of its mean (`ShoreNoise::swell`).
float water_shore_swell(float3 away)
{
    uint4 salts[2] = { uint4(WATER_SHORE.noise_salts.z, 0u, 0u, 0u), (uint4)0u };
    float4 n = water_shore_octaves(away, WATER_CONTINENT_OCTAVES + WATER_PHASE_NOISE_OCTAVES,
                                   WATER_SHORE.swell.x, 1u, salts, float2(-1.0e30, 1.0e30));
    return 1.0 + WATER_SHORE.swell.y * (2.0 * n.w - 1.0);
}

// How many times taller a wave of `omega` and wavenumber `k` stands in water
// `depth` meters deep than in deep water (`shoaling_at`).
float water_shore_shoaling(float omega, float k, float depth)
{
    float kh = k * depth;
    float n = kh > 20.0 ? 0.5 : 0.5 * (1.0 + 2.0 * kh / sinh(2.0 * kh));
    return sqrt(0.5 * WATER_GRAVITY / omega / (n * omega / k));
}

// The shore waves at one point (`ShoreSurface`), in the planet's axes, and
// the carrier's wavenumber and slope.
struct WaterShoreSurface
{
    float height;
    float3 shift;
    float3 slope;
    float foam;
    // Periods since the crest passed, and the unit direction it travels.
    float age;
    float3 toward;
    float k;
    float steepness;
};

WaterShoreSurface water_shore_surface(WaterShorePoint p, float4 noise, float swell)
{
    WaterShoreSurface o = (WaterShoreSurface)0;
    if (p.weight <= 0.0)
    {
        return o;
    }
    float shallow = max(p.depth, WATER_SHALLOWEST);
    float re = 0.0;
    float im = 0.0;
    float total = 0.0;
    float energy = 0.0;
    float k_sum = 0.0;
    for (uint i = 0u; i < WATER_SHORE_TRAINS; i++)
    {
        float a = p.weight * swell * WATER_SHORE.amplitudes[i]
                * water_shore_shoaling(WATER_SHORE.omegas[i], p.k[i], shallow);
        float s, c;
        sincos(-(WATER_SHORE.phases[i] + p.phases[i] + noise.w), s, c);
        re += a * c;
        im += a * s;
        total += a;
        energy += a * a;
        k_sum += a * p.k[i];
    }
    float envelope = sqrt(re * re + im * im);
    float carrier = envelope > 0.0 ? atan2(im, re) : 0.0;
    float3 travel = p.landward * (k_sum / max(total, 1e-12)) - noise.xyz;
    float k = length(travel);
    if (k <= 1e-9)
    {
        return o;
    }
    float3 toward = travel / k;
    float cap = max(WATER_BREAKER_INDEX * p.depth, 1e-6);
    float ratio = 2.0 * envelope / cap;
    float held = min(envelope, 0.5 * cap);
    float lean = WATER_MAX_LEAN * smoothstep(0.5, 1.0, ratio);
    float u = carrier + lean * (1.0 - cos(carrier));
    float du = 1.0 + lean * sin(carrier);
    float pinch = WATER_MAX_PINCH * smoothstep(0.4, 1.0, ratio) / (1.0 + lean);
    float su, cu;
    sincos(u, su, cu);
    float rise = -held * su * du * k;
    float spread = -pinch * cu * du;
    float since = -carrier;
    since -= WATER_TAU * floor(since / WATER_TAU);
    float trail = exp(-since / (WATER_TAU * WATER_FOAM_TRAIL));
    float lead = smoothstep(WATER_TAU - WATER_FOAM_LEAD, WATER_TAU, since);
    float crest = smoothstep(0.85, 1.15, ratio) * max(trail, lead);
    float rms = 2.0 * sqrt(energy) / cap;
    o.height = held * cu;
    o.shift = toward * (-pinch * su / k);
    o.slope = toward * (rise / (1.0 + spread));
    o.foam = max(crest, WATER_RESIDUAL_FOAM * smoothstep(0.8, 1.5, rms));
    o.age = since / WATER_TAU;
    o.toward = toward;
    o.k = k;
    o.steepness = held * k;
    return o;
}

// The surf's foam: webs of foam along the borders of cells two sizes wide,
// laid out in the planet's axes, which one shows drifting with smooth noise of
// cells as wide as the patches the foam breaks into.
static const float WATER_SURF_CELL = 0.64;
static const float WATER_SURF_COARSE = 1.28;
static const float WATER_SURF_PATCH = 3.2;
// How far the webs' cells stretch along the wave's travel as the foam ages.
static const float WATER_SURF_STRETCH = 1.6;
// How far, in meters, the webs' lines wander off their cells' borders.
static const float WATER_SURF_WARP = 1.0;

// How far the surf's foam has drifted shoreward `age` periods after its
// crest passed, on a wave of wavenumber `k` (`foam_drift`).
float water_surf_drift(float age, float k)
{
    float a = saturate(age);
    return WATER_FOAM_RIDE * WATER_TAU / max(k, 1e-4) * a * (1.0 - a) * (1.0 - a);
}

// Value noise in [0, 1] with a quintic step, so no lattice line shows, at
// lattice position `p`, the lattice repeating every `period` cells.
float water_surf_noise(float3 p, float period, uint salt)
{
    float3 cell = floor(p);
    float3 f = p - cell;
    float3 s = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float3 c0 = cell - period * floor(cell / period);
    float3 c1 = c0 + 1.0;
    c1 = select(c1 >= period, (float3)0.0, c1);
    int3 a = int3(c0);
    int3 b = int3(c1);
    float x00 = lerp(ground_lattice(int3(a.x, a.y, a.z), salt), ground_lattice(int3(b.x, a.y, a.z), salt), s.x);
    float x10 = lerp(ground_lattice(int3(a.x, b.y, a.z), salt), ground_lattice(int3(b.x, b.y, a.z), salt), s.x);
    float x01 = lerp(ground_lattice(int3(a.x, a.y, b.z), salt), ground_lattice(int3(b.x, a.y, b.z), salt), s.x);
    float x11 = lerp(ground_lattice(int3(a.x, b.y, b.z), salt), ground_lattice(int3(b.x, b.y, b.z), salt), s.x);
    return lerp(lerp(x00, x10, s.y), lerp(x01, x11, s.y), s.z);
}

// How far lattice position `p` lies from the border between its nearest two
// cells of a jittered lattice repeating every `period` cells, in cells,
// distances along unit `toward` shrunk by `stretch` so the cells run long
// that way.
float water_surf_border(float3 p, float period, float3 toward, float stretch)
{
    float3 cell = floor(p);
    float3 f = p - cell;
    float squash = 1.0 - 1.0 / (stretch * stretch);
    float2 nearest = (float2)1.0e9;
    for (int z = -1; z <= 1; z++)
    {
        for (int y = -1; y <= 1; y++)
        {
            for (int x = -1; x <= 1; x++)
            {
                float3 d = float3(x, y, z);
                float3 c = cell + d;
                c -= period * floor(c / period);
                int3 at = int3(c);
                float3 jitter = float3(ground_lattice(at, 0x68e31da4u), ground_lattice(at, 0xb5297a4du),
                                       ground_lattice(at, 0x1b56c4e9u));
                float3 delta = d + jitter - f;
                float along = dot(delta, toward);
                // Squared: the root is taken of the nearest two alone.
                float dist = max(dot(delta, delta) - squash * along * along, 0.0);
                nearest = dist < nearest.x ? float2(dist, nearest.x) : float2(nearest.x, min(nearest.y, dist));
            }
        }
    }
    return sqrt(nearest.y) - sqrt(nearest.x);
}

// Three value noises in [0, 1] at lattice position `p`, repeating every
// `period` cells, for a warp.
float3 water_surf_warp(float3 p, float period)
{
    return float3(water_surf_noise(p, period, 0x2545f491u), water_surf_noise(p + 0.31, period, 0x9e3779b9u),
                  water_surf_noise(p + 0.67, period, 0x7feb352du));
}

// The surf's foam `cover` at `p`, meters in the planet's axes within one
// period, `age` periods after its crest passed, the wave traveling along
// unit `toward`, seen through a pixel `footprint` meters across. Fresh foam
// behind a bore is a nearly whole sheet with round holes; as it ages it thins
// into translucent webs along the borders of cells streaked along the
// travel, coarse in some places and fine in others, their lines bent at two
// scales and swelling and pinching along their length, and breaks into
// patches with clear water between. It fades to its cover as its cells
// shrink below the pixel, so it never aliases into streaks at grazing views.
float water_surf_lace(float cover, float age, float3 p, float3 toward, float footprint)
{
    if (cover <= 0.0)
    {
        return 0.0;
    }
    float fade = water_fade(WATER_SURF_CELL, footprint);
    if (fade <= 0.0)
    {
        return cover;
    }
    float period = WATER_SHORE.lace.w;
    // How old the foam reads, back to fresh as the next crest arrives.
    float old = saturate(2.5 * age) * (1.0 - smoothstep(0.85, 1.0, age));
    float patch = water_surf_noise(p / WATER_SURF_PATCH, period / WATER_SURF_PATCH, 0x51ed270bu);
    float coarseness = water_surf_noise(p / WATER_SURF_PATCH + 0.5, period / WATER_SURF_PATCH, 0x0bb2c7a3u);
    // A broad warp bends the coarse cells' borders and a finer one every
    // short run of them.
    float3 q = p + (water_surf_warp(p / WATER_SURF_COARSE, period / WATER_SURF_COARSE) - 0.5) * WATER_SURF_WARP;
    q += (water_surf_warp(q / (0.5 * WATER_SURF_CELL), 2.0 * period / WATER_SURF_CELL) - 0.5)
       * (0.4 * WATER_SURF_CELL);
    float stretch = lerp(1.0, WATER_SURF_STRETCH, old);
    // The foam lies along the borders, as wide as it is thick: thick fresh
    // foam leaves only holes, thin old foam only the webs, their width
    // wandering along them and breaking them into strands as they thin.
    float thick = saturate(cover * (0.6 + 0.8 * patch));
    float vary = water_surf_noise(q / (0.5 * WATER_SURF_CELL), 2.0 * period / WATER_SURF_CELL, 0x165667b1u);
    float width = (0.015 + 0.5 * thick * thick) * lerp(1.0, 0.6, old)
                * saturate(lerp(0.3 + 1.4 * vary, 2.0 * vary - 0.3, old * (1.0 - smoothstep(0.5, 0.9, thick))));
    // Each scale's cells are searched only where they show.
    float which = smoothstep(0.35, 0.65, coarseness);
    float fine_web = 0.0;
    if (which < 1.0)
    {
        float fine = WATER_SURF_CELL
                   * water_surf_border(q / WATER_SURF_CELL, period / WATER_SURF_CELL, toward, stretch);
        fine_web = 1.0 - smoothstep(0.4 * width, width, fine);
    }
    float coarse_web = 0.0;
    if (which > 0.0)
    {
        float coarse = WATER_SURF_COARSE
                     * water_surf_border(q / WATER_SURF_COARSE, period / WATER_SURF_COARSE, toward, stretch);
        coarse_web = 1.0 - smoothstep(0.5 * width, 1.4 * width, coarse);
    }
    float web = lerp(fine_web, coarse_web, which);
    // Thin foam gathers into patches, clear water between them, a
    // translucent film inside them laced by the webs; the thicker the foam
    // the more of the sea its patches cover, until behind the bore it is
    // one sheet.
    float gather = 0.4 * patch + 0.6 * water_surf_noise(q / WATER_SURF_COARSE, period / WATER_SURF_COARSE,
                                                        0x3c6ef372u);
    float edge = 0.9 - 1.1 * thick;
    // The webs fade in only well inside a patch, after its film, so no lone
    // junction is left on clear water at a patch's margin.
    float inside = smoothstep(edge, edge + 0.15, gather);
    float film = lerp(0.2, 0.9, thick);
    float lace = max(web * smoothstep(edge + 0.03, edge + 0.12, gather), film * inside)
               * lerp(1.0, 0.55, old * (1.0 - thick));
    return lerp(cover, lace, fade);
}

// How the wake fades at frame point `p` (x, z): whole inside its clear
// area, nothing at its window's edge or while none runs.
float water_wake_fade(float2 p)
{
    WaterWake w = params.sea.wake;
    float2 off = abs(p - w.lookup.zw);
    float edge = max(off.x, off.y);
    return w.fade.w > 0.0 ? saturate((w.fade.x - edge) * w.fade.y + 1.0) : 0.0;
}

// The wake at frame point `p` (x, z), read at the mip a footprint
// `footprint` meters across needs: (height, slope x, slope z, foam).
float4 water_wake_level(float2 p, float footprint)
{
    float fade = water_wake_fade(p);
    if (fade <= 0.0)
    {
        return (float4)0.0;
    }
    float lod = log2(max(footprint / params.sea.wake.lookup.y, 1.0));
    return fade * wake_field.SampleLevel(ocean_sampler, p * params.sea.wake.lookup.x, lod);
}

// The wake at frame point `p` (x, z), read through a pixel whose footprint
// moves `dx`, `dy` meters across it.
float4 water_wake(float2 p, float2 dx, float2 dy)
{
    float fade = water_wake_fade(p);
    if (fade <= 0.0)
    {
        return (float4)0.0;
    }
    float s = params.sea.wake.lookup.x;
    return fade * wake_field.SampleGrad(ocean_sampler, p * s, dx * s, dy * s);
}

// The tallest the wake stands at coast `field`, which breaks it where the
// water grows shallow; unbounded out at sea.
float water_wake_limit(float4 field)
{
    return water_shore_reaches(field) ? params.sea.wake.fade.z * max(water_shore_depth(field.x), 0.0)
                                      : 1e6;
}

// Below these the wake is still water: a millimeter of height, a thousandth
// of slope, a hundredth of foam.
static const float WATER_WAKE_QUIET_HEIGHT = 1e-3;
static const float WATER_WAKE_QUIET_SLOPE = 1e-3;
static const float WATER_WAKE_QUIET_FOAM = 1e-2;

// How far, in meters, the wake must stand over what the depth holds before it
// breaks into foam, and by how far it is fully broken; and how much foam
// breaking adds at most.
static const float2 WATER_WAKE_BREAK = float2(0.05, 0.3);
static const float WATER_WAKE_BREAK_FOAM = 0.5;

// Whether `wake` stirs the surface enough to draw.
bool water_wake_stirs(float4 wake)
{
    return abs(wake.x) > WATER_WAKE_QUIET_HEIGHT || wake.w > WATER_WAKE_QUIET_FOAM
           || max(abs(wake.y), abs(wake.z)) > WATER_WAKE_QUIET_SLOPE;
}

// The wake's foam lies in lines through noise drawn out along the way it
// streaks, on cells this many meters across: strands, finer threads between
// them, and the patches the foam breaks into as it ages.
static const float3 WATER_WAKE_LACE_CELLS = float3(0.25, 0.09, 1.6);

// How much longer than wide the strands' cells are along a streak.
static const float WATER_WAKE_STREAK = 4.0;

// How wide a web is in meters, the thinnest thread and the thickest clump,
// for the strands and the finer webs.
static const float4 WATER_WAKE_WEB = float4(0.004, 0.06, 0.003, 0.016);

// How far, in meters, the foam's outline wanders off the field, on noise
// cells this wide, so the field's cells never show; and how far the webs
// wander on finer cells, so they bend and the lattices' rows never show.
static const float WATER_WAKE_WARP = 0.35;
static const float WATER_WAKE_WARP_CELL = 0.7;
static const float WATER_WAKE_FINE_WARP = 0.06;
static const float WATER_WAKE_FINE_WARP_CELL = 0.3;

// The ragged edge and the bubbly grain of the foam up close: noise cells in
// meters, and how far the edge frays in meters and the grain darkens it.
static const float2 WATER_WAKE_GRAIN_CELLS = float2(0.04, 0.02);
static const float2 WATER_WAKE_GRAIN = float2(0.012, 0.35);

// Patches the foam breaks into, in meters, and the share of the water the
// patches leave open, in fresh foam and in old.
static const float WATER_WAKE_PATCH = 5.0;
static const float2 WATER_WAKE_OPEN = float2(0.05, 0.6);

// The streaky film that clumps between the lines of the thickest foam.
static const float WATER_WAKE_FILM = 0.45;

// How opaque the thinnest foam is, as a share of the thickest: thin foam is
// sheer as well as sparse.
static const float WATER_WAKE_THIN_OPACITY = 0.3;

// The wake's foam is wet, thin and full of water: it takes this share of the
// light the whitecaps do, and that much nearer gray.
static const float WATER_WAKE_FOAM_LIGHT = 0.82;
static const float WATER_WAKE_FOAM_GRAY = 0.35;

// How the way the foam streaks is read: off the field's mip this many meters
// across, at taps this far apart, and how strongly it leans to the frame's x
// where the foam is too even to tell.
static const float WATER_WAKE_STREAK_READ = 2.0;
static const float WATER_WAKE_STREAK_REACH = 2.5;
static const float WATER_WAKE_STREAK_EVEN = 2e-3;

// Value noise in [0, 1] at frame point `p` on cells `cell` meters across.
float water_wake_noise(float2 p, float cell)
{
    float2 q = p / cell;
    q -= floor(q / 256.0) * 256.0;
    return water_periodic_noise(q, 256.0);
}

// A wander of up to `reach` meters at frame point `p`, on noise cells `cell`
// meters across.
float2 water_wake_wander(float2 p, float cell, float reach)
{
    float2 n = float2(water_wake_noise(p, cell), water_wake_noise(p + float2(37.3, 11.9), cell));
    return (2.0 * n - 1.0) * reach;
}

// How far the wake's foam is read off frame point `p`, in meters.
float2 water_wake_warp(float2 p)
{
    return water_wake_wander(p, WATER_WAKE_WARP_CELL, WATER_WAKE_WARP);
}

// A web `width` meters wide along borders `border` meters off, its edge as
// sharp as a pixel `footprint` meters across can show, faded by `fade` to
// `mean` as its cells shrink below the pixel.
float water_wake_web(float border, float width, float footprint, float fade, float mean)
{
    float soft = 0.5 * max(0.8 * width, footprint);
    return lerp(mean, 1.0 - smoothstep(width - soft, width + soft, border), fade);
}

// The angle in [0, pi) from the frame's x that the wake's foam streaks along
// at frame point `p`: across the way it changes most, which behind a body is
// along its trail. Read off a coarse mip, so it turns only as the trail does.
float water_wake_streak(float2 p)
{
    float r = WATER_WAKE_STREAK_REACH;
    float d = 0.70710678 * r;
    float at = water_wake_level(p, WATER_WAKE_STREAK_READ).w;
    float2 offs[4] = { float2(r, 0.0), float2(d, d), float2(0.0, r), float2(-d, d) };
    float change[4];
    for (uint k = 0u; k < 4u; k++)
    {
        change[k] = abs(water_wake_level(p + offs[k], WATER_WAKE_STREAK_READ).w - at)
                  + abs(water_wake_level(p - offs[k], WATER_WAKE_STREAK_READ).w - at);
    }
    float most = 0.5 * atan2(change[1] - change[3], change[0] - change[2] + WATER_WAKE_STREAK_EVEN);
    float along = most + 1.5707963;
    return along - floor(along / 3.1415927) * 3.1415927;
}

// How far, in meters, frame point `q` lies from where noise on cells `cell`
// meters across crosses its middle, the noise drawn out along `along` by
// the streak: lines that run along the streak, braiding as they go.
float water_wake_line(float2 q, float2 along, float cell)
{
    float2 s = float2(dot(q, along) / WATER_WAKE_STREAK, dot(q, float2(-along.y, along.x))) / cell;
    s -= floor(s / 256.0) * 256.0;
    float h = 0.05;
    float n = water_periodic_noise(s, 256.0);
    float2 rise = float2(water_periodic_noise(s + float2(h, 0.0), 256.0) - n,
                         water_periodic_noise(s + float2(0.0, h), 256.0) - n)
                / (h * cell * float2(WATER_WAKE_STREAK, 1.0));
    return abs(n - 0.5) / max(length(rise), 1e-3);
}

// One streaked lace of foam `thick` thick at frame point `q`, its lines drawn
// out along angle `angle`, seen through a pixel `footprint` meters across;
// `fray` ragged at its edges.
float water_wake_streaks(float2 q, float angle, float thick, float footprint, float fray)
{
    float2 along = float2(cos(angle), sin(angle));
    float3 cells = WATER_WAKE_LACE_CELLS;
    // Threads swell into clumps and pinch out along their length.
    float2 s = float2(dot(q, along) / WATER_WAKE_STREAK, dot(q, float2(-along.y, along.x)));
    float vary = water_wake_noise(s, 0.5 * cells.x);
    float width = lerp(WATER_WAKE_WEB.x, WATER_WAKE_WEB.y, vary * vary) * lerp(0.5, 1.0, thick);
    float fade = water_fade(cells.x, footprint);
    float mean = saturate(2.0 * width / cells.x);
    float lace = fade > 0.0
        ? water_wake_web(water_wake_line(q, along, cells.x) + fray, width, footprint, fade, mean)
        : mean;
    lace *= lerp(0.55, 1.0, vary);
    float filling = smoothstep(0.1, 0.5, thick);
    if (filling > 0.0)
    {
        float web = lerp(WATER_WAKE_WEB.z, WATER_WAKE_WEB.w, 1.0 - vary) * thick;
        float fine_fade = water_fade(cells.y, footprint);
        float fine_mean = saturate(2.0 * web / cells.y);
        float fine = fine_fade > 0.0
            ? water_wake_web(water_wake_line(q + 17.3, along, cells.y) + fray, web, footprint, fine_fade, fine_mean)
            : fine_mean;
        lace = max(lace, 0.8 * filling * fine);
        float clump = smoothstep(0.5, 0.85, water_wake_noise(s + 5.3, 2.0 * cells.x));
        lace = max(lace, WATER_WAKE_FILM * clump * smoothstep(0.4, 1.0, thick) * (0.5 + 0.5 * vary));
    }
    return lace;
}

// How opaque the wake's foam stands at frame point `p` where it covers
// `cover` of the water, seen through a pixel `footprint` meters across. Fresh
// foam right behind a body is a dense run of threads and clumps streaked
// along its trail; as it ages it breaks into uneven patches over open water,
// and its opacity thins with it.
float water_wake_lace(float cover, float2 p, float footprint)
{
    cover = saturate(cover);
    float2 q = p + water_wake_wander(p, WATER_WAKE_FINE_WARP_CELL, WATER_WAKE_FINE_WARP);
    float patch = 0.65 * water_wake_noise(p, WATER_WAKE_PATCH)
                + 0.35 * water_wake_noise(q, WATER_WAKE_LACE_CELLS.z);
    float open = lerp(WATER_WAKE_OPEN.y, WATER_WAKE_OPEN.x, smoothstep(0.03, 0.35, cover));
    float gather = smoothstep(open - 0.12, open + 0.12, patch);
    float thick = saturate(2.5 * cover * gather * (0.6 + 0.6 * patch));
    if (thick <= 0.0)
    {
        return 0.0;
    }
    float2 grain_fade = float2(water_fade(2.0 * WATER_WAKE_GRAIN_CELLS.x, footprint),
                               water_fade(2.0 * WATER_WAKE_GRAIN_CELLS.y, footprint));
    float fray = grain_fade.x > 0.0
        ? grain_fade.x * WATER_WAKE_GRAIN.x * (water_wake_noise(q, WATER_WAKE_GRAIN_CELLS.x) - 0.5)
        : 0.0;
    // Between two of four fixed directions, so no cell slides as the streak
    // turns.
    float bin = water_wake_streak(p) / 0.78539816;
    float first = floor(bin);
    float turn = smoothstep(0.2, 0.8, bin - first);
    float lace = water_wake_streaks(q, first * 0.78539816, thick, footprint, fray);
    if (turn > 0.0)
    {
        lace = lerp(lace, water_wake_streaks(q, (first + 1.0) * 0.78539816, thick, footprint, fray), turn);
    }
    if (grain_fade.y > 0.0)
    {
        float grain = water_wake_noise(q + float2(7.1, 3.7), WATER_WAKE_GRAIN_CELLS.y);
        lace *= 1.0 - grain_fade.y * WATER_WAKE_GRAIN.y * grain * grain;
    }
    return lace * lerp(WATER_WAKE_THIN_OPACITY, 1.0, thick);
}

// The simulated ocean's displacement at a rest point, each face's read at
// the mip a pixel `footprint` meters across needs, the heights along the
// sea's up there.
float3 water_sea_displace(float3 rest, float footprint, float4 shore_field, float4 shore_noise,
                          float shore_swell)
{
    float3 up = water_sea_up(rest);
    float3 w = water_sea_weights(rest);
    WaterShorePoint shore = water_shore_point(shore_field);
    water_shore_damp = shore.damp;
    float3 d = (float3)0.0;
    for (uint f = 0u; f < WATER_SEA_FACES; f++)
    {
        if (w[f] <= 0.0)
        {
            continue;
        }
        water_face = f;
        WaterSeaPoint p = water_sea_point(f, rest);
        float3 o = water_ocean_displace(p.s, footprint * p.stretch);
        d += w[f] * (o.y * up + (o.x * p.tu + o.z * p.tv) / p.stretch);
    }
    if (shore.weight > 0.0)
    {
        WaterShoreSurface waves = water_shore_surface(shore, shore_noise, shore_swell);
        float fade = water_fade(WATER_TAU / max(waves.k, 1e-4), footprint);
        d += fade * (waves.height * up + water_shore_to_frame(waves.shift));
    }
    return d;
}

// The whitecap cover, the surf's foam and the wake's `water_sea_shading`
// found.
static float water_sea_cover = 0.0;
static float water_sea_surf = 0.0;
static float water_sea_wake = 0.0;

// The shading at a fragment's rest point: every live face's waves and
// ripples, read through the pixel's footprint on the sea (`dx`, `dy`, from
// `water_footprint`) taken onto the face, their slopes turned onto the sea
// and blended by each face's share. The gusts are read where the grass reads
// them.
WaterShading water_sea_shading(WaterVertexOut i, float3 dx, float3 dy)
{
    float4 at = float4(i.rest + params.anchor.xyz, 1.0);
    water_gust_xz = float2(dot(params.sea.wind_map[0], at), dot(params.sea.wind_map[1], at));
    float3 up = water_sea_up(i.rest);
    float3 w = water_sea_weights(i.rest);
    float3 slope = (float3)0.0;
    float alpha2 = 0.0;
    float cover = 0.0;
    // The shore waves, faded as their crests shrink below the pixel; what the
    // fade takes goes to the roughness, the foam to the lace below.
    WaterShorePoint shore = water_shore_point(i.shore_field);
    water_shore_damp = shore.damp;
    float footprint = max(length(dx), length(dy));
    float surf = 0.0;
    if (shore.weight > 0.0)
    {
        WaterShoreSurface waves = water_shore_surface(shore, i.shore_noise, i.shore_swell);
        float fade = water_fade(WATER_TAU / max(waves.k, 1e-4), footprint);
        slope += fade * water_shore_to_frame(waves.slope);
        alpha2 += (1.0 - fade * fade) * 0.5 * waves.steepness * waves.steepness;
        float3 at = WATER_SHORE.lace.xyz
                  + float3(dot(WATER_SHORE.planet_rows[0].xyz, i.rest), dot(WATER_SHORE.planet_rows[1].xyz, i.rest),
                           dot(WATER_SHORE.planet_rows[2].xyz, i.rest));
        // The foam rides in with its bore and slips back, not painted on the sea.
        at -= waves.toward * water_surf_drift(waves.age, waves.k);
        surf = water_surf_lace(saturate(waves.foam * params.foam_intensity), waves.age, at, waves.toward,
                               footprint);
    }
    // The wake, read first: its churned water calms the short waves below.
    float2 p = (i.rest + params.anchor.xyz).xz;
    float4 wake = water_wake(p, dx.xz, dy.xz);
    bool stirred = water_wake_stirs(wake);
    // From the quiet foam up, so the calm meets 1 where the wake goes quiet.
    if (stirred)
    {
        water_calm = 1.0 - WATER_WAKE_SLICK * smoothstep(WATER_WAKE_QUIET_FOAM, WATER_WAKE_SLICK_FOAM, wake.w);
    }
    for (uint f = 0u; f < WATER_SEA_FACES; f++)
    {
        if (w[f] <= 0.0)
        {
            continue;
        }
        water_face = f;
        WaterSeaPoint p = water_sea_point(f, i.rest);
        float2 sdx = p.stretch * float2(dot(dx, p.tu), dot(dx, p.tv));
        float2 sdy = p.stretch * float2(dot(dy, p.tu), dot(dy, p.tv));
        WaterShading face = water_shading(p.s, sdx, sdy);
        float2 rise = -face.normal.xz / max(face.normal.y, 1e-3);
        slope += w[f] * p.stretch * (rise.x * p.tu + rise.y * p.tv);
        alpha2 += w[f] * w[f] * face.alpha * face.alpha;
        if (water_whitecaps())
        {
            cover += w[f] * w[f] * water_whitecap_cover(p.s, sdx, sdy);
        }
    }
    // The wake over everything, held to what the depth lets stand; what
    // the shallows break off turns to foam.
    if (stirred)
    {
        float limit = water_wake_limit(i.shore_field);
        float height = abs(wake.x);
        slope += float3(wake.y, 0.0, wake.z) * saturate(limit / max(height, 1e-6));
        // Breaking in the shallows foams only where the wake stands well over
        // what the depth holds, in a narrow band.
        float broken = WATER_WAKE_BREAK_FOAM * smoothstep(WATER_WAKE_BREAK.x, WATER_WAKE_BREAK.y, height - limit);
        if (saturate((wake.w + broken) * params.foam_intensity) > WATER_WAKE_QUIET_FOAM)
        {
            // Read off a wandering point, so the foam's outline never traces
            // the field's cells.
            float2 q = p + water_wake_warp(p);
            float churn = saturate((water_wake(q, dx.xz, dy.xz).w + broken) * params.foam_intensity);
            water_sea_wake = water_wake_lace(churn, q, footprint);
        }
    }
    water_sea_cover = cover;
    water_sea_surf = surf;
    WaterShading o;
    o.normal = normalize(up - slope);
    o.alpha = sqrt(alpha2);
    return o;
}

[shader("vertex")]
WaterVertexOut water_vertex(WaterVertexIn v)
{
    // The grid is laid out in face meters around the anchor; scaled to
    // meters along the sea, held within the horizon, and wrapped onto the
    // sphere along the great circles out from the anchor.
    float2 g = v.pos.xz * params.sea.tangent_u.w;
    float d = length(g);
    float reach = params.sea.tangent_v.w;
    if (d > reach)
    {
        g *= reach / d;
        d = reach;
    }
    float radius = params.sea.up.w;
    float turn = d / radius;
    float half_turn = sin(0.5 * turn);
    float drop = 2.0 * radius * half_turn * half_turn;
    g *= d > 0.0 ? radius * sin(turn) / d : 1.0;
    float3 rest = g.x * params.sea.tangent_u.xyz + g.y * params.sea.tangent_v.xyz - drop * params.sea.up.xyz;
    float rest_w = mul(view.vp, float4(rest + params.anchor.xyz, 1.0)).w;
    float footprint = water_pixel_meters(rest_w);
    float3 dir;
    float3 away;
    water_shore_place(rest, dir, away);
    float4 shore_field = water_shore_field(dir, away);
    bool reaches = water_shore_reaches(shore_field);
    float4 shore_noise = reaches ? water_shore_noise(dir, away) : (float4)0.0;
    float shore_swell = reaches ? water_shore_swell(away) : 1.0;
    float3 pos = rest;
    // Nothing finer than two cells, which the grid would alias.
    float cell = water_grid_cell(v.pos.xz) * params.sea.tangent_u.w;
    if (params.ocean.layers.x > 0u)
    {
        pos += water_sea_displace(rest, max(footprint, 2.0 * cell), shore_field, shore_noise, shore_swell);
    }
    float4 wake = water_wake_level((rest + params.anchor.xyz).xz, max(footprint, 2.0 * cell));
    if (abs(wake.x) > WATER_WAKE_QUIET_HEIGHT)
    {
        float limit = water_wake_limit(shore_field);
        pos += clamp(wake.x, -limit, limit) * water_sea_up(rest);
    }
    pos += params.anchor.xyz;

    WaterVertexOut o;
    o.world_pos = pos;
    o.rest = rest;
    o.shore_field = shore_field;
    o.shore_noise = shore_noise;
    o.shore_swell = shore_swell;
    o.position = mul(view.vp, float4(pos, 1.0));
    return o;
}

#else

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

#endif

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

#ifdef WATER_SEA

// Where the camera ray through screen position `pixel` meets the sea's
// sphere, relative to the camera; far along the ray when it never does. The
// near root is taken in the form that keeps its digits beside the camera.
float3 water_sea_hit(float2 pixel)
{
    float2 uv = pixel / max(view.viewport, (float2)(1.0));
    float2 ndc = float2(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));
    float3 dir = camera_view_ray(view.inv_vp, ndc, view.camera_pos.xyz);
    float3 e = view.camera_pos.xyz - params.anchor.xyz;
    float radius = params.sea.up.w;
    float b = dot(e, dir) + radius * dot(params.sea.up.xyz, dir);
    float c = dot(e, e) + 2.0 * radius * dot(e, params.sea.up.xyz);
    float disc = b * b - c;
    float t = (c > 0.0 && b < 0.0 && disc >= 0.0) ? c / (-b + sqrt(disc)) : 1e7;
    return t * dir;
}

// How far the point on the sea moves across one pixel right (`dx`) and one
// down (`dy`), from the camera rays through the pixel and its neighbors, as
// for a rectangle's rest plane.
void water_footprint(float2 pixel, out float3 dx, out float3 dy)
{
    float3 center = water_sea_hit(pixel);
    dx = water_sea_hit(pixel + float2(1.0, 0.0)) - center;
    dy = water_sea_hit(pixel + float2(0.0, 1.0)) - center;
}

#else

// Where the camera ray through screen position `pixel` meets the flat rest
// plane, in XZ relative to the camera; far along the ray when it never does.
float2 water_rest_hit(float2 pixel)
{
    float2 uv = pixel / max(view.viewport, (float2)(1.0));
    float2 ndc = float2(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));
    float3 dir = camera_view_ray(view.inv_vp, ndc, view.camera_pos.xyz);
    float height = view.camera_pos.y - params.anchor.y;
    float t = -height / (abs(dir.y) > 1e-6 ? dir.y : -1e-6);
    return (t > 0.0 ? t : 1e7) * dir.xz;
}

// How far the rest XZ moves across one pixel right (`dx`) and one down (`dy`)
// from screen position `pixel`, which the wave fades, the cascade filtering
// and the roughness all read as the pixel's footprint. It is taken from the
// camera rays through the pixel and its neighbors rather than from screen
// derivatives of `local_xz`: over the far field's sliver triangles most quads
// are only partly covered, and there some GPUs return different derivatives
// for the same quad from one draw to the next, which flickers the shading of
// whole triangles. The rays depend on the pixel and the view alone. They meet
// the flat rest plane rather than the displaced crests.
void water_footprint(float2 pixel, out float2 dx, out float2 dy)
{
    float2 center = water_rest_hit(pixel);
    dx = water_rest_hit(pixel + float2(1.0, 0.0)) - center;
    dy = water_rest_hit(pixel + float2(0.0, 1.0)) - center;
}

#endif

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

#define DAYLIGHT view.daylight
{DAYLIGHT}

// Only a planet sea's daylight variants light it from its own sky.
#if defined(WATER_SEA) && defined(SURFACE_DAYLIGHT)
#define WATER_SEA_DAYLIGHT 1
#endif

// The radiance a white Lambertian surface facing `n` at `pos` reflects: the
// sun's irradiance through `shadow` plus the sky's, both over pi. Without an
// irradiance cube the sky is the main pass's flat ambient fill. On a planet's
// sea the bodies reach the point through the air above it, and away from the
// camera its own sky lights it.
static const float WATER_INV_PI = 0.318309886;
static const float WATER_FLAT_AMBIENT = 0.03;

// The daylight at a surface point, looked up once for all the ways it is
// lit; none off a planet's sea.
#ifdef WATER_SEA_DAYLIGHT
typedef DaylightPoint WaterDaylight;

WaterDaylight water_daylight(float3 pos)
{
    return daylight_point(pos);
}
#else
struct WaterDaylight
{
    float reach;
};

WaterDaylight water_daylight(float3 pos)
{
    WaterDaylight day;
    day.reach = 0.0;
    return day;
}
#endif

float3 water_lambert(float3 n, float shadow, WaterDaylight day)
{
    float3 sun = view.sun_color.rgb * (saturate(dot(n, view.sun_dir.xyz)) * shadow * WATER_INV_PI);
    float3 sky = view.ambient.x > 0.5
        ? irradiance_cube.SampleLevel(prefilter_cube_sampler, SKY_DIR(n), 0.0).rgb * WATER_INV_PI
        : (float3)(WATER_FLAT_AMBIENT);
#ifdef WATER_SEA_DAYLIGHT
    if (day.reach > 0.0)
    {
        float3 lit = (float3)(0.0);
        [unroll]
        for (uint b = 0u; b < DAYLIGHT_BODIES; b++)
        {
            if (daylight_body(b))
            {
                float s = b == 0u ? shadow : 1.0;
                lit += day.light[b] * (saturate(dot(n, DAYLIGHT.body_dir[b].xyz)) * s);
            }
        }
        sun = lerp(sun, lit * WATER_INV_PI, day.reach);
        sky = lerp(sky, daylight_point_irradiance(day, n) * WATER_INV_PI, day.reach);
    }
#endif
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
    float3 world_pos;
    float3 normal;
    // GGX alpha, the base roughness widened by the slopes the pixel filtered.
    float alpha;
    // The whitecap cover, which hides the reflection and the glint.
    float foam;
    float2 frag_uv;
    float3 below;
    float camera_distance;
    WaterDaylight day;
};

// `dx`, `dy` are the pixel's footprint on the rest plane (`water_footprint`).
WaterSurfacePoint water_surface(WaterVertexOut i, WATER_STEP dx, WATER_STEP dy)
{
    WaterSurfacePoint s;
    WaterShading shading = WATER_SHADE(i, dx, dy);
    s.normal = shading.normal;
    s.alpha = shading.alpha;
    s.foam = 0.0;

    float2 vp_dim = max(view.viewport, (float2)(1.0));
    s.frag_uv = i.position.xy / vp_dim;

    // Refraction: perturb the screen lookup by the waves' slope, the normal's
    // tilt off the surface's up, so the seabed bends under the waves. A planet
    // sea's own curvature tilts its up, which must not shift the lookup.
    float3 up = WATER_UP(i);
    float3 tilt = s.normal - up * dot(s.normal, up);
    float2 refract_uv = clamp(s.frag_uv + tilt.xz * params.refraction_strength,
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
    s.world_pos = i.world_pos;
    s.day = water_daylight(i.world_pos);
    float water_depth = max(scene_dist - water_dist, 0.0);
#ifdef WATER_SEA
    // At a grazing view over a seabed a wave can bend the lookup onto ground
    // nearer than the surface, which would read as no water at all; the
    // column and the refraction are read straight through instead.
    if (scene_dist < water_dist)
    {
        refracted = scene_sample(s.frag_uv).rgb;
        scene_dist = water_scene_distance(ndc_xy, water_scene_depth(water_clamp_pixel(i.position.xy)));
        water_depth = max(scene_dist - water_dist, 0.0);
    }
#endif

    // The body as a lit single-scatter layer: light entering the flat surface
    // (the sun past its entry Fresnel, and the sky) is scattered back out by
    // the column as a Lambertian reflector of the shallow-to-deep reflectance,
    // its share growing as the column thickens over `depth_falloff` meters.
    // The seabed seen through the shallows is filtered by the shallow color.
    float shadow = water_sun_shadow(i.world_pos, i.position.xy);
    float sun_entry = 1.0 - (0.02 + 0.98 * pow(1.0 - saturate(WATER_SUN_HEIGHT(i)), 5.0));
    float3 body_light = water_lambert(WATER_UP(i), shadow * sun_entry, s.day);
    float depth_t = 1.0 - exp(-water_depth / max(params.depth_falloff, 1e-3));
    float3 scattered = lerp(params.shallow_color.rgb, params.deep_color.rgb, depth_t) * body_light;
    float3 below = lerp(refracted * params.shallow_color.rgb, scattered, depth_t);

    // Foam: a white diffuse surface on the waves in a soft band where the
    // seabed is just below the surface, which is both the shoreline and any
    // intersection line with standing geometry, then the whitecaps over what
    // that leaves.
    float foam_t = saturate(1.0 - water_depth / max(params.foam_width, 1e-3));
    float shore_foam = foam_t * foam_t * params.foam_intensity;
    if (water_whitecaps())
    {
        s.foam = saturate(WATER_WHITECAP(i, dx, dy) * params.wind_gust.z);
    }
    // Foam is lit only where some of it shows.
    bool foamy = shore_foam > 0.0 || s.foam > 0.0;
#ifdef WATER_SEA
    foamy = foamy || water_sea_surf > 0.0 || water_sea_wake > 0.0;
#endif
    float3 foam = foamy ? water_lambert(s.normal, shadow, s.day) : (float3)(0.0);
    s.below = lerp(below, foam, shore_foam);
    if (water_whitecaps())
    {
        s.below = lerp(s.below, foam, s.foam);
    }
#ifdef WATER_SEA
    // The surf's foam, over the whitecaps, and the wake's over both.
    if (water_sea_surf > 0.0)
    {
        s.below = lerp(s.below, foam, water_sea_surf);
        s.foam = 1.0 - (1.0 - s.foam) * (1.0 - water_sea_surf);
    }
    if (water_sea_wake > 0.0)
    {
        float3 wake_foam = WATER_WAKE_FOAM_LIGHT * lerp(foam, (float3)dot(foam, float3(0.2126, 0.7152, 0.0722)),
                                                        WATER_WAKE_FOAM_GRAY);
        s.below = lerp(s.below, wake_foam, water_sea_wake);
        s.foam = 1.0 - (1.0 - s.foam) * (1.0 - water_sea_wake);
    }
#endif
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
float3 water_environment(float3 world_pos, float3 r, float alpha, WaterDaylight day)
{
    float horizon = saturate(r.y * 0.5 + 0.5);
    float3 gradient = lerp(float3(0.55, 0.62, 0.7), float3(0.25, 0.45, 0.7), horizon);
    // Blurrier water reads a coarser cube level.
    float3 env = transparent_environment(world_pos, r, saturate(sqrt(alpha)), gradient);
#ifdef WATER_SEA_DAYLIGHT
    // On a planet's sea away from the camera, the sky over the point.
    if (day.reach > 0.0)
    {
        env = lerp(env, daylight_point_sky(day, r), day.reach);
    }
#endif
    return env;
}

// The specular lobe of a light toward `sun_dir` of `sun_color` off this
// fragment's wave normal at GGX `alpha`: GGX with a height-correlated Smith
// visibility term and the water-vs-air F0. A high sun draws a compact disc; a
// low one spreads the same lobe into a path running toward the camera, since
// the wave slopes that mirror it span more of the surface. Zero for a light of
// no color, as when the world declares no directional light.
float3 water_glint(float3 normal, float3 view_dir, float alpha, float3 sun_dir, float3 sun_color)
{
    float n_dot_l = dot(normal, sun_dir);
    if (n_dot_l <= 0.0)
    {
        return (float3)(0.0);
    }
    float3 h = normalize(sun_dir + view_dir);
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
    return sun_color * (d * vis * f * n_dot_l);
}

// The glint of the first directional light at `pos`; on a planet's sea, of
// each body, lit through the air above the point.
float3 water_sun_glint(float3 normal, float3 view_dir, float alpha, WaterDaylight day)
{
    float3 glint = water_glint(normal, view_dir, alpha, view.sun_dir.xyz, view.sun_color.rgb);
#ifdef WATER_SEA_DAYLIGHT
    if (day.reach > 0.0)
    {
        float3 lit = (float3)(0.0);
        [unroll]
        for (uint b = 0u; b < DAYLIGHT_BODIES; b++)
        {
            if (daylight_body(b))
            {
                lit += water_glint(normal, view_dir, alpha, DAYLIGHT.body_dir[b].xyz, day.light[b]);
            }
        }
        glint = lerp(glint, lit, day.reach);
    }
#endif
    return glint;
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

// The share of the surface's color its reflection takes before whitecaps.
float water_fresnel(WaterSurfacePoint s, float3 view_dir)
{
    float n_dot_v = saturate(dot(s.normal, view_dir));
    return 0.02 + 0.98 * pow(1.0 - n_dot_v, max(params.fresnel_power, 1e-3));
}

// The whitecap cover that hides the reflection and the glint, 0 where this
// surface draws none.
float water_cover(WaterSurfacePoint s)
{
#ifdef WATER_SEA
    bool covered = water_whitecaps() || water_sea_surf > 0.0 || water_sea_wake > 0.0;
#else
    bool covered = water_whitecaps();
#endif
    return covered ? s.foam : 0.0;
}

TransparentOut water_resolve(WaterSurfacePoint s, float3 view_dir, float3 reflection)
{
    float fresnel = water_fresnel(s, view_dir);
    float3 glint = water_sun_glint(s.normal, view_dir, s.alpha, s.day);
    float3 color = lerp(s.below, reflection, fresnel) + glint;
    // Whitecaps lie on top of the water: what they cover neither mirrors nor
    // glints.
    color = lerp(color, s.below, water_cover(s));
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

// The on-screen reflection weight below which the trace is not worth a ray:
// head-on water reflects 2%, so this skips most of a view looking down.
static const float WATER_TRACE_MIN_WEIGHT = 0.04;
// The GGX alpha above which the prefiltered environment is the right answer.
static const float WATER_TRACE_MAX_ALPHA = 0.25;

// Whether the reflection at `s` shows enough, sharply enough, to trace.
bool water_trace_worth(WaterSurfacePoint s, float3 view_dir)
{
    float weight = water_fresnel(s, view_dir) * (1.0 - water_cover(s));
    return params.planar.z > 0.5 && weight >= WATER_TRACE_MIN_WEIGHT
        && s.alpha <= WATER_TRACE_MAX_ALPHA;
}

[shader("pixel")]
TransparentOut water_rt_fragment(WaterVertexOut i)
{
    if (water_occluded(i))
    {
        discard;
    }
    WATER_STEP dx, dy;
    water_footprint(i.position.xy, dx, dy);
    float3 view_dir = normalize(view.camera_pos.xyz - i.world_pos);
    WaterSurfacePoint s = water_surface(i, dx, dy);

    float3 reflection;
    if (params.planar.x > 0.5)
    {
        reflection = water_planar_reflection(s);
    }
    // No mirror plane for this surface, so trace instead where it shows. The
    // per-fragment wave normal makes `R` vary across the surface, so the traced
    // reflection follows the waves rather than mirroring one flat plane, and
    // the ripples hide the hit's sun shadow, so it casts none.
    else
    {
        float3 r = reflect(-view_dir, s.normal);
        if (!water_trace_worth(s, view_dir)
            || !rt_trace_reflection(i.world_pos + s.normal * 0.02, r,
                                    view.prefilter_mip_count > 0.5,
                                    view.prefilter_mip_count - 1.0, false, reflection))
        {
            reflection = water_environment(i.world_pos, r, s.alpha, s.day);
        }
    }
    return water_resolve(s, view_dir, reflection);
}

#else

[shader("pixel")]
TransparentOut water_fragment(WaterVertexOut i)
{
    if (water_occluded(i))
    {
        discard;
    }
    WATER_STEP dx, dy;
    water_footprint(i.position.xy, dx, dy);
    float3 view_dir = normalize(view.camera_pos.xyz - i.world_pos);
    WaterSurfacePoint s = water_surface(i, dx, dy);

    float3 reflection;
    if (params.planar.x > 0.5)
    {
        reflection = water_planar_reflection(s);
    }
    else
    {
        reflection = water_environment(i.world_pos, reflect(-view_dir, s.normal), s.alpha, s.day);
    }
    return water_resolve(s, view_dir, reflection);
}

#endif
