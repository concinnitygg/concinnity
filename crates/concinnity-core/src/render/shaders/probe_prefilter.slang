// Reflection-probe prefilter: the runtime GGX convolution that turns a freshly
// captured cube into the prefiltered radiance mip chain the specular term
// samples. One kernel per compile, selected by a define so each variant
// declares exactly the resources it binds (Metal and DXIL indices are assigned
// in declaration order, so an unused declaration would shift the live ones):
//
//   PROBE_MIP0       - probe_mip0: firefly-clamped copy of the capture into
//                      mip 0 of the probe cube, the mirror mip.
//   PROBE_DOWNSAMPLE - probe_downsample: 2x2 box reduction of the capture
//                      cube's mip M into mip M+1, building the pyramid the GGX
//                      kernel selects a level from.
//   PROBE_GGX        - probe_ggx: importance-sampled GGX convolution of the
//                      capture pyramid into one mip of the probe cube.
//
// These mirror the CPU convolution in `bake::environment_map`, which still runs
// at build time for imported HDR environment maps: same Hammersley sequence,
// same GGX importance sample, same split-sum V = R = N, same N.L-weighted
// average, same firefly clamp, and the roughness per mip comes from the same
// `prefilter_roughness`. The deliberate difference is where a sample's radiance
// comes from. The CPU kernel taps the full-resolution capture every time; this
// one taps the pyramid at the level whose texel solid angle matches the
// sample's, so a small bright source cannot alias between samples at high
// roughness (Karis, Real Shading in Unreal Engine 4).
//
// That is also why the clamp applies while the pyramid is built: a blown texel's
// energy would otherwise survive the box average and reappear through a coarse
// tap. Clamping is monotone -- a level whose texels are all under the threshold
// cannot produce one over it -- so no level past the first is changed by it.
//
// Source and destination are always different mips, bound as single-level views,
// so no dispatch reads the texels it writes.

static const float PI = 3.14159265358979323846;

struct ProbePrefilterParams
{
    // Destination face edge in texels.
    uint dst_size;
    // Source cube face edge at mip 0, in texels.
    uint src_size;
    // GGX samples per output texel.
    uint sample_count;
    // Source mip the downsample kernel reduces (writes src_mip + 1).
    uint src_mip;
    // GGX roughness of the destination mip.
    float roughness;
    // Firefly clamp luminance; <= 0 disables the cap.
    float clamp_lum;
    // Mip levels the source pyramid has, so the solid-angle lod stays in range.
    float src_mip_count;
    float _pad;
};

// The params slot is a host difference, not a target one: Vulkan pushes them,
// DirectX hands them over as root constants at b0 (where a bare push constant
// lands on that target), and the Metal encoder writes them to buffer(0), which
// is where this declaration lands as the kernel's first buffer.
#ifdef METAL_BINDINGS
ConstantBuffer<ProbePrefilterParams> params;
#else
[[vk::push_constant]] ConstantBuffer<ProbePrefilterParams> params;
#endif

// Direction a face texel at (u, v) in [-1, 1] looks along, not normalised.
// Mirrors `gfx::cubemap::face_dir`, which is also the hardware cube convention,
// so a face rendered into cube slice `face` is the face a sampler finds there.
float3 probe_face_dir(uint face, float u, float v)
{
    switch (face)
    {
    case 0u:
        return float3(1.0, -v, -u);
    case 1u:
        return float3(-1.0, -v, u);
    case 2u:
        return float3(u, 1.0, v);
    case 3u:
        return float3(u, -1.0, -v);
    case 4u:
        return float3(u, -v, 1.0);
    default:
        return float3(-u, -v, -1.0);
    }
}

// Unit direction for texel `xy` of a `size` square face. Mirrors
// `gfx::cubemap::texel_dir`.
float3 probe_texel_dir(uint face, uint2 xy, uint size)
{
    float u = (float(xy.x) + 0.5) / float(size) * 2.0 - 1.0;
    float v = (float(xy.y) + 0.5) / float(size) * 2.0 - 1.0;
    return normalize(probe_face_dir(face, u, v));
}

// Cap a radiance so one very bright texel cannot dominate a glossy mip, scaling
// RGB uniformly to keep its hue. Mirrors `clamp_radiance` in the CPU bake.
float3 probe_clamp_radiance(float3 rgb, float clamp_lum)
{
    if (clamp_lum <= 0.0)
    {
        return rgb;
    }
    float lum = dot(rgb, float3(0.2126, 0.7152, 0.0722));
    return lum > clamp_lum ? rgb * (clamp_lum / lum) : rgb;
}

#if defined(PROBE_MIP0)

[[vk::binding(0, 0)]]
[format("rgba16f")]
RWTexture2DArray<float4> src_mip;

[[vk::binding(1, 0)]]
[format("rgba16f")]
RWTexture2DArray<float4> dst_mip;

// Mip 0 of the probe cube: the capture copied through, firefly-clamped.
//
// A probe cube is sampled only by the specular term (never drawn as a skybox),
// so the mirror mip is capped too: it suppresses a lone blown highlight aliasing
// into a bright square on a near-mirror surface that falls back to the probe.
// The CPU counterpart is `prefilter_mip0` with `clamp_mip0` set.
[shader("compute")]
[numthreads(8, 8, 1)]
void probe_mip0(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.dst_size || tid.y >= params.dst_size)
    {
        return;
    }
    float3 rgb = src_mip[uint3(tid.xy, tid.z)].rgb;
    dst_mip[uint3(tid.xy, tid.z)] = float4(probe_clamp_radiance(rgb, params.clamp_lum), 1.0);
}

#elif defined(PROBE_DOWNSAMPLE)

[[vk::binding(0, 0)]]
[format("rgba16f")]
RWTexture2DArray<float4> src_mip;

[[vk::binding(1, 0)]]
[format("rgba16f")]
RWTexture2DArray<float4> dst_mip;

// One level of the capture pyramid: the 2x2 box average of the level above,
// each tap clamped first so a firefly cannot smuggle its energy down the chain.
// Per-face, so the corner texels average slightly fewer neighbours than a
// seamless filter would; the GGX kernel taps the finished pyramid through a cube
// sampler, which is where cross-face continuity actually matters.
[shader("compute")]
[numthreads(8, 8, 1)]
void probe_downsample(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.dst_size || tid.y >= params.dst_size)
    {
        return;
    }
    uint2 s = tid.xy * 2u;
    float3 sum = probe_clamp_radiance(src_mip[uint3(s, tid.z)].rgb, params.clamp_lum)
               + probe_clamp_radiance(src_mip[uint3(s + uint2(1u, 0u), tid.z)].rgb, params.clamp_lum)
               + probe_clamp_radiance(src_mip[uint3(s + uint2(0u, 1u), tid.z)].rgb, params.clamp_lum)
               + probe_clamp_radiance(src_mip[uint3(s + uint2(1u, 1u), tid.z)].rgb, params.clamp_lum);
    dst_mip[uint3(tid.xy, tid.z)] = float4(sum * 0.25, 1.0);
}

#elif defined(PROBE_GGX)

[[vk::binding(0, 0)]]
TextureCube<float4> src_cube;

[[vk::binding(1, 0)]]
SamplerState src_sampler;

[[vk::binding(2, 0)]]
[format("rgba16f")]
RWTexture2DArray<float4> dst_mip;

// Hammersley quasi-random point `i` of `n`. The van der Corput term is the bit
// reversal of `i`; the CPU bake spells that reversal out by hand (rotate, then
// swap bytes, nibbles, pairs, bits), which is the same permutation.
float2 probe_hammersley(uint i, uint n)
{
    return float2(float(i) / float(n), float(reversebits(i)) * 2.3283064365386963e-10);
}

// Orthonormal basis around `n`, matching `make_tbn` in the CPU bake.
void probe_tbn(float3 n, out float3 t, out float3 b)
{
    float3 up = abs(n.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    t = normalize(cross(up, n));
    b = cross(n, t);
}

// GGX half-vector for Hammersley point `xi` around `n`. `a2m1` is a^2 - 1, held
// constant across a texel's sample set, exactly as `importance_sample_ggx` does.
float3 probe_importance_sample_ggx(float2 xi, float3 n, float3 t, float3 b, float a2m1)
{
    float phi = 2.0 * PI * xi.x;
    float cos_theta = sqrt((1.0 - xi.y) / (1.0 + a2m1 * xi.y));
    float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    float3 h_local = float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    return normalize(t * h_local.x + b * h_local.y + n * h_local.z);
}

// GGX convolution of the capture pyramid into one mip of the probe cube.
//
// Split-sum approximation: V = R = N, so a sampled half-vector H gives the light
// direction L = 2 (N.H) H - N and the samples average weighted by N.L. Each
// sample reads the pyramid level whose texel solid angle matches the sample's
// own (`sa_sample / sa_texel`), which is what keeps a sparse sample set from
// aliasing a small bright source across a wide lobe.
[shader("compute")]
[numthreads(8, 8, 1)]
void probe_ggx(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.dst_size || tid.y >= params.dst_size)
    {
        return;
    }
    float3 n = probe_texel_dir(tid.z, tid.xy, params.dst_size);
    float3 t;
    float3 b;
    probe_tbn(n, t, b);

    float a = params.roughness * params.roughness;
    float a2 = a * a;
    // Solid angle of one source texel: the sphere split evenly over six faces.
    float sa_texel = 4.0 * PI / (6.0 * float(params.src_size) * float(params.src_size));
    float max_lod = max(params.src_mip_count - 1.0, 0.0);

    float3 accum = float3(0.0);
    float total_weight = 0.0;
    for (uint i = 0u; i < params.sample_count; ++i)
    {
        float3 h = probe_importance_sample_ggx(
            probe_hammersley(i, params.sample_count), n, t, b, a2 - 1.0);
        float ndh = dot(n, h);
        if (ndh <= 0.0)
        {
            continue;
        }
        float3 l = normalize(2.0 * ndh * h - n);
        float ndl = max(dot(n, l), 0.0);
        if (ndl <= 0.0)
        {
            continue;
        }
        // With V = N the GGX sample pdf is D / 4 (the 4 (V.H) Jacobian cancels
        // against N.H). Spreading `sample_count` samples over the lobe gives each
        // one that much solid angle to represent.
        float d = ndh * ndh * (a2 - 1.0) + 1.0;
        float pdf = a2 / (PI * d * d * 4.0);
        float sa_sample = 1.0 / (float(params.sample_count) * pdf + 1e-4);
        float lod = clamp(0.5 * log2(sa_sample / sa_texel), 0.0, max_lod);
        float3 env = probe_clamp_radiance(src_cube.SampleLevel(src_sampler, l, lod).rgb,
                                          params.clamp_lum);
        accum += env * ndl;
        total_weight += ndl;
    }

    // No sample landed in the hemisphere (possible only at a tiny sample count):
    // fall back to the mirror direction, as the CPU kernel does.
    float3 rgb = total_weight > 0.0
                     ? accum / total_weight
                     : src_cube.SampleLevel(src_sampler, n, 0.0).rgb;
    dst_mip[uint3(tid.xy, tid.z)] = float4(rgb, 1.0);
}

#else
#error "probe_prefilter.slang: define PROBE_MIP0, PROBE_DOWNSAMPLE, or PROBE_GGX"
#endif
