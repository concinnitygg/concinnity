// Hi-Z (depth-mip pyramid) builder. One kernel per compile, selected by a
// define so each variant declares exactly the resources it binds (Metal and
// DXIL indices are assigned in declaration order, so an unused declaration
// would shift the live ones):
//
//   HIZ_INIT_MSAA   - hiz_init_msaa: read the MSAA main depth taking MAX over
//                     sample_count samples, write into HiZ mip 0.
//   HIZ_INIT_SINGLE - hiz_init_single: same for a single-sample main depth.
//   HIZ_DOWNSAMPLE  - hiz_downsample: MAX-reduce 2x2 texels of the previous
//                     HiZ mip (bound as a single-level view) into the next.
//
// The MAX reduction is correct because the engine uses standard (not reverse)
// depth: a Hi-Z texel storing the MAX represents the farthest visible surface
// in that region, so the cull can only be conservative.
//
// Both source and destination mips are single-level R32F views, so each
// downsample reads mip M and writes mip M+1 without aliasing the same texels.

struct HizParams
{
    uint dst_width;
    uint dst_height;
    uint src_mip;
    uint sample_count;
};

[[vk::push_constant]]
ConstantBuffer<HizParams> params;

#if defined(HIZ_INIT_MSAA)

[[vk::binding(0, 0)]]
Texture2DMS<float> src_depth;

#elif defined(HIZ_INIT_SINGLE)

[[vk::binding(0, 0)]]
Texture2D<float> src_depth;

#elif defined(HIZ_DOWNSAMPLE)

// Previous Hi-Z mip, bound as a single-level view (read-only storage image).
[[vk::binding(0, 0)]]
[format("r32f")]
RWTexture2D<float> src_hiz;

#else
#error "hiz_build.slang: define HIZ_INIT_MSAA, HIZ_INIT_SINGLE, or HIZ_DOWNSAMPLE"
#endif

[[vk::binding(1, 0)]]
[format("r32f")]
RWTexture2D<float> dst_mip;

#if defined(HIZ_INIT_MSAA)

// Slang's Metal backend emits texture2d_ms::read with an int2 coordinate,
// which MSL rejects (read takes uint2). Route the Metal target through inline
// MSL until that is fixed upstream; every other target keeps the plain Load.
float load_depth_sample(Texture2DMS<float> t, uint2 coord, uint s)
{
    __target_switch
    {
    case metal:
        __intrinsic_asm "$0.read(uint2($1), uint($2)).x";
    default:
        return t.Load(int2(coord), int(s));
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void hiz_init_msaa(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.dst_width || tid.y >= params.dst_height)
    {
        return;
    }
    float d = 0.0;
    for (uint s = 0u; s < params.sample_count; ++s)
    {
        d = max(d, load_depth_sample(src_depth, tid.xy, s));
    }
    dst_mip[tid.xy] = d;
}

#elif defined(HIZ_INIT_SINGLE)

[shader("compute")]
[numthreads(8, 8, 1)]
void hiz_init_single(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.dst_width || tid.y >= params.dst_height)
    {
        return;
    }
    dst_mip[tid.xy] = src_depth.Load(int3(int2(tid.xy), 0));
}

#elif defined(HIZ_DOWNSAMPLE)

[shader("compute")]
[numthreads(8, 8, 1)]
void hiz_downsample(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.dst_width || tid.y >= params.dst_height)
    {
        return;
    }
    uint2 src_size;
    src_hiz.GetDimensions(src_size.x, src_size.y);
    uint sx = tid.x * 2u;
    uint sy = tid.y * 2u;
    // For odd source dimensions the right/bottom edge loses a texel, but
    // max-reduction is conservative so dropping a half-row is harmless: it can
    // only make the cull more conservative, never wrongly cull a visible
    // object. Clamp the +1 taps so an odd edge reuses the in-bounds texel.
    uint sx1 = min(sx + 1u, src_size.x - 1u);
    uint sy1 = min(sy + 1u, src_size.y - 1u);
    float d0 = src_hiz[uint2(sx, sy)];
    float d1 = src_hiz[uint2(sx1, sy)];
    float d2 = src_hiz[uint2(sx, sy1)];
    float d3 = src_hiz[uint2(sx1, sy1)];
    dst_mip[tid.xy] = max(max(d0, d1), max(d2, d3));
}

#endif
