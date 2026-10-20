// HIZ_TEST marker: the pyramid half of a Hi-Z occlusion test. A screen rect,
// already clipped to the viewport, is tested against the farthest occluder the
// pyramid holds over it, at the mip whose texels are about the rect's size.
// Conservative: any uncertain case returns false. The includer defines
// `float hiz_load(int3 texel_mip)` ahead of the splice and carries
// DEPTH_CONVENTION. Mirrored by `render::hiz_cull::Footprint`.

// Widest footprint per axis, in texels, the test gathers at the deepest mip of
// a pyramid shorter than the full chain: the last level of a 16384x16384
// target. A wider footprint is kept. Mirrored by
// `render::hiz_cull::MAX_GATHER_SPAN`.
static const uint HIZ_GATHER_MAX_SPAN = 16u;

// Whether everything in the NDC rect `ndc_min..ndc_max` (inside [-1, 1]) at or
// behind device depth `nearest_depth` is hidden, against a `mip_count`-deep
// pyramid over a `hiz_size` base.
bool hiz_rect_occluded(float2 ndc_min, float2 ndc_max, float nearest_depth, float2 hiz_size,
                       uint mip_count)
{
    // A bound whose nearest point falls outside the [0, 1] depth range
    // crosses the near or far plane, so it is conservatively kept.
    if (nearest_depth < 0.0 || nearest_depth > 1.0)
    {
        return false;
    }
    // Map NDC -> UV (y flips because NDC y is up, UV v is down).
    float2 uv_min = float2(ndc_min.x * 0.5 + 0.5, 0.5 - ndc_max.y * 0.5);
    float2 uv_max = float2(ndc_max.x * 0.5 + 0.5, 0.5 - ndc_min.y * 0.5);
    // Size of the rect at mip 0, in texels.
    float2 size_tex = (uv_max - uv_min) * hiz_size;
    float max_dim = max(size_tex.x, size_tex.y);
    // Pick the mip whose texels are roughly the rect size so a 2x2 footprint
    // covers the rect (the standard Hi-Z 4-tap pattern).
    int mip = int(ceil(log2(max(max_dim, 1.0))));
    mip = clamp(mip, 0, int(mip_count) - 1);
    float2 mip_dim = max(hiz_size / float(1u << uint(mip)), float2(1.0, 1.0));
    int2 lo = int2(floor(uv_min * mip_dim));
    int2 hi = int2(floor(uv_max * mip_dim));
    int2 max_xy = int2(mip_dim) - int2(1, 1);
    lo = clamp(lo, int2(0, 0), max_xy);
    hi = clamp(hi, int2(0, 0), max_xy);
    int2 span = hi - lo + int2(1, 1);
    float occluder_depth;
    if (all(span <= int2(2, 2)))
    {
        float d0 = hiz_load(int3(lo.x, lo.y, mip));
        float d1 = hiz_load(int3(hi.x, lo.y, mip));
        float d2 = hiz_load(int3(lo.x, hi.y, mip));
        float d3 = hiz_load(int3(hi.x, hi.y, mip));
        occluder_depth = depth_farther(depth_farther(d0, d1), depth_farther(d2, d3));
    }
    else
    {
        // A pyramid shorter than the full chain can leave the rect wider than
        // 2x2 at its deepest mip, so read every texel it covers there.
        if (any(span > int2(HIZ_GATHER_MAX_SPAN, HIZ_GATHER_MAX_SPAN)))
        {
            return false;
        }
        occluder_depth = DEPTH_NEAR;
        [loop] for (int y = 0; y < int(HIZ_GATHER_MAX_SPAN); ++y)
        {
            if (y >= span.y)
            {
                break;
            }
            [loop] for (int x = 0; x < int(HIZ_GATHER_MAX_SPAN); ++x)
            {
                if (x >= span.x)
                {
                    break;
                }
                occluder_depth = depth_farther(occluder_depth, hiz_load(int3(lo.x + x, lo.y + y, mip)));
            }
        }
    }
    // If the bound's nearest depth is strictly behind the farthest surface the
    // pyramid holds over the rect, the whole bound is hidden.
    return depth_behind(nearest_depth, occluder_depth);
}
