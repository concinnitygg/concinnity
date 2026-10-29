// Writing radiance into a half-float target: the largest value a channel
// keeps, and the cap that scales a brighter color down whole so it keeps its
// hue. Mirrored in `render::radiance_storage`.
#ifndef HDR_STORE_SPLICED
#define HDR_STORE_SPLICED

static const float HDR_STORE_MAX = 60000.0;

float3 hdr_store_cap(float3 c)
{
    float peak = max(max(c.r, c.g), c.b);
    if (peak <= HDR_STORE_MAX)
    {
        return c;
    }
    return isinf(peak) ? min(c, (float3)HDR_STORE_MAX) : c * (HDR_STORE_MAX / peak);
}

#endif
