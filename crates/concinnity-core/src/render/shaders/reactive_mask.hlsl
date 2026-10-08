// What the particle and transparent passes write to the reactive mask.
//
// A value is the share of a pixel's color that does not follow the motion
// vector under it; 1 tells a temporal pass to trust this frame over its
// history. Writes stop at `REACTIVE_WRITE_MAX`, since a pixel that keeps no
// history aliases, and the readers that cannot cap the mask themselves read it
// as written.

static const float REACTIVE_WRITE_MAX = 0.9;

float reactive_write(float share)
{
    return min(saturate(share), REACTIVE_WRITE_MAX);
}

float reactive_luminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}
