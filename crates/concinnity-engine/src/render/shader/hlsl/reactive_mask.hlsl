// What the passes that write the reactive mask store, and how a reader tells
// the disocclusion mark from the layered writers.
//
// A value is the share of a pixel's color that does not follow the motion
// vector under it; 1 tells a temporal pass to trust this frame over its
// history. The particle and transparent passes stop at `REACTIVE_WRITE_MAX`,
// since a pixel that keeps no history aliases, and the readers that cannot cap
// the mask themselves read it as written. The disocclusion mark alone stores
// the full 1: an uncovered pixel has no history worth keeping.

static const float REACTIVE_WRITE_MAX = 0.9;

// Past this a stored value is the disocclusion mark, not a capped layer; it
// sits midway so the 8-bit store of either stays on its side.
static const float REACTIVE_DISOCCLUDED_ABOVE = 0.95;

float reactive_write(float share)
{
    return min(saturate(share), REACTIVE_WRITE_MAX);
}

float reactive_luminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}
