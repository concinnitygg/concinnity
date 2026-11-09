// Declarations shared by the fullscreen post passes, spliced into a shader at
// its POST_COMMON marker (see shader_source.rs). Not a standalone program: it
// declares no entry point and no resources, so it never appears in a program
// table. Nothing here may spell the marker itself -- the splice would
// reintroduce it into the assembled source.

// The G-buffer pixel a reflection texel at `uv` traces from, as the UV of that
// pixel's center. At a reduced trace resolution a texel covers a block of
// pixels; this picks one real pixel of the block, and addressing its center
// makes a linear sampler return it unblended. At full resolution it is the
// pixel itself.
float2 reflection_source_uv(float2 uv, float2 gbuffer_size)
{
    float2 px = floor(uv * gbuffer_size - 0.25);
    return (px + 0.5) / gbuffer_size;
}
