// A texture's size without the out-parameter boilerplate, spliced into a shader
// at its TEXTURE_SIZE marker (see shader_source.rs). Not a standalone program.
// Nothing here may spell the marker itself.

// The top mip's size in texels, as float2 for UV math.
float2 texture_size(Texture2D<float4> t)
{
    uint w, h;
    t.GetDimensions(w, h);
    return float2(float(w), float(h));
}
