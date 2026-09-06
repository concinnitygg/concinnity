// Declarations shared by the fullscreen post passes, spliced into a shader at
// its POST_COMMON marker (see slang_source.rs). Not a standalone program: it
// declares no entry point and no resources, so it never appears in a backend's
// program table. Nothing here may spell the marker itself -- the splice would
// reintroduce it into the assembled source.

// Dimensions of a combined texture-sampler, as out parameters.
//
// slangc mis-lowers GetDimensions on a combined type for HLSL: it splits the
// type and then passes the sampler as the first argument
// (`tex.GetDimensions(samp, w, h)`), which no HLSL overload takes, so the DXIL
// leg fails in dxc. Spelling the call here fixes the argument list. A split
// pair occupies two operand slots, so `$0` is the texture, `$1` the sampler it
// dropped, and the out parameters land at `$2` and `$3`.
void combined_dims(Sampler2D<float4> s, out uint w, out uint h)
{
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "$0.GetDimensions($2, $3)";
    default:
        s.GetDimensions(w, h);
    }
}

// Texture dimensions of a combined texture-sampler, as floats.
float2 combined_size(Sampler2D<float4> s)
{
    uint w, h;
    combined_dims(s, w, h);
    return float2(float(w), float(h));
}
