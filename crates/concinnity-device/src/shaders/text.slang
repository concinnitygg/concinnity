// Screen-space text and sprite overlay: single source for every backend.
//
// One alpha-blended quad per glyph, sprite or background box, drawn in the
// composite pass so it sits on top of the tonemapped image. Vertex positions
// arrive in logical pixels with the origin at the top left; the vertex stage
// maps them straight to NDC.
//
// The fragment stage carries three cases, selected per vertex rather than per
// pipeline: a negative u marks a solid background box, a positive `mode` marks
// a textured sprite quad whose `mode` is the alpha multiplier, and everything
// else samples the glyph atlas as a signed distance field.
//
// METAL_BINDINGS selects the Metal host's constant shape, which is a host
// difference rather than a target one: Vulkan pushes the window size while the
// Metal encoder writes it to vertex buffer(0).

// Window size in logical points, 16 B. Mirrors `TextUniforms` in
// gfx/render_types.rs.
struct TextUniforms
{
    float win_width;
    float win_height;
    float _pad0;
    float _pad1;
};

#ifdef METAL_BINDINGS
ConstantBuffer<TextUniforms> uni : register(b0);
#else
[[vk::push_constant]] ConstantBuffer<TextUniforms> uni;
#endif

// The glyph atlas, or a sprite's own RGBA image on the textured path. A
// combined sampler lowers to a texture(0) + sampler(0) pair on Metal, which is
// what that host binds.
[[vk::binding(0, 0)]] Sampler2D<float4> atlas;

// Matches `TextVertex` (32 B: pos at 0, uv at 8, colour at 16, mode at 28),
// which the host input layouts declare and `text_vertex_layout_matches_shaders`
// pins.
//
// `mode` takes a semantic of its own rather than TEXCOORD1: slangc appends its
// own index to whatever the semantic spells, so a trailing digit is multiplied
// by ten on the way out (TEXCOORD1 lands as TEXCOORD index 10). That is
// harmless between two stages that agree, which is why the varyings below keep
// their TEXCOORDn names, but a vertex input has to match the index a host input
// layout declares.
struct TextVertexIn
{
    [[vk::location(0)]] float2 pos : POSITION;
    [[vk::location(1)]] float2 uv : TEXCOORD0;
    [[vk::location(2)]] float3 color : COLOR;
    [[vk::location(3)]] float mode : MODE;
};

// The varyings lead deliberately: D3D packs a stage signature in declaration
// order and links the two stages by matching semantic *and* register, and this
// fragment reads no position at all.
struct TextVertexOut
{
    [[vk::location(0)]] float2 uv : TEXCOORD0;
    [[vk::location(1)]] float3 color : TEXCOORD1;
    [[vk::location(2)]] float mode : TEXCOORD2;
    float4 position : SV_Position;
};

// Vulkan is the exception rather than the rule, exactly as in `fullscreen.slang`:
// the composite pass rasterises through a standard positive-height viewport
// there, so pixel (0,0) already maps to NDC (-1,-1) and the remap is linear.
// Metal and DirectX put NDC +1 at the top of the screen, so they flip.
float text_ndc_y(float pixel_y, float win_height)
{
    __target_switch
    {
    case spirv:
        return (pixel_y / win_height) * 2.0 - 1.0;
    default:
        return 1.0 - (pixel_y / win_height) * 2.0;
    }
}

[shader("vertex")]
TextVertexOut text_vertex_main(TextVertexIn v)
{
    TextVertexOut o;
    o.position = float4(
        (v.pos.x / uni.win_width) * 2.0 - 1.0,
        text_ndc_y(v.pos.y, uni.win_height),
        0.0,
        1.0);
    o.uv = v.uv;
    o.color = v.color;
    o.mode = v.mode;
    return o;
}

[shader("fragment")]
float4 text_fragment_main(TextVertexOut i) : SV_Target
{
    // A negative u marks a solid background-box vertex (a TextLabel.background
    // quad emitted by gfx::text::build_text_calls): emit the colour directly,
    // alpha carried through in v, no atlas sample.
    if (i.uv.x < 0.0)
    {
        return float4(i.color, i.uv.y);
    }
    // A positive mode marks a textured quad (a Sprite with a texture): the
    // bound atlas is the sprite's own RGBA image, tinted by the vertex colour
    // with the mode value as the quad's alpha multiplier.
    if (i.mode > 0.0)
    {
        float4 tex = atlas.Sample(i.uv);
        return float4(tex.rgb * i.color, tex.a * i.mode);
    }
    // The atlas stores a signed distance field: 0.5 = edge, > 0.5 = inside.
    // fwidth gives the screen-space derivative of d, so the smoothstep spans
    // exactly one screen pixel and the glyph stays crisp at any text scale or
    // display density. Sampling d directly as alpha ramps the whole distance
    // field and reads as blurry.
    float d = atlas.Sample(i.uv).r;
    float aa = fwidth(d);
    float a = smoothstep(0.5 - aa, 0.5 + aa, d);
    return float4(i.color, a);
}
