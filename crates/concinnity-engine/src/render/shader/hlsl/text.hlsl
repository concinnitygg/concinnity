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

// Window size in logical points, 16 B. Mirrors `TextUniforms` in
// gfx/render_types.rs. Vulkan pushes it; Metal and DirectX take the same
// declaration as a constant buffer at the register named here, which is also
// the Metal buffer index.
struct TextUniforms
{
    float win_width;
    float win_height;
    float _pad0;
    float _pad1;
};

[[vk::push_constant]] ConstantBuffer<TextUniforms> uni : register(b0);

// The glyph atlas, or a sprite's own RGBA image on the textured path.
[[vk::binding(0, 0)]] Texture2D<float4> atlas : register(t0);
[[vk::binding(1, 0)]] SamplerState atlas_sampler : register(s0);

// Matches `TextVertex` (32 B: pos at 0, uv at 8, color at 16, mode at 28),
// which the host input layouts declare and `text_vertex_layout_matches_shaders`
// pins.
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

// Vulkan is the exception rather than the rule: the composite pass rasterizes
// through a standard positive-height viewport there, so pixel (0,0) already
// maps to NDC (-1,-1) and the remap is linear. Metal and DirectX put NDC +1 at
// the top of the screen, so they flip.
float text_ndc_y(float pixel_y, float win_height)
{
#ifdef CN_BACKEND_VULKAN
    return (pixel_y / win_height) * 2.0 - 1.0;
#else
    return 1.0 - (pixel_y / win_height) * 2.0;
#endif
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

[shader("pixel")]
float4 text_fragment_main(TextVertexOut i) : SV_Target
{
    // A negative u marks a solid background-box vertex (a TextLabel.background
    // quad emitted by gfx::text::build_text_calls): emit the color directly,
    // alpha carried through in v, no atlas sample.
    if (i.uv.x < 0.0)
    {
        return float4(i.color, i.uv.y);
    }
    // A positive mode marks a textured quad (a Sprite with a texture): the
    // bound atlas is the sprite's own RGBA image, tinted by the vertex color
    // with the mode value as the quad's alpha multiplier.
    if (i.mode > 0.0)
    {
        float4 tex = atlas.Sample(atlas_sampler, i.uv);
        return float4(tex.rgb * i.color, tex.a * i.mode);
    }
    // The atlas stores a signed distance field: 0.5 = edge, > 0.5 = inside.
    // fwidth gives the screen-space derivative of d, so the smoothstep spans
    // exactly one screen pixel and the glyph stays crisp at any text scale or
    // display density. Sampling d directly as alpha ramps the whole distance
    // field and reads as blurry.
    float d = atlas.Sample(atlas_sampler, i.uv).r;
    float aa = fwidth(d);
    float a = smoothstep(0.5 - aa, 0.5 + aa, d);
    return float4(i.color, a);
}
