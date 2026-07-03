Texture2D    atlas         : register(t0);
SamplerState text_sampler  : register(s0);

struct PsIn
{
    float4 sv_pos : SV_POSITION;
    float2 uv     : TEXCOORD0;
    float3 color  : TEXCOORD1;
    float  mode   : TEXCOORD2;
};

float4 main(PsIn p) : SV_TARGET
{
    // A negative u marks a solid background-box vertex (a TextLabel.background
    // quad emitted by `gfx::text::build_text_calls`): emit the colour directly,
    // alpha carried through in v, no atlas sample. Mirrors the Metal text
    // fragment shader.
    if (p.uv.x < 0.0)
    {
        return float4(p.color, p.uv.y);
    }
    // A positive mode marks a textured quad (a Sprite with a texture): the
    // bound atlas is the sprite's own RGBA image, tinted by the vertex colour
    // with the mode value as the quad's alpha multiplier. Mirrors Metal.
    if (p.mode > 0.0)
    {
        float4 tex = atlas.Sample(text_sampler, p.uv);
        return float4(tex.rgb * p.color, tex.a * p.mode);
    }
    // Atlas stores a signed distance field: 0.5 = edge, >0.5 = inside, <0.5 = outside.
    // fwidth gives the screen-space derivative of d, so the smoothstep spans exactly
    // one screen pixel regardless of text scale or display density.
    float d = atlas.Sample(text_sampler, p.uv).r;
    float aa = fwidth(d);
    float a = smoothstep(0.5 - aa, 0.5 + aa, d);
    return float4(p.color, a);
}
