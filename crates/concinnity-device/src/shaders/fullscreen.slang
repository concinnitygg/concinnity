// Fullscreen-triangle vertex stage, shared by every screen-space post pass.
//
// Three SV_VertexID-generated vertices cover the framebuffer, so these passes
// need no vertex buffer and no vertex descriptor. Pairing is by semantic:
// a post fragment declares `[[vk::location(0)]] float2 uv : TEXCOORD0` and
// slangc emits the matching `[[user(TEXCOORD)]]` varying on both sides, so the
// vertex compiles once and every fragment links against it.
//
// The Y flip is the one genuinely target-specific line, and Vulkan is the
// exception rather than the rule: it rasterises its HDR targets through a
// negative-height viewport, so the sampled image is already upright there and a
// plain [0,1] map is correct. Metal and DirectX both have a top-left texture
// origin and no such flip, so they map it here.

// `uv` leads deliberately. D3D packs a stage signature in declaration order and
// links the two stages by matching semantic *and* register, so with SV_Position
// first the vertex would hand TEXCOORD0 out on register 1 while a fragment that
// declares only `uv` reads it from register 0, and every PSO fails to create.
// Metal and Vulkan are order-blind here (both varyings carry an attribute or an
// explicit location), so the order costs them nothing.
struct FullscreenVertex
{
    [[vk::location(0)]] float2 uv : TEXCOORD0;
    float4 position : SV_Position;
};

float2 fullscreen_uv(float2 pos)
{
    __target_switch
    {
    case spirv:
        return (pos + 1.0) * 0.5;
    default:
        return float2((pos.x + 1.0) * 0.5, 1.0 - (pos.y + 1.0) * 0.5);
    }
}

[shader("vertex")]
FullscreenVertex fullscreen_vertex(uint vid : SV_VertexID)
{
    float2 pos = float2((vid == 2) ? 3.0 : -1.0, (vid == 1) ? 3.0 : -1.0);
    FullscreenVertex o;
    o.position = float4(pos, 0.0, 1.0);
    o.uv = fullscreen_uv(pos);
    return o;
}
