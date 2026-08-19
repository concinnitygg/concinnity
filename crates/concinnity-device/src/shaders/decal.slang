// Projected (deferred) decal pass: single source for every backend.
//
// Runs after the main HDR pass. The vertex stage draws a unit cube (positions
// in [-0.5, 0.5]^3 in local space) under the decal's model matrix and the
// camera VP, jittered when TAA is on so the rasterised pixel grid matches the
// main pass exactly. The fragment stage reconstructs the world-space sample
// point of each rasterised pixel from the main pass's depth attachment,
// transforms it back into decal-local space, and discards where the point
// falls outside the unit box. Pixels near the top / bottom face along local +Y
// fade out so the stamp shows no hard edge where it meets a tilted or curved
// surface.
//
// The composited colour is alpha-blended into the resolved HDR target by the
// pipeline's blend state (Src.A * Src + (1 - Src.A) * Dst). The pass binds no
// depth attachment: the unit-box test is the depth test.
//
// USE_MSAA is a HOST difference rather than a target one: Vulkan reads the
// multisampled main depth while Metal reads the resolved copy.

#ifndef USE_MSAA
#define USE_MSAA 0
#endif

// Per-frame view inputs, 144 B. Mirrors `DecalView` in each backend's uniforms
// module.
struct DecalView
{
    // The main pass's view-projection (jittered when TAA is on).
    float4x4 vp;
    // Inverse of the same, for the world-space reconstruction.
    float4x4 inv_vp;
    // Attachment width, height in pixels.
    float2 viewport;
    float2 _pad;
};

// Per-decal uniforms, 160 B. Mirrors `DecalParams`.
struct DecalParams
{
    float4x4 model;     // local -> world
    float4x4 inv_model; // world -> local
    float4 tint;        // linear RGB x alpha
    float fade_pow;     // exponent on the soft fade along local +Y
    float _pad0;
    float _pad1;
    float _pad2;
};

// Vulkan binds both as UBOs in the per-frame set (the params through a
// dynamic offset into one MAX_DECALS-slot ring); Metal writes both with
// setBytes at the buffer indices the register numbers name.
[[vk::binding(0, 0)]] ConstantBuffer<DecalView> view : register(b0);
[[vk::binding(1, 0)]] ConstantBuffer<DecalParams> params : register(b1);

// Declaration order is the Metal texture index: the scene depth at 0 and this
// decal's albedo at 1, which is what that host binds. The albedo lives in its
// own Vulkan set so swapping decals rebinds set 1 rather than the per-frame set.
#if USE_MSAA
[[vk::binding(2, 0)]] Texture2DMS<float> scene_depth;
#else
[[vk::binding(2, 0)]] Texture2D<float> scene_depth;
#endif
[[vk::binding(0, 1)]] Sampler2D<float4> decal_tex;

struct DecalVertexIn
{
    // Unit cube vertex; the bound buffer carries 8 corners in [-0.5, 0.5]^3.
    [[vk::location(0)]] float3 pos : POSITION;
};

// The cube is only a rasterisation proxy, so the fragment derives everything it
// needs from the pixel position and carries no varying.
struct DecalVertexOut
{
    float4 position : SV_Position;
};

[shader("vertex")]
DecalVertexOut decal_vertex(DecalVertexIn v)
{
    DecalVertexOut o;
    float4 world = mul(params.model, float4(v.pos, 1.0));
    o.position = mul(view.vp, world);
    return o;
}

float decal_scene_depth(int2 pixel)
{
#if USE_MSAA
    return scene_depth.Load(pixel, 0);
#else
    return scene_depth.Load(int3(pixel, 0));
#endif
}

[shader("fragment")]
float4 decal_fragment(DecalVertexOut i) : SV_Target
{
    // The single-sample depth resolve shares the rasterised pixel grid with the
    // composite target, so integer texel coordinates address it directly.
    int2 pixel = int2(i.position.xy);
    if (pixel.x < 0 || pixel.y < 0 || pixel.x >= int(view.viewport.x)
        || pixel.y >= int(view.viewport.y))
    {
        discard;
    }
    // 1.0 is the cleared / "no geometry" sentinel: the main pass left this
    // pixel empty (the sky writes near-far-plane depth instead). Nothing to
    // project onto.
    float depth = decal_scene_depth(pixel);
    if (depth >= 1.0)
    {
        discard;
    }

    // Reconstruct world space at this pixel through the inverse VP. Every
    // backend rasterises this pass with the framebuffer's Y running downwards
    // (Vulkan through the same negative-height viewport as the main pass), so
    // the NDC flip is shared rather than target-specific.
    float2 ndc_xy = (i.position.xy / view.viewport) * 2.0 - 1.0;
    ndc_xy.y = -ndc_xy.y;
    float4 clip = float4(ndc_xy, depth, 1.0);
    float4 world = mul(view.inv_vp, clip);
    world /= world.w;

    // Decal-local clip against the unit box.
    float4 local = mul(params.inv_model, world);
    float3 ab = abs(local.xyz);
    if (ab.x > 0.5 || ab.y > 0.5 || ab.z > 0.5)
    {
        discard;
    }

    // Soft fade along the projection axis (local +Y) so the stamp shows no hard
    // band where the surface tilts away from the projection plane: alpha rolls
    // off as |local.y| approaches 0.5.
    float fade = saturate(1.0 - (ab.y * 2.0));
    fade = pow(fade, max(params.fade_pow, 1.0));

    // Sample on local X-Z; UV in [0, 1] with V = 0 at the top, matching the
    // rest of the engine's textures.
    float2 uv = local.xz + 0.5;
    uv.y = 1.0 - uv.y;
    float4 tex = decal_tex.Sample(uv);

    return float4(tex.rgb * params.tint.rgb, tex.a * params.tint.a * fade);
}
