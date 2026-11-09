// Where this frame shows a surface that last frame's pixels did not: what a
// moving object uncovers. A temporal pass that judges its history by depth
// alone keeps the object's history there wherever the two depths agree, as
// they do where an object meets the ground. One fragment per compile,
// selected by a define:
//
//   DISOCCLUSION_MARK     - the reactive mask's value for each pixel: full
//                           where the motion vectors carry the pixel onto
//                           none of last frame's surfaces near it, else none.
//                           A surface is told from another by its view normal
//                           or its motion: the uncovered ground faces and moves
//                           unlike the object that hid it, while a surface that
//                           only turns, or a camera that only starts moving,
//                           changes one and keeps the other.
//   DISOCCLUSION_SNAPSHOT - the G-buffer's view normal and the motion vectors,
//                           kept for the next frame's mark.
//
// Pairs with `fullscreen_vertex` in fullscreen.hlsl. Every read is a point
// load by pixel position; the register numbers are the Metal indices and the
// D3D root signature's slots alike, and Vulkan binds a variant's n textures at
// 0..n-1 and their samplers at n..2n-1.

{TEXTURE_SIZE}

// A snapshot texel with no surface: its motion lies past every motion vector
// the G-buffer stores (they clamp to 2), so nothing matches it.
static const float DISOCCLUSION_NO_SURFACE = 4.0;

// A unit vector folded onto the octahedron, in [-1, 1]^2.
float2 disocclusion_oct_encode(float3 n)
{
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    if (n.z < 0.0)
    {
        float2 fold = (1.0 - abs(n.yx)) * select(n.xy >= 0.0, (float2)(1.0), (float2)(-1.0));
        return fold;
    }
    return n.xy;
}

float3 disocclusion_oct_decode(float2 e)
{
    float3 n = float3(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0)
    {
        n.xy = (1.0 - abs(n.yx)) * select(n.xy >= 0.0, (float2)(1.0), (float2)(-1.0));
    }
    return normalize(n);
}

#if defined(DISOCCLUSION_MARK)

{REACTIVE_MASK}

// binding 0 = the G-buffer (rgb = view normal, a = linear view depth); 1 = the
// motion vectors; 2 = last frame's snapshot.
[[vk::binding(0, 0)]] Texture2D<float4> gbuffer : register(t0);
[[vk::binding(3, 0)]] SamplerState gbuffer_samp : register(s0);
[[vk::binding(1, 0)]] Texture2D<float4> velocity : register(t1);
[[vk::binding(4, 0)]] SamplerState velocity_samp : register(s1);
[[vk::binding(2, 0)]] Texture2D<float4> previous : register(t2);
[[vk::binding(5, 0)]] SamplerState previous_samp : register(s2);

// Least cosine between last frame's normal and this pixel's at which it may
// be the same surface.
static const float DISOCCLUSION_NORMAL_TOLERANCE = 0.9;
// Motion difference, in pixels, within which it may be the same surface: a
// pixel's worth, plus a share of the motion itself for a camera that
// accelerates.
static const float DISOCCLUSION_MOTION_SLACK = 0.5;
static const float DISOCCLUSION_MOTION_SHARE = 0.25;

// Whether last frame's texel `q` may have shown the surface facing `normal`
// and moving by `motion` pixels.
bool disocclusion_same_surface(int2 q, float3 normal, float2 motion, float2 size)
{
    float4 last = previous.Load(int3(q, 0));
    if (last.z >= DISOCCLUSION_NO_SURFACE)
    {
        return false;
    }
    float2 last_motion = last.zw * size;
    float slack = DISOCCLUSION_MOTION_SLACK
        + DISOCCLUSION_MOTION_SHARE * max(length(motion), length(last_motion));
    return dot(disocclusion_oct_decode(last.xy), normal) >= DISOCCLUSION_NORMAL_TOLERANCE
        || length(motion - last_motion) <= slack;
}

[shader("pixel")]
float4 disocclusion_mark_fragment(
    [[vk::location(0)]] float2 vertex_uv : TEXCOORD0,
    float4 pixel : SV_Position) : SV_Target
{
    int2 p = int2(pixel.xy);
    float4 now = gbuffer.Load(int3(p, 0));
    if (now.a <= 0.0)
    {
        return (float4)(0.0);
    }
    int2 size = int2(texture_size(gbuffer));
    float2 v = velocity.Load(int3(p, 0)).rg;
    float2 prev_uv = (float2(p) + 0.5) / float2(size) + v;
    // Off screen last frame, the reader has no history here to keep.
    if (any(prev_uv < 0.0) || any(prev_uv > 1.0))
    {
        return (float4)(0.0);
    }
    // The 3x3 around where the pixel was absorbs the jitter's sub-pixel shift
    // of every edge, so only an uncovered area wider than that is marked.
    float3 normal = normalize(now.xyz);
    float2 motion = v * float2(size);
    int2 c = int2(prev_uv * float2(size));
    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            int2 q = clamp(c + int2(x, y), 0, size - 1);
            if (disocclusion_same_surface(q, normal, motion, float2(size)))
            {
                return (float4)(0.0);
            }
        }
    }
    return float4(reactive_write(1.0), 0.0, 0.0, 0.0);
}

#elif defined(DISOCCLUSION_SNAPSHOT)

// binding 0 = the G-buffer; 1 = the motion vectors.
[[vk::binding(0, 0)]] Texture2D<float4> gbuffer : register(t0);
[[vk::binding(2, 0)]] SamplerState gbuffer_samp : register(s0);
[[vk::binding(1, 0)]] Texture2D<float4> velocity : register(t1);
[[vk::binding(3, 0)]] SamplerState velocity_samp : register(s1);

// rg = the octahedral view normal, ba = the motion vector, or
// `DISOCCLUSION_NO_SURFACE` in b where there is no surface.
[shader("pixel")]
float4 disocclusion_snapshot_fragment(
    [[vk::location(0)]] float2 vertex_uv : TEXCOORD0,
    float4 pixel : SV_Position) : SV_Target
{
    int3 p = int3(int2(pixel.xy), 0);
    float4 now = gbuffer.Load(p);
    if (now.a <= 0.0)
    {
        return float4(0.0, 0.0, DISOCCLUSION_NO_SURFACE, 0.0);
    }
    return float4(disocclusion_oct_encode(normalize(now.xyz)), velocity.Load(p).rg);
}

#else
#error "disocclusion.hlsl: define DISOCCLUSION_MARK or DISOCCLUSION_SNAPSHOT"
#endif
