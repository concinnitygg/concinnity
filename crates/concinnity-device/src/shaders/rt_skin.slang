// Compute skinning for ray tracing: single source for every backend.
//
// The main pass skins in the vertex shader, so no deformed-vertex buffer exists
// for the BVH to trace against. This kernel produces one: it reads the bind-pose
// skinned vertices + a per-object joint palette and writes posed (model-space)
// plain `Vertex`s into a shared deformed buffer, which the RT
// acceleration-structure build then traces. One dispatch per skinned object over
// its vertex range; the deformed buffer mirrors the skinned vertex buffer's
// indexing so the existing skinned index buffer addresses it directly.
//
// The mesh payloads are read and written as raw bytes rather than through
// `StructuredBuffer<T>`, because a structured-buffer `float3` does not lay out
// the same way on all three targets: Metal and DXIL pack it to 12 bytes, SPIR-V
// pads it to 16, which would stride `SkinnedVertex` at 96 instead of 80. Byte
// addressing has no layout rule to disagree over, so the one source reproduces
// the CPU strides everywhere. `mesh_payload_offsets_match_the_kernel` in
// `shader_layout` pins the constants below to the `#[repr(C)]` mirrors.

// Which slice of the shared buffers this dispatch deforms.
struct SkinParams
{
    // First vertex of this object in the shared buffers.
    uint vertex_base;
    // Vertices to deform this dispatch.
    uint vertex_count;
    // Palette size; joint indices are clamped below it.
    uint joint_count;
    // Morph targets in `morph_data`; 0 = no morphing.
    uint target_count;
};

// Byte stride and field offsets of `gfx::mesh_payload::SkinnedVertex`. `joints`
// is four u16s, read as the two uints at `SKINNED_JOINTS`.
static const uint SKINNED_STRIDE = 80;
static const uint SKINNED_POS = 0;
static const uint SKINNED_NORMAL = 12;
static const uint SKINNED_TANGENT = 24;
static const uint SKINNED_COLOR = 36;
static const uint SKINNED_UV = 48;
static const uint SKINNED_JOINTS = 56;
static const uint SKINNED_WEIGHTS = 64;

// Byte stride and field offsets of `gfx::mesh_payload::Vertex`, the layout the
// static RT vertex fetchers read the deformed buffer back at.
static const uint VERTEX_STRIDE = 56;
static const uint VERTEX_POS = 0;
static const uint VERTEX_NORMAL = 12;
static const uint VERTEX_TANGENT = 24;
static const uint VERTEX_COLOR = 36;
static const uint VERTEX_UV = 48;

// Byte stride and field offsets of `gfx::mesh_payload::MorphEntry`: one sparse
// morph delta naming its target, a bind-space position + normal offset scaled by
// that target's weight.
static const uint MORPH_STRIDE = 28;
static const uint MORPH_TARGET = 0;
static const uint MORPH_POSITION = 4;
static const uint MORPH_NORMAL = 16;

// METAL_BINDINGS is a host difference, not a target one. The two branches carry
// the same Vulkan bindings and differ only in the register numbers, because
// Metal has one buffer index space where DirectX has three: a `t0` and a `u0`
// that never collide on DXIL would both land on buffer(0). The Metal numbers are
// the indices its encoder already binds; Vulkan pushes the params and DirectX
// takes them as root constants at b0, where a bare push constant lands there.
#ifdef METAL_BINDINGS
[[vk::binding(0, 0)]] ByteAddressBuffer src : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<float4x4> palette : register(t2);
[[vk::binding(2, 0)]] RWByteAddressBuffer dst : register(u1);
[[vk::binding(3, 0)]] ByteAddressBuffer morph_data : register(t4);
[[vk::binding(4, 0)]] StructuredBuffer<float> morph_weights : register(t5);
[[vk::push_constant]] ConstantBuffer<SkinParams> params : register(b3);
#else
[[vk::binding(0, 0)]] ByteAddressBuffer src : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<float4x4> palette : register(t1);
[[vk::binding(2, 0)]] RWByteAddressBuffer dst : register(u0);
[[vk::binding(3, 0)]] ByteAddressBuffer morph_data : register(t2);
[[vk::binding(4, 0)]] StructuredBuffer<float> morph_weights : register(t3);
[[vk::push_constant]] ConstantBuffer<SkinParams> params;
#endif

// Byte offset of the entry list in the packed morph buffer
// (`PayloadMorphs::packed_words`): `vertex_count + 1` uint entry offsets, then
// the `MorphEntry` list at a 16-byte-aligned word.
uint morph_entry_byte_base(uint vertex_count)
{
    return ((vertex_count + 1u + 3u) & ~3u) * 4u;
}

// The four joint indices of one vertex: two u16s per word, low half first.
uint4 unpack_joints(uint2 words)
{
    return uint4(words.x & 0xFFFFu, words.x >> 16, words.y & 0xFFFFu, words.y >> 16);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rt_skin(uint3 gid : SV_DispatchThreadID)
{
    if (gid.x >= params.vertex_count)
    {
        return;
    }
    uint idx = params.vertex_base + gid.x;
    uint sbase = idx * SKINNED_STRIDE;

    float3 pos = asfloat(src.Load3(sbase + SKINNED_POS));
    float3 normal = asfloat(src.Load3(sbase + SKINNED_NORMAL));
    float3 tangent = asfloat(src.Load3(sbase + SKINNED_TANGENT));
    float3 color = asfloat(src.Load3(sbase + SKINNED_COLOR));
    float2 uv = asfloat(src.Load2(sbase + SKINNED_UV));
    uint4 joints = unpack_joints(src.Load2(sbase + SKINNED_JOINTS));
    float4 weights = asfloat(src.Load4(sbase + SKINNED_WEIGHTS));

    // Morph deltas apply in bind space, before the skin matrix. The sparse
    // buffer is vertex-major: this thread walks only the entries that touch its
    // own LOCAL vertex index.
    if (params.target_count != 0u)
    {
        uint first = morph_data.Load(gid.x * 4u);
        uint end = morph_data.Load(gid.x * 4u + 4u);
        uint ebase = morph_entry_byte_base(params.vertex_count);
        for (uint e = first; e < end; ++e)
        {
            uint dbase = ebase + e * MORPH_STRIDE;
            float w = morph_weights[morph_data.Load(dbase + MORPH_TARGET)];
            pos += w * asfloat(morph_data.Load3(dbase + MORPH_POSITION));
            normal += w * asfloat(morph_data.Load3(dbase + MORPH_NORMAL));
        }
    }
    normal = normalize(normal);

    uint last = params.joint_count == 0u ? 0u : params.joint_count - 1u;
    float4x4 skin = weights.x * palette[min(joints.x, last)]
                  + weights.y * palette[min(joints.y, last)]
                  + weights.z * palette[min(joints.z, last)]
                  + weights.w * palette[min(joints.w, last)];
    float3x3 skin3 = (float3x3)skin;

    uint dbase = idx * VERTEX_STRIDE;
    dst.Store3(dbase + VERTEX_POS, asuint(mul(skin, float4(pos, 1.0)).xyz));
    dst.Store3(dbase + VERTEX_NORMAL, asuint(normalize(mul(skin3, normal))));
    dst.Store3(dbase + VERTEX_TANGENT, asuint(mul(skin3, tangent)));
    dst.Store3(dbase + VERTEX_COLOR, asuint(color));
    dst.Store2(dbase + VERTEX_UV, asuint(uv));
}
