#include <metal_stdlib>
using namespace metal;

// Compute skinning for ray tracing. The main pass skins in the vertex shader, so
// no deformed-vertex buffer exists for the BVH to trace against. This kernel
// produces one: it reads the bind-pose skinned vertices + a per-object joint
// palette and writes posed (model-space) plain `Vertex`s into a shared deformed
// buffer, which the RT acceleration-structure build then traces. One dispatch
// per skinned object over its vertex range; the deformed buffer mirrors the
// skinned vertex buffer's indexing so the existing skinned index buffer
// addresses it directly.

// Matches gfx::mesh_payload::SkinnedVertex (repr(C), 80-byte stride). Packed
// types keep the field offsets identical to the Rust struct.
struct SkinnedVtxIn {
    packed_float3 pos;      // 0
    packed_float3 normal;   // 12
    packed_float3 tangent;  // 24
    packed_float3 color;    // 36
    packed_float2 uv;       // 48
    ushort        joints[4];// 56
    packed_float4 weights;  // 64  (..80)
};

// Matches gfx::mesh_payload::Vertex (repr(C), 56-byte stride) - the same layout
// the static RT vertex fetchers read.
struct VtxOut {
    packed_float3 pos;      // 0
    packed_float3 normal;   // 12
    packed_float3 tangent;  // 24
    packed_float3 color;    // 36
    packed_float2 uv;       // 48  (..56)
};

// Matches gfx::mesh_payload::MorphEntry (repr(C), 28-byte stride): one sparse
// morph delta naming its target, a bind-space position + normal offset scaled
// by that target's weight.
struct MorphEntry {
    uint          target;   // 0
    packed_float3 position; // 4
    packed_float3 normal;   // 16 (..28)
};

// The packed morph buffer (PayloadMorphs::packed_words): `vertex_count + 1`
// uint entry offsets, then the MorphEntry list at a 16-byte-aligned word.
inline uint morph_entry_word_base(uint vertex_count) {
    return (vertex_count + 1u + 3u) & ~3u;
}

// buffer(3): which slice of the shared buffers this dispatch deforms.
struct SkinParams {
    uint vertex_base;   // first vertex of this object in the shared buffers
    uint vertex_count;  // vertices to deform this dispatch
    uint joint_count;   // palette size (joint indices are clamped below it)
    uint target_count;  // morph targets in buffer(4); 0 = no morphing
};

kernel void rt_skin(
    device const SkinnedVtxIn* src     [[buffer(0)]],
    device VtxOut*             dst     [[buffer(1)]],
    constant float4x4*         palette [[buffer(2)]],
    constant SkinParams&       p       [[buffer(3)]],
    device const uint*         morphs  [[buffer(4)]],
    constant float*            mweights [[buffer(5)]],
    uint                       gid     [[thread_position_in_grid]]
) {
    if (gid >= p.vertex_count) return;
    uint idx = p.vertex_base + gid;
    SkinnedVtxIn v = src[idx];

    // Morph deltas apply in bind space, before the skin matrix. The sparse
    // buffer is vertex-major: this thread walks only the entries that touch
    // its own LOCAL vertex index.
    float3 pos = float3(v.pos);
    float3 nrm = float3(v.normal);
    if (p.target_count != 0u) {
        uint first = morphs[gid];
        uint end   = morphs[gid + 1u];
        device const MorphEntry* entries =
            (device const MorphEntry*)(morphs + morph_entry_word_base(p.vertex_count));
        for (uint e = first; e < end; ++e) {
            MorphEntry d = entries[e];
            float w = mweights[d.target];
            pos += w * float3(d.position);
            nrm += w * float3(d.normal);
        }
    }
    nrm = normalize(nrm);

    uint last = p.joint_count == 0u ? 0u : p.joint_count - 1u;
    float4x4 skin = v.weights.x * palette[min((uint)v.joints[0], last)]
                  + v.weights.y * palette[min((uint)v.joints[1], last)]
                  + v.weights.z * palette[min((uint)v.joints[2], last)]
                  + v.weights.w * palette[min((uint)v.joints[3], last)];
    float3x3 skin3 = float3x3(skin[0].xyz, skin[1].xyz, skin[2].xyz);

    VtxOut o;
    o.pos     = (skin * float4(pos, 1.0)).xyz;
    o.normal  = normalize(skin3 * nrm);
    o.tangent = skin3 * float3(v.tangent);
    o.color   = v.color;
    o.uv      = v.uv;
    dst[idx]  = o;
}
