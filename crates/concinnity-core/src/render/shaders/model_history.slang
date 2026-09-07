// Previous-frame model snapshot for the G-buffer pre-pass's motion vectors.
// One thread per cull record: copies this frame's model matrix out of the
// bindless object buffer into the frame's slot of the model-history ring, which
// the NEXT frame's pre-pass reads as its previous-frame transform.
//
// The snapshot lives on the GPU because the object buffer already carries every
// model the pre-pass needs: a host-built parallel table would write the same
// 64 bytes per record a second time, once per backend.
//
// Records are indexed exactly as the object buffer indexes them (static prefix,
// instances, runtime reserve, skinned tail), so a record's history entry is only
// meaningful while that record keeps its occupant. A record whose occupant
// changed carries `DRAW_NO_HISTORY` in its draw args and the pre-pass falls back
// to the current model, which is what makes the copy unconditional here.
//
// Single source for every backend. Metal and DXIL assign the same slots from
// declaration order (params 0, objects 1, history 2), matching the hand-assigned
// host bindings on all three; Vulkan binds set 0 bindings 0-2.

{OBJECT_COMMON}

// Matches `ModelHistoryParams` in render/uniforms/geometry.rs (16 B).
struct ModelHistoryParams
{
    uint record_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[[vk::binding(0, 0)]]
ConstantBuffer<ModelHistoryParams> params;

[[vk::binding(1, 0)]]
StructuredBuffer<GpuObjectData> objects;

[[vk::binding(2, 0)]]
RWStructuredBuffer<float4x4> history;

[shader("compute")]
[numthreads(64, 1, 1)]
void model_history_kernel(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= params.record_count)
    {
        return;
    }
    history[i] = objects[i].model;
}
