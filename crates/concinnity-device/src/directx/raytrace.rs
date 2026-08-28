// src/directx/raytrace.rs
//
// DXR (DirectX Raytracing) acceleration structures for the hardware ray-traced
// reflection pass. Builds, from the shared static vertex / index buffers and the
// `DrawObject` + `InstancedCluster` lists, the bottom- and top-level
// acceleration structures (BLAS / TLAS) the inline-`RayQuery` reflection shader
// traces against, plus a per-instance geometry table the shader uses to fetch
// the hit triangle and shade it.
//
// One triangle BLAS per participating static object (over its slice of the
// shared buffers) and one per instanced cluster; one TLAS instance per object
// and one per cluster instance (transform = the object/instance model matrix,
// `InstanceID` = the geometry-table index). The BLAS describe object-space
// geometry and never change for a rigid transform; only the TLAS instance
// transforms (and the geometry table's per-instance model matrices the shader
// shades with) move when a prop moves.
//
// Mirrors `metal/raytrace.rs`. Skinned geometry is added per frame
// (`rebuild_skinned`): a compute pass deforms each skinned object's bind-pose
// vertices into a model-space buffer, one BLAS per skinned object is
// built or refit over it, and the TLAS + geometry table are rebuilt over the
// persistent static/cluster BLAS plus the skinned tail.
//
// Every resource those two per-frame paths write lives in a ring rather than
// being allocated fresh: `skinned_ring` is one slot per frame in flight, keyed on
// `frame_idx`, and `static_ring` advances a cursor one slot per dynamic-transform
// rebuild. Both are rewritten in place and grown only on demand, so a steady
// scene allocates nothing after warm-up. The ring rule they rest on is that the
// frame-begin fence wait retires a slot's previous writer before the next one
// touches it -- sound for the skinned path because it runs on EVERY frame, and
// for the static path because its cursor advances per rebuild rather than per
// frame (a sparsely-moving scene traces one TLAS across many frames, so a
// frame-keyed slot could be reused while a live trace still reads it). See
// `SkinnedFrameRing` / `StaticFrameRing`. Only the rare incremental topology
// refresh still allocates fresh and parks its orphans in `retire`.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use super::allocator::{DeviceAllocator, PooledBuffer};
use crate::gfx::render_types::{DrawObject, InstancedCluster, RtGeomEntry, SkinnedDrawObject};
use crate::gfx::rt_geom::{cluster_geom_entry, geom_entry, models_dirty, skinned_geom_entry};
use crate::gfx::rt_refit::{BlasUpdate, SkinnedRefit, SkinnedShape};
use crate::gfx::rt_topology::{GeomSig, plan_topology_refresh};
// The dynamic-update mode ladder lives in concinnity-render; re-exported so the
// `super::raytrace::RtDynamicMode` path (init + context) keeps resolving.
pub(super) use crate::gfx::rt_geom::RtDynamicMode;

use super::com;
use super::context::FRAMES;
use super::texture::{create_buffer, create_uav_buffer, transition_barrier};
use crate::directx::slang_builtins::SlangCompile;

// Byte stride of a `Vertex` in the shared vertex buffer (pos + normal + tangent
// + colour + uv = 14 floats). The BLAS reads positions at this stride and the
// shader fetches attributes at this stride. The deformed (posed) skinned vertex
// buffer the skin kernel writes carries the same 56-byte layout.
const VERTEX_STRIDE: u64 = 56;

// Shared with the Metal and Vulkan hosts: one `.slang` declares it now.
// Re-exported so `crate::directx::raytrace::SkinParams` keeps resolving.
pub(in crate::directx) use concinnity_render::uniforms::SkinParams;

// Whether the active GPU supports the DXR feature tier inline `RayQuery` needs.
// Tier 1.1 is required because the reflection pass traces from a pixel shader
// (`RayQuery::TraceRayInline`), which Tier 1.0 (DispatchRays-only) does not
// expose. Mirrors `metal::raytrace::raytracing_supported`.
pub(super) fn raytracing_supported(device: &ID3D12Device) -> bool {
    let mut opts5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
    // SAFETY: a query on a live COM object; the descriptor it reads and the out-parameters it fills
    // are live locals that outlive the call.
    let ok = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS5,
            &mut opts5 as *mut _ as *mut std::ffi::c_void,
            std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>() as u32,
        )
    };
    ok.is_ok() && opts5.RaytracingTier.0 >= D3D12_RAYTRACING_TIER_1_1.0
}

// Pack a column-major object-to-world `model` matrix into a DXR instance
// transform: a 3x4 ROW-major affine stored flat as `[f32; 12]` (rows
// `[m00 m01 m02 m03][m10 ...][m20 ...]`), where element (row r, col c) is the
// world-matrix value. The Rust `model` is column-major, so the math element
// (r, c) lives at `model[c][r]`; the row/column transpose here is the opposite
// handedness from Metal's `MTLPackedFloat4x3` (which drops the bottom row of
// each column), so getting it wrong silently mirrors / shears every reflection.
// Unit-tested.
pub(super) fn pack_instance_transform(model: [[f32; 4]; 4]) -> [f32; 12] {
    [
        model[0][0],
        model[1][0],
        model[2][0],
        model[3][0],
        model[0][1],
        model[1][1],
        model[2][1],
        model[3][1],
        model[0][2],
        model[1][2],
        model[2][2],
        model[3][2],
    ]
}

// One DXR instance descriptor with an explicit 3x4 transform, `InstanceID`
// (indexes the geometry table), full visibility mask, and the BLAS GPU virtual
// address. Hit-group contribution + flags are zero (inline tracing ignores hit
// groups).
fn instance_desc(
    model: [[f32; 4]; 4],
    instance_id: u32,
    blas_gva: u64,
) -> D3D12_RAYTRACING_INSTANCE_DESC {
    D3D12_RAYTRACING_INSTANCE_DESC {
        Transform: pack_instance_transform(model),
        // InstanceID in the low 24 bits, InstanceMask (0xFF) in the high 8.
        _bitfield1: (instance_id & 0x00FF_FFFF) | (0xFFu32 << 24),
        // InstanceContributionToHitGroupIndex (24) + Flags (8), both zero.
        _bitfield2: 0,
        AccelerationStructure: blas_gva,
    }
}

// A triangle geometry descriptor over a slice of the shared buffers, declared
// opaque. `vertex_start`/`index_start` are absolute GPU virtual addresses into
// the shared buffers (already offset for the object's base vertex / index).
fn triangle_geometry(
    vertex_start: u64,
    vertex_count: u32,
    index_start: u64,
    index_count: u32,
) -> D3D12_RAYTRACING_GEOMETRY_DESC {
    D3D12_RAYTRACING_GEOMETRY_DESC {
        Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
        Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
        Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
            Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                Transform3x4: 0,
                IndexFormat: DXGI_FORMAT_R32_UINT,
                VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                IndexCount: index_count,
                VertexCount: vertex_count,
                IndexBuffer: index_start,
                VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                    StartAddress: vertex_start,
                    StrideInBytes: VERTEX_STRIDE,
                },
            },
        },
    }
}

// A triangle geometry descriptor over the deformed (posed) skinned vertex buffer
// with an `R32_UINT` index buffer. The skinned BLAS bakes absolute indices into
// the deformed buffer (base vertex folded to 0), so `vertex_start` is the
// deformed buffer's base GVA and `index_start` is the index buffer offset for
// this object. Same 56-byte vertex stride as the static path.
fn skinned_triangle_geometry(
    vertex_start: u64,
    vertex_count: u32,
    index_start: u64,
    index_count: u32,
) -> D3D12_RAYTRACING_GEOMETRY_DESC {
    D3D12_RAYTRACING_GEOMETRY_DESC {
        Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
        Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
        Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
            Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                Transform3x4: 0,
                IndexFormat: DXGI_FORMAT_R32_UINT,
                VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                IndexCount: index_count,
                VertexCount: vertex_count,
                IndexBuffer: index_start,
                VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                    StartAddress: vertex_start,
                    StrideInBytes: VERTEX_STRIDE,
                },
            },
        },
    }
}

// Create an acceleration-structure backing buffer (default heap,
// `ALLOW_UNORDERED_ACCESS`, initial state `RAYTRACING_ACCELERATION_STRUCTURE`).
fn create_as_buffer(device: &ID3D12Device, size: u64) -> Result<ID3D12Resource, String> {
    create_uav_buffer(
        device,
        size.max(256),
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
    )
}

// Create a build scratch buffer (default heap, `ALLOW_UNORDERED_ACCESS`). D3D12
// buffers are always created in `COMMON` regardless of the requested state, so
// pass `COMMON` explicitly to avoid the debug-layer "Ignoring InitialState"
// warning; the buffer implicitly promotes to `UNORDERED_ACCESS` on the AS
// build's first UAV access (and decays back to `COMMON` after each
// `ExecuteCommandLists`, re-promoting on the next reused-scratch rebuild).
fn create_scratch(device: &ID3D12Device, size: u64) -> Result<ID3D12Resource, String> {
    create_uav_buffer(device, size.max(256), D3D12_RESOURCE_STATE_COMMON)
}

// Upload a `Copy` slice to a fresh UPLOAD-heap buffer (host-visible,
// GPU-readable). Used for the TLAS instance-descriptor buffer (read by the AS
// build) and the geometry table (read as a `StructuredBuffer` root SRV by the
// trace).
fn upload_slice<T: Copy>(
    alloc: &DeviceAllocator,
    data: &[T],
    label: &str,
) -> Result<PooledBuffer, String> {
    let bytes = std::mem::size_of_val(data).max(16) as u64;
    let buf = create_buffer(
        alloc,
        bytes,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { buf.Map(0, None, Some(&mut ptr)) }.map_err(|e| format!("map {label}: {e}"))?;
    // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and the source
    // is a separate allocation, so the ranges cannot overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            ptr as *mut u8,
            std::mem::size_of_val(data),
        );
        buf.Unmap(0, None);
    }
    Ok(buf)
}

// A global UAV barrier (null resource): orders every preceding acceleration-
// structure / UAV write before subsequent reads on the same command list. Used
// between BLAS builds sharing one scratch buffer and before the TLAS build.
fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

// Prebuild sizes for one acceleration structure.
fn prebuild_info(
    device: &ID3D12Device5,
    inputs: &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS,
) -> D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO {
    let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
    // SAFETY: a query on a live COM object; the descriptor it reads and the out-parameters it fills
    // are live locals that outlive the call.
    unsafe { device.GetRaytracingAccelerationStructurePrebuildInfo(inputs, &mut info) };
    info
}

// The BOTTOM_LEVEL build inputs for a single geometry desc. `geo` must outlive
// the returned inputs (the inputs hold a pointer to it).
fn blas_inputs(
    geo: &D3D12_RAYTRACING_GEOMETRY_DESC,
) -> D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
        Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
        NumDescs: 1,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
            pGeometryDescs: geo,
        },
    }
}

// The BOTTOM_LEVEL build inputs for one skinned geometry desc. Always carries
// `ALLOW_UPDATE`, which is what makes a later in-place refit legal (DXR requires
// it at build time and it also makes the prebuild report an update scratch size);
// `Refit` additionally sets `PERFORM_UPDATE`, turning the build into a refit of
// the structure named by `SourceAccelerationStructureData`. Pass `Build` when
// sizing: a prebuild only needs the allocation flags.
fn skinned_blas_inputs(
    geo: &D3D12_RAYTRACING_GEOMETRY_DESC,
    update: BlasUpdate,
) -> D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
    let mut flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
        | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    if update == BlasUpdate::Refit {
        flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    }
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
        Flags: flags,
        NumDescs: 1,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
            pGeometryDescs: geo,
        },
    }
}

// The TOP_LEVEL build inputs over `instance_count` instances at
// `instance_descs_gva` (0 during prebuild, where only the count + layout
// matter).
fn tlas_inputs(
    instance_count: u32,
    instance_descs_gva: u64,
) -> D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
        Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
        NumDescs: instance_count,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
            InstanceDescs: instance_descs_gva,
        },
    }
}

// The compute pipeline that deforms skinned vertices for ray tracing
// (`rt_skin.slang`): a root SRV for the bind-pose skinned vertices (t0), a root
// SRV for the per-object joint palette (t1), a root UAV for the deformed output
// (u0), and a 4-DWORD `SkinParams` root-constant block (b0). Built alongside the
// RT PSO and held on `RtAccelData`; mirrors Metal's `skin_pipeline`.
pub(super) struct SkinPipeline {
    pub(super) root_sig: ID3D12RootSignature,
    pub(super) pso: ID3D12PipelineState,
}

// DWORD count of the `SkinParams` root-constant block (vertex_base, vertex_count,
// joint_count, target_count).
const SKIN_PARAMS_DWORDS: u32 = 4;

// Root signature for the `rt_skin` compute kernel: `SkinParams` root constants at
// b0, the skinned vertex buffer as a root SRV (t0), the joint palette as a root
// SRV (t1), the deformed output as a root UAV (u0), the morph deltas as a root
// SRV (t2), and the morph weights as a root SRV (t3).
fn create_skin_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let params = [
        // [0] b0 SkinParams root constants
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: SKIN_PARAMS_DWORDS,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [1] t0 skinned vertex buffer (raw)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [2] t1 joint palette (structured)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [3] u0 deformed output (raw)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [4] t2 morph deltas (raw, dense target-major)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 2,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [5] t3 morph weights (raw, one f32 per target)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 3,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        ..Default::default()
    };
    super::pipeline::serialize_desc_and_create(device, &desc, "rt skin root sig")
}

// Build the `rt_skin` compute pipeline (root signature + PSO). slangc emits it
// as `cs_6_5` DXIL, the same SM the RT reflection shader needs. Returns `Err`
// when the kernel fails to compile; the caller then leaves the skin pipeline
// `None` and skinned geometry is absent from the BVH (the RT pass still runs for
// static geometry).
fn build_skin_pipeline(device: &ID3D12Device, hot_reload: bool) -> Result<SkinPipeline, String> {
    let cs = super::slang_builtins::RT_SKIN.compile(hot_reload)?;
    let root_sig = create_skin_root_signature(device)?;
    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: com::borrowed(&root_sig),
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: cs.as_ptr() as _,
            BytecodeLength: cs.len(),
        },
        ..Default::default()
    };
    // SAFETY: `desc` outlives this synchronous call, and so do the root signature, shader bytecode
    // and input-element array whose raw pointers it borrows.
    let pso = unsafe { crate::directx::pso_library::create_compute(device, &desc) }
        .map_err(|e| format!("create rt skin PSO: {e}"))?;
    Ok(SkinPipeline { root_sig, pso })
}

// The per-frame skinned-geometry inputs `rebuild_skinned` needs to deform and
// add skinned objects to the BVH. Assembled by `rt_dynamic_update` from the
// context's skinned state.
pub(super) struct SkinnedRtInputs<'a> {
    // One entry per skinned mesh (only `visible`, real-triangle objects build).
    pub objects: &'a [SkinnedDrawObject],
    // GPU virtual address of the shared bind-pose skinned vertex buffer
    // (`SkinnedVertex`, 80-byte stride) the skin kernel reads.
    pub vertex_gva: u64,
    // GPU virtual address of the shared skinned index buffer the skinned BLAS
    // and the reflection trace address the deformed buffer with.
    pub index_gva: u64,
    // This frame's per-object joint palettes, parallel to `objects` (each is that
    // object's `MAX_JOINTS`-matrix upload buffer for the current frame). Borrowed
    // from the main pass's per-frame palettes rather than uploaded again here, so
    // the RT skin dispatch costs no extra buffer per object per frame.
    pub joint_buffers: &'a [PooledBuffer],
}

// Whether a ring slot must (re)allocate to satisfy `needed` bytes: it is either
// empty or its current capacity is too small. Pure so the grow decision is unit-
// testable without a device.
fn ring_slot_needs_grow(present: bool, capacity: u64, needed: u64) -> bool {
    !present || capacity < needed
}

// One frame slot of the per-frame skinned-rebuild buffers. The skinned RT rebuild
// reuses these in place every frame and only (re)allocates a slot when a larger
// size is needed, so the steady state allocates nothing. Reuse is hazard-free:
// the frame-begin fence wait gates this slot's prior GPU work (`FRAMES` deep), so
// the prior trace that read the slot has finished before the rebuild overwrites
// it. This replaces the old allocate-fresh-every-frame + retire-pool path, whose
// per-frame committed-resource churn grew the driver's video-memory pool without
// bound. Each buffer tracks its byte capacity alongside the resource. The deformed
// vertex buffer rests in the combined shader-read state after its first rebuild
// (it is created in `COMMON`), so the per-frame skin dispatch transitions it from
// whichever state it is in.
#[derive(Default)]
struct SkinnedFrameRing {
    deformed: Option<ID3D12Resource>,
    deformed_cap: u64,
    // One BLAS per skinned object, paired with its byte capacity.
    blas: Vec<(ID3D12Resource, u64)>,
    // Whether this slot's BLAS hold a tree the next update can refit rather than
    // rebuild, and the geometry that tree was built over. Per slot because the
    // slots are written on different frames, so their rebuild cadences stagger.
    refit: SkinnedRefit,
    tlas: Option<ID3D12Resource>,
    tlas_cap: u64,
    scratch: Option<ID3D12Resource>,
    scratch_cap: u64,
    instance: Option<PooledBuffer>,
    instance_cap: u64,
    geom: Option<PooledBuffer>,
    geom_cap: u64,
}

// One ring slot of the per-rebuild static-transform buffers (the TLAS + its
// instance descriptors + the geometry table). The dynamic-transform rebuild
// advances `static_cursor` to the next slot each rebuild and reuses that slot's
// buffers in place, growing one only when a later rebuild outgrows it (the static
// instance count is fixed, so the steady state allocates nothing). Reuse is
// hazard-free: the cursor revisits a slot only after a full ring cycle, by which
// point the frame-begin fence wait (`FRAMES` deep) has retired every trace that
// read it. This replaces the allocate-fresh-every-rebuild + retire-pool path,
// whose per-frame committed-resource churn grew the driver's video-memory pool
// without bound when a prop animated continuously. The live `self.tlas` /
// `geom_table` / `instance_buffer` are clones (AddRefs) of the current slot's
// resources, so the trace's root SRVs stay valid across the rotation.
#[derive(Default)]
struct StaticFrameRing {
    tlas: Option<ID3D12Resource>,
    tlas_cap: u64,
    instance: Option<PooledBuffer>,
    instance_cap: u64,
    geom: Option<PooledBuffer>,
    geom_cap: u64,
}

// Advance a ring cursor to the next slot, wrapping at `len`. Pure so the
// wrap-around is unit-testable without a device.
fn next_slot(cursor: usize, len: usize) -> usize {
    (cursor + 1) % len.max(1)
}

// Re-collect the participating objects' current model matrices into `out`, in
// BLAS order. Returns `false` (leaving `out` unspecified) when the draw list
// changed shape -- an index is now out of range or non-resident -- in which case
// the caller leaves the structure as-is for this frame; the topology-refresh path
// is what handles a changed object set. Free-standing and filling a caller-owned
// buffer so the per-frame `Vec` lives in the update scratch rather than being
// collected fresh, and so it can be called while another field of the accel is
// mutably borrowed.
fn collect_models(
    object_indices: &[usize],
    draw_objects: &[DrawObject],
    out: &mut Vec<[[f32; 4]; 4]>,
) -> bool {
    out.clear();
    for &idx in object_indices {
        match draw_objects.get(idx) {
            Some(o) if o.resident && o.index_count >= 3 => out.push(o.model),
            _ => return false,
        }
    }
    true
}

// Outgoing acceleration-structure / scratch resources parked by an incremental
// topology refresh for deferred free. A topology refresh runs on the frame's
// start command list (async, no fence-wait), so an orphaned draw BLAS the
// still-live TLAS references, and the build scratch the just-recorded builds
// keep reading, must outlive the frames whose in-flight trace could reach them.
// Freed `FRAMES` frames later, by when the frame-begin fence wait has retired
// every trace that could have referenced them. (The per-frame TLAS/skinned
// rebuild paths recycle through their rings instead, so this pool only ever
// holds a rare topology change's orphans + scratch.)
// The scene-scaled `Vec`s the per-frame dynamic update fills. Kept on the accel
// and swapped out with `mem::take` for the duration of an update, so each frame
// reuses the heap capacity instead of collecting fresh ones at frame rate.
#[derive(Default)]
struct RtUpdateScratch {
    // Indices into the frame's skinned draw objects, for those visible with real
    // triangles, in skinned-BLAS order.
    skinned: Vec<usize>,
    // The participating draw objects' current model matrices, in BLAS order.
    models: Vec<[[f32; 4]; 4]>,
    // The geometry each skinned BLAS covers, parallel to `skinned`; compared
    // against the ring slot's last set to decide build vs refit.
    shapes: Vec<SkinnedShape>,
    // This frame's skinned geometry descriptors, parallel to `skinned`. Held
    // across the sizing and recording loops, which both point build inputs at it.
    geo: Vec<D3D12_RAYTRACING_GEOMETRY_DESC>,
    // This frame's TLAS instance descriptors and per-instance geometry entries,
    // in instance order.
    instances: Vec<D3D12_RAYTRACING_INSTANCE_DESC>,
    geom: Vec<RtGeomEntry>,
}

struct RetiredBlas {
    free_at: u64,
    // Never read: held only so its COM references (the orphaned BLAS + build
    // scratch) stay alive until this entry is dropped, once `free_at` passes.
    #[expect(
        dead_code,
        reason = "held so the orphaned BLAS and scratch stay alive until free_at passes"
    )]
    resources: Vec<ID3D12Resource>,
}

// Write `data` into a reused UPLOAD-heap ring slot, growing it only when the
// current capacity is too small, then map / copy / unmap. The slot's resource is
// CPU-written every frame, so an UPLOAD buffer (persistently re-mappable) is the
// right home; reuse avoids the per-frame committed-resource churn the skinned
// rebuild used to do via `upload_slice`.
fn write_upload_ring<T: Copy>(
    slot: &mut Option<PooledBuffer>,
    cap: &mut u64,
    alloc: &DeviceAllocator,
    data: &[T],
    label: &str,
) -> Result<(), String> {
    let len_bytes = std::mem::size_of_val(data);
    let needed = (len_bytes as u64).max(4);
    if ring_slot_needs_grow(slot.is_some(), *cap, needed) {
        *slot = Some(
            create_buffer(
                alloc,
                needed,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )
            .map_err(|e| format!("{label}: {e}"))?,
        );
        *cap = needed;
    }
    let buf = slot
        .as_ref()
        .expect("the upload buffer was just allocated or already met the capacity");
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and the source
    // is a separate allocation, so the ranges cannot overlap.
    unsafe {
        buf.Map(0, None, Some(&mut ptr))
            .map_err(|e| format!("{label} map: {e}"))?;
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr as *mut u8, len_bytes);
        buf.Unmap(0, None);
    }
    Ok(())
}

// The DXR acceleration structures + geometry table for hardware ray tracing.
// Held on the context behind an `Option`; present only when RT reflections are
// enabled, the GPU supports the DXR tier, and the scene has resident geometry.
pub(super) struct RtAccelData {
    // BLAS in build order: one per participating static object (in
    // `object_indices` order), then one per instanced cluster, then one per
    // skinned object. The leading `static_blas_count` entries are the persistent
    // static + cluster BLAS, built once and never rebuilt (a rigid transform
    // leaves object-space geometry unchanged); the skinned tail
    // (`blas[static_blas_count..]`) is rebuilt each frame from the current pose.
    blas: Vec<ID3D12Resource>,
    // How many leading `blas` entries are the persistent static + cluster BLAS. A
    // skinned object's BLAS index is `static_blas_count + si`.
    static_blas_count: usize,
    // The top-level (instance) acceleration structure the trace reads.
    tlas: ID3D12Resource,
    // `[RtGeomEntry; instance_count]` (UPLOAD heap), bound as a `StructuredBuffer`
    // root SRV; indexed by the trace's instance id.
    geom_table: PooledBuffer,
    // The TLAS instance-descriptor buffer (UPLOAD heap). Only the TLAS *build*
    // reads it; a clone of the live `static_ring` / `skinned_ring` slot's buffer.
    instance_buffer: PooledBuffer,
    // Scratch sized for the largest of every BLAS build and the TLAS build;
    // reused by the per-frame TLAS rebuild (the instance count is fixed).
    scratch: ID3D12Resource,
    // Size the TLAS prebuild reported; the static rebuild grows the ring slot's
    // TLAS to this size (once, since the static instance count is fixed).
    tlas_size: u64,

    // Per-frame update state.
    // Indices into the frame's `draw.objects` for the participating objects, in
    // BLAS / instance order. Lets a rebuild re-read current transforms in build
    // order and detect a changed draw list.
    object_indices: Vec<usize>,
    // The geometry signature each draw-object BLAS (`blas[..object_indices.len()]`)
    // was built from, parallel to `object_indices`. An incremental topology
    // refresh compares these against the current draw set to reuse every
    // unchanged BLAS and build only the new / changed ones.
    draw_blas_sigs: Vec<GeomSig>,
    // Each participating object's model matrix as baked into the live TLAS. The
    // `Auto` dirty check compares the live draw list against these.
    cached_models: Vec<[[f32; 4]; 4]>,
    // The TLAS instance descriptors for every cluster instance, re-appended
    // verbatim on a rebuild (clusters are baked static into the BVH).
    cluster_instances: Vec<D3D12_RAYTRACING_INSTANCE_DESC>,
    // The geometry-table entries for the cluster instances, parallel to
    // `cluster_instances`.
    cluster_geom: Vec<RtGeomEntry>,

    // Per-rebuild static-transform buffers (see `StaticFrameRing`), reused in
    // place by the static `rebuild_tlas` path. `static_cursor` advances one slot
    // per rebuild; a slot is revisited only after a full ring cycle, so its prior
    // trace has retired. The skinned path uses `skinned_ring` instead.
    static_ring: Vec<StaticFrameRing>,
    static_cursor: usize,

    // Per-frame skinned-rebuild buffers, one slot per frame in flight, reused in
    // place and grown on demand (see `SkinnedFrameRing`). Indexed by the frame's
    // `frame_idx`.
    skinned_ring: Vec<SkinnedFrameRing>,

    // Skinned geometry.
    // The compute-skinning pipeline (`rt_skin`). `Some` only when the kernel
    // compiled; without it skinned geometry is absent from the BVH.
    skin: Option<SkinPipeline>,
    // The fresh-per-rebuild deformed (posed) skinned vertex buffer the skin pass
    // writes and the skinned BLAS + reflection trace read. A 1-element dummy when
    // the scene has no skinned geometry, so the trace's t8 binding is always
    // valid. The skinned rebuild allocates a new one each frame and retires the
    // old (a prior frame's trace may still read it).
    deformed_verts: ID3D12Resource,
    // GPU virtual address of the shared skinned index buffer the skinned BLAS
    // + trace address the deformed buffer with. A dummy buffer's GVA when there
    // is no skinned geometry, so the t9 binding is always valid. Cloned here so
    // the trace encoder can bind it.
    skinned_indices: PooledBuffer,
    // Whether any skinned object is currently live in the BVH (drives whether the
    // per-frame update runs `rebuild_skinned` or the static `rebuild_tlas`).
    has_skinned: bool,
    // Real-texture count in the shared pool, so the dynamic-rebuild paths can
    // recompute each geometry's albedo / normal pool indices (the flat-normal
    // fallback sits at `albedo_count`) without re-querying the descriptor pools.
    albedo_count: u32,
    // Shared vertex buffer's vertex count, so an incremental topology refresh can
    // bound a freshly-built BLAS's `VertexCount` exactly as `build_rt_accel` does
    // (`total_vertices - base_vertex`), without re-threading it from the context.
    total_vertices: u32,
    // GPU virtual addresses of the shared static vertex / index buffers, so a
    // topology refresh can build a fresh draw BLAS over a slice of them. Stable
    // for the buffers' lifetime (the persistent static BLAS already bake this
    // assumption), so they are cached here rather than re-threaded each frame.
    vbuf_gva: u64,
    ibuf_gva: u64,

    // Deferred-free pool for the rare incremental-topology-refresh orphans +
    // build scratch, drained by `dynamic_update` once `frame_counter` passes each
    // entry's `free_at`. `frame_counter` is a monotonic per-update counter (the
    // per-frame ring paths need no absolute counter, so it lives here rather than
    // threading one in from the context).
    retire: Vec<RetiredBlas>,
    frame_counter: u64,

    // Persistent CPU scratch for the per-frame dynamic update, swapped out with
    // `mem::take` so its heap capacity survives the frame.
    update_scratch: RtUpdateScratch,
}

impl RtAccelData {
    // GPU virtual address of the TLAS (bound as a root SRV for inline tracing).
    pub(super) fn tlas_gva(&self) -> u64 {
        com::gpu_va(&self.tlas)
    }

    // GPU virtual address of the geometry table (bound as a `StructuredBuffer`
    // root SRV).
    pub(super) fn geom_table_gva(&self) -> u64 {
        com::gpu_va(&self.geom_table)
    }

    // GPU virtual address of the deformed (posed) skinned vertex buffer (bound as
    // the trace's t8 root SRV). A valid 1-element dummy GVA when the scene has no
    // skinned geometry, so the binding is always live.
    pub(super) fn deformed_verts_gva(&self) -> u64 {
        com::gpu_va(&self.deformed_verts)
    }

    // GPU virtual address of the skinned index buffer (bound as the trace's
    // t9 root SRV). A valid 1-element dummy GVA when there is no skinned geometry.
    pub(super) fn skinned_index_gva(&self) -> u64 {
        com::gpu_va(&self.skinned_indices)
    }

    // Attach the compute-skinning pipeline, built alongside the RT PSO (gated on
    // `rt_reflections.is_some()` + DXR support). Called once at init after the
    // accel data is built; skinned geometry is seeded on the first dynamic frame.
    pub(super) fn set_skin_pipeline(&mut self, skin: SkinPipeline) {
        self.skin = Some(skin);
    }
}

// Build the `rt_skin` compute pipeline for the RT skinning pass. A thin wrapper
// over `build_skin_pipeline` so the caller (init / RT-resources setup) does not
// reach into the private pipeline type. Returns `Err` when the kernel fails to
// compile (the caller then skips skinned RT geometry).
pub(super) fn build_rt_skin_pipeline(
    device: &ID3D12Device,
    hot_reload: bool,
) -> Result<SkinPipeline, String> {
    build_skin_pipeline(device, hot_reload)
}

// Whether a draw object contributes geometry to the BVH. When the Layer 2
// see-through path is enabled, see-through glass meshes are left out: they trace
// their own per-pixel reflection in the transparent pass, and excluding them
// means glass neither reflects glass nor self-hits. Off keeps every transparent
// mesh IN the BVH so Layer 1 opaque glass reflects and is reflected like any
// other surface. Driven by `seethrough_meshes_enabled` (opt-in per
// `Material::see_through`), not a global flag.
fn participates_in_bvh(o: &DrawObject, exclude_seethrough: bool) -> bool {
    o.resident && o.index_count >= 3 && !(exclude_seethrough && o.material.see_through != 0)
}

// Geometry + counts the RT acceleration-structure build reads.
#[derive(Clone, Copy)]
pub(super) struct RtInitGeometry<'a> {
    // Placement pool the build allocates through; also carries the device and
    // the command queue the build records and fences on.
    pub alloc: &'a DeviceAllocator,
    // Shared static vertex buffer (positions at VERTEX_STRIDE).
    pub vertex_buffer: &'a ID3D12Resource,
    // Shared static u32 index buffer.
    pub index_buffer: &'a ID3D12Resource,
    // Every participating draw object (filtered by residency + index count inside).
    pub draw_objects: &'a [DrawObject],
    // Every participating instanced cluster.
    pub clusters: &'a [InstancedCluster],
    // Shared vertex buffer's vertex count (bounds each geometry's VertexCount).
    pub total_vertices: usize,
    // Real-texture count in the shared pool (resolves per-object pool indices;
    // the flat-normal fallback sits at this index).
    pub albedo_count: u32,
    // Leave see-through glass meshes out of the BVH (see `participates_in_bvh`).
    pub exclude_seethrough: bool,
}

// Per-frame dynamic-update policy + skinned inputs for `dynamic_update`.
pub(super) struct RtDynamicInputs<'a> {
    // Rebuild gate: off / auto (dirty check) / rebuild (every frame) / tlas.
    pub mode: RtDynamicMode,
    // Per-frame joint palettes + visible skinned objects (None skips the skinned path).
    pub skinned: Option<SkinnedRtInputs<'a>>,
    // Index into the per-frame ring (frame_idx % FRAMES).
    pub frame_idx: usize,
    // Set when the participating draw set changed since the last update.
    pub topology_dirty: bool,
    // Leave see-through glass meshes out of the BVH (see `participates_in_bvh`).
    // Must match what the init build used, or a refresh would silently re-add
    // geometry the transparent pass is already drawing.
    pub exclude_seethrough: bool,
}

// Build the BLAS / TLAS / geometry table for the scene on a one-shot command
// list (committed and fence-waited so the structures are ready before the first
// frame traces them). Returns `Ok(None)` when there is no resident triangle
// geometry to trace: the caller then leaves RT disabled and falls back to SSR.
//
// `total_vertices` is the shared vertex buffer's vertex count (used to bound
// each geometry's `VertexCount`); `albedo_count` is the shared pool's
// real-texture count, used to resolve each geometry's albedo / normal pool
// indices (the flat-normal fallback sits at `albedo_count`) for the RT hit shader.
pub(super) fn build_rt_accel(geometry: RtInitGeometry) -> Result<Option<RtAccelData>, String> {
    let RtInitGeometry {
        alloc,
        vertex_buffer,
        index_buffer,
        draw_objects,
        clusters,
        total_vertices,
        albedo_count,
        exclude_seethrough,
    } = geometry;
    let device = alloc.device();
    let queue = alloc.queue();
    let device5: ID3D12Device5 = device
        .cast()
        .map_err(|e| format!("ID3D12Device5 cast (DXR unsupported?): {e}"))?;

    // Participating static objects + clusters (real triangles, resident, and not
    // rerouted to the see-through transparent path).
    let object_indices: Vec<usize> = draw_objects
        .iter()
        .enumerate()
        .filter(|(_, o)| participates_in_bvh(o, exclude_seethrough))
        .map(|(i, _)| i)
        .collect();
    let cluster_list: Vec<(usize, &InstancedCluster)> = clusters
        .iter()
        .enumerate()
        .filter(|(_, c)| c.index_count >= 3 && !c.instances.is_empty())
        .collect();
    if object_indices.is_empty() && cluster_list.is_empty() {
        return Ok(None);
    }

    let vbuf_gva = com::gpu_va(vertex_buffer);
    let ibuf_gva = com::gpu_va(index_buffer);

    // One geometry desc per BLAS: participating objects first, then clusters.
    let mut geo_descs: Vec<D3D12_RAYTRACING_GEOMETRY_DESC> =
        Vec::with_capacity(object_indices.len() + cluster_list.len());
    for &i in &object_indices {
        let obj = &draw_objects[i];
        let base_vertex = obj.base_vertex as u64;
        let vcount = (total_vertices as u64).saturating_sub(base_vertex) as u32;
        geo_descs.push(triangle_geometry(
            vbuf_gva + base_vertex * VERTEX_STRIDE,
            vcount,
            ibuf_gva + obj.index_offset as u64 * 4,
            obj.index_count as u32,
        ));
    }
    for (_, c) in &cluster_list {
        geo_descs.push(triangle_geometry(
            vbuf_gva,
            total_vertices as u32,
            ibuf_gva + c.index_offset as u64 * 4,
            c.index_count as u32,
        ));
    }

    // Size + allocate each BLAS; track the largest scratch requirement.
    let mut blas: Vec<ID3D12Resource> = Vec::with_capacity(geo_descs.len());
    let mut max_scratch: u64 = 0;
    for geo in &geo_descs {
        let inputs = blas_inputs(geo);
        let info = prebuild_info(&device5, &inputs);
        blas.push(create_as_buffer(device, info.ResultDataMaxSizeInBytes)?);
        max_scratch = max_scratch.max(info.ScratchDataSizeInBytes);
    }

    // Instance descriptors + geometry table, in instance order: static objects
    // (each referencing its own BLAS), then every cluster instance (referencing
    // the cluster's single BLAS, each with its own transform + geom entry).
    let draw_blas_count = object_indices.len();
    let mut instance_descs: Vec<D3D12_RAYTRACING_INSTANCE_DESC> =
        Vec::with_capacity(object_indices.len());
    let mut geom_entries: Vec<RtGeomEntry> = Vec::with_capacity(object_indices.len());
    for (slot, &i) in object_indices.iter().enumerate() {
        let obj = &draw_objects[i];
        instance_descs.push(instance_desc(
            obj.model,
            slot as u32,
            com::gpu_va(&blas[slot]),
        ));
        geom_entries.push(geom_entry(obj, albedo_count));
    }
    let mut cluster_instances: Vec<D3D12_RAYTRACING_INSTANCE_DESC> = Vec::new();
    let mut cluster_geom: Vec<RtGeomEntry> = Vec::new();
    for (ci, (_cluster_idx, c)) in cluster_list.iter().enumerate() {
        let blas_gva = com::gpu_va(&blas[draw_blas_count + ci]);
        for model in &c.instances {
            let id = (instance_descs.len() + cluster_instances.len()) as u32;
            cluster_instances.push(instance_desc(*model, id, blas_gva));
            cluster_geom.push(cluster_geom_entry(c, *model, albedo_count));
        }
    }
    instance_descs.extend_from_slice(&cluster_instances);
    geom_entries.extend_from_slice(&cluster_geom);

    let instance_buffer = upload_slice(alloc, &instance_descs, "RT instance descriptors")?;
    let geom_table = upload_slice(alloc, &geom_entries, "RT geometry table")?;

    // Size + allocate the TLAS + the shared scratch (>= the largest BLAS/TLAS).
    let tlas_pre = prebuild_info(&device5, &tlas_inputs(instance_descs.len() as u32, 0));
    max_scratch = max_scratch.max(tlas_pre.ScratchDataSizeInBytes);
    let tlas = create_as_buffer(device, tlas_pre.ResultDataMaxSizeInBytes)?;
    let scratch = create_scratch(device, max_scratch)?;
    let scratch_gva = com::gpu_va(&scratch);

    // Record every BLAS build (UAV-barrier-serialised over the shared scratch),
    // then the TLAS build, on a one-shot command list; fence-wait so the BVH is
    // ready before the first trace.
    // SAFETY: the command list is in the recording state, and every resource, descriptor and slice
    // these commands name is live for the call.
    record_builds(alloc, queue, |cmd4| unsafe {
        for (slot, geo) in geo_descs.iter().enumerate() {
            let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: com::gpu_va(&blas[slot]),
                Inputs: blas_inputs(geo),
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: scratch_gva,
            };
            cmd4.BuildRaytracingAccelerationStructure(&desc, None);
            cmd4.ResourceBarrier(&[uav_barrier()]);
        }
        let tlas_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: com::gpu_va(&tlas),
            Inputs: tlas_inputs(instance_descs.len() as u32, com::gpu_va(&instance_buffer)),
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: scratch_gva,
        };
        cmd4.BuildRaytracingAccelerationStructure(&tlas_desc, None);
    })?;

    let cached_models = object_indices
        .iter()
        .map(|&i| draw_objects[i].model)
        .collect();
    let draw_blas_sigs = object_indices
        .iter()
        .map(|&i| GeomSig::of(&draw_objects[i]))
        .collect();

    // Skinned geometry is seeded on the first dynamic frame (like Metal), so the
    // init build is static-only. Allocate dummy deformed-vertex / skinned-index
    // buffers so the trace's t8/t9 root SRVs always bind a valid resource; the
    // first `rebuild_skinned` replaces the deformed buffer with the real one.
    // D3D12 buffers are always created in COMMON regardless of the requested
    // state (so pass COMMON to avoid the debug-layer "Ignoring InitialState"
    // warning); COMMON implicitly promotes to a shader-read state on the trace's
    // first t8/t9 access, so the dummies need no transition.
    let deformed_verts = create_uav_buffer(device, VERTEX_STRIDE, D3D12_RESOURCE_STATE_COMMON)?;
    let skinned_indices = create_buffer(
        alloc,
        4,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
    )?;
    let static_blas_count = blas.len();

    // Seed ring slot 0 with the init structures so the static-transform rebuild
    // path reuses them in place; the live `tlas` / `geom_table` / `instance_buffer`
    // fields hold a parallel clone (AddRef), so slot 0's resources stay alive until
    // the cursor wraps back to it a full ring cycle later. The remaining slots fill
    // lazily on their first rebuild.
    let mut static_ring: Vec<StaticFrameRing> =
        (0..FRAMES).map(|_| StaticFrameRing::default()).collect();
    static_ring[0] = StaticFrameRing {
        tlas: Some(tlas.clone()),
        tlas_cap: tlas_pre.ResultDataMaxSizeInBytes.max(256),
        instance: Some(instance_buffer.clone()),
        instance_cap: (std::mem::size_of_val(instance_descs.as_slice()) as u64).max(16),
        geom: Some(geom_table.clone()),
        geom_cap: (std::mem::size_of_val(geom_entries.as_slice()) as u64).max(16),
    };

    Ok(Some(RtAccelData {
        blas,
        static_blas_count,
        tlas,
        geom_table,
        instance_buffer,
        scratch,
        tlas_size: tlas_pre.ResultDataMaxSizeInBytes,
        object_indices,
        draw_blas_sigs,
        cached_models,
        cluster_instances,
        cluster_geom,
        retire: Vec::new(),
        frame_counter: 0,
        update_scratch: RtUpdateScratch::default(),
        static_ring,
        static_cursor: 0,
        skinned_ring: (0..FRAMES).map(|_| SkinnedFrameRing::default()).collect(),
        skin: None,
        deformed_verts,
        skinned_indices,
        has_skinned: false,
        albedo_count,
        total_vertices: total_vertices as u32,
        vbuf_gva,
        ibuf_gva,
    }))
}

// Create a one-shot DIRECT command list, cast it to `ID3D12GraphicsCommandList4`
// (for `BuildRaytracingAccelerationStructure`), run `record`, submit, and
// fence-wait. A self-contained variant of `texture::one_shot_submit` that adds
// the List4 cast + error propagation. Mirrors the AS-build commit+wait Metal does.
fn record_builds<F>(
    alloc: &DeviceAllocator,
    queue: &ID3D12CommandQueue,
    record: F,
) -> Result<(), String>
where
    F: FnOnce(&ID3D12GraphicsCommandList4),
{
    let device = alloc.device();
    let alloc: ID3D12CommandAllocator =
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
            .map_err(|e| format!("RT build allocator: {e}"))?;
    let cmd: ID3D12GraphicsCommandList =
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None) }
            .map_err(|e| format!("RT build cmd list: {e}"))?;
    let cmd4: ID3D12GraphicsCommandList4 = cmd
        .cast()
        .map_err(|e| format!("ID3D12GraphicsCommandList4 cast: {e}"))?;

    record(&cmd4);

    // SAFETY: the command list is live and in the recording state, which is what `Close` requires.
    unsafe { cmd.Close() }.map_err(|e| format!("RT build close: {e}"))?;
    let list: ID3D12CommandList = cmd.cast().map_err(|e| format!("RT build cast: {e}"))?;
    // SAFETY: every command list in the submission is live and closed, and the slice outlives the
    // call.
    unsafe { queue.ExecuteCommandLists(&[Some(list)]) };

    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
        .map_err(|e| format!("RT build fence: {e}"))?;
    let event =
        // SAFETY: an auto-reset, initially unsignalled event with no name and no security
        // attributes; the call borrows nothing.
        unsafe { windows::Win32::System::Threading::CreateEventW(None, false, false, None) }
            .map_err(|e| format!("RT build event: {e}"))?;
    // SAFETY: the fence and the event were created from this device and are live for the call.
    unsafe { queue.Signal(&fence, 1) }.map_err(|e| format!("RT build signal: {e}"))?;
    // SAFETY: the fence and the event were created from this device and are live for the call.
    if unsafe { fence.GetCompletedValue() } < 1 {
        // SAFETY: the fence and the event were created from this device and are live for the call.
        unsafe { fence.SetEventOnCompletion(1, event) }
            .map_err(|e| format!("RT build set event: {e}"))?;
        // SAFETY: `event` is the handle created above and is still open.
        unsafe { windows::Win32::System::Threading::WaitForSingleObject(event, u32::MAX) };
    }
    // SAFETY: `event` was created above, every wait on it has returned, and it is closed exactly
    // once.
    unsafe { windows::Win32::Foundation::CloseHandle(event) }.ok();
    Ok(())
}

impl RtAccelData {
    // Per-frame dynamic update, recorded onto `cmd` (the frame's "start" cmd
    // list, submitted before every per-pass trace on the serial DIRECT queue).
    // Keeps the BVH current: when any skinned object is visible this frame it
    // always re-skins + rebuilds the skinned BLAS + TLAS (the pose changes every
    // frame); otherwise, when the mode + dirty gate call for it, it rebuilds the
    // TLAS + geometry table from current transforms. Both paths reuse ring buffers
    // in place (`static_ring` / `skinned_ring`), so the steady state allocates
    // nothing. A transient failure is non-fatal (keeps the live BVH).
    //
    // `frame_idx` selects the per-frame joint buffer the skin dispatch reads;
    // `skinned`, when present, carries this frame's skinned-geometry inputs.
    // `topology_dirty` is set when a runtime change (cloned prop, streamed chunk
    // added/removed) altered the participating draw set since the last update: the
    // BLAS head is refreshed (`refresh_topology`) before the transform path, so
    // the new/removed geometry enters/leaves the BVH instead of being ignored
    // (the `Auto` dirty check only watches the transforms of the prior set).
    pub(super) fn dynamic_update(
        &mut self,
        alloc: &DeviceAllocator,
        cmd: &ID3D12GraphicsCommandList,
        draw_objects: &[DrawObject],
        inputs: RtDynamicInputs,
    ) {
        // Persistent CPU scratch, swapped out so its heap capacity survives the
        // frame and put back on every exit path.
        let mut scratch = std::mem::take(&mut self.update_scratch);
        self.dynamic_update_inner(alloc, cmd, draw_objects, inputs, &mut scratch);
        self.update_scratch = scratch;
    }

    fn dynamic_update_inner(
        &mut self,
        alloc: &DeviceAllocator,
        cmd: &ID3D12GraphicsCommandList,
        draw_objects: &[DrawObject],
        inputs: RtDynamicInputs,
        scratch: &mut RtUpdateScratch,
    ) {
        let RtDynamicInputs {
            mode,
            skinned,
            frame_idx,
            topology_dirty,
            exclude_seethrough,
        } = inputs;
        // Advance the deferred-free clock and drop any topology-refresh orphans /
        // scratch whose frames-in-flight window has elapsed (the frame-begin fence
        // wait has by now retired every trace that could have referenced them).
        self.frame_counter += 1;
        let now = self.frame_counter;
        let mut i = 0;
        while i < self.retire.len() {
            if self.retire[i].free_at <= now {
                self.retire.swap_remove(i);
            } else {
                i += 1;
            }
        }

        if !mode.is_dynamic() {
            return;
        }

        // Skinned objects visible this frame, as indices into the skinned draw
        // list (which is also the joint-palette list's order). The skin pipeline
        // must be present (the kernel compiled); with none, skinned geometry stays absent
        // (the static path runs).
        scratch.skinned.clear();
        if let (Some(_), Some(s)) = (&self.skin, &skinned) {
            scratch.skinned.extend(
                s.objects
                    .iter()
                    .enumerate()
                    .filter(|(_, o)| o.visible && o.index_count >= 3)
                    .map(|(i, _)| i),
            );
        }

        // Fold any added/removed/cloned draw geometry into the BLAS head + rebuild
        // the static TLAS FIRST (before the transform path re-reads `object_indices`).
        // The refresh always rebuilds a static TLAS; on the skinned path
        // `rebuild_skinned` below then overlays the skinned tail on top.
        if topology_dirty
            && let Err(e) = self.refresh_topology(alloc, cmd, draw_objects, exclude_seethrough, now)
        {
            tracing::warn!("RT topology refresh failed (keeping live BVH): {e}");
        }

        // Skinned geometry present: always re-skin + rebuild (the pose changes
        // every frame), regardless of the dirty gate.
        if !scratch.skinned.is_empty() {
            let s = skinned.expect("scratch.skinned non-empty implies inputs present");
            if !collect_models(&self.object_indices, draw_objects, &mut scratch.models) {
                return;
            }
            if let Err(e) = self.rebuild_skinned(alloc, cmd, draw_objects, &s, frame_idx, scratch) {
                tracing::warn!("RT skinned rebuild failed (keeping live BVH): {e}");
            }
            return;
        }

        // No skinned geometry this frame. The topology refresh above already
        // rebuilt the TLAS + geometry table over the current set, so nothing more
        // is needed this frame.
        if topology_dirty {
            return;
        }

        // Re-collect current transforms in BLAS order. A changed draw-list shape
        // (an index now out of range / non-resident) is left for the topology
        // path; skip this frame.
        if !collect_models(&self.object_indices, draw_objects, &mut scratch.models) {
            return;
        }

        // If the BVH still carries a skinned tail (the last skinned object just
        // turned invisible), drop it back to the static head with a fresh TLAS so
        // the trace stops reaching stale skinned BLAS. Otherwise fall through to
        // the dirty-gated static rebuild.
        let needs_rebuild = match mode {
            RtDynamicMode::Auto => {
                self.has_skinned || models_dirty(&self.cached_models, &scratch.models)
            }
            RtDynamicMode::Rebuild | RtDynamicMode::Tlas => true,
            RtDynamicMode::Off => false,
        };
        if !needs_rebuild {
            return;
        }

        if let Err(e) = self.rebuild_tlas(alloc, cmd, draw_objects, scratch) {
            tracing::warn!("RT dynamic TLAS rebuild failed (keeping live BVH): {e}");
        }
    }

    // Incrementally bring the draw-object BLAS head in line with the current
    // participating draw set: reuse every BLAS whose geometry slice is unchanged
    // (clone = AddRef, no rebuild), build only the new / changed ones, retire the
    // orphans. The cluster BLAS are kept verbatim; any skinned tail is dropped (its
    // BLAS live in `skinned_ring`, so releasing the clone frees nothing in flight,
    // and `rebuild_skinned` re-adds the tail this frame on the skinned path). The
    // TLAS + geometry table are ALWAYS rebuilt inline over [refreshed head +
    // clusters], recycling the next `static_ring` slot like `rebuild_tlas` -- even
    // on the skinned path, where `rebuild_skinned` then overlays the skinned tail on
    // top. Rebuilding the static TLAS here (rather than deferring it to
    // `rebuild_skinned`) keeps two invariants the caller relies on: `self.tlas` is
    // replaced with a structure that does NOT reference the orphaned BLAS before
    // they are retired (so a failing / skipped `rebuild_skinned` can never leave the
    // trace reading a freed orphan), and `self.tlas_size` tracks the current static
    // instance count (so a later static `rebuild_tlas` does not under-size the ring
    // TLAS after the draw count grew).
    //
    // Recorded onto `cmd` (the frame's start cmd list), so the builds order before
    // this frame's trace by submission (no fence-wait, no stall). The orphaned draw
    // BLAS + the dedicated build scratch are parked in `retire` (freed `FRAMES`
    // frames later): the just-replaced TLAS an in-flight prior frame still traces
    // references the orphans, the just-recorded builds keep reading the scratch after
    // this returns, and the frame-begin fence wait bounds when an in-flight trace can
    // still reach them. Every `self`-field mutation is deferred to the commit block
    // at the end, past all fallible allocations, so a mid-refresh allocation failure
    // leaves the live BVH untouched (`?` returns with `self` unchanged).
    fn refresh_topology(
        &mut self,
        alloc: &DeviceAllocator,
        cmd: &ID3D12GraphicsCommandList,
        draw_objects: &[DrawObject],
        exclude_seethrough: bool,
        now: u64,
    ) -> Result<(), String> {
        let device = alloc.device();
        let device5: ID3D12Device5 = device
            .cast()
            .map_err(|e| format!("ID3D12Device5 cast (topology refresh): {e}"))?;
        let cmd4: ID3D12GraphicsCommandList4 = cmd
            .cast()
            .map_err(|e| format!("ID3D12GraphicsCommandList4 cast (topology refresh): {e}"))?;

        // Current participating draw set (same predicate as `build_rt_accel`).
        let new_indices: Vec<usize> = draw_objects
            .iter()
            .enumerate()
            .filter(|(_, o)| participates_in_bvh(o, exclude_seethrough))
            .map(|(i, _)| i)
            .collect();
        let new_sigs: Vec<GeomSig> = new_indices
            .iter()
            .map(|&i| GeomSig::of(&draw_objects[i]))
            .collect();

        // Keep the last-good BVH rather than build a degenerate zero-instance TLAS
        // when the refresh would leave no draw + cluster geometry (all removed).
        if new_indices.is_empty() && self.cluster_instances.is_empty() {
            return Ok(());
        }

        // Each cluster instance bakes an `InstanceID = draw_count + ci` indexing the
        // geometry table (draw entries first, then per cluster instance). The draw
        // count may have changed, so re-bake into a LOCAL copy for this refresh's
        // TLAS build; the copy is committed to `self.cluster_instances` at the end
        // (so a mid-refresh failure does not desync the stored IDs from the draw
        // count), and every later `rebuild_tlas` / `rebuild_skinned` appends the
        // committed copy verbatim. Transform + BLAS GVA are preserved (the cluster
        // BLAS are kept verbatim, so their addresses stay valid).
        let new_draw_count = new_indices.len();
        let mut rebaked_clusters = self.cluster_instances.clone();
        for (ci, inst) in rebaked_clusters.iter_mut().enumerate() {
            let id = (new_draw_count + ci) as u32;
            inst._bitfield1 = (id & 0x00FF_FFFF) | (0xFFu32 << 24);
        }

        let plan = plan_topology_refresh(
            &self.object_indices,
            &self.draw_blas_sigs,
            &new_indices,
            &new_sigs,
        );
        let old_draw_count = self.object_indices.len();
        let cluster_count = self.static_blas_count - old_draw_count;

        // Build the new draw-BLAS head: reuse each unchanged old draw BLAS, allocate
        // + record a fresh build for each new slot. `fresh_builds` holds the geometry
        // desc + its dest BLAS so the builds can be recorded below (after the shared
        // scratch is sized over all of them + the TLAS).
        let mut new_draw_blas: Vec<ID3D12Resource> = Vec::with_capacity(new_indices.len());
        let mut fresh_builds: Vec<(D3D12_RAYTRACING_GEOMETRY_DESC, ID3D12Resource)> = Vec::new();
        let mut max_scratch: u64 = 0;
        for (j, reuse) in plan.reuse.iter().enumerate() {
            match reuse {
                Some(k) => new_draw_blas.push(self.blas[*k].clone()),
                None => {
                    let obj = &draw_objects[new_indices[j]];
                    let base_vertex = obj.base_vertex as u64;
                    let vcount = (self.total_vertices as u64).saturating_sub(base_vertex) as u32;
                    let geo = triangle_geometry(
                        self.vbuf_gva + base_vertex * VERTEX_STRIDE,
                        vcount,
                        self.ibuf_gva + obj.index_offset as u64 * 4,
                        obj.index_count as u32,
                    );
                    let info = prebuild_info(&device5, &blas_inputs(&geo));
                    let blas = create_as_buffer(device, info.ResultDataMaxSizeInBytes)?;
                    max_scratch = max_scratch.max(info.ScratchDataSizeInBytes);
                    fresh_builds.push((geo, blas.clone()));
                    new_draw_blas.push(blas);
                }
            }
        }

        // Orphaned old draw BLAS (not reused): the just-replaced TLAS an in-flight
        // prior frame still traces references them, so park them for deferred free
        // rather than drop now (an AddRef is residency, not lifetime).
        let mut orphans: Vec<ID3D12Resource> =
            plan.retire.iter().map(|&k| self.blas[k].clone()).collect();

        // Assemble the new static head: refreshed draw BLAS ++ cluster BLAS (kept
        // verbatim). Any skinned tail is left out (ring-owned; the clone drop frees
        // nothing in flight).
        let cluster_blas: Vec<ID3D12Resource> =
            self.blas[old_draw_count..self.static_blas_count].to_vec();
        let mut new_blas = new_draw_blas;
        new_blas.extend(cluster_blas);
        let new_static_blas_count = new_indices.len() + cluster_count;

        // Static TLAS + geometry table over [refreshed draw head + clusters].
        let mut instance_descs: Vec<D3D12_RAYTRACING_INSTANCE_DESC> =
            Vec::with_capacity(new_indices.len() + rebaked_clusters.len());
        let mut geom_entries: Vec<RtGeomEntry> = Vec::with_capacity(instance_descs.capacity());
        for (slot, &idx) in new_indices.iter().enumerate() {
            let obj = &draw_objects[idx];
            instance_descs.push(instance_desc(
                obj.model,
                slot as u32,
                com::gpu_va(&new_blas[slot]),
            ));
            geom_entries.push(geom_entry(obj, self.albedo_count));
        }
        instance_descs.extend_from_slice(&rebaked_clusters);
        geom_entries.extend_from_slice(&self.cluster_geom);
        let tlas_pre = prebuild_info(&device5, &tlas_inputs(instance_descs.len() as u32, 0));
        max_scratch = max_scratch.max(tlas_pre.ScratchDataSizeInBytes);
        let tlas_needed = tlas_pre.ResultDataMaxSizeInBytes;

        // A single dedicated scratch covers every fresh BLAS build + the TLAS build;
        // retired below (the async builds keep reading it after this returns).
        let scratch = create_scratch(device, max_scratch.max(256))?;
        let scratch_gva = com::gpu_va(&scratch);

        // Recycle the next static ring slot (last live a full cycle ago, so its
        // trace has retired), growing it to this refresh's sizes. The cursor advance
        // + slot take is the only pre-commit `self` mutation; on a later `?` failure
        // it (like the existing `rebuild_tlas`) leaves the slot recreated next use --
        // the live `self.tlas` / `blas` are untouched.
        self.static_cursor = next_slot(self.static_cursor, self.static_ring.len());
        let mut slot = std::mem::take(&mut self.static_ring[self.static_cursor]);
        write_upload_ring(
            &mut slot.instance,
            &mut slot.instance_cap,
            alloc,
            &instance_descs,
            "RT instance descriptors",
        )?;
        write_upload_ring(
            &mut slot.geom,
            &mut slot.geom_cap,
            alloc,
            &geom_entries,
            "RT geometry table",
        )?;
        if ring_slot_needs_grow(slot.tlas.is_some(), slot.tlas_cap, tlas_needed) {
            slot.tlas = Some(create_as_buffer(device, tlas_needed)?);
            slot.tlas_cap = tlas_needed;
        }
        let instance_buffer = slot
            .instance
            .clone()
            .expect("RT instance buffer was sized by write_upload_ring above");
        let geom_table = slot
            .geom
            .clone()
            .expect("RT geometry table was sized by write_upload_ring above");
        let tlas = slot.tlas.clone().expect("RT TLAS buffer was sized above");

        // Record the fresh draw-BLAS builds (UAV-barrier-serialised over the shared
        // scratch), then the TLAS build. Infallible from here on.
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            for (geo, dest) in &fresh_builds {
                let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                    DestAccelerationStructureData: com::gpu_va(dest),
                    Inputs: blas_inputs(geo),
                    SourceAccelerationStructureData: 0,
                    ScratchAccelerationStructureData: scratch_gva,
                };
                cmd4.BuildRaytracingAccelerationStructure(&desc, None);
                cmd.ResourceBarrier(&[uav_barrier()]);
            }
            let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: com::gpu_va(&tlas),
                Inputs: tlas_inputs(instance_descs.len() as u32, com::gpu_va(&instance_buffer)),
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: scratch_gva,
            };
            cmd4.BuildRaytracingAccelerationStructure(&desc, None);
            cmd.ResourceBarrier(&[uav_barrier()]);
        }

        // Commit: swap in the refreshed structures + book-keeping. Any skinned tail
        // was left out of `new_blas`; `rebuild_skinned` re-adds it (and re-sets
        // `has_skinned`) on the skinned path this same frame, replacing this static
        // TLAS with a static+skinned one.
        self.blas = new_blas;
        self.static_blas_count = new_static_blas_count;
        self.draw_blas_sigs = new_sigs;
        self.cluster_instances = rebaked_clusters;
        self.has_skinned = false;
        // The skinned tail is gone from `blas`, so no ring slot's refit bookkeeping
        // describes a published tree any more. On the skinned path
        // `rebuild_skinned` re-adds the tail this same frame and rebuilds it from
        // scratch, which is also the right answer for the change that triggered
        // this refresh.
        for ring in &mut self.skinned_ring {
            ring.refit.reset();
        }
        self.tlas = tlas;
        self.geom_table = geom_table;
        self.instance_buffer = instance_buffer;
        self.tlas_size = tlas_needed;
        self.static_ring[self.static_cursor] = slot;
        // Snapshot the transforms baked into the new TLAS for the next dirty check.
        // (On the skinned path `rebuild_skinned` overwrites `cached_models`.)
        self.cached_models = new_indices.iter().map(|&i| draw_objects[i].model).collect();
        self.object_indices = new_indices;
        orphans.push(scratch);
        self.retire.push(RetiredBlas {
            free_at: now + FRAMES as u64,
            resources: orphans,
        });
        Ok(())
    }

    // Rebuild the TLAS + geometry table from `current` transforms, reusing the
    // next `static_ring` slot's buffers in place, and record the build onto `cmd`.
    // The BLAS are kept (rigid transforms leave object-space geometry unchanged).
    fn rebuild_tlas(
        &mut self,
        alloc: &DeviceAllocator,
        cmd: &ID3D12GraphicsCommandList,
        draw_objects: &[DrawObject],
        scratch: &mut RtUpdateScratch,
    ) -> Result<(), String> {
        let device = alloc.device();
        let RtUpdateScratch {
            models,
            instances: instance_descs,
            geom: geom_entries,
            ..
        } = scratch;
        // Freshly-transformed draw-object instances, then the cluster instances
        // re-appended verbatim. The geometry table mirrors this order.
        instance_descs.clear();
        geom_entries.clear();
        for (slot, &idx) in self.object_indices.iter().enumerate() {
            let obj = &draw_objects[idx];
            instance_descs.push(instance_desc(
                obj.model,
                slot as u32,
                com::gpu_va(&self.blas[slot]),
            ));
            geom_entries.push(geom_entry(obj, self.albedo_count));
        }
        // Cluster instances keep their stored BLAS GVA + transform; only their
        // instance id shifts to follow the (unchanged-count) object instances,
        // which it already does since the object count is fixed.
        instance_descs.extend_from_slice(&self.cluster_instances);
        geom_entries.extend_from_slice(&self.cluster_geom);

        // Advance to the next ring slot and reuse its buffers in place. The slot
        // was last current a full ring cycle ago, so the frame-begin fence wait has
        // retired every trace that read it; the static instance count is fixed, so
        // the upload buffers + TLAS are reused without growing after warm-up.
        self.static_cursor = next_slot(self.static_cursor, self.static_ring.len());
        let mut slot = std::mem::take(&mut self.static_ring[self.static_cursor]);
        write_upload_ring(
            &mut slot.instance,
            &mut slot.instance_cap,
            alloc,
            instance_descs,
            "RT instance descriptors",
        )?;
        write_upload_ring(
            &mut slot.geom,
            &mut slot.geom_cap,
            alloc,
            geom_entries,
            "RT geometry table",
        )?;
        if ring_slot_needs_grow(slot.tlas.is_some(), slot.tlas_cap, self.tlas_size) {
            slot.tlas = Some(create_as_buffer(device, self.tlas_size)?);
            slot.tlas_cap = self.tlas_size;
        }
        let instance_buffer = slot
            .instance
            .clone()
            .expect("RT instance buffer was sized by write_upload_ring above");
        let geom_table = slot
            .geom
            .clone()
            .expect("RT geometry table was sized by write_upload_ring above");
        let tlas = slot.tlas.clone().expect("RT TLAS buffer was sized above");

        let cmd4: ID3D12GraphicsCommandList4 = cmd
            .cast()
            .map_err(|e| format!("ID3D12GraphicsCommandList4 cast (rebuild): {e}"))?;
        let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: com::gpu_va(&tlas),
            Inputs: tlas_inputs(instance_descs.len() as u32, com::gpu_va(&instance_buffer)),
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: com::gpu_va(&self.scratch),
        };
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd4.BuildRaytracingAccelerationStructure(&desc, None);
            // Order the build before this frame's trace reads the TLAS / table.
            cmd.ResourceBarrier(&[uav_barrier()]);
        }

        // Point the live BVH at this slot's buffers (clones AddRef the slot's
        // resources, not GPU allocations), then park the slot back for reuse a full
        // ring cycle later. A skinned tail still owned from a prior skinned frame
        // (the last skinned object just turned invisible) drops back to the static
        // head: the rebuilt TLAS no longer references it, and the skinned BLAS
        // resources persist in `skinned_ring`, so dropping these clones frees
        // nothing still in flight.
        self.tlas = tlas;
        self.geom_table = geom_table;
        self.instance_buffer = instance_buffer;
        if self.blas.len() > self.static_blas_count {
            self.blas.truncate(self.static_blas_count);
            self.has_skinned = false;
            // The ring's skinned BLAS are no longer published. A refit continues
            // the tree its last full build produced, so re-entering the skinned
            // path after an arbitrary gap must rebuild rather than refit from a
            // pose the tree was never fitted for.
            for ring in &mut self.skinned_ring {
                ring.refit.reset();
            }
        }
        self.static_ring[self.static_cursor] = slot;
        self.cached_models.clear();
        self.cached_models.extend_from_slice(models);
        Ok(())
    }

    // Per-frame skinned update, recorded onto `cmd` (the frame's "start" DIRECT
    // cmd list, which supports `Dispatch`). Keeps the persistent static + cluster
    // BLAS, re-skins this frame's pose into the deformed buffer, builds or refits
    // one BLAS per skinned object over it, and rebuilds the TLAS + geometry
    // table over the static head plus the skinned tail.
    //
    // The skinned BLAS carry `ALLOW_UPDATE` and are refit IN PLACE
    // (`PERFORM_UPDATE` with `SourceAccelerationStructureData` = the destination)
    // while the triangle set is unchanged, with a full rebuild every
    // `rt_refit::REFIT_LIMIT` refits per slot to bound the traversal-quality drift
    // a refit accumulates as the pose walks away from the tree's build pose.
    //
    // All per-frame buffers (deformed verts, skinned BLAS, TLAS, scratch, instance
    // descriptors, geometry table) live in `skinned_ring[frame_idx]` and are
    // rebuilt IN PLACE: they are allocated once and only grown when a larger size
    // is needed, so the steady state allocates nothing. Reuse is hazard-free
    // because the frame-begin fence wait gates this slot's prior GPU work
    // (`FRAMES` deep), so the prior frame's trace that read this slot has finished.
    // (The previous design allocated all of these fresh every frame and parked the
    // outgoing copies in a retire pool; even though the bookkeeping was bounded,
    // the per-frame committed-resource alloc/free churn grew the driver's video-
    // memory pool without bound. See `SkinnedFrameRing`.)
    //
    // The three GPU steps are recorded in dependency order on the one DIRECT cmd
    // list: skin dispatch (writes the deformed buffer), a UAV barrier + transition
    // to a shader-readable state, then the BLAS/TLAS build (reads it). The start
    // cmd list is submitted before every per-pass trace, so build -> trace is
    // ordered by submission too.
    fn rebuild_skinned(
        &mut self,
        alloc: &DeviceAllocator,
        cmd: &ID3D12GraphicsCommandList,
        draw_objects: &[DrawObject],
        skinned: &SkinnedRtInputs,
        frame_idx: usize,
        scratch: &mut RtUpdateScratch,
    ) -> Result<(), String> {
        let RtUpdateScratch {
            skinned: skinned_objects,
            models,
            shapes,
            geo: skinned_geo,
            instances: instance_descs,
            geom: geom_entries,
        } = scratch;
        let device = alloc.device();
        let device5: ID3D12Device5 = device
            .cast()
            .map_err(|e| format!("ID3D12Device5 cast (skinned rebuild): {e}"))?;
        let cmd4: ID3D12GraphicsCommandList4 = cmd
            .cast()
            .map_err(|e| format!("ID3D12GraphicsCommandList4 cast (skinned rebuild): {e}"))?;

        // Take this frame slot's buffers out to sidestep the `&mut self` borrow
        // while reading other fields (`skin`, `object_indices`, the static `blas`
        // head); it is put back at the end. Cheap: `SkinnedFrameRing` is `Default`.
        let mut ring = std::mem::take(&mut self.skinned_ring[frame_idx]);

        // Deformed-vertex buffer (default heap, ALLOW_UNORDERED_ACCESS): the skin
        // pass writes posed `Vertex`s here, mirroring the skinned vertex buffer's
        // indexing so the index buffer addresses it directly. Sized to the
        // highest vertex the skinned objects reach, grown on demand. Created in
        // COMMON (D3D12 buffers always are); after its first rebuild it rests in
        // the combined shader-read state.
        let deformed_extent: u64 = skinned_objects
            .iter()
            .map(|&i| {
                skinned.objects[i].vertex_base as u64 + skinned.objects[i].vertex_count as u64
            })
            .max()
            .unwrap_or(0);
        let deformed_bytes = (deformed_extent * VERTEX_STRIDE).max(VERTEX_STRIDE);
        let read_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
            | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        let deformed_realloc =
            ring_slot_needs_grow(ring.deformed.is_some(), ring.deformed_cap, deformed_bytes);
        if deformed_realloc {
            ring.deformed = Some(create_uav_buffer(
                device,
                deformed_bytes,
                D3D12_RESOURCE_STATE_COMMON,
            )?);
            ring.deformed_cap = deformed_bytes;
        }
        let deformed_verts = ring
            .deformed
            .clone()
            .expect("deformed vertex buffer was sized above");
        let deformed_gva = com::gpu_va(&deformed_verts);

        // A freshly (re)allocated buffer rests in COMMON; a reused one rests in
        // `read_state` from its previous rebuild. Either is a valid source for the
        // transition into UNORDERED_ACCESS the skin dispatch writes through.
        let deformed_before = if deformed_realloc {
            D3D12_RESOURCE_STATE_COMMON
        } else {
            read_state
        };
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.ResourceBarrier(&[transition_barrier(
                &deformed_verts,
                deformed_before,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }

        // Stage 1: skin dispatch per skinned object, writing the deformed buffer.
        {
            let skin = self
                .skin
                .as_ref()
                .ok_or("rebuild_skinned called without a skin pipeline")?;
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                cmd.SetComputeRootSignature(&skin.root_sig);
                cmd.SetPipelineState(&skin.pso);
            }
        }
        for &obj_idx in skinned_objects.iter() {
            let obj = &skinned.objects[obj_idx];
            let Some(joint) = skinned.joint_buffers.get(obj_idx) else {
                continue;
            };
            let joint_gva = com::gpu_va(joint);
            if joint_gva == 0 {
                continue;
            }
            // The RT skin runs at bind pose (before per-frame morph weights
            // exist); morphing happens in the per-frame main fold. `target_count
            // == 0` leaves t2/t3 unread, so a dummy binding (the vertex buffer)
            // satisfies the root SRVs.
            let params = SkinParams {
                vertex_base: obj.vertex_base,
                vertex_count: obj.vertex_count as u32,
                joint_count: obj.joint_count.max(1) as u32,
                target_count: 0,
            };
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                cmd.SetComputeRoot32BitConstants(
                    0,
                    SKIN_PARAMS_DWORDS,
                    &params as *const SkinParams as *const std::ffi::c_void,
                    0,
                );
                cmd.SetComputeRootShaderResourceView(1, skinned.vertex_gva);
                cmd.SetComputeRootShaderResourceView(2, joint_gva);
                cmd.SetComputeRootUnorderedAccessView(3, deformed_gva);
                cmd.SetComputeRootShaderResourceView(4, skinned.vertex_gva);
                cmd.SetComputeRootShaderResourceView(5, skinned.vertex_gva);
                cmd.Dispatch((obj.vertex_count as u32).div_ceil(64), 1, 1);
            }
        }
        // Order the skin writes before the BLAS build reads them, then transition
        // the deformed buffer to a state both the BLAS build (NON_PIXEL_SHADER_
        // RESOURCE: AS-build input geometry) and the later hit-shader read
        // (PIXEL_SHADER_RESOURCE: the trace samples it as the t8 root SRV in a
        // pixel shader) accept. The combined read state satisfies both and is the
        // resting state of the deformed buffer thereafter.
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.ResourceBarrier(&[uav_barrier()]);
            cmd.ResourceBarrier(&[transition_barrier(
                &deformed_verts,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                read_state,
            )]);
        }

        // Stage 2: one BLAS per skinned object over the deformed buffer, then
        // the TLAS over the static/cluster head + the skinned tail.
        let skinned_idx_gva = skinned.index_gva;
        skinned_geo.clear();
        shapes.clear();
        for &i in skinned_objects.iter() {
            let obj = &skinned.objects[i];
            skinned_geo.push(skinned_triangle_geometry(
                deformed_gva,
                deformed_extent as u32,
                skinned_idx_gva + obj.index_offset as u64 * 4,
                obj.index_count as u32,
            ));
            shapes.push(SkinnedShape {
                index_offset: obj.index_offset,
                index_count: obj.index_count,
                vertex_extent: deformed_extent as u32,
            });
        }

        // Size each skinned BLAS in the ring (grown on demand), tracking the
        // largest scratch either a full build or a refit needs -- both run over
        // this one buffer, and which of the two this frame takes is only settled
        // below. Stale tail entries from a higher-count past frame are left in
        // place (bounded by the max skinned count); only the active prefix is used.
        // A (re)allocated BLAS holds no tree, so it forces a full build, as does a
        // regrown deformed buffer (the geometry the tree was fitted to moved).
        let mut max_scratch: u64 = 0;
        let mut storage_changed = deformed_realloc;
        for (si, geo) in skinned_geo.iter().enumerate() {
            let info = prebuild_info(&device5, &skinned_blas_inputs(geo, BlasUpdate::Build));
            let needed = info.ResultDataMaxSizeInBytes;
            if si >= ring.blas.len() {
                ring.blas.push((create_as_buffer(device, needed)?, needed));
                storage_changed = true;
            } else if ring_slot_needs_grow(true, ring.blas[si].1, needed) {
                ring.blas[si] = (create_as_buffer(device, needed)?, needed);
                storage_changed = true;
            }
            max_scratch = max_scratch
                .max(info.ScratchDataSizeInBytes)
                .max(info.UpdateScratchDataSizeInBytes);
        }

        // Instance descriptors + geometry table, in instance order: static
        // objects (current transforms), then the cluster instances verbatim, then
        // one per skinned object (BLAS index `static_blas_count + si`).
        instance_descs.clear();
        geom_entries.clear();
        for (slot, &idx) in self.object_indices.iter().enumerate() {
            let obj = &draw_objects[idx];
            instance_descs.push(instance_desc(
                obj.model,
                slot as u32,
                com::gpu_va(&self.blas[slot]),
            ));
            geom_entries.push(geom_entry(obj, self.albedo_count));
        }
        instance_descs.extend_from_slice(&self.cluster_instances);
        geom_entries.extend_from_slice(&self.cluster_geom);
        for (si, &obj_idx) in skinned_objects.iter().enumerate() {
            let obj = &skinned.objects[obj_idx];
            let id = instance_descs.len() as u32;
            let blas_gva = com::gpu_va(&ring.blas[si].0);
            instance_descs.push(instance_desc(obj.model, id, blas_gva));
            // Albedo / normal resolve through the shared flat pool by the skinned
            // object's own material slots, like any static object.
            geom_entries.push(skinned_geom_entry(obj, self.albedo_count));
        }

        write_upload_ring(
            &mut ring.instance,
            &mut ring.instance_cap,
            alloc,
            instance_descs,
            "RT instance descriptors",
        )?;
        write_upload_ring(
            &mut ring.geom,
            &mut ring.geom_cap,
            alloc,
            geom_entries,
            "RT geometry table",
        )?;
        let instance_buffer = ring
            .instance
            .clone()
            .expect("RT instance buffer was sized by write_upload_ring above");
        let geom_table = ring
            .geom
            .clone()
            .expect("RT geometry table was sized by write_upload_ring above");

        // Size the TLAS + scratch in the ring (>= the largest skinned BLAS + the
        // TLAS). The skinned instance count can change frame to frame, so size the
        // TLAS from this frame's prebuild rather than the cached size.
        let tlas_pre = prebuild_info(&device5, &tlas_inputs(instance_descs.len() as u32, 0));
        max_scratch = max_scratch.max(tlas_pre.ScratchDataSizeInBytes);
        let tlas_needed = tlas_pre.ResultDataMaxSizeInBytes;
        if ring_slot_needs_grow(ring.tlas.is_some(), ring.tlas_cap, tlas_needed) {
            ring.tlas = Some(create_as_buffer(device, tlas_needed)?);
            ring.tlas_cap = tlas_needed;
        }
        let scratch_needed = max_scratch.max(256);
        if ring_slot_needs_grow(ring.scratch.is_some(), ring.scratch_cap, scratch_needed) {
            ring.scratch = Some(create_scratch(device, scratch_needed)?);
            ring.scratch_cap = scratch_needed;
        }
        let tlas = ring.tlas.clone().expect("RT TLAS buffer was sized above");
        let scratch_buffer = ring
            .scratch
            .clone()
            .expect("RT scratch buffer was sized above");
        let scratch_gva = com::gpu_va(&scratch_buffer);

        // Settle build-or-refit last, once every fallible step above has passed:
        // recording a build the command list never gets would leave the slot
        // claiming a tree a later refit could not update.
        let update = ring.refit.plan(shapes, storage_changed);

        // Record the skinned BLAS updates (UAV-barrier-serialised over the shared
        // scratch), then the TLAS build, on `cmd`. A `Build` is a full rebuild
        // (`SourceAccelerationStructureData = 0`) into the ring buffer, overwriting
        // the prior frame's structure; a `Refit` names that same structure as the
        // source, which DXR defines as an in-place update.
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            for (si, geo) in skinned_geo.iter().enumerate() {
                let dest = com::gpu_va(&ring.blas[si].0);
                let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                    DestAccelerationStructureData: dest,
                    Inputs: skinned_blas_inputs(geo, update),
                    SourceAccelerationStructureData: match update {
                        BlasUpdate::Build => 0,
                        BlasUpdate::Refit => dest,
                    },
                    ScratchAccelerationStructureData: scratch_gva,
                };
                cmd4.BuildRaytracingAccelerationStructure(&desc, None);
                cmd.ResourceBarrier(&[uav_barrier()]);
            }
            let tlas_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: com::gpu_va(&tlas),
                Inputs: tlas_inputs(instance_descs.len() as u32, com::gpu_va(&instance_buffer)),
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: scratch_gva,
            };
            cmd4.BuildRaytracingAccelerationStructure(&tlas_desc, None);
            // Order the TLAS build before this frame's trace reads it.
            cmd.ResourceBarrier(&[uav_barrier()]);
        }

        // Point the live BVH at this frame's ring buffers (clones are AddRefs on
        // the persistent ring resources, not GPU allocations). The static/cluster
        // head of `blas` is untouched; only the skinned tail rotates.
        self.blas.truncate(self.static_blas_count);
        for (blas, _) in &ring.blas[..skinned_geo.len()] {
            self.blas.push(blas.clone());
        }
        self.tlas = tlas;
        self.geom_table = geom_table;
        self.instance_buffer = instance_buffer;
        self.scratch = scratch_buffer;
        self.deformed_verts = deformed_verts;
        self.skinned_ring[frame_idx] = ring;
        self.has_skinned = true;
        self.cached_models.clear();
        self.cached_models.extend_from_slice(models);
        Ok(())
    }
}

impl super::context::DxContext {
    // Per-frame main-pass skinning compute pass. Deforms every skinned object's
    // bind-pose vertices into this frame's deformed-vertex buffer (the bindless
    // main pass's 2nd `ExecuteIndirect` draws that buffer as rigid geometry).
    // A no-op when there is no skin pipeline / deformed buffer (no skinned mesh,
    // or the bindless fold is inactive). Runs in the Cull graph arm, before Main;
    // mirrors the stage-1 skin dispatch in `rebuild_skinned` but targets a
    // per-frame buffer that rests in VERTEX_AND_CONSTANT_BUFFER for the draw
    // instead of the RT ring's shader-read state, and is independent of RT (the
    // RT path keeps its own skin dispatch + ring, untouched). The deformed buffer
    // mirrors the skinned vertex buffer's global indexing, so the draws read it
    // with `base_vertex = 0` and the skinned index buffer unchanged.
    pub(in crate::directx) fn encode_skin(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
    ) {
        let (Some(skin), Some(deformed), Some(vb)) = (
            self.skinned.skin_pipeline.as_ref(),
            self.skinned.deformed_buffers.get(frame_idx),
            self.skinned.vertex_buffer.as_ref(),
        ) else {
            return;
        };
        if self.skinned.draw_objects.is_empty() {
            return;
        }
        let src_gva = com::gpu_va(vb);
        let dst_gva = com::gpu_va(deformed);

        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.ResourceBarrier(&[transition_barrier(
                deformed,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            cmd.SetComputeRootSignature(&skin.root_sig);
            cmd.SetPipelineState(&skin.pso);
        }
        for (i, obj) in self.skinned.draw_objects.iter().enumerate() {
            let joint_gva = self.skinned_joint_gva(frame_idx, i);
            let target_count = self
                .skinned
                .morph_target_counts
                .get(i)
                .copied()
                .unwrap_or(0);
            let params = SkinParams {
                vertex_base: obj.vertex_base,
                vertex_count: obj.vertex_count as u32,
                joint_count: obj.joint_count.max(1) as u32,
                target_count,
            };
            // Real morph deltas + this frame's weights when the object morphs;
            // otherwise the vertex buffer as an unread dummy (target_count == 0).
            let delta_gva = self
                .skinned
                .morph_delta_buffers
                .get(i)
                .and_then(|b| b.as_ref())
                .map(|b| com::gpu_va(b))
                .unwrap_or(src_gva);
            let weight_gva = self.morph_weight_gva(frame_idx, i).unwrap_or(src_gva);
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                cmd.SetComputeRoot32BitConstants(
                    0,
                    SKIN_PARAMS_DWORDS,
                    &params as *const SkinParams as *const std::ffi::c_void,
                    0,
                );
                cmd.SetComputeRootShaderResourceView(1, src_gva);
                cmd.SetComputeRootShaderResourceView(2, joint_gva);
                cmd.SetComputeRootUnorderedAccessView(3, dst_gva);
                cmd.SetComputeRootShaderResourceView(4, delta_gva);
                cmd.SetComputeRootShaderResourceView(5, weight_gva);
                cmd.Dispatch((obj.vertex_count as u32).div_ceil(64), 1, 1);
            }
        }
        // Orders the skin writes before the main pass's vertex fetch and returns
        // the buffer to its resting VERTEX_AND_CONSTANT_BUFFER state (read by both
        // Main and Main2's skinned ExecuteIndirect this frame).
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.ResourceBarrier(&[transition_barrier(
                deformed,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            )]);
        }
    }

    // Run the per-frame dynamic acceleration-structure update on `cmd` (the
    // frame's "start" DIRECT cmd list, submitted before every per-pass trace on
    // the serial DIRECT queue). A no-op when RT reflections are off. Assembles
    // this frame's skinned-geometry inputs (the skinned VB/IB GVAs + per-object
    // joint-buffer GVAs for `frame_idx`) so the skin dispatch binds the right
    // per-frame pose. Disjoint field borrows: `rt_accel` (mut) vs the rest
    // (shared); the joint GVAs are collected up-front so `skinned_joint_gva`'s
    // `&self` borrow does not overlap the `rt_accel` mutable borrow.
    //
    // Consumes `rt_topology_dirty` (set when a cloned prop / streamed chunk
    // altered the draw set): the accel's `dynamic_update` folds the change into
    // the BLAS head. When RT is on but the scene had no resident geometry at build
    // time (`rt_accel` is `None`), a topology change that introduces the first
    // participating geometry seeds the BVH from scratch here.
    pub(super) fn rt_dynamic_update(&mut self, cmd: &ID3D12GraphicsCommandList, frame_idx: usize) {
        let topology_dirty = std::mem::take(&mut self.rt_topology_dirty);

        // Seed-from-empty: RT enabled + a topology change added the first
        // participating geometry to a scene that had none at build time. The
        // one-shot build is fence-waited internally (a rare, one-time stall); the
        // DXR trace reads the TLAS + table by GPU virtual address each frame, so
        // the fresh accel is picked up with no descriptor rewire.
        if self.rt_accel.is_none() {
            if topology_dirty && self.rt_reflections.is_some() && self.rt_dynamic_mode.is_dynamic()
            {
                self.seed_rt_accel();
            }
            return;
        }

        // Build the skinned inputs while `self` is still fully borrowable. `None`
        // when there is no skinned geometry resident or the launch excluded it
        // (the static path runs). Only the two shared GVAs are read up-front; the
        // per-object joint palettes are borrowed straight out of this frame's slot
        // below (a disjoint field borrow), so the skin dispatch costs no per-frame
        // list of its own.
        let skinned_inputs = match (
            self.skinned.vertex_buffer.as_ref(),
            self.skinned.index_buffer.as_ref(),
        ) {
            (Some(vb), Some(ib))
                if self.rt_skinned_geometry && !self.skinned.draw_objects.is_empty() =>
            {
                let vertex_gva = com::gpu_va(vb);
                let index_gva = com::gpu_va(ib);
                Some((vertex_gva, index_gva))
            }
            _ => None,
        };
        let joint_buffers: &[PooledBuffer] = self
            .skinned
            .joint_buffers
            .get(frame_idx)
            .map(|b| b.as_slice())
            .unwrap_or(&[]);

        // Read before `rt_accel` is borrowed mutably below.
        let exclude_seethrough = self.seethrough_meshes_enabled();

        let Some(accel) = self.rt_accel.as_mut() else {
            return;
        };
        let skinned = skinned_inputs.map(|(v, i)| SkinnedRtInputs {
            objects: &self.skinned.draw_objects,
            vertex_gva: v,
            index_gva: i,
            joint_buffers,
        });
        accel.dynamic_update(
            &self.alloc,
            cmd,
            &self.draw.objects,
            RtDynamicInputs {
                mode: self.rt_dynamic_mode,
                skinned,
                frame_idx,
                topology_dirty,
                exclude_seethrough,
            },
        );
    }

    // Build the scene acceleration structure from scratch (mirrors the init /
    // `build_rt_runtime` accel block) when a runtime topology change introduces
    // the first participating geometry into an RT-enabled scene that had none.
    // A build failure / still-empty scene is non-fatal: `rt_accel` stays `None`
    // and the next topology change retries.
    fn seed_rt_accel(&mut self) {
        if let Some(accel) = self.build_scene_accel() {
            self.rt_accel = Some(accel);
        }
    }

    // Replace the live acceleration structure with one built over the current
    // shared vertex / index buffers. Called by `rebuild_static_geometry`, which
    // swaps both buffers and re-lays out every draw underneath the BVH: its BLAS
    // then trace the old geometry and its geometry table indexes offsets into a
    // buffer that no longer exists. An empty scene or a failed build drops the
    // BVH rather than keeping the stale one (which would have the trace read the
    // new, possibly smaller, buffers at old offsets); RT falls back to SSR until
    // the next topology change re-seeds it.
    pub(super) fn rebuild_rt_accel(&mut self) {
        self.rt_accel = self.build_scene_accel();
    }

    // Build a scene acceleration structure from scratch over the current draw set
    // + shared geometry buffers, with the skin pipeline attached. `None` when the
    // scene has no participating geometry or the build failed (warned).
    fn build_scene_accel(&self) -> Option<RtAccelData> {
        let hot_reload = self.hot_reload.enabled;
        let mut accel = match build_rt_accel(RtInitGeometry {
            alloc: &self.alloc,
            vertex_buffer: &self.geometry.vertex_buffer,
            index_buffer: &self.geometry.index_buffer,
            draw_objects: &self.draw.objects,
            clusters: &self.instanced.clusters,
            total_vertices: self.rt_static_vertex_count,
            albedo_count: self.descriptors.textures.len() as u32,
            exclude_seethrough: self.seethrough_meshes_enabled(),
        }) {
            Ok(Some(accel)) => accel,
            Ok(None) => return None,
            Err(e) => {
                tracing::warn!("RT acceleration-structure build failed: {e}");
                return None;
            }
        };
        match build_rt_skin_pipeline(&self.device, hot_reload) {
            Ok(skin) => accel.set_skin_pipeline(skin),
            Err(e) => {
                tracing::warn!("RT skin pipeline build failed (skinned meshes absent): {e}")
            }
        }
        Some(accel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_slot_grows_when_empty_or_undersized_only() {
        // Empty slot always grows.
        assert!(ring_slot_needs_grow(false, 0, 0));
        assert!(ring_slot_needs_grow(false, 0, 1024));
        // Present and large enough: reuse in place (the steady-state case).
        assert!(!ring_slot_needs_grow(true, 1024, 1024));
        assert!(!ring_slot_needs_grow(true, 4096, 1024));
        // Present but too small: grow.
        assert!(ring_slot_needs_grow(true, 512, 1024));
    }

    #[test]
    fn next_slot_wraps_around_the_ring() {
        // Advancing the static-rebuild cursor cycles through every slot and wraps
        // at the end, so a slot is revisited only after a full ring cycle.
        assert_eq!(next_slot(0, 3), 1);
        assert_eq!(next_slot(1, 3), 2);
        assert_eq!(next_slot(2, 3), 0);
        // A degenerate single-slot ring always returns slot 0.
        assert_eq!(next_slot(0, 1), 0);
    }

    #[test]
    fn pack_instance_transform_transposes_column_major_to_3x4_row_major() {
        // A column-major model with a known translation column [10, 20, 30].
        let model = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [10.0, 20.0, 30.0, 1.0],
        ];
        let t = pack_instance_transform(model);
        // The DXR transform is 3x4 row-major (flat); the translation is the
        // last entry of each 4-wide row.
        assert_eq!(
            t,
            [
                1.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0, 20.0, 0.0, 0.0, 1.0, 30.0
            ]
        );
    }

    #[test]
    fn pack_instance_transform_preserves_a_rotation_shear() {
        // Distinct values in every cell so a row/col swap would be detectable.
        let model = [
            [1.0, 2.0, 3.0, 0.0],
            [4.0, 5.0, 6.0, 0.0],
            [7.0, 8.0, 9.0, 0.0],
            [10.0, 11.0, 12.0, 1.0],
        ];
        let t = pack_instance_transform(model);
        // Flat row-major: row r is [model[0][r], model[1][r], model[2][r], model[3][r]].
        assert_eq!(
            t,
            [
                1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0
            ]
        );
    }

    #[test]
    fn instance_desc_packs_id_and_full_mask() {
        let d = instance_desc(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            7,
            0xDEAD_BEEF,
        );
        // InstanceID in the low 24 bits, mask 0xFF in the high 8.
        assert_eq!(d._bitfield1 & 0x00FF_FFFF, 7);
        assert_eq!(d._bitfield1 >> 24, 0xFF);
        assert_eq!(d._bitfield2, 0);
        assert_eq!(d.AccelerationStructure, 0xDEAD_BEEF);
    }

    // The `SkinParams` layout test lives with the struct in
    // `concinnity_render::directx::uniforms`. The root-constant DWORD-count
    // cross-check stays here, where `SKIN_PARAMS_DWORDS` is defined.
    #[test]
    fn skin_params_dwords_matches_size() {
        assert_eq!(
            SKIN_PARAMS_DWORDS as usize,
            std::mem::size_of::<super::SkinParams>() / 4
        );
    }
}
