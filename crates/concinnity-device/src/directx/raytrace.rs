//! DXR (DirectX Raytracing) acceleration structures for the hardware ray-traced
//! reflection pass. Builds, from the shared static vertex / index buffers and the
//! `DrawObject` + `InstancedCluster` lists, the bottom- and top-level
//! acceleration structures (BLAS / TLAS) the inline-`RayQuery` reflection shader
//! traces against, plus a per-instance geometry table the shader uses to fetch
//! the hit triangle and shade it.
//!
//! One triangle BLAS per participating static object (over its slice of the
//! shared buffers) and one per instanced cluster; one TLAS instance per object
//! and one per cluster instance (transform = the object/instance model matrix,
//! `InstanceID` = the geometry-table index). The BLAS describe object-space
//! geometry and never change for a rigid transform; only the TLAS instance
//! transforms (and the geometry table's per-instance model matrices the shader
//! shades with) move when a prop moves.
//!
//! Mirrors `metal/raytrace.rs`. Skinned geometry is added per frame
//! (`rebuild_skinned`): a compute pass deforms each skinned object's bind-pose
//! vertices into a model-space buffer, one BLAS per skinned object is
//! built or refit over it, and the TLAS + geometry table are rebuilt over the
//! persistent static/cluster BLAS plus the skinned tail.
//!
//! Every resource those two per-frame paths write lives in a ring rather than
//! being allocated fresh: `skinned_ring` is one slot per frame in flight, keyed on
//! `frame_idx`, `static_ring` advances a cursor one slot per dynamic-transform
//! rebuild, and the build scratch both paths record over is one buffer per frame
//! in flight too (see `ScratchRing`). All are rewritten in place and grown only on
//! demand, so a steady scene allocates nothing after warm-up. The ring rule they
//! rest on is that the frame-begin fence wait retires a slot's previous writer
//! before the next one touches it -- sound for the skinned path because it runs on
//! EVERY frame, and for the static path because its cursor advances per rebuild
//! rather than per frame (a sparsely-moving scene traces one TLAS across many
//! frames, so a frame-keyed slot could be reused while a live trace still reads
//! it). See `SkinnedFrameRing` / `StaticFrameRing`. Only a topology refresh (rare
//! in a running scene, every frame under the `Rebuild` diagnostic) still
//! allocates fresh; its orphans and its own dedicated
//! scratch go to the allocator's deferred free (orphans waiting, when the refresh
//! built no TLAS, for the next TLAS to publish). The bookkeeping over all of
//! it -- which draws and clusters the BLAS cover, the instance and geometry-table
//! order, and when to update -- is the shared `AccelBook`.

use concinnity_core::gfx::render_types::{DrawObject, InstancedCluster, SkinnedDrawObject};
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::rt_accel::{
    AccelBook, EmptyHead, FrameRing, HeadRefresh, InstanceBlas, RefreshMode, RtStep, RtUpdate,
    ScratchRing, SeedSet, StaticRing, empty_head, seed_wanted,
};
use concinnity_core::render::rt_geom::{RtDynamicMode, instance_id_and_mask, pack_row_major_3x4};
use concinnity_core::render::rt_refit::{BlasUpdate, SkinnedRefit};
use concinnity_core::render::rt_topology::blas_vertex_count;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::com;
use super::context::{DxGeometry, FRAMES};
use super::error::{map_hresult, map_pso_hresult};
use super::texture::{create_uav_buffer, transition_barrier};
use crate::directx::builtin_shaders::CompileProgram;
use crate::directx::root_constants::{RootConstants, root_dwords};

// Byte stride of a `Vertex` in the shared vertex buffer (pos + normal + tangent
// + color + uv = 14 floats). The BLAS reads positions at this stride and the
// shader fetches attributes at this stride. The deformed (posed) skinned vertex
// buffer the skin kernel writes carries the same 56-byte layout.
const VERTEX_STRIDE: u64 = 56;

// Shared with the Metal and Vulkan hosts: one `.hlsl` declares it.
use concinnity_core::render::uniforms::SkinParams;

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
        Transform: pack_row_major_3x4(model),
        // InstanceID in the low 24 bits, InstanceMask (0xFF) in the high 8.
        _bitfield1: instance_id_and_mask(instance_id),
        // InstanceContributionToHitGroupIndex (24) + Flags (8), both zero.
        _bitfield2: 0,
        AccelerationStructure: blas_gva,
    }
}

// The GPU virtual address of the BLAS an instance the book laid out references:
// `fresh` holds the addresses of the BLAS a refresh in progress builds, indexed by
// head slot, and `skinned` this frame's skinned BLAS. A missing one reads as 0,
// which DXR treats as an inactive instance.
fn instance_blas_gva(
    blas: InstanceBlas<'_, ID3D12Resource>,
    fresh: &[u64],
    skinned: &[u64],
) -> u64 {
    match blas {
        InstanceBlas::Head { blas, .. } => com::gpu_va(blas),
        InstanceBlas::Fresh { index } => fresh.get(index).copied().unwrap_or(0),
        InstanceBlas::Skinned { n } => skinned.get(n).copied().unwrap_or(0),
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

// The shared static vertex / index buffers every draw-object and cluster BLAS is
// built over, read from the live `DxGeometry` at each build: chunk streaming and
// geometry rebuilds replace both buffers, so an address or vertex count captured
// earlier can point at a released heap or bound streamed geometry short.
#[derive(Clone, Copy)]
pub(super) struct SharedGeometry {
    vertex_gva: u64,
    index_gva: u64,
    // Vertices the vertex buffer holds, streaming headroom included.
    vertex_count: u64,
}

impl SharedGeometry {
    pub(super) fn of(geometry: &DxGeometry) -> Self {
        Self {
            vertex_gva: com::gpu_va(&geometry.vertex_buffer),
            index_gva: com::gpu_va(&geometry.index_buffer),
            vertex_count: u64::from(geometry.vertex_buffer_view.SizeInBytes) / VERTEX_STRIDE,
        }
    }

    // The BLAS geometry for one draw object's slice. Its indices are offset by
    // `base_vertex`, which folds into the vertex start address.
    fn draw_geometry(&self, obj: &DrawObject) -> D3D12_RAYTRACING_GEOMETRY_DESC {
        let base_vertex = u64::try_from(obj.base_vertex).unwrap_or(0);
        triangle_geometry(
            self.vertex_gva + base_vertex * VERTEX_STRIDE,
            blas_vertex_count(obj.base_vertex, self.vertex_count),
            self.index_gva + obj.index_offset as u64 * 4,
            obj.index_count as u32,
        )
    }

    // The BLAS geometry for one instanced cluster (absolute indices).
    fn cluster_geometry(&self, cluster: &InstancedCluster) -> D3D12_RAYTRACING_GEOMETRY_DESC {
        triangle_geometry(
            self.vertex_gva,
            blas_vertex_count(0, self.vertex_count),
            self.index_gva + cluster.index_offset as u64 * 4,
            cluster.index_count as u32,
        )
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
fn create_as_buffer(device: &ID3D12Device, size: u64) -> RenderResult<ID3D12Resource> {
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
fn create_scratch(device: &ID3D12Device, size: u64) -> RenderResult<ID3D12Resource> {
    create_uav_buffer(device, size.max(256), D3D12_RESOURCE_STATE_COMMON)
}

// Byte size of a scratch slot covering a build requiring `needed` bytes. D3D12
// buffers have a 256-byte minimum, so a slot never sizes below that; keeping the
// rule here is what lets `ScratchRing::ensure` compare a request against a slot's
// recorded capacity without under-counting the rounding `create_scratch` applies.
fn scratch_capacity(needed: u64) -> u64 {
    needed.max(256)
}

// This frame's build scratch (see `ScratchRing`), covering a build requiring
// `needed` bytes. A replaced buffer is released in place: its last writer was
// this slot's frame a full ring cycle ago, which the frame-begin fence wait
// (`FRAMES` deep) has retired. At most one path replaces a frame's slot per frame
// (`rebuild_tlas` and `rebuild_skinned` are mutually exclusive, and a topology
// refresh builds over its own dedicated scratch), so a replacement can never pull
// the buffer out from under a build already recorded this frame.
fn ensure_scratch(
    ring: &mut ScratchRing<ID3D12Resource>,
    device: &ID3D12Device,
    frame_idx: usize,
    needed: u64,
) -> RenderResult<u64> {
    ring.ensure(frame_idx, scratch_capacity(needed), |capacity| {
        create_scratch(device, capacity)
    })
    .map(com::gpu_va)
}

// Upload a `Copy` slice to a fresh UPLOAD-heap buffer (host-visible,
// GPU-readable). Used for the TLAS instance-descriptor buffer (read by the AS
// build) and the geometry table (read as a `StructuredBuffer` root SRV by the
// trace).
fn upload_slice<T: Copy>(
    alloc: &DeviceAllocator,
    data: &[T],
    label: &str,
) -> RenderResult<PooledBuffer> {
    let bytes = std::mem::size_of_val(data).max(16) as u64;
    let buf = alloc.alloc_buffer(
        bytes,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { buf.Map(0, None, Some(&mut ptr)) }
        .map_err(|e| map_hresult(e.code(), &format!("map {label}")))?;
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
// (`rt_skin.hlsl`): a root SRV for the bind-pose skinned vertices (t0), a root
// SRV for the per-object joint palette (t1), a root UAV for the deformed output
// (u0), and a 4-DWORD `SkinParams` root-constant block (b0). Built alongside the
// RT PSO and held on `RtAccelData`; mirrors Metal's `skin_pipeline`.
pub(super) struct SkinPipeline {
    pub(super) root_sig: ID3D12RootSignature,
    pub(super) pso: ID3D12PipelineState,
}

// Root signature for the `rt_skin` compute kernel: `SkinParams` root constants at
// b0, the skinned vertex buffer as a root SRV (t0), the joint palette as a root
// SRV (t1), the deformed output as a root UAV (u0), the morph deltas as a root
// SRV (t2), and the morph weights as a root SRV (t3).
fn create_skin_root_signature(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    let params = [
        // [0] b0 SkinParams root constants
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: root_dwords::<SkinParams>(),
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

// Build the `rt_skin` compute pipeline (root signature + PSO). dxc emits it
// as `cs_6_5` DXIL, the same SM the RT reflection shader needs. Returns `Err`
// when the kernel fails to compile; the caller then leaves the skin pipeline
// `None` and skinned geometry is absent from the BVH (the RT pass still runs for
// static geometry).
fn build_skin_pipeline(device: &ID3D12Device, hot_reload: bool) -> RenderResult<SkinPipeline> {
    let cs = super::builtin_shaders::RT_SKIN.compile(hot_reload)?;
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
        .map_err(|e| map_pso_hresult(e.code(), "create rt skin PSO"))?;
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
    instance: Option<PooledBuffer>,
    instance_cap: u64,
    geom: Option<PooledBuffer>,
    geom_cap: u64,
}

// One ring slot of the per-rebuild static-transform buffers (the TLAS + its
// instance descriptors + the geometry table). The dynamic-transform rebuild
// advances the ring cursor to the next slot each rebuild and reuses that slot's
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
) -> RenderResult<PooledBuffer> {
    let len_bytes = std::mem::size_of_val(data);
    let needed = upload_ring_size(data);
    if ring_slot_needs_grow(slot.is_some(), *cap, needed) {
        *slot = Some(
            alloc
                .alloc_buffer(
                    needed,
                    D3D12_HEAP_TYPE_UPLOAD,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                )
                .map_err(|e| e.context(label))?,
        );
        *cap = needed;
    }
    let buf = slot.clone().ok_or_else(missing_slot_buffer)?;
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and the source
    // is a separate allocation, so the ranges cannot overlap.
    unsafe {
        buf.Map(0, None, Some(&mut ptr))
            .map_err(|e| map_hresult(e.code(), &format!("{label} map")))?;
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr as *mut u8, len_bytes);
        buf.Unmap(0, None);
    }
    Ok(buf)
}

// The bytes an upload ring slot holding `data` needs: at least one whole
// element, so a shader reading it as an array (the geometry table of a
// zero-instance TLAS) sees a full entry, and never under 4.
fn upload_ring_size<T>(data: &[T]) -> u64 {
    (std::mem::size_of_val(data).max(std::mem::size_of::<T>()) as u64).max(4)
}

// Ensure a ring slot holds an acceleration-structure buffer of at least `needed`
// bytes, replacing it only when it is missing or too small, and hand it back.
fn ensure_as_buffer(
    slot: &mut Option<ID3D12Resource>,
    cap: &mut u64,
    device: &ID3D12Device,
    needed: u64,
) -> RenderResult<ID3D12Resource> {
    if ring_slot_needs_grow(slot.is_some(), *cap, needed) {
        *slot = Some(create_as_buffer(device, needed)?);
        *cap = needed;
    }
    slot.clone().ok_or_else(missing_slot_buffer)
}

fn missing_slot_buffer() -> RenderError {
    RenderError::Other("RT ring slot is missing a buffer it was just sized for".into())
}

// The DXR acceleration structures + geometry table for hardware ray tracing.
// Held on the context behind an `Option`; present only when RT reflections are
// enabled, the GPU supports the DXR tier, and the scene has resident geometry.
pub(super) struct RtAccelData {
    // The persistent static + cluster BLAS (the book's head, built once and
    // never rebuilt: a rigid transform leaves object-space geometry unchanged)
    // plus, as the book's tail, copies (AddRefs) of this frame's skinned BLAS,
    // which their `skinned_ring` slot owns. Also the order the TLAS instances and
    // geometry table follow, and the update policy over them.
    book: AccelBook<ID3D12Resource, D3D12_RAYTRACING_INSTANCE_DESC>,
    // The top-level (instance) acceleration structure the trace reads.
    tlas: ID3D12Resource,
    // `[RtGeomEntry; instance_count]` (UPLOAD heap), bound as a `StructuredBuffer`
    // root SRV; indexed by the trace's instance id.
    geom_table: PooledBuffer,
    // The TLAS instance-descriptor buffer (UPLOAD heap). Only the TLAS *build*
    // reads it; a clone of the live `static_ring` / `skinned_ring` slot's buffer.
    instance_buffer: PooledBuffer,
    // Build scratch, one buffer per frame in flight (see `ScratchRing`). Sized at
    // init for the largest of every BLAS build and the TLAS build; each frame's
    // own slot grows on demand when a later build outgrows it.
    scratch: ScratchRing<ID3D12Resource>,
    // Size the TLAS prebuild reported; the static rebuild grows the ring slot's
    // TLAS to this size (once, since the static instance count is fixed).
    tlas_size: u64,

    // Per-rebuild static-transform buffers (see `StaticFrameRing`), reused in
    // place by the static `rebuild_tlas` path. The cursor advances one slot per
    // rebuild; a slot is revisited only after a full ring cycle, so its prior
    // trace has retired. The skinned path uses `skinned_ring` instead.
    static_ring: StaticRing<StaticFrameRing>,

    // Per-frame skinned-rebuild buffers, one slot per frame in flight, reused in
    // place and grown on demand (see `SkinnedFrameRing`). Indexed by the frame's
    // `frame_idx`.
    skinned_ring: FrameRing<SkinnedFrameRing>,

    // Skinned geometry.
    // The compute-skinning pipeline (`rt_skin`). `Some` only when the kernel
    // compiled; without it skinned geometry is absent from the BVH.
    skin: Option<SkinPipeline>,
    // The deformed (posed) skinned vertex buffer the skin pass writes and the
    // skinned BLAS + reflection trace read, owned by the `skinned_ring` slot that
    // last rebuilt it. A 1-element dummy when the scene has no skinned geometry,
    // so the trace's t8 binding is always valid.
    deformed_verts: ID3D12Resource,
    // GPU virtual address of the shared skinned index buffer the skinned BLAS
    // + trace address the deformed buffer with. A dummy buffer's GVA when there
    // is no skinned geometry, so the t9 binding is always valid. Cloned here so
    // the trace encoder can bind it.
    skinned_indices: PooledBuffer,

    // Persistent CPU scratch for the skinned rebuild.
    skinned_scratch: SkinnedScratch,
}

// The skinned rebuild's per-frame lists the book does not keep, held on the accel
// so their capacity is reused from frame to frame.
#[derive(Default)]
struct SkinnedScratch {
    // This frame's skinned geometry descriptors, parallel to the book's selected
    // skinned objects. Held across the sizing and recording loops, which both
    // point build inputs at it.
    geo: Vec<D3D12_RAYTRACING_GEOMETRY_DESC>,
    // GPU virtual addresses of this frame's skinned BLAS, in the same order.
    blas_gvas: Vec<u64>,
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
) -> RenderResult<SkinPipeline> {
    build_skin_pipeline(device, hot_reload)
}

// Geometry + counts the RT acceleration-structure build reads.
#[derive(Clone, Copy)]
pub(super) struct RtInitGeometry<'a> {
    // Placement pool the build allocates through; also carries the device and
    // the command queue the build records and fences on.
    pub alloc: &'a DeviceAllocator,
    // The shared vertex / index buffers the draw and cluster BLAS read.
    pub shared: SharedGeometry,
    // Every participating draw object (filtered by residency + index count inside).
    pub draw_objects: &'a [DrawObject],
    // Every participating instanced cluster.
    pub clusters: &'a [InstancedCluster],
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
    // The live shared buffers a topology refresh builds new draw BLAS over.
    pub shared: SharedGeometry,
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
// `albedo_count` is the shared pool's real-texture count, used to resolve each
// geometry's albedo / normal pool indices (the flat-normal fallback sits at
// `albedo_count`) for the RT hit shader.
pub(super) fn build_rt_accel(geometry: RtInitGeometry) -> RenderResult<Option<RtAccelData>> {
    let RtInitGeometry {
        alloc,
        shared,
        draw_objects,
        clusters,
        albedo_count,
        exclude_seethrough,
    } = geometry;
    let device = alloc.device();
    let queue = alloc.queue();
    let device5: ID3D12Device5 = device
        .cast()
        .map_err(|e| map_hresult(e.code(), "ID3D12Device5 cast (DXR unsupported?)"))?;

    // Participating static objects + clusters (real triangles, resident, and not
    // rerouted to the see-through transparent path).
    let seed = SeedSet::new(draw_objects, clusters, exclude_seethrough);
    if seed.is_empty() {
        return Ok(None);
    }

    // One geometry desc per BLAS: participating objects first, then clusters.
    let geo_descs: Vec<D3D12_RAYTRACING_GEOMETRY_DESC> = seed
        .objects
        .iter()
        .map(|&i| shared.draw_geometry(&draw_objects[i]))
        .chain(seed.clusters.iter().map(|c| shared.cluster_geometry(c)))
        .collect();

    // Size + allocate each BLAS; track the largest scratch requirement.
    let mut blas: Vec<ID3D12Resource> = Vec::with_capacity(geo_descs.len());
    let mut max_scratch: u64 = 0;
    for geo in &geo_descs {
        let inputs = blas_inputs(geo);
        let info = prebuild_info(&device5, &inputs);
        blas.push(create_as_buffer(device, info.ResultDataMaxSizeInBytes)?);
        max_scratch = max_scratch.max(info.ScratchDataSizeInBytes);
    }

    // Instance descriptors + geometry table, in the book's instance order.
    let mut book = AccelBook::new(&seed, blas, draw_objects, albedo_count)?;
    book.fill_instances(draw_objects, None, |model, id, blas| {
        instance_desc(model, id, instance_blas_gva(blas, &[], &[]))
    });
    let instance_count = book.instances().len() as u32;
    let instance_buffer = upload_slice(alloc, book.instances(), "RT instance descriptors")?;
    let geom_table = upload_slice(alloc, book.geom_table(), "RT geometry table")?;

    // Size + allocate the TLAS + the shared scratch (>= the largest BLAS/TLAS).
    let tlas_pre = prebuild_info(&device5, &tlas_inputs(instance_count, 0));
    max_scratch = max_scratch.max(tlas_pre.ScratchDataSizeInBytes);
    let tlas = create_as_buffer(device, tlas_pre.ResultDataMaxSizeInBytes)?;
    // One scratch buffer per frame in flight. The init builds below are their own
    // fence-waited submit, so they record over slot 0 before any frame exists.
    let scratch = ScratchRing::filled(FRAMES, scratch_capacity(max_scratch), |capacity| {
        create_scratch(device, capacity)
    })?;
    let scratch_gva = scratch.get(0).map_or(0, com::gpu_va);

    // Record every BLAS build (UAV-barrier-serialized over the shared scratch),
    // then the TLAS build, on a one-shot command list; fence-wait so the BVH is
    // ready before the first trace.
    // SAFETY: the command list is in the recording state, and every resource, descriptor and slice
    // these commands name is live for the call.
    record_builds(alloc, queue, |cmd4| unsafe {
        for (dest, geo) in book.head().iter().zip(&geo_descs) {
            let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: com::gpu_va(dest),
                Inputs: blas_inputs(geo),
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: scratch_gva,
            };
            cmd4.BuildRaytracingAccelerationStructure(&desc, None);
            cmd4.ResourceBarrier(&[uav_barrier()]);
        }
        let tlas_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: com::gpu_va(&tlas),
            Inputs: tlas_inputs(instance_count, com::gpu_va(&instance_buffer)),
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: scratch_gva,
        };
        cmd4.BuildRaytracingAccelerationStructure(&tlas_desc, None);
    })?;

    // Skinned geometry is seeded on the first dynamic frame (like Metal), so the
    // init build is static-only. Allocate dummy deformed-vertex / skinned-index
    // buffers so the trace's t8/t9 root SRVs always bind a valid resource; the
    // first `rebuild_skinned` replaces the deformed buffer with the real one.
    // D3D12 buffers are always created in COMMON regardless of the requested
    // state (so pass COMMON to avoid the debug-layer "Ignoring InitialState"
    // warning); COMMON implicitly promotes to a shader-read state on the trace's
    // first t8/t9 access, so the dummies need no transition.
    let deformed_verts = create_uav_buffer(device, VERTEX_STRIDE, D3D12_RESOURCE_STATE_COMMON)?;
    let skinned_indices =
        alloc.alloc_buffer(4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON)?;

    // Seed ring slot 0 with the init structures so the static-transform rebuild
    // path reuses them in place; the live `tlas` / `geom_table` / `instance_buffer`
    // fields hold a parallel clone (AddRef), so slot 0's resources stay alive until
    // the cursor wraps back to it a full ring cycle later. The remaining slots fill
    // lazily on their first rebuild.
    let static_ring = StaticRing::new(
        FRAMES,
        StaticFrameRing {
            tlas: Some(tlas.clone()),
            tlas_cap: tlas_pre.ResultDataMaxSizeInBytes.max(256),
            instance: Some(instance_buffer.clone()),
            instance_cap: (std::mem::size_of_val(book.instances()) as u64).max(16),
            geom: Some(geom_table.clone()),
            geom_cap: (std::mem::size_of_val(book.geom_table()) as u64).max(16),
        },
    );

    Ok(Some(RtAccelData {
        book,
        tlas,
        geom_table,
        instance_buffer,
        scratch,
        tlas_size: tlas_pre.ResultDataMaxSizeInBytes,
        static_ring,
        skinned_ring: FrameRing::new(FRAMES),
        skin: None,
        deformed_verts,
        skinned_indices,
        skinned_scratch: SkinnedScratch::default(),
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
) -> RenderResult<()>
where
    F: FnOnce(&ID3D12GraphicsCommandList4),
{
    let device = alloc.device();
    let alloc: ID3D12CommandAllocator =
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
            .map_err(|e| map_hresult(e.code(), "RT build allocator"))?;
    let cmd: ID3D12GraphicsCommandList =
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None) }
            .map_err(|e| map_hresult(e.code(), "RT build cmd list"))?;
    let cmd4: ID3D12GraphicsCommandList4 = cmd
        .cast()
        .map_err(|e| map_hresult(e.code(), "ID3D12GraphicsCommandList4 cast"))?;

    record(&cmd4);

    // SAFETY: the command list is live and in the recording state, which is what `Close` requires.
    unsafe { cmd.Close() }.map_err(|e| map_hresult(e.code(), "RT build close"))?;
    let list: ID3D12CommandList = cmd
        .cast()
        .map_err(|e| map_hresult(e.code(), "RT build cast"))?;
    // SAFETY: every command list in the submission is live and closed, and the slice outlives the
    // call.
    unsafe { queue.ExecuteCommandLists(&[Some(list)]) };

    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
        .map_err(|e| map_hresult(e.code(), "RT build fence"))?;
    let event =
        // SAFETY: an auto-reset, initially unsignaled event with no name and no security
        // attributes; the call borrows nothing.
        unsafe { windows::Win32::System::Threading::CreateEventW(None, false, false, None) }
            .map_err(|e| map_hresult(e.code(), "RT build event"))?;
    // SAFETY: the fence and the event were created from this device and are live for the call.
    unsafe { queue.Signal(&fence, 1) }.map_err(|e| map_hresult(e.code(), "RT build signal"))?;
    // SAFETY: the fence and the event were created from this device and are live for the call.
    if unsafe { fence.GetCompletedValue() } < 1 {
        // SAFETY: the fence and the event were created from this device and are live for the call.
        unsafe { fence.SetEventOnCompletion(1, event) }
            .map_err(|e| map_hresult(e.code(), "RT build set event"))?;
        // SAFETY: `event` is the handle created above and is still open.
        unsafe { windows::Win32::System::Threading::WaitForSingleObject(event, u32::MAX) };
    }
    // SAFETY: `event` was created above, every wait on it has returned, and it is closed exactly
    // once.
    unsafe { windows::Win32::Foundation::CloseHandle(event) }.ok();
    Ok(())
}

// What one topology refresh needs beyond the allocator, the command list and the
// draw list: the buffers new draw BLAS are built over, the BVH membership rule,
// and whether unchanged BLAS are reused.
#[derive(Clone, Copy)]
struct TopologyRefresh {
    shared: SharedGeometry,
    exclude_seethrough: bool,
    mode: RefreshMode,
}

impl RtAccelData {
    // Per-frame dynamic update, recorded onto `cmd` (the frame's "start" cmd
    // list, submitted before every per-pass trace on the serial DIRECT queue),
    // following the book's plan: refresh the draw BLAS head when the
    // participating draw set changed (a cloned prop, a streamed chunk
    // added/removed), then re-skin, rebuild the TLAS, or keep it. Both rebuild
    // paths reuse ring buffers in place (`static_ring` / `skinned_ring`), so the
    // steady state allocates nothing. A failure is non-fatal: the live BVH is
    // kept and the first error comes back for the caller to report; a failed
    // refresh still lets the step after it run. Once the last draw and cluster
    // geometry is gone the BVH follows `empty_head`, and the caller drops a
    // spent one.
    //
    // `frame_idx` selects the per-frame joint buffer the skin dispatch reads;
    // `skinned`, when present, carries this frame's skinned-geometry inputs.
    pub(super) fn dynamic_update(
        &mut self,
        alloc: &DeviceAllocator,
        cmd: &ID3D12GraphicsCommandList,
        draw_objects: &[DrawObject],
        inputs: RtDynamicInputs,
    ) -> RenderResult<RtUpdate> {
        let RtDynamicInputs {
            mode,
            skinned,
            frame_idx,
            shared,
            topology_dirty,
            exclude_seethrough,
        } = inputs;
        // Skinned geometry takes part only with the skin pipeline (the kernel
        // compiled); without it the static path runs.
        let skinned = skinned.filter(|_| self.skin.is_some());
        self.book.tick();
        let Some(plan) = self
            .book
            .plan(mode, topology_dirty, skinned.as_ref().map(|s| s.objects))
        else {
            return Ok(RtUpdate::Done);
        };

        // Fold any added/removed/cloned draw geometry into the BLAS head + rebuild
        // the static TLAS FIRST. On the skinned path `rebuild_skinned` below then
        // overlays the skinned tail on top.
        let mut refreshed = Ok(());
        if let Some(mode) = plan.refresh {
            let req = TopologyRefresh {
                shared,
                exclude_seethrough,
                mode,
            };
            let if_empty = empty_head(plan.skinned, skinned.is_some());
            refreshed = self.refresh_topology(alloc, cmd, draw_objects, req, if_empty);
            if refreshed.is_err() {
                self.book.owe_refresh();
            }
        }

        let stepped = match self.book.next_step(mode, &plan, draw_objects) {
            RtStep::Keep => RtUpdate::Done,
            // Nothing static is left and no skinned geometry can rejoin: stop
            // publishing the skinned tail, which leaves the BVH spent for the
            // caller to drop.
            RtStep::Tlas if self.book.is_empty() && skinned.is_none() => {
                self.book.release_skinned();
                RtUpdate::Done
            }
            // An empty head still gets a (zero-instance) TLAS, so the trace stops
            // reaching what left.
            RtStep::Tlas => {
                self.rebuild_tlas(alloc, cmd, draw_objects, frame_idx)?;
                RtUpdate::Done
            }
            RtStep::Skinned => match skinned {
                Some(s) => self.rebuild_skinned(SkinnedRebuild {
                    alloc,
                    cmd,
                    draw_objects,
                    skinned: &s,
                    frame_idx,
                    full_build: plan.full_skinned_build,
                })?,
                None => RtUpdate::Done,
            },
        };
        refreshed.map(|()| stepped)
    }

    // Whether the BVH has nothing left to trace and nothing that could rejoin
    // it: the last draw and cluster geometry is gone, no skinned geometry is
    // published, and `skinned_present` says none exists to publish.
    pub(super) fn is_spent(&self, skinned_present: bool) -> bool {
        self.book.is_empty()
            && !self.book.has_skinned()
            && empty_head(false, skinned_present && self.skin.is_some()) == EmptyHead::Drop
    }

    // Bring the draw-object BLAS head in line with the current participating
    // draw set: reuse every unchanged BLAS (or none, under
    // `RefreshMode::RebuildAll`), build only the new / changed ones, retire the
    // orphans. The cluster BLAS are kept verbatim; any skinned tail is dropped
    // (its BLAS live in `skinned_ring`, so releasing the copy frees nothing in
    // flight, and `rebuild_skinned` re-adds the tail this frame on the skinned
    // path). The TLAS + geometry table are ALWAYS rebuilt inline over [refreshed
    // head + clusters], recycling the next `static_ring` slot like `rebuild_tlas`
    // -- even on the skinned path, where `rebuild_skinned` then overlays the
    // skinned tail on top. Rebuilding the static TLAS here keeps two invariants
    // the caller relies on: `self.tlas` is replaced with a structure that does NOT
    // reference the orphaned BLAS before they are retired (so a failing / skipped
    // `rebuild_skinned` can never leave the trace reading a freed orphan), and
    // `self.tlas_size` tracks the current static instance count (so a later
    // static `rebuild_tlas` does not under-size the ring TLAS).
    //
    // Recorded onto `cmd` (the frame's start cmd list), so the builds order before
    // this frame's trace by submission (no fence-wait, no stall). The orphaned
    // draw BLAS + the dedicated build scratch go to the allocator's deferred free:
    // the just-replaced TLAS an in-flight prior frame still traces references the
    // orphans, and the just-recorded builds keep reading the scratch after this
    // returns. Every `self` mutation is deferred past all fallible allocations,
    // so a mid-refresh failure leaves the live BVH untouched.
    //
    // A refresh that leaves no draw or cluster geometry follows `if_empty`. With
    // skinned geometry following, it commits the empty head and parks the
    // orphans until the skinned TLAS this frame publishes. With skinned geometry
    // that could rejoin, it builds a zero-instance static TLAS like any other
    // refresh. With none, it empties the book for the caller to drop the whole
    // BVH, orphans included, through a deferred free.
    fn refresh_topology(
        &mut self,
        alloc: &DeviceAllocator,
        cmd: &ID3D12GraphicsCommandList,
        draw_objects: &[DrawObject],
        req: TopologyRefresh,
        if_empty: EmptyHead,
    ) -> RenderResult<()> {
        let refresh = self
            .book
            .plan_refresh(draw_objects, req.exclude_seethrough, req.mode);
        if if_empty != EmptyHead::Build && self.book.refresh_leaves_nothing(&refresh) {
            let orphans = self.book.commit_refresh(refresh, Vec::new(), draw_objects);
            self.book.park(orphans);
            if if_empty == EmptyHead::Drop {
                self.book.release_skinned();
            }
            return Ok(());
        }
        // The slot after the live one becomes live only when the refresh publishes.
        let (next, mut slot) = self.static_ring.take_next();
        let result = self.refresh_topology_into(alloc, cmd, draw_objects, refresh, req, &mut slot);
        if result.is_ok() {
            self.static_ring.publish(next, slot);
        } else {
            self.static_ring.put(next, slot);
        }
        result
    }

    fn refresh_topology_into(
        &mut self,
        alloc: &DeviceAllocator,
        cmd: &ID3D12GraphicsCommandList,
        draw_objects: &[DrawObject],
        refresh: HeadRefresh,
        req: TopologyRefresh,
        slot: &mut StaticFrameRing,
    ) -> RenderResult<()> {
        let device = alloc.device();
        let device5: ID3D12Device5 = device
            .cast()
            .map_err(|e| map_hresult(e.code(), "ID3D12Device5 cast (topology refresh)"))?;
        let cmd4: ID3D12GraphicsCommandList4 = cmd.cast().map_err(|e| {
            map_hresult(
                e.code(),
                "ID3D12GraphicsCommandList4 cast (topology refresh)",
            )
        })?;

        // Allocate a fresh BLAS for every slot the plan could not reuse. `fresh_builds`
        // holds the geometry desc + its dest BLAS so the builds can be recorded below
        // (after the shared scratch is sized over all of them + the TLAS).
        let mut fresh: Vec<Option<ID3D12Resource>> =
            (0..refresh.indices().len()).map(|_| None).collect();
        let mut fresh_builds: Vec<(D3D12_RAYTRACING_GEOMETRY_DESC, ID3D12Resource)> = Vec::new();
        let mut max_scratch: u64 = 0;
        for (j, idx) in refresh.fresh_slots() {
            let geo = req.shared.draw_geometry(&draw_objects[idx]);
            let info = prebuild_info(&device5, &blas_inputs(&geo));
            let blas = create_as_buffer(device, info.ResultDataMaxSizeInBytes)?;
            max_scratch = max_scratch.max(info.ScratchDataSizeInBytes);
            fresh_builds.push((geo, blas.clone()));
            fresh[j] = Some(blas);
        }
        let fresh_gvas: Vec<u64> = fresh
            .iter()
            .map(|b| b.as_ref().map_or(0, com::gpu_va))
            .collect();

        // Static TLAS + geometry table over [refreshed draw head + clusters].
        self.book
            .fill_refresh_instances(&refresh, draw_objects, |model, id, blas| {
                instance_desc(model, id, instance_blas_gva(blas, &fresh_gvas, &[]))
            });
        let instance_count = self.book.instances().len() as u32;
        let tlas_pre = prebuild_info(&device5, &tlas_inputs(instance_count, 0));
        max_scratch = max_scratch.max(tlas_pre.ScratchDataSizeInBytes);
        let tlas_needed = tlas_pre.ResultDataMaxSizeInBytes;

        // A single dedicated scratch covers every fresh BLAS build + the TLAS build;
        // retired below (the async builds keep reading it after this returns).
        let scratch = create_scratch(device, scratch_capacity(max_scratch))?;
        let scratch_gva = com::gpu_va(&scratch);

        // Recycle this static ring slot (last live a full cycle ago, so its trace
        // has retired), growing it to this refresh's sizes.
        let instance_buffer = write_upload_ring(
            &mut slot.instance,
            &mut slot.instance_cap,
            alloc,
            self.book.instances(),
            "RT instance descriptors",
        )?;
        let geom_table = write_upload_ring(
            &mut slot.geom,
            &mut slot.geom_cap,
            alloc,
            self.book.geom_table(),
            "RT geometry table",
        )?;
        let tlas = ensure_as_buffer(&mut slot.tlas, &mut slot.tlas_cap, device, tlas_needed)?;
        // The commit below must follow the builds recorded next, so it is checked
        // now, while a failure still leaves nothing recorded.
        self.book.check_refresh(&refresh, &fresh)?;

        // Record the fresh draw-BLAS builds (UAV-barrier-serialized over the shared
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
                Inputs: tlas_inputs(instance_count, com::gpu_va(&instance_buffer)),
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: scratch_gva,
            };
            cmd4.BuildRaytracingAccelerationStructure(&desc, None);
            cmd.ResourceBarrier(&[uav_barrier()]);
        }

        // Commit: swap in the refreshed head and the static TLAS over it. Any
        // skinned tail is dropped; `rebuild_skinned` re-adds it on the skinned path
        // this same frame, replacing this static TLAS with a static+skinned one.
        alloc.retire(scratch);
        let orphans = self.book.commit_refresh(refresh, fresh, draw_objects);
        for orphan in orphans.into_iter().chain(self.book.take_parked()) {
            alloc.retire(orphan);
        }
        self.book.release_skinned();
        self.skinned_ring.unpublish(self.book.clock());
        // The skinned tail is gone, so no ring slot's refit bookkeeping describes a
        // published tree any more. On the skinned path `rebuild_skinned` re-adds the
        // tail this same frame and rebuilds it from scratch, which is also the right
        // answer for the change that triggered this refresh.
        for ring in self.skinned_ring.slots_mut() {
            ring.refit.reset();
        }
        self.tlas = tlas;
        self.geom_table = geom_table;
        self.instance_buffer = instance_buffer;
        self.tlas_size = tlas_needed;
        Ok(())
    }

    // Rebuild the TLAS + geometry table from the transforms the book collected,
    // reusing the next `static_ring` slot's buffers in place, and record the
    // build onto `cmd`. The BLAS are kept (rigid transforms leave object-space
    // geometry unchanged).
    fn rebuild_tlas(
        &mut self,
        alloc: &DeviceAllocator,
        cmd: &ID3D12GraphicsCommandList,
        draw_objects: &[DrawObject],
        frame_idx: usize,
    ) -> RenderResult<()> {
        // Take the slot after the live one and reuse its buffers in place. It was
        // last live a full ring of publishes ago, so the frame-begin fence wait has
        // retired every trace that read it. It is put back on every exit path and
        // becomes the live slot only on success.
        let (next, mut slot) = self.static_ring.take_next();
        let result = self.rebuild_tlas_into(alloc, cmd, draw_objects, frame_idx, &mut slot);
        if result.is_ok() {
            self.static_ring.publish(next, slot);
            for orphan in self.book.take_parked() {
                alloc.retire(orphan);
            }
        } else {
            self.static_ring.put(next, slot);
        }
        result
    }

    fn rebuild_tlas_into(
        &mut self,
        alloc: &DeviceAllocator,
        cmd: &ID3D12GraphicsCommandList,
        draw_objects: &[DrawObject],
        frame_idx: usize,
        slot: &mut StaticFrameRing,
    ) -> RenderResult<()> {
        let device = alloc.device();
        let device5: ID3D12Device5 = device
            .cast()
            .map_err(|e| map_hresult(e.code(), "ID3D12Device5 cast (rebuild)"))?;
        // Freshly-transformed draw-object instances, then the cluster instances.
        // The geometry table mirrors this order.
        self.book
            .fill_instances(draw_objects, None, |model, id, blas| {
                instance_desc(model, id, instance_blas_gva(blas, &[], &[]))
            });
        let instance_count = self.book.instances().len() as u32;

        // The static instance count is fixed, so the upload buffers + TLAS are
        // reused without growing after warm-up.
        let instance_buffer = write_upload_ring(
            &mut slot.instance,
            &mut slot.instance_cap,
            alloc,
            self.book.instances(),
            "RT instance descriptors",
        )?;
        let geom_table = write_upload_ring(
            &mut slot.geom,
            &mut slot.geom_cap,
            alloc,
            self.book.geom_table(),
            "RT geometry table",
        )?;
        let tlas = ensure_as_buffer(&mut slot.tlas, &mut slot.tlas_cap, device, self.tlas_size)?;

        // Ensure this frame's scratch slot covers this TLAS build. The instance
        // count is fixed between topology refreshes, so after warm-up this reuses
        // the slot in place; a refresh that grew the count is what makes the
        // init-time size too small, and asking the prebuild each rebuild is what
        // keeps the ring self-sufficient without the refresh path touching it.
        let scratch_needed =
            prebuild_info(&device5, &tlas_inputs(instance_count, 0)).ScratchDataSizeInBytes;
        let scratch_gva = ensure_scratch(&mut self.scratch, device, frame_idx, scratch_needed)?;

        let cmd4: ID3D12GraphicsCommandList4 = cmd
            .cast()
            .map_err(|e| map_hresult(e.code(), "ID3D12GraphicsCommandList4 cast (rebuild)"))?;
        let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: com::gpu_va(&tlas),
            Inputs: tlas_inputs(instance_count, com::gpu_va(&instance_buffer)),
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: scratch_gva,
        };
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd4.BuildRaytracingAccelerationStructure(&desc, None);
            // Order the build before this frame's trace reads the TLAS / table.
            cmd.ResourceBarrier(&[uav_barrier()]);
        }

        // Point the live BVH at this slot's buffers (clones AddRef the slot's
        // resources, not GPU allocations). A skinned tail still published from a
        // prior skinned frame (the last skinned object just turned invisible)
        // drops back to the static head: the rebuilt TLAS no longer references
        // it, and the skinned BLAS resources persist in `skinned_ring`, so
        // dropping these copies frees nothing still in flight. A refit continues
        // the tree its last full build produced, so re-entering the skinned path
        // after an arbitrary gap must rebuild rather than refit from a pose the
        // tree was never fitted for.
        self.tlas = tlas;
        self.geom_table = geom_table;
        self.instance_buffer = instance_buffer;
        if self.book.commit_static().is_some() {
            self.skinned_ring.unpublish(self.book.clock());
            for ring in self.skinned_ring.slots_mut() {
                ring.refit.reset();
            }
        }
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
    // All per-frame buffers (deformed verts, skinned BLAS, TLAS, instance
    // descriptors, geometry table) live in `skinned_ring[frame_idx]`, and the
    // build scratch in the `ScratchRing` slot for the same frame; all are rebuilt
    // IN PLACE: they are allocated once and only grown when a larger size is
    // needed, so the steady state allocates nothing. Reuse is hazard-free because
    // the frame-begin fence wait gates this slot's prior GPU work (`FRAMES` deep),
    // so the prior frame's trace that read this slot has finished.
    //
    // The three GPU steps are recorded in dependency order on the one DIRECT cmd
    // list: skin dispatch (writes the deformed buffer), a UAV barrier + transition
    // to a shader-readable state, then the BLAS/TLAS build (reads it). The start
    // cmd list is submitted before every per-pass trace, so build -> trace is
    // ordered by submission too.
    fn rebuild_skinned(&mut self, req: SkinnedRebuild) -> RenderResult<RtUpdate> {
        // This frame slot's buffers, taken out for the duration (sidesteps the
        // `&mut self` borrow while the rebuild reads other fields) and put back
        // on every exit path, so a failed rebuild leaves the ring intact. A slot a
        // failed rebuild left live is still traced by the frames since, so this
        // frame skips rather than rewrite it.
        let frame_idx = req.frame_idx;
        let alloc = req.alloc;
        let now = self.book.clock();
        let Some(mut ring) = self.skinned_ring.take(frame_idx, now)? else {
            return Ok(RtUpdate::Skipped);
        };
        let result = self.rebuild_skinned_into(req, &mut ring);
        if result.is_ok() {
            self.skinned_ring.publish(frame_idx, ring, now);
            for orphan in self.book.take_parked() {
                alloc.retire(orphan);
            }
        } else {
            // A failure can leave freshly (re)allocated BLAS in the slot that no
            // build was recorded into, so the next visit must build, not refit.
            ring.refit.reset();
            self.skinned_ring.put(frame_idx, ring);
        }
        result.map(|()| RtUpdate::Done)
    }

    fn rebuild_skinned_into(
        &mut self,
        req: SkinnedRebuild,
        ring: &mut SkinnedFrameRing,
    ) -> RenderResult<()> {
        let SkinnedRebuild {
            alloc,
            cmd,
            draw_objects,
            skinned,
            frame_idx,
            full_build,
        } = req;
        let device = alloc.device();
        let device5: ID3D12Device5 = device
            .cast()
            .map_err(|e| map_hresult(e.code(), "ID3D12Device5 cast (skinned rebuild)"))?;
        let cmd4: ID3D12GraphicsCommandList4 = cmd.cast().map_err(|e| {
            map_hresult(
                e.code(),
                "ID3D12GraphicsCommandList4 cast (skinned rebuild)",
            )
        })?;

        // Deformed-vertex buffer (default heap, ALLOW_UNORDERED_ACCESS): the skin
        // pass writes posed `Vertex`s here, mirroring the skinned vertex buffer's
        // indexing so the index buffer addresses it directly. Sized to the
        // highest vertex the skinned objects reach, grown on demand. Created in
        // COMMON (D3D12 buffers always are); after its first rebuild it rests in
        // the combined shader-read state.
        let deformed_extent = self.book.skinned_vertex_extent(skinned.objects);
        self.book
            .fill_skinned_shapes(skinned.objects, deformed_extent as u32);
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
        let deformed_verts = ring.deformed.clone().ok_or_else(missing_slot_buffer)?;
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
        let skin = self.skin.as_ref().ok_or_else(|| {
            RenderError::Other("rebuild_skinned called without a skin pipeline".to_string())
        })?;
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.SetComputeRootSignature(&skin.root_sig);
            cmd.SetPipelineState(&skin.pso);
        }
        let visible = self.book.visible_skinned();
        for &obj_idx in visible {
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
                cmd.set_compute_root_constants(0, &params);
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
        let SkinnedScratch {
            geo: skinned_geo,
            blas_gvas: skinned_gvas,
        } = &mut self.skinned_scratch;
        skinned_geo.clear();
        skinned_geo.extend(visible.iter().map(|&i| {
            let obj = &skinned.objects[i];
            skinned_triangle_geometry(
                deformed_gva,
                deformed_extent as u32,
                skinned.index_gva + obj.index_offset as u64 * 4,
                obj.index_count as u32,
            )
        }));

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
        skinned_gvas.clear();
        skinned_gvas.extend(
            ring.blas
                .iter()
                .take(skinned_geo.len())
                .map(|(b, _)| com::gpu_va(b)),
        );

        // Instance descriptors + geometry table, in the book's instance order:
        // static objects (current transforms), the cluster instances, then one per
        // skinned object.
        self.book
            .fill_instances(draw_objects, Some(skinned.objects), |model, id, blas| {
                instance_desc(model, id, instance_blas_gva(blas, &[], skinned_gvas))
            });
        let instance_count = self.book.instances().len() as u32;
        let instance_buffer = write_upload_ring(
            &mut ring.instance,
            &mut ring.instance_cap,
            alloc,
            self.book.instances(),
            "RT instance descriptors",
        )?;
        let geom_table = write_upload_ring(
            &mut ring.geom,
            &mut ring.geom_cap,
            alloc,
            self.book.geom_table(),
            "RT geometry table",
        )?;

        // Size the TLAS in the ring, and fold its scratch requirement into
        // `max_scratch` (>= the largest skinned BLAS + the TLAS). The skinned
        // instance count can change frame to frame, so size the TLAS from this
        // frame's prebuild rather than the cached size.
        let tlas_pre = prebuild_info(&device5, &tlas_inputs(instance_count, 0));
        max_scratch = max_scratch.max(tlas_pre.ScratchDataSizeInBytes);
        let tlas = ensure_as_buffer(
            &mut ring.tlas,
            &mut ring.tlas_cap,
            device,
            tlas_pre.ResultDataMaxSizeInBytes,
        )?;
        // Ensure this frame's scratch slot covers the skinned BLAS builds/refits
        // plus this frame's TLAS.
        let scratch_gva = ensure_scratch(&mut self.scratch, device, frame_idx, max_scratch)?;

        // Settle build-or-refit last, once every fallible step above has passed:
        // recording a build the command list never gets would leave the slot
        // claiming a tree a later refit could not update.
        let update = ring
            .refit
            .plan(self.book.skinned_shapes(), storage_changed || full_build);

        // Record the skinned BLAS updates (UAV-barrier-serialized over the shared
        // scratch), then the TLAS build, on `cmd`. A `Build` is a full rebuild
        // (`SourceAccelerationStructureData = 0`) into the ring buffer, overwriting
        // the prior frame's structure; a `Refit` names that same structure as the
        // source, which DXR defines as an in-place update.
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            let SkinnedScratch { geo, blas_gvas } = &self.skinned_scratch;
            for (geo, &dest) in geo.iter().zip(blas_gvas) {
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
                Inputs: tlas_inputs(instance_count, com::gpu_va(&instance_buffer)),
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: scratch_gva,
            };
            cmd4.BuildRaytracingAccelerationStructure(&tlas_desc, None);
            // Order the TLAS build before this frame's trace reads it.
            cmd.ResourceBarrier(&[uav_barrier()]);
        }

        // Point the live BVH at this frame's ring buffers (clones are AddRefs on
        // the persistent ring resources, not GPU allocations). The static/cluster
        // head is untouched; only the skinned tail rotates.
        let count = self.skinned_scratch.geo.len();
        self.book
            .replace_tail(ring.blas.iter().take(count).map(|(b, _)| b.clone()));
        self.tlas = tlas;
        self.geom_table = geom_table;
        self.instance_buffer = instance_buffer;
        self.deformed_verts = deformed_verts;
        self.book.commit_skinned();
        Ok(())
    }
}

// Everything one skinned rebuild reads beyond the accel itself: the allocator,
// the command list it records onto, the frame's draw list + skinned inputs, and
// which ring slot to build into.
struct SkinnedRebuild<'a> {
    alloc: &'a DeviceAllocator,
    cmd: &'a ID3D12GraphicsCommandList,
    draw_objects: &'a [DrawObject],
    skinned: &'a SkinnedRtInputs<'a>,
    frame_idx: usize,
    // Build every skinned BLAS from scratch rather than refitting.
    full_build: bool,
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
        if self.state.skinned.draw_objects.is_empty() {
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
        for (i, obj) in self.state.skinned.draw_objects.iter().enumerate() {
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
                cmd.set_compute_root_constants(0, &params);
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
    // per-frame pose. Disjoint field borrows: `rt.accel` (mut) vs the rest
    // (shared); the joint GVAs are collected up-front so `skinned_joint_gva`'s
    // `&self` borrow does not overlap the `rt.accel` mutable borrow.
    //
    // Consumes `rt.topology_dirty` (set when a cloned prop / streamed chunk
    // altered the draw set): the accel's `dynamic_update` folds the change into
    // the BLAS head. When RT is on but the scene had no resident geometry at build
    // time (`rt.accel` is `None`), a topology change that introduces the first
    // participating geometry seeds the BVH from scratch here.
    pub(super) fn rt_dynamic_update(&mut self, cmd: &ID3D12GraphicsCommandList, frame_idx: usize) {
        let topology_dirty = std::mem::take(&mut self.state.gpu_dirty.rt_topology);
        // Drop the dropped BVHs no in-flight frame can trace any more: the
        // frame-begin fence wait bounds that at `FRAMES` frames, plus the one this
        // frame records.
        self.rt.retire_tick += 1;
        self.rt
            .retired
            .collect(self.rt.retire_tick, FRAMES as u64 + 1);

        // Seed-from-empty: RT enabled + a topology change added the first
        // participating geometry to a scene that had none at build time. The
        // one-shot build is fence-waited internally (a rare, one-time stall); the
        // DXR trace reads the TLAS + table by GPU virtual address each frame, so
        // the fresh accel is picked up with no descriptor rewire. The seed build
        // is static-only, so skinned geometry alone cannot seed one.
        if self.rt.accel.is_none() {
            if self.rt_reflections.is_some()
                && seed_wanted(self.rt.dynamic_mode, topology_dirty, false)
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
                if self.rt.skinned_geometry && !self.state.skinned.draw_objects.is_empty() =>
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

        // Read before `rt.accel` is borrowed mutably below.
        let exclude_seethrough = self.seethrough_meshes_enabled();
        let shared = SharedGeometry::of(&self.scene.geometry);

        let Some(accel) = self.rt.accel.as_mut() else {
            return;
        };
        let skinned = skinned_inputs.map(|(v, i)| SkinnedRtInputs {
            objects: &self.state.skinned.draw_objects,
            vertex_gva: v,
            index_gva: i,
            joint_buffers,
        });
        let updated = accel.dynamic_update(
            &self.hw.alloc,
            cmd,
            &self.state.draw.objects,
            RtDynamicInputs {
                mode: self.rt.dynamic_mode,
                skinned,
                frame_idx,
                shared,
                topology_dirty,
                exclude_seethrough,
            },
        );
        crate::rt_report::report_rt_update(&mut self.rt.update_streak, updated);
        // The last draw + cluster geometry is gone and no skinned geometry can
        // rejoin: drop the BVH so a later add re-seeds it, holding it until the
        // frames still in flight have finished tracing it. The trace falls back
        // to SSR meanwhile.
        let skinned_present = skinned_inputs.is_some();
        if self
            .rt
            .accel
            .as_ref()
            .is_some_and(|a| a.is_spent(skinned_present))
            && let Some(accel) = self.rt.accel.take()
        {
            self.rt.retired.push(self.rt.retire_tick, accel);
        }
    }

    // Build the scene acceleration structure from scratch (mirrors the init /
    // `build_rt_runtime` accel block) when a runtime topology change introduces
    // the first participating geometry into an RT-enabled scene that had none.
    // A build failure / still-empty scene is non-fatal: `rt.accel` stays `None`
    // and the next topology change retries.
    fn seed_rt_accel(&mut self) {
        if let Some(accel) = self.build_scene_accel() {
            self.rt.accel = Some(accel);
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
        self.rt.accel = self.build_scene_accel();
    }

    // Build a scene acceleration structure from scratch over the current draw set
    // + shared geometry buffers, with the skin pipeline attached. `None` when the
    // scene has no participating geometry or the build failed (warned).
    fn build_scene_accel(&self) -> Option<RtAccelData> {
        let hot_reload = self.hot_reload.enabled;
        let mut accel = match build_rt_accel(RtInitGeometry {
            alloc: &self.hw.alloc,
            shared: SharedGeometry::of(&self.scene.geometry),
            draw_objects: &self.state.draw.objects,
            clusters: &self.instanced.clusters,
            albedo_count: self.scene.textures.len() as u32,
            exclude_seethrough: self.seethrough_meshes_enabled(),
        }) {
            Ok(Some(accel)) => accel,
            Ok(None) => return None,
            Err(e) => {
                tracing::warn!("RT acceleration-structure build failed: {e}");
                return None;
            }
        };
        match build_rt_skin_pipeline(&self.hw.device, hot_reload) {
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
    fn an_empty_upload_slot_still_holds_one_element() {
        assert_eq!(
            upload_ring_size::<concinnity_core::gfx::render_types::RtGeomEntry>(&[]),
            128
        );
        assert_eq!(upload_ring_size::<u8>(&[]), 4);
        assert_eq!(upload_ring_size(&[0u32; 3]), 12);
    }

    #[test]
    fn scratch_capacity_never_drops_below_the_buffer_minimum() {
        // A build asking for more than the minimum is sized exactly.
        assert_eq!(scratch_capacity(1024), 1024);
        assert_eq!(scratch_capacity(257), 257);
        // A tiny (or zero-instance) build still gets D3D12's 256-byte floor, which
        // is what `create_scratch` allocates, so the recorded capacity matches.
        assert_eq!(scratch_capacity(0), 256);
        assert_eq!(scratch_capacity(255), 256);
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
}
