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
//! it). See `SkinnedFrameRing` / `StaticFrameRing`. A topology refresh records
//! over its own per-frame scratch ring and builds its new BLAS into spares its
//! earlier orphans left in a `BlasPool`, allocating only when none fits; orphans
//! the pool has no room for go to the allocator's deferred free (orphans waiting,
//! when the refresh built no TLAS, for the next TLAS to publish). The static
//! update refits the live TLAS while its instance layout holds, with a periodic
//! full rebuild. The bookkeeping over all of it -- which draws
//! and clusters the BLAS cover, the instance and geometry-table order, and when
//! to update -- is the shared `AccelBook`.

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types::{DrawObject, InstancedCluster, SkinnedDrawObject};
use concinnity_engine::profile::FrameSpans;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::rt_accel::geom::{
    InstanceTag, RtDynamicMode, instance_id_and_mask, pack_row_major_3x4,
};
use concinnity_engine::render::rt_accel::pool::BlasPool;
use concinnity_engine::render::rt_accel::refit::{AccelUpdate, SkinnedRefit, TlasRefit, TlasTree};
use concinnity_engine::render::rt_accel::topology::blas_vertex_count;
use concinnity_engine::render::rt_accel::{
    AccelBook, EmptyHead, FrameRing, HeadRefresh, InstanceBlas, RefreshMode, RefreshStep,
    RtAccelSteps, RtDriveInputs, RtUpdate, ScratchRing, SeedSet, StaticRing, drive_update,
    seed_wanted,
};
use concinnity_engine::render::shader::programs::abi::skin::SKIN_GROUP;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::com;
use super::context::Geometry;
use super::error::map_hresult;
use super::texture::{create_acceleration_structure_buffer, create_uav_buffer, transition_barrier};
use crate::directx::builtin_shaders::CompileProgram;
use crate::directx::pso::compute_pso;
use crate::directx::root_constants::RootConstants;
use crate::directx::root_sig::{RootSig, Visibility};

// Byte stride of a `Vertex` in the shared vertex buffer (pos + normal + tangent
// + color + uv = 14 floats). The BLAS reads positions at this stride and the
// shader fetches attributes at this stride. The deformed (posed) skinned vertex
// buffer the skin kernel writes carries the same 56-byte layout.
const VERTEX_STRIDE: u64 = 56;

// Shared with the Metal and Vulkan hosts: one `.hlsl` declares it.
use concinnity_engine::render::uniforms::SkinParams;

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
            (&raw mut opts5).cast::<std::ffi::c_void>(),
            u32::saturating_from(size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>()),
        )
    };
    ok.is_ok() && opts5.RaytracingTier.0 >= D3D12_RAYTRACING_TIER_1_1.0
}

// One DXR instance descriptor with an explicit 3x4 transform, `InstanceID`
// (indexes the geometry table) and visibility mask from `tag`, and the BLAS
// GPU virtual address. Hit-group contribution + flags are zero (inline tracing ignores hit
// groups).
fn instance_desc(
    model: [[f32; 4]; 4],
    tag: InstanceTag,
    blas_gva: u64,
) -> D3D12_RAYTRACING_INSTANCE_DESC {
    D3D12_RAYTRACING_INSTANCE_DESC {
        Transform: pack_row_major_3x4(model),
        // InstanceID in the low 24 bits, InstanceMask in the high 8.
        _bitfield1: instance_id_and_mask(tag),
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
// built over, read from the live `Geometry` at each build: chunk streaming and
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
    pub(super) fn of(geometry: &Geometry) -> Self {
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
            u32::saturating_from(obj.index_count),
        )
    }

    // The BLAS geometry for one instanced cluster (absolute indices).
    fn cluster_geometry(&self, cluster: &InstancedCluster) -> D3D12_RAYTRACING_GEOMETRY_DESC {
        triangle_geometry(
            self.vertex_gva,
            blas_vertex_count(0, self.vertex_count),
            self.index_gva + cluster.index_offset as u64 * 4,
            u32::saturating_from(cluster.index_count),
        )
    }
}

// Triangle geometry over the deformed skinned vertices with `R32_UINT` indices,
// the base vertex folded into the absolute indices.
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

fn create_as_buffer(device: &ID3D12Device, size: u64) -> RenderResult<ID3D12Resource> {
    create_acceleration_structure_buffer(device, size.max(256))
}

// A build scratch buffer, which promotes to `UNORDERED_ACCESS` on the build's
// first UAV access and decays back to `COMMON` after each `ExecuteCommandLists`.
fn create_scratch(device: &ID3D12Device, size: u64) -> RenderResult<ID3D12Resource> {
    create_uav_buffer(device, size.max(256))
}

// Byte size of a scratch slot covering a build requiring `needed` bytes. D3D12
// buffers have a 256-byte minimum, so a slot never sizes below that; keeping the
// rule here is what lets `ScratchRing::ensure` compare a request against a slot's
// recorded capacity without under-counting the rounding `create_scratch` applies.
fn scratch_capacity(needed: u64) -> u64 {
    needed.max(256)
}

// This frame's build scratch, covering `needed` bytes. A replaced buffer is
// released in place: the frame fence retired its last writer, and only one
// path replaces a slot per frame.
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
    let bytes = size_of_val(data).max(16) as u64;
    let buf = alloc.alloc_buffer(
        bytes,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { buf.Map(0, None, Some(&raw mut ptr)) }
        .map_err(|e| map_hresult(e.code(), &format!("map {label}")))?;
    // SAFETY: `ptr` is the mapping from the call above, of a buffer sized for at least `data`, and
    // `data` is a separate allocation, so the ranges cannot overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().cast::<u8>(),
            ptr.cast::<u8>(),
            size_of_val(data),
        );
    }
    // SAFETY: unmaps subresource 0 of `buf`, mapped above; the copy is done.
    unsafe { buf.Unmap(0, None) };
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
    unsafe { device.GetRaytracingAccelerationStructurePrebuildInfo(inputs, &raw mut info) };
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

// The BOTTOM_LEVEL inputs for one skinned geometry. Always `ALLOW_UPDATE`,
// which a later refit needs; `Refit` sets `PERFORM_UPDATE`. Pass `Build` when
// sizing.
fn skinned_blas_inputs(
    geo: &D3D12_RAYTRACING_GEOMETRY_DESC,
    update: AccelUpdate,
) -> D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
    let mut flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
        | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    if update == AccelUpdate::Refit {
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
// matter). Every TLAS allows updates so a later static update can refit it;
// `Refit` performs one.
fn tlas_inputs(
    instance_count: u32,
    instance_descs_gva: u64,
    update: AccelUpdate,
) -> D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
    let mut flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
        | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    if update == AccelUpdate::Refit {
        flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    }
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
        Flags: flags,
        NumDescs: instance_count,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
            InstanceDescs: instance_descs_gva,
        },
    }
}

// The bytes a BLAS resource holds, for the pool to match it against a build.
fn blas_bytes(blas: &ID3D12Resource) -> u64 {
    // SAFETY: a query on a live COM object that borrows nothing.
    unsafe { blas.GetDesc() }.Width
}

// The `rt_skin` compute pipeline: root SRVs for the bind-pose vertices (t0) and
// joint palette (t1), a root UAV for the output (u0), and `SkinParams` (b0).
pub(super) struct SkinPipeline {
    pub(super) root_sig: ID3D12RootSignature,
    pub(super) pso: ID3D12PipelineState,
}

// Root signature for the `rt_skin` compute kernel: `SkinParams` root constants at
// b0, the skinned vertex buffer as a root SRV (t0), the joint palette as a root
// SRV (t1), the deformed output as a root UAV (u0), the morph deltas as a root
// SRV (t2), and the morph weights as a root SRV (t3).
fn create_skin_root_signature(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    use Visibility::All;
    use concinnity_engine::render::shader::programs::abi::skin::dx as skin;
    RootSig::new()
        .constants::<SkinParams>(skin::PARAMS, All) // [0]
        .srv(skin::SRC, All) // [1] skinned vertex buffer (raw)
        .srv(skin::PALETTE, All) // [2] joint palette (structured)
        .uav(skin::DST, All) // [3] deformed output (raw)
        .srv(skin::MORPH_DATA, All) // [4] morph deltas (raw, dense target-major)
        .srv(skin::MORPH_WEIGHTS, All) // [5] morph weights (raw, one f32 per target)
        .build(device, "rt skin root sig")
}

// One object's `rt_skin` inputs, in root-parameter order.
struct SkinBindings {
    params: SkinParams,
    src_gva: u64,
    joint_gva: u64,
    dst_gva: u64,
    delta_gva: u64,
    weight_gva: u64,
}

// Bind `b` on the skin root signature the caller set and dispatch one thread per vertex.
fn dispatch_skin(cmd: &ID3D12GraphicsCommandList, b: &SkinBindings) {
    // SAFETY: the skin root signature declares `SkinParams` constants at 0, which `b.params` is.
    unsafe { cmd.set_compute_root_constants(0, &b.params) };
    // SAFETY: t0 takes the bind-pose skinned vertex buffer, live for the frame.
    unsafe { cmd.SetComputeRootShaderResourceView(1, b.src_gva) };
    // SAFETY: t1 takes the object's joint palette, live for the frame.
    unsafe { cmd.SetComputeRootShaderResourceView(2, b.joint_gva) };
    // SAFETY: u0 takes the deformed output, which the caller moved to UNORDERED_ACCESS.
    unsafe { cmd.SetComputeRootUnorderedAccessView(3, b.dst_gva) };
    // SAFETY: t2 takes the morph deltas, or a live dummy the kernel leaves unread.
    unsafe { cmd.SetComputeRootShaderResourceView(4, b.delta_gva) };
    // SAFETY: t3 takes the morph weights, or a live dummy the kernel leaves unread.
    unsafe { cmd.SetComputeRootShaderResourceView(5, b.weight_gva) };
    let [x, y, z] = SKIN_GROUP.groups_1d(b.params.vertex_count);
    // SAFETY: the skin PSO and every root binding the kernel reads are set.
    unsafe { cmd.Dispatch(x, y, z) };
}

// Build the `rt_skin` pipeline as `cs_6_5`. `Err` on a compile failure; the
// BVH then omits skinned geometry.
fn build_skin_pipeline(device: &ID3D12Device, hot_reload: bool) -> RenderResult<SkinPipeline> {
    let cs = super::builtin_shaders::RT_SKIN.compile(hot_reload)?;
    let root_sig = create_skin_root_signature(device)?;
    let pso = compute_pso(device, &root_sig, &cs, "rt skin")?;
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
    // The compute-skinning pipeline the skin dispatch runs.
    pub skin: &'a SkinPipeline,
}

// Whether a ring slot must (re)allocate to satisfy `needed` bytes: it is either
// empty or its current capacity is too small. Pure so the grow decision is unit-
// testable without a device.
fn ring_slot_needs_grow(present: bool, capacity: u64, needed: u64) -> bool {
    !present || capacity < needed
}

// One frame slot of the skinned-rebuild buffers, reused in place and grown only
// when needed; the frame fence retires the slot's previous trace first. The
// deformed buffer rests in the combined shader-read state after its first
// rebuild.
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

// One ring slot of the static-transform buffers (TLAS, instance descriptors,
// geometry table), reused in place and grown only when needed; the cursor
// revisits a slot only after a full cycle, past the fence. The live `tlas`,
// `geom_table` and `instance_buffer` are AddRef clones of the current slot's.
#[derive(Default)]
struct StaticFrameRing {
    tlas: Option<ID3D12Resource>,
    tlas_cap: u64,
    instance: Option<PooledBuffer>,
    instance_cap: u64,
    geom: Option<PooledBuffer>,
    geom_cap: u64,
    // The tree `tlas` was last built as and the refit run since.
    tlas_refit: TlasRefit,
}

impl StaticFrameRing {
    // The slot's TLAS, grown to `needed` bytes; a new buffer holds no tree to
    // refit.
    fn ensure_tlas(&mut self, device: &ID3D12Device, needed: u64) -> RenderResult<ID3D12Resource> {
        if ring_slot_needs_grow(self.tlas.is_some(), self.tlas_cap, needed) {
            self.tlas_refit.reset();
        }
        ensure_as_buffer(&mut self.tlas, &mut self.tlas_cap, device, needed)
    }
}

// Write `data` into a reused UPLOAD-heap ring slot, growing it only when too
// small.
fn write_upload_ring<T: Copy>(
    slot: &mut Option<PooledBuffer>,
    cap: &mut u64,
    alloc: &DeviceAllocator,
    data: &[T],
    label: &str,
) -> RenderResult<PooledBuffer> {
    let len_bytes = size_of_val(data);
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
    // SAFETY: `buf` is a live UPLOAD-heap buffer, and `ptr` is a live local that receives the
    // mapping.
    unsafe { buf.Map(0, None, Some(&raw mut ptr)) }
        .map_err(|e| map_hresult(e.code(), &format!("{label} map")))?;
    // SAFETY: `ptr` maps a slot grown to at least `len_bytes`, and `data` is a separate
    // allocation, so the ranges cannot overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), ptr.cast::<u8>(), len_bytes);
    }
    // SAFETY: unmaps subresource 0 of `buf`, mapped above; the copy is done.
    unsafe { buf.Unmap(0, None) };
    Ok(buf)
}

// The bytes an upload ring slot holding `data` needs: at least one whole
// element, so a shader reading it as an array (the geometry table of a
// zero-instance TLAS) sees a full entry, and never under 4.
fn upload_ring_size<T>(data: &[T]) -> u64 {
    (size_of_val(data).max(size_of::<T>()) as u64).max(4)
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
    // The persistent static + cluster BLAS (built once), plus AddRef copies of this
    // frame's skinned BLAS as the tail; the order the TLAS and geometry table
    // follow, and the update policy.
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
    // The deformed skinned vertex buffer, owned by the ring slot that last rebuilt
    // it; a 1-element dummy without skinned geometry so t8 always binds.
    deformed_verts: ID3D12Resource,
    // GPU virtual address of the shared skinned index buffer the skinned BLAS
    // + trace address the deformed buffer with. A dummy buffer's GVA when there
    // is no skinned geometry, so the t9 binding is always valid. Cloned here so
    // the trace encoder can bind it.
    skinned_indices: PooledBuffer,

    // Persistent CPU scratch for the skinned rebuild.
    skinned_scratch: SkinnedScratch,

    // Build scratch for topology refreshes, one buffer per frame in flight. Kept
    // apart from `scratch` so a refresh and the skinned update that follows it
    // never replace a buffer the other recorded against this frame.
    refresh_scratch: ScratchRing<ID3D12Resource>,
    // Draw BLAS a refresh orphaned, rebuilt in place by a later refresh.
    blas_pool: BlasPool<ID3D12Resource>,
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

// The RT skin pipeline, built with the RT pass. A build failure is non-fatal
// (warned): reflections still trace the static geometry, without skinned hits.
pub(super) fn build_rt_skin(device: &ID3D12Device, hot_reload: bool) -> Option<SkinPipeline> {
    build_skin_pipeline(device, hot_reload)
        .inspect_err(|e| {
            tracing::warn!(
                "RT skin pipeline build failed (skinned meshes absent from reflections): {e}"
            );
        })
        .ok()
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
    // Skinned geometry exists to join the BVH, visible or not (see
    // `SeedSet::builds_nothing`).
    pub skinned_present: bool,
}

// Per-frame dynamic-update policy + skinned inputs for `dynamic_update`.
pub(super) struct RtDynamicInputs<'a> {
    // Rebuild gate: off / auto (dirty check) / rebuild (every frame) / tlas.
    pub mode: RtDynamicMode,
    // Per-frame joint palettes + visible skinned objects (None skips the skinned path).
    pub skinned: Option<SkinnedRtInputs<'a>>,
    // Index into the per-frame ring.
    pub frame_idx: usize,
    // The live shared buffers a topology refresh builds new draw BLAS over.
    pub shared: SharedGeometry,
    // Set when the participating draw set changed since the last update.
    pub topology_dirty: bool,
    // Leave see-through glass meshes out of the BVH (see `participates_in_bvh`).
    // Must match what the init build used, or a refresh would silently re-add
    // geometry the transparent pass is already drawing.
    pub exclude_seethrough: bool,
    // Where the update's step times are recorded.
    pub spans: &'a std::cell::Cell<FrameSpans>,
}

// Build the BLAS / TLAS / geometry table on a one-shot, fence-waited list.
// `Ok(None)` when the seed builds nothing; with only skinned geometry the TLAS
// starts empty. `albedo_count` places the flat-normal fallback for the hit
// shader's pool indices.
pub(super) fn build_rt_accel(geometry: RtInitGeometry<'_>) -> RenderResult<Option<RtAccelData>> {
    let RtInitGeometry {
        alloc,
        shared,
        draw_objects,
        clusters,
        albedo_count,
        exclude_seethrough,
        skinned_present,
    } = geometry;
    let device = alloc.device();
    let queue = alloc.queue();
    let device5: ID3D12Device5 = device
        .cast()
        .map_err(|e| map_hresult(e.code(), "ID3D12Device5 cast (DXR unsupported?)"))?;

    // Participating static objects + clusters (real triangles, resident, and not
    // rerouted to the see-through transparent path).
    let seed = SeedSet::new(draw_objects, clusters, exclude_seethrough);
    if seed.builds_nothing(skinned_present) {
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
    book.fill_instances(draw_objects, None, |model, tag, blas| {
        Some(instance_desc(model, tag, instance_blas_gva(blas, &[], &[])))
    });
    let instance_count = u32::saturating_from(book.instances().len());
    let instance_buffer = upload_slice(alloc, book.instances(), "RT instance descriptors")?;
    let geom_table = upload_slice(alloc, book.geom_table(), "RT geometry table")?;

    // Size + allocate the TLAS + the shared scratch (>= the largest BLAS/TLAS).
    let tlas_pre = prebuild_info(
        &device5,
        &tlas_inputs(instance_count, 0, AccelUpdate::Build),
    );
    max_scratch = max_scratch.max(tlas_pre.ScratchDataSizeInBytes);
    let tlas = create_as_buffer(device, tlas_pre.ResultDataMaxSizeInBytes)?;
    // One scratch buffer per frame in flight. The init builds below are their own
    // fence-waited submit, so they record over slot 0 before any frame exists.
    let frames = alloc.frames_in_flight();
    let scratch = ScratchRing::filled(frames, scratch_capacity(max_scratch), |capacity| {
        create_scratch(device, capacity)
    })?;
    let scratch_gva = scratch.get(0).map_or(0, com::gpu_va);

    // Record every BLAS build (UAV-barrier-serialized over the shared scratch),
    // then the TLAS build, on a one-shot command list; fence-wait so the BVH is
    // ready before the first trace.
    record_builds(alloc, queue, |cmd4| {
        for (dest, geo) in book.head().iter().zip(&geo_descs) {
            let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: com::gpu_va(dest),
                Inputs: blas_inputs(geo),
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: scratch_gva,
            };
            // SAFETY: the one-shot list is recording, `geo` outlives the inputs pointing at it,
            // and the BLAS and scratch buffers are live through the fence wait.
            unsafe { cmd4.BuildRaytracingAccelerationStructure(&raw const desc, None) };
            // SAFETY: the one-shot list is recording, and a global UAV barrier names no resource.
            unsafe { cmd4.ResourceBarrier(&[uav_barrier()]) };
        }
        let tlas_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: com::gpu_va(&tlas),
            Inputs: tlas_inputs(
                instance_count,
                com::gpu_va(&instance_buffer),
                AccelUpdate::Build,
            ),
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: scratch_gva,
        };
        // SAFETY: the list is recording, the barrier above ordered every BLAS build, and the
        // TLAS, instance and scratch buffers are live through the fence wait.
        unsafe { cmd4.BuildRaytracingAccelerationStructure(&raw const tlas_desc, None) };
    })?;

    // Skinned geometry joins on the dynamic update, so bind dummy t8/t9 buffers
    // for now. Created in COMMON (as D3D12 buffers always are), which promotes to
    // a shader-read state on first access.
    let deformed_verts = create_uav_buffer(device, VERTEX_STRIDE)?;
    let skinned_indices =
        alloc.alloc_buffer(4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON)?;

    // Seed ring slot 0 with the init structures; the live fields hold clones. The
    // other slots fill on their first rebuild.
    let static_ring = StaticRing::new(
        frames,
        StaticFrameRing {
            tlas: Some(tlas.clone()),
            tlas_cap: tlas_pre.ResultDataMaxSizeInBytes.max(256),
            instance: Some(instance_buffer.clone()),
            instance_cap: (size_of_val(book.instances()) as u64).max(16),
            geom: Some(geom_table.clone()),
            geom_cap: (size_of_val(book.geom_table()) as u64).max(16),
            tlas_refit: TlasRefit::default(),
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
        skinned_ring: FrameRing::new(frames),
        deformed_verts,
        skinned_indices,
        skinned_scratch: SkinnedScratch::default(),
        refresh_scratch: ScratchRing::filled(frames, scratch_capacity(0), |capacity| {
            create_scratch(device, capacity)
        })?,
        blas_pool: BlasPool::default(),
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
    frame_idx: usize,
}

impl RtAccelData {
    // Per-frame dynamic update on the frame's start list: refresh the draw BLAS
    // when the draw set changed, then re-skin, rebuild the TLAS, or keep it. A
    // failure keeps the live BVH and returns the first error; the caller drops a
    // spent BVH. `frame_idx` picks the joint buffer the skin dispatch reads.
    pub(super) fn dynamic_update(
        &mut self,
        alloc: &DeviceAllocator,
        cmd: &ID3D12GraphicsCommandList,
        draw_objects: &[DrawObject],
        inputs: RtDynamicInputs<'_>,
    ) -> RenderResult<RtUpdate> {
        let RtDynamicInputs {
            mode,
            skinned,
            frame_idx,
            shared,
            topology_dirty,
            exclude_seethrough,
            spans,
        } = inputs;
        let now = self.book.tick();
        while let Some(spare) = self.blas_pool.pop_expired(now) {
            alloc.retire(spare);
        }
        let skinned_objects = skinned.as_ref().map(|s| s.objects);
        let mut steps = RtSteps {
            accel: self,
            alloc,
            cmd,
            draw_objects,
            frame_idx,
            shared,
            exclude_seethrough,
            skinned: skinned.as_ref(),
        };
        drive_update(
            &mut steps,
            RtDriveInputs {
                mode,
                topology_dirty,
                draw_objects,
                skinned: skinned_objects,
                now_us: concinnity_host::clock::monotonic_micros,
                spans,
            },
        )
    }

    // Whether the BVH has nothing left to trace and nothing that could rejoin
    // it.
    pub(super) fn is_spent(&self, skinned_present: bool) -> bool {
        self.book.is_spent(skinned_present)
    }

    // Bring the draw BLAS in line with the draw set: reuse unchanged ones, build
    // new ones, drop the skinned tail for `rebuild_skinned`, and rebuild the TLAS
    // into the next `static_ring` slot. Orphans go to the BLAS pool. `self`
    // changes only once every allocation succeeds. With no geometry left it
    // follows `if_empty`.
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

        // A BLAS for every slot the plan could not reuse, rebuilt into a spare when
        // one fits, each with its own range of this frame's refresh scratch so the
        // builds need no barrier between them. `fresh_builds` holds the geometry
        // desc, the dest BLAS and the scratch offset for recording below.
        let now = self.book.clock();
        let depth = alloc.frames_in_flight() as u64;
        let mut fresh: Vec<Option<ID3D12Resource>> =
            (0..refresh.indices().len()).map(|_| None).collect();
        let mut fresh_builds: Vec<(D3D12_RAYTRACING_GEOMETRY_DESC, ID3D12Resource, u64)> =
            Vec::new();
        let mut blas_scratch: u64 = 0;
        for (j, idx) in refresh.fresh_slots() {
            let geo = req.shared.draw_geometry(&draw_objects[idx]);
            let info = prebuild_info(&device5, &blas_inputs(&geo));
            let bytes = info.ResultDataMaxSizeInBytes.max(256);
            let blas = match self.blas_pool.take(bytes, now, depth) {
                Some((blas, _)) => blas,
                None => create_as_buffer(device, bytes)?,
            };
            fresh_builds.push((geo, blas.clone(), blas_scratch));
            blas_scratch += info.ScratchDataSizeInBytes.next_multiple_of(u64::from(
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT,
            ));
            fresh[j] = Some(blas);
        }
        let fresh_gvas: Vec<u64> = fresh
            .iter()
            .map(|b| b.as_ref().map_or(0, com::gpu_va))
            .collect();

        // Static TLAS + geometry table over [refreshed draw head + clusters].
        self.book
            .fill_refresh_instances(&refresh, draw_objects, |model, tag, blas| {
                Some(instance_desc(
                    model,
                    tag,
                    instance_blas_gva(blas, &fresh_gvas, &[]),
                ))
            });
        let instance_count = u32::saturating_from(self.book.instances().len());
        let tlas_pre = prebuild_info(
            &device5,
            &tlas_inputs(instance_count, 0, AccelUpdate::Build),
        );
        let tlas_needed = tlas_pre.ResultDataMaxSizeInBytes;

        // The TLAS build follows a barrier after every BLAS build, so it reuses the
        // scratch from the start.
        let scratch_gva = ensure_scratch(
            &mut self.refresh_scratch,
            device,
            req.frame_idx,
            blas_scratch.max(tlas_pre.ScratchDataSizeInBytes),
        )?;

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
        let tlas = slot.ensure_tlas(device, tlas_needed)?;
        // The commit below must follow the builds recorded next, so it is checked
        // now, while a failure still leaves nothing recorded.
        self.book.check_refresh(&refresh, &fresh)?;

        // Record the fresh draw-BLAS builds over disjoint scratch ranges, one UAV
        // barrier, then the TLAS build. Infallible from here on.
        for (geo, dest, offset) in &fresh_builds {
            let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: com::gpu_va(dest),
                Inputs: blas_inputs(geo),
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: scratch_gva + offset,
            };
            // SAFETY: the list is recording, `geo` outlives the inputs pointing at it, and the
            // fresh BLAS and this frame's refresh scratch stay alive until the build completes.
            unsafe { cmd4.BuildRaytracingAccelerationStructure(&raw const desc, None) };
        }
        if !fresh_builds.is_empty() {
            // SAFETY: the list is recording, and the null-resource UAV barrier borrows nothing.
            unsafe { cmd.ResourceBarrier(&[uav_barrier()]) };
        }
        let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: com::gpu_va(&tlas),
            Inputs: tlas_inputs(
                instance_count,
                com::gpu_va(&instance_buffer),
                AccelUpdate::Build,
            ),
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: scratch_gva,
        };
        // SAFETY: the list is recording, and the ring slot's TLAS and instance buffers stay live
        // until the slot's trace retires a full cycle later.
        unsafe { cmd4.BuildRaytracingAccelerationStructure(&raw const desc, None) };
        // SAFETY: the list is recording, and a global UAV barrier needs no live resource.
        unsafe { cmd.ResourceBarrier(&[uav_barrier()]) };

        // Commit the refreshed head and its static TLAS; `rebuild_skinned` re-adds any
        // skinned tail this frame. Orphans go to the pool.
        let orphans = self.book.commit_refresh(refresh, fresh, draw_objects);
        for orphan in orphans.into_iter().chain(self.book.take_parked()) {
            let bytes = blas_bytes(&orphan);
            if let Some(orphan) = self.blas_pool.give(orphan, bytes, now) {
                alloc.retire(orphan);
            }
        }
        self.book.release_skinned();
        slot.tlas_refit.published(
            AccelUpdate::Build,
            TlasTree {
                head: self.book.head_generation(),
                instances: self.book.instances().len(),
            },
        );
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
            .fill_instances(draw_objects, None, |model, tag, blas| {
                Some(instance_desc(model, tag, instance_blas_gva(blas, &[], &[])))
            });
        let instance_count = u32::saturating_from(self.book.instances().len());

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
        let tlas = slot.ensure_tlas(device, self.tlas_size)?;
        let tree = TlasTree {
            head: self.book.head_generation(),
            instances: self.book.instances().len(),
        };
        let update = slot.tlas_refit.plan(tree);

        // Size this frame's scratch for the TLAS build; after warm-up it is reused.
        let prebuild = prebuild_info(
            &device5,
            &tlas_inputs(instance_count, 0, AccelUpdate::Build),
        );
        let scratch_needed = match update {
            AccelUpdate::Build => prebuild.ScratchDataSizeInBytes,
            AccelUpdate::Refit => prebuild.UpdateScratchDataSizeInBytes,
        };
        let scratch_gva = ensure_scratch(&mut self.scratch, device, frame_idx, scratch_needed)?;

        let cmd4: ID3D12GraphicsCommandList4 = cmd
            .cast()
            .map_err(|e| map_hresult(e.code(), "ID3D12GraphicsCommandList4 cast (rebuild)"))?;
        // A refit updates the slot's own tree in place, which no frame in flight
        // traces.
        let source = match update {
            AccelUpdate::Build => 0,
            AccelUpdate::Refit => com::gpu_va(&tlas),
        };
        let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: com::gpu_va(&tlas),
            Inputs: tlas_inputs(instance_count, com::gpu_va(&instance_buffer), update),
            SourceAccelerationStructureData: source,
            ScratchAccelerationStructureData: scratch_gva,
        };
        // SAFETY: the list is recording; the slot's TLAS and instance buffers and this frame's
        // scratch are live for the frame, and a refit runs only over a tree the slot built with
        // updates allowed over this same head.
        unsafe { cmd4.BuildRaytracingAccelerationStructure(&raw const desc, None) };
        // Order the build before this frame's trace reads the TLAS / table.
        // SAFETY: the frame's start list is recording; the barrier's resource is null.
        unsafe { cmd.ResourceBarrier(&[uav_barrier()]) };

        // Point the live BVH at this slot's buffers. A still-published skinned tail
        // drops back to the static head, and since a refit continues its last build,
        // re-entering the skinned path must rebuild.
        self.tlas = tlas;
        self.geom_table = geom_table;
        self.instance_buffer = instance_buffer;
        slot.tlas_refit.published(update, tree);
        if self.book.commit_static().is_some() {
            self.skinned_ring.unpublish(self.book.clock());
            for ring in self.skinned_ring.slots_mut() {
                ring.refit.reset();
            }
        }
        Ok(())
    }

    // Per-frame skinned update on the start list: re-skin, build or refit each
    // skinned BLAS (rebuilding every `rt_refit::REFIT_LIMIT` refits), and rebuild
    // the TLAS over the head and the skinned tail.
    fn rebuild_skinned(&mut self, req: SkinnedRebuild<'_>) -> RenderResult<RtUpdate> {
        // This slot's buffers, taken out for the duration and put back on every exit.
        // A slot a failed rebuild left live is still traced, so this frame skips.
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
        req: SkinnedRebuild<'_>,
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

        // The deformed vertex buffer, sized to the highest vertex the skinned objects
        // reach and grown on demand.
        let deformed_extent = self.book.skinned_vertex_extent(skinned.objects);
        self.book
            .fill_skinned_shapes(skinned.objects, u32::saturating_from(deformed_extent));
        let deformed_bytes = (deformed_extent * VERTEX_STRIDE).max(VERTEX_STRIDE);
        let read_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
            | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        let deformed_realloc =
            ring_slot_needs_grow(ring.deformed.is_some(), ring.deformed_cap, deformed_bytes);
        if deformed_realloc {
            ring.deformed = Some(create_uav_buffer(device, deformed_bytes)?);
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
        let to_uav = transition_barrier(
            &deformed_verts,
            deformed_before,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        // SAFETY: the list is recording, and the ring slot owns the deformed buffer.
        unsafe { cmd.ResourceBarrier(&[to_uav]) };

        // Stage 1: skin dispatch per skinned object, writing the deformed buffer.
        let skin = skinned.skin;
        // SAFETY: the list is recording, and the skin root signature is owned by `skin`.
        unsafe { cmd.SetComputeRootSignature(&skin.root_sig) };
        // SAFETY: the list is recording, and the skin PSO is owned by `skin`.
        unsafe { cmd.SetPipelineState(&skin.pso) };
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
                vertex_count: u32::saturating_from(obj.vertex_count),
                joint_count: u32::saturating_from(obj.joint_count.max(1)),
                target_count: 0,
            };
            dispatch_skin(
                cmd,
                &SkinBindings {
                    params,
                    src_gva: skinned.vertex_gva,
                    joint_gva,
                    dst_gva: deformed_gva,
                    delta_gva: skinned.vertex_gva,
                    weight_gva: skinned.vertex_gva,
                },
            );
        }
        // Order the skin writes before the BLAS build, then move the deformed buffer to
        // the combined shader-read state both the build and the pixel-shader trace
        // accept, where it rests.
        // SAFETY: the list is recording, and the UAV barrier names no resource to keep alive.
        unsafe { cmd.ResourceBarrier(&[uav_barrier()]) };
        let to_read = transition_barrier(
            &deformed_verts,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            read_state,
        );
        // SAFETY: the list is recording, and the ring slot owns the deformed buffer.
        unsafe { cmd.ResourceBarrier(&[to_read]) };

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
                u32::saturating_from(deformed_extent),
                skinned.index_gva + obj.index_offset as u64 * 4,
                u32::saturating_from(obj.index_count),
            )
        }));

        // Size each skinned BLAS, growing only when needed, and track the larger of
        // the build and refit scratch. A new BLAS or a regrown deformed buffer forces a
        // full build.
        let mut max_scratch: u64 = 0;
        let mut storage_changed = deformed_realloc;
        for (si, geo) in skinned_geo.iter().enumerate() {
            let info = prebuild_info(&device5, &skinned_blas_inputs(geo, AccelUpdate::Build));
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
            .fill_instances(draw_objects, Some(skinned.objects), |model, tag, blas| {
                Some(instance_desc(
                    model,
                    tag,
                    instance_blas_gva(blas, &[], skinned_gvas),
                ))
            });
        let instance_count = u32::saturating_from(self.book.instances().len());
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
        let tlas_pre = prebuild_info(
            &device5,
            &tlas_inputs(instance_count, 0, AccelUpdate::Build),
        );
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

        // Record the skinned BLAS builds or refits, serialized over the shared scratch,
        // then the TLAS build.
        let SkinnedScratch { geo, blas_gvas } = &self.skinned_scratch;
        for (geo, &dest) in geo.iter().zip(blas_gvas) {
            let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: dest,
                Inputs: skinned_blas_inputs(geo, update),
                SourceAccelerationStructureData: match update {
                    AccelUpdate::Build => 0,
                    AccelUpdate::Refit => dest,
                },
                ScratchAccelerationStructureData: scratch_gva,
            };
            // SAFETY: the list is recording, `geo` in `self.skinned_scratch` outlives the inputs
            // pointing at it, and the ring's BLAS and this frame's scratch are live for the frame.
            unsafe { cmd4.BuildRaytracingAccelerationStructure(&raw const desc, None) };
            // SAFETY: the list is recording, and this global barrier borrows no resource.
            unsafe { cmd.ResourceBarrier(&[uav_barrier()]) };
        }
        let tlas_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: com::gpu_va(&tlas),
            Inputs: tlas_inputs(
                instance_count,
                com::gpu_va(&instance_buffer),
                AccelUpdate::Build,
            ),
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: scratch_gva,
        };
        // SAFETY: the list is recording, the barriers above ordered every BLAS update, and the
        // ring's TLAS and instance buffers are live for the frame.
        unsafe { cmd4.BuildRaytracingAccelerationStructure(&raw const tlas_desc, None) };
        // Order the TLAS build before this frame's trace reads it.
        // SAFETY: the list is recording; the TLAS barrier is global and names no resource.
        unsafe { cmd.ResourceBarrier(&[uav_barrier()]) };

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
#[derive(Clone, Copy)]
struct SkinnedRebuild<'a> {
    alloc: &'a DeviceAllocator,
    cmd: &'a ID3D12GraphicsCommandList,
    draw_objects: &'a [DrawObject],
    skinned: &'a SkinnedRtInputs<'a>,
    frame_idx: usize,
    // Build every skinned BLAS from scratch rather than refitting.
    full_build: bool,
}

// One frame's update of the live accel, recorded onto the frame's start command
// list: the steps `drive_update` schedules, over this frame's geometry.
struct RtSteps<'a> {
    accel: &'a mut RtAccelData,
    alloc: &'a DeviceAllocator,
    cmd: &'a ID3D12GraphicsCommandList,
    draw_objects: &'a [DrawObject],
    frame_idx: usize,
    shared: SharedGeometry,
    exclude_seethrough: bool,
    skinned: Option<&'a SkinnedRtInputs<'a>>,
}

impl RtAccelSteps for RtSteps<'_> {
    type Blas = ID3D12Resource;
    type Instance = D3D12_RAYTRACING_INSTANCE_DESC;

    fn book(&mut self) -> &mut AccelBook<ID3D12Resource, D3D12_RAYTRACING_INSTANCE_DESC> {
        &mut self.accel.book
    }

    fn refresh_topology(&mut self, refresh: RefreshStep) -> RenderResult<()> {
        let req = TopologyRefresh {
            shared: self.shared,
            exclude_seethrough: self.exclude_seethrough,
            mode: refresh.mode,
            frame_idx: self.frame_idx,
        };
        self.accel.refresh_topology(
            self.alloc,
            self.cmd,
            self.draw_objects,
            req,
            refresh.if_empty(),
        )
    }

    fn rebuild_tlas(&mut self) -> RenderResult<()> {
        self.accel
            .rebuild_tlas(self.alloc, self.cmd, self.draw_objects, self.frame_idx)
    }

    fn release_skinned(&mut self) {
        self.accel.book.release_skinned();
    }

    fn rebuild_skinned(&mut self, full_build: bool) -> RenderResult<RtUpdate> {
        let Some(skinned) = self.skinned else {
            return Ok(RtUpdate::Done);
        };
        self.accel.rebuild_skinned(SkinnedRebuild {
            alloc: self.alloc,
            cmd: self.cmd,
            draw_objects: self.draw_objects,
            skinned,
            frame_idx: self.frame_idx,
            full_build,
        })
    }
}

impl super::context::DxContext {
    // Deform every skinned object into this frame's deformed buffer for the main
    // pass's second `ExecuteIndirect`, in the Cull arm before Main. Independent of
    // RT; a no-op when the fold is inactive.
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

        let to_uav = transition_barrier(
            deformed,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        // SAFETY: the list is recording, and this frame's deformed buffer is owned by
        // `self.skinned`.
        unsafe { cmd.ResourceBarrier(&[to_uav]) };
        // SAFETY: the list is recording, and the skin root signature is owned by `skin`.
        unsafe { cmd.SetComputeRootSignature(&skin.root_sig) };
        // SAFETY: the list is recording, and the skin PSO is owned by `skin`.
        unsafe { cmd.SetPipelineState(&skin.pso) };
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
                vertex_count: u32::saturating_from(obj.vertex_count),
                joint_count: u32::saturating_from(obj.joint_count.max(1)),
                target_count,
            };
            // Real morph deltas + this frame's weights when the object morphs;
            // otherwise the vertex buffer as an unread dummy (target_count == 0).
            let delta_gva = self
                .skinned
                .morph_delta_buffers
                .get(i)
                .and_then(|b| b.as_ref())
                .map_or(src_gva, |b| com::gpu_va(b));
            let weight_gva = self.morph_weight_gva(frame_idx, i).unwrap_or(src_gva);
            dispatch_skin(
                cmd,
                &SkinBindings {
                    params,
                    src_gva,
                    joint_gva,
                    dst_gva,
                    delta_gva,
                    weight_gva,
                },
            );
        }
        // Orders the skin writes before the main pass's vertex fetch and returns
        // the buffer to its resting VERTEX_AND_CONSTANT_BUFFER state (read by both
        // Main and MainLate's skinned ExecuteIndirect this frame).
        let to_vertex = transition_barrier(
            deformed,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
        );
        // SAFETY: the list is recording, and this frame's deformed buffer is owned by
        // `self.skinned`.
        unsafe { cmd.ResourceBarrier(&[to_vertex]) };
    }

    // Update the acceleration structure on the start list, with this frame's
    // skinned inputs. Consumes `rt.topology_dirty`, and seeds a BVH once geometry
    // appears. A no-op with RT off.
    pub(super) fn rt_dynamic_update(&mut self, cmd: &ID3D12GraphicsCommandList, frame_idx: usize) {
        let mut topology_dirty = std::mem::take(&mut self.state.gpu_dirty.rt_topology);
        // Drop the dropped BVHs no in-flight frame can trace any more: the
        // frame-begin fence wait bounds that at the frames in flight, plus the one this
        // frame records.
        self.rt.retire_tick += 1;
        self.rt
            .retired
            .collect(self.rt.retire_tick, self.hw.frames() as u64 + 1);

        // Only the two shared skinned GVAs are read up-front; the per-object
        // joint palettes are borrowed straight out of this frame's slot below (a
        // disjoint field borrow), so the skin dispatch costs no per-frame list of
        // its own.
        let skinned_gvas = self.rt_skinned_gvas();
        let skinned_present = skinned_gvas.is_some();

        // Seed-from-empty. The one-shot build is fence-waited internally (a rare
        // stall); the DXR trace reads the TLAS + table by GPU virtual address each
        // frame, so the fresh accel is picked up with no descriptor rewire.
        if self.rt.accel.is_none() {
            if self.rt_reflections.is_none()
                || !seed_wanted(self.rt.dynamic_mode, topology_dirty, skinned_present)
            {
                return;
            }
            match self.build_scene_accel(skinned_present) {
                Ok(Some(accel)) => self.rt.accel = Some(accel),
                Ok(None) => return,
                Err(e) => {
                    crate::rt_report::report_rt_update(&mut self.rt.update_streak, Err(e));
                    return;
                }
            }
            // The seed already covers this frame's draw set.
            topology_dirty = false;
        }

        let joint_buffers: &[PooledBuffer] = self
            .skinned
            .joint_buffers
            .get(frame_idx)
            .map(Vec::as_slice)
            .unwrap_or_default();

        // Read before `rt.accel` is borrowed mutably below.
        let exclude_seethrough = self.seethrough_meshes_enabled();
        let shared = SharedGeometry::of(&self.scene.geometry);

        let Some(accel) = self.rt.accel.as_mut() else {
            return;
        };
        let skinned = skinned_gvas
            .zip(self.rt.skin.as_ref())
            .map(|((v, i), skin)| SkinnedRtInputs {
                objects: &self.state.skinned.draw_objects,
                vertex_gva: v,
                index_gva: i,
                joint_buffers,
                skin,
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
                spans: &self.state.spans,
            },
        );
        crate::rt_report::report_rt_update(&mut self.rt.update_streak, updated);
        // The last draw + cluster geometry is gone and no skinned geometry can
        // rejoin: drop the BVH so a later add re-seeds it, holding it until the
        // frames still in flight have finished tracing it. The trace falls back
        // to SSR meanwhile.
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

    // The shared skinned vertex / index buffer GVAs the RT skin dispatch reads,
    // or `None` when skinned geometry cannot join the BVH: none is resident, the
    // launch excluded it, or the skin pipeline failed to build.
    fn rt_skinned_gvas(&self) -> Option<(u64, u64)> {
        if !self.rt.skinned_geometry
            || self.rt.skin.is_none()
            || self.state.skinned.draw_objects.is_empty()
        {
            return None;
        }
        let vb = self.skinned.vertex_buffer.as_ref()?;
        let ib = self.skinned.index_buffer.as_ref()?;
        Some((com::gpu_va(vb), com::gpu_va(ib)))
    }

    // Whether skinned geometry can join the BVH (see `rt_skinned_gvas`).
    pub(super) fn rt_skinned_present(&self) -> bool {
        self.rt_skinned_gvas().is_some()
    }

    // Rebuild the acceleration structure over the current draw set and shared
    // buffers, on RT toggle-on and after `rebuild_static_geometry` replaced them.
    // An empty scene or a failed build leaves no BVH until the next update.
    pub(super) fn rebuild_rt_accel(&mut self) {
        let skinned_present = self.rt_skinned_present();
        self.rt.accel = self.build_scene_accel(skinned_present).unwrap_or_else(|e| {
            tracing::warn!("RT acceleration-structure build failed: {e}");
            None
        });
    }

    // Build a scene acceleration structure from scratch over the current draw set
    // + shared geometry buffers. `None` when the seed builds nothing.
    pub(super) fn build_scene_accel(
        &self,
        skinned_present: bool,
    ) -> RenderResult<Option<RtAccelData>> {
        build_rt_accel(RtInitGeometry {
            alloc: &self.hw.alloc,
            shared: SharedGeometry::of(&self.scene.geometry),
            draw_objects: &self.state.draw.objects,
            clusters: &self.instanced.clusters,
            albedo_count: u32::saturating_from(self.scene.textures.len()),
            exclude_seethrough: self.seethrough_meshes_enabled(),
            skinned_present,
        })
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
            upload_ring_size::<concinnity_engine::gfx::render_types::RtGeomEntry>(&[]),
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
    fn instance_desc_packs_id_and_mask() {
        let d = instance_desc(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            InstanceTag { id: 7, mask: 0x03 },
            0xDEAD_BEEF,
        );
        // InstanceID in the low 24 bits, the mask in the high 8.
        assert_eq!(d._bitfield1 & 0x00FF_FFFF, 7);
        assert_eq!(d._bitfield1 >> 24, 0x03);
        assert_eq!(d._bitfield2, 0);
        assert_eq!(d.AccelerationStructure, 0xDEAD_BEEF);
    }
}
