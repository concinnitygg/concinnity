//! Hardware ray-tracing acceleration structures for the Metal backend. Builds,
//! from the shared static vertex / index buffers and the `DrawObject` list, the
//! bottom- and top-level acceleration structures (BLAS / TLAS) the RT-reflection
//! kernel traces against, plus a per-instance geometry table the kernel uses to
//! fetch the hit triangle and shade it.
//!
//! One primitive BLAS per object over its slice of the shared buffers, one
//! instance in the TLAS per object (transform = the object's model matrix,
//! `instance_id` = the object index). The BLAS describe object-space geometry and
//! never change for a rigid transform; only the TLAS instance transforms (and
//! the geometry table's per-instance model matrices the kernel shades with) move
//! when a prop moves.
//!
//! Dynamic transforms (`RtDynamicMode`) update the structures per frame. The
//! per-frame skinned update (`rebuild_skinned`) keeps the persistent static +
//! cluster BLAS, re-skins the current pose, and rebuilds only the skinned BLAS +
//! TLAS + geometry table. It is fully asynchronous: NO `waitUntilCompleted`. The
//! three GPU steps are committed on the one shared queue in dependency order: skin
//! compute (writes the deformed buffer), then the BLAS/TLAS build (reads it), all
//! in `rt_dynamic_update`, strictly before the reflection-trace command buffer
//! (committed later in `execute_graph`). Same-queue FIFO commit order runs them
//! skin → build → trace; the render graph already depends on exactly this
//! event-free ordering for every cross-pass read. Faults (which can no longer be
//! caught synchronously) are surfaced from completion handlers.
//!
//! (Historical note: an earlier bisect concluded no GPU-side primitive orders these
//! steps and kept the rebuild synchronous. That predated the fix for the actual
//! fault (a CPU/GPU `RtGeomEntry` struct-layout mismatch that made the trace read
//! out of bounds), so those fault observations were the layout bug, not an ordering
//! failure. With it fixed, same-queue commit order alone orders the rebuild.)
//!
//! The one-time seed build and the incremental topology refresh allocate fresh
//! and park the outgoing structures / Shared buffers in a frame-tagged
//! deferred-free pool (`RetirePool`) until the frames-in-flight fence retires the
//! frames whose still-in-flight trace could read them. The per-frame skinned
//! update instead rebuilds in place in a ring slot (`rt_ring`), which is sound
//! precisely because it runs on every frame: see that module's header for why the
//! static paths cannot use the same trick. The bookkeeping over all of it -- which
//! draws and clusters the BLAS cover, the instance and geometry-table order, and
//! when to update -- is the shared `AccelBook`.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types::{DrawObject, InstancedCluster, SkinnedDrawObject};
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::buffers::retire_pool::RetirePool;
use concinnity_engine::render::post::rt_reflections::RtReflectionSettings;
use concinnity_engine::render::rt_accel::geom::RtDynamicMode;
use concinnity_engine::render::rt_accel::pool::BlasPool;
use concinnity_engine::render::rt_accel::refit::{AccelUpdate, SkinnedShape, TlasTree};
use concinnity_engine::render::rt_accel::{
    AccelBook, EmptyHead, FailureStreak, InstanceBlas, RefreshMode, RefreshStep, RtAccelSteps,
    RtUpdate, SeedSet, SlotLiveness, StaticRing, empty_head,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSArray;
use objc2_metal::{
    MTLAccelerationStructure, MTLAccelerationStructureCommandEncoder,
    MTLAccelerationStructureGeometryDescriptor, MTLAccelerationStructureInstanceDescriptorType,
    MTLAccelerationStructureInstanceOptions, MTLAccelerationStructureTriangleGeometryDescriptor,
    MTLAccelerationStructureUsage, MTLAttributeFormat, MTLBuffer, MTLCommandBuffer as _,
    MTLCommandEncoder as _, MTLCommandQueue as _, MTLComputeCommandEncoder as _,
    MTLComputePipelineState, MTLDevice as _, MTLIndexType,
    MTLIndirectAccelerationStructureInstanceDescriptor, MTLInstanceAccelerationStructureDescriptor,
    MTLPackedFloat3, MTLPackedFloat4x3, MTLPrimitiveAccelerationStructureDescriptor,
    MTLRenderCommandEncoder, MTLRenderPipelineState, MTLRenderStages, MTLResource, MTLResourceID,
    MTLResourceOptions, MTLResourceUsage, MTLSize,
};
use std::ptr::NonNull;
// Shared with the Vulkan and DirectX hosts: one `.hlsl` declares it.
use concinnity_engine::render::uniforms::SkinParams;

use super::builtin_shaders::compute_pipeline;
use super::context::write_buffer_slice;
use super::encode::ComputeEncode;
use super::error::{allocation_failed, completed_command_buffer};
use super::rt_ring::{RtFrameRing, RtFrameSlot, SkinnedBlasSet, TlasKey};

// The alignment of each build's range in a shared scratch buffer.
const SCRATCH_OFFSET_ALIGN: usize = 256;

// Byte stride of a `Vertex` in the shared vertex buffer (pos + normal + tangent
// + color + uv = 14 floats). The RT kernel reads positions at this stride; the
// main-pass skinned fold sizes its deformed buffer by it too.
pub(in crate::metal) const VERTEX_STRIDE: usize = 56;

type Structure = Retained<ProtocolObject<dyn MTLAccelerationStructure>>;

// All hardware-ray-traced-reflection state grouped into one feature unit: the
// resolved tunables, the scene acceleration structure, the dynamic-update
// mode + failure-streak flag, and the resolve / textured-resolve / skinning
// pipelines. `settings`/`accel`/the pipelines are `Some` only when RT
// reflections are on and the GPU supports ray tracing (see the per-field
// docs on [`MtlContext`](super::context::MtlContext) for the exact gates).
pub(crate) struct RtState {
    // Resolved + clamped tunables. `Some` only when the world's
    // `Reflections` sets `ray_traced` AND the GPU supports
    // ray tracing; gates the RT pass. RT takes precedence over SSR resolve
    // and reuses `ssr.reflection` as its resolve target.
    pub settings: Option<RtReflectionSettings>,
    // Scene acceleration structure (BLAS/TLAS) + geometry table. `Some` only
    // when RT reflections are on and the scene has resident geometry; updated
    // per frame when `dynamic_mode` is dynamic. Resolution-independent.
    pub accel: Option<RtAccelData>,
    // How the acceleration structure is kept current as props move (the
    // launch's `--rt-dynamic` request; `Auto` by default).
    pub dynamic_mode: RtDynamicMode,
    // Whether skinned meshes join the BVH (the launch's
    // `--rt-skinned-geometry` request; in by default). Clear it and the BVH
    // covers static + instanced geometry only, isolating the skinned trace path.
    pub skinned_geometry: bool,
    // Whether the per-frame BVH update is failing. A transient rebuild failure
    // is non-fatal (keep last frame's BVH) and logged once per streak rather
    // than every frame.
    pub update_streak: FailureStreak,
    pub pipelines: RtPipelines,
    // Dropped BVHs, held until no frame still in flight can reach them.
    pub retired: RetirePool<RtAccelData>,
}

impl RtState {
    // Drop the BVH, so the trace falls back to SSR until a later update seeds
    // one. Retained command buffers keep alive what they bind or declare with
    // `useResource`, but nothing guarantees that for a buffer a build reads only
    // through its descriptor (an instance or deformed-vertex buffer), so the BVH
    // waits out the frames in flight rather than being freed now.
    pub(crate) fn retire_accel(&mut self, frame_id: u64) {
        if let Some(accel) = self.accel.take() {
            self.retired.push(frame_id, accel);
        }
    }

    // Free the dropped BVHs the frames-in-flight fence now guarantees no
    // in-flight frame can reach. Called once per frame.
    pub(crate) fn collect_retired(&mut self, frame_id: u64, depth: usize) {
        self.retired.collect(frame_id, depth as u64);
    }
}

// The ray-traced reflection pipelines, `Some` only when RT reflections are on.
// The quality rebuild swaps them as a unit.
pub(crate) struct RtPipelines {
    // Resolve pipeline, flat-tint hit shading. Used for non-bindless worlds
    // (no albedo pool).
    pub resolve: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // Resolve pipeline, textured hit shading (samples the bindless albedo pool
    // at buffer(7)). Preferred over `resolve` when the bindless texture
    // argument buffer is available this frame.
    pub resolve_textured: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // Compute-skinning pipeline that deforms skinned vertices into a buffer the
    // BVH can trace. Consumed each frame by `rebuild_rt_accel` to pose skinned
    // geometry before the skinned BLAS build.
    pub skin: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

// The acceleration structures + geometry table for hardware ray tracing. Held
// on the context behind an `Option`; present only when the world enables RT
// reflections, the GPU supports ray tracing, and the scene has geometry.
pub(crate) struct RtAccelData {
    // The BLAS head (one per participating draw object, then one per cluster)
    // plus the skinned tail, and the update policy over them. A TLAS does not
    // retain the structures it references, so the book keeps every BLAS a live
    // TLAS reaches for later frames to declare resident; a frame already in
    // flight holds its own retain through the `useResource` it encoded.
    book: AccelBook<Structure, MTLIndirectAccelerationStructureInstanceDescriptor>,
    // The top-level (instance) acceleration structure the kernel traces.
    pub tlas: Structure,
    // `[RtGeomEntry; instance_count]`, indexed by the intersector's
    // `instance_id`. Lets the kernel find the hit triangle + shade it. Carries
    // each instance's model matrix, which the kernel uses to bring the hit
    // normal to world space, so it moves in lockstep with the TLAS transforms.
    pub geom_table: Retained<ProtocolObject<dyn MTLBuffer>>,

    // Deformed (posed) skinned vertices in the static 56-byte `Vertex` layout,
    // written by the `rt_skin` compute pass and traced by the skinned BLAS. The
    // reflection kernel reads it (buffer 5) for skinned hits. A 1-element dummy
    // when the scene has no skinned geometry, so the encoder always has a buffer
    // to bind. The skinned rebuild allocates a fresh one each frame and retires
    // the old through `retire_pool` (it cannot overwrite in place: a prior
    // frame's trace may still be reading it).
    pub deformed_verts: Retained<ProtocolObject<dyn MTLBuffer>>,
    // The shared skinned index buffer, cloned here so the reflection kernel
    // can bind it (buffer 6) for skinned hits. A 1-element dummy when there is
    // no skinned geometry.
    pub skinned_indices: Retained<ProtocolObject<dyn MTLBuffer>>,
    // Outgoing structures / Shared buffers from prior rebuilds, held alive until
    // the frames-in-flight fence retires the frames whose still-in-flight trace
    // could read them. Drained once per frame in `rt_dynamic_update`. The seed
    // build and an incremental topology refresh allocate fresh and park the old
    // here rather than freeing in place; the per-frame skinned update only pushes
    // when it takes over from one of them (see `ring_published`).
    retire_pool: RetirePool<RetiredRt>,

    // Per-in-flight-frame storage the skinned update rebuilds in place instead of
    // allocating fresh. One slot per frame in flight; see `super::rt_ring`.
    ring: RtFrameRing,
    // The TLAS, instance, geometry-table and scratch buffers the static
    // `rebuild_tlas` rewrites in place, one slot per published rebuild. Slot 0
    // holds the seed build's structures.
    static_ring: StaticRing<RtFrameSlot>,
    // A 1-vertex Shared buffer bound at the deformed-vertex slot whenever no
    // skinned geometry is being traced, so the encoder always has a buffer for
    // the binding the shader declares (the skinned branch is never taken then).
    // Also what `release_skinned` falls back to when the ring stops publishing.
    deformed_dummy: Retained<ProtocolObject<dyn MTLBuffer>>,
    // A single-identity joint palette the skin dispatch binds for an object with
    // no pose, so it deforms to bind pose rather than reading whatever the
    // frame's pre-built palette buffer happens to hold.
    identity_palette: Retained<ProtocolObject<dyn MTLBuffer>>,
    // Whether `tlas` / `geom_table` / `deformed_verts` / the skinned tail of
    // `blas` are currently ring-owned clones. When they are not -- right after
    // the seed build or a topology refresh built fresh ones -- the next skinned
    // update must park the outgoing handles in `retire_pool` instead of dropping
    // them, because a prior in-flight frame's trace can still reach them.
    ring_published: bool,
    // Which ring slot the live TLAS was built from, so a slot a failed skinned
    // update left live is not rewritten while frames still in flight trace it.
    ring_liveness: SlotLiveness,
    // Draw BLAS a refresh orphaned, rebuilt in place by a later refresh.
    blas_pool: BlasPool<Structure>,
    // Reused storage for declaring the BLAS a TLAS build reads resident.
    resident: ResidentList,
    // The head BLAS's resource IDs, in order, carried through each refresh.
    // `None` stands for one `book_instance` asks the structure for.
    head_ids: Vec<Option<MTLResourceID>>,
}

// Outgoing RT resources parked by a skinned rebuild or an incremental topology
// refresh for deferred free. Never read again: they exist only to keep the Metal
// handles (and thus the GPU allocations) valid until `RetirePool` drops them,
// once the fence guarantees no in-flight trace can still reference them.
struct RetiredRt {
    #[expect(
        dead_code,
        reason = "held so the acceleration structures stay valid until RetirePool drops them"
    )]
    structures: Vec<Retained<ProtocolObject<dyn MTLAccelerationStructure>>>,
    #[expect(
        dead_code,
        reason = "held so the backing buffers stay valid until RetirePool drops them"
    )]
    buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

// The per-frame skinned-geometry inputs `build_rt_accel` needs to deform and
// add skinned objects to the BVH. Assembled from the context's skinned state;
// `None` skips skinned geometry entirely (the static-only path).
#[derive(Clone, Copy)]
pub(crate) struct SkinnedRtInputs<'a> {
    // One entry per skinned mesh (only `visible`, real-triangle objects build).
    pub objects: &'a [SkinnedDrawObject],
    // Shared skinned vertex buffer (`SkinnedVertex`, 80-byte stride) the skin
    // kernel reads bind-pose vertices from.
    pub vertex_buffer: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    // Shared skinned index buffer (absolute indices) the skinned BLAS and
    // the reflection kernel address the deformed buffer with. Cloned into
    // `RtAccelData` so the reflection encoder can bind it.
    pub index_buffer: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    // Per-object joint palettes, parallel to `objects`; uploaded transiently
    // and consumed by the skin kernel.
    pub joint_matrices: &'a [Vec<[[f32; 4]; 4]>],
    // The compiled `rt_skin` compute pipeline.
    pub skin_pipeline: &'a ProtocolObject<dyn MTLComputePipelineState>,
}

// The device + queue every acceleration-structure build encodes on, plus the
// frames-in-flight depth the per-frame ring is sized to.
#[derive(Clone, Copy)]
pub(crate) struct RtGpu<'a> {
    pub device: &'a ProtocolObject<dyn objc2_metal::MTLDevice>,
    pub command_queue: &'a ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    pub frames_in_flight: usize,
}

// Which frame a per-frame RT update belongs to: the id outgoing resources are
// tagged with in the retire pool, and the ring slot whose storage this frame's
// skinned structures are rebuilt in. Both come from the same frame counter the
// rest of the backend's per-frame rings use.
#[derive(Clone, Copy)]
pub(crate) struct RtFrame {
    pub id: u64,
    pub ring_slot: usize,
}

// The shared static geometry buffers a BLAS build addresses: the u32-indexed
// vertex + index buffers that hold every non-skinned draw object and cluster.
#[derive(Clone, Copy)]
pub(crate) struct RtStaticGeometry<'a> {
    pub vertex_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub index_buffer: &'a ProtocolObject<dyn MTLBuffer>,
}

// The scene geometry a full BVH build spans: the draw objects and the instanced
// clusters (both filtered to resident, real-triangle participants inside).
#[derive(Clone, Copy)]
pub(crate) struct RtSceneGeometry<'a> {
    pub draw_objects: &'a [DrawObject],
    pub clusters: &'a [InstancedCluster],
}

// The shared texture pool's real-texture count a geometry table resolves its
// per-object albedo / normal indices against (the flat-normal fallback sits at
// this index), so an index never points past the resident pool.
#[derive(Clone, Copy)]
pub(crate) struct RtTextureCounts {
    pub albedo_count: usize,
}

// Where skinned geometry stands on the frame of a topology refresh.
#[derive(Clone, Copy)]
pub(crate) struct RefreshShape {
    // This frame's skinned update follows the refresh and builds the TLAS.
    pub skinned_follows: bool,
    // Skinned geometry exists, visible or not.
    pub skinned_present: bool,
}

// The trailing knobs of a topology refresh: whether see-through glass is
// excluded from the BLAS, whether unchanged BLAS are reused, where skinned
// geometry stands, and the frame id the retired resources are parked under.
#[derive(Clone, Copy)]
pub(crate) struct RtTopologyRefreshOptions {
    pub exclude_seethrough: bool,
    pub mode: RefreshMode,
    pub shape: RefreshShape,
    pub frame_id: u64,
}

// The first OS whose TLAS instances can name their BLAS by resource ID
// (`MTLAccelerationStructureInstanceDescriptorType::Indirect`), as `(major, minor)`.
const RT_MIN_OS: (isize, isize) = if cfg!(target_os = "ios") {
    (17, 0)
} else {
    (14, 0)
};

// Whether the GPU supports hardware ray tracing and the OS has the indirect
// instance descriptors the TLAS builds use. Apple-silicon GPUs report `true`;
// Intel / most AMD Macs and older OS versions report `false`, in which case the
// caller falls back to SSR (or no reflections). Mirrors the capability gates the
// MetalFX / HDR paths use at init.
pub(crate) fn raytracing_supported(device: &ProtocolObject<dyn objc2_metal::MTLDevice>) -> bool {
    device.supportsRaytracing() && os_supports_indirect_instances()
}

fn os_supports_indirect_instances() -> bool {
    let (major, minor) = RT_MIN_OS;
    objc2_foundation::NSProcessInfo::processInfo().isOperatingSystemAtLeastVersion(
        objc2_foundation::NSOperatingSystemVersion {
            majorVersion: major,
            minorVersion: minor,
            patchVersion: 0,
        },
    )
}

// Pack a column-major object-to-world `model` matrix into Metal's
// `MTLPackedFloat4x3` instance transform. The packed form is the first three
// rows of each of the four columns (the affine `[0,0,0,1]` bottom row is
// dropped), so `columns[c] = (model[c][0], model[c][1], model[c][2])`. Getting
// this transpose wrong silently mirrors / shears every reflection, so it is
// unit-tested.
pub(crate) fn pack_instance_transform(model: [[f32; 4]; 4]) -> MTLPackedFloat4x3 {
    let col = |c: usize| MTLPackedFloat3 {
        x: model[c][0],
        y: model[c][1],
        z: model[c][2],
    };
    MTLPackedFloat4x3 {
        columns: [col(0), col(1), col(2), col(3)],
    }
}

// A primitive (triangle) BLAS descriptor over a slice of the shared buffers.
// `vertexBufferOffset = base_vertex * stride` so a chunk with mesh-relative
// indices and a non-zero base vertex still resolves; static geometry and
// instanced clusters use base_vertex 0 (their indices are already absolute).
// `index_type` selects the index width; every shared buffer is `UInt32`.
// `usage` is `Refit` only for the skinned structures the
// per-frame update re-fits in place; Metal requires it at build time for a later
// refit to be legal, and it is left `None` everywhere else so the static
// structures keep the better-optimized default tree.
fn prim_desc_for(
    vertex_buffer: &ProtocolObject<dyn MTLBuffer>,
    index_buffer: &ProtocolObject<dyn MTLBuffer>,
    base_vertex: usize,
    index_offset: usize,
    index_count: usize,
    index_type: MTLIndexType,
    usage: MTLAccelerationStructureUsage,
) -> Retained<MTLPrimitiveAccelerationStructureDescriptor> {
    let index_bytes = match index_type {
        MTLIndexType::UInt16 => 2,
        _ => 4,
    };
    let geo = MTLAccelerationStructureTriangleGeometryDescriptor::descriptor();
    geo.setVertexBuffer(Some(vertex_buffer));
    // SAFETY: `base_vertex` addresses a vertex inside `vertex_buffer`, so the offset is in range.
    unsafe { geo.setVertexBufferOffset(base_vertex * VERTEX_STRIDE) };
    geo.setVertexStride(VERTEX_STRIDE);
    geo.setVertexFormat(MTLAttributeFormat::Float3);
    geo.setIndexBuffer(Some(index_buffer));
    // SAFETY: `index_offset` is the mesh's first index inside `index_buffer`, in range.
    unsafe { geo.setIndexBufferOffset(index_offset * index_bytes) };
    geo.setIndexType(index_type);
    geo.setTriangleCount(index_count / 3);
    let geo_ref: &MTLAccelerationStructureGeometryDescriptor = &geo;
    let geos = NSArray::from_slice(&[geo_ref]);
    let prim = MTLPrimitiveAccelerationStructureDescriptor::descriptor();
    prim.setGeometryDescriptors(Some(&geos));
    prim.setUsage(usage);
    prim
}

// The instance descriptor for an instance the book laid out, naming its BLAS by
// resource ID so the TLAS descriptor carries no array of every BLAS (encoding a
// build over such an array costs time in proportion to its length). The shader
// indexes the geometry table by the intersector's `instance_id`, the instance's
// position in the instance buffer, so the table carries one entry per instance,
// in instance order (cluster instances share one BLAS but get distinct
// entries). `head_ids` caches the head's IDs (asking a structure for its ID is
// a message send per instance); `skinned` holds this frame's skinned BLAS. A
// fresh head slot has no BLAS to name and is left out.
fn book_instance(
    model: [[f32; 4]; 4],
    id: u32,
    blas: InstanceBlas<'_, Structure>,
    head_ids: &[Option<MTLResourceID>],
    skinned: &[Structure],
) -> Option<MTLIndirectAccelerationStructureInstanceDescriptor> {
    let structure_id = match blas {
        InstanceBlas::Head { index, blas } => head_ids
            .get(index)
            .copied()
            .flatten()
            .unwrap_or_else(|| blas.gpuResourceID()),
        InstanceBlas::Skinned { n } => skinned.get(n)?.gpuResourceID(),
        InstanceBlas::Fresh { .. } => return None,
    };
    Some(MTLIndirectAccelerationStructureInstanceDescriptor {
        transformationMatrix: pack_instance_transform(model),
        options: MTLAccelerationStructureInstanceOptions::Opaque,
        mask: 0xFF,
        intersectionFunctionTableOffset: 0,
        userID: id,
        accelerationStructureID: structure_id,
    })
}

// The resource IDs of `head`, in order, for `book_instance`.
fn resource_ids(head: &[Structure], out: &mut Vec<Option<MTLResourceID>>) {
    out.clear();
    out.extend(head.iter().map(|b| Some(b.gpuResourceID())));
}

// The TLAS descriptor over `instance_count` instances read from
// `instance_buffer`, built so a later update can refit it.
fn make_tlas_desc(
    instance_buffer: &ProtocolObject<dyn MTLBuffer>,
    instance_count: usize,
) -> Retained<MTLInstanceAccelerationStructureDescriptor> {
    let desc = MTLInstanceAccelerationStructureDescriptor::descriptor();
    desc.setInstanceCount(instance_count);
    desc.setInstanceDescriptorBuffer(Some(instance_buffer));
    desc.setInstanceDescriptorType(MTLAccelerationStructureInstanceDescriptorType::Indirect);
    desc.setUsage(MTLAccelerationStructureUsage::Refit);
    desc
}

// Reused storage for declaring BLAS resident in one batched call. A TLAS build
// reads the primitive structures its instances point to; unlike a direct buffer
// binding, that indirect reference does not make them resident, so Metal
// requires an explicit `useResource` (or `useHeap`) or the build can read a
// non-resident structure and fault. Only structures built on an *earlier*
// command buffer need this: BLAS built in the same encoder are already resident
// and ordered by same-encoder hazard tracking, so they must NOT be passed here.
#[derive(Default)]
struct ResidentList(Vec<NonNull<ProtocolObject<dyn MTLResource>>>);

impl ResidentList {
    fn declare<'a>(
        &mut self,
        enc: &ProtocolObject<dyn MTLAccelerationStructureCommandEncoder>,
        blas: impl IntoIterator<Item = &'a Structure>,
    ) {
        self.0.clear();
        self.0.extend(
            blas.into_iter()
                .map(|b| NonNull::from(ProtocolObject::from_ref(&**b))),
        );
        if let Some(first) = NonNull::new(self.0.as_mut_ptr())
            && !self.0.is_empty()
        {
            // SAFETY: `first` heads `self.0.len()` contiguous pointers to resources the
            // caller's iterator borrowed, all live for this call; the encoder reads the
            // array only during it.
            unsafe { enc.useResources_count_usage(first, self.0.len(), MTLResourceUsage::Read) };
        }
        self.0.clear();
    }
}

// Declare every BLAS resident for a fragment-stage trace in ONE batched
// `useResources` call rather than N per-BLAS ones. A trace render pass (the
// transparent glass/water pass, the RT-reflection resolve) reaches each BLAS
// indirectly through the TLAS, which does NOT make them resident, so the pass
// must declare them itself. Batching collapses the per-frame Obj-C message-send
// count on BLAS-heavy worlds (the driver records the same residency set either
// way, just in one call). A no-op when there are no BLAS.
pub(in crate::metal) fn use_blas_resident_fragment<'a>(
    enc: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    blas: impl IntoIterator<Item = &'a Structure>,
) {
    let res: Vec<NonNull<ProtocolObject<dyn MTLResource>>> = blas
        .into_iter()
        .map(|b| NonNull::from(ProtocolObject::from_ref(&**b)))
        .collect();
    if res.is_empty() {
        return;
    }
    // SAFETY: `res` is a non-empty, contiguous array of `res.len()` live resource
    // pointers; the encoder reads it for the duration of the call only.
    unsafe {
        enc.useResources_count_usage_stages(
            NonNull::new(res.as_ptr() as *mut NonNull<ProtocolObject<dyn MTLResource>>)
                .expect("non-empty blas slice has a non-null pointer"),
            res.len(),
            MTLResourceUsage::Read,
            MTLRenderStages::Fragment,
        );
    }
}

// Identity matrix used as a one-joint fallback palette so a skinned object with
// no pose yet still has a valid (undeformed) palette to dispatch against.
const IDENTITY4: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

// Run the `rt_skin` compute pass: deform each skinned object's bind-pose
// vertices into `deformed_verts` (posed, model-space, 56-byte `Vertex` layout)
// using its joint palette. Runs on its OWN command buffer, committed and
// waited, so the deformed buffer is complete before the acceleration-structure
// build reads it: an AS build does not synchronize against a prior compute
// pass that wrote its input vertex buffer (it is outside the normal encoder
// hazard tracking), so without this wait the build would race the skinning and
// bake a BLAS from half-written vertices.
fn dispatch_skin(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    command_queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    skinned: &SkinnedRtInputs<'_>,
    skinned_objects: &[usize],
    deformed_verts: &ProtocolObject<dyn MTLBuffer>,
) -> RenderResult<()> {
    let skin_cmd = command_queue
        .commandBuffer()
        .ok_or_else(|| RenderError::Other("failed to create RT skin command buffer".into()))?;
    let cenc = skin_cmd
        .computeCommandEncoder()
        .ok_or_else(|| RenderError::Other("failed to create RT skin compute encoder".into()))?;
    // Transient palette buffers must outlive the GPU work; held until the wait
    // below completes.
    let palette_bufs = encode_skin_dispatch(
        &cenc,
        skinned,
        skinned_objects,
        deformed_verts,
        SkinPalettes::Upload(device),
    )?;
    cenc.endEncoding();
    skin_cmd.commit();
    skin_cmd.waitUntilCompleted();
    drop(palette_bufs);
    check_build_status(&skin_cmd, "skinning compute")
}

// Where the skin dispatch gets each object's joint palette.
#[derive(Clone, Copy)]
enum SkinPalettes<'a> {
    // Upload one transient buffer per object. Used by the one-time seed, which
    // runs before any per-frame palette buffer exists; the caller keeps the
    // returned buffers alive across the commit-and-wait.
    Upload(&'a ProtocolObject<dyn objc2_metal::MTLDevice>),
    // Bind the palette buffers the main and shadow passes already built for this
    // frame's ring slot, which live for the whole frame. `identity` stands in for
    // an object with no pose, so it deforms to bind pose rather than reading
    // whatever that object's slot happens to hold.
    Prebuilt {
        buffers: &'a [Retained<ProtocolObject<dyn MTLBuffer>>],
        identity: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    },
}

// Encode the `rt_skin` dispatch for each skinned object into `cenc` (setting the
// pipeline state first) and return any transient joint-palette buffers it
// uploaded, which must outlive the GPU work. Empty on the per-frame path, whose
// palettes are owned by the frame. The caller owns command-buffer lifetime
// (commit + wait, or commit + a completion handler).
fn encode_skin_dispatch(
    cenc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    skinned: &SkinnedRtInputs<'_>,
    skinned_objects: &[usize],
    deformed_verts: &ProtocolObject<dyn MTLBuffer>,
    palettes: SkinPalettes<'_>,
) -> RenderResult<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>> {
    cenc.set_pipeline(skinned.skin_pipeline);
    let threadgroup = skinned
        .skin_pipeline
        .maxTotalThreadsPerThreadgroup()
        .clamp(1, 64);
    let mut uploaded: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> = Vec::new();
    for &obj_idx in skinned_objects {
        let obj = &skinned.objects[obj_idx];
        let matrices: &[[[f32; 4]; 4]] = skinned
            .joint_matrices
            .get(obj_idx)
            .map(Vec::as_slice)
            .unwrap_or_default();
        // Empty pose -> a single identity joint (undeformed) so the dispatch
        // always has a valid palette to index.
        let joint_count = matrices.len().max(1);
        let palette = match &palettes {
            SkinPalettes::Upload(device) => {
                let slice = if matrices.is_empty() {
                    std::slice::from_ref(&IDENTITY4)
                } else {
                    matrices
                };
                let buf = upload_buffer(device, slice, "RT skin palette")?;
                uploaded.push(buf.clone());
                buf
            }
            SkinPalettes::Prebuilt { buffers, identity } => match buffers.get(obj_idx) {
                Some(buf) if !matrices.is_empty() => buf.clone(),
                _ => (*identity).clone(),
            },
        };
        encode_skin_object(
            cenc,
            SkinDispatchBuffers {
                vertex: skinned.vertex_buffer.as_ref(),
                deformed: deformed_verts,
                palette: palette.as_ref(),
            },
            obj,
            joint_count,
            threadgroup,
        );
    }
    Ok(uploaded)
}

// The three buffers one `rt_skin` dispatch binds: the shared bind-pose vertices
// it reads, the deformed buffer it writes, and the object's joint palette.
#[derive(Clone, Copy)]
struct SkinDispatchBuffers<'a> {
    vertex: &'a ProtocolObject<dyn MTLBuffer>,
    deformed: &'a ProtocolObject<dyn MTLBuffer>,
    palette: &'a ProtocolObject<dyn MTLBuffer>,
}

// Encode one skinned object's `rt_skin` dispatch: one thread per vertex, writing
// its posed model-space `Vertex` into the deformed buffer.
fn encode_skin_object(
    cenc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    bufs: SkinDispatchBuffers<'_>,
    obj: &SkinnedDrawObject,
    joint_count: usize,
    threadgroup: usize,
) {
    // The RT dispatch runs the base pose; morphing happens in the per-frame main
    // fold. `target_count == 0` leaves the morph slots unread, so a dummy binding
    // satisfies them.
    let params = SkinParams {
        vertex_base: obj.vertex_base,
        vertex_count: u32::saturating_from(obj.vertex_count),
        joint_count: u32::saturating_from(joint_count),
        target_count: 0,
    };
    cenc.set_buffer(bufs.vertex, 0, 0);
    cenc.set_buffer(bufs.deformed, 0, 1);
    cenc.set_buffer(bufs.palette, 0, 2);
    cenc.set_value(&params, 3);
    // Both morph slots take the vertex buffer as a dummy binding; the kernel
    // leaves them unread because `target_count` is zero.
    cenc.set_buffer(bufs.vertex, 0, 4);
    cenc.set_buffer(bufs.vertex, 0, 5);
    cenc.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: obj.vertex_count.max(1),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threadgroup,
            height: 1,
            depth: 1,
        },
    );
}

// The per-object buffers the per-frame skin fold binds, both built from this
// frame's ring slot and live for the whole frame.
#[derive(Clone, Copy)]
pub(in crate::metal) struct MainSkinBuffers<'a> {
    pub joints: &'a [Retained<ProtocolObject<dyn MTLBuffer>>],
    pub morph_weights: &'a [Retained<ProtocolObject<dyn MTLBuffer>>],
}

impl crate::metal::context::MtlContext {
    // Per-frame pre-skin for the GPU-driven skinned fold: deform every
    // skinned object's bind-pose vertices into `deformed` (this frame's ring
    // slot) using the per-object joint-palette buffers `build_joint_buffers`
    // writes into the joint ring each frame. Reuses the `rt_skin` kernel.
    //
    // Encoded into the Cull pass's command buffer (its own compute encoder),
    // which commits before the Main pass: Metal's automatic hazard tracking then
    // orders this compute write before the main pass's vertex read of `deformed`
    // (the same cross-command-buffer mechanism the static cull → ICB relies on).
    // Unlike the RT seed it binds the pre-built joint and morph-weight buffers
    // instead of uploading transient palettes -- the parallel per-pass encoder
    // cannot keep a transient buffer alive past the worker, while these live for
    // the whole frame. A no-op when the skin pipeline / skinned VB are unset.
    pub(in crate::metal) fn encode_main_skin(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        deformed: &ProtocolObject<dyn MTLBuffer>,
        bufs: MainSkinBuffers<'_>,
    ) -> RenderResult<()> {
        let MainSkinBuffers {
            joints: joint_bufs,
            morph_weights: weight_bufs,
        } = bufs;
        let (Some(skin_pipeline), Some(svb)) = (
            self.skinned.skin_pipeline.as_ref(),
            self.skinned.vertex_buffer.as_ref(),
        ) else {
            return Ok(());
        };
        if self.state.skinned.draw_objects.is_empty() {
            return Ok(());
        }
        let cenc = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            RenderError::Other("failed to create main-skin compute encoder".into())
        })?;
        cenc.set_pipeline(skin_pipeline);
        let tg = skin_pipeline.maxTotalThreadsPerThreadgroup().clamp(1, 64);
        for (i, obj) in self.state.skinned.draw_objects.iter().enumerate() {
            let Some(joint_buf) = joint_bufs.get(i) else {
                continue;
            };
            // Palette length = this object's matrix count (seeded to >= 1, and
            // `update_skinned_pose` never leaves it empty), matching the buffer
            // the kernel indexes.
            let joint_count = self
                .state
                .skinned
                .joint_matrices
                .get(i)
                .map_or(1, |m| m.len().max(1));
            // Morphing needs this object's deltas AND this frame's weights.
            // Both slots are bound unconditionally, so a missing one takes the
            // vertex buffer as a dummy and `target_count` goes to zero, which
            // leaves the kernel reading neither.
            let morph = self.skinned.morphs.get(i).and_then(|m| m.as_ref());
            let weights = weight_bufs.get(i);
            let params = SkinParams {
                vertex_base: obj.vertex_base,
                vertex_count: u32::saturating_from(obj.vertex_count),
                joint_count: u32::saturating_from(joint_count),
                target_count: match (morph, weights) {
                    (Some(m), Some(_)) => m.target_count,
                    _ => 0,
                },
            };
            cenc.set_buffer(svb.as_ref(), 0, 0);
            cenc.set_buffer(deformed, 0, 1);
            cenc.set_buffer(joint_buf.as_ref(), 0, 2);
            cenc.set_value(&params, 3);
            cenc.set_buffer(
                morph.map_or_else(|| svb.as_ref(), |m| m.buffer.as_ref()),
                0,
                4,
            );
            cenc.set_buffer(weights.map_or_else(|| svb.as_ref(), |b| b.as_ref()), 0, 5);
            cenc.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: obj.vertex_count.max(1),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: tg,
                    height: 1,
                    depth: 1,
                },
            );
        }
        cenc.endEncoding();
        Ok(())
    }
}

// Build the BLAS / TLAS / geometry table for the scene. Returns `None` (not an
// error) when the seed builds nothing (`SeedSet::builds_nothing`): the caller
// then leaves RT disabled and the pass falls back to the base scene.
//
// `skinned`, when present, adds skeletally-animated geometry: a compute pass
// deforms each skinned object's vertices into a fresh model-space buffer, and
// one BLAS per skinned object is built over that buffer. Because
// the pose changes every frame the whole structure is rebuilt per frame (the
// caller forces this); fresh allocations keep it hazard-free.
pub(crate) fn build_rt_accel(
    gpu: RtGpu<'_>,
    static_geometry: RtStaticGeometry<'_>,
    scene: RtSceneGeometry<'_>,
    texture_counts: RtTextureCounts,
    skinned: Option<SkinnedRtInputs<'_>>,
    // Layer 2 see-through: when set, see-through glass meshes are left out of the
    // BLAS (they trace their own per-pixel reflection in the transparent pass, and
    // excluding them means glass does not reflect glass). Off keeps every
    // transparent mesh IN the BVH so Layer 1 opaque glass reflects + is reflected
    // like any other surface. Driven by `seethrough_meshes_enabled` (opt-in per
    // `Material::see_through`), not a global flag.
    exclude_seethrough: bool,
) -> RenderResult<Option<RtAccelData>> {
    let RtGpu {
        device,
        command_queue,
        frames_in_flight,
    } = gpu;
    let RtStaticGeometry {
        vertex_buffer,
        index_buffer,
    } = static_geometry;
    let RtSceneGeometry {
        draw_objects,
        clusters,
    } = scene;
    let RtTextureCounts { albedo_count } = texture_counts;
    let seed = SeedSet::new(draw_objects, clusters, exclude_seethrough);
    // Skinned geometry joins the seed build alongside the static head. Clusters
    // and skinned geometry coexist in the BVH; the combination once page-faulted
    // the trace, but that was a per-frame VRAM leak (no autorelease pool around
    // the frame), fixed separately.
    let skinned_list: &[SkinnedDrawObject] = skinned.as_ref().map_or(&[], |s| s.objects);
    if seed.builds_nothing(skinned.is_some()) {
        return Ok(None);
    }

    // One BLAS per draw object, then one per cluster: the book's head.
    let mut prim_descs: Vec<Retained<MTLPrimitiveAccelerationStructureDescriptor>> = seed
        .objects
        .iter()
        .map(|&i| {
            let obj = &draw_objects[i];
            prim_desc_for(
                vertex_buffer,
                index_buffer,
                usize::saturating_from(obj.base_vertex),
                obj.index_offset,
                obj.index_count,
                MTLIndexType::UInt32,
                MTLAccelerationStructureUsage::None,
            )
        })
        .chain(seed.clusters.iter().map(|c| {
            prim_desc_for(
                vertex_buffer,
                index_buffer,
                0,
                c.index_offset,
                c.index_count,
                MTLIndexType::UInt32,
                MTLAccelerationStructureUsage::None,
            )
        }))
        .collect();
    let mut max_scratch: usize = 0;
    let mut allocate = |prim: &MTLPrimitiveAccelerationStructureDescriptor| {
        let sizes = device.accelerationStructureSizesWithDescriptor(prim);
        max_scratch = max_scratch.max(sizes.buildScratchBufferSize);
        device
            .newAccelerationStructureWithSize(sizes.accelerationStructureSize)
            .ok_or_else(|| allocation_failed("BLAS"))
    };
    let head = prim_descs
        .iter()
        .map(|prim| allocate(prim))
        .collect::<RenderResult<Vec<Structure>>>()?;
    let albedo_count = u32::saturating_from(albedo_count);
    let mut book = AccelBook::new(&seed, head, draw_objects, albedo_count)?;
    book.select_skinned(Some(skinned_list));
    let skinned_objects = book.visible_skinned().to_vec();

    // Deformed-vertex buffer for skinned geometry: the `rt_skin` kernel writes
    // posed model-space `Vertex`s here, mirroring the skinned vertex buffer's
    // indexing so the skinned index buffer addresses it directly. Sized to the
    // highest vertex the skinned objects reach.
    //
    // Shared, not Private: the buffer is written by the skin compute pass and
    // then read both by the acceleration-structure build and (per hit) by the
    // reflection fragment shader, which run in *separate* command buffers. A
    // Private buffer in that cross-command-buffer producer/consumer pattern was
    // observed to GPU page-fault on the fragment read under the parallel per-
    // pass encoder; Shared is always host-resident and coherent, sidestepping it.
    //
    // The 1-vertex dummy is allocated unconditionally: it is what the encoder
    // binds whenever no skinned geometry is traced, both here and after the
    // per-frame update stops publishing its ring slot.
    let deformed_bytes = (usize::saturating_from(book.skinned_vertex_extent(skinned_list))
        * VERTEX_STRIDE)
        .max(VERTEX_STRIDE);
    let deformed_dummy = device
        .newBufferWithLength_options(VERTEX_STRIDE, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| allocation_failed("RT deformed-vertex dummy buffer"))?;
    let deformed_verts = if skinned_objects.is_empty() {
        deformed_dummy.clone()
    } else {
        device
            .newBufferWithLength_options(deformed_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| allocation_failed("RT deformed-vertex buffer"))?
    };
    // The shared skinned index buffer the kernel + skinned BLAS address; a
    // dummy when there is no skinned geometry. The dummy is one u32 rather than
    // one u16 because the trace reads the buffer as packed u32 words (two
    // indices each) -- Metal API validation rejects a 2-byte buffer bound to a
    // 4-byte element type, even where the branch that reads it never runs.
    let skinned_indices: Retained<ProtocolObject<dyn MTLBuffer>> = match &skinned {
        Some(s) if !skinned_objects.is_empty() => s.index_buffer.clone(),
        _ => device
            .newBufferWithLength_options(size_of::<u32>(), MTLResourceOptions::StorageModePrivate)
            .ok_or_else(|| allocation_failed("RT skinned-index dummy buffer"))?,
    };

    // Skinned BLAS trace the deformed buffer (absolute indices, base_vertex 0)
    // and follow the head as the book's tail. The buffer's contents are written
    // by the skin pass below, before these build.
    let skinned_descs: Vec<_> = skinned_objects
        .iter()
        .map(|&i| {
            let obj = &skinned_list[i];
            prim_desc_for(
                deformed_verts.as_ref(),
                skinned_indices.as_ref(),
                0,
                obj.index_offset,
                obj.index_count,
                MTLIndexType::UInt32,
                MTLAccelerationStructureUsage::Refit,
            )
        })
        .collect();
    let tail = skinned_descs
        .iter()
        .map(|prim| allocate(prim))
        .collect::<RenderResult<Vec<Structure>>>()?;
    book.replace_tail(tail);
    prim_descs.extend(skinned_descs);

    // The geometry table is indexed PER INSTANCE, by the intersector's
    // `instance_id`, which is the instance's position in the instance buffer, so
    // there is exactly one entry per TLAS instance, in the book's instance order.
    let skinned_blas = book.blas()[book.static_blas_count()..].to_vec();
    let mut head_ids = Vec::new();
    resource_ids(book.head(), &mut head_ids);
    book.fill_instances(draw_objects, Some(skinned_list), |model, id, blas| {
        book_instance(model, id, blas, &head_ids, &skinned_blas)
    });
    let instance_buffer = upload_buffer(device, book.instances(), "RT instance descriptors")?;
    let geom_table = upload_buffer(device, book.geom_table(), "RT geometry table")?;

    let tlas_desc = make_tlas_desc(&instance_buffer, book.instances().len());
    let tlas_sizes = device.accelerationStructureSizesWithDescriptor(&tlas_desc);
    let tlas = device
        .newAccelerationStructureWithSize(tlas_sizes.accelerationStructureSize)
        .ok_or_else(|| allocation_failed("TLAS"))?;
    // Size the scratch for the largest of every BLAS build and the TLAS build
    // so the per-frame TLAS rebuild can reuse the same buffer.
    max_scratch = max_scratch.max(tlas_sizes.buildScratchBufferSize);

    // Freed once the build below completes; the static ring sizes its own.
    let scratch = device
        .newBufferWithLength_options(max_scratch.max(1), MTLResourceOptions::StorageModePrivate)
        .ok_or_else(|| allocation_failed("RT scratch buffer"))?;

    // Skin first (on its own committed-and-waited command buffer) so the
    // deformed buffer is complete before the BLAS build reads it; see
    // `dispatch_skin` for why the wait is required.
    if let Some(s) = &skinned
        && !skinned_objects.is_empty()
    {
        dispatch_skin(
            device,
            command_queue,
            s,
            &skinned_objects,
            deformed_verts.as_ref(),
        )?;
    }

    // Build each BLAS in its own acceleration-structure encoder, then the TLAS in
    // a final encoder. Metal does not order (or prevent overlap of) builds within
    // a single encoder, so a one-encoder build of "all BLAS then the TLAS" lets
    // the TLAS read half-built BLAS and lets builds sharing the scratch buffer
    // stomp on it. Separate encoders within the command buffer serialize, which
    // both orders the TLAS after its BLAS and makes the shared scratch safe.
    // Synchronous: wait for the GPU so the structures (and the skinning that feeds
    // the skinned BLAS) are ready before the first frame traces them.
    let cmd = command_queue
        .commandBuffer()
        .ok_or_else(|| RenderError::Other("failed to create RT build command buffer".into()))?;
    for (acc, prim) in book.blas().iter().zip(prim_descs.iter()) {
        let enc = cmd.accelerationStructureCommandEncoder().ok_or_else(|| {
            RenderError::Other("failed to create acceleration-structure encoder".into())
        })?;
        enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
            acc, prim, &scratch, 0,
        );
        enc.endEncoding();
    }
    // The TLAS references every BLAS, all built on earlier encoders, so declare
    // them resident in this encoder.
    let enc = cmd.accelerationStructureCommandEncoder().ok_or_else(|| {
        RenderError::Other("failed to create acceleration-structure encoder".into())
    })?;
    ResidentList::default().declare(&enc, book.blas());
    enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
        &tlas, &tlas_desc, &scratch, 0,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    check_build_status(&cmd, "acceleration-structure build")?;

    let identity_palette = upload_buffer(device, &[IDENTITY4], "RT identity palette")?;
    let static_ring = StaticRing::new(
        frames_in_flight,
        RtFrameSlot::holding_tlas(
            tlas.clone(),
            tlas_sizes.accelerationStructureSize,
            instance_buffer,
            geom_table.clone(),
        ),
    );

    Ok(Some(RtAccelData {
        book,
        tlas,
        geom_table,
        deformed_verts,
        skinned_indices,
        retire_pool: RetirePool::new(),
        ring: RtFrameRing::new(frames_in_flight),
        static_ring,
        deformed_dummy,
        identity_palette,
        // Everything above is a fresh allocation, not a ring clone, so the first
        // skinned update has to retire it rather than drop it.
        ring_published: false,
        ring_liveness: SlotLiveness::new(frames_in_flight),
        blas_pool: BlasPool::default(),
        resident: ResidentList::default(),
        head_ids,
    }))
}

// Allocate one BLAS per skinned object over the deformed buffer (absolute
// indices, base_vertex 0), with the descriptors they were sized from and the
// largest build scratch any of them needs. `Refit` usage is required at build
// time for the per-frame in-place refit to be legal.
fn allocate_skinned_blas(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    deformed_verts: &ProtocolObject<dyn MTLBuffer>,
    skinned_indices: &ProtocolObject<dyn MTLBuffer>,
    shapes: &[SkinnedShape],
) -> RenderResult<SkinnedBlasSet> {
    let mut blas = Vec::with_capacity(shapes.len());
    let mut descs = Vec::with_capacity(shapes.len());
    let mut scratch_bytes = 0usize;
    for shape in shapes {
        let prim = prim_desc_for(
            deformed_verts,
            skinned_indices,
            0,
            shape.index_offset,
            shape.index_count,
            MTLIndexType::UInt32,
            MTLAccelerationStructureUsage::Refit,
        );
        let sizes = device.accelerationStructureSizesWithDescriptor(&prim);
        let acc = device
            .newAccelerationStructureWithSize(sizes.accelerationStructureSize)
            .ok_or_else(|| allocation_failed("skinned BLAS"))?;
        acc.setLabel(Some(&crate::metal::pipeline::ns_str("rt_skinned_blas")));
        scratch_bytes = scratch_bytes.max(sizes.buildScratchBufferSize);
        blas.push(acc);
        descs.push(prim);
    }
    Ok(SkinnedBlasSet {
        blas,
        descs,
        scratch_bytes,
    })
}

// One frame's build-scratch requirement: the largest of every skinned BLAS build
// and the TLAS build, which share the slot's one scratch buffer (separate
// encoders serialize them). At least one byte so Metal never sees a zero-length
// buffer. Pure so the sizing is unit-testable.
fn slot_scratch_bytes(blas_scratch: usize, tlas_scratch: usize) -> usize {
    blas_scratch.max(tlas_scratch).max(1)
}

// One frame's update of the live accel: the steps `drive_update` schedules,
// over this frame's device, geometry and skinned inputs. `skinned` is `None`
// when a skinned resource is missing, which skips the skinned step.
pub(crate) struct MtlRtSteps<'a> {
    pub accel: &'a mut RtAccelData,
    pub gpu: RtGpu<'a>,
    pub geometry: RtStaticGeometry<'a>,
    pub draw_objects: &'a [DrawObject],
    pub exclude_seethrough: bool,
    pub frame: RtFrame,
    pub joint_buffers: &'a [Retained<ProtocolObject<dyn MTLBuffer>>],
    pub skinned: Option<SkinnedRtInputs<'a>>,
}

impl RtAccelSteps for MtlRtSteps<'_> {
    type Blas = Structure;
    type Instance = MTLIndirectAccelerationStructureInstanceDescriptor;

    fn book(
        &mut self,
    ) -> &mut AccelBook<Structure, MTLIndirectAccelerationStructureInstanceDescriptor> {
        &mut self.accel.book
    }

    fn refresh_topology(&mut self, refresh: RefreshStep) -> RenderResult<()> {
        self.accel.refresh_static_topology(
            self.gpu,
            self.geometry,
            self.draw_objects,
            RtTopologyRefreshOptions {
                exclude_seethrough: self.exclude_seethrough,
                mode: refresh.mode,
                shape: RefreshShape {
                    skinned_follows: refresh.skinned_follows,
                    skinned_present: refresh.skinned_present,
                },
                frame_id: self.frame.id,
            },
        )
    }

    fn rebuild_tlas(&mut self) -> RenderResult<()> {
        self.accel.rebuild_tlas(
            self.gpu.device,
            self.gpu.command_queue,
            self.draw_objects,
            self.frame.id,
        )
    }

    fn release_skinned(&mut self) {
        self.accel.release_skinned_tail(self.frame.id);
    }

    fn rebuild_skinned(&mut self, full_build: bool) -> RenderResult<RtUpdate> {
        let Some(skinned) = self.skinned.take() else {
            return Ok(RtUpdate::Skipped);
        };
        self.accel.rebuild_skinned(
            self.gpu,
            self.draw_objects,
            skinned,
            self.joint_buffers,
            self.frame,
            full_build,
        )
    }
}

impl RtAccelData {
    // Keep the static BLAS; update the TLAS + geometry table from the current
    // transforms into the next static ring slot (see `StaticRing`), committed
    // WITHOUT waiting: a full build, or a refit of the tree the slot already
    // holds over the current head (see `TlasRefit`). Same-queue commit order runs
    // the update before this frame's reflection trace, as on the skinned path. The slot after the live
    // one is no longer traced by any frame in flight, so its buffers are
    // rewritten in place; a failure puts it back untouched and leaves the live
    // TLAS as it was.
    pub(crate) fn rebuild_tlas(
        &mut self,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        command_queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
        draw_objects: &[DrawObject],
        frame_id: u64,
    ) -> RenderResult<()> {
        let head_ids = &self.head_ids;
        self.book
            .fill_instances(draw_objects, None, |model, id, blas| {
                book_instance(model, id, blas, head_ids, &[])
            });
        let (index, mut slot) = self.static_ring.take_next();
        let (tlas, geom_table) = match self.build_static_tlas(device, command_queue, &mut slot) {
            Ok(built) => built,
            Err(e) => {
                self.static_ring.put(index, slot);
                return Err(e);
            }
        };
        self.static_ring.publish(index, slot);

        self.tlas = tlas;
        self.geom_table = geom_table;
        if let Some(tail) = self.book.commit_static() {
            self.release_skinned(tail, frame_id);
        }
        self.retire_parked(frame_id);
        // Not skinned-ring storage, so a later skinned update has to retire
        // rather than drop them.
        self.ring_published = false;
        Ok(())
    }

    // Write the book's instances + geometry table into `slot` and commit the TLAS
    // build over the static head, or a refit in place of the tree the slot
    // already holds over that head, returning the TLAS and table to publish.
    fn build_static_tlas(
        &mut self,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        command_queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
        slot: &mut RtFrameSlot,
    ) -> RenderResult<(Structure, Retained<ProtocolObject<dyn MTLBuffer>>)> {
        let instances = self.book.instances();
        let geom = self.book.geom_table();
        let instance_buffer = slot.instances(device, size_of_val(instances))?;
        let geom_table = slot.geom_table(device, size_of_val(geom))?;
        write_buffer_slice(&instance_buffer, instances)?;
        write_buffer_slice(&geom_table, geom)?;

        let key = TlasKey {
            slot_generation: slot.generation(),
            instance_count: instances.len(),
        };
        let tlas_desc = if let Some(desc) = slot.tlas_desc(key) {
            desc
        } else {
            let desc = make_tlas_desc(&instance_buffer, instances.len());
            slot.set_tlas_desc(key, desc.clone());
            desc
        };
        let sizes = device.accelerationStructureSizesWithDescriptor(&tlas_desc);
        let tlas = slot.tlas(device, sizes.accelerationStructureSize)?;
        let scratch = slot.scratch(
            device,
            sizes
                .buildScratchBufferSize
                .max(sizes.refitScratchBufferSize),
        )?;
        let tree = TlasTree {
            head: self.book.head_generation(),
            instances: instances.len(),
        };
        let update = slot.tlas_refit.plan(tree);

        let cmd = command_queue.commandBuffer().ok_or_else(|| {
            RenderError::Other("failed to create RT rebuild command buffer".into())
        })?;
        cmd.setLabel(Some(&crate::metal::pipeline::ns_str("rt_tlas_build")));
        let enc = cmd.accelerationStructureCommandEncoder().ok_or_else(|| {
            RenderError::Other("failed to create acceleration-structure encoder".into())
        })?;
        // Every BLAS the TLAS references was built on an earlier command buffer
        // (none are rebuilt here), so all must be declared resident.
        self.resident.declare(&enc, self.book.head());
        match update {
            AccelUpdate::Build => {
                enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
                    &tlas, &tlas_desc, &scratch, 0,
                );
            }
            // A nil destination refits in place, which no frame in flight
            // traces: the slot after the live one has retired its tracers.
            // SAFETY: the slot's tree was built with `Refit` usage over this same
            // head, so the descriptor lays out the same instances over the same
            // BLAS; the TLAS and the scratch, sized to the descriptor's refit
            // size, are owned by the slot for longer than the command buffer runs.
            AccelUpdate::Refit => unsafe {
                enc.refitAccelerationStructure_descriptor_destination_scratchBuffer_scratchBufferOffset(
                    &tlas,
                    &tlas_desc,
                    None,
                    Some(&scratch),
                    0,
                );
            },
        }
        enc.endEncoding();
        super::fault_log::attach_fault_logger(&cmd, "RT TLAS update");
        cmd.commit();
        slot.tlas_refit.published(update, tree);
        Ok((tlas, geom_table))
    }

    // Whether the BVH has nothing left to trace and nothing that could rejoin it.
    pub(crate) fn is_spent(&self, skinned_present: bool) -> bool {
        self.book.is_spent(skinned_present)
    }

    // Stop publishing the skinned tail without building a TLAS, for a BVH with
    // nothing static left that no skinned geometry can rejoin.
    pub(crate) fn release_skinned_tail(&mut self, frame_id: u64) {
        if let Some(tail) = self.book.release_skinned() {
            self.release_skinned(tail, frame_id);
        }
    }

    // Every BLAS, the head then any skinned tail.
    pub(crate) fn blas(&self) -> &[Structure] {
        self.book.blas()
    }

    // Every BLAS the live TLAS may reference, parked orphans included, for a
    // trace pass to declare resident.
    pub(crate) fn traced_blas(&self) -> impl Iterator<Item = &Structure> {
        self.book.traced_blas()
    }

    // Follow a change to the shared texture pool's real-texture count.
    pub(crate) fn set_albedo_count(&mut self, albedo_count: usize) {
        self.book
            .set_albedo_count(u32::saturating_from(albedo_count));
    }

    // Bring the draw-object BLAS head in line with the current participating draw
    // set: reuse every BLAS whose geometry is unchanged (or none, under
    // `RefreshMode::RebuildAll`), build the new / changed ones into spare storage
    // where a recycled BLAS fits, and commit the refreshed head. The cluster BLAS
    // and any skinned tail are preserved. Used when streamed chunks are
    // added/removed, props are cloned, or a material edit changes RT
    // participation.
    //
    // The fresh BLAS build on one command buffer committed WITHOUT waiting, all in
    // one encoder: each build has its own scratch range and destination, so the
    // builds need no order between them. The orphans stay parked in the book, and
    // declared resident by every trace, until a TLAS that does not reference them
    // publishes. On the static path that is the `rebuild_tlas` that follows here,
    // on its own command buffer committed after the builds, so same-queue commit
    // order runs it after them; on the skinned path (`skinned_follows`) the
    // caller's `rebuild_skinned` builds this frame's TLAS instead. A refresh that
    // leaves no draw or cluster geometry follows `empty_head`: with no skinned
    // geometry at all it builds nothing and the caller drops the BVH. Every
    // allocation precedes the commit, so a failure leaves the live BVH untouched.
    pub(crate) fn refresh_static_topology(
        &mut self,
        gpu: RtGpu<'_>,
        static_geometry: RtStaticGeometry<'_>,
        draw_objects: &[DrawObject],
        options: RtTopologyRefreshOptions,
    ) -> RenderResult<()> {
        let RtGpu {
            device,
            command_queue,
            frames_in_flight,
        } = gpu;
        let RtStaticGeometry {
            vertex_buffer,
            index_buffer,
        } = static_geometry;
        let RtTopologyRefreshOptions {
            exclude_seethrough,
            mode,
            shape:
                RefreshShape {
                    skinned_follows,
                    skinned_present,
                },
            frame_id,
        } = options;
        let refresh = self
            .book
            .plan_refresh(draw_objects, exclude_seethrough, mode);
        let empty = self
            .book
            .refresh_leaves_nothing(&refresh)
            .then(|| empty_head(skinned_follows, skinned_present));

        // A BLAS for every slot the plan did not match to an existing one, each
        // with its own range of one scratch buffer.
        let mut fresh: Vec<Option<Structure>> =
            (0..refresh.indices().len()).map(|_| None).collect();
        let mut build_jobs: Vec<(
            usize,
            Retained<MTLPrimitiveAccelerationStructureDescriptor>,
            usize,
        )> = Vec::new();
        let mut scratch_bytes: usize = 0;
        for (j, idx) in refresh.fresh_slots() {
            let obj = &draw_objects[idx];
            let prim = prim_desc_for(
                vertex_buffer,
                index_buffer,
                usize::saturating_from(obj.base_vertex),
                obj.index_offset,
                obj.index_count,
                MTLIndexType::UInt32,
                MTLAccelerationStructureUsage::None,
            );
            let sizes = device.accelerationStructureSizesWithDescriptor(&prim);
            let bytes = u64::saturating_from(sizes.accelerationStructureSize);
            let acc = if let Some((acc, _)) =
                self.blas_pool
                    .take(bytes, frame_id, frames_in_flight as u64)
            {
                acc
            } else {
                let acc = device
                    .newAccelerationStructureWithSize(sizes.accelerationStructureSize)
                    .ok_or_else(|| allocation_failed("topology-refresh BLAS"))?;
                acc.setLabel(Some(&crate::metal::pipeline::ns_str("rt_topology_blas")));
                acc
            };
            build_jobs.push((j, prim, scratch_bytes));
            scratch_bytes += sizes
                .buildScratchBufferSize
                .next_multiple_of(SCRATCH_OFFSET_ALIGN);
            fresh[j] = Some(acc);
        }

        // The commit must follow the builds committed next, so it is checked now,
        // while a failure still leaves nothing committed.
        self.book.check_refresh(&refresh, &fresh)?;

        if !build_jobs.is_empty() {
            let scratch = device
                .newBufferWithLength_options(
                    scratch_bytes.max(1),
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| allocation_failed("topology-refresh scratch buffer"))?;
            let cmd = command_queue.commandBuffer().ok_or_else(|| {
                RenderError::Other("failed to create topology-refresh command buffer".into())
            })?;
            cmd.setLabel(Some(&crate::metal::pipeline::ns_str("rt_topology_build")));
            let enc = cmd.accelerationStructureCommandEncoder().ok_or_else(|| {
                RenderError::Other("failed to create acceleration-structure encoder".into())
            })?;
            for (j, prim, offset) in &build_jobs {
                if let Some(acc) = fresh[*j].as_ref() {
                    enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
                        acc, prim, &scratch, *offset,
                    );
                }
            }
            enc.endEncoding();
            super::fault_log::attach_fault_logger(&cmd, "RT topology build");
            cmd.commit();
            // The async builds keep reading the scratch after this returns.
            self.retire_pool.push(
                frame_id,
                RetiredRt {
                    structures: Vec::new(),
                    buffers: vec![scratch],
                },
            );
        }

        // Swap in the refreshed head. The live TLAS still references the orphans,
        // so they wait parked for one that does not.
        let old_ids = core::mem::take(&mut self.head_ids);
        self.book.remap_head(
            &refresh,
            &old_ids,
            |j| {
                fresh
                    .get(j)
                    .and_then(Option::as_ref)
                    .map(|b| b.gpuResourceID())
            },
            &mut self.head_ids,
        );
        let orphans = self.book.commit_refresh(refresh, fresh, draw_objects);
        self.book.park(orphans);
        match empty {
            // Nothing can rejoin, so stop publishing the skinned tail and leave the
            // BVH spent for the caller to drop, parked orphans and all.
            Some(EmptyHead::Drop) => {
                self.release_skinned_tail(frame_id);
                Ok(())
            }
            _ if skinned_follows => Ok(()),
            // An empty head that skinned geometry can rejoin still gets a
            // (zero-instance) TLAS, so the trace stops reaching what left.
            _ => self.rebuild_tlas(device, command_queue, draw_objects, frame_id),
        }
    }

    // Per-frame skinned update: keep the persistent static + cluster BLAS,
    // re-skin this frame's pose, update the skinned BLAS, and rebuild the TLAS +
    // geometry table over the static head plus the skinned tail.
    //
    // Fully asynchronous: NO `waitUntilCompleted`. The skin compute (which writes
    // `deformed_verts`) and the BLAS/TLAS build (which reads it) run on separate
    // command buffers, both committed on the shared queue in order (skin, then
    // build) ahead of this frame's reflection-trace command buffer. An
    // acceleration-structure build is outside Metal's automatic hazard tracking,
    // so skin → build and build → trace are ordered by same-queue FIFO commit order,
    // the same mechanism the render graph uses for every cross-pass read.
    // Per-frame GPU stalls are gone; faults are surfaced from completion handlers.
    //
    // Allocation-free in steady state. Every structure and buffer it writes lives
    // in this frame's ring slot (`super::rt_ring`) and is rebuilt in place: the
    // update runs on every frame, so the frames-in-flight fence guarantees the
    // slot's previous writer has retired. The skinned BLAS are re-fit rather than
    // rebuilt while the triangle set is unchanged, with a periodic full rebuild to
    // bound the traversal-quality drift, and the joint palettes are the buffers
    // the main pass already built for this frame.
    //
    // Runs over the skinned objects the book selected for this frame; the book's
    // plan only takes this step when there is at least one. `full_build` builds
    // every skinned BLAS from scratch rather than refitting.
    pub(crate) fn rebuild_skinned(
        &mut self,
        gpu: RtGpu<'_>,
        draw_objects: &[DrawObject],
        skinned: SkinnedRtInputs<'_>,
        joint_buffers: &[Retained<ProtocolObject<dyn MTLBuffer>>],
        frame: RtFrame,
        full_build: bool,
    ) -> RenderResult<RtUpdate> {
        let RtGpu {
            device,
            command_queue,
            ..
        } = gpu;
        // A slot a failed update left live is still traced by the frames since, so
        // this frame skips rather than rewrite it.
        if !self.ring_liveness.writable(frame.ring_slot, frame.id) {
            return Ok(RtUpdate::Skipped);
        }
        // The deformed buffer mirrors the skinned vertex buffer's indexing, so it
        // spans the highest vertex any visible skinned object reaches.
        let deformed_bytes =
            (usize::saturating_from(self.book.skinned_vertex_extent(skinned.objects))
                * VERTEX_STRIDE)
                .max(VERTEX_STRIDE);
        self.book.fill_skinned_shapes(skinned.objects, 0);

        let skinned_indices = skinned.index_buffer.clone();
        let slot = self.ring.slot(frame.ring_slot);

        // A (re)grown deformed buffer invalidates every descriptor built over the
        // old one, so it counts as a shape change even when the triangles did not
        // move.
        let (deformed_verts, deformed_fresh) = slot.deformed(device, deformed_bytes)?;
        let shape_changed = deformed_fresh || !slot.refit.matches(self.book.skinned_shapes());
        if shape_changed {
            slot.set_skinned(allocate_skinned_blas(
                device,
                deformed_verts.as_ref(),
                skinned_indices.as_ref(),
                self.book.skinned_shapes(),
            )?);
        }

        // TLAS instances + geometry table in the book's order, the skinned
        // instances naming this slot's skinned BLAS.
        let head_ids = &self.head_ids;
        self.book
            .fill_instances(draw_objects, Some(skinned.objects), |model, id, blas| {
                book_instance(model, id, blas, head_ids, slot.skinned_blas())
            });
        let book = &self.book;
        let skinned_objects = book.visible_skinned();
        let shapes = book.skinned_shapes();
        let instances = book.instances();
        let geom = book.geom_table();

        // This frame's instance descriptors + geometry entries, written straight
        // into the slot's upload buffers.
        let instance_buffer = slot.instances(device, size_of_val(instances))?;
        let geom_table = slot.geom_table(device, size_of_val(geom))?;
        write_buffer_slice(&instance_buffer, instances)?;
        write_buffer_slice(&geom_table, geom)?;

        // The TLAS descriptor pins the instance buffer and the instance count;
        // while neither changes the cached one drives every rebuild.
        let key = TlasKey {
            slot_generation: slot.generation(),
            instance_count: instances.len(),
        };
        let tlas_desc = if let Some(desc) = slot.tlas_desc(key) {
            desc
        } else {
            let desc = make_tlas_desc(&instance_buffer, instances.len());
            slot.set_tlas_desc(key, desc.clone());
            desc
        };
        let tlas_sizes = device.accelerationStructureSizesWithDescriptor(&tlas_desc);
        let tlas = slot.tlas(device, tlas_sizes.accelerationStructureSize)?;
        let scratch_buffer = slot.scratch(
            device,
            slot_scratch_bytes(slot.blas_scratch(), tlas_sizes.buildScratchBufferSize),
        )?;

        // Stage 1: skin compute on its own command buffer, committed WITHOUT
        // waiting. Same-queue commit order runs it before the build below (which
        // reads the deformed buffer it writes), the same FIFO ordering the build →
        // trace step and the whole render graph rely on. The palettes it binds are
        // this frame's pre-built joint buffers, which live for the whole frame, so
        // nothing transient has to outlive the async dispatch. A fault can no
        // longer be caught synchronously, so it is logged from a completion handler.
        {
            let skin_cmd = command_queue.commandBuffer().ok_or_else(|| {
                RenderError::Other("failed to create RT skin command buffer".into())
            })?;
            skin_cmd.setLabel(Some(&crate::metal::pipeline::ns_str("rt_skin")));
            let cenc = skin_cmd.computeCommandEncoder().ok_or_else(|| {
                RenderError::Other("failed to create RT skin compute encoder".into())
            })?;
            encode_skin_dispatch(
                &cenc,
                &skinned,
                skinned_objects,
                deformed_verts.as_ref(),
                SkinPalettes::Prebuilt {
                    buffers: joint_buffers,
                    identity: &self.identity_palette,
                },
            )?;
            cenc.endEncoding();
            super::fault_log::attach_fault_logger(&skin_cmd, "RT skinning compute");
            skin_cmd.commit();
        }

        // Settle build-or-refit last, once every fallible step above has passed:
        // recording a build the encoder never ran would leave the slot claiming a
        // tree a later refit could not update.
        let update = slot.refit.plan(shapes, shape_changed || full_build);

        // Stage 2: skinned BLAS + TLAS update, committed WITHOUT waiting:
        // same-queue commit order runs it after the skin compute above and before
        // this frame's reflection trace (committed later on the shared queue), the
        // same FIFO ordering the render graph relies on for every cross-pass read.
        //
        // Each BLAS gets its OWN acceleration-structure encoder. Metal does not
        // guarantee the order (or non-overlap) of builds within a single encoder,
        // so a TLAS that references BLAS built in the same encoder can read them
        // half-built, and builds sharing one scratch buffer can stomp on it.
        // Separate encoders within the command buffer serialize, which both orders
        // the TLAS after its BLAS and makes the shared scratch safe to reuse. (The
        // static-only `rebuild_tlas` path never hit this because its BLAS were all
        // built on earlier command buffers.)
        {
            let cmd = command_queue.commandBuffer().ok_or_else(|| {
                RenderError::Other("failed to create RT skinned rebuild command buffer".into())
            })?;
            cmd.setLabel(Some(&crate::metal::pipeline::ns_str("rt_build")));
            for (acc, prim) in slot.skinned_blas().iter().zip(slot.skinned_descs()) {
                let enc = cmd.accelerationStructureCommandEncoder().ok_or_else(|| {
                    RenderError::Other("failed to create acceleration-structure encoder".into())
                })?;
                match update {
                    AccelUpdate::Build => {
                        enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
                            acc,
                            prim,
                            &scratch_buffer,
                            0,
                        );
                    }
                    // A nil destination refits in place, which is legal here
                    // because the slot's previous writer has retired and the
                    // structure was built with `MTLAccelerationStructureUsage::Refit`.
                    // SAFETY: `acc` and `scratch_buffer` are owned by the ring for
                    // longer than this command buffer runs, the descriptor is the
                    // one `acc` was built from, and the scratch was sized from that
                    // same descriptor's reported build size.
                    AccelUpdate::Refit => unsafe {
                        enc.refitAccelerationStructure_descriptor_destination_scratchBuffer_scratchBufferOffset(
                            acc,
                            prim,
                            None,
                            Some(&scratch_buffer),
                            0,
                        );
                    },
                }
                enc.endEncoding();
            }
            // The TLAS, in its own encoder after every BLAS is updated. It
            // references the persistent static/cluster head AND this frame's
            // skinned BLAS, all built on earlier encoders / command buffers, so
            // every one must be declared resident here.
            let enc = cmd.accelerationStructureCommandEncoder().ok_or_else(|| {
                RenderError::Other("failed to create acceleration-structure encoder".into())
            })?;
            self.resident
                .declare(&enc, book.head().iter().chain(slot.skinned_blas().iter()));
            enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
                &tlas,
                &tlas_desc,
                &scratch_buffer,
                0,
            );
            enc.endEncoding();
            super::fault_log::attach_fault_logger(&cmd, "RT skinned BLAS + TLAS build");
            cmd.commit();
        }

        // Publish this slot's structures. Only the skinned tail of `blas` rotates;
        // the static/cluster head is untouched. Handles that were NOT ring-owned
        // (the seed build's, or a topology refresh's) are parked in the retire pool
        // rather than dropped, because a prior in-flight frame's trace can still
        // reach them; once the ring owns them there is nothing left to retire.
        let takeover = !self.ring_published;
        let old_skinned = self.book.replace_tail(slot.skinned_blas().iter().cloned());
        let old_tlas = std::mem::replace(&mut self.tlas, tlas);
        let old_geom_table = std::mem::replace(&mut self.geom_table, geom_table);
        let old_deformed = std::mem::replace(&mut self.deformed_verts, deformed_verts);
        if takeover {
            let mut structures = old_skinned;
            structures.push(old_tlas);
            self.retire_pool.push(
                frame.id,
                RetiredRt {
                    structures,
                    buffers: vec![old_geom_table, old_deformed],
                },
            );
        }
        self.skinned_indices = skinned_indices;
        self.book.commit_skinned();
        self.ring_published = true;
        self.ring_liveness.publish(frame.ring_slot, frame.id);
        self.retire_parked(frame.id);
        Ok(RtUpdate::Done)
    }

    // Hand the orphans a refresh parked to the BLAS pool now that a TLAS built
    // after it published, retiring those it has no room for.
    fn retire_parked(&mut self, frame_id: u64) {
        let structures: Vec<Structure> = self
            .book
            .take_parked()
            .into_iter()
            .filter_map(|b| {
                let bytes = u64::saturating_from(b.size());
                self.blas_pool.give(b, bytes, frame_id)
            })
            .collect();
        if !structures.is_empty() {
            self.retire_pool.push(
                frame_id,
                RetiredRt {
                    structures,
                    buffers: Vec::new(),
                },
            );
        }
    }

    // Stop publishing the skinned structures the TLAS no longer references: fall
    // back to the persistent dummy deformed buffer and let every ring slot forget
    // the trees it built. A ring slot may be rewritten in place only because the
    // frame that wrote it is the only frame that binds it, so the moment the
    // skinned path stops running its handles have to go with it. The `tail` may
    // still be reached by an in-flight trace through the TLAS it was published
    // in, so it is retired rather than dropped: a ring slot dropping its handles
    // does not free a structure a retired copy still holds.
    fn release_skinned(&mut self, tail: Vec<Structure>, frame_id: u64) {
        if !tail.is_empty() {
            self.retire_pool.push(
                frame_id,
                RetiredRt {
                    structures: tail,
                    buffers: Vec::new(),
                },
            );
        }
        self.deformed_verts = self.deformed_dummy.clone();
        self.ring.release_all();
        self.ring_liveness.unpublish(frame_id);
    }

    // Drop resources parked by prior skinned rebuilds that the frames-in-flight
    // fence now guarantees no in-flight frame can still read (`depth` =
    // frames-in-flight; see [`RetirePool::collect`]). Called once per frame.
    pub(crate) fn retire_completed(&mut self, frame_id: u64, depth: usize) {
        self.retire_pool.collect(frame_id, depth as u64);
        while self.blas_pool.pop_expired(frame_id).is_some() {}
    }
}

// Build the compute pipeline that deforms skinned vertices for ray tracing
// (`rt_skin.hlsl`). Compiled only when RT reflections are on and the GPU
// supports ray tracing, alongside the reflection pipelines.
pub(crate) fn build_rt_skin_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    compute_pipeline(device, &super::builtin_shaders::RT_SKIN, hot_reload)
}

// Fail if a command buffer faulted on the GPU. `waitUntilCompleted` returns
// regardless of success, so without this a faulted build/skin would leave a
// corrupt structure the trace then reads. Surfacing it as an `Err` lets the
// non-fatal per-frame update skip the frame (keeping the last good BVH) instead
// of tracing garbage. `what` names the stage so a fault points at the actual
// culprit (the skin compute vs the acceleration-structure build) rather than a
// generic message, and a downstream `SubmissionsIgnored` cascade is
// distinguishable from an original fault by its error code.
fn check_build_status(
    cmd: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
    what: &str,
) -> RenderResult<()> {
    completed_command_buffer(cmd, format_args!("RT {what}"))
}

// Upload a `#[repr(C)]` slice to a new shared GPU buffer.
pub(super) fn upload_buffer<T: Copy>(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    data: &[T],
    what: &str,
) -> RenderResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let bytes = size_of_val(data);
    if bytes == 0 {
        // Metal rejects a zero-length buffer, and `data.as_ptr()` on an empty
        // slice is dangling, so there is no byte to copy from. Hand back one
        // zeroed element instead: a shader binding the buffer as an array of `T`
        // (the geometry table of a zero-instance TLAS) needs a whole one.
        return device
            .newBufferWithLength_options(
                size_of::<T>().max(1),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| allocation_failed(format_args!("buffer for {what}")));
    }
    let ptr = NonNull::new(data.as_ptr() as *mut std::ffi::c_void)
        .ok_or_else(|| RenderError::Other(format!("{what}: null data pointer")))?;
    // SAFETY: `ptr`/`bytes` describe the live, non-empty `data` slice, and
    // Metal copies those bytes into the new buffer before the call returns.
    unsafe {
        device.newBufferWithBytes_length_options(ptr, bytes, MTLResourceOptions::StorageModeShared)
    }
    .ok_or_else(|| allocation_failed(format_args!("buffer for {what}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::gfx::render_types;

    #[test]
    fn pack_instance_transform_drops_affine_row_and_keeps_columns() {
        // A model with a clear translation column and a scaled basis. The packed
        // 4x3 keeps the top three rows of each column; the [0,0,0,1] row is gone.
        let model = [
            [2.0, 0.0, 0.0, 0.0], // column 0: scaled x basis
            [0.0, 3.0, 0.0, 0.0], // column 1: scaled y basis
            [0.0, 0.0, 4.0, 0.0], // column 2: scaled z basis
            [5.0, 6.0, 7.0, 1.0], // column 3: translation
        ];
        let p = pack_instance_transform(model);
        assert_eq!(
            (p.columns[0].x, p.columns[0].y, p.columns[0].z),
            (2.0, 0.0, 0.0)
        );
        assert_eq!(
            (p.columns[1].x, p.columns[1].y, p.columns[1].z),
            (0.0, 3.0, 0.0)
        );
        assert_eq!(
            (p.columns[2].x, p.columns[2].y, p.columns[2].z),
            (0.0, 0.0, 4.0)
        );
        // The translation lands in the fourth column, not a transposed row.
        assert_eq!(
            (p.columns[3].x, p.columns[3].y, p.columns[3].z),
            (5.0, 6.0, 7.0)
        );
    }

    #[test]
    fn rt_geom_entry_is_128_bytes() {
        // The kernel's matching struct relies on this exact size + 16-byte
        // alignment for the array stride to agree. tint+roughness fill one
        // float4; metallic + emissive[3] fill the next so the float4x4 model
        // lands on a 16-byte boundary, exactly as MSL lays the struct out
        // (emissive is a `packed_float3` there, matching `[f32; 3]` here).
        assert_eq!(size_of::<render_types::RtGeomEntry>(), 128);
    }

    #[test]
    fn slot_scratch_covers_the_largest_build_and_never_reaches_zero() {
        // One scratch buffer serves every skinned BLAS and the TLAS, so it takes
        // the larger requirement whichever side it comes from.
        assert_eq!(slot_scratch_bytes(4096, 1024), 4096);
        assert_eq!(slot_scratch_bytes(1024, 4096), 4096);
        // Metal rejects a zero-length buffer, so a scene whose structures need no
        // scratch still asks for a byte.
        assert_eq!(slot_scratch_bytes(0, 0), 1);
    }
}
