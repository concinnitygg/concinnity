//! Hardware ray-tracing acceleration structures for the Metal backend. Builds,
//! from the shared static vertex / index buffers and the `DrawObject` list, the
//! bottom- and top-level acceleration structures (BLAS / TLAS) the RT-reflection
//! kernel traces against, plus a per-instance geometry table the kernel uses to
//! fetch the hit triangle and shade it.
//!
//! One primitive BLAS per object over its slice of the shared buffers, one
//! instance in the TLAS per object (transform = the object's model matrix,
//! instance_id = the object index). The BLAS describe object-space geometry and
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

use concinnity_core::gfx::render_types::{DrawObject, InstancedCluster, SkinnedDrawObject};
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::post::rt_reflections::RtReflectionSettings;
use concinnity_core::render::retire_pool::RetirePool;
use concinnity_core::render::rt_accel::{
    AccelBook, EmptyHead, FailureStreak, InstanceBlas, RefreshMode, RtUpdate, SeedSet,
    SlotLiveness, empty_head,
};
use concinnity_core::render::rt_geom::RtDynamicMode;
use concinnity_core::render::rt_refit::{BlasUpdate, SkinnedShape};
use concinnity_core::render::rt_topology::traced_skinned;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSArray;
use objc2_metal::{
    MTLAccelerationStructure, MTLAccelerationStructureCommandEncoder,
    MTLAccelerationStructureGeometryDescriptor, MTLAccelerationStructureInstanceDescriptor,
    MTLAccelerationStructureInstanceDescriptorType, MTLAccelerationStructureInstanceOptions,
    MTLAccelerationStructureTriangleGeometryDescriptor, MTLAccelerationStructureUsage,
    MTLAttributeFormat, MTLBuffer, MTLCommandBuffer as _, MTLCommandEncoder as _,
    MTLCommandQueue as _, MTLComputeCommandEncoder as _, MTLComputePipelineState, MTLDevice as _,
    MTLIndexType, MTLInstanceAccelerationStructureDescriptor, MTLPackedFloat3, MTLPackedFloat4x3,
    MTLPrimitiveAccelerationStructureDescriptor, MTLRenderCommandEncoder, MTLRenderPipelineState,
    MTLRenderStages, MTLResource, MTLResourceOptions, MTLResourceUsage, MTLSize,
};
use std::ptr::NonNull;
// Shared with the Vulkan and DirectX hosts: one `.hlsl` declares it.
use concinnity_core::render::uniforms::SkinParams;

use super::builtin_shaders::compute_pipeline;
use super::context::write_buffer_slice;
use super::encode::ComputeEncode;
use super::error::{allocation_failed, completed_command_buffer};
use super::rt_ring::{RtFrameRing, SkinnedBlasSet, TlasKey};

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
    // `PostProcessConfig` sets `ray_traced_reflections` AND the GPU supports
    // ray tracing; gates the RT pass. RT takes precedence over SSR resolve
    // and reuses `ssr.targets.output` as its resolve target.
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
    // plus the skinned tail, and the update policy over them. The book is the
    // sole CPU owner that keeps every BLAS alive: a TLAS does not retain the
    // structures it references, and the `useResource` an encoder issues only
    // declares residency, so a BLAS must stay owned as long as any in-flight
    // trace can reach it through the TLAS.
    book: AccelBook<Structure, MTLAccelerationStructureInstanceDescriptor>,
    // The top-level (instance) acceleration structure the kernel traces.
    pub tlas: Structure,
    // `[RtGeomEntry; instance_count]`, indexed by the intersector's
    // `instance_id`. Lets the kernel find the hit triangle + shade it. Carries
    // each instance's model matrix, which the kernel uses to bring the hit
    // normal to world space, so it moves in lockstep with the TLAS transforms.
    pub geom_table: Retained<ProtocolObject<dyn MTLBuffer>>,
    // Private scratch for the seed build and the static `rebuild_tlas`, which
    // grows it as the instance count rises. One buffer serves every frame
    // because both paths commit and wait; the paths that do not (the skinned
    // update, a topology refresh) take scratch of their own.
    scratch: Retained<ProtocolObject<dyn MTLBuffer>>,
    // The TLAS instance-descriptor buffer of the last *asynchronous* static
    // build. Only the TLAS build reads it (the built TLAS bakes the instances),
    // so it is not bound to the trace; it is held here because a topology
    // refresh commits without waiting and the build keeps reading it after the
    // call returns. The per-frame skinned update does not use it -- its instance
    // buffer is a ring slot the ring keeps alive.
    instance_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,

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
    // A 1-vertex Shared buffer bound at the deformed-vertex slot whenever no
    // skinned geometry is being traced, so the encoder always has a buffer for
    // the binding the shader declares (the skinned branch is never taken then).
    // Also what `release_skinned` falls back to when the ring stops publishing.
    deformed_dummy: Retained<ProtocolObject<dyn MTLBuffer>>,
    // A single-identity joint palette the skin dispatch binds for an object with
    // no pose, so it deforms to bind pose rather than reading whatever the
    // frame's pre-built palette buffer happens to hold.
    identity_palette: Retained<ProtocolObject<dyn MTLBuffer>>,
    // Bumped whenever the persistent BLAS head (`blas[..static_blas_count]`)
    // changes identity, so every ring slot's cached TLAS descriptor rebuilds.
    head_generation: u64,
    // Whether `tlas` / `geom_table` / `deformed_verts` / the skinned tail of
    // `blas` are currently ring-owned clones. When they are not -- right after
    // the seed build or a topology refresh built fresh ones -- the next skinned
    // update must park the outgoing handles in `retire_pool` instead of dropping
    // them, because a prior in-flight frame's trace can still reach them.
    ring_published: bool,
    // Which ring slot the live TLAS was built from, so a slot a failed skinned
    // update left live is not rewritten while frames still in flight trace it.
    ring_liveness: SlotLiveness,
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

// Whether the GPU supports hardware ray tracing. Apple-silicon GPUs report
// `true`; Intel / most AMD Macs report `false`, in which case the caller falls
// back to SSR (or no reflections). Mirrors the capability gates the MetalFX /
// HDR paths use at init.
pub(crate) fn raytracing_supported(device: &ProtocolObject<dyn objc2_metal::MTLDevice>) -> bool {
    device.supportsRaytracing()
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
    // SAFETY: plain descriptor property setters, all values in range.
    let geo = unsafe {
        let g = MTLAccelerationStructureTriangleGeometryDescriptor::descriptor();
        g.setVertexBuffer(Some(vertex_buffer));
        g.setVertexBufferOffset(base_vertex * VERTEX_STRIDE);
        g.setVertexStride(VERTEX_STRIDE);
        g.setVertexFormat(MTLAttributeFormat::Float3);
        g.setIndexBuffer(Some(index_buffer));
        g.setIndexBufferOffset(index_offset * index_bytes);
        g.setIndexType(index_type);
        g.setTriangleCount(index_count / 3);
        g
    };
    let geo_ref: &MTLAccelerationStructureGeometryDescriptor = &geo;
    let geos = NSArray::from_slice(&[geo_ref]);
    let prim = MTLPrimitiveAccelerationStructureDescriptor::descriptor();
    prim.setGeometryDescriptors(Some(&geos));
    prim.setUsage(usage);
    prim
}

// An instance descriptor with an explicit transform + BLAS index
// (`accelerationStructureIndex` selects which BLAS this instance uses). The
// shader indexes the geometry table by the intersector's `instance_id`, which
// for `MTLAccelerationStructureInstanceDescriptorType::Default` is the
// instance's position in the instance buffer (NOT the
// `accelerationStructureIndex`), so the table carries one entry per instance,
// in instance order (multiple cluster instances share one BLAS but get distinct
// entries). See `build_rt_accel`.
fn instance_desc_at(
    model: [[f32; 4]; 4],
    blas_index: u32,
) -> MTLAccelerationStructureInstanceDescriptor {
    MTLAccelerationStructureInstanceDescriptor {
        transformationMatrix: pack_instance_transform(model),
        options: MTLAccelerationStructureInstanceOptions::Opaque,
        mask: 0xFF,
        intersectionFunctionTableOffset: 0,
        accelerationStructureIndex: blas_index,
    }
}

// The instance descriptor for an instance the book laid out. Its BLAS index is
// the BLAS's position in the TLAS's array, where the skinned BLAS follow the
// `static_blas_count` head entries.
fn book_instance(
    model: [[f32; 4]; 4],
    blas: InstanceBlas<'_, Structure>,
    static_blas_count: usize,
) -> MTLAccelerationStructureInstanceDescriptor {
    let index = match blas {
        InstanceBlas::Head { index, .. } | InstanceBlas::Fresh { index } => index,
        InstanceBlas::Skinned { n } => static_blas_count + n,
    };
    instance_desc_at(model, index as u32)
}

// The TLAS descriptor over `blas_refs`, reading transforms from
// `instance_buffer`. Takes plain references so the BLAS array can be assembled
// from more than one source (e.g. persistent static BLAS followed by this
// frame's fresh skinned BLAS).
fn make_tlas_desc_from_refs(
    blas_refs: &[&ProtocolObject<dyn MTLAccelerationStructure>],
    instance_buffer: &ProtocolObject<dyn MTLBuffer>,
    instance_count: usize,
) -> Retained<MTLInstanceAccelerationStructureDescriptor> {
    let blas_array = NSArray::from_slice(blas_refs);
    let desc = MTLInstanceAccelerationStructureDescriptor::descriptor();
    desc.setInstancedAccelerationStructures(Some(&blas_array));
    desc.setInstanceCount(instance_count);
    desc.setInstanceDescriptorBuffer(Some(instance_buffer));
    desc.setInstanceDescriptorType(MTLAccelerationStructureInstanceDescriptorType::Default);
    desc
}

// The TLAS descriptor over `blas`, reading transforms from `instance_buffer`.
fn make_tlas_desc(
    blas: &[Retained<ProtocolObject<dyn MTLAccelerationStructure>>],
    instance_buffer: &ProtocolObject<dyn MTLBuffer>,
    instance_count: usize,
) -> Retained<MTLInstanceAccelerationStructureDescriptor> {
    let blas_refs: Vec<&ProtocolObject<dyn MTLAccelerationStructure>> =
        blas.iter().map(|b| b.as_ref()).collect();
    make_tlas_desc_from_refs(&blas_refs, instance_buffer, instance_count)
}

// Declare BLAS that a TLAS build references resident on the build encoder. A
// TLAS build reads the primitive structures its instances point to; unlike a
// direct buffer binding, that indirect reference does not make them resident,
// so Metal requires an explicit `useResource` (or `useHeap`) or the build can
// read a non-resident structure and fault. Only structures built on an *earlier*
// command buffer need this: BLAS built in the same encoder are already resident
// and ordered by same-encoder hazard tracking, so they must NOT be passed here.
fn declare_blas_resident<'a>(
    enc: &ProtocolObject<dyn MTLAccelerationStructureCommandEncoder>,
    blas: impl IntoIterator<Item = &'a Retained<ProtocolObject<dyn MTLAccelerationStructure>>>,
) {
    for b in blas {
        enc.useResource_usage(ProtocolObject::from_ref(&**b), MTLResourceUsage::Read);
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
    skinned: &SkinnedRtInputs,
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
    skinned: &SkinnedRtInputs,
    skinned_objects: &[usize],
    deformed_verts: &ProtocolObject<dyn MTLBuffer>,
    palettes: SkinPalettes,
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
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
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
    bufs: SkinDispatchBuffers,
    obj: &SkinnedDrawObject,
    joint_count: usize,
    threadgroup: usize,
) {
    // The RT dispatch runs the base pose; morphing happens in the per-frame main
    // fold. `target_count == 0` leaves the morph slots unread, so a dummy binding
    // satisfies them.
    let params = SkinParams {
        vertex_base: obj.vertex_base,
        vertex_count: obj.vertex_count as u32,
        joint_count: joint_count as u32,
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
                .map(|m| m.len().max(1))
                .unwrap_or(1);
            // Morphing needs this object's deltas AND this frame's weights.
            // Both slots are bound unconditionally, so a missing one takes the
            // vertex buffer as a dummy and `target_count` goes to zero, which
            // leaves the kernel reading neither.
            let morph = self.skinned.morphs.get(i).and_then(|m| m.as_ref());
            let weights = weight_bufs.get(i);
            let params = SkinParams {
                vertex_base: obj.vertex_base,
                vertex_count: obj.vertex_count as u32,
                joint_count: joint_count as u32,
                target_count: match (morph, weights) {
                    (Some(m), Some(_)) => m.target_count,
                    _ => 0,
                },
            };
            cenc.set_buffer(svb.as_ref(), 0, 0);
            cenc.set_buffer(deformed, 0, 1);
            cenc.set_buffer(joint_buf.as_ref(), 0, 2);
            cenc.set_value(&params, 3);
            cenc.set_buffer(morph.map_or(svb.as_ref(), |m| m.buffer.as_ref()), 0, 4);
            cenc.set_buffer(weights.map_or(svb.as_ref(), |b| b.as_ref()), 0, 5);
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
// error) when there is no resident triangle geometry to trace: the caller then
// leaves RT disabled and the pass falls back to the base scene.
//
// `skinned`, when present, adds skeletally-animated geometry: a compute pass
// deforms each skinned object's vertices into a fresh model-space buffer, and
// one BLAS per skinned object is built over that buffer. Because
// the pose changes every frame the whole structure is rebuilt per frame (the
// caller forces this); fresh allocations keep it hazard-free.
pub(crate) fn build_rt_accel(
    gpu: RtGpu,
    static_geometry: RtStaticGeometry,
    scene: RtSceneGeometry,
    texture_counts: RtTextureCounts,
    skinned: Option<SkinnedRtInputs>,
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
    if seed.is_empty() && !skinned_list.iter().any(traced_skinned) {
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
                obj.base_vertex as usize,
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
    let albedo_count = albedo_count as u32;
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
    let deformed_bytes =
        (book.skinned_vertex_extent(skinned_list) as usize * VERTEX_STRIDE).max(VERTEX_STRIDE);
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
            .newBufferWithLength_options(
                std::mem::size_of::<u32>(),
                MTLResourceOptions::StorageModePrivate,
            )
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
    // `instance_id`, which is the instance's position in the instance buffer
    // (NOT the `accelerationStructureIndex`), so there is exactly one entry per
    // TLAS instance, in the book's instance order.
    let static_blas_count = book.static_blas_count();
    book.fill_instances(draw_objects, Some(skinned_list), |model, _, blas| {
        book_instance(model, blas, static_blas_count)
    });
    let instance_buffer = upload_buffer(device, book.instances(), "RT instance descriptors")?;
    let geom_table = upload_buffer(device, book.geom_table(), "RT geometry table")?;

    let tlas_desc = make_tlas_desc(book.blas(), &instance_buffer, book.instances().len());
    let tlas_sizes = device.accelerationStructureSizesWithDescriptor(&tlas_desc);
    let tlas = device
        .newAccelerationStructureWithSize(tlas_sizes.accelerationStructureSize)
        .ok_or_else(|| allocation_failed("TLAS"))?;
    // Size the scratch for the largest of every BLAS build and the TLAS build
    // so the per-frame TLAS rebuild can reuse the same buffer.
    max_scratch = max_scratch.max(tlas_sizes.buildScratchBufferSize);

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
    declare_blas_resident(&enc, book.blas());
    enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
        &tlas, &tlas_desc, &scratch, 0,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    check_build_status(&cmd, "acceleration-structure build")?;

    let identity_palette = upload_buffer(device, &[IDENTITY4], "RT identity palette")?;

    Ok(Some(RtAccelData {
        book,
        tlas,
        geom_table,
        scratch,
        instance_buffer,
        deformed_verts,
        skinned_indices,
        retire_pool: RetirePool::new(),
        ring: RtFrameRing::new(frames_in_flight),
        deformed_dummy,
        identity_palette,
        head_generation: 0,
        // Everything above is a fresh allocation, not a ring clone, so the first
        // skinned update has to retire it rather than drop it.
        ring_published: false,
        ring_liveness: SlotLiveness::new(frames_in_flight),
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

impl RtAccelData {
    // Keep the static BLAS; rebuild the TLAS + geometry table from the
    // transforms the book's `next_step` collected, with fresh allocations, then
    // build on a separate command buffer (committed and waited). Fresh
    // allocations mean no prior in-flight frame can observe a half-updated
    // structure: the old TLAS / table stay alive (retained by their command
    // buffers) until those frames complete.
    pub(crate) fn rebuild_tlas(
        &mut self,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        command_queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
        draw_objects: &[DrawObject],
        frame_id: u64,
    ) -> RenderResult<()> {
        // Freshly-transformed draw-object instances, then the cluster instances
        // (clusters are baked static; their BLAS never move in the head).
        let static_blas_count = self.book.static_blas_count();
        self.book
            .fill_instances(draw_objects, None, |model, _, blas| {
                book_instance(model, blas, static_blas_count)
            });
        let instance_count = self.book.instances().len();
        let instance_buffer =
            upload_buffer(device, self.book.instances(), "RT instance descriptors")?;
        let geom_table = upload_buffer(device, self.book.geom_table(), "RT geometry table")?;
        let tlas_desc = make_tlas_desc(self.book.head(), &instance_buffer, instance_count);
        let sizes = device.accelerationStructureSizesWithDescriptor(&tlas_desc);
        let tlas = device
            .newAccelerationStructureWithSize(sizes.accelerationStructureSize)
            .ok_or_else(|| allocation_failed("TLAS"))?;
        // Reuse the scratch sized at init (the prior frame's build completed
        // before we got here, so it is free) -- but a topology refresh can change
        // the instance count, and a larger TLAS needs more build scratch than the
        // init sizing. Grow it when so. Replacing the handle is safe: this path
        // is synchronous (commit + wait), and an in-flight command buffer that
        // still references the old scratch retains it independently of this Vec.
        if (sizes.buildScratchBufferSize as u64) > self.scratch.length() as u64 {
            self.scratch = device
                .newBufferWithLength_options(
                    sizes.buildScratchBufferSize.max(1),
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| allocation_failed("grown RT scratch buffer"))?;
        }

        let cmd = command_queue.commandBuffer().ok_or_else(|| {
            RenderError::Other("failed to create RT rebuild command buffer".into())
        })?;
        let enc = cmd.accelerationStructureCommandEncoder().ok_or_else(|| {
            RenderError::Other("failed to create acceleration-structure encoder".into())
        })?;
        // Every BLAS the rebuilt TLAS references was built on an earlier command
        // buffer (none are rebuilt here), so all must be declared resident.
        declare_blas_resident(&enc, self.book.head());
        enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
            &tlas,
            &tlas_desc,
            &self.scratch,
            0,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        check_build_status(&cmd, "TLAS rebuild")?;

        self.tlas = tlas;
        self.geom_table = geom_table;
        if let Some(tail) = self.book.commit_static() {
            self.release_skinned(tail, frame_id);
        }
        self.retire_parked(frame_id);
        // Fresh allocations, so a later skinned update has to retire rather than
        // drop them.
        self.ring_published = false;
        Ok(())
    }

    // Whether the BVH has no draw-object and no cluster geometry left.
    pub(crate) fn is_empty(&self) -> bool {
        self.book.is_empty()
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
        self.book.set_albedo_count(albedo_count as u32);
    }

    // The update policy: what this frame's update plans and runs.
    pub(crate) fn book_mut(
        &mut self,
    ) -> &mut AccelBook<Structure, MTLAccelerationStructureInstanceDescriptor> {
        &mut self.book
    }

    // Bring the draw-object BLAS head in line with the current participating draw
    // set: reuse every BLAS whose geometry is unchanged (or none, under
    // `RefreshMode::RebuildAll`), build the new / changed ones, and commit the
    // refreshed head. The cluster BLAS and any skinned tail are preserved. Used
    // when streamed chunks are added/removed, props are cloned, or a material
    // edit changes RT participation.
    //
    // On the skinned path (`skinned_follows`) the caller's `rebuild_skinned`
    // builds this frame's TLAS over the refreshed head + the skinned tail, so no
    // TLAS is built here and the ring is left alone; the orphans stay parked in
    // the book, and declared resident by every trace, until a TLAS that does not
    // reference them publishes, since a failed or skipped skinned step leaves the
    // old one live. On the static path the TLAS + geometry table are rebuilt over
    // the refreshed head in the same command buffer and any skinned tail stops
    // being published. A refresh that leaves no draw or cluster geometry follows
    // `empty_head`: with no skinned geometry at all it builds nothing and the
    // caller drops the BVH.
    //
    // Fully asynchronous, mirroring `rebuild_skinned`: NO `waitUntilCompleted`.
    // The new BLAS (and, on the static path, the TLAS) build on one command buffer
    // committed on the shared queue ahead of this frame's reflection-trace command
    // buffer, ordered by same-queue FIFO commit. Outgoing / transient resources
    // (orphan BLAS, the replaced TLAS + geometry table + instance buffer, and the
    // build scratch) are parked in `retire_pool` rather than freed in place:
    // `useResource` declares residency not lifetime, and the build keeps reading
    // the scratch / instance buffer after this returns, so they must outlive the
    // frames whose still-in-flight trace could reach them. Every allocation
    // precedes the commit, so a failure leaves the live BVH untouched.
    pub(crate) fn refresh_static_topology(
        &mut self,
        gpu: RtGpu,
        static_geometry: RtStaticGeometry,
        draw_objects: &[DrawObject],
        options: RtTopologyRefreshOptions,
    ) -> RenderResult<()> {
        let RtGpu {
            device,
            command_queue,
            ..
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
        let leaves_nothing = self.book.refresh_leaves_nothing(&refresh);

        // Allocate (but do not yet build) a fresh BLAS for every slot the plan did
        // not match to an existing one.
        let mut fresh: Vec<Option<Structure>> =
            (0..refresh.indices().len()).map(|_| None).collect();
        let mut build_jobs: Vec<(usize, Retained<MTLPrimitiveAccelerationStructureDescriptor>)> =
            Vec::new();
        let mut max_scratch: usize = 0;
        for (j, idx) in refresh.fresh_slots() {
            let obj = &draw_objects[idx];
            let prim = prim_desc_for(
                vertex_buffer,
                index_buffer,
                obj.base_vertex as usize,
                obj.index_offset,
                obj.index_count,
                MTLIndexType::UInt32,
                MTLAccelerationStructureUsage::None,
            );
            let sizes = device.accelerationStructureSizesWithDescriptor(&prim);
            let acc = device
                .newAccelerationStructureWithSize(sizes.accelerationStructureSize)
                .ok_or_else(|| allocation_failed("topology-refresh BLAS"))?;
            acc.setLabel(Some(&crate::metal::pipeline::ns_str("rt_topology_blas")));
            max_scratch = max_scratch.max(sizes.buildScratchBufferSize);
            fresh[j] = Some(acc);
            build_jobs.push((j, prim));
        }

        // On the static path, rebuild the TLAS + geometry table over the refreshed
        // head with the current transforms. An empty head that skinned geometry
        // can rejoin still gets a (zero-instance) TLAS, so the trace stops
        // reaching what left.
        let empty = leaves_nothing.then(|| empty_head(skinned_follows, skinned_present));
        let build_tlas = !skinned_follows && empty != Some(EmptyHead::Drop);
        let tlas_build = if build_tlas {
            self.book
                .fill_refresh_instances(&refresh, draw_objects, |model, _, blas| {
                    book_instance(model, blas, 0)
                });
            let instance_count = self.book.instances().len();
            let instance_buffer =
                upload_buffer(device, self.book.instances(), "RT instance descriptors")?;
            let geom_table = upload_buffer(device, self.book.geom_table(), "RT geometry table")?;
            let head: Vec<&ProtocolObject<dyn MTLAccelerationStructure>> = self
                .book
                .refreshed_head(&refresh, &fresh)
                .into_iter()
                .map(|b| b.as_ref())
                .collect();
            let tlas_desc = make_tlas_desc_from_refs(&head, &instance_buffer, instance_count);
            let tlas_sizes = device.accelerationStructureSizesWithDescriptor(&tlas_desc);
            max_scratch = max_scratch.max(tlas_sizes.buildScratchBufferSize);
            let tlas = device
                .newAccelerationStructureWithSize(tlas_sizes.accelerationStructureSize)
                .ok_or_else(|| allocation_failed("TLAS"))?;
            tlas.setLabel(Some(&crate::metal::pipeline::ns_str("rt_tlas")));
            Some((tlas, tlas_desc, instance_buffer, geom_table))
        } else {
            None
        };

        // The commit must follow the builds committed next, so it is checked now,
        // while a failure still leaves nothing committed.
        self.book.check_refresh(&refresh, &fresh)?;

        // Build everything on ONE command buffer, committed WITHOUT waiting. Each
        // new BLAS in its own encoder (Metal does not order builds within an
        // encoder, and they share the scratch); then, when building the TLAS, a
        // final encoder that declares every referenced BLAS resident (all were
        // built on this or an earlier command buffer, so the TLAS build needs the
        // explicit `useResource`, exactly as the full build does).
        let mut retire_buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> = Vec::new();
        if !build_jobs.is_empty() || tlas_build.is_some() {
            let scratch = device
                .newBufferWithLength_options(
                    max_scratch.max(1),
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| allocation_failed("topology-refresh scratch buffer"))?;
            let cmd = command_queue.commandBuffer().ok_or_else(|| {
                RenderError::Other("failed to create topology-refresh command buffer".into())
            })?;
            cmd.setLabel(Some(&crate::metal::pipeline::ns_str("rt_topology_build")));
            for (j, prim) in &build_jobs {
                let Some(acc) = fresh[*j].as_ref() else {
                    continue;
                };
                let enc = cmd.accelerationStructureCommandEncoder().ok_or_else(|| {
                    RenderError::Other("failed to create acceleration-structure encoder".into())
                })?;
                enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
                    acc, prim, &scratch, 0,
                );
                enc.endEncoding();
            }
            if let Some((tlas, tlas_desc, _, _)) = &tlas_build {
                let enc = cmd.accelerationStructureCommandEncoder().ok_or_else(|| {
                    RenderError::Other("failed to create acceleration-structure encoder".into())
                })?;
                declare_blas_resident(&enc, self.book.refreshed_head(&refresh, &fresh));
                enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
                    tlas, tlas_desc, &scratch, 0,
                );
                enc.endEncoding();
            }
            super::fault_log::attach_fault_logger(&cmd, "RT topology build");
            cmd.commit();
            // The async build keeps reading the scratch after this returns.
            retire_buffers.push(scratch);
        }

        // Swap in the refreshed head. The orphaned draw BLAS are referenced by the
        // current (not yet replaced) TLAS, which an in-flight trace may still be
        // reading, and `useResource` is residency not lifetime, so they are retired
        // rather than dropped, and only once a TLAS without them is live.
        let orphans = self.book.commit_refresh(refresh, fresh, draw_objects);
        // The persistent BLAS head changed identity, so every ring slot's cached
        // TLAS descriptor (which pins the array of referenced structures) is stale.
        self.head_generation = self.head_generation.wrapping_add(1);
        let mut retire_structures = Vec::new();
        if skinned_follows {
            self.book.park(orphans);
        } else {
            retire_structures.extend(orphans);
        }
        if let Some((tlas, _, instance_buffer, geom_table)) = tlas_build {
            retire_structures.extend(self.book.take_parked());
            retire_structures.push(std::mem::replace(&mut self.tlas, tlas));
            retire_buffers.push(std::mem::replace(&mut self.geom_table, geom_table));
            retire_buffers.push(std::mem::replace(
                &mut self.instance_buffer,
                instance_buffer,
            ));
            if let Some(tail) = self.book.release_skinned() {
                self.release_skinned(tail, frame_id);
            }
            // Fresh allocations, so a later skinned update has to retire rather
            // than drop them.
            self.ring_published = false;
        }
        if !retire_structures.is_empty() || !retire_buffers.is_empty() {
            self.retire_pool.push(
                frame_id,
                RetiredRt {
                    structures: retire_structures,
                    buffers: retire_buffers,
                },
            );
        }
        Ok(())
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
        gpu: RtGpu,
        draw_objects: &[DrawObject],
        skinned: SkinnedRtInputs,
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
        let deformed_bytes = (self.book.skinned_vertex_extent(skinned.objects) as usize
            * VERTEX_STRIDE)
            .max(VERTEX_STRIDE);
        self.book.fill_skinned_shapes(skinned.objects, 0);

        // TLAS instances + geometry table in the book's order. Skinned BLAS follow
        // the static/cluster head, so their `accelerationStructureIndex` is
        // `static_blas_count + n`. Built before the ring slot is borrowed so the
        // reads of `self` stay disjoint from it.
        let static_blas_count = self.book.static_blas_count();
        let head_generation = self.head_generation;
        self.book
            .fill_instances(draw_objects, Some(skinned.objects), |model, _, blas| {
                book_instance(model, blas, static_blas_count)
            });
        let book = &self.book;
        let skinned_objects = book.visible_skinned();
        let shapes = book.skinned_shapes();
        let instances = book.instances();
        let geom = book.geom_table();

        let skinned_indices = skinned.index_buffer.clone();
        let slot = self.ring.slot(frame.ring_slot);

        // A (re)grown deformed buffer invalidates every descriptor built over the
        // old one, so it counts as a shape change even when the triangles did not
        // move.
        let (deformed_verts, deformed_fresh) = slot.deformed(device, deformed_bytes)?;
        let shape_changed = deformed_fresh || !slot.refit.matches(shapes);
        if shape_changed {
            slot.set_skinned(allocate_skinned_blas(
                device,
                deformed_verts.as_ref(),
                skinned_indices.as_ref(),
                shapes,
            )?);
        }

        // This frame's instance descriptors + geometry entries, written straight
        // into the slot's upload buffers.
        let instance_buffer = slot.instances(device, std::mem::size_of_val(instances))?;
        let geom_table = slot.geom_table(device, std::mem::size_of_val(geom))?;
        write_buffer_slice(&instance_buffer, instances)?;
        write_buffer_slice(&geom_table, geom)?;

        // The TLAS descriptor pins the BLAS array, the instance buffer and the
        // instance count; while none of those change the cached one drives every
        // rebuild, so the per-frame `Vec` of BLAS references is only built when
        // something actually moved.
        let key = TlasKey {
            head_generation,
            slot_generation: slot.generation(),
            instance_count: instances.len(),
        };
        let cached = slot.tlas_desc(key);
        let tlas_desc = match cached {
            Some(desc) => desc,
            None => {
                let refs: Vec<&ProtocolObject<dyn MTLAccelerationStructure>> = book
                    .head()
                    .iter()
                    .map(|b| b.as_ref())
                    .chain(slot.skinned_blas().iter().map(|b| b.as_ref()))
                    .collect();
                let desc = make_tlas_desc_from_refs(&refs, &instance_buffer, instances.len());
                slot.set_tlas_desc(key, desc.clone());
                desc
            }
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
                    BlasUpdate::Build => {
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
                    BlasUpdate::Refit => unsafe {
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
            declare_blas_resident(&enc, book.head().iter().chain(slot.skinned_blas().iter()));
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

    // Retire the orphans a refresh parked until a TLAS built after it published.
    fn retire_parked(&mut self, frame_id: u64) {
        let structures = self.book.take_parked();
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
    }
}

// Build the compute pipeline that deforms skinned vertices for ray tracing
// (`rt_skin.hlsl`). Compiled only when RT reflections are on and the GPU
// supports ray tracing, alongside the reflection pipelines.
pub(crate) fn build_rt_skin_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>> {
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
fn upload_buffer<T: Copy>(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    data: &[T],
    what: &str,
) -> RenderResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let bytes = std::mem::size_of_val(data);
    if bytes == 0 {
        // Metal rejects a zero-length buffer, and `data.as_ptr()` on an empty
        // slice is dangling, so there is no byte to copy from. Hand back one
        // zeroed element instead: a shader binding the buffer as an array of `T`
        // (the geometry table of a zero-instance TLAS) needs a whole one.
        return device
            .newBufferWithLength_options(
                std::mem::size_of::<T>().max(1),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| allocation_failed(format_args!("buffer for {what}")));
    }
    let ptr = std::ptr::NonNull::new(data.as_ptr() as *mut std::ffi::c_void)
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
    use concinnity_core::gfx::render_types;

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
        assert_eq!(std::mem::size_of::<render_types::RtGeomEntry>(), 128);
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
