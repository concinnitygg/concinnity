//! Vulkan ray-query acceleration structures for the hardware ray-traced
//! reflection pass. Builds, from the shared static vertex / index buffers and the
//! `DrawObject` + `InstancedCluster` lists, the bottom- and top-level
//! acceleration structures (BLAS / TLAS) the inline-`rayQueryEXT` reflection
//! shader traces against, plus a per-instance geometry table the shader uses to
//! fetch the hit triangle and shade it.
//!
//! One triangle BLAS per participating static object (over its slice of the
//! shared buffers) and one per instanced cluster; one TLAS instance per object
//! and one per cluster instance (transform = the object/instance model matrix,
//! `instanceCustomIndex` = the geometry-table index). The BLAS describe
//! object-space geometry and never change for a rigid transform; only the TLAS
//! instance transforms (and the geometry table's per-instance model matrices the
//! shader shades with) move when a prop moves.
//!
//! Mirrors `directx/raytrace.rs` (DXR inline ray tracing). Skinned geometry is
//! added per frame (`rebuild_skinned`): a compute pass deforms each skinned
//! object's bind-pose vertices into a model-space buffer, one BLAS
//! per skinned object is built or updated over it, and the TLAS + geometry table
//! are rebuilt over the persistent static/cluster BLAS plus the skinned tail.
//!
//! Every resource those two per-frame paths write lives in a ring rather than
//! being allocated fresh: `skinned_ring` is one slot per frame in flight, keyed on
//! `frame_idx`, `static_ring` advances a cursor one slot per dynamic-transform
//! rebuild, and the build scratch every path records over is one slot per frame in
//! flight too (see `ScratchRing`). Each slot OWNS its resources for the accel's
//! lifetime and rebuilds them in place, growing them only on demand, so a steady
//! scene allocates nothing after warm-up. `RtAccelData`'s `live_*` fields are
//! plain handle copies of whichever slot last built -- Vulkan has no refcount, so
//! the ownership split has to be explicit. Nothing rotates between slots: a slot
//! handing its buffer to the next one would make every handle-keyed cache
//! (`SkinPipeline::wired`) miss on every visit. The ring rule they rest on is that
//! the `in_flight` fence wait retires a slot's previous writer before the next one
//! touches it -- sound for the skinned path because it runs on EVERY frame, and
//! for the static path because its cursor advances per rebuild rather than per
//! frame (a sparsely-moving scene traces one TLAS across many frames, so a
//! frame-keyed slot could be reused while a live trace still reads it). See
//! `SkinnedFrameRing` / `StaticFrameRing`. Only a topology refresh's orphaned draw
//! BLAS (and what a growing slot displaces) still go through the deferred-free
//! `Retired` pool. The bookkeeping over all of it -- which draws and clusters the
//! BLAS cover, the instance and geometry-table order, and when to update -- is the
//! shared `AccelBook`.
//!
//! Unlike DXR (which binds the TLAS as a root SRV by GPU virtual address each
//! frame), Vulkan binds the TLAS + geometry table through a descriptor set, so the
//! RT pass re-points the current frame's set at the live handles every frame; see
//! `post::rt_reflections::VkContext::rt_update_descriptors`. That re-point is
//! unconditional, so ring slot reuse needs nothing extra from it: the set for
//! frame `R` is written while frame `R` is the only frame that can bind it, the
//! same fence window the ring itself relies on.
//!
//! TODO(rt-pipeline-vulkan): this uses `VK_KHR_ray_query` (inline tracing in the
//! reflection fragment shader), the direct analog of the DXR 1.1 `RayQuery` path.
//! A future `VK_KHR_ray_tracing_pipeline` path (raygen/closest-hit/miss + a shader
//! binding table) would only be worth it if a feature needs recursive tracing or
//! per-material hit shaders, which screen-space reflections do not.

use ash::vk;
use concinnity_core::gfx::render_types::{DrawObject, InstancedCluster, SkinnedDrawObject};
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::fullscreen::align_up;
use concinnity_core::render::retire_pool::RetirePool;
use concinnity_core::render::rt_accel::{
    AccelBook, EmptyHead, FrameRing, HeadRefresh, InstanceBlas, RefreshMode, RtStep, RtUpdate,
    ScratchRing, SeedSet, StaticRing, empty_head,
};
use concinnity_core::render::rt_geom::{RtDynamicMode, pack_row_major_3x4};
use concinnity_core::render::rt_refit::{BlasUpdate, SkinnedRefit};
use concinnity_core::render::rt_topology::blas_vertex_count;
use concinnity_core::render::uniforms::SkinParams;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::context::VkGeometry;
use super::pipeline::{SHADER_ENTRY, spv_module};
use crate::vulkan::builtin_shaders::CompileProgram;
use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedPipeline, OwnedPipelineLayout, OwnedSetLayout, VkDevice,
};

// Byte stride of a `Vertex` in the shared vertex buffer (pos + normal + tangent
// + color + uv = 14 floats). The BLAS reads positions at this stride and the
// shader fetches attributes at this stride. The deformed (posed) skinned vertex
// buffer the skin kernel writes carries the same 56-byte layout.
const VERTEX_STRIDE: u64 = 56;

// One TLAS instance descriptor: explicit 3x4 transform, custom index (indexes
// the geometry table), full visibility mask, no SBT offset / flags, and the BLAS
// device address. Inline tracing ignores hit groups so the SBT fields are zero.
fn tlas_instance(
    model: [[f32; 4]; 4],
    custom_index: u32,
    blas_address: u64,
) -> vk::AccelerationStructureInstanceKHR {
    vk::AccelerationStructureInstanceKHR {
        transform: vk::TransformMatrixKHR {
            matrix: pack_row_major_3x4(model),
        },
        // instanceCustomIndex (low 24) + mask (high 8 = 0xFF).
        instance_custom_index_and_mask: vk::Packed24_8::new(custom_index & 0x00FF_FFFF, 0xFFu8),
        // instanceShaderBindingTableRecordOffset (24) + flags (8), both zero.
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, 0u8),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            device_handle: blas_address,
        },
    }
}

// The device address of the BLAS an instance the book laid out references:
// `fresh` holds the addresses of the BLAS a refresh in progress builds, indexed by
// head slot, and `skinned` this frame's skinned BLAS. A missing one reads as 0,
// which Vulkan defines as an inactive instance.
fn instance_blas_address(
    blas: InstanceBlas<'_, AccelBuffer>,
    fresh: &[u64],
    skinned: &[u64],
) -> u64 {
    match blas {
        InstanceBlas::Head { blas, .. } => blas.address,
        InstanceBlas::Fresh { index } => fresh.get(index).copied().unwrap_or(0),
        InstanceBlas::Skinned { n } => skinned.get(n).copied().unwrap_or(0),
    }
}

// The byte size a build scratch buffer needs to serve a build requiring
// `required` bytes: the requirement plus the offset alignment, so the aligned
// device address inside the buffer still leaves room for it.
fn scratch_capacity(required: u64, align: u64) -> u64 {
    required + align
}

// One frame's acceleration-structure build scratch (see `ScratchRing`). `addr`
// is the buffer's device address pre-aligned to
// `minAccelerationStructureScratchOffsetAlignment`. A replaced slot is dropped in
// place: the allocator withholds its range and handle for `frames_in_flight + 1`
// ticks, which outlasts both the builds this frame already recorded against it
// and any still in flight.
struct ScratchSlot {
    // Owns the memory the builds write; nothing reads it afterwards.
    _pooled: PooledBuffer,
    addr: u64,
}

// Allocate one scratch slot of `capacity` bytes, its address aligned to `align`.
fn alloc_scratch(
    alloc: &DeviceAllocator,
    device: &VkDevice,
    capacity: u64,
    align: u64,
) -> RenderResult<ScratchSlot> {
    let pooled = alloc.create_buffer(
        capacity,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let addr = align_up(buffer_address(device, pooled.buffer()), align);
    Ok(ScratchSlot {
        _pooled: pooled,
        addr,
    })
}

// This frame's build scratch, holding a build requiring `required` bytes.
fn ensure_scratch(
    ring: &mut ScratchRing<ScratchSlot>,
    ctx: RtDeviceCtx,
    frame_idx: usize,
    required: u64,
) -> RenderResult<u64> {
    let align = scratch_alignment(ctx.instance, ctx.pd);
    ring.ensure(frame_idx, scratch_capacity(required, align), |capacity| {
        alloc_scratch(ctx.alloc, ctx.device, capacity, align)
    })
    .map(|slot| slot.addr)
}

// A device-local buffer holding an acceleration structure plus its handle.
// `size` is the backing buffer's byte size, so a recycled `AccelBuffer` can be
// reused in place when a later build still fits.
struct AccelBuffer {
    accel: vk::AccelerationStructureKHR,
    // Backing buffer, held so the acceleration structure's memory outlives it.
    _pooled: PooledBuffer,
    size: u64,
    // The structure's device address, which TLAS instances reference it by.
    address: u64,
}

impl AccelBuffer {
    // The backing buffer retires through the allocator when the value drops;
    // only the acceleration-structure handle is destroyed by hand.
    fn destroy(&self, as_loader: &ash::khr::acceleration_structure::Device) {
        // SAFETY: the handle was created from this device and is destroyed exactly once, by its
        // owner, when no submission can still reference it: at teardown after the device idles,
        // from the retire pool once the frames-in-flight window since it was displaced has
        // passed, or by the ring slot that owns it, which is only rewritten after the
        // frame-begin fence retired every frame that traced it.
        unsafe {
            as_loader.destroy_acceleration_structure(self.accel, None);
        }
    }
}

// A host-visible buffer (the geometry table + the TLAS instance descriptors),
// filled once at creation and read by the GPU. A fresh one is allocated on each
// dynamic rebuild, so there is no need to keep it mapped past the initial copy.
struct HostBuffer {
    buffer: vk::Buffer,
    pooled: PooledBuffer,
    size: vk::DeviceSize,
}

// A plain device-local buffer (the deformed-vertex buffer the skin pass writes
// and the skinned BLAS + reflection trace read). Owns its memory + cached device
// address. `size` is the byte size, so a recycled buffer can be reused in place
// when a later rebuild still fits.
pub(super) struct DeviceBuffer {
    pub(super) buffer: vk::Buffer,
    // Owns the buffer; held so it outlives every reference.
    _pooled: PooledBuffer,
    address: u64,
    size: u64,
}

impl DeviceBuffer {
    // Name the buffer without borrowing it, so a rebuild can keep addressing it
    // while the ring slot that owns it is borrowed for its other members.
    fn handle(&self) -> DeviceBufferRef {
        DeviceBufferRef {
            buffer: self.buffer,
            address: self.address,
        }
    }
}

// A non-owning name for a `DeviceBuffer`: the handle plus its device address.
// Vulkan buffers are not refcounted, so the live BVH names its slot-owned
// deformed-vertex buffer through one of these rather than holding the buffer.
#[derive(Clone, Copy)]
struct DeviceBufferRef {
    buffer: vk::Buffer,
    address: u64,
}

// The compute pipeline that deforms skinned vertices for ray tracing
// (`rt_skin.hlsl`): set 0 = [src skinned verts, joint palette, deformed output,
// morph deltas, morph weights] (five storage buffers) + a 16-byte `SkinParams`
// push-constant block. Built in `build_rt_accel` (gated on RT) and held on
// `RtAccelData`; mirrors DirectX's `SkinPipeline` / Metal's `skin_pipeline`.
pub(super) struct SkinPipeline {
    set_layout: OwnedSetLayout,
    pipeline_layout: OwnedPipelineLayout,
    pipeline: OwnedPipeline,
    // Per-(frame, object) compute descriptor sets, sized + allocated lazily on
    // the first `rebuild_skinned` (the skinned object count is unknown at init,
    // before `upload_skinned` runs). Indexed `[frame_idx][object]`; rewritten in
    // place each rebuild at the current frame's slot (fence-gated, so safe, like
    // the RT resolve set's per-frame re-point). `upload_skinned_morphs` re-points
    // the morph bindings (3, 4) on the main fold's sets.
    descriptor_pool: OwnedDescriptorPool,
    pub(in crate::vulkan) sets: Vec<Vec<vk::DescriptorSet>>,
    // The [source verts, joint palette, deformed output] triple each set was last
    // pointed at, parallel to `sets`. The RT skin path re-points a set only when
    // its triple actually moved; see the compare site for what that does and does
    // not skip. Only the RT path (`rebuild_skinned`) maintains this; the main
    // fold's own `SkinPipeline` writes its sets once and leaves this empty.
    wired: Vec<Vec<[vk::Buffer; 3]>>,
    // A never-read storage buffer bound to the morph slots (3, 4) of every set
    // whose object carries no morph targets, so those bindings stay valid
    // without borrowing an unrelated buffer for the job.
    pub(in crate::vulkan) morph_dummy: vk::Buffer,
    _morph_dummy_pooled: PooledBuffer,
}

impl SkinPipeline {
    pub(super) fn destroy(&self, _device: &VkDevice) {}
}

// Whether a skin descriptor set can be left alone: it already names `want`, and
// nothing it names was (re)allocated this frame. The second clause is what makes
// the handle-value compare safe -- a destroy + create can hand back the same
// `VkBuffer` value for a different allocation, which would otherwise skip on a
// stale descriptor. Pure so the rule is unit-testable without a device.
fn skin_set_current(
    wired: &[vk::Buffer; 3],
    want: &[vk::Buffer; 3],
    storage_changed: bool,
) -> bool {
    !storage_changed && wired == want
}

// The per-frame skinned-geometry inputs `rebuild_skinned` needs to deform and
// add skinned objects to the BVH. Assembled by `rt_dynamic_update` from the
// context's skinned state.
pub(super) struct SkinnedRtInputs<'a> {
    // One entry per skinned mesh (only `visible`, real-triangle objects build).
    pub objects: &'a [SkinnedDrawObject],
    // The shared bind-pose skinned vertex buffer (`SkinnedVertex`, 80-byte
    // stride) the skin kernel reads, bound as the compute set's binding 0.
    pub vertex_buffer: vk::Buffer,
    // The shared skinned index buffer the skinned BLAS + reflection trace
    // address the deformed buffer with. Its device address is the BLAS index
    // input; the buffer handle is the trace's SSBO.
    pub index_buffer: vk::Buffer,
    // This frame's per-object joint palettes, parallel to `objects` (each is that
    // object's `MAX_JOINTS`-matrix upload buffer for the current frame), bound as
    // the compute set's binding 1. Borrowed from the main pass's per-frame
    // palettes rather than uploaded again here, so the RT skin dispatch costs no
    // extra buffer per object per frame.
    pub joint_buffers: &'a [PooledBuffer],
}

// Everything one skinned rebuild reads beyond the accel itself: the device
// context, the command buffer it records onto, the frame's draw list + skinned
// inputs, and which ring slot to build into. Bundled so the rebuild can also take
// the slot it writes without running past the argument limit.
struct SkinnedRebuild<'a> {
    ctx: RtDeviceCtx<'a>,
    cmd: vk::CommandBuffer,
    draw_objects: &'a [DrawObject],
    skinned: SkinnedRtInputs<'a>,
    frame_idx: usize,
    // Build every skinned BLAS from scratch rather than refitting.
    full_build: bool,
}

// A resource parked for deferred free: a draw BLAS a topology refresh orphaned,
// or whatever a growing ring slot displaced. Something a rebuild replaces cannot
// be freed in place -- a prior frame's in-flight trace may still reach it, and a
// live handle may still name it if a later step of the same rebuild fails (the
// live BVH is the very slot being rebuilt when the ring is one deep) -- so it is
// freed only once the frames-in-flight window has elapsed, by which point the
// fence guarantees neither is true. Growth is rare, so the steady state never
// parks anything.
enum Retired {
    // A structure whose handle has to be destroyed by hand.
    Accel(AccelBuffer),
    // Buffers that free on `Drop`, parked only so that drop waits out the window.
    Device { _buffer: DeviceBuffer },
    Host { _buffer: HostBuffer },
}

impl Retired {
    fn destroy(&self, as_loader: &ash::khr::acceleration_structure::Device) {
        if let Retired::Accel(b) = self {
            b.destroy(as_loader);
        }
    }
}

// The deferred-free pool as a growing ring slot sees it: somewhere to hand the
// resource it displaced, plus the update tick it was displaced on. Passed by
// value so the borrow of the pool lasts only the one `ensure_*` call that needs
// it.
struct RetireSink<'a> {
    pool: &'a mut RetirePool<Retired>,
    now: u64,
}

impl<'a> RetireSink<'a> {
    fn new(pool: &'a mut RetirePool<Retired>, now: u64) -> Self {
        Self { pool, now }
    }

    fn accel(self, resource: AccelBuffer) {
        self.pool.push(self.now, Retired::Accel(resource));
    }

    fn device(self, resource: DeviceBuffer) {
        self.pool
            .push(self.now, Retired::Device { _buffer: resource });
    }

    fn host(self, resource: HostBuffer) {
        self.pool
            .push(self.now, Retired::Host { _buffer: resource });
    }
}

// One frame slot's skinned-rebuild resources, owned by the slot for the accel's
// lifetime. The skinned rebuild for frame `R` builds into slot `R % depth` in
// place and publishes the slot's handles as the live BVH (`RtAccelData`'s
// `live_*` fields); it never hands a resource to another slot. Reuse is
// hazard-free because the in-flight fence retires slot `s`'s previous writer
// (frame `R - depth`) before the next one records, the same window the retire
// pool's deferred free rested on; the difference is the resources are reused
// rather than freed + reallocated, so the steady state allocates nothing.
//
// Slot ownership (rather than swapping the live set with the slot) is what keeps
// a slot's handles stable across cycles, which is what lets `SkinPipeline::wired`
// skip the per-object descriptor re-point: a swap would rotate `depth + 1`
// buffers through each slot, so no two consecutive visits would see the same one.
// Each resource self-describes its byte size, so a slot is only ever grown when a
// later build outgrows it.
#[derive(Default)]
struct SkinnedFrameRing {
    deformed: Option<DeviceBuffer>,
    // One BLAS per skinned object.
    blas: Vec<AccelBuffer>,
    // Whether this slot's BLAS hold a tree the next update can refit rather than
    // rebuild, and the geometry that tree was built over. Per slot because the
    // slots are written on different frames, so their rebuild cadences stagger.
    refit: SkinnedRefit,
    tlas: Option<AccelBuffer>,
    instance: Option<HostBuffer>,
    geom: Option<HostBuffer>,
}

impl SkinnedFrameRing {
    // The buffers retire through the allocator when the slot drops; only the
    // acceleration-structure handles are destroyed by hand.
    fn destroy(&mut self, as_loader: &ash::khr::acceleration_structure::Device) {
        for b in &self.blas {
            b.destroy(as_loader);
        }
        if let Some(t) = &self.tlas {
            t.destroy(as_loader);
        }
    }
}

// One ring slot of the per-rebuild static-transform buffers (the TLAS + its
// instance descriptors + the geometry table), owned by the slot for the accel's
// lifetime like `SkinnedFrameRing`. The dynamic-transform rebuild advances
// the ring cursor to the next slot each rebuild, rebuilds that slot's buffers in
// place (re-map + copy / build-over) and publishes its handles as the live BVH,
// growing one only when a later rebuild outgrows it (the static instance count is
// fixed, so the steady state allocates nothing). Reuse is hazard-free: the cursor
// revisits a slot only after a full ring cycle, and a cycle spans at least
// `frames_in_flight` frames, by which point the fence has retired every trace
// that read it. The initial `build_rt_accel` structures live in slot 0.
#[derive(Default)]
struct StaticFrameRing {
    tlas: Option<AccelBuffer>,
    instance: Option<HostBuffer>,
    geom: Option<HostBuffer>,
}

impl StaticFrameRing {
    // The host buffers retire through the allocator when the slot drops.
    fn destroy(&self, as_loader: &ash::khr::acceleration_structure::Device) {
        if let Some(t) = &self.tlas {
            t.destroy(as_loader);
        }
    }
}

// The skinned rebuild's per-frame lists the book does not keep, held on the
// accel so their capacity is reused from frame to frame.
#[derive(Default)]
struct SkinnedScratch {
    // This frame's skinned geometry parameters, parallel to the book's selected
    // skinned objects. Held across the sizing and recording loops, which both
    // rebuild the temporary `vk::*` geometry structs from it.
    params: Vec<BlasParams>,
    // Device addresses of this frame's skinned BLAS, in the same order.
    blas_addresses: Vec<u64>,
}

// The Vulkan ray-query acceleration structures + geometry table for hardware ray
// tracing. Held on the context behind an `Option`; present only when RT
// reflections are enabled, the GPU exposes the ray-query extensions, and the
// scene has resident geometry.
pub(super) struct RtAccelData {
    as_loader: ash::khr::acceleration_structure::Device,

    // The persistent static + cluster BLAS (the book's head, built once and
    // never rebuilt: a rigid transform leaves object-space geometry unchanged),
    // the order the TLAS instances and geometry table follow, and the update
    // policy over them. The per-frame skinned BLAS are owned by their
    // `skinned_ring` slot, not by the book.
    book: AccelBook<AccelBuffer, vk::AccelerationStructureInstanceKHR>,
    // The top-level (instance) acceleration structure the trace reads, owned by
    // the ring slot that last rebuilt it (`static_ring` on the static path,
    // `skinned_ring` on the skinned path).
    live_tlas: vk::AccelerationStructureKHR,
    // `[RtGeomEntry; instance_count]` (host-visible), bound as a storage buffer;
    // indexed by the trace's `instanceCustomIndex`. Owned by the same slot as
    // `live_tlas`; `live_geom_size` is its byte size.
    live_geom: vk::Buffer,
    live_geom_size: vk::DeviceSize,
    // Build scratch, one slot per frame in flight (see `ScratchRing`). Sized at
    // init for the largest of every BLAS build and the TLAS build; a rebuild
    // whose builds need more replaces the slot it records over.
    scratch: ScratchRing<ScratchSlot>,
    // Size the TLAS prebuild reported; the static rebuild recycles the ring slot's
    // TLAS at this size (the static instance count is fixed).
    tlas_size: u64,
    // Scratch that TLAS build needs, kept with it so the static rebuild sizes
    // the ring slot it records over: a topology refresh grows only its own
    // frame's slot.
    tlas_scratch: u64,
    instance_count: u32,
    // Frames-in-flight depth; a retired structure is freed this many updates
    // after the one that displaced it (by then its frame's fence has signaled,
    // so no in-flight trace can still read it).
    frames_in_flight: u64,

    // Deferred-free pool for the draw BLAS a topology refresh orphans and what a
    // growing ring slot displaces, timed against the book's update clock. Every
    // per-frame resource is owned by a ring slot, so a steady scene parks nothing
    // here; a refresh does, and the `Rebuild` diagnostic retires every draw BLAS
    // on every frame.
    retire: RetirePool<Retired>,

    // Per-rebuild static-transform buffers (see `StaticFrameRing`), owned by their
    // slot and rebuilt in place by the static `rebuild_tlas` path. The cursor
    // advances one slot per rebuild; a slot is revisited only after a full ring
    // cycle, so its prior trace has retired. Slot 0 holds the initial build's
    // structures. The skinned path uses `skinned_ring` instead.
    static_ring: StaticRing<StaticFrameRing>,

    // Per-frame skinned-rebuild resources, one slot per frame in flight, owned by
    // their slot and rebuilt in place (see `SkinnedFrameRing`). Indexed by
    // `frame_idx`.
    skinned_ring: FrameRing<SkinnedFrameRing>,

    // Skinned geometry.
    // The compute-skinning pipeline (`rt_skin`). `Some` only when the GLSL
    // compile + pipeline creation succeeded; without it skinned geometry is
    // absent from the BVH (the RT pass still runs for static geometry).
    skin: Option<SkinPipeline>,
    // The deformed (posed) skinned vertex buffer the skin pass writes and the
    // skinned BLAS + reflection trace read, owned by the `skinned_ring` slot that
    // last rebuilt it. Re-pointed onto the RT descriptor set each frame, like the
    // TLAS.
    live_deformed: vk::Buffer,
    // A 1-element deformed-vertex buffer, named by `live_deformed` until the first
    // skinned rebuild so the trace's skinned-verts SSBO always binds a valid
    // resource. Never read again; held so it outlives that binding.
    _deformed_dummy: DeviceBuffer,
    // The shared skinned index buffer (the BLAS index input + the trace's
    // SSBO). A dummy `vk::Buffer::null()`-backed handle when there is no skinned
    // geometry; the post pass binds a dummy SSBO in that case.
    skinned_indices: vk::Buffer,
    frames_in_flight_usize: usize,

    // Persistent CPU scratch for the skinned rebuild.
    skinned_scratch: SkinnedScratch,
}

// SAFETY: Raw pointers in `HostBuffer` are host-mapped and only touched on the render
// thread; the acceleration-structure loader holds plain fn pointers. The whole
// struct lives inside `VkContext`, which is already `unsafe impl Send`.
unsafe impl Send for RtAccelData {}

impl RtAccelData {
    // The live TLAS handle (bound through the RT pass's descriptor set).
    pub(super) fn tlas(&self) -> vk::AccelerationStructureKHR {
        self.live_tlas
    }

    // The live geometry-table buffer + its byte range (bound as a storage buffer).
    pub(super) fn geom_table(&self) -> (vk::Buffer, vk::DeviceSize) {
        (self.live_geom, self.live_geom_size)
    }

    // The live deformed (posed) skinned vertex buffer (bound as the RT pass's
    // skinned-verts SSBO). It moves between ring slots as the frame advances, so
    // the RT pass re-points its descriptor at this every frame, like the TLAS. A
    // 1-element dummy until the first skinned rebuild, so the binding is always
    // valid.
    pub(super) fn deformed_verts(&self) -> vk::Buffer {
        self.live_deformed
    }

    // The shared skinned index buffer (bound as the RT pass's skinned-index
    // SSBO). `vk::Buffer::null()` when there is no skinned geometry; the post
    // pass substitutes a dummy SSBO so the binding is always live.
    pub(super) fn skinned_indices(&self) -> vk::Buffer {
        self.skinned_indices
    }
}

// Per-build geometry parameters captured once, used both for sizing and for the
// recorded build (so the temporary `vk::*` builder structs can be reconstructed
// cheaply inside the command-buffer recording closure).
struct BlasParams {
    vertex_address: u64,
    max_vertex: u32,
    index_byte_offset: u32,
    primitive_count: u32,
}

// The shared static vertex / index buffers every draw-object and cluster BLAS is
// built over, read from the live `VkGeometry` at each build: chunk streaming and
// geometry rebuilds replace both buffers, so an address or vertex count captured
// earlier can name a destroyed buffer or bound streamed geometry short.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct SharedGeometry {
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    // Vertices the vertex buffer holds, streaming headroom included.
    vertex_count: u64,
}

impl SharedGeometry {
    pub(in crate::vulkan) fn of(geometry: &VkGeometry) -> Self {
        Self {
            vertex_buffer: geometry.vertex_buffer.buffer(),
            index_buffer: geometry.index_buffer.buffer(),
            vertex_count: geometry.vertex_buffer_bytes / VERTEX_STRIDE,
        }
    }

    fn addresses(&self, device: &VkDevice) -> SharedAddresses {
        SharedAddresses {
            vertex: buffer_address(device, self.vertex_buffer),
            index: buffer_address(device, self.index_buffer),
            vertex_count: self.vertex_count,
        }
    }
}

// `SharedGeometry` resolved to device addresses for one build.
#[derive(Clone, Copy)]
struct SharedAddresses {
    vertex: u64,
    index: u64,
    vertex_count: u64,
}

impl SharedAddresses {
    // The BLAS parameters for one draw object's slice. Its indices are offset by
    // `base_vertex`, which folds into the vertex address.
    fn draw_params(&self, obj: &DrawObject) -> BlasParams {
        let base_vertex = u64::try_from(obj.base_vertex).unwrap_or(0);
        BlasParams {
            vertex_address: self.vertex + base_vertex * VERTEX_STRIDE,
            max_vertex: blas_vertex_count(obj.base_vertex, self.vertex_count).saturating_sub(1),
            index_byte_offset: obj.index_offset as u32 * 4,
            primitive_count: (obj.index_count / 3) as u32,
        }
    }

    // The BLAS parameters for one instanced cluster (absolute indices).
    fn cluster_params(&self, cluster: &InstancedCluster) -> BlasParams {
        BlasParams {
            vertex_address: self.vertex,
            max_vertex: blas_vertex_count(0, self.vertex_count).saturating_sub(1),
            index_byte_offset: cluster.index_offset as u32 * 4,
            primitive_count: (cluster.index_count / 3) as u32,
        }
    }
}

fn blas_geometry(p: &BlasParams, index_address: u64) -> vk::AccelerationStructureGeometryKHR<'_> {
    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: p.vertex_address,
        })
        .vertex_stride(VERTEX_STRIDE)
        .max_vertex(p.max_vertex)
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR {
            device_address: index_address,
        });
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
        .flags(vk::GeometryFlagsKHR::OPAQUE)
}

// Same as `blas_geometry` but over the skinned index buffer + the deformed
// (posed) skinned vertex buffer. The skinned BLAS bakes absolute indices into
// the deformed buffer (base vertex folded to 0), so `vertex_address` is the
// deformed buffer's base address and `index_address` is the index buffer offset
// for this object. Same 56-byte vertex stride as the static path.
fn skinned_blas_geometry(
    p: &BlasParams,
    index_address: u64,
) -> vk::AccelerationStructureGeometryKHR<'_> {
    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: p.vertex_address,
        })
        .vertex_stride(VERTEX_STRIDE)
        .max_vertex(p.max_vertex)
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR {
            device_address: index_address,
        });
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
        .flags(vk::GeometryFlagsKHR::OPAQUE)
}

// The BOTTOM_LEVEL build info for one skinned geometry. Always carries
// `ALLOW_UPDATE`, which is what makes a later in-place update legal (Vulkan
// requires it on the build that produced the source structure, and it also makes
// the size query report an `update_scratch_size`); `Refit` additionally selects
// `MODE_UPDATE`. The caller fills in the destination, the source (the destination
// itself, which the spec allows and defines as an in-place update) and the
// scratch address. Pass `Build` when sizing: a size query only needs the
// allocation flags.
fn skinned_blas_build_info<'a>(
    geo: &'a vk::AccelerationStructureGeometryKHR<'a>,
    update: BlasUpdate,
) -> vk::AccelerationStructureBuildGeometryInfoKHR<'a> {
    let mode = match update {
        BlasUpdate::Build => vk::BuildAccelerationStructureModeKHR::BUILD,
        BlasUpdate::Refit => vk::BuildAccelerationStructureModeKHR::UPDATE,
    };
    vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
        )
        .mode(mode)
        .geometries(std::slice::from_ref(geo))
}

fn tlas_geometry(instance_address: u64) -> vk::AccelerationStructureGeometryKHR<'static> {
    let instances = vk::AccelerationStructureGeometryInstancesDataKHR::default()
        .array_of_pointers(false)
        .data(vk::DeviceOrHostAddressConstKHR {
            device_address: instance_address,
        });
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { instances })
        .flags(vk::GeometryFlagsKHR::OPAQUE)
}

// A from-scratch build of a `ty` structure over the one geometry `geo`, preferring
// trace speed. The destination and scratch are set by the caller.
fn build_info<'a>(
    ty: vk::AccelerationStructureTypeKHR,
    geo: &'a vk::AccelerationStructureGeometryKHR<'a>,
) -> vk::AccelerationStructureBuildGeometryInfoKHR<'a> {
    vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(ty)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(std::slice::from_ref(geo))
}

// The structure and scratch sizes `info` needs over `primitive_count` primitives.
fn build_sizes(
    as_loader: &ash::khr::acceleration_structure::Device,
    info: &vk::AccelerationStructureBuildGeometryInfoKHR<'_>,
    primitive_count: u32,
) -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    // SAFETY: a property query on a live handle; it only reads.
    unsafe {
        as_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            info,
            &[primitive_count],
            &mut sizes,
        );
    }
    sizes
}

// The sizes a from-scratch BLAS build over `geo` needs.
fn blas_build_sizes(
    as_loader: &ash::khr::acceleration_structure::Device,
    geo: &vk::AccelerationStructureGeometryKHR<'_>,
    primitive_count: u32,
) -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
    let info = build_info(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL, geo);
    build_sizes(as_loader, &info, primitive_count)
}

// The sizes a TLAS build over `instance_count` instances needs.
fn tlas_build_sizes(
    as_loader: &ash::khr::acceleration_structure::Device,
    geo: &vk::AccelerationStructureGeometryKHR<'_>,
    instance_count: u32,
) -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
    let info = build_info(vk::AccelerationStructureTypeKHR::TOP_LEVEL, geo);
    build_sizes(as_loader, &info, instance_count)
}

// The range of one build: `primitive_count` primitives starting `primitive_offset`
// bytes into the geometry's index (or instance) data.
fn build_range(
    primitive_count: u32,
    primitive_offset: u32,
) -> vk::AccelerationStructureBuildRangeInfoKHR {
    vk::AccelerationStructureBuildRangeInfoKHR::default()
        .primitive_count(primitive_count)
        .primitive_offset(primitive_offset)
        .first_vertex(0)
        .transform_offset(0)
}

// Record one acceleration-structure build onto `cmd`.
fn record_build(
    as_loader: &ash::khr::acceleration_structure::Device,
    cmd: vk::CommandBuffer,
    info: vk::AccelerationStructureBuildGeometryInfoKHR<'_>,
    range: vk::AccelerationStructureBuildRangeInfoKHR,
) {
    // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
    // these commands name is live for the call.
    unsafe {
        as_loader.cmd_build_acceleration_structures(
            cmd,
            std::slice::from_ref(&info),
            &[std::slice::from_ref(&range)],
        );
    }
}

// Device address of a buffer (core in Vulkan 1.2; the device enables
// `bufferDeviceAddress` for the RT path).
fn buffer_address(device: &VkDevice, buffer: vk::Buffer) -> u64 {
    // SAFETY: `buffer` was created from this device with SHADER_DEVICE_ADDRESS usage and the info
    // struct borrows it for the call; the query only reads.
    unsafe {
        device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
    }
}

// Allocate a fresh acceleration-structure backing buffer + create the AS handle.
fn create_accel(
    alloc: &DeviceAllocator,
    as_loader: &ash::khr::acceleration_structure::Device,
    size: u64,
    ty: vk::AccelerationStructureTypeKHR,
) -> RenderResult<AccelBuffer> {
    let size = size.max(256);
    let pooled = alloc.create_buffer(
        size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let buffer = pooled.buffer();
    let info = vk::AccelerationStructureCreateInfoKHR::default()
        .buffer(buffer)
        .offset(0)
        .size(size)
        .ty(ty);
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    let accel = unsafe { as_loader.create_acceleration_structure(&info, None) }
        .map_err(|e| super::error::map_vk_result(e, "create acceleration structure"))?;
    // SAFETY: the acceleration structure was just created from this device and the info struct
    // borrows its handle for the call; the query only reads.
    let address = unsafe {
        as_loader.get_acceleration_structure_device_address(
            &vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(accel),
        )
    };
    Ok(AccelBuffer {
        accel,
        _pooled: pooled,
        size,
        address,
    })
}

// The bytes a host buffer holding `data` needs: at least one whole element, so
// a shader binding it as an array (the geometry table of a zero-instance TLAS)
// sees a full entry, and never under 16.
fn host_buffer_size<T>(data: &[T]) -> vk::DeviceSize {
    (std::mem::size_of_val(data).max(std::mem::size_of::<T>()) as vk::DeviceSize).max(16)
}

// Allocate a host-visible, persistently-mapped buffer of `size` bytes with the
// given usage, copy `data` into it, and return the mapped handle.
fn create_host_buffer<T: Copy>(
    alloc: &DeviceAllocator,
    data: &[T],
    usage: vk::BufferUsageFlags,
    _label: &str,
) -> RenderResult<HostBuffer> {
    let size = host_buffer_size(data);
    let pooled = alloc.create_buffer(
        size,
        usage,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let buffer = pooled.buffer();
    pooled.write_slice(0, data);
    Ok(HostBuffer {
        buffer,
        pooled,
        size,
    })
}

// Write `data` into `slot`'s host buffer, reusing it in place when it can hold
// the data and replacing it with a larger one when it cannot. The ring's host
// buffers are rewritten every frame, so this keeps them allocation-free in the
// steady state while still growing on demand. `slot`'s buffer must have been
// created with `usage` (the ring only ever stores a buffer of the matching usage
// in each slot).
//
// The replacement is allocated before the old buffer drops, so a failure leaves
// `slot` -- and any live handle naming it -- untouched.
fn write_or_recreate_host<T: Copy>(
    slot: &mut Option<HostBuffer>,
    alloc: &DeviceAllocator,
    data: &[T],
    usage: vk::BufferUsageFlags,
    label: &str,
    retire: RetireSink,
) -> RenderResult<()> {
    let needed = host_buffer_size(data);
    if let Some(buf) = slot.as_ref()
        && buf.size >= needed
    {
        buf.pooled.write_slice(0, data);
        return Ok(());
    }
    let fresh = create_host_buffer(alloc, data, usage, label)?;
    if let Some(old) = slot.replace(fresh) {
        retire.host(old);
    }
    Ok(())
}

// The buffer a ring slot's host-buffer write just ensured.
fn host_buffer(slot: &Option<HostBuffer>) -> RenderResult<vk::Buffer> {
    slot.as_ref()
        .map(|b| b.buffer)
        .ok_or_else(missing_slot_buffer)
}

// The structure a ring slot's `ensure_accel` just ensured.
fn live_accel(slot: &Option<AccelBuffer>) -> RenderResult<vk::AccelerationStructureKHR> {
    slot.as_ref()
        .map(|b| b.accel)
        .ok_or_else(missing_slot_buffer)
}

fn missing_slot_buffer() -> RenderError {
    RenderError::Other("RT ring slot is missing a buffer it was just sized for".into())
}

// Ensure `slot` holds an acceleration structure of at least `size` bytes, keeping
// the one it already holds when that still fits. Returns whether the structure
// was (re)created, which leaves no tree for a later update to continue.
//
// The replacement is created before the old one is displaced, so a failure leaves
// `slot` untouched, and the displaced structure goes to the deferred-free pool
// rather than being destroyed here -- see `Retired` for why freeing in place is
// not safe even though the create succeeded.
fn ensure_accel(
    slot: &mut Option<AccelBuffer>,
    alloc: &DeviceAllocator,
    as_loader: &ash::khr::acceleration_structure::Device,
    size: u64,
    ty: vk::AccelerationStructureTypeKHR,
    retire: RetireSink,
) -> RenderResult<bool> {
    if slot.as_ref().is_some_and(|b| b.size >= size) {
        return Ok(false);
    }
    let fresh = create_accel(alloc, as_loader, size, ty)?;
    if let Some(old) = slot.replace(fresh) {
        retire.accel(old);
    }
    Ok(true)
}

// Ensure `slot` holds a device-local buffer of at least `size` bytes, keeping the
// one it already holds when that still fits. Returns whether the buffer was
// (re)created, which both invalidates the descriptors pointing at it and leaves
// no tree for a later update to continue. Same create-then-retire ordering as
// `ensure_accel`.
fn ensure_device_buffer(
    slot: &mut Option<DeviceBuffer>,
    alloc: &DeviceAllocator,
    device: &VkDevice,
    size: u64,
    retire: RetireSink,
) -> RenderResult<bool> {
    if slot.as_ref().is_some_and(|b| b.size >= size) {
        return Ok(false);
    }
    let fresh = create_device_buffer(alloc, device, size)?;
    if let Some(old) = slot.replace(fresh) {
        retire.device(old);
    }
    Ok(true)
}

// Allocate a fresh device-local buffer usable as the deformed-vertex buffer: a
// storage buffer (skin compute writes it, the trace reads it), a BLAS vertex
// input, and device-addressable (the BLAS reads it by address). Caches the
// device address.
fn create_device_buffer(
    alloc: &DeviceAllocator,
    device: &VkDevice,
    size: u64,
) -> RenderResult<DeviceBuffer> {
    let size = size.max(VERTEX_STRIDE);
    let pooled = alloc.create_buffer(
        size,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let buffer = pooled.buffer();
    let address = buffer_address(device, buffer);
    Ok(DeviceBuffer {
        buffer,
        _pooled: pooled,
        address,
        size,
    })
}

// Build the `rt_skin` compute pipeline: a 3-storage-buffer descriptor set layout
// (set 0: src skinned verts, joint palette, deformed output) + a 16-byte
// `SkinParams` push constant. Returns `Err` when the compiler is unavailable or the
// kernel fails to compile; the caller then leaves the skin pipeline absent and
// skinned geometry is omitted from the BVH (the RT pass still runs for static
// geometry). Per-(frame, object) descriptor sets are allocated lazily on the
// first `rebuild_skinned`, when the skinned object count is known.
pub(super) fn build_skin_pipeline(
    alloc: &DeviceAllocator,
    device: &VkDevice,
    hot_reload: bool,
) -> RenderResult<SkinPipeline> {
    let spv = super::builtin_shaders::RT_SKIN.compile(hot_reload)?;
    let module = spv_module(device, &spv)?;

    // Five storage buffers: src verts (0), joint palette (1), deformed output
    // (2), morph deltas (3), morph weights (4).
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..5u32)
        .map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let set_layout = device
        .create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
        )
        .map_err(|e| super::error::map_vk_result(e, "rt skin descriptor set layout"))?;

    let pc = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<SkinParams>() as u32);
    let set_layouts = [set_layout.handle()];
    let pipeline_layout = device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(std::slice::from_ref(&pc)),
        )
        .map_err(|e| super::error::map_vk_result(e, "rt skin pipeline layout"))?;

    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module.handle())
        .name(SHADER_ENTRY);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout.handle());
    let pipeline = crate::vulkan::pipeline_cache::create_compute_pipeline(device, &info);
    let pipeline =
        pipeline.map_err(|e| super::error::map_vk_result(e, "create rt skin pipeline"))?;

    // Sized to one `MorphEntry` so even a stray read of slot 0 stays in
    // bounds; `target_count == 0` keeps it unread.
    let morph_dummy_pooled = alloc.create_buffer(
        28,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    Ok(SkinPipeline {
        set_layout,
        pipeline_layout,
        pipeline,
        descriptor_pool: OwnedDescriptorPool::null(),
        sets: Vec::new(),
        wired: Vec::new(),
        morph_dummy: morph_dummy_pooled.buffer(),
        _morph_dummy_pooled: morph_dummy_pooled,
    })
}

// A global acceleration-structure-build memory barrier: orders one build's writes
// before whatever `dst_stages` names next reads or writes them -- the scratch
// slot the next build rewrites and the BLAS the TLAS build reads. Mirrors the
// DXR UAV barrier between builds.
fn build_barrier(device: &VkDevice, cmd: vk::CommandBuffer, dst_stages: vk::PipelineStageFlags) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
        .dst_access_mask(
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
        );
    // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice these
    // commands name is live for the call.
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            dst_stages,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&barrier),
            &[],
            &[],
        );
    }
}

// The dst stages between two builds: the scratch slot they share is rewritten by
// the next build, and a TLAS build reads the BLAS just built.
const BUILD_TO_BUILD: vk::PipelineStageFlags =
    vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR;

// The dst stages that close a recorded build sequence: this frame's trace reads
// the structures in the fragment shader, and any sequence recorded after this one
// on the same command buffer writes the same scratch slot (a topology refresh is
// followed by the skinned rebuild on the same frame).
const BUILD_TO_TRACE: vk::PipelineStageFlags = vk::PipelineStageFlags::from_raw(
    vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR.as_raw()
        | vk::PipelineStageFlags::FRAGMENT_SHADER.as_raw(),
);

// Query the device's minimum scratch-offset alignment for AS builds.
fn scratch_alignment(instance: &ash::Instance, pd: vk::PhysicalDevice) -> u64 {
    let mut as_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut as_props);
    // SAFETY: a property query on a live handle; it only reads.
    unsafe { instance.get_physical_device_properties2(pd, &mut props2) };
    (as_props.min_acceleration_structure_scratch_offset_alignment as u64).max(1)
}

// The Vulkan device handles every acceleration-structure build reads from. `pd`
// is Copy; `instance` / `device` are borrowed. Shared by the one-shot
// `build_rt_accel` and the per-frame rebuild methods so they thread one context
// rather than three loose handles.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct RtDeviceCtx<'a> {
    pub(in crate::vulkan) alloc: &'a DeviceAllocator,
    pub(in crate::vulkan) instance: &'a ash::Instance,
    pub(in crate::vulkan) device: &'a VkDevice,
    pub(in crate::vulkan) pd: vk::PhysicalDevice,
}

// The scene geometry + bindless-pool sizing `build_rt_accel` bakes into the
// initial BVH: the shared static vertex / index buffers, the participating draw
// objects + instanced clusters, and the pool counts the geometry-table indices
// offset against. Borrowed for the duration of the build.
pub(in crate::vulkan) struct RtSceneGeometry<'a> {
    // The shared vertex / index buffers the draw and cluster BLAS read.
    pub(in crate::vulkan) shared: SharedGeometry,
    // Every draw object; the resident, real-triangle ones participate.
    pub(in crate::vulkan) draw_objects: &'a [DrawObject],
    // Every instanced cluster; the non-empty, real-triangle ones participate.
    pub(in crate::vulkan) clusters: &'a [InstancedCluster],
    // The shared pool's real-texture count (resolves each geometry's albedo /
    // normal indices; the flat-normal fallback sits at this index).
    pub(in crate::vulkan) albedo_count: usize,
    // Leave see-through glass meshes out of the BVH (see `participates_in_bvh`).
    pub(in crate::vulkan) exclude_seethrough: bool,
}

// Build the BLAS / TLAS / geometry table for the scene on a one-shot command
// buffer (submitted and fence-waited so the structures are ready before the
// first frame traces them). Returns `Ok(None)` when there is no resident
// triangle geometry to trace: the RT pass then skips its trace until a
// topology change seeds one.
pub(super) fn build_rt_accel(
    ctx: RtDeviceCtx,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    geometry: RtSceneGeometry,
    frames_in_flight: usize,
    hot_reload: bool,
) -> RenderResult<Option<RtAccelData>> {
    let RtDeviceCtx {
        alloc,
        instance,
        device,
        pd,
    } = ctx;
    let RtSceneGeometry {
        shared,
        draw_objects,
        clusters,
        albedo_count,
        exclude_seethrough,
    } = geometry;
    let as_loader = ash::khr::acceleration_structure::Device::new(instance, device);

    // Participating static objects + clusters (real triangles, resident, and not
    // rerouted to the see-through transparent path).
    let seed = SeedSet::new(draw_objects, clusters, exclude_seethrough);
    if seed.is_empty() {
        return Ok(None);
    }

    let shared = shared.addresses(device);
    let ibuf_addr = shared.index;

    // One BLAS-build params entry per participating object first, then clusters.
    // Each object folds its base_vertex into the vertex device address + uses its
    // mesh-relative indices (the shader adds base_vertex back via the geom table),
    // mirroring the DirectX vertex-address fold.
    let params: Vec<BlasParams> = seed
        .objects
        .iter()
        .map(|&i| shared.draw_params(&draw_objects[i]))
        .chain(seed.clusters.iter().map(|c| shared.cluster_params(c)))
        .collect();

    // Size + allocate each BLAS; track the largest scratch requirement.
    let mut blas: Vec<AccelBuffer> = Vec::with_capacity(params.len());
    let mut max_scratch: u64 = 0;
    for p in &params {
        let geo = blas_geometry(p, ibuf_addr);
        let sizes = blas_build_sizes(&as_loader, &geo, p.primitive_count);
        blas.push(create_accel(
            alloc,
            &as_loader,
            sizes.acceleration_structure_size,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        )?);
        max_scratch = max_scratch.max(sizes.build_scratch_size);
    }

    // Instance descriptors + geometry table, in the book's instance order.
    let mut book = AccelBook::new(&seed, blas, draw_objects, albedo_count as u32)?;
    book.fill_instances(draw_objects, None, |model, id, blas| {
        tlas_instance(model, id, instance_blas_address(blas, &[], &[]))
    });
    let instance_count = book.instances().len() as u32;

    let instance_buffer = create_host_buffer(
        alloc,
        book.instances(),
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        "RT instance buffer",
    )?;
    let geom_table = create_host_buffer(
        alloc,
        book.geom_table(),
        vk::BufferUsageFlags::STORAGE_BUFFER,
        "RT geometry table",
    )?;

    // Size + allocate the TLAS + the scratch ring (>= the largest BLAS/TLAS).
    let tlas_geo = tlas_geometry(buffer_address(device, instance_buffer.buffer));
    let tlas_sizes = tlas_build_sizes(&as_loader, &tlas_geo, instance_count);
    max_scratch = max_scratch.max(tlas_sizes.build_scratch_size);
    let tlas = create_accel(
        alloc,
        &as_loader,
        tlas_sizes.acceleration_structure_size,
        vk::AccelerationStructureTypeKHR::TOP_LEVEL,
    )?;

    // One scratch slot per frame in flight, each sized to the largest build. This
    // one-shot build is fence-waited before the first frame records, so it can
    // take slot 0.
    let align = scratch_alignment(instance, pd);
    let scratch = ScratchRing::filled(
        frames_in_flight,
        scratch_capacity(max_scratch, align),
        |capacity| alloc_scratch(alloc, device, capacity, align),
    )?;
    let scratch_addr = scratch.get(0).map_or(0, |slot| slot.addr);

    // Record every BLAS build (build-barrier-serialized over the one scratch slot
    // they share), then the TLAS build, on a one-shot command buffer; fence-wait so the BVH is
    // ready before the first trace.
    super::texture::one_shot_submit(device, command_pool, queue, |cmd| {
        for (dst, p) in book.head().iter().zip(&params) {
            let geo = blas_geometry(p, ibuf_addr);
            record_build(
                &as_loader,
                cmd,
                build_info(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL, &geo)
                    .dst_acceleration_structure(dst.accel)
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: scratch_addr,
                    }),
                build_range(p.primitive_count, p.index_byte_offset),
            );
            build_barrier(device, cmd, BUILD_TO_BUILD);
        }
        let tlas_geo = tlas_geometry(buffer_address(device, instance_buffer.buffer));
        record_build(
            &as_loader,
            cmd,
            build_info(vk::AccelerationStructureTypeKHR::TOP_LEVEL, &tlas_geo)
                .dst_acceleration_structure(tlas.accel)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch_addr,
                }),
            build_range(instance_count, 0),
        );
    })?;

    // Skinned geometry is seeded on the first dynamic frame (like DirectX /
    // Metal), so the init build is static-only. Allocate a 1-element dummy
    // deformed-vertex buffer so the trace's skinned-verts SSBO always binds a
    // valid resource; the first `rebuild_skinned` points it at a ring slot's.
    let deformed_dummy = create_device_buffer(alloc, device, VERTEX_STRIDE)?;

    // The structures just built are the live BVH; home them in static ring slot 0,
    // which owns them from here on. The first dynamic rebuild advances past it, so
    // slot 0 is only reused a full ring cycle later -- the same window every other
    // slot rests on.
    let live_tlas = tlas.accel;
    let live_geom = geom_table.buffer;
    let live_geom_size = geom_table.size;
    let static_ring = StaticRing::new(
        frames_in_flight,
        StaticFrameRing {
            tlas: Some(tlas),
            instance: Some(instance_buffer),
            geom: Some(geom_table),
        },
    );

    // The compute-skinning pipeline (gated on RT, which is the only path that
    // reaches `build_rt_accel`). A build failure is non-fatal: the RT pass still
    // runs for static geometry, just without skinned hits.
    let skin = match build_skin_pipeline(alloc, device, hot_reload) {
        Ok(s) => Some(s),
        Err(e) => {
            tracing::warn!(
                "RT skin pipeline build failed (skinned meshes absent from reflections): {e}"
            );
            None
        }
    };

    Ok(Some(RtAccelData {
        as_loader,
        book,
        live_tlas,
        live_geom,
        live_geom_size,
        scratch,
        tlas_size: tlas_sizes.acceleration_structure_size,
        tlas_scratch: tlas_sizes.build_scratch_size,
        instance_count,
        frames_in_flight: (frames_in_flight.max(1)) as u64,
        retire: RetirePool::new(),
        static_ring,
        skinned_ring: FrameRing::new(frames_in_flight),
        skin,
        live_deformed: deformed_dummy.buffer,
        _deformed_dummy: deformed_dummy,
        skinned_indices: vk::Buffer::null(),
        frames_in_flight_usize: frames_in_flight.max(1),
        skinned_scratch: SkinnedScratch::default(),
    }))
}

// The rebuild policy for one dynamic update: the mode gate plus whether the
// participating draw set changed since the last update.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct RtRebuildPolicy {
    pub mode: RtDynamicMode,
    pub topology_dirty: bool,
    // Leave see-through glass meshes out of the BVH (see `participates_in_bvh`).
    // Must match what the init build used, or a refresh would silently re-add
    // geometry the transparent pass is already drawing.
    pub exclude_seethrough: bool,
}

// Everything one `dynamic_update` needs beyond the device context, the command
// buffer and the draw list: the rebuild gate, which per-frame ring slot to write,
// and this frame's skinned inputs. Bundled so the entry point stays under the
// argument limit and mirrors DirectX's `RtDynamicInputs`.
pub(in crate::vulkan) struct RtDynamicInputs<'a> {
    pub policy: RtRebuildPolicy,
    // Index into the per-frame ring (the frame's `frame_idx`).
    pub frame_idx: usize,
    // The live shared buffers a topology refresh builds new draw BLAS over.
    pub shared: SharedGeometry,
    // Per-frame joint palettes + the shared skinned buffers; `None` skips the
    // skinned path (the static path runs).
    pub skinned: Option<SkinnedRtInputs<'a>>,
}

// What one topology refresh needs beyond the device context, the command buffer
// and the draw list: the buffers new draw BLAS are built over, the BVH
// membership rule, whether unchanged BLAS are reused, and which per-frame scratch
// slot its builds record over.
#[derive(Clone, Copy)]
struct TopologyRefresh {
    shared: SharedGeometry,
    exclude_seethrough: bool,
    mode: RefreshMode,
    frame_idx: usize,
}

// The static TLAS and geometry table a rebuild recorded, to publish as the live
// BVH once the recording is done.
#[derive(Clone, Copy)]
struct StaticBuilt {
    tlas: vk::AccelerationStructureKHR,
    geom: vk::Buffer,
    geom_size: vk::DeviceSize,
    instance_count: u32,
}

impl StaticBuilt {
    // The structures `slot` holds once its rebuild has sized them.
    fn of(slot: &StaticFrameRing, instance_count: u32) -> RenderResult<Self> {
        let geom = slot.geom.as_ref().ok_or_else(missing_slot_buffer)?;
        Ok(Self {
            tlas: live_accel(&slot.tlas)?,
            geom: geom.buffer,
            geom_size: geom.size,
            instance_count,
        })
    }
}

// A topology refresh in flight: its plan, and the fresh BLAS built for the slots
// the plan could not reuse, indexed by head slot.
struct PendingRefresh {
    refresh: HeadRefresh,
    fresh: Vec<Option<AccelBuffer>>,
}

impl RtAccelData {
    // Per-frame dynamic update, recorded onto `cmd` (the frame's "start" command
    // buffer, submitted before every per-pass trace on the single graphics
    // queue), following the book's plan: drain the retire pool, refresh the draw
    // BLAS head when the participating draw set changed (a cloned prop, a
    // streamed chunk added/removed), then re-skin, rebuild the TLAS, or keep it.
    // A failure is non-fatal: the live BVH is kept and the first error comes
    // back for the caller to report; a failed refresh still lets the step after
    // it run.
    pub(super) fn dynamic_update(
        &mut self,
        ctx: RtDeviceCtx,
        cmd: vk::CommandBuffer,
        draw_objects: &[DrawObject],
        inputs: RtDynamicInputs,
    ) -> RenderResult<RtUpdate> {
        let RtDynamicInputs {
            policy:
                RtRebuildPolicy {
                    mode,
                    topology_dirty,
                    exclude_seethrough,
                },
            frame_idx,
            shared,
            skinned,
        } = inputs;
        // Free any retired resources whose frames-in-flight window has elapsed.
        let now = self.book.tick();
        while let Some(r) = self.retire.pop_due(now, self.frames_in_flight) {
            r.destroy(&self.as_loader);
        }

        // Skinned geometry takes part only with the skin pipeline (GLSL compiled);
        // without it the static path runs.
        let skinned = skinned.filter(|_| self.skin.is_some());
        let Some(plan) = self
            .book
            .plan(mode, topology_dirty, skinned.as_ref().map(|s| s.objects))
        else {
            return Ok(RtUpdate::Done);
        };

        // Fold any added/removed/cloned draw geometry into the BLAS head + rebuild
        // the static TLAS FIRST. On the skinned path `rebuild_skinned` below then
        // overlays the skinned BLAS on top.
        let mut refreshed = Ok(());
        if let Some(mode) = plan.refresh {
            let req = TopologyRefresh {
                shared,
                exclude_seethrough,
                mode,
                frame_idx,
            };
            let if_empty = empty_head(plan.skinned, skinned.is_some());
            refreshed = self.refresh_topology(ctx, cmd, draw_objects, req, if_empty);
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
                self.rebuild_tlas(ctx, cmd, draw_objects, frame_idx)?;
                RtUpdate::Done
            }
            RtStep::Skinned => match skinned {
                Some(skinned) => self.rebuild_skinned(SkinnedRebuild {
                    ctx,
                    cmd,
                    draw_objects,
                    skinned,
                    frame_idx,
                    full_build: plan.full_skinned_build,
                })?,
                None => RtUpdate::Done,
            },
        };
        refreshed.map(|()| stepped)
    }

    // Whether the BVH has nothing left to trace and nothing that could rejoin
    // it. Skinned geometry rejoins only through the skin pipeline.
    pub(super) fn is_spent(&self, skinned_present: bool) -> bool {
        self.book.is_spent(skinned_present && self.skin.is_some())
    }

    // Bring the draw-object BLAS head in line with the current participating
    // draw set: reuse every BLAS whose geometry slice is unchanged (or none, under
    // `RefreshMode::RebuildAll`), build only the new / changed ones, retire the
    // orphans through the deferred-free pool. The cluster BLAS are kept verbatim.
    // The TLAS + geometry table are rebuilt inline over [refreshed head +
    // clusters] into the next `static_ring` slot, like `rebuild_tlas`; on the
    // skinned path `rebuild_skinned` overlays its own TLAS over that the same
    // frame. Building the static TLAS here keeps the live TLAS from referencing an
    // orphan once it is retired, and keeps `tlas_size` in step with the static
    // instance count. The skinned BLAS are untouched -- they belong to their
    // `skinned_ring` slot -- so their slots only have their refit bookkeeping
    // reset, which makes the next skinned update rebuild.
    //
    // Recorded onto `cmd` (the frame's start command buffer), so the builds order
    // before this frame's trace by submission. The orphaned BLAS are freed once
    // the frames-in-flight fence retires the frames whose in-flight trace could
    // still reach them through the not-yet-replaced TLAS; this frame's scratch
    // slot is replaced when this refresh's builds need more than its capacity.
    // A refresh that would leave no draw or cluster geometry follows `if_empty`.
    // With skinned geometry following, it commits the empty head and parks the
    // orphans until the skinned TLAS this frame publishes. With skinned geometry
    // that could rejoin, it builds a zero-instance static TLAS like any other
    // refresh. With none, it empties the book for the caller to drop the whole
    // BVH, orphans included, through a deferred free.
    fn refresh_topology(
        &mut self,
        ctx: RtDeviceCtx,
        cmd: vk::CommandBuffer,
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
        // Take the slot after the live one out, which sidesteps the `&mut self`
        // borrow while the refresh reads the rest of the accel. It is put back on
        // every exit path, and becomes the live slot only when the refresh
        // publishes, so a failed refresh leaves the ring -- and the live handles
        // naming it -- intact.
        let (next, mut slot) = self.static_ring.take_next();
        let mut pending = PendingRefresh {
            fresh: (0..refresh.indices().len()).map(|_| None).collect(),
            refresh,
        };
        let recorded = self.record_refresh(ctx, cmd, draw_objects, &mut pending, req, &mut slot);
        match recorded {
            Ok((tlas_sizes, built)) => {
                self.commit_refresh(pending, draw_objects, built, tlas_sizes);
                self.static_ring.publish(next, slot);
                Ok(())
            }
            Err(e) => {
                // Nothing was recorded against the structures built so far.
                for b in pending.fresh.into_iter().flatten() {
                    b.destroy(&self.as_loader);
                }
                self.static_ring.put(next, slot);
                Err(e)
            }
        }
    }

    // The fallible half of a topology refresh: allocate a fresh BLAS for every
    // slot that needs one, write the TLAS instances and geometry table over the
    // refreshed head into `slot`, size the TLAS and this frame's scratch, then
    // record the fresh BLAS builds and the TLAS build. Returns the TLAS sizes.
    fn record_refresh(
        &mut self,
        ctx: RtDeviceCtx,
        cmd: vk::CommandBuffer,
        draw_objects: &[DrawObject],
        pending: &mut PendingRefresh,
        req: TopologyRefresh,
        slot: &mut StaticFrameRing,
    ) -> RenderResult<(
        vk::AccelerationStructureBuildSizesInfoKHR<'static>,
        StaticBuilt,
    )> {
        let RtDeviceCtx { alloc, device, .. } = ctx;
        let PendingRefresh { refresh, fresh } = pending;
        let shared = req.shared.addresses(device);
        let now = self.book.clock();

        // A fresh BLAS per new / changed slot; reused slots keep their structure.
        let mut fresh_params: Vec<(BlasParams, usize)> = Vec::new();
        let mut max_scratch: u64 = 0;
        for (j, idx) in refresh.fresh_slots() {
            let p = shared.draw_params(&draw_objects[idx]);
            let geo = blas_geometry(&p, shared.index);
            let sizes = blas_build_sizes(&self.as_loader, &geo, p.primitive_count);
            fresh[j] = Some(create_accel(
                alloc,
                &self.as_loader,
                sizes.acceleration_structure_size,
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            )?);
            max_scratch = max_scratch.max(sizes.build_scratch_size);
            fresh_params.push((p, j));
        }
        let fresh_addresses: Vec<u64> = fresh
            .iter()
            .map(|b| b.as_ref().map_or(0, |b| b.address))
            .collect();

        // Static TLAS instances + geometry table over [refreshed draw head +
        // clusters], rebuilt into this ring slot's host buffers in place (growing
        // on demand), exactly like `rebuild_tlas`. The slot was last written a
        // full ring cycle ago, so its trace has retired.
        self.book
            .fill_refresh_instances(refresh, draw_objects, |model, id, blas| {
                tlas_instance(
                    model,
                    id,
                    instance_blas_address(blas, &fresh_addresses, &[]),
                )
            });
        let instance_count = self.book.instances().len() as u32;
        write_or_recreate_host(
            &mut slot.instance,
            alloc,
            self.book.instances(),
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            "RT instance buffer",
            RetireSink::new(&mut self.retire, now),
        )?;
        write_or_recreate_host(
            &mut slot.geom,
            alloc,
            self.book.geom_table(),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "RT geometry table",
            RetireSink::new(&mut self.retire, now),
        )?;
        let instance_buffer = host_buffer(&slot.instance)?;

        // Size the TLAS for this (possibly new) static instance count, then ensure
        // this frame's scratch slot covers every fresh BLAS build + this TLAS.
        let tlas_geo = tlas_geometry(buffer_address(device, instance_buffer));
        let tlas_sizes = tlas_build_sizes(&self.as_loader, &tlas_geo, instance_count);
        max_scratch = max_scratch.max(tlas_sizes.build_scratch_size);
        let scratch_addr = ensure_scratch(&mut self.scratch, ctx, req.frame_idx, max_scratch)?;
        ensure_accel(
            &mut slot.tlas,
            alloc,
            &self.as_loader,
            tlas_sizes.acceleration_structure_size,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            RetireSink::new(&mut self.retire, now),
        )?;
        let built = StaticBuilt::of(slot, instance_count)?;
        let tlas = built.tlas;
        // The commit must follow the builds recorded next, so it is checked now,
        // while a failure still leaves nothing recorded.
        self.book.check_refresh(refresh, fresh)?;

        // Record the fresh draw-BLAS builds (build-barrier-serialized over the one
        // scratch slot they share), then the TLAS build, on `cmd`. Infallible from here on.
        let scratch = vk::DeviceOrHostAddressKHR {
            device_address: scratch_addr,
        };
        for (p, j) in &fresh_params {
            let Some(dst) = fresh[*j].as_ref() else {
                continue;
            };
            let geo = blas_geometry(p, shared.index);
            record_build(
                &self.as_loader,
                cmd,
                build_info(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL, &geo)
                    .dst_acceleration_structure(dst.accel)
                    .scratch_data(scratch),
                build_range(p.primitive_count, p.index_byte_offset),
            );
            build_barrier(device, cmd, BUILD_TO_BUILD);
        }
        record_build(
            &self.as_loader,
            cmd,
            build_info(vk::AccelerationStructureTypeKHR::TOP_LEVEL, &tlas_geo)
                .dst_acceleration_structure(tlas)
                .scratch_data(scratch),
            build_range(instance_count, 0),
        );
        build_barrier(device, cmd, BUILD_TO_TRACE);
        Ok((tlas_sizes, built))
    }

    // The infallible half of a topology refresh: swap the refreshed head into the
    // book, retire the orphans, and publish `slot`'s structures as the live BVH.
    fn commit_refresh(
        &mut self,
        pending: PendingRefresh,
        draw_objects: &[DrawObject],
        built: StaticBuilt,
        tlas_sizes: vk::AccelerationStructureBuildSizesInfoKHR<'static>,
    ) {
        // Publish first: the TLAS just recorded references none of the orphans, so
        // once it is live they are safe to retire.
        self.publish_static(built);
        let PendingRefresh { refresh, fresh } = pending;
        let orphans = self.book.commit_refresh(refresh, fresh, draw_objects);
        self.retire_orphans(orphans);
        self.tlas_size = tlas_sizes.acceleration_structure_size;
        self.tlas_scratch = tlas_sizes.build_scratch_size;
        // The TLAS just built references no skinned BLAS, so no ring slot's refit
        // bookkeeping describes a published tree any more. On the skinned path
        // `rebuild_skinned` re-adds the skinned instances this same frame and
        // rebuilds their BLAS from scratch, which is also the right answer for the
        // change that triggered this refresh. The slots keep their structures for
        // reuse; nothing else references them.
        self.book.release_skinned();
        self.skinned_ring.unpublish(self.book.clock());
        for ring in self.skinned_ring.slots_mut() {
            ring.refit.reset();
        }
    }

    // Retire BLAS the live TLAS no longer references, together with any a
    // refresh parked until a TLAS built after it published.
    fn retire_orphans(&mut self, orphans: Vec<AccelBuffer>) {
        let now = self.book.clock();
        for orphan in orphans.into_iter().chain(self.book.take_parked()) {
            self.retire.push(now, Retired::Accel(orphan));
        }
    }

    // Point the live BVH at a static rebuild's TLAS and geometry table; the ring
    // slot keeps owning them until the cursor comes back around a full ring cycle
    // later (by then its fence has signaled, so no in-flight trace still reads it).
    fn publish_static(&mut self, built: StaticBuilt) {
        self.live_tlas = built.tlas;
        self.live_geom = built.geom;
        self.live_geom_size = built.geom_size;
        self.instance_count = built.instance_count;
    }

    // Rebuild the TLAS + geometry table from the transforms the book collected,
    // rebuilding the next `static_ring` slot's buffers in place, and record the
    // build onto `cmd`. The BLAS are kept (rigid transforms leave object-space
    // geometry unchanged).
    fn rebuild_tlas(
        &mut self,
        ctx: RtDeviceCtx,
        cmd: vk::CommandBuffer,
        draw_objects: &[DrawObject],
        frame_idx: usize,
    ) -> RenderResult<()> {
        // Take the slot after the live one out (see `refresh_topology`); it is put
        // back on every exit path and becomes live only on success.
        let (next, mut slot) = self.static_ring.take_next();
        let result = self.rebuild_tlas_into(ctx, cmd, draw_objects, frame_idx, &mut slot);
        if result.is_ok() {
            self.static_ring.publish(next, slot);
        } else {
            self.static_ring.put(next, slot);
        }
        result
    }

    fn rebuild_tlas_into(
        &mut self,
        ctx: RtDeviceCtx,
        cmd: vk::CommandBuffer,
        draw_objects: &[DrawObject],
        frame_idx: usize,
        slot: &mut StaticFrameRing,
    ) -> RenderResult<()> {
        let RtDeviceCtx { alloc, device, .. } = ctx;
        let now = self.book.clock();
        // Freshly-transformed draw-object instances, then the cluster instances.
        // The geometry table mirrors this order.
        self.book
            .fill_instances(draw_objects, None, |model, id, blas| {
                tlas_instance(model, id, instance_blas_address(blas, &[], &[]))
            });
        let instance_count = self.book.instances().len() as u32;

        // Rebuild this ring slot's buffers in place. The slot was last written a
        // full ring cycle ago, so the frames-in-flight fence has retired every
        // trace that read it; the static instance count is fixed, so the host
        // buffers + TLAS are reused without growing after warm-up.
        write_or_recreate_host(
            &mut slot.instance,
            alloc,
            self.book.instances(),
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            "RT instance buffer",
            RetireSink::new(&mut self.retire, now),
        )?;
        write_or_recreate_host(
            &mut slot.geom,
            alloc,
            self.book.geom_table(),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "RT geometry table",
            RetireSink::new(&mut self.retire, now),
        )?;
        ensure_accel(
            &mut slot.tlas,
            alloc,
            &self.as_loader,
            self.tlas_size,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            RetireSink::new(&mut self.retire, now),
        )?;
        let instance_buffer = host_buffer(&slot.instance)?;
        let built = StaticBuilt::of(slot, instance_count)?;
        // This frame's scratch slot was sized at init; a topology refresh that
        // raised the instance count grew only the slot it recorded over.
        let scratch_addr = ensure_scratch(&mut self.scratch, ctx, frame_idx, self.tlas_scratch)?;

        let tlas_geo = tlas_geometry(buffer_address(device, instance_buffer));
        record_build(
            &self.as_loader,
            cmd,
            build_info(vk::AccelerationStructureTypeKHR::TOP_LEVEL, &tlas_geo)
                .dst_acceleration_structure(built.tlas)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch_addr,
                }),
            build_range(instance_count, 0),
        );
        build_barrier(device, cmd, BUILD_TO_TRACE);

        self.publish_static(built);
        // If skinned instances were still live (the last skinned object just turned
        // invisible), the rebuilt static TLAS no longer references their BLAS. The
        // ring slots keep them for reuse, but an update continues the tree its last
        // full build produced, so re-entering the skinned path after an arbitrary
        // gap must rebuild rather than update from a pose the tree was never fitted
        // for.
        if self.book.commit_static().is_some() {
            self.skinned_ring.unpublish(self.book.clock());
            for ring in self.skinned_ring.slots_mut() {
                ring.refit.reset();
            }
        }
        self.retire_orphans(Vec::new());
        Ok(())
    }

    // Per-frame skinned update, recorded onto `cmd` (the frame's "start" command
    // buffer, which supports compute dispatch + AS builds). Keeps the persistent
    // static + cluster BLAS, re-skins this frame's pose into the deformed buffer,
    // builds or updates one BLAS per skinned object over it, and rebuilds the
    // TLAS + geometry table over the static BLAS plus those skinned instances.
    //
    // Every buffer and structure it writes belongs to `skinned_ring[frame_idx]`
    // and is rewritten in place, grown only on demand, so the steady state
    // allocates nothing (see `SkinnedFrameRing`). The skinned BLAS carry
    // `ALLOW_UPDATE` and are updated IN PLACE (`MODE_UPDATE` with the destination
    // as its own source) while the triangle set is unchanged, with a full rebuild
    // every `rt_refit::REFIT_LIMIT` updates per slot to bound the traversal-quality
    // drift an update accumulates as the pose walks away from the tree's build pose.
    //
    // The three GPU steps are recorded in dependency order on the one command
    // buffer: skin dispatch (writes the deformed buffer), a pipeline barrier
    // (COMPUTE write -> AS-build + FRAGMENT read), then the BLAS/TLAS build (reads
    // it). The start buffer is submitted before every per-pass trace, so build ->
    // trace is ordered by submission too.
    fn rebuild_skinned(&mut self, req: SkinnedRebuild) -> RenderResult<RtUpdate> {
        // This frame slot's resources, taken out for the duration (sidesteps the
        // `&mut self` borrow while the rebuild reads other fields) and put back on
        // every exit path, so a failed rebuild leaves the ring -- and the live
        // handles naming it -- intact. A slot a failed rebuild left live is still
        // traced by the frames since, so this frame skips rather than rewrite it.
        let frame_idx = req.frame_idx;
        let now = self.book.clock();
        let Some(mut slot) = self.skinned_ring.take(frame_idx, now)? else {
            return Ok(RtUpdate::Skipped);
        };
        let result = self.rebuild_skinned_into(req, &mut slot);
        if result.is_ok() {
            self.skinned_ring.publish(frame_idx, slot, now);
            self.retire_orphans(Vec::new());
        } else {
            // A failure can leave freshly (re)allocated BLAS in the slot that no
            // build was recorded into, so the next visit must build, not refit.
            slot.refit.reset();
            self.skinned_ring.put(frame_idx, slot);
        }
        result.map(|()| RtUpdate::Done)
    }

    fn rebuild_skinned_into(
        &mut self,
        req: SkinnedRebuild,
        slot: &mut SkinnedFrameRing,
    ) -> RenderResult<()> {
        let SkinnedRebuild {
            ctx,
            cmd,
            draw_objects,
            skinned,
            frame_idx,
            full_build,
        } = req;
        let skinned = &skinned;
        let RtDeviceCtx { alloc, device, .. } = ctx;
        let now = self.book.clock();
        let frames = self.frames_in_flight_usize;
        let skin = self.skin.as_mut().ok_or_else(|| {
            RenderError::Other("rebuild_skinned called without a skin pipeline".into())
        })?;
        let pipeline = skin.pipeline.handle();
        let pipeline_layout = skin.pipeline_layout.handle();

        // Deformed-vertex buffer: the skin pass writes posed `Vertex`s here,
        // mirroring the skinned VB's indexing so the index buffer addresses it
        // directly. Sized to the highest vertex the skinned objects reach. Owned by
        // this slot, rebuilt in place and grown only when a later frame outgrows it.
        let deformed_extent = self.book.skinned_vertex_extent(skinned.objects);
        self.book
            .fill_skinned_shapes(skinned.objects, deformed_extent as u32);
        let deformed_bytes = (deformed_extent * VERTEX_STRIDE).max(VERTEX_STRIDE);
        // A (re)allocated buffer leaves no tree for an update to continue, so it
        // forces this frame's BLAS to be built from scratch.
        let mut storage_changed = ensure_device_buffer(
            &mut slot.deformed,
            alloc,
            device,
            deformed_bytes,
            RetireSink::new(&mut self.retire, now),
        )?;
        let deformed = slot
            .deformed
            .as_ref()
            .ok_or_else(missing_slot_buffer)?
            .handle();

        // Ensure per-(frame, object) compute descriptor sets exist for this
        // skinned object count, then point this frame's sets at the skinned VB
        // (binding 0), each object's current-frame joint buffer (binding 1), and
        // the fresh deformed buffer (binding 2).
        ensure_skin_sets(device, skin, frames, skinned.objects.len())?;
        let visible = self.book.visible_skinned();
        let frame_sets = &skin.sets[frame_idx];
        let frame_wired = &mut skin.wired[frame_idx];
        for &obj_idx in visible {
            let joint_buffer = skinned
                .joint_buffers
                .get(obj_idx)
                .map(|b| b.buffer())
                .unwrap_or(vk::Buffer::null());
            if joint_buffer == vk::Buffer::null() {
                continue;
            }
            // Skip the re-point when this set already names these three buffers.
            // All three are stable per (frame, object): the skinned VB is shared,
            // the joint buffer is that object's slot in the frame's palette ring,
            // and the deformed buffer belongs to this ring slot for good. The
            // steady state therefore re-points nothing. `storage_changed` guards
            // the handle-value compare against a `VkBuffer` handle a grow recycled
            // into a new allocation.
            let want = [skinned.vertex_buffer, joint_buffer, deformed.buffer];
            if skin_set_current(&frame_wired[obj_idx], &want, storage_changed) {
                continue;
            }
            frame_wired[obj_idx] = want;
            let src_info = vk::DescriptorBufferInfo::default()
                .buffer(skinned.vertex_buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let pal_info = vk::DescriptorBufferInfo::default()
                .buffer(joint_buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let dst_info = vk::DescriptorBufferInfo::default()
                .buffer(deformed.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let set = frame_sets[obj_idx];
            // The RT skin runs at bind pose (before per-frame morph weights
            // exist); morphing happens in the per-frame main fold. Bindings 3/4
            // take the dummy SSBO and target_count is 0, so they go unread.
            let dummy_info = vk::DescriptorBufferInfo::default()
                .buffer(skin.morph_dummy)
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&src_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&pal_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&dst_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&dummy_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&dummy_info)),
            ];
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        // Stage 1: skin dispatch per visible skinned object onto `cmd`.
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        }
        for &obj_idx in visible {
            let obj = &skinned.objects[obj_idx];
            let joint_buffer = skinned
                .joint_buffers
                .get(obj_idx)
                .map(|b| b.buffer())
                .unwrap_or(vk::Buffer::null());
            if joint_buffer == vk::Buffer::null() {
                continue;
            }
            let params = SkinParams {
                vertex_base: obj.vertex_base,
                vertex_count: obj.vertex_count as u32,
                joint_count: obj.joint_count.max(1) as u32,
                target_count: 0,
            };
            // SAFETY: `SkinParams` is `#[repr(C)]` with only 4-byte scalar fields, so it has no
            // padding and all 16 of its bytes are initialized; the slice borrows it and does not
            // outlive it.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    &params as *const SkinParams as *const u8,
                    std::mem::size_of::<SkinParams>(),
                )
            };
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline_layout,
                    0,
                    std::slice::from_ref(&frame_sets[obj_idx]),
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytes,
                );
                device.cmd_dispatch(cmd, (obj.vertex_count as u32).div_ceil(64), 1, 1);
            }
        }

        // Order the skin writes before the BLAS build (AS-build input geometry)
        // and the later hit-shader read (the trace samples the deformed buffer as
        // an SSBO in a fragment shader). An AS build does not auto-synchronize
        // against a prior compute write to its input vertex buffer, so this
        // cross-pass residency barrier is required (Metal / DirectX document the
        // same).
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags::SHADER_READ,
                );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                    | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }

        // Stage 2: one BLAS per skinned object over the deformed buffer.
        let skinned_idx_addr = buffer_address(device, skinned.index_buffer);
        let max_vertex = deformed_extent.saturating_sub(1) as u32;
        let SkinnedScratch {
            params: skinned_params,
            blas_addresses: skinned_blas_addresses,
        } = &mut self.skinned_scratch;
        skinned_params.clear();
        skinned_params.extend(visible.iter().map(|&i| {
            let obj = &skinned.objects[i];
            BlasParams {
                vertex_address: deformed.address,
                max_vertex,
                // u32 indices = 4 bytes each.
                index_byte_offset: obj.index_offset as u32 * 4,
                primitive_count: (obj.index_count / 3) as u32,
            }
        }));

        // Size each skinned BLAS, rebuilding this slot's own BLAS in place when it
        // still fits (else growing); track the largest scratch either a full build
        // or an update needs, since both run over the one scratch slot and which of
        // the two this frame takes is only settled below. A (re)created structure
        // holds no tree, so it forces a full build.
        let mut max_scratch: u64 = 0;
        for (si, p) in skinned_params.iter().enumerate() {
            let geo = skinned_blas_geometry(p, skinned_idx_addr);
            let info = skinned_blas_build_info(&geo, BlasUpdate::Build);
            let sizes = build_sizes(&self.as_loader, &info, p.primitive_count);
            let needed = sizes.acceleration_structure_size;
            match slot.blas.get(si) {
                Some(b) if b.size >= needed => {}
                Some(_) => {
                    let fresh = create_accel(
                        alloc,
                        &self.as_loader,
                        needed,
                        vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                    )?;
                    std::mem::replace(&mut slot.blas[si], fresh).destroy(&self.as_loader);
                    storage_changed = true;
                }
                None => {
                    slot.blas.push(create_accel(
                        alloc,
                        &self.as_loader,
                        needed,
                        vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                    )?);
                    storage_changed = true;
                }
            }
            max_scratch = max_scratch
                .max(sizes.build_scratch_size)
                .max(sizes.update_scratch_size);
        }
        // Structures past this frame's skinned count are dropped in the commit
        // below, once every fallible step has passed. Losing them still changes the
        // published set, so it forces a full build like any other (re)allocation.
        storage_changed |= slot.blas.len() > skinned_params.len();
        skinned_blas_addresses.clear();
        skinned_blas_addresses.extend(
            slot.blas
                .iter()
                .take(skinned_params.len())
                .map(|b| b.address),
        );

        // Instance descriptors + geometry table, in the book's instance order:
        // static objects (current transforms), the cluster instances, then one per
        // skinned object.
        self.book
            .fill_instances(draw_objects, Some(skinned.objects), |model, id, blas| {
                tlas_instance(
                    model,
                    id,
                    instance_blas_address(blas, &[], skinned_blas_addresses),
                )
            });
        let instance_count = self.book.instances().len() as u32;

        // Rewrite this slot's own host buffers in place (re-map + copy) when they
        // still fit, else grow.
        write_or_recreate_host(
            &mut slot.instance,
            alloc,
            self.book.instances(),
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            "RT instance buffer",
            RetireSink::new(&mut self.retire, now),
        )?;
        write_or_recreate_host(
            &mut slot.geom,
            alloc,
            self.book.geom_table(),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "RT geometry table",
            RetireSink::new(&mut self.retire, now),
        )?;
        let instance_buffer = host_buffer(&slot.instance)?;

        // Size the TLAS + scratch (>= the largest skinned BLAS + the TLAS). The
        // skinned instance count can change frame to frame, so size the TLAS from
        // this frame's prebuild rather than the cached size.
        let tlas_geo = tlas_geometry(buffer_address(device, instance_buffer));
        let tlas_sizes = tlas_build_sizes(&self.as_loader, &tlas_geo, instance_count);
        max_scratch = max_scratch.max(tlas_sizes.build_scratch_size);
        // Rebuild this slot's own TLAS in place when it still fits, else grow.
        ensure_accel(
            &mut slot.tlas,
            alloc,
            &self.as_loader,
            tlas_sizes.acceleration_structure_size,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            RetireSink::new(&mut self.retire, now),
        )?;
        let tlas = live_accel(&slot.tlas)?;

        // The scratch was sized for the static build; the skinned BLAS + this
        // frame's TLAS may need more, so re-size this frame's slot if so.
        let scratch_addr = ensure_scratch(&mut self.scratch, ctx, frame_idx, max_scratch)?;

        // Settle build-or-update last, once every fallible step above has passed:
        // recording a build the command buffer never gets would leave the slot
        // claiming a tree a later update could not continue.
        let update = slot
            .refit
            .plan(self.book.skinned_shapes(), storage_changed || full_build);

        // Record the skinned BLAS updates (build-barrier-serialized over the one
        // scratch slot they share), then the TLAS build, on `cmd`. A `Build` writes the structure
        // from scratch; a `Refit` names it as its own source, which the spec defines
        // as an in-place update.
        let scratch = vk::DeviceOrHostAddressKHR {
            device_address: scratch_addr,
        };
        for (p, dst) in self.skinned_scratch.params.iter().zip(&slot.blas) {
            let geo = skinned_blas_geometry(p, skinned_idx_addr);
            let mut bi = skinned_blas_build_info(&geo, update)
                .dst_acceleration_structure(dst.accel)
                .scratch_data(scratch);
            if update == BlasUpdate::Refit {
                bi.src_acceleration_structure = dst.accel;
            }
            record_build(
                &self.as_loader,
                cmd,
                bi,
                build_range(p.primitive_count, p.index_byte_offset),
            );
            build_barrier(device, cmd, BUILD_TO_BUILD);
        }
        record_build(
            &self.as_loader,
            cmd,
            build_info(vk::AccelerationStructureTypeKHR::TOP_LEVEL, &tlas_geo)
                .dst_acceleration_structure(tlas)
                .scratch_data(scratch),
            build_range(instance_count, 0),
        );
        build_barrier(device, cmd, BUILD_TO_TRACE);

        // Publish this slot's resources as the live BVH. The slot keeps owning
        // everything it just built into, so the handles it hands out are the same
        // ones it will hand out next cycle -- which is what lets the skin
        // descriptor cache above skip. The static/cluster head is untouched.
        // BLAS this slot no longer needs (the visible skinned count shrank). Freed
        // in place, not retired: unlike a resource a grow REPLACES, nothing names
        // these -- the TLAS built above does not reference them, no live handle
        // does, and the only TLAS that did was this same slot's, whose frame the
        // fence retired before this one recorded.
        for leftover in slot.blas.drain(self.skinned_scratch.params.len()..) {
            leftover.destroy(&self.as_loader);
        }
        let geom = slot.geom.as_ref().ok_or_else(missing_slot_buffer)?;
        self.live_tlas = tlas;
        self.live_geom = geom.buffer;
        self.live_geom_size = geom.size;
        self.live_deformed = deformed.buffer;
        self.instance_count = instance_count;
        self.skinned_indices = skinned.index_buffer;
        self.book.commit_skinned();
        Ok(())
    }

    // Destroy every acceleration-structure resource. The caller has already
    // idled the device.
    pub(super) fn destroy(&mut self, device: &VkDevice) {
        for r in self.retire.drain() {
            r.destroy(&self.as_loader);
        }
        for slot in self.skinned_ring.slots_mut() {
            slot.destroy(&self.as_loader);
        }
        for slot in self.static_ring.slots_mut() {
            slot.destroy(&self.as_loader);
        }
        for b in self.book.drain_blas() {
            b.destroy(&self.as_loader);
        }
        if let Some(skin) = &self.skin {
            skin.destroy(device);
        }
    }
}

// Grow a `SkinPipeline`'s per-(frame, object) descriptor-set pool to hold at least
// `object_count` objects per frame, reallocating the pool from scratch when it must
// grow. A no-op when the pool already holds enough (or `object_count == 0`). Shared
// by the RT skin path (`RtAccelData::ensure_skin_sets`) and the GPU-driven main-pass
// skin fold (`VkContext::build_main_skin`).
pub(super) fn ensure_skin_sets(
    device: &VkDevice,
    skin: &mut SkinPipeline,
    frames: usize,
    object_count: usize,
) -> RenderResult<()> {
    let have = skin.sets.first().map(|s| s.len()).unwrap_or(0);
    if object_count == 0 || have >= object_count {
        return Ok(());
    }
    // Re-allocate the pool from scratch sized for the (possibly grown) count. The
    // old pool's sets are only ever bound on the frame's own command buffer, which
    // has completed (the per-frame fence gated the frame at the top of
    // `draw_frame`), so freeing the old pool here is safe.
    let total = (frames * object_count) as u32;
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(total * 5);
    let pool = device
        .create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(std::slice::from_ref(&pool_size))
                .max_sets(total),
        )
        .map_err(|e| super::error::map_vk_result(e, "skin descriptor pool"))?;
    let mut sets: Vec<Vec<vk::DescriptorSet>> = Vec::with_capacity(frames);
    for _ in 0..frames {
        let layouts: Vec<vk::DescriptorSetLayout> = (0..object_count)
            .map(|_| skin.set_layout.handle())
            .collect();
        // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
        // it names belongs to this device.
        let alloc = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool.handle())
                    .set_layouts(&layouts),
            )
        }
        .map_err(|e| super::error::map_vk_result(e, "alloc skin descriptor sets"))?;
        sets.push(alloc);
    }
    skin.descriptor_pool = pool;
    skin.sets = sets;
    // Fresh sets point at nothing yet, so the RT path's re-point cache starts
    // empty and its first frame writes every binding.
    skin.wired = (0..frames)
        .map(|_| vec![[vk::Buffer::null(); 3]; object_count])
        .collect();
    Ok(())
}

// Allocate a device-local buffer for the GPU-driven main pass's per-frame deformed
// skinned vertices: a storage buffer the `rt_skin` compute writes + a vertex buffer
// the bindless main pass draws. Unlike the RT deformed buffer it needs no
// acceleration-structure / device-address usage (the main pass binds it as a vertex
// buffer, not by address), so this stays independent of the RT feature being enabled.
pub(super) fn create_main_deformed_buffer(
    alloc: &DeviceAllocator,
    size: u64,
) -> RenderResult<DeviceBuffer> {
    let size = size.max(VERTEX_STRIDE);
    let pooled = alloc.create_buffer(
        size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let buffer = pooled.buffer();
    Ok(DeviceBuffer {
        buffer,
        _pooled: pooled,
        address: 0,
        size,
    })
}

impl super::context::VkRayTracing {
    // Advance the retire clock and destroy every dropped BVH no frame in flight
    // can still trace: the frame-begin fence wait bounds those at
    // `frames_in_flight`, plus the one this frame records.
    pub(in crate::vulkan) fn collect_retired(
        &mut self,
        device: &VkDevice,
        frames_in_flight: usize,
    ) {
        self.retire_tick += 1;
        let depth = frames_in_flight as u64 + 1;
        while let Some(mut accel) = self.retired.pop_due(self.retire_tick, depth) {
            accel.destroy(device);
        }
    }

    // Drop the live BVH, holding it until the frames in flight have finished
    // tracing it.
    pub(in crate::vulkan) fn retire_accel(&mut self) {
        if let Some(accel) = self.accel.take() {
            self.retired.push(self.retire_tick, accel);
        }
    }

    // Destroy the live BVH and every dropped one. The caller has already idled
    // the device.
    pub(in crate::vulkan) fn destroy_accels(&mut self, device: &VkDevice) {
        if let Some(mut accel) = self.accel.take() {
            accel.destroy(device);
        }
        for mut accel in self.retired.drain() {
            accel.destroy(device);
        }
    }
}

impl super::context::VkContext {
    // Build a scene acceleration structure over the current draw set and shared
    // geometry buffers. `None` when the scene has no participating geometry or
    // the build failed (warned).
    pub(in crate::vulkan) fn build_scene_accel(&self) -> Option<RtAccelData> {
        let built = build_rt_accel(
            RtDeviceCtx {
                alloc: &self.hw.alloc,
                instance: &self.hw.instance,
                device: &self.hw.device,
                pd: self.hw.physical_device,
            },
            self.commands.command_pool,
            self.hw.graphics_queue,
            RtSceneGeometry {
                shared: SharedGeometry::of(&self.geometry),
                draw_objects: &self.state.draw.objects,
                clusters: &self.instanced.clusters,
                albedo_count: self.scene.textures.len(),
                exclude_seethrough: self.seethrough_meshes_enabled(),
            },
            self.frames_in_flight,
            self.hot_reload.enabled,
        );
        built.unwrap_or_else(|e| {
            tracing::warn!("RT acceleration-structure build failed: {e}");
            None
        })
    }

    // Forget what the RT descriptor sets point at, so the next frame with a BVH
    // rewires them. Called whenever the BVH is replaced: a new one can reuse a
    // destroyed one's handles.
    pub(in crate::vulkan) fn forget_wired_accel(&mut self) {
        if let Some(rt) = self.rt_reflections.as_mut() {
            rt.forget_accel();
        }
        if let Some(transparent) = self.transparent.as_mut() {
            transparent.forget_rt_dynamic();
        }
    }

    // Replace the live acceleration structure with one built over the current
    // shared vertex / index buffers, and re-point every pass that reads those
    // buffers directly. Called by `rebuild_static_geometry`, which destroys both
    // buffers and re-lays out every draw underneath the BVH: its BLAS then trace
    // the old geometry, its geometry table indexes offsets into freed memory,
    // and the RT / glass descriptor sets still name the destroyed buffers.
    //
    // An empty scene or a failed build leaves no BVH rather than the stale one;
    // the pass stays and the next topology change seeds a new one. The caller
    // has already drained the device.
    pub(in crate::vulkan) fn rebuild_rt_accel(&mut self) {
        self.rt.destroy_accels(&self.hw.device);
        self.rt.accel = self.build_scene_accel();
        self.forget_wired_accel();
        self.rewire_shared_geometry_readers();
    }

    // Re-point the RT resolve + glass sets at the current shared vertex / index
    // buffers. Both bind them directly (the trace fetches attributes at hit
    // points), so every path that replaces the buffers calls this, or the sets
    // keep descriptors on destroyed buffers.
    pub(in crate::vulkan) fn rewire_shared_geometry_readers(&self) {
        let (vertex_buffer, index_buffer) = (
            self.geometry.vertex_buffer.buffer(),
            self.geometry.index_buffer.buffer(),
        );
        if let Some(rt) = self.rt_reflections.as_ref() {
            rt.rewire_geometry(&self.hw.device, vertex_buffer, index_buffer);
        }
        if let Some(transparent) = self.transparent.as_ref() {
            transparent.wire_rt_geometry(&self.hw.device, vertex_buffer, index_buffer);
        }
    }

    // Build the GPU-driven main-pass skinning resources: the `rt_skin` compute
    // pipeline (reused independently of RT), one deformed-vertex buffer per
    // frame-in-flight (storage + vertex usage), and the per-(frame, object)
    // descriptor sets pointing at [skinned bind-pose VB, this object's joint
    // buffer, this frame's deformed buffer]. The deformed + joint buffers are
    // stable for the world's lifetime, so the sets are written once here (no
    // per-frame re-point). Sets `self.state.draw.n_skinned`, which engages the fold. Called
    // from `upload_skinned` when the bindless cull path is active. Mirrors the
    // DirectX `upload_skinned` skin block.
    pub(in crate::vulkan) fn build_main_skin(&mut self, vertex_total: usize) -> RenderResult<()> {
        let device = self.hw.device.clone();
        let frames = self.frames_in_flight.max(1);
        let n = self.state.skinned.draw_objects.len();
        if n == 0 {
            return Ok(());
        }

        let mut skin = build_skin_pipeline(&self.hw.alloc, &device, self.hot_reload.enabled)?;
        ensure_skin_sets(&device, &mut skin, frames, n)?;
        let deformed = self.build_deformed_ring(&skin.sets, vertex_total)?;

        // Point every set at its stable buffers once: binding 1 = this object's
        // joint buffer for that frame. Morph bindings 3 (deltas) + 4 (weights)
        // start on the dummy SSBO; `upload_skinned_morphs` re-points them for
        // objects that carry morph targets. target_count == 0 leaves them
        // unread.
        for f in 0..frames {
            for o in 0..n {
                let set = skin.sets[f][o];
                let pal_info = vk::DescriptorBufferInfo::default()
                    .buffer(self.skinned.joint_buffers[f][o].buffer())
                    .offset(0)
                    .range(vk::WHOLE_SIZE);
                let dummy_info = vk::DescriptorBufferInfo::default()
                    .buffer(skin.morph_dummy)
                    .offset(0)
                    .range(vk::WHOLE_SIZE);
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&pal_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(3)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&dummy_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(4)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&dummy_info)),
                ];
                // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
                // every set and resource it names belongs to this device.
                unsafe { device.update_descriptor_sets(&writes, &[]) };
            }
        }

        self.skinned.skin = Some(skin);
        self.skinned.deformed = deformed;
        self.state.draw.n_skinned = n;
        Ok(())
    }

    // Re-point the skin fold at a replaced bind-pose vertex buffer and re-size
    // the deformed ring to the new vertex total. Called by
    // `rebuild_skinned_geometry` after its swap commits (the device is idle):
    // every (frame, object) set's binding 0 still names the replaced buffer,
    // and the deformed buffers were sized for the old layout. The joint and
    // morph bindings (1/3/4) are untouched; their buffers did not move. A
    // no-op when the fold is inactive. Reached only through the bin's
    // `cn debug` geometry-rebuild path (dead in the FFI lib, live in the bin).
    pub(in crate::vulkan) fn refresh_main_skin_geometry(
        &mut self,
        vertex_total: usize,
    ) -> RenderResult<()> {
        let Some(skin) = self.skinned.skin.as_ref() else {
            return Ok(());
        };
        let deformed = self.build_deformed_ring(&skin.sets, vertex_total)?;
        self.skinned.deformed = deformed;
        Ok(())
    }

    // Create the per-frame deformed ring sized for `vertex_total` and point
    // every (frame, object) set's geometry bindings at it: binding 0 = the
    // shared bind-pose skinned VB, binding 2 = that frame's deformed output.
    // The caller installs the returned ring. Marks the ring unposed: no slot
    // has been posed yet, so the G-buffer velocity must treat the previous
    // deformed buffer as the current one until a full frame has primed it.
    fn build_deformed_ring(
        &self,
        sets: &[Vec<vk::DescriptorSet>],
        vertex_total: usize,
    ) -> RenderResult<Vec<DeviceBuffer>> {
        let frames = self.frames_in_flight.max(1);
        let n = self.state.skinned.draw_objects.len();

        let deformed_bytes = (vertex_total as u64 * VERTEX_STRIDE).max(VERTEX_STRIDE);
        let mut deformed: Vec<DeviceBuffer> = Vec::with_capacity(frames);
        for _ in 0..frames {
            deformed.push(create_main_deformed_buffer(&self.hw.alloc, deformed_bytes)?);
        }

        let src_buffer = self.skinned.vertex_buffer.buffer();
        for (f, deformed_buf) in deformed.iter().enumerate() {
            for &set in sets[f].iter().take(n) {
                let src_info = vk::DescriptorBufferInfo::default()
                    .buffer(src_buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE);
                let dst_info = vk::DescriptorBufferInfo::default()
                    .buffer(deformed_buf.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE);
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&src_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&dst_info)),
                ];
                // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
                // every set and resource it names belongs to this device.
                unsafe { self.hw.device.update_descriptor_sets(&writes, &[]) };
            }
        }

        self.skinned
            .deformed_primed
            .store(false, std::sync::atomic::Ordering::Relaxed);
        Ok(deformed)
    }

    // Per-frame main-pass skinning compute pass: deform every skinned object's
    // bind-pose vertices into this frame's deformed buffer, which the bindless
    // main pass's 2nd indirect draw reads as a vertex buffer. A no-op when the
    // fold is inactive (no skin pipeline / deformed buffer). Run in the Cull graph
    // arm after `encode_cull`, before Main; mirrors the stage-1 skin dispatch in
    // `rebuild_skinned` but targets a per-frame vertex buffer and barriers to
    // VERTEX_ATTRIBUTE_READ instead of the RT BLAS-build read. Independent of RT.
    pub(in crate::vulkan) fn encode_skin(&self, cmd: vk::CommandBuffer, frame_idx: usize) {
        let Some(skin) = self.skinned.skin.as_ref() else {
            return;
        };
        if self.state.draw.n_skinned == 0 || self.skinned.deformed.len() <= frame_idx {
            return;
        }
        let device = &self.hw.device;
        let frame_sets = &skin.sets[frame_idx];
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, skin.pipeline.handle());
        }
        for (o, obj) in self
            .state
            .skinned
            .draw_objects
            .iter()
            .take(self.state.draw.n_skinned)
            .enumerate()
        {
            let params = SkinParams {
                vertex_base: obj.vertex_base,
                vertex_count: obj.vertex_count as u32,
                joint_count: obj.joint_count.max(1) as u32,
                target_count: self
                    .skinned
                    .morph_target_counts
                    .get(o)
                    .copied()
                    .unwrap_or(0),
            };
            // SAFETY: `SkinParams` is `#[repr(C)]` with only 4-byte scalar fields, so it has no
            // padding and all 16 of its bytes are initialized; the slice borrows it and does not
            // outlive it.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    &params as *const SkinParams as *const u8,
                    std::mem::size_of::<SkinParams>(),
                )
            };
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    skin.pipeline_layout.handle(),
                    0,
                    std::slice::from_ref(&frame_sets[o]),
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    skin.pipeline_layout.handle(),
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytes,
                );
                device.cmd_dispatch(cmd, (obj.vertex_count as u32).div_ceil(64), 1, 1);
            }
        }
        // Order the skin writes before the main pass's vertex fetch of the
        // deformed buffer.
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::VERTEX_ATTRIBUTE_READ);
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::VERTEX_INPUT,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::gfx::mesh_payload;

    // Distinct fake buffer handles for the descriptor-cache rule.
    fn buf(raw: u64) -> vk::Buffer {
        use ash::vk::Handle;
        vk::Buffer::from_raw(raw)
    }

    #[test]
    fn skin_set_skips_when_the_slot_hands_back_the_same_buffers() {
        // The steady state slot ownership buys: the skinned VB, the joint buffer
        // and the slot's own deformed buffer are all the same as last cycle, so
        // the set is left alone.
        let wired = [buf(1), buf(2), buf(3)];
        assert!(skin_set_current(&wired, &wired.clone(), false));
    }

    #[test]
    fn skin_set_repoints_when_the_deformed_buffer_moves() {
        // What a swap-recycled ring produced every frame: the first two elements
        // match but the third names a different buffer, so the set is re-pointed.
        let wired = [buf(1), buf(2), buf(3)];
        let want = [buf(1), buf(2), buf(4)];
        assert!(!skin_set_current(&wired, &want, false));
    }

    #[test]
    fn skin_set_repoints_when_a_named_resource_was_reallocated() {
        // A grow can hand back a recycled `VkBuffer` value for a new allocation,
        // so an equal triple is not enough on a frame that (re)allocated.
        let wired = [buf(1), buf(2), buf(3)];
        assert!(!skin_set_current(&wired, &wired.clone(), true));
    }

    #[test]
    fn instance_packs_custom_index_and_full_mask() {
        let d = tlas_instance(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            7,
            0xDEAD_BEEF,
        );
        assert_eq!(d.instance_custom_index_and_mask.low_24(), 7);
        assert_eq!(d.instance_custom_index_and_mask.high_8(), 0xFF);
        assert_eq!(
            // SAFETY: the union was built from `device_handle` two lines above, so that is the live
            // variant.
            unsafe { d.acceleration_structure_reference.device_handle },
            0xDEAD_BEEF
        );
    }

    #[test]
    fn an_empty_host_buffer_still_holds_one_element() {
        assert_eq!(
            host_buffer_size::<concinnity_core::gfx::render_types::RtGeomEntry>(&[]),
            128
        );
        assert_eq!(host_buffer_size::<u8>(&[]), 16);
        assert_eq!(host_buffer_size(&[0u64; 3]), 24);
    }

    #[test]
    fn scratch_capacity_leaves_room_for_the_aligned_address() {
        // The slot is sized so an address aligned up from anywhere inside the
        // buffer still has `required` bytes ahead of it.
        assert_eq!(scratch_capacity(1000, 256), 1256);
        // A device reporting no alignment requirement asks for the bare size.
        assert_eq!(scratch_capacity(1000, 1), 1001);
    }

    #[test]
    fn rt_skin_kernel_compiles() {
        concinnity_shader::require_dxc!();
        // The skin compute kernel compiles to SPIR-V. Its payload offsets and
        // the `SkinParams` block are checked against the Rust mirrors in
        // `shader_layout`, on all three targets rather than this one.
        let spv = crate::vulkan::builtin_shaders::RT_SKIN
            .compile(false)
            .expect("rt skin kernel compiles");
        assert!(super::super::pipeline::is_spirv(&spv));
    }

    #[test]
    fn vertex_stride_matches_the_deformed_payload() {
        // The BLAS strides the deformed buffer by this constant, and the skin
        // kernel writes it in the static `Vertex` layout.
        assert_eq!(size_of::<mesh_payload::Vertex>() as u64, VERTEX_STRIDE);
    }
}
