// src/vulkan/raytrace.rs
//
// Vulkan ray-query acceleration structures for the hardware ray-traced
// reflection pass. Builds, from the shared static vertex / index buffers and the
// `DrawObject` + `InstancedCluster` lists, the bottom- and top-level
// acceleration structures (BLAS / TLAS) the inline-`rayQueryEXT` reflection
// shader traces against, plus a per-instance geometry table the shader uses to
// fetch the hit triangle and shade it.
//
// One triangle BLAS per participating static object (over its slice of the
// shared buffers) and one per instanced cluster; one TLAS instance per object
// and one per cluster instance (transform = the object/instance model matrix,
// `instanceCustomIndex` = the geometry-table index). The BLAS describe
// object-space geometry and never change for a rigid transform; only the TLAS
// instance transforms (and the geometry table's per-instance model matrices the
// shader shades with) move when a prop moves.
//
// Mirrors `directx/raytrace.rs` (DXR inline ray tracing). Skinned geometry is
// added per frame (`rebuild_skinned`): a compute pass deforms each skinned
// object's bind-pose vertices into a model-space buffer, one BLAS
// per skinned object is built or updated over it, and the TLAS + geometry table
// are rebuilt over the persistent static/cluster BLAS plus the skinned tail.
//
// Every resource those two per-frame paths write lives in a ring rather than
// being allocated fresh: `skinned_ring` is one slot per frame in flight, keyed on
// `frame_idx`, and `static_ring` advances a cursor one slot per dynamic-transform
// rebuild. Each slot OWNS its resources for the accel's lifetime and rebuilds
// them in place, growing them only on demand, so a steady scene allocates nothing
// after warm-up. `RtAccelData`'s `live_*` fields are plain handle copies of
// whichever slot last built -- Vulkan has no refcount, so the ownership split has
// to be explicit. Nothing rotates between slots: a slot handing its buffer to the
// next one would make every handle-keyed cache (`SkinPipeline::wired`) miss on
// every visit. The ring rule they rest on is that the `in_flight` fence wait
// retires a slot's previous writer before the next one touches it -- sound for
// the skinned path because it runs on EVERY frame, and for the static path
// because its cursor advances per rebuild rather than per frame (a sparsely-moving
// scene traces one TLAS across many frames, so a frame-keyed slot could be reused
// while a live trace still reads it). See `SkinnedFrameRing` / `StaticFrameRing`.
// Only a topology refresh's orphaned draw BLAS still go through the deferred-free
// `Retired` pool.
//
// Unlike DXR (which binds the TLAS as a root SRV by GPU virtual address each
// frame), Vulkan binds the TLAS + geometry table through a descriptor set, so the
// RT pass re-points the current frame's set at the live handles every frame; see
// `post::rt_reflections::VkContext::rt_update_descriptors`. That re-point is
// unconditional, so ring slot reuse needs nothing extra from it: the set for
// frame `R` is written while frame `R` is the only frame that can bind it, the
// same fence window the ring itself relies on.
//
// TODO(rt-pipeline-vulkan): this uses `VK_KHR_ray_query` (inline tracing in the
// reflection fragment shader), the direct analog of the DXR 1.1 `RayQuery` path.
// A future `VK_KHR_ray_tracing_pipeline` path (raygen/closest-hit/miss + a shader
// binding table) would only be worth it if a feature needs recursive tracing or
// per-material hit shaders, which screen-space reflections do not.

use ash::vk;

use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedPipeline, OwnedPipelineLayout, OwnedSetLayout, VkDevice,
};

use crate::gfx::render_types::{DrawObject, InstancedCluster, RtGeomEntry, SkinnedDrawObject};
use crate::gfx::rt_geom::{cluster_geom_entry, geom_entry, models_dirty, skinned_geom_entry};
use crate::gfx::rt_refit::{BlasUpdate, SkinnedRefit, SkinnedShape};
use crate::gfx::rt_topology::{GeomSig, plan_topology_refresh};
use concinnity_render::uniforms::SkinParams;
// The dynamic-update mode ladder lives in concinnity-render; re-exported so the
// `crate::vulkan::raytrace::RtDynamicMode` path (init + context) keeps resolving.
pub(super) use crate::gfx::rt_geom::RtDynamicMode;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::pipeline::spv_module;
use crate::vulkan::slang_builtins::SlangCompile;

// Byte stride of a `Vertex` in the shared vertex buffer (pos + normal + tangent
// + colour + uv = 14 floats). The BLAS reads positions at this stride and the
// shader fetches attributes at this stride. The deformed (posed) skinned vertex
// buffer the skin kernel writes carries the same 56-byte layout.
const VERTEX_STRIDE: u64 = 56;

// Pack a column-major object-to-world `model` matrix into a Vulkan instance
// transform: a 3x4 ROW-major affine (`VkTransformMatrixKHR`, `matrix[3][4]`),
// row r = `[m_r0 m_r1 m_r2 m_r3]` where element (row r, col c) is the world-matrix
// value. The Rust `model` is column-major, so math element (r, c) lives at
// `model[c][r]`. `VkTransformMatrixKHR` and the DXR 3x4 row-major transform are
// byte-identical, so this is the same packing as `directx::raytrace`. Unit-tested.
pub(super) fn pack_instance_transform(model: [[f32; 4]; 4]) -> vk::TransformMatrixKHR {
    vk::TransformMatrixKHR {
        matrix: [
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
        ],
    }
}

// One TLAS instance descriptor: explicit 3x4 transform, custom index (indexes
// the geometry table), full visibility mask, no SBT offset / flags, and the BLAS
// device address. Inline tracing ignores hit groups so the SBT fields are zero.
fn tlas_instance(
    model: [[f32; 4]; 4],
    custom_index: u32,
    blas_address: u64,
) -> vk::AccelerationStructureInstanceKHR {
    vk::AccelerationStructureInstanceKHR {
        transform: pack_instance_transform(model),
        // instanceCustomIndex (low 24) + mask (high 8 = 0xFF).
        instance_custom_index_and_mask: vk::Packed24_8::new(custom_index & 0x00FF_FFFF, 0xFFu8),
        // instanceShaderBindingTableRecordOffset (24) + flags (8), both zero.
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, 0u8),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            device_handle: blas_address,
        },
    }
}

// Round `value` up to a multiple of `align` (a power of two). Used for the
// scratch buffer's `minAccelerationStructureScratchOffsetAlignment`.
fn align_up(value: u64, align: u64) -> u64 {
    if align <= 1 {
        value
    } else {
        (value + align - 1) & !(align - 1)
    }
}

// A device-local buffer holding an acceleration structure plus its handle.
// `size` is the backing buffer's byte size, so a recycled `AccelBuffer` can be
// reused in place when a later build still fits.
struct AccelBuffer {
    accel: vk::AccelerationStructureKHR,
    // Backing buffer, held so the acceleration structure's memory outlives it.
    _pooled: PooledBuffer,
    size: u64,
}

impl AccelBuffer {
    // The backing buffer retires through the allocator when the value drops;
    // only the acceleration-structure handle is destroyed by hand.
    fn destroy(&self, as_loader: &ash::khr::acceleration_structure::Device) {
        // SAFETY: the handle was created from this device and is destroyed exactly once; the caller
        // has already waited for the device to go idle, so no submission still references it.
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
// (`rt_skin.slang`): set 0 = [src skinned verts, joint palette, deformed output,
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
}

// Resources parked for deferred free: the draw BLAS a topology refresh orphaned,
// and whatever a growing ring slot displaced. Something a rebuild replaces cannot
// be freed in place -- a prior frame's in-flight trace may still reach it, and a
// live handle may still name it if a later step of the same rebuild fails (the
// live BVH is the very slot being rebuilt when the ring is one deep) -- so it is
// freed only once `free_at` updates have elapsed, by which point the
// frames-in-flight fence guarantees neither is true. Growth is rare, so the
// steady state never pushes here.
struct Retired {
    free_at: u64,
    // Structures whose handle has to be destroyed by hand.
    accel: Vec<AccelBuffer>,
    // Buffers that free on `Drop`; parked only so that drop waits out the window,
    // so these are never read.
    _device: Vec<DeviceBuffer>,
    _host: Vec<HostBuffer>,
}

impl Retired {
    fn new(free_at: u64) -> Self {
        Self {
            free_at,
            accel: Vec::new(),
            _device: Vec::new(),
            _host: Vec::new(),
        }
    }

    fn destroy(&self, as_loader: &ash::khr::acceleration_structure::Device) {
        for b in &self.accel {
            b.destroy(as_loader);
        }
    }
}

// The deferred-free pool as a growing ring slot sees it: somewhere to hand the
// resource it displaced, plus the update that resource must survive to. Passed by
// value so the borrow of the pool lasts only the one `ensure_*` call that needs
// it, and consumed by the push so a sink can never park two resources under
// separate deadlines.
struct RetireSink<'a> {
    pool: &'a mut Vec<Retired>,
    free_at: u64,
}

impl<'a> RetireSink<'a> {
    // Built from the accel's fields rather than from `&mut self`, so the borrow
    // stays on `retire` alone and a call can still pass `&self.as_loader`.
    // `free_at` is the same frames-in-flight window a topology refresh's orphans
    // wait out.
    fn new(pool: &'a mut Vec<Retired>, now: u64, depth: u64) -> Self {
        Self {
            pool,
            free_at: now + depth,
        }
    }

    fn accel(self, resource: AccelBuffer) {
        let mut entry = Retired::new(self.free_at);
        entry.accel.push(resource);
        self.pool.push(entry);
    }

    fn device(self, resource: DeviceBuffer) {
        let mut entry = Retired::new(self.free_at);
        entry._device.push(resource);
        self.pool.push(entry);
    }

    fn host(self, resource: HostBuffer) {
        let mut entry = Retired::new(self.free_at);
        entry._host.push(resource);
        self.pool.push(entry);
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
// `static_cursor` to the next slot each rebuild, rebuilds that slot's buffers in
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

// Advance a ring cursor to the next slot, wrapping at `len`. Pure so the
// wrap-around is unit-testable without a device.
fn next_slot(cursor: usize, len: usize) -> usize {
    (cursor + 1) % len.max(1)
}

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
    // against the ring slot's last set to decide build vs update.
    shapes: Vec<SkinnedShape>,
    // This frame's skinned geometry parameters, parallel to `skinned`. Held
    // across the sizing and recording loops, which both rebuild the temporary
    // `vk::*` geometry structs from it.
    params: Vec<BlasParams>,
    // Device addresses of this frame's skinned BLAS, parallel to `skinned`.
    blas_addresses: Vec<u64>,
    // This frame's TLAS instance descriptors and per-instance geometry entries,
    // in instance order.
    instances: Vec<vk::AccelerationStructureInstanceKHR>,
    geom: Vec<RtGeomEntry>,
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

// The Vulkan ray-query acceleration structures + geometry table for hardware ray
// tracing. Held on the context behind an `Option`; present only when RT
// reflections are enabled, the GPU exposes the ray-query extensions, and the
// scene has resident geometry.
pub(super) struct RtAccelData {
    as_loader: ash::khr::acceleration_structure::Device,

    // The persistent static + cluster BLAS in build order: one per participating
    // static object (in `object_indices` order), then one per instanced cluster.
    // Built once and never rebuilt (a rigid transform leaves object-space geometry
    // unchanged). The per-frame skinned BLAS are owned by their `skinned_ring`
    // slot, not held here.
    blas: Vec<AccelBuffer>,
    // How many `blas` entries are the persistent static + cluster BLAS, which is
    // also the base a skinned object's TLAS instance index counts from.
    static_blas_count: usize,
    // The top-level (instance) acceleration structure the trace reads, owned by
    // the ring slot that last rebuilt it (`static_ring` on the static path,
    // `skinned_ring` on the skinned path).
    live_tlas: vk::AccelerationStructureKHR,
    // `[RtGeomEntry; instance_count]` (host-visible), bound as a storage buffer;
    // indexed by the trace's `instanceCustomIndex`. Owned by the same slot as
    // `live_tlas`; `live_geom_size` is its byte size.
    live_geom: vk::Buffer,
    live_geom_size: vk::DeviceSize,
    // Scratch sized for the largest of every BLAS build and the TLAS build;
    // reused by the per-frame TLAS rebuild (the instance count is fixed). Its
    // device address is pre-aligned to the scratch-offset alignment. The skinned
    // rebuild grows it (retiring the old) when a skinned BLAS + TLAS build needs
    // more; `scratch_capacity` is the buffer's byte size.
    scratch: PooledBuffer,
    scratch_addr: u64,
    scratch_capacity: u64,
    // Size the TLAS prebuild reported; the static rebuild recycles the ring slot's
    // TLAS at this size (the static instance count is fixed).
    tlas_size: u64,
    instance_count: u32,
    // Frames-in-flight depth; a retired structure is freed this many frames
    // after the rebuild that displaced it (by then its frame's fence has
    // signalled, so no in-flight trace can still read it).
    frames_in_flight: u64,

    // Per-frame update state.
    // Indices into the frame's `draw.objects` for the participating objects, in
    // BLAS / instance order. Lets a rebuild re-read current transforms in build
    // order and detect a changed draw list.
    object_indices: Vec<usize>,
    // The geometry signature each draw-object BLAS (`blas[..object_indices.len()]`)
    // was built from, parallel to `object_indices`. An incremental topology
    // refresh compares these against the current draw set to reuse every unchanged
    // BLAS and build only the new / changed ones.
    draw_blas_sigs: Vec<GeomSig>,
    // BLAS device addresses, parallel to `blas`, cached so a rebuild re-emits the
    // instance descriptors without re-querying.
    blas_addresses: Vec<u64>,
    // Each participating object's model matrix as baked into the live TLAS. The
    // `Auto` dirty check compares the live draw list against these.
    cached_models: Vec<[[f32; 4]; 4]>,
    // The TLAS instance descriptors for every cluster instance, re-appended
    // verbatim on a rebuild (clusters are baked static into the BVH).
    cluster_instances: Vec<vk::AccelerationStructureInstanceKHR>,
    // The geometry-table entries for the cluster instances, parallel to
    // `cluster_instances`.
    cluster_geom: Vec<RtGeomEntry>,
    // Shared-pool real-texture count for the geometry-table pool indices on a
    // rebuild (the flat-normal fallback sits at this index).
    albedo_count: usize,
    // Shared static vertex / index buffer device addresses + vertex count, so an
    // incremental topology refresh can build a fresh draw BLAS over a slice of the
    // shared buffers (bounding its `max_vertex` exactly as `build_rt_accel` does)
    // without re-threading them from the context. Stable for the buffers'
    // lifetime, which the persistent static BLAS already assume.
    vbuf_addr: u64,
    ibuf_addr: u64,
    total_vertices: usize,

    // Deferred-free pool (for the draw BLAS a topology refresh orphans) + the
    // monotonic per-update counter that drives it. Every per-frame resource is
    // owned by a ring slot, so this never churns on the steady-state path.
    retire: Vec<Retired>,
    frame_counter: u64,

    // Per-rebuild static-transform buffers (see `StaticFrameRing`), owned by their
    // slot and rebuilt in place by the static `rebuild_tlas` path. `static_cursor`
    // advances one slot per rebuild; a slot is revisited only after a full ring
    // cycle, so its prior trace has retired. Slot 0 holds the initial build's
    // structures. The skinned path uses `skinned_ring` instead.
    static_ring: Vec<StaticFrameRing>,
    static_cursor: usize,

    // Per-frame skinned-rebuild resources, one slot per frame in flight, owned by
    // their slot and rebuilt in place (see `SkinnedFrameRing`). Indexed by
    // `frame_idx`.
    skinned_ring: Vec<SkinnedFrameRing>,

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
    // Whether any skinned object is currently live in the BVH (drives whether the
    // per-frame update runs `rebuild_skinned` or the static `rebuild_tlas`).
    has_skinned: bool,
    frames_in_flight_usize: usize,

    // Persistent CPU scratch for the per-frame dynamic update, swapped out with
    // `mem::take` so its heap capacity survives the frame.
    update_scratch: RtUpdateScratch,
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
) -> Result<AccelBuffer, String> {
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
        .map_err(|e| format!("create acceleration structure: {e}"))?;
    Ok(AccelBuffer {
        accel,
        _pooled: pooled,
        size,
    })
}

// Allocate a host-visible, persistently-mapped buffer of `size` bytes with the
// given usage, copy `data` into it, and return the mapped handle.
fn create_host_buffer<T: Copy>(
    alloc: &DeviceAllocator,
    data: &[T],
    usage: vk::BufferUsageFlags,
    _label: &str,
) -> Result<HostBuffer, String> {
    let size = (std::mem::size_of_val(data) as vk::DeviceSize).max(16);
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
) -> Result<(), String> {
    let needed = (std::mem::size_of_val(data) as vk::DeviceSize).max(16);
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
) -> Result<bool, String> {
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
) -> Result<bool, String> {
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
) -> Result<DeviceBuffer, String> {
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
// `SkinParams` push constant. Returns `Err` when shaderc is unavailable or the
// kernel fails to compile; the caller then leaves the skin pipeline absent and
// skinned geometry is omitted from the BVH (the RT pass still runs for static
// geometry). Per-(frame, object) descriptor sets are allocated lazily on the
// first `rebuild_skinned`, when the skinned object count is known.
pub(super) fn build_skin_pipeline(
    alloc: &DeviceAllocator,
    device: &VkDevice,
    hot_reload: bool,
) -> Result<SkinPipeline, String> {
    let spv = super::slang_builtins::RT_SKIN.compile(&super::builtins::Ctx::plain(hot_reload))?;
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
        .map_err(|e| format!("rt skin descriptor set layout: {e}"))?;

    let pc = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<SkinParams>() as u32);
    let set_layouts = [set_layout.handle()];
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    let pipeline_layout = device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(std::slice::from_ref(&pc)),
        )
        .map_err(|e| {
            // SAFETY: the set layout was created from this device and is destroyed exactly once here,
            // with no pipeline layout referencing it yet.
            format!("rt skin pipeline layout: {e}")
        })?;

    let entry = std::ffi::CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module.handle())
        .name(&entry);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout.handle());
    let pipeline = crate::vulkan::pipeline_cache::create_compute_pipeline(device, &info);
    let pipeline = pipeline.map_err(|e| format!("create rt skin pipeline: {e}"))?;

    // Sized to one `MorphEntry` so even a stray read of slot 0 stays in
    // bounds; `target_count == 0` keeps it unread.
    let morph_dummy_pooled = alloc
        .create_buffer(
            28,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .map_err(|e| format!("rt skin morph dummy buffer: {e}"))?;

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

// A global acceleration-structure-build memory barrier: orders one build's
// writes before the next build reads/writes (shared scratch reuse + TLAS reading
// the just-built BLAS). Mirrors the DXR UAV barrier between builds.
fn build_barrier(device: &VkDevice, cmd: vk::CommandBuffer) {
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
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&barrier),
            &[],
            &[],
        );
    }
}

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

// initial BVH: the shared static vertex / index buffers, the participating draw
// objects + instanced clusters, and the pool counts the geometry-table indices
// offset against. Borrowed for the duration of the build.
pub(in crate::vulkan) struct RtSceneGeometry<'a> {
    // The shared static vertex buffer the BLAS reads positions from.
    pub(in crate::vulkan) vertex_buffer: vk::Buffer,
    // The shared static u32 index buffer the BLAS reads triangles from.
    pub(in crate::vulkan) index_buffer: vk::Buffer,
    // Every draw object; the resident, real-triangle ones participate.
    pub(in crate::vulkan) draw_objects: &'a [DrawObject],
    // Every instanced cluster; the non-empty, real-triangle ones participate.
    pub(in crate::vulkan) clusters: &'a [InstancedCluster],
    // The shared pool's real-texture count (resolves each geometry's albedo /
    // normal indices; the flat-normal fallback sits at this index).
    pub(in crate::vulkan) albedo_count: usize,
    // The shared vertex buffer's vertex count (used to bound each geometry's
    // `max_vertex`).
    pub(in crate::vulkan) total_vertices: usize,
    // Leave see-through glass meshes out of the BVH (see `participates_in_bvh`).
    pub(in crate::vulkan) exclude_seethrough: bool,
}

// Build the BLAS / TLAS / geometry table for the scene on a one-shot command
// buffer (submitted and fence-waited so the structures are ready before the
// first frame traces them). Returns `Ok(None)` when there is no resident
// triangle geometry to trace: the caller then leaves RT disabled and falls back
// to SSR.
pub(super) fn build_rt_accel(
    ctx: RtDeviceCtx,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    geometry: RtSceneGeometry,
    frames_in_flight: usize,
    hot_reload: bool,
) -> Result<Option<RtAccelData>, String> {
    let RtDeviceCtx {
        alloc,
        instance,
        device,
        pd,
    } = ctx;
    let RtSceneGeometry {
        vertex_buffer,
        index_buffer,
        draw_objects,
        clusters,
        albedo_count,
        total_vertices,
        exclude_seethrough,
    } = geometry;
    let as_loader = ash::khr::acceleration_structure::Device::new(instance, device);

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

    let vbuf_addr = buffer_address(device, vertex_buffer);
    let ibuf_addr = buffer_address(device, index_buffer);

    // One BLAS-build params entry per participating object first, then clusters.
    // Each object folds its base_vertex into the vertex device address + uses its
    // mesh-relative indices (the shader adds base_vertex back via the geom table),
    // mirroring the DirectX vertex-address fold.
    let mut params: Vec<BlasParams> = Vec::with_capacity(object_indices.len() + cluster_list.len());
    for &i in &object_indices {
        let obj = &draw_objects[i];
        let base_vertex = obj.base_vertex as u64;
        params.push(BlasParams {
            vertex_address: vbuf_addr + base_vertex * VERTEX_STRIDE,
            max_vertex: (total_vertices as u64)
                .saturating_sub(base_vertex)
                .saturating_sub(1) as u32,
            index_byte_offset: obj.index_offset as u32 * 4,
            primitive_count: (obj.index_count / 3) as u32,
        });
    }
    for (_, c) in &cluster_list {
        params.push(BlasParams {
            vertex_address: vbuf_addr,
            max_vertex: (total_vertices as u64).saturating_sub(1) as u32,
            index_byte_offset: c.index_offset as u32 * 4,
            primitive_count: (c.index_count / 3) as u32,
        });
    }

    // Size + allocate each BLAS; track the largest scratch requirement.
    let mut blas: Vec<AccelBuffer> = Vec::with_capacity(params.len());
    let mut max_scratch: u64 = 0;
    for p in &params {
        let geo = blas_geometry(p, ibuf_addr);
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&geo));
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: a property query on a live handle; it only reads.
        unsafe {
            as_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[p.primitive_count],
                &mut sizes,
            );
        }
        blas.push(create_accel(
            alloc,
            &as_loader,
            sizes.acceleration_structure_size,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        )?);
        max_scratch = max_scratch.max(sizes.build_scratch_size);
    }
    let blas_addresses: Vec<u64> = blas
        .iter()
        // SAFETY: the acceleration structure was created from this device and the info struct
        // borrows its handle for the call; the query only reads.
        .map(|b| unsafe {
            as_loader.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(b.accel),
            )
        })
        .collect();

    // Instance descriptors + geometry table, in instance order: static objects
    // (each referencing its own BLAS), then every cluster instance (referencing
    // the cluster's single BLAS, each with its own transform + geom entry).
    let draw_blas_count = object_indices.len();
    let mut instances: Vec<vk::AccelerationStructureInstanceKHR> =
        Vec::with_capacity(object_indices.len());
    let mut geom_entries: Vec<RtGeomEntry> = Vec::with_capacity(object_indices.len());
    for (slot, &i) in object_indices.iter().enumerate() {
        let obj = &draw_objects[i];
        instances.push(tlas_instance(obj.model, slot as u32, blas_addresses[slot]));
        geom_entries.push(geom_entry(obj, albedo_count as u32));
    }
    let mut cluster_instances: Vec<vk::AccelerationStructureInstanceKHR> = Vec::new();
    let mut cluster_geom: Vec<RtGeomEntry> = Vec::new();
    for (ci, (_, c)) in cluster_list.iter().enumerate() {
        let blas_address = blas_addresses[draw_blas_count + ci];
        for model in &c.instances {
            let id = (instances.len() + cluster_instances.len()) as u32;
            cluster_instances.push(tlas_instance(*model, id, blas_address));
            cluster_geom.push(cluster_geom_entry(c, *model, albedo_count as u32));
        }
    }
    instances.extend_from_slice(&cluster_instances);
    geom_entries.extend_from_slice(&cluster_geom);
    let instance_count = instances.len() as u32;

    let instance_buffer = create_host_buffer(
        alloc,
        &instances,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        "RT instance buffer",
    )?;
    let geom_table = create_host_buffer(
        alloc,
        &geom_entries,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        "RT geometry table",
    )?;

    // Size + allocate the TLAS + the shared scratch (>= the largest BLAS/TLAS).
    let tlas_geo = tlas_geometry(buffer_address(device, instance_buffer.buffer));
    let tlas_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(std::slice::from_ref(&tlas_geo));
    let mut tlas_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    // SAFETY: a property query on a live handle; it only reads.
    unsafe {
        as_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &tlas_build_info,
            &[instance_count],
            &mut tlas_sizes,
        );
    }
    max_scratch = max_scratch.max(tlas_sizes.build_scratch_size);
    let tlas = create_accel(
        alloc,
        &as_loader,
        tlas_sizes.acceleration_structure_size,
        vk::AccelerationStructureTypeKHR::TOP_LEVEL,
    )?;

    // Scratch sized to the largest build + the offset alignment so the aligned
    // device address still leaves room for the largest scratch requirement.
    let align = scratch_alignment(instance, pd);
    let scratch_capacity = max_scratch + align;
    let scratch = alloc.create_buffer(
        scratch_capacity,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let scratch_addr = align_up(buffer_address(device, scratch.buffer()), align);

    // Record every BLAS build (build-barrier-serialised over the shared scratch),
    // then the TLAS build, on a one-shot command buffer; fence-wait so the BVH is
    // ready before the first trace.
    super::texture::one_shot_submit(device, command_pool, queue, |cmd| {
        for (slot, p) in params.iter().enumerate() {
            let geo = blas_geometry(p, ibuf_addr);
            let mut bi = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(std::slice::from_ref(&geo));
            bi.dst_acceleration_structure = blas[slot].accel;
            bi.scratch_data = vk::DeviceOrHostAddressKHR {
                device_address: scratch_addr,
            };
            let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(p.primitive_count)
                .primitive_offset(p.index_byte_offset)
                .first_vertex(0)
                .transform_offset(0);
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                as_loader.cmd_build_acceleration_structures(
                    cmd,
                    std::slice::from_ref(&bi),
                    &[std::slice::from_ref(&range)],
                );
            }
            build_barrier(device, cmd);
        }
        let tlas_geo = tlas_geometry(buffer_address(device, instance_buffer.buffer));
        let mut bi = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&tlas_geo));
        bi.dst_acceleration_structure = tlas.accel;
        bi.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_addr,
        };
        let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(instance_count)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0);
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            as_loader.cmd_build_acceleration_structures(
                cmd,
                std::slice::from_ref(&bi),
                &[std::slice::from_ref(&range)],
            );
        }
    })?;

    let cached_models = object_indices
        .iter()
        .map(|&i| draw_objects[i].model)
        .collect();
    let draw_blas_sigs = object_indices
        .iter()
        .map(|&i| GeomSig::of(&draw_objects[i]))
        .collect();
    let static_blas_count = blas.len();

    // Skinned geometry is seeded on the first dynamic frame (like DirectX /
    // Metal), so the init build is static-only. Allocate a 1-element dummy
    // deformed-vertex buffer so the trace's skinned-verts SSBO always binds a
    // valid resource; the first `rebuild_skinned` points it at a ring slot's.
    let deformed_dummy = create_device_buffer(alloc, device, VERTEX_STRIDE)?;

    // The structures just built are the live BVH; home them in static ring slot 0,
    // which owns them from here on. `static_cursor` starts there, so the first
    // dynamic rebuild advances past it and slot 0 is only reused a full ring cycle
    // later -- the same window every other slot rests on.
    let mut static_ring: Vec<StaticFrameRing> = (0..frames_in_flight.max(1))
        .map(|_| StaticFrameRing::default())
        .collect();
    let live_tlas = tlas.accel;
    let live_geom = geom_table.buffer;
    let live_geom_size = geom_table.size;
    static_ring[0] = StaticFrameRing {
        tlas: Some(tlas),
        instance: Some(instance_buffer),
        geom: Some(geom_table),
    };

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
        blas,
        static_blas_count,
        live_tlas,
        live_geom,
        live_geom_size,
        scratch,
        scratch_addr,
        scratch_capacity,
        tlas_size: tlas_sizes.acceleration_structure_size,
        instance_count,
        frames_in_flight: (frames_in_flight.max(1)) as u64,
        object_indices,
        draw_blas_sigs,
        blas_addresses,
        cached_models,
        cluster_instances,
        cluster_geom,
        albedo_count,
        vbuf_addr,
        ibuf_addr,
        total_vertices,
        retire: Vec::new(),
        frame_counter: 0,
        static_ring,
        static_cursor: 0,
        skinned_ring: (0..frames_in_flight.max(1))
            .map(|_| SkinnedFrameRing::default())
            .collect(),
        skin,
        live_deformed: deformed_dummy.buffer,
        _deformed_dummy: deformed_dummy,
        skinned_indices: vk::Buffer::null(),
        has_skinned: false,
        frames_in_flight_usize: frames_in_flight.max(1),
        update_scratch: RtUpdateScratch::default(),
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
    // Per-frame joint palettes + the shared skinned buffers; `None` skips the
    // skinned path (the static path runs).
    pub skinned: Option<SkinnedRtInputs<'a>>,
}

impl RtAccelData {
    // Per-frame dynamic update, recorded onto `cmd` (the frame's "start" command
    // buffer, submitted before every per-pass trace on the single graphics
    // queue). Drains the retire pool, then, when the mode + dirty gate call for
    // it, rebuilds the TLAS + geometry table from current transforms with fresh
    // allocations and parks the outgoing structures for deferred free. A
    // transient failure is non-fatal (keeps the live BVH).
    //
    // `topology_dirty` is set when a runtime change (cloned prop, streamed chunk
    // added/removed) altered the participating draw set since the last update: the
    // BLAS head is refreshed (`refresh_topology`) before the transform path, so
    // the new/removed geometry enters/leaves the BVH instead of being ignored (the
    // `Auto` dirty check only watches transforms of the prior set).
    pub(super) fn dynamic_update(
        &mut self,
        ctx: RtDeviceCtx,
        cmd: vk::CommandBuffer,
        draw_objects: &[DrawObject],
        inputs: RtDynamicInputs,
    ) {
        // Persistent CPU scratch, swapped out so its heap capacity survives the
        // frame and put back on every exit path.
        let mut scratch = std::mem::take(&mut self.update_scratch);
        self.dynamic_update_inner(ctx, cmd, draw_objects, inputs, &mut scratch);
        self.update_scratch = scratch;
    }

    fn dynamic_update_inner(
        &mut self,
        ctx: RtDeviceCtx,
        cmd: vk::CommandBuffer,
        draw_objects: &[DrawObject],
        inputs: RtDynamicInputs,
        scratch: &mut RtUpdateScratch,
    ) {
        let RtDynamicInputs {
            policy:
                RtRebuildPolicy {
                    mode,
                    topology_dirty,
                    exclude_seethrough,
                },
            frame_idx,
            skinned,
        } = inputs;
        self.frame_counter += 1;
        let now = self.frame_counter;
        // Free any retired resources whose frames-in-flight window has elapsed.
        let mut i = 0;
        while i < self.retire.len() {
            if self.retire[i].free_at <= now {
                let r = self.retire.swap_remove(i);
                r.destroy(&self.as_loader);
            } else {
                i += 1;
            }
        }

        if !mode.is_dynamic() {
            return;
        }

        // Skinned objects visible this frame, as indices into the skinned draw
        // list (which is also the joint-palette list's order). The skin pipeline
        // must be present (GLSL compiled); with none, skinned geometry stays
        // absent (the static path runs).
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
            && let Err(e) = self.refresh_topology(ctx, cmd, draw_objects, exclude_seethrough, now)
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
            let req = SkinnedRebuild {
                ctx,
                cmd,
                draw_objects,
                skinned: s,
                frame_idx,
            };
            if let Err(e) = self.rebuild_skinned(req, scratch) {
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

        if let Err(e) = self.rebuild_tlas(ctx, cmd, draw_objects, scratch) {
            tracing::warn!("RT dynamic TLAS rebuild failed (keeping live BVH): {e}");
        }
    }

    // Device address of a BLAS handle (for the instance descriptors).
    fn blas_device_address(&self, accel: vk::AccelerationStructureKHR) -> u64 {
        // SAFETY: the acceleration structure was created from this device and the info struct
        // borrows its handle for the call; the query only reads.
        unsafe {
            self.as_loader.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(accel),
            )
        }
    }

    // Incrementally bring the draw-object BLAS head in line with the current
    // participating draw set: reuse every BLAS whose geometry slice is unchanged
    // (moved, not rebuilt), build only the new / changed ones, retire the orphans
    // through the deferred-free pool. The cluster BLAS are kept verbatim. The TLAS
    // + geometry table are rebuilt inline over [refreshed head + clusters] into the
    // next `static_ring` slot, like `rebuild_tlas`; on the skinned path
    // `rebuild_skinned` overlays its own TLAS over that the same frame. The skinned
    // BLAS are untouched either way -- they belong to their `skinned_ring` slot and
    // are not referenced by the TLAS built here -- so their slots only have their
    // refit bookkeeping reset, which makes the next skinned update rebuild.
    //
    // Recorded onto `cmd` (the frame's start command buffer), so the builds order
    // before this frame's trace by submission. The orphaned BLAS go through
    // `retire` (freed once the frames-in-flight fence retires the frames whose
    // in-flight trace could still reach them through the not-yet-replaced TLAS);
    // the shared scratch is grown (retiring the old) when this refresh's builds
    // need more than the current capacity.
    fn refresh_topology(
        &mut self,
        ctx: RtDeviceCtx,
        cmd: vk::CommandBuffer,
        draw_objects: &[DrawObject],
        exclude_seethrough: bool,
        now: u64,
    ) -> Result<(), String> {
        // Advance to the next ring slot and take it out, which sidesteps the
        // `&mut self` borrow while the refresh reads the rest of the accel. It is
        // put back on every exit path, so a failed refresh leaves the ring -- and
        // the live handles naming it -- intact.
        self.static_cursor = next_slot(self.static_cursor, self.static_ring.len());
        let cursor = self.static_cursor;
        let mut slot = std::mem::take(&mut self.static_ring[cursor]);
        let result =
            self.refresh_topology_into(ctx, cmd, draw_objects, exclude_seethrough, now, &mut slot);
        self.static_ring[cursor] = slot;
        result
    }

    fn refresh_topology_into(
        &mut self,
        ctx: RtDeviceCtx,
        cmd: vk::CommandBuffer,
        draw_objects: &[DrawObject],
        exclude_seethrough: bool,
        now: u64,
        slot: &mut StaticFrameRing,
    ) -> Result<(), String> {
        let RtDeviceCtx {
            alloc,
            instance,
            device,
            pd,
        } = ctx;
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

        // Each cluster instance bakes an `instanceCustomIndex = draw_count + ci`
        // indexing the geometry table (draw entries first, then per cluster instance).
        // The draw count may have changed, so re-bake into a LOCAL copy for this
        // refresh's TLAS build; the copy is committed to `self.cluster_instances` at
        // the end (so a mid-refresh failure does not desync the stored IDs from the
        // draw count), and every later `rebuild_tlas` / `rebuild_skinned` appends the
        // committed copy verbatim. Transform + BLAS reference are preserved (the
        // cluster BLAS are kept verbatim, so their addresses stay valid).
        let new_draw_count = new_indices.len();
        let mut rebaked_clusters = self.cluster_instances.clone();
        for (ci, inst) in rebaked_clusters.iter_mut().enumerate() {
            let id = (new_draw_count + ci) as u32;
            inst.instance_custom_index_and_mask = vk::Packed24_8::new(id & 0x00FF_FFFF, 0xFFu8);
        }

        let plan = plan_topology_refresh(
            &self.object_indices,
            &self.draw_blas_sigs,
            &new_indices,
            &new_sigs,
        );
        let old_draw_count = self.object_indices.len();
        let cluster_count = self.static_blas_count - old_draw_count;

        // --- Fallible allocation phase: everything below reads `self` but does NOT
        // move `self.blas` / `self.blas_addresses` out; a mid-phase `?` therefore
        // leaves the live BVH intact (`self` unchanged except the ring cursor +
        // scratch, whose failure mode is a bounded leak like the existing
        // `rebuild_tlas`, never a desync). The reused draw BLAS are moved out only in
        // the infallible commit at the end. `AccelBuffer` is not `Clone`, so this
        // deferral is what keeps `self.blas` consistent with `object_indices` on
        // failure (an early take + late restore would empty it and later panic). ---

        // Fresh BLAS per new/changed slot; reused slots read their cached address.
        // `fresh_slots[j]` holds the fresh `AccelBuffer` (moved into `new_blas` at
        // commit); `new_addrs[j]` is that slot's BLAS device address for the TLAS.
        let mut fresh_slots: Vec<Option<AccelBuffer>> =
            (0..new_indices.len()).map(|_| None).collect();
        let mut new_addrs: Vec<u64> = vec![0; new_indices.len()];
        let mut fresh_params: Vec<(BlasParams, usize)> = Vec::new();
        let mut max_scratch: u64 = 0;
        for (j, reuse) in plan.reuse.iter().enumerate() {
            match reuse {
                Some(k) => new_addrs[j] = self.blas_addresses[*k],
                None => {
                    let obj = &draw_objects[new_indices[j]];
                    let base_vertex = obj.base_vertex as u64;
                    let p = BlasParams {
                        vertex_address: self.vbuf_addr + base_vertex * VERTEX_STRIDE,
                        max_vertex: (self.total_vertices as u64)
                            .saturating_sub(base_vertex)
                            .saturating_sub(1) as u32,
                        index_byte_offset: obj.index_offset as u32 * 4,
                        primitive_count: (obj.index_count / 3) as u32,
                    };
                    let geo = blas_geometry(&p, self.ibuf_addr);
                    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                        .geometries(std::slice::from_ref(&geo));
                    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
                    // SAFETY: a property query on a live handle; it only reads.
                    unsafe {
                        self.as_loader.get_acceleration_structure_build_sizes(
                            vk::AccelerationStructureBuildTypeKHR::DEVICE,
                            &build_info,
                            &[p.primitive_count],
                            &mut sizes,
                        );
                    }
                    let blas = create_accel(
                        alloc,
                        &self.as_loader,
                        sizes.acceleration_structure_size,
                        vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                    )?;
                    new_addrs[j] = self.blas_device_address(blas.accel);
                    max_scratch = max_scratch.max(sizes.build_scratch_size);
                    fresh_slots[j] = Some(blas);
                    fresh_params.push((p, j));
                }
            }
        }

        // Static TLAS instances + geometry table over [refreshed draw head +
        // clusters]. The skinned tail is NOT included: the static TLAS built here is
        // superseded the same frame by `rebuild_skinned` (which overlays the skinned
        // tail) on the skinned path, and is the live TLAS as-is on the no-skinned
        // path. Building it here (rather than only on the no-skinned path) keeps
        // `self.tlas` referencing no orphaned BLAS before they are retired, and keeps
        // `self.tlas_size` in step with the static instance count.
        let mut instances: Vec<vk::AccelerationStructureInstanceKHR> =
            Vec::with_capacity(new_indices.len() + rebaked_clusters.len());
        let mut geom_entries: Vec<RtGeomEntry> = Vec::with_capacity(instances.capacity());
        for (inst, &idx) in new_indices.iter().enumerate() {
            let obj = &draw_objects[idx];
            instances.push(tlas_instance(obj.model, inst as u32, new_addrs[inst]));
            geom_entries.push(geom_entry(obj, self.albedo_count as u32));
        }
        instances.extend_from_slice(&rebaked_clusters);
        geom_entries.extend_from_slice(&self.cluster_geom);
        let instance_count = instances.len() as u32;

        // Rebuild this ring slot's host buffers in place (growing on demand),
        // exactly like `rebuild_tlas`. The slot was last written a full ring cycle
        // ago, so its trace has retired.
        write_or_recreate_host(
            &mut slot.instance,
            alloc,
            &instances,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            "RT instance buffer",
            RetireSink::new(&mut self.retire, self.frame_counter, self.frames_in_flight),
        )?;
        write_or_recreate_host(
            &mut slot.geom,
            alloc,
            &geom_entries,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "RT geometry table",
            RetireSink::new(&mut self.retire, self.frame_counter, self.frames_in_flight),
        )?;
        let instance_buffer = slot
            .instance
            .as_ref()
            .expect("instance buffer written above")
            .buffer;

        // Size the TLAS for this (possibly new) static instance count.
        let tlas_geo = tlas_geometry(buffer_address(device, instance_buffer));
        let tlas_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&tlas_geo));
        let mut tlas_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: a property query on a live handle; it only reads.
        unsafe {
            self.as_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &tlas_build_info,
                &[instance_count],
                &mut tlas_sizes,
            );
        }
        max_scratch = max_scratch.max(tlas_sizes.build_scratch_size);

        // Ensure the shared scratch covers every fresh BLAS build + this TLAS.
        let align = scratch_alignment(instance, pd);
        if max_scratch + align > self.scratch_capacity {
            self.grow_scratch(alloc, device, max_scratch, align)?;
        }
        let scratch_addr = self.scratch_addr;
        ensure_accel(
            &mut slot.tlas,
            alloc,
            &self.as_loader,
            tlas_sizes.acceleration_structure_size,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            RetireSink::new(&mut self.retire, self.frame_counter, self.frames_in_flight),
        )?;
        let tlas = slot.tlas.as_ref().expect("TLAS sized above").accel;

        // Record the fresh draw-BLAS builds (build-barrier-serialised over the shared
        // scratch), then the TLAS build, on `cmd`. Infallible from here on.
        for (p, j) in &fresh_params {
            let geo = blas_geometry(p, self.ibuf_addr);
            let mut bi = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(std::slice::from_ref(&geo));
            bi.dst_acceleration_structure =
                fresh_slots[*j].as_ref().expect("fresh BLAS present").accel;
            bi.scratch_data = vk::DeviceOrHostAddressKHR {
                device_address: scratch_addr,
            };
            let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(p.primitive_count)
                .primitive_offset(p.index_byte_offset)
                .first_vertex(0)
                .transform_offset(0);
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                self.as_loader.cmd_build_acceleration_structures(
                    cmd,
                    std::slice::from_ref(&bi),
                    &[std::slice::from_ref(&range)],
                );
            }
            build_barrier(device, cmd);
        }
        let tlas_geo = tlas_geometry(buffer_address(device, instance_buffer));
        let mut bi = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&tlas_geo));
        bi.dst_acceleration_structure = tlas;
        bi.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_addr,
        };
        let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(instance_count)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0);
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            self.as_loader.cmd_build_acceleration_structures(
                cmd,
                std::slice::from_ref(&bi),
                &[std::slice::from_ref(&range)],
            );
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }

        // --- Commit (infallible): move the reused BLAS out of the live `self.blas`,
        // assemble the new head, retire the orphans, and publish this slot's
        // structures as the live BVH. ---
        let mut old_blas = std::mem::take(&mut self.blas);
        let _ = std::mem::take(&mut self.blas_addresses);
        let cluster_blas: Vec<AccelBuffer> = old_blas.split_off(old_draw_count);
        let mut draw_head: Vec<Option<AccelBuffer>> = old_blas.into_iter().map(Some).collect();

        let mut new_blas: Vec<AccelBuffer> = Vec::with_capacity(new_indices.len() + cluster_count);
        for (j, reuse) in plan.reuse.iter().enumerate() {
            match reuse {
                Some(k) => new_blas.push(draw_head[*k].take().expect("reused draw BLAS present")),
                None => new_blas.push(fresh_slots[j].take().expect("fresh draw BLAS present")),
            }
        }
        let orphans: Vec<AccelBuffer> = plan
            .retire
            .iter()
            .map(|&k| draw_head[k].take().expect("orphan draw BLAS present"))
            .collect();
        if !orphans.is_empty() {
            let mut entry = Retired::new(now + self.frames_in_flight);
            entry.accel = orphans;
            self.retire.push(entry);
        }
        new_blas.extend(cluster_blas);
        // `new_addrs` holds the draw-head addresses; append the (unchanged) cluster
        // addresses, recomputed from the moved cluster BLAS, to stay parallel.
        for b in &new_blas[new_indices.len()..] {
            new_addrs.push(self.blas_device_address(b.accel));
        }

        // Publish this slot's structures as the live BVH; the slot keeps owning
        // them until the cursor comes back around.
        let geom = slot.geom.as_ref().expect("geometry table written above");
        self.live_tlas = tlas;
        self.live_geom = geom.buffer;
        self.live_geom_size = geom.size;

        self.blas = new_blas;
        self.blas_addresses = new_addrs;
        self.static_blas_count = new_indices.len() + cluster_count;
        self.draw_blas_sigs = new_sigs;
        self.cluster_instances = rebaked_clusters;
        self.tlas_size = tlas_sizes.acceleration_structure_size;
        self.instance_count = instance_count;
        self.has_skinned = false;
        // The TLAS just built references no skinned BLAS, so no ring slot's refit
        // bookkeeping describes a published tree any more. On the skinned path
        // `rebuild_skinned` re-adds the skinned instances this same frame and
        // rebuilds their BLAS from scratch, which is also the right answer for the
        // change that triggered this refresh. The slots keep their structures for
        // reuse; nothing else references them.
        for ring in &mut self.skinned_ring {
            ring.refit.reset();
        }
        // Snapshot the transforms baked into the new TLAS for the next dirty check.
        // (On the skinned path `rebuild_skinned` overwrites `cached_models`.)
        self.cached_models = new_indices.iter().map(|&i| draw_objects[i].model).collect();
        self.object_indices = new_indices;
        Ok(())
    }

    // Rebuild the TLAS + geometry table from `current` transforms, rebuilding the
    // next `static_ring` slot's buffers in place, and record the build onto `cmd`.
    // The BLAS are kept (rigid transforms leave object-space geometry unchanged).
    fn rebuild_tlas(
        &mut self,
        ctx: RtDeviceCtx,
        cmd: vk::CommandBuffer,
        draw_objects: &[DrawObject],
        scratch: &mut RtUpdateScratch,
    ) -> Result<(), String> {
        // Advance to the next ring slot and take it out (see `refresh_topology`);
        // it is put back on every exit path.
        self.static_cursor = next_slot(self.static_cursor, self.static_ring.len());
        let cursor = self.static_cursor;
        let mut slot = std::mem::take(&mut self.static_ring[cursor]);
        let result = self.rebuild_tlas_into(ctx, cmd, draw_objects, scratch, &mut slot);
        self.static_ring[cursor] = slot;
        result
    }

    fn rebuild_tlas_into(
        &mut self,
        ctx: RtDeviceCtx,
        cmd: vk::CommandBuffer,
        draw_objects: &[DrawObject],
        scratch: &mut RtUpdateScratch,
        slot: &mut StaticFrameRing,
    ) -> Result<(), String> {
        let RtDeviceCtx {
            alloc,
            device,
            instance: _,
            pd: _,
        } = ctx;
        let RtUpdateScratch {
            models,
            instances,
            geom: geom_entries,
            ..
        } = scratch;
        // Freshly-transformed draw-object instances, then the cluster instances
        // re-appended verbatim. The geometry table mirrors this order.
        instances.clear();
        geom_entries.clear();
        for (inst, &idx) in self.object_indices.iter().enumerate() {
            let obj = &draw_objects[idx];
            instances.push(tlas_instance(
                obj.model,
                inst as u32,
                self.blas_addresses[inst],
            ));
            geom_entries.push(geom_entry(obj, self.albedo_count as u32));
        }
        instances.extend_from_slice(&self.cluster_instances);
        geom_entries.extend_from_slice(&self.cluster_geom);

        // Refresh the live instance count so the TLAS build below covers exactly
        // this rebuild's descriptors. A prior skinned rebuild may have left a
        // larger count; reusing it would read past the valid instance buffer.
        self.instance_count = instances.len() as u32;

        // Rebuild this ring slot's buffers in place. The slot was last written a
        // full ring cycle ago, so the frames-in-flight fence has retired every
        // trace that read it; the static instance count is fixed, so the host
        // buffers + TLAS are reused without growing after warm-up.
        write_or_recreate_host(
            &mut slot.instance,
            alloc,
            instances.as_slice(),
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            "RT instance buffer",
            RetireSink::new(&mut self.retire, self.frame_counter, self.frames_in_flight),
        )?;
        write_or_recreate_host(
            &mut slot.geom,
            alloc,
            geom_entries.as_slice(),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "RT geometry table",
            RetireSink::new(&mut self.retire, self.frame_counter, self.frames_in_flight),
        )?;
        ensure_accel(
            &mut slot.tlas,
            alloc,
            &self.as_loader,
            self.tlas_size,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            RetireSink::new(&mut self.retire, self.frame_counter, self.frames_in_flight),
        )?;
        let instance_buffer = slot
            .instance
            .as_ref()
            .expect("instance buffer written above")
            .buffer;
        let tlas = slot.tlas.as_ref().expect("TLAS sized above").accel;

        let tlas_geo = tlas_geometry(buffer_address(device, instance_buffer));
        let mut bi = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&tlas_geo));
        bi.dst_acceleration_structure = tlas;
        bi.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: self.scratch_addr,
        };
        let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(self.instance_count)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0);
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            self.as_loader.cmd_build_acceleration_structures(
                cmd,
                std::slice::from_ref(&bi),
                &[std::slice::from_ref(&range)],
            );
            // Order the build before this frame's trace reads the TLAS.
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }

        // Publish this slot's structures as the live BVH; the slot keeps owning
        // them until the cursor comes back around a full ring cycle later (by then
        // its fence has signalled, so no in-flight trace still reads it).
        let geom = slot.geom.as_ref().expect("geometry table written above");
        self.live_tlas = tlas;
        self.live_geom = geom.buffer;
        self.live_geom_size = geom.size;

        // If skinned instances were still live (the last skinned object just turned
        // invisible), the rebuilt static TLAS no longer references their BLAS. The
        // ring slots keep them for reuse, but an update continues the tree its last
        // full build produced, so re-entering the skinned path after an arbitrary
        // gap must rebuild rather than update from a pose the tree was never fitted
        // for.
        if self.has_skinned {
            self.has_skinned = false;
            for ring in &mut self.skinned_ring {
                ring.refit.reset();
            }
        }
        self.cached_models.clear();
        self.cached_models.extend_from_slice(models);
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
    fn rebuild_skinned(
        &mut self,
        req: SkinnedRebuild,
        scratch: &mut RtUpdateScratch,
    ) -> Result<(), String> {
        // This frame slot's resources, taken out for the duration (sidesteps the
        // `&mut self` borrow while the rebuild reads other fields) and put back on
        // every exit path, so a failed rebuild leaves the ring -- and the live
        // handles naming it -- intact.
        let frame_idx = req.frame_idx;
        let mut slot = std::mem::take(&mut self.skinned_ring[frame_idx]);
        let result = self.rebuild_skinned_into(req, scratch, &mut slot);
        self.skinned_ring[frame_idx] = slot;
        result
    }

    fn rebuild_skinned_into(
        &mut self,
        req: SkinnedRebuild,
        scratch: &mut RtUpdateScratch,
        slot: &mut SkinnedFrameRing,
    ) -> Result<(), String> {
        let SkinnedRebuild {
            ctx,
            cmd,
            draw_objects,
            skinned,
            frame_idx,
        } = req;
        let skinned = &skinned;
        let RtUpdateScratch {
            skinned: skinned_objects,
            models,
            shapes,
            params: skinned_params,
            blas_addresses: skinned_blas_addresses,
            instances,
            geom: geom_entries,
        } = scratch;
        let RtDeviceCtx {
            alloc,
            instance,
            device,
            pd,
        } = ctx;
        let skin = self
            .skin
            .as_ref()
            .ok_or("rebuild_skinned called without a skin pipeline")?;
        let pipeline = skin.pipeline.handle();
        let pipeline_layout = skin.pipeline_layout.handle();

        // Deformed-vertex buffer: the skin pass writes posed `Vertex`s here,
        // mirroring the skinned VB's indexing so the index buffer addresses it
        // directly. Sized to the highest vertex the skinned objects reach. Owned by
        // this slot, rebuilt in place and grown only when a later frame outgrows it.
        let deformed_extent: u64 = skinned_objects
            .iter()
            .map(|&i| {
                skinned.objects[i].vertex_base as u64 + skinned.objects[i].vertex_count as u64
            })
            .max()
            .unwrap_or(0);
        let deformed_bytes = (deformed_extent * VERTEX_STRIDE).max(VERTEX_STRIDE);
        // A (re)allocated buffer leaves no tree for an update to continue, so it
        // forces this frame's BLAS to be built from scratch.
        let mut storage_changed = ensure_device_buffer(
            &mut slot.deformed,
            alloc,
            device,
            deformed_bytes,
            RetireSink::new(&mut self.retire, self.frame_counter, self.frames_in_flight),
        )?;
        let deformed = slot
            .deformed
            .as_ref()
            .expect("deformed buffer sized above")
            .handle();

        // Ensure per-(frame, object) compute descriptor sets exist for this
        // skinned object count, then point this frame's sets at the skinned VB
        // (binding 0), each object's current-frame joint buffer (binding 1), and
        // the fresh deformed buffer (binding 2).
        self.ensure_skin_sets(device, skinned.objects.len())?;
        let skin = self.skin.as_mut().expect("skin pipeline present");
        let frame_sets = &skin.sets[frame_idx];
        let frame_wired = &mut skin.wired[frame_idx];
        for &obj_idx in skinned_objects.iter() {
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
        let skin = self.skin.as_ref().expect("skin pipeline present");
        let frame_sets = &skin.sets[frame_idx];
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        }
        for &obj_idx in skinned_objects.iter() {
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
            // padding and all 16 of its bytes are initialised; the slice borrows it and does not
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
        // an SSBO in a fragment shader). An AS build does not auto-synchronise
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
        skinned_params.clear();
        shapes.clear();
        for &i in skinned_objects.iter() {
            let obj = &skinned.objects[i];
            skinned_params.push(BlasParams {
                vertex_address: deformed.address,
                max_vertex,
                // u32 indices = 4 bytes each.
                index_byte_offset: obj.index_offset as u32 * 4,
                primitive_count: (obj.index_count / 3) as u32,
            });
            shapes.push(SkinnedShape {
                index_offset: obj.index_offset,
                index_count: obj.index_count,
                vertex_extent: deformed_extent as u32,
            });
        }

        // Size each skinned BLAS, rebuilding this slot's own BLAS in place when it
        // still fits (else growing); track the largest scratch either a full build
        // or an update needs, since both run over the shared scratch and which of
        // the two this frame takes is only settled below. A (re)created structure
        // holds no tree, so it forces a full build.
        let mut max_scratch: u64 = 0;
        for (si, p) in skinned_params.iter().enumerate() {
            let geo = skinned_blas_geometry(p, skinned_idx_addr);
            let build_info = skinned_blas_build_info(&geo, BlasUpdate::Build);
            let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
            // SAFETY: a property query on a live handle; it only reads.
            unsafe {
                self.as_loader.get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    &[p.primitive_count],
                    &mut sizes,
                );
            }
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
        skinned_blas_addresses.extend(slot.blas[..skinned_params.len()].iter().map(|b| {
            // SAFETY: the acceleration structure was created from this device and the info struct
            // borrows its handle for the call; the query only reads.
            unsafe {
                self.as_loader.get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(b.accel),
                )
            }
        }));

        // Instance descriptors + geometry table, in instance order: static
        // objects (current transforms), then the cluster instances verbatim, then
        // one per skinned object (BLAS index `static_blas_count + si`).
        instances.clear();
        geom_entries.clear();
        for (inst, &idx) in self.object_indices.iter().enumerate() {
            let obj = &draw_objects[idx];
            instances.push(tlas_instance(
                obj.model,
                inst as u32,
                self.blas_addresses[inst],
            ));
            geom_entries.push(geom_entry(obj, self.albedo_count as u32));
        }
        instances.extend_from_slice(&self.cluster_instances);
        geom_entries.extend_from_slice(&self.cluster_geom);
        for (si, &obj_idx) in skinned_objects.iter().enumerate() {
            let obj = &skinned.objects[obj_idx];
            let id = instances.len() as u32;
            instances.push(tlas_instance(obj.model, id, skinned_blas_addresses[si]));
            // The skinned object's textures bake into the shared bindless pool
            // from its own `texture_slot` / `normal_map_slot`, so the pool index
            // reads off `obj` directly (no list-position dependence).
            geom_entries.push(skinned_geom_entry(obj, self.albedo_count as u32));
        }
        let instance_count = instances.len() as u32;

        // Rewrite this slot's own host buffers in place (re-map + copy) when they
        // still fit, else grow.
        write_or_recreate_host(
            &mut slot.instance,
            alloc,
            instances.as_slice(),
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            "RT instance buffer",
            RetireSink::new(&mut self.retire, self.frame_counter, self.frames_in_flight),
        )?;
        write_or_recreate_host(
            &mut slot.geom,
            alloc,
            geom_entries.as_slice(),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "RT geometry table",
            RetireSink::new(&mut self.retire, self.frame_counter, self.frames_in_flight),
        )?;
        let instance_buffer = slot
            .instance
            .as_ref()
            .expect("instance buffer written above")
            .buffer;

        // Size the TLAS + scratch (>= the largest skinned BLAS + the TLAS). The
        // skinned instance count can change frame to frame, so size the TLAS from
        // this frame's prebuild rather than the cached size.
        let tlas_geo = tlas_geometry(buffer_address(device, instance_buffer));
        let tlas_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&tlas_geo));
        let mut tlas_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: a property query on a live handle; it only reads.
        unsafe {
            self.as_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &tlas_build_info,
                &[instance_count],
                &mut tlas_sizes,
            );
        }
        max_scratch = max_scratch.max(tlas_sizes.build_scratch_size);
        // Rebuild this slot's own TLAS in place when it still fits, else grow.
        ensure_accel(
            &mut slot.tlas,
            alloc,
            &self.as_loader,
            tlas_sizes.acceleration_structure_size,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            RetireSink::new(&mut self.retire, self.frame_counter, self.frames_in_flight),
        )?;
        let tlas = slot.tlas.as_ref().expect("TLAS sized above").accel;

        // The shared scratch was sized for the static build; the skinned BLAS +
        // this frame's TLAS may need more. Grow it (retire the old) if so.
        let align = scratch_alignment(instance, pd);
        if max_scratch + align > self.scratch_size() {
            self.grow_scratch(alloc, device, max_scratch, align)?;
        }
        let scratch_addr = self.scratch_addr;

        // Settle build-or-update last, once every fallible step above has passed:
        // recording a build the command buffer never gets would leave the slot
        // claiming a tree a later update could not continue.
        let update = slot.refit.plan(shapes, storage_changed);

        // Record the skinned BLAS updates (build-barrier-serialised over the shared
        // scratch), then the TLAS build, on `cmd`. A `Build` writes the structure
        // from scratch; a `Refit` names it as its own source, which the spec defines
        // as an in-place update.
        for (si, p) in skinned_params.iter().enumerate() {
            let geo = skinned_blas_geometry(p, skinned_idx_addr);
            let mut bi = skinned_blas_build_info(&geo, update);
            bi.dst_acceleration_structure = slot.blas[si].accel;
            if update == BlasUpdate::Refit {
                bi.src_acceleration_structure = slot.blas[si].accel;
            }
            bi.scratch_data = vk::DeviceOrHostAddressKHR {
                device_address: scratch_addr,
            };
            let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(p.primitive_count)
                .primitive_offset(p.index_byte_offset)
                .first_vertex(0)
                .transform_offset(0);
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                self.as_loader.cmd_build_acceleration_structures(
                    cmd,
                    std::slice::from_ref(&bi),
                    &[std::slice::from_ref(&range)],
                );
            }
            build_barrier(device, cmd);
        }
        let tlas_geo = tlas_geometry(buffer_address(device, instance_buffer));
        let mut bi = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&tlas_geo));
        bi.dst_acceleration_structure = tlas;
        bi.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_addr,
        };
        let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(instance_count)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0);
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            self.as_loader.cmd_build_acceleration_structures(
                cmd,
                std::slice::from_ref(&bi),
                &[std::slice::from_ref(&range)],
            );
            // Order the TLAS build before this frame's trace reads it.
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }

        // Publish this slot's resources as the live BVH. The slot keeps owning
        // everything it just built into, so the handles it hands out are the same
        // ones it will hand out next cycle -- which is what lets the skin
        // descriptor cache above skip. The static/cluster `blas` head is untouched.
        // BLAS this slot no longer needs (the visible skinned count shrank). Freed
        // in place, not retired: unlike a resource a grow REPLACES, nothing names
        // these -- the TLAS built above does not reference them, no live handle
        // does, and the only TLAS that did was this same slot's, whose frame the
        // fence retired before this one recorded.
        for leftover in slot.blas.drain(skinned_params.len()..) {
            leftover.destroy(&self.as_loader);
        }
        let geom = slot.geom.as_ref().expect("geometry table written above");
        self.live_tlas = tlas;
        self.live_geom = geom.buffer;
        self.live_geom_size = geom.size;
        self.live_deformed = deformed.buffer;
        self.instance_count = instance_count;
        self.skinned_indices = skinned.index_buffer;
        self.has_skinned = true;
        self.cached_models.clear();
        self.cached_models.extend_from_slice(models);
        Ok(())
    }

    // Current scratch buffer byte size (queried lazily; the scratch was sized
    // `max_scratch + align` at the build that allocated it).
    fn scratch_size(&self) -> u64 {
        self.scratch_capacity
    }

    // Grow the shared scratch buffer to cover `required + align` bytes and retire
    // the old (a prior frame's build may still read it). Re-aligns the cached
    // scratch device address.
    fn grow_scratch(
        &mut self,
        alloc: &DeviceAllocator,
        device: &VkDevice,
        required: u64,
        align: u64,
    ) -> Result<(), String> {
        let new_capacity = required + align;
        let buffer = alloc.create_buffer(
            new_capacity,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let addr = align_up(buffer_address(device, buffer.buffer()), align);
        // The replaced scratch retires through the allocator once no in-flight
        // build can reference it.
        self.scratch = buffer;
        self.scratch_addr = addr;
        self.scratch_capacity = new_capacity;
        Ok(())
    }

    // Ensure the per-(frame, object) compute descriptor sets cover `object_count`
    // skinned objects. Allocated lazily on the first skinned rebuild (the count
    // is unknown at init, before `upload_skinned`). Idempotent once sized.
    fn ensure_skin_sets(&mut self, device: &VkDevice, object_count: usize) -> Result<(), String> {
        let frames = self.frames_in_flight_usize;
        let skin = self
            .skin
            .as_mut()
            .ok_or("ensure_skin_sets called without a skin pipeline")?;
        ensure_skin_sets(device, skin, frames, object_count)
    }

    // Destroy every acceleration-structure resource. The caller has already
    // idled the device.
    pub(super) fn destroy(&mut self, device: &VkDevice) {
        for r in self.retire.drain(..) {
            r.destroy(&self.as_loader);
        }
        for slot in &mut self.skinned_ring {
            slot.destroy(&self.as_loader);
        }
        for slot in &self.static_ring {
            slot.destroy(&self.as_loader);
        }
        for b in &self.blas {
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
) -> Result<(), String> {
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
        .map_err(|e| format!("skin descriptor pool: {e}"))?;
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
        .map_err(|e| format!("alloc skin descriptor sets: {e}"))?;
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
) -> Result<DeviceBuffer, String> {
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

impl super::context::VkContext {
    // Replace the live acceleration structure with one built over the current
    // shared vertex / index buffers, and re-point every pass that reads those
    // buffers directly. Called by `rebuild_static_geometry`, which destroys both
    // buffers and re-lays out every draw underneath the BVH: its BLAS then trace
    // the old geometry, its geometry table indexes offsets into freed memory,
    // and the RT / glass descriptor sets still name the destroyed buffers.
    //
    // An empty scene or a failed build drops the BVH rather than keeping the
    // stale one; RT falls back to SSR. The caller has already drained the device.
    pub(in crate::vulkan) fn rebuild_rt_accel(&mut self) {
        let fresh = match build_rt_accel(
            RtDeviceCtx {
                alloc: &self.alloc,
                instance: &self.instance,
                device: &self.device,
                pd: self.physical_device,
            },
            self.commands.command_pool,
            self.graphics_queue,
            RtSceneGeometry {
                vertex_buffer: self.geometry.vertex_buffer.buffer(),
                index_buffer: self.geometry.index_buffer.buffer(),
                draw_objects: &self.draw.objects,
                clusters: &self.instanced.clusters,
                albedo_count: self.textures.len(),
                total_vertices: self.rt_static_vertex_count,
                exclude_seethrough: self.seethrough_meshes_enabled(),
            },
            self.frames_in_flight,
            self.hot_reload.enabled,
        ) {
            Ok(accel) => accel,
            Err(e) => {
                tracing::warn!("RT acceleration-structure rebuild failed (dropping BVH): {e}");
                None
            }
        };
        if let Some(mut old) = self.rt_accel.take() {
            old.destroy(&self.device);
        }
        self.rt_accel = fresh;

        // The resolve + glass sets bind the shared vertex / index buffers
        // directly (the trace fetches attributes at hit points), so they must
        // follow the swap even when the BVH itself was dropped.
        let device = self.device.clone();
        let (vertex_buffer, index_buffer) = (
            self.geometry.vertex_buffer.buffer(),
            self.geometry.index_buffer.buffer(),
        );
        if let Some(rt) = self.rt_reflections.as_ref() {
            rt.rewire_geometry(&device, vertex_buffer, index_buffer);
        }
        if let Some(transparent) = self.transparent.as_ref() {
            transparent.wire_rt_geometry(&device, vertex_buffer, index_buffer);
        }
    }

    // Build the GPU-driven main-pass skinning resources: the `rt_skin` compute
    // pipeline (reused independently of RT), one deformed-vertex buffer per
    // frame-in-flight (storage + vertex usage), and the per-(frame, object)
    // descriptor sets pointing at [skinned bind-pose VB, this object's joint
    // buffer, this frame's deformed buffer]. The deformed + joint buffers are
    // stable for the world's lifetime, so the sets are written once here (no
    // per-frame re-point). Sets `self.draw.n_skinned`, which engages the fold. Called
    // from `upload_skinned` when the bindless cull path is active. Mirrors the
    // DirectX `upload_skinned` skin block.
    pub(in crate::vulkan) fn build_main_skin(&mut self, vertex_total: usize) -> Result<(), String> {
        let device = self.device.clone();
        let frames = self.frames_in_flight.max(1);
        let n = self.skinned.draw_objects.len();
        if n == 0 {
            return Ok(());
        }

        let mut skin = build_skin_pipeline(&self.alloc, &device, self.hot_reload.enabled)?;
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
        self.draw.n_skinned = n;
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
    ) -> Result<(), String> {
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
    ) -> Result<Vec<DeviceBuffer>, String> {
        let frames = self.frames_in_flight.max(1);
        let n = self.skinned.draw_objects.len();

        let deformed_bytes = (vertex_total as u64 * VERTEX_STRIDE).max(VERTEX_STRIDE);
        let mut deformed: Vec<DeviceBuffer> = Vec::with_capacity(frames);
        for _ in 0..frames {
            deformed.push(create_main_deformed_buffer(&self.alloc, deformed_bytes)?);
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
                unsafe { self.device.update_descriptor_sets(&writes, &[]) };
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
        if self.draw.n_skinned == 0 || self.skinned.deformed.len() <= frame_idx {
            return;
        }
        let device = &self.device;
        let frame_sets = &skin.sets[frame_idx];
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, skin.pipeline.handle());
        }
        for (o, obj) in self
            .skinned
            .draw_objects
            .iter()
            .take(self.draw.n_skinned)
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
            // padding and all 16 of its bytes are initialised; the slice borrows it and does not
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
    fn pack_instance_transform_transposes_column_major_to_3x4_row_major() {
        // A column-major model with a known translation column [10, 20, 30].
        let model = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [10.0, 20.0, 30.0, 1.0],
        ];
        let t = pack_instance_transform(model);
        // VkTransformMatrixKHR is 3x4 row-major (flat); the translation is the
        // last entry of each 4-wide row.
        assert_eq!(
            t.matrix,
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
            t.matrix,
            [
                1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0
            ]
        );
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
    fn align_up_rounds_to_power_of_two() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
        // align <= 1 is identity.
        assert_eq!(align_up(123, 1), 123);
    }

    #[test]
    fn rt_skin_kernel_compiles() {
        if !crate::slangc_gate::slangc_available() {
            return;
        }
        // The skin compute kernel compiles to SPIR-V. Its payload offsets and
        // the `SkinParams` block are checked against the Rust mirrors in
        // `shader_layout`, on all three targets rather than this one.
        let spv = crate::vulkan::slang_builtins::RT_SKIN
            .compile(&crate::vulkan::builtins::Ctx::plain(false))
            .expect("rt skin kernel compiles");
        assert!(super::super::pipeline::is_spirv(&spv));
    }

    #[test]
    fn vertex_stride_matches_the_deformed_payload() {
        // The BLAS strides the deformed buffer by this constant, and the skin
        // kernel writes it in the static `Vertex` layout.
        assert_eq!(
            size_of::<crate::gfx::mesh_payload::Vertex>() as u64,
            VERTEX_STRIDE
        );
    }
}
