// Vulkan rendering context. Owns all GPU resources, the GLFW window, and input state.
// Mirrors the public API of metal::MtlContext so GraphicsSystem can drive both
// backends identically.

use ash::vk;

use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedFramebuffer, OwnedPipeline, OwnedPipelineLayout, OwnedRenderPass,
    OwnedSampler, OwnedSetLayout, VkDevice,
};

use crate::gfx::backend::FrameParams;
use crate::gfx::render_types::*;

use super::allocator::PooledBuffer;
use super::draw::*;
use super::input::*;
use super::post::*;
use super::texture::*;

// Off-screen HDR render-target format. The main pass renders linear-light
// radiance into this; the composite pass tonemaps it down to the swapchain's
// 8-bit format. `R16G16B16A16_SFLOAT` is universally supported as a colour
// attachment + sampled image on desktop GPUs.
pub(super) const HDR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

// Cap on runtime-cloned static draws (`clone_static_draw_object`). The
// clone descriptor pool is sized for this many (albedo, normal) sets at
// init. Editor-only: exhausting the pool only happens under
// `world.jsonl` hot-reload churn that adds 129+ new Props referencing
// existing meshes; the call returns an error past that. Mirrors
// `directx::context::MAX_CLONE_DRAWS`. Used by `clone_static_draw_object`,
// which is reached only through the bin's `cn debug` runtime-mutation path
// (dead in the FFI lib, live in the bin) -- hence the allow, matching DirectX.
#[allow(dead_code)]
pub(super) const MAX_CLONE_DRAWS: usize = 128;

// MAX_BLOOM_MIPS now lives in `crate::vulkan::post::bloom` (re-exported as
// `crate::vulkan::post::MAX_BLOOM_MIPS`).

// Cascaded-shadow-map resources, grouped off the flat `VkContext` field soup
// (mirrors the DirectX backend's `self.shadow`). The barrier executor resolves
// `shadow_map` through `build_barrier_registry`, so moving these fields behind
// one field left the parallel emit path untouched.
pub(super) struct VkShadow {
    pub(super) render_pass: OwnedRenderPass,
    pub(super) map: GpuImage,
    pub(super) map_size: u32,
    // One framebuffer per cascade slice. Empty when the shadow pass is disabled.
    pub(super) framebuffers: Vec<OwnedFramebuffer>,
    pub(super) pipeline: Option<OwnedPipeline>,
    pub(super) pipeline_layout: Option<OwnedPipelineLayout>,
    pub(super) global_set_layout: Option<OwnedSetLayout>,
    pub(super) global_sets: Vec<vk::DescriptorSet>,
    pub(super) sampler: OwnedSampler,
    pub(super) skinned_pipeline: Option<OwnedPipeline>,
    pub(super) skinned_pipeline_layout: Option<OwnedPipelineLayout>,
    // Per-frame-in-flight `ShadowUniforms` ring, persistently mapped. One slot
    // per frame: a single buffer would let this frame's cascade VPs overwrite
    // memory an in-flight frame is still sampling, which under `Hybrid` pairs a
    // freshly-jumped far-cascade VP with depth rasterized from the old one.
    pub(super) ubos: Vec<PooledBuffer>,
    // Carried CSM uniforms: skipped cascades keep the VP their slice was last
    // rendered with. Splits refresh every frame; per-cascade light VPs only when
    // `render_mask` includes that cascade. Written to this frame's `ubos` slot
    // each frame.
    pub(super) uniforms: ShadowUniforms,
    // World-space direction toward the first directional light, cached at init.
    // Per-frame CSM updates use this; refresh it when lights change for a moving
    // sun.
    pub(super) light_dir: [f32; 3],
    // Cascade re-render policy from GraphicsConfig.shadow_update. Hybrid
    // refreshes the near cascade every frame and the far cascades round-robin.
    pub(super) update: crate::assets::ShadowUpdate,
    // Shadow distance in world units (GraphicsConfig.shadow_distance), read by the
    // per-frame cascade-split computation and capped at the camera far plane.
    pub(super) distance: u32,
    // Active shadow cascade count, 1..=4 (GraphicsConfig.shadow_cascades). The
    // per-frame split + schedule read it; only the first `cascades` of the four
    // slots are rendered + sampled. Stored at init (applies at the next launch).
    pub(super) cascades: u32,
    // Round-robin clock + primed-set for the cascade schedule; advanced once per
    // frame in draw_frame.
    pub(super) scheduler: crate::gfx::shadow_schedule::ShadowCascadeScheduler,
    // Cascades re-rendered this frame (bit `i` = cascade `i`). Set in draw_frame
    // and read by encode_shadow_pass so the two agree on which slices to refresh
    // and which to leave intact.
    pub(super) render_mask: u32,
}

impl VkShadow {
    // Destroy every owned GPU object. Called from `VkContext::drop` after
    // `wait_idle`. The per-frame `global_sets` are freed with the shared
    // descriptor pool, so they are not destroyed here.
    pub(super) fn destroy(&mut self, _device: &VkDevice) {
        self.map = GpuImage::null();
        self.ubos.clear();
    }
}

// Spot shadow map resources: one depth array layer per shadow-casting spot
// light, plus the `SpotShadowData` buffer holding each slice's light-space
// projection. Local lights are static, so the slice assignment and every matrix
// are decided once at init and only the depth contents refresh. A world with no
// shadowed spot still gets a 1x1 fallback array and a one-element buffer, so
// the main pass's descriptors are always valid. Reuses the cascade pass's
// render pass, pipeline, and comparison sampler.
pub(super) struct VkSpotShadow {
    pub(super) map: GpuImage,
    // One framebuffer per shadowed spot; empty when the world has none.
    pub(super) framebuffers: Vec<OwnedFramebuffer>,
    pub(super) slice_size: u32,
    // `SpotShadowData` per slice, uploaded once at init.
    pub(super) data_buffer: PooledBuffer,

    // One `ShadowUniforms` per slice, each carrying that spot's matrix in
    // `light_vps[0]` so the shared shadow vertex shader renders a spot slice by
    // pushing cascade_idx = 0. Written once at init: the projections are fixed
    // for the world's lifetime, so unlike the cascade UBO this needs no
    // per-frame copy. One descriptor set per slice binds its own range.
    pub(super) ubo: PooledBuffer,

    pub(super) sets: Vec<vk::DescriptorSet>,
    pub(super) _descriptor_pool: OwnedDescriptorPool,
    // Round-robin clock + primed set, advanced once per frame in draw_frame.
    pub(super) scheduler: crate::gfx::spot_shadow::SpotShadowScheduler,
    // Slices re-rendered this frame (bit `i` = slice `i`).
    pub(super) render_mask: u32,
}

impl VkSpotShadow {
    // Slices actually handed out; the array layers, the framebuffers, and the
    // data buffer all carry exactly this many entries.
    pub(super) fn count(&self) -> u32 {
        self.framebuffers.len() as u32
    }

    // Advance the round-robin clock and record which slices re-render this
    // frame. A no-op (mask stays 0) when the world has no shadowed spot.
    pub(super) fn advance(&mut self, every_frame: bool) {
        let count = self.framebuffers.len();
        self.render_mask = self.scheduler.next_mask(every_frame, count);
    }

    // Destroy every owned GPU object. Called from `VkContext::drop` after
    // `wait_idle`; `sets` are freed with `descriptor_pool`.
    pub(super) fn destroy(&mut self, _device: &VkDevice) {
        self.map = GpuImage::null();
        self.data_buffer = PooledBuffer::null();
        self.ubo = PooledBuffer::null();
    }
}

// Rectangular area lights: the per-scene `AreaLightData` table indexed by
// `GpuLight.data_index`, plus the two LTC lookup tables the shading path
// samples. All three are static for the world's lifetime. The tables are
// scene-independent (fitted at build time), so they are uploaded even with no
// area light declared -- the shader simply never samples them.
pub(super) struct VkAreaLight {
    pub(super) buffer: PooledBuffer,
    pub(super) ltc_matrix: GpuImage,
    pub(super) ltc_magnitude: GpuImage,
    // Linear clamp-to-edge sampler for both tables.
    pub(super) sampler: OwnedSampler,
}

impl VkAreaLight {
    // Destroy every owned GPU object. Called from `VkContext::drop` after
    // `wait_idle`.
    pub(super) fn destroy(&mut self, _device: &VkDevice) {
        self.ltc_matrix = GpuImage::null();
        self.ltc_magnitude = GpuImage::null();
        self.buffer = PooledBuffer::null();
    }
}

// Skinned (skeletally animated) mesh resources, grouped off the flat `VkContext`
// field soup. All `None` / empty until `upload_skinned` runs; with no
// `SkinnedMesh` in the world every skinned pass is skipped. The joint matrices
// live in per-(frame, object) storage buffers bound through `joint_sets`: set 2
// for the main pass, set 1 for the shadow pass; the descriptor set layout is
// identical so one set serves both.
pub(super) struct VkSkinned {
    pub(super) pipeline: Option<OwnedPipeline>,
    pub(super) pipeline_layout: Option<OwnedPipelineLayout>,
    pub(super) joint_set_layout: Option<OwnedSetLayout>,
    pub(super) descriptor_pool: Option<OwnedDescriptorPool>,
    pub(super) vertex_buffer: PooledBuffer,
    pub(super) index_buffer: PooledBuffer,
    // Current byte sizes of the skinned VB / IB. Used by
    // `update_skinned_mesh_geometry` to bound-check the slot region the asset
    // hot-reload write lands in. Zero until `upload_skinned` runs.
    pub(super) vertex_buffer_bytes: u64,
    pub(super) index_buffer_bytes: u64,
    pub(super) draw_objects: Vec<SkinnedDrawObject>,
    // Per-object (albedo, normal) descriptor sets (set 1 for the main pass).
    pub(super) object_sets: Vec<vk::DescriptorSet>,
    // Per-(frame, object) joint storage buffers (host-mapped) + their
    // descriptor sets. Indexed [frame_idx][skinned_idx].
    pub(super) joint_buffers: Vec<Vec<PooledBuffer>>,
    pub(super) joint_sets: Vec<Vec<vk::DescriptorSet>>,
    // Current skinning matrices per skinned object, parallel to `draw_objects`.
    // Rewritten each frame by `update_skinned_pose`.
    pub(super) joint_matrices: Vec<Vec<[[f32; 4]; 4]>>,
    // GPU-driven main-pass skinning fold. `skin` is the `rt_skin` compute pipeline
    // (reused independently of RT) + its per-(frame, object) descriptor sets,
    // written once in `build_main_skin`. `deformed` is one storage+vertex buffer
    // per frame-in-flight holding this frame's posed 56-byte `Vertex`s (global
    // skinned indexing, so the draw uses `base_vertex = 0`); `encode_skin` writes
    // it each frame and the bindless main pass's 2nd indirect draw reads it. Both
    // `None`/empty until `upload_skinned` runs with the bindless cull path active.
    pub(super) skin: Option<super::raytrace::SkinPipeline>,
    pub(super) deformed: Vec<super::raytrace::DeviceBuffer>,
    // Morph targets, parallel to `draw_objects`. `morph_delta_unique` owns the
    // per-mesh dense target-major `MorphDelta` device buffers (deduped by source
    // `Arc`); `morph_delta_buffers[i]` is object `i`'s handle into them (null =
    // morphless). `morph_target_counts[i]` is its target count (0 = none).
    // `morph_weights[i]` is the object's current weights (empty without morphs),
    // rewritten by `update_morph_weights` and copied into the per-(frame, object)
    // host-mapped `morph_weight_buffers` ([frame_idx][skinned_idx], one f32 per
    // target) by `upload_morph_weights`. The weight buffers/memories/ptrs are
    // empty when no skinned object carries morphs. The skin descriptor sets'
    // morph bindings (3 = deltas, 4 = weights) are re-pointed in
    // `upload_skinned_morphs`.
    pub(super) morph_delta_unique: Vec<PooledBuffer>,
    pub(super) morph_delta_buffers: Vec<vk::Buffer>,
    pub(super) morph_target_counts: Vec<u32>,
    pub(super) morph_weights: Vec<Vec<f32>>,
    pub(super) morph_weight_buffers: Vec<Vec<PooledBuffer>>,
    // `false` until the deformed-vertex ring has been posed at least one full
    // frame. While false the GPU-driven G-buffer velocity binds the current
    // deformed buffer as the previous one (prev_pos == cur_pos), so an unposed
    // ring slot never feeds a garbage skinned motion vector on the first frame
    // (or after a runtime ring rebuild). Mirrors the legacy joint priming. Reset
    // by `build_main_skin` / `upload_skinned`. Atomic, not `Cell`: the G-buffer
    // pass encodes on a `jobs::pool()` rayon worker thread (the parallel per-pass
    // encoder shares `&self` across workers), so any interior mutation reachable
    // from `encode_pass_into` must be atomic, like `draw_calls_accum`.
    pub(super) deformed_primed: std::sync::atomic::AtomicBool,
}

impl VkSkinned {
    // Destroy every owned GPU object. Called from `VkContext::drop` after
    // `wait_idle`. The per-object `object_sets` and per-frame `joint_sets` are
    // freed with `descriptor_pool`, so they are not destroyed here.
    pub(super) fn destroy(&mut self, device: &VkDevice) {
        self.vertex_buffer = PooledBuffer::null();
        self.index_buffer = PooledBuffer::null();
        self.joint_buffers.clear();
        self.morph_delta_unique.clear();
        self.morph_weight_buffers.clear();
        // GPU-driven main-pass skinning resources.
        if let Some(skin) = self.skin.take() {
            skin.destroy(device);
        }
        self.deformed.clear();
    }
}

// Shared static vertex/index buffers plus the byte-range sub-allocators that
// carve streamed-mesh regions out of them, grouped off the flat `VkContext`
// field soup. Created at init and live for the context's lifetime; the
// streaming and geometry-rebuild paths swap the buffers in place.
pub(super) struct VkGeometry {
    pub(super) vertex_buffer: PooledBuffer,
    pub(super) index_buffer: PooledBuffer,
    // Byte-range sub-allocators for the streamed-mesh regions of the shared
    // vertex/index buffers. Empty until mesh streaming is active; `evict_mesh`
    // seeds them with each streamed draw's build-time region at init, then
    // `upload_mesh` / `evict_mesh` allocate and free byte ranges so a streamed
    // mesh lands wherever there is room.
    pub(super) mesh_vtx_alloc: crate::suballoc::range_alloc::RangeAllocator,
    pub(super) mesh_idx_alloc: crate::suballoc::range_alloc::RangeAllocator,
    // Current byte sizes of the shared vertex/index buffers. Tracked so
    // `setup_chunk_streaming` knows how much build-time geometry to copy when
    // it grows them.
    pub(super) vertex_buffer_bytes: u64,
    pub(super) index_buffer_bytes: u64,
}

impl VkGeometry {
    // Drop the shared vertex/index buffers (they retire through the
    // allocator). The range allocators and byte counts are plain CPU state.
    pub(super) fn destroy(&mut self) {
        self.vertex_buffer = PooledBuffer::null();
        self.index_buffer = PooledBuffer::null();
    }
}

// The main geometry-path descriptor set layouts plus the shared pool the
// per-frame sets are allocated from, grouped off the flat `VkContext` field
// soup. Global set 0 (camera / lights / shadow / IBL / SSAO), object set 1
// (per-draw albedo + normal), and the text-overlay set; the `*_sets` are
// allocated from `descriptor_pool` at init and freed with it. Post, instanced,
// chunk, clone, and skinned descriptors live in their own pools, not here.
pub(super) struct VkDescriptors {
    pub(super) global_set_layout: OwnedSetLayout,
    // Whether `global_set_layout` was created with
    // `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT`, so it budgets
    // against the update-after-bind sampler limit instead of Metal's 16-entry
    // per-stage table and costs the layouts that bind it nothing. Every pool that
    // allocates a global set (here, `planar.rs`, `probe.rs`) must declare the
    // matching flag. True on a sampler-constrained device (MoltenVK), false on
    // every desktop driver.
    pub(super) global_update_after_bind: bool,
    // Descriptor count of `global_set_layout`'s reflection-probe cube array
    // (binding 8), derived at init from the device's per-stage sampler limit by
    // `descriptor_layout::probe_cube_array_count`. Every later probe cube write,
    // re-rendered global set, and probe-shader recompile reads it from here so
    // they stay sized to the layout the pipelines were built against.
    pub(super) probe_cube_count: u32,
    pub(super) object_set_layout: OwnedSetLayout,
    pub(super) _text_set_layout: OwnedSetLayout,
    pub(super) _descriptor_pool: OwnedDescriptorPool,
    pub(super) global_sets: Vec<vk::DescriptorSet>,
    pub(super) object_sets: Vec<vk::DescriptorSet>,
    pub(super) text_atlas_sets: Vec<vk::DescriptorSet>,
}

impl VkDescriptors {
    // Destroy the shared descriptor pool (which frees every set allocated from
    // it: global_sets / object_sets / text_atlas_sets) and the three set
    // layouts. Called from `VkContext::drop` after `wait_idle`.
    pub(super) fn destroy(&self, _device: &VkDevice) {}
}

// Instanced-prop rendering: the pipeline + per-cluster material sets + the
// per-(frame, cluster) instance storage buffers and their descriptor sets,
// grouped off the flat `VkContext` field soup. All `None` / empty when the
// world declares no `InstancedProp` clusters. `clusters` holds the declared
// clusters (each with its per-instance transforms); `lod_buckets` is the
// per-frame LOD partition every instanced draw site shares.
pub(super) struct VkInstanced {
    // Instanced pipeline; None when no InstancedProp clusters were declared.
    pub(super) pipeline: Option<OwnedPipeline>,
    pub(super) pipeline_layout: Option<OwnedPipelineLayout>,
    pub(super) set_layout: Option<OwnedSetLayout>,
    pub(super) clusters: Vec<InstancedCluster>,
    // Per-cluster (albedo, normal) sets used by the instanced pipeline.
    // Indexed by cluster index. Empty when no clusters are declared.
    pub(super) object_sets: Vec<vk::DescriptorSet>,
    // Per-frame, per-cluster instance buffer descriptor sets bound to set=2.
    // Indexed [frame_idx][cluster_idx].
    pub(super) sets: Vec<Vec<vk::DescriptorSet>>,
    // Per-frame, per-cluster instance storage buffers (host-mapped).
    pub(super) buffers: Vec<Vec<PooledBuffer>>,
    // Per-cluster LOD-bucket partition for the current frame, indexed by
    // cluster index. Recomputed once per frame by `prepare_instanced_clusters`
    // (on `&mut self`, before the parallel pass fan-out) and consumed read-only
    // by every instanced draw site (main, shadow, SSR / SSAO / velocity
    // pre-passes) so all passes agree on the per-instance LOD pick and the
    // bucket-ordered byte layout uploaded into each cluster's instance SSBO.
    // Empty until the first frame / when no clusters are declared.
    pub(super) lod_buckets: Vec<Vec<InstancedLodBucket>>,
}

impl VkInstanced {
    // Destroy the pipeline, pipeline layout, instance set layout, and the
    // per-frame instance storage buffers + their mapped memory. Called from
    // `VkContext::drop` after `wait_idle`. The descriptor sets (`object_sets`,
    // `sets`) are freed with the shared descriptor pool, so they are not
    // destroyed here; `clusters` / `lod_buckets` are plain CPU state.
    pub(super) fn destroy(&mut self, _device: &VkDevice) {
        self.buffers.clear();
    }
}

// Streamed VoxelWorld chunk rendering resources, grouped off the flat
// `VkContext` field soup (mirrors the DirectX backend's `chunk_stream:
// ChunkStreamState`, though Vulkan needs the extra descriptor pool + set and
// the reload-tracking material slots where DX reuses stable SRV-heap slots).
// All `None` / empty until `setup_chunk_streaming` runs; with no streamed
// chunks every field stays inert.
pub(super) struct VkChunkStream {
    // Byte-range sub-allocators for the headroom region appended to the shared
    // vertex/index buffers, disjoint from the build-time geometry and the
    // mesh-streaming allocators.
    pub(super) vtx_alloc: crate::suballoc::range_alloc::RangeAllocator,
    pub(super) idx_alloc: crate::suballoc::range_alloc::RangeAllocator,
    // Dedicated pool + one shared (albedo, normal) descriptor set for streamed
    // chunks.
    pub(super) descriptor_pool: Option<OwnedDescriptorPool>,
    pub(super) object_set: Option<vk::DescriptorSet>,
    // Albedo / normal-map pool slots the shared chunk material samples, stored
    // (already clamped) so a streamed swap of either slot re-points `object_set`.
    pub(super) texture_slot: Option<usize>,
    pub(super) normal_map_slot: Option<usize>,
}

impl VkChunkStream {
    // Destroy the chunk descriptor pool (which frees `object_set`). Called from
    // `VkContext::drop` after `wait_idle`. The allocators, free-slot list, and
    // material slots are plain CPU state with nothing to free.
    pub(super) fn destroy(&self, _device: &VkDevice) {}
}

// GPU-driven cull + bindless static main pass (+ optional two-pass Hi-Z
// occlusion), grouped off the flat `VkContext` field soup. Mirrors the DirectX
// backend's `cull: CullState`. A compute kernel frustum/distance-tests the
// build-time static objects and writes one indirect draw per survivor; the
// bindless main pass issues the whole buffer with one indirect draw. All
// `Some` / non-empty only when the world uses the built-in bindless shader with
// build-time geometry; non-bindless shaders keep the legacy per-draw loop. Field
// names are kept verbatim (heterogeneous prefixes, no single cluster prefix to
// drop). The two-pass Hi-Z pyramid + its temporal state live here too. The
// legacy `main_pipeline` and the CPU `draw.bvh` are NOT part of this.
pub(super) struct VkCull {
    // Bindless static main pass. `Some` only on the built-in shader; `None`
    // keeps the legacy per-draw main pass. The bindless descriptor sets are
    // freed with the shared descriptor pool.
    pub(super) bindless_pipeline: Option<OwnedPipeline>,
    pub(super) bindless_pipeline_layout: Option<OwnedPipelineLayout>,
    pub(super) bindless_set_layout: Option<OwnedSetLayout>,
    // Descriptor count `bindless_set_layout`'s binding 1 was built with, and the
    // `{POOL_SIZE}` every pool-sized shader must compile against. 0 when the
    // bindless path is inactive. Every recompile reads this rather than
    // re-deriving from the texture table: a shader that declares fewer array
    // elements than the layout is legal Vulkan, so drift is silent and costs the
    // trailing flat-normal fallback slot.
    pub(super) bindless_pool_size: usize,
    // Whether `bindless_set_layout` was created with
    // `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT`, which every
    // descriptor pool that allocates it must declare in turn. Set on a
    // sampler-constrained device (MoltenVK) whose plain per-stage sampler budget
    // cannot seat the texture pool; false on every desktop driver.
    pub(super) bindless_update_after_bind: bool,
    // Material-referenced world shader pipelines, indexed by `shader_bucket - 1`
    // (bucket 0 is `bindless_pipeline`). Each renders its bucket's slice of the
    // GPU-culled command buffer through the shared bindless pipeline layout.
    // `None` marks a bucket whose Shader is not resident yet: its scene has not
    // pinned, so the pass skips those draws (see `world_shaders.rs`).
    pub(super) world_pipelines: Vec<Option<OwnedPipeline>>,
    // Commands reserved per shader-bucket region in the indirect buffers, fixed at
    // init to the record capacity the buffers were sized for. Bucket `b`'s region
    // starts at command `b * bucket_stride`.
    pub(super) bucket_stride: usize,
    // The engine's compiled bindless main-pass SPIR-V, retained so a bucket that
    // resolves to the engine default can build its pipeline without recompiling
    // the GLSL. Empty when the world authored its own main shader.
    pub(super) bindless_main_spv: (Vec<u8>, Vec<u8>),
    // One bindless descriptor set per frame-in-flight: binding 0 is that frame's
    // GpuObjectData storage buffer, binding 1 the shared texture pool.
    pub(super) bindless_sets: Vec<vk::DescriptorSet>,
    // Per-frame GpuObjectData storage buffers, persistently mapped; rebuilt each
    // frame from `draw.objects[..draw.n_objects]`.
    pub(super) object_buffers: Vec<PooledBuffer>,
    // Compute cull pipeline + its per-frame sets (bindings 0/1/2 = that frame's
    // object SSBO, draw-args SSBO, indirect-command SSBO). Sets are pool-freed.
    pub(super) cull_pipeline: Option<OwnedPipeline>,
    pub(super) cull_pipeline_layout: Option<OwnedPipelineLayout>,
    pub(super) cull_set_layout: Option<OwnedSetLayout>,
    pub(super) cull_sets: Vec<vk::DescriptorSet>,
    // Per-frame `GpuDrawArgs` storage buffers, persistently mapped.
    pub(super) draw_args_buffers: Vec<PooledBuffer>,
    // Per-frame indirect draw-command buffers the cull kernel writes and the
    // main pass consumes (`INDIRECT_BUFFER`). Device-local.
    pub(super) indirect_buffers: Vec<PooledBuffer>,
    // Per-frame per-object cull-status buffers (one u32 each): phase-1 writes,
    // phase-2 reads. Device-local storage.
    pub(super) cull_status_buffers: Vec<PooledBuffer>,
    // Two-pass Hi-Z occlusion (HizBuild -> Cull2 -> Main2). `occlusion_two_pass`
    // records the world's request; the live resources below are `Some` /
    // non-empty only when it AND the bindless cull path are active.
    pub(super) occlusion_two_pass: bool,
    // Phase-2 cull pipeline (same layout as `cull_pipeline`) + its per-frame
    // sets, allocated from `two_pass_pool`.
    pub(super) cull_pipeline_phase2: Option<OwnedPipeline>,
    pub(super) cull_sets2: Vec<vk::DescriptorSet>,
    pub(super) _two_pass_pool: Option<OwnedDescriptorPool>,
    // Per-frame second indirect draw-command buffers `Cull2` writes and `Main2`
    // consumes. Device-local.
    pub(super) indirect_buffers2: Vec<PooledBuffer>,
    // Phase-1 / phase-2 main render passes (render-pass-compatible with the
    // main-pass `framebuffers`).
    pub(super) main_render_pass_phase1: Option<OwnedRenderPass>,
    pub(super) main_render_pass_phase2: Option<OwnedRenderPass>,
    // Hi-Z occlusion culling. The depth-mip pyramid (built at end of frame
    // from this frame's main depth) + its build pipelines + the cull pipeline's
    // set 1 (`sampler2D` Hi-Z + per-frame `CullHizParams` UBO). `Some` exactly
    // when the GPU-cull pipeline is active (same gating as `cull_pipeline`):
    // the next frame's `Cull` kernel projects each AABB through the previous
    // frame's un-jittered VP and discards objects fully behind the pyramid.
    pub(super) hiz: Option<crate::vulkan::hiz::HiZResources>,
    // False on the first frame and immediately after a swapchain resize (no
    // valid pyramid yet); drives the cull UBO's `hiz_enabled` so the cull
    // kernel falls back to frustum + distance only until a pyramid at the
    // current resolution exists. Set true at the end of `record_frame` once a
    // build has run.
    pub(super) hiz_valid: bool,
    // Previous frame's un-jittered camera view-projection, fed to the Hi-Z cull
    // test. Updated every frame (independent of TAA, which keeps its own
    // `prev_view_proj`). The pyramid is reduced from depth rendered with the
    // jittered VP; the sub-pixel discrepancy is conservative, matching DirectX
    // / Metal which also project through the previous un-jittered VP.
    pub(super) hiz_prev_view_proj: [[f32; 4]; 4],
    // GPU-driven shadow pass. `shadow_cull_pipeline` is a frustum +
    // distance only cull kernel (`SHADOW_CULL`, no Hi-Z / status) over a lean
    // 3-SSBO set (objects + draw-args + this cascade's indirect-command buffer);
    // one dispatch per re-rendered cascade writes that cascade's indirect buffer.
    // `shadow_bindless_pipeline` is a depth-only graphics pipeline whose VS reads
    // `model` from the GpuObjectData SSBO (gl_InstanceIndex) and projects through
    // `light_vps[cascade_idx]` (a push constant); each cascade is then issued with
    // one `cmd_draw_indexed_indirect` (static+instance prefix) + one for the
    // skinned tail. `shadow_indirect_buffers` / `shadow_cull_sets` are indexed
    // [frame][cascade]. All `Some`/non-empty only when the bindless cull path is
    // active AND shadows are enabled.
    pub(super) shadow_cull_pipeline: Option<OwnedPipeline>,
    pub(super) shadow_cull_pipeline_layout: Option<OwnedPipelineLayout>,
    pub(super) _shadow_cull_set_layout: Option<OwnedSetLayout>,
    pub(super) shadow_cull_sets: Vec<Vec<vk::DescriptorSet>>,
    pub(super) shadow_bindless_pipeline: Option<OwnedPipeline>,
    pub(super) shadow_bindless_pipeline_layout: Option<OwnedPipelineLayout>,
    pub(super) shadow_indirect_buffers: Vec<Vec<PooledBuffer>>,
    // GPU-driven G-buffer pre-pass. A 3-MRT bindless pipeline whose VS
    // reads `model` + `roughness` from the GpuObjectData SSBO (gl_InstanceIndex)
    // and the previous-frame model from `prev_model_buffers`; the velocity history
    // for the skinned tail rides the previous-frame deformed buffer. The pass
    // reuses the main pass's `indirect_buffers` (camera frustum, no extra cull).
    // Set 0 (`gbuffer_set_layout`) = GbView UBO + prev_model SSBO; set 1 = the
    // shared bindless set. `gbuffer_sets` is one set 0 per frame; the per-frame
    // `prev_model_*` buffers are host-mapped (instance region init-written, static
    // + skinned regions rewritten each frame). All `Some`/non-empty only when the
    // bindless cull path is active AND the G-buffer is enabled.
    pub(super) gbuffer_bindless_pipeline: Option<OwnedPipeline>,
    pub(super) gbuffer_bindless_pipeline_layout: Option<OwnedPipelineLayout>,
    pub(super) _gbuffer_set_layout: Option<OwnedSetLayout>,
    pub(super) gbuffer_sets: Vec<vk::DescriptorSet>,
    pub(super) prev_model_buffers: Vec<PooledBuffer>,
}

impl VkCull {
    // Destroy every owned GPU object. Called from `VkContext::drop` after
    // `wait_idle`. The bindless / cull / phase-2 descriptor sets are freed with
    // the shared descriptor pool + `two_pass_pool`, so they are not destroyed
    // here. `occlusion_two_pass` is plain CPU state. Takes `&mut self` because
    // `HiZResources::destroy` nulls out its handles as it frees them.
    pub(super) fn destroy(&mut self, device: &VkDevice) {
        // Hi-Z occlusion resources (image + build pipelines + cull-read sets +
        // per-frame cull UBOs).
        if let Some(hiz) = &mut self.hiz {
            hiz.destroy(device);
        }
        // GPU-driven shadow pass. The per-(frame, cascade) `shadow_cull_sets`
        // are freed with the shared descriptor pool, so only the pipelines,
        // the set layout, and the per-cascade indirect buffers are destroyed.
        // GPU-driven G-buffer pre-pass. The per-frame `gbuffer_sets` are freed
        // with the shared descriptor pool, so only the pipeline, layout, set
        // layout, and the per-frame prev_model buffers are destroyed here.
        self.object_buffers.clear();
        self.draw_args_buffers.clear();
        self.indirect_buffers.clear();
        self.cull_status_buffers.clear();
        self.indirect_buffers2.clear();
        self.shadow_indirect_buffers.clear();
        self.prev_model_buffers.clear();
    }
}

// Per-frame-in-flight CPU/GPU synchronization primitives, grouped off the flat
// `VkContext` field soup. `image_available` + `in_flight` are one-per-frame-in-
// flight (`frames_in_flight` deep); `render_finished` is one-per-swapchain-image
// (its length tracks the swapchain, so a resize rebuilds it). The ring cursor
// (`current_frame`) and depth (`frames_in_flight`) stay flat on `VkContext`:
// they are read pervasively and are frame-pacing counters, not sync handles.
pub(super) struct VkFrameSync {
    // Signalled by `acquire_next_image`, waited on by that frame's submit.
    pub(super) image_available: Vec<vk::Semaphore>,
    // Signalled by the frame's submit, waited on by its present. Indexed by
    // swapchain image, so one per swapchain image (not per frame-in-flight).
    pub(super) render_finished: Vec<vk::Semaphore>,
    // Per-frame-in-flight submission fence; gates reuse of that slot's
    // resources (command buffers, mapped UBOs, per-pass pools).
    pub(super) in_flight: Vec<vk::Fence>,
}

impl VkFrameSync {
    // Destroy every owned semaphore + fence. Called from `VkContext::drop`
    // after `wait_idle`, so none are still in flight.
    pub(super) fn destroy(&self, device: &VkDevice) {
        // SAFETY: the handle was created from this device and is destroyed exactly once; the caller
        // has already waited for the device to go idle, so no submission still references it.
        unsafe {
            for &s in &self.image_available {
                device.destroy_semaphore(s, None);
            }
            for &s in &self.render_finished {
                device.destroy_semaphore(s, None);
            }
            for &f in &self.in_flight {
                device.destroy_fence(f, None);
            }
        }
    }
}

// Per-frame command pools + buffers, grouped off the flat `VkContext` field
// soup. Each frame's submission splits into three tiers: a "start" outer buffer
// (leading timestamp), one buffer per render-graph pass recorded in parallel
// (each from its own externally-synchronized pool), and an "end" outer buffer
// (Composite + post-graph work). `command_pool` also doubles as the shared
// one-shot pool for upload / layout-transition submits during resource
// creation. DX keeps the analogous allocators / lists flat on `DxContext`, so
// there is no DX sub-struct to mirror here.
pub(super) struct VkCommands {
    // Shared pool: allocates the per-frame "end" buffers below AND backs every
    // one-shot upload / layout-transition submit during resource creation.
    pub(super) command_pool: vk::CommandPool,
    // Per-frame outer "end" command buffer (one per frame-in-flight). Carries
    // the Composite pass + the inline end-of-frame Hi-Z build + the
    // shadow-cascade reset + the trailing timestamp. Submitted last in the
    // per-frame batch.
    pub(super) command_buffers: Vec<vk::CommandBuffer>,
    // Per-frame outer "start" command buffer (one per frame-in-flight): just
    // the leading timestamp-pool reset + TOP_OF_PIPE write. Submitted first so
    // the timestamp brackets the whole frame. From its own pool (timestamp
    // reset must precede every pass).
    pub(super) start_command_pools: Vec<vk::CommandPool>,
    pub(super) start_command_buffers: Vec<vk::CommandBuffer>,
    // Per-(frame, pass) command pools + primary command buffers for parallel
    // command-buffer recording: each non-composite render-graph pass records
    // into its own buffer on a `jobs::pool()` worker, then the whole frame is
    // submitted in graph order as one `vkQueueSubmit`. Vulkan command pools are
    // externally synchronized, so each (frame, pass) slot owns its own pool;
    // no two workers ever touch the same pool. Length `frames_in_flight *
    // PASS_COUNT`, indexed `frame_idx * PASS_COUNT + pass_id as usize`. Mirrors
    // the DirectX `pass_allocators` / `pass_cmd_lists` pool.
    pub(super) pass_command_pools: Vec<vk::CommandPool>,
    pub(super) pass_command_buffers: Vec<vk::CommandBuffer>,
}

impl VkCommands {
    // Destroy every command pool, which frees the buffers allocated from it.
    // Called from `VkContext::drop` after `wait_idle`.
    pub(super) fn destroy(&self, device: &VkDevice) {
        // SAFETY: the handle was created from this device and is destroyed exactly once; the caller
        // has already waited for the device to go idle, so no submission still references it.
        unsafe {
            // The shared pool (also frees the per-frame "end" buffers).
            device.destroy_command_pool(self.command_pool, None);
            // The parallel-recording pools (each frees its own buffer).
            for &pool in self
                .start_command_pools
                .iter()
                .chain(self.pass_command_pools.iter())
            {
                device.destroy_command_pool(pool, None);
            }
        }
    }
}

// The main-pass view + light uniform buffers, grouped off the flat `VkContext`
// field soup. `view_ubo_*` is one host-mapped buffer per frame-in-flight (the
// per-frame `ViewUniforms` write in `record_frame`); `light_ubo` is a single
// buffer uploaded once at init and bound into the object descriptor sets. NOTE
// the field names collide with the per-pass resource structs (decal / glass /
// raymarch / particle / gbuffer each own their own `view_ubo_*`), so accesses
// are always anchored on the `self.<field>` form, never a bare leading-dot.
pub(super) struct VkUniforms {
    // Per-frame-in-flight `ViewUniforms` UBO (camera + IBL params), persistently
    // mapped. `record_frame` memcpys this frame's view into its slot.
    pub(super) view_ubo_buffers: Vec<PooledBuffer>,
    // Per-frame-in-flight `ProbeSet` UBO (reflection-probe count + per-probe
    // parallax boxes), bound at global set 0 binding 7, persistently mapped.
    // `record_frame` memcpys `self.probe.set` into this frame's slot each
    // frame; it stays `EMPTY` (count 0 = sky reflection) until a probe bakes.
    pub(super) probe_set_ubo_buffers: Vec<PooledBuffer>,
    // Single `LightUniforms` UBO, uploaded once at init and bound into every
    // object descriptor set.
    pub(super) light_ubo: PooledBuffer,
    // Single per-scene local-light storage buffer (SSBO), uploaded once at init
    // and bound at global set 0 binding 9. Static like `light_ubo` (never
    // rewritten per-frame).
    pub(super) local_light_buffer: PooledBuffer,
    // Byte size of `local_light_buffer`, kept so passes that rebind the global
    // set from `ctx` (probe bake) can set the SSBO descriptor range without the
    // original `local_lights` slice.
    pub(super) local_light_size: u64,
    // CPU-side copy of the values in `light_ubo`, kept so a live Ambient-slider
    // change can mutate `ambient_intensity` and re-upload. The light UBO is a
    // single (not per-frame) buffer, so `set_ambient_intensity` `wait_idle`s
    // before the rewrite to avoid racing an in-flight read; ambient changes only
    // on a slider drag, so the stall is rare.
    pub(super) light_uniforms: crate::gfx::render_types::LightUniforms,
}

impl VkUniforms {
    // Drop the per-frame view UBOs, the light UBO, and the local-light SSBO
    // (they retire through the allocator).
    pub(super) fn destroy(&mut self) {
        self.view_ubo_buffers.clear();
        self.probe_set_ubo_buffers.clear();
        self.light_ubo = PooledBuffer::null();
        self.local_light_buffer = PooledBuffer::null();
    }
}

// Projected decals. `resources` (pipeline + unit-cube buffers + per-frame
// uniforms + per-decal albedo sets) is always built so runtime `add_decal`
// works from a world that started empty; the encoder simply skips when every
// slot is `None` or every live decal culls. `records` and `free_slots` mirror
// Metal / DirectX's freelist pattern so id reuse stays bounded.
pub(super) struct DecalState {
    pub resources: Option<crate::vulkan::decal::DecalResources>,
    pub records: Vec<Option<crate::gfx::decal::DecalRecord>>,
    pub free_slots: Vec<usize>,
}

// Volumetric fog. `resources` is `Some` only when the world declared a
// `VolumetricFog` asset; with none it and `settings` both stay `None` and the
// fog pass is skipped entirely. The settings are cached so the per-frame
// encoder can build its `FogParams` without re-resolving the asset. `sun_dir` /
// `sun_color` mirror the first directional light captured at init: the Vulkan
// backend uploads `LightUniforms` once, so the sun is fixed.
pub(super) struct FogState {
    pub resources: Option<crate::vulkan::fog::FogResources>,
    pub settings: Option<crate::gfx::volumetric_fog::FogSettings>,
    pub sun_dir: [f32; 3],
    pub sun_color: [f32; 3],
}

// Auto-exposure (EV adaptation). `resources` is `Some` only when
// `PostProcessConfig.auto_exposure` is enabled; it holds the build + average
// compute pipelines, histogram + output buffers, and the per-frame readback
// buffers. `state` carries the EMA target, `settings` the clamped tunables,
// `bias_ev` the authored EV bias added to the target, and `last_elapsed` the
// previous frame's elapsed time used to derive `dt` for the EMA.
pub(super) struct AutoExposureState {
    pub resources: Option<crate::vulkan::auto_exposure::AutoExposureResources>,
    pub settings: Option<crate::gfx::auto_exposure::AutoExposureSettings>,
    pub state: Option<crate::gfx::auto_exposure::AutoExposureState>,
    pub bias_ev: f32,
    pub last_elapsed: f32,
}

// Built-in shader hot reload. `enabled` is true only under `cn debug`: it
// routes every built-in GLSL source resolve through `pipeline::shader_source`'s
// disk-first path and gates the `vulkan/shaders/` filesystem watcher. Under
// `cn run` the `include_str!`-baked GLSL is the only source the binary sees.
// `reload_pending` is the atomic flag set by the `notify` watcher or the debug
// WS `reload-shaders` command, polled at the top of `draw_frame` to trigger a
// pipeline rebuild. `watcher` is the live `notify` handle held purely for
// lifetime; dropping it stops the watcher. Both are `Some` only when `enabled`.
pub(super) struct HotReloadState {
    pub enabled: bool,
    pub reload_pending: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    #[allow(dead_code)]
    pub watcher: Option<crate::vulkan::hot_reload::WatcherHandle>,
}

// GPU-compute particle system. `resources` (pipelines + per-frame view UBO +
// descriptor pool + framebuffers) is built only when the world declared at
// least one `ParticleEmitter` (or when runtime `add_particle_emitter` fires);
// the encoder is a no-op otherwise. `records` and `emitter_state` mirror Metal /
// DirectX's parallel-vec freelist pattern so id reuse stays bounded.
// `last_elapsed` + `frame_index` live in `Cell`s because `encode_particles` is
// reached through `&self` from the graph executor (per-frame mutable state has
// to be interior-mut).
pub(super) struct ParticleState {
    pub resources: Option<crate::vulkan::particle::ParticleResources>,
    pub records: Vec<Option<crate::gfx::particles::ParticleEmitterRecord>>,
    pub emitter_state: Vec<Option<crate::vulkan::particle::ParticleEmitterGpuState>>,
    pub free_slots: Vec<usize>,
    pub last_elapsed: std::cell::Cell<f32>,
    pub frame_index: std::cell::Cell<u32>,
}

// Composite (post-process) pass: tonemaps the HDR resolve image onto the
// swapchain, with the text overlay drawn here too, post-tonemap. The
// framebuffers are one per swapchain image; `sets` is one per frame-in-flight
// slot, binding the matching HDR resolve image (binding 0), bloom mip 0
// (binding 1), and the 3D colour LUT (binding 2). `sampler` is the linear-clamp
// sampler the composite + bloom shaders read HDR images with, the colour LUT
// included.
pub(super) struct CompositeState {
    pub render_pass: OwnedRenderPass,
    pub framebuffers: Vec<OwnedFramebuffer>,
    pub pipeline: OwnedPipeline,
    pub pipeline_layout: OwnedPipelineLayout,
    pub _set_layout: OwnedSetLayout,
    pub sets: Vec<vk::DescriptorSet>,
    pub sampler: OwnedSampler,
}

// Bloom chain. The mips, framebuffers, and input descriptor sets are all
// per-frame-in-flight slot (outer Vec): concurrent slots must not share a bloom
// target. Render passes / pipelines / layouts are slot-agnostic. `mips` is
// `[frame][mip]`, largest first, with mip 0 at half the HDR resolution;
// `mip_extents` is shared across frame slots. `blend_framebuffers` has one fewer
// entry than `write_framebuffers` (the smallest mip is never upsampled into).
// `input_sets` is `[frame][input]`: input 0 binds the HDR resolve image, input
// `1 + m` binds bloom mip `m`.
pub(super) struct BloomState {
    pub write_pass: OwnedRenderPass,
    pub blend_pass: OwnedRenderPass,
    pub pipeline_prefilter: OwnedPipeline,
    pub pipeline_downsample: OwnedPipeline,
    pub pipeline_upsample: OwnedPipeline,
    pub pipeline_layout: OwnedPipelineLayout,
    pub set_layout: OwnedSetLayout,
    pub descriptor_pool: OwnedDescriptorPool,
    pub mips: Vec<Vec<GpuImage>>,
    pub mip_extents: Vec<vk::Extent2D>,
    pub write_framebuffers: Vec<Vec<OwnedFramebuffer>>,
    pub blend_framebuffers: Vec<Vec<OwnedFramebuffer>>,
    pub input_sets: Vec<Vec<vk::DescriptorSet>>,
}

// HUD text pass: the glyph atlases, the pipeline (`None` until the first frame
// that publishes text), its layout, the sampler held for lifetime, and the
// per-frame-slot persistent upload buffers for transient text geometry. Each
// upload slot's cursor resets and its buffer grows inside the ring's `reserve`,
// which the composite pass calls once the frame fence confirms the GPU is done
// with that slot.
pub(super) struct TextState {
    pub atlas_textures: Vec<GpuImage>,
    pub pipeline: Option<OwnedPipeline>,
    pub pipeline_layout: OwnedPipelineLayout,
    pub _sampler: OwnedSampler,
    pub upload: super::upload_ring::UploadRing,
}

// Descriptor state for `clone_static_draw_object`. `descriptor_pool` is
// pre-allocated at init regardless of whether any clone exists yet, holding up
// to `MAX_CLONE_DRAWS` per-object (albedo, normal) sets so an asset hot-reload
// that adds a new authored Prop referencing an existing mesh / model can wire
// its descriptors without growing any other pool. `object_sets` is indexed by
// clone offset; an offset in `free_offsets` is allocated but unreferenced, ready
// for the next clone to reuse (re-pointed only if its textures differ).
// `slot_by_draw_idx` is the `draw_idx -> clone_offset` lookup the legacy main
// pass uses to pick the right set for an entry past `n_objects` (chunks fall
// through to `chunk_object_set`). `texture_slots` / `normal_map_slots` are
// parallel to `object_sets` and read by `rewrite_texture_slot` so a streamed
// swap repoints the matching clone sets.
pub(super) struct CloneState {
    pub descriptor_pool: Option<OwnedDescriptorPool>,
    pub object_sets: Vec<vk::DescriptorSet>,
    pub free_offsets: Vec<usize>,
    pub slot_by_draw_idx: std::collections::HashMap<usize, usize>,
    pub texture_slots: Vec<usize>,
    pub normal_map_slots: Vec<usize>,
}

// The scene draw list plus the CPU cull inputs derived from it, and the record
// counts that partition the GPU-driven bindless cull buffers.
pub(super) struct DrawState {
    pub objects: Vec<DrawObject>,
    pub bvh: crate::gfx::bvh::Bvh,
    pub always: Vec<u32>,
    // Parallel to `objects`: true where that slot is a member of `always`, so
    // `ensure_always_draw` adds a recycled slot at most once.
    pub always_member: Vec<bool>,
    // Per-frame scratch for the legacy CPU draw path's visible set (BVH-culled
    // cullables + `always` fallback). `mem::take`d at the top of record_frame
    // and returned at the bottom so the heap allocation is reused across frames
    // instead of `Vec::with_capacity`'d each tick.
    pub visible_scratch: Vec<u32>,
    // The last compiled frame graph, keyed by the `FrameGraphInputs` it was
    // built from. `build_frame_graph` is a pure function of those inputs (which
    // change only when a feature toggles or a target resizes), so a frame whose
    // inputs match the cached key reuses the compiled graph instead of
    // rebuilding it. Taken out during `execute_graph` (which needs `&mut self`)
    // and put back after, so a steady scene compiles the graph once.
    pub graph_cache: Option<(
        crate::gfx::render_graph::FrameGraphInputs,
        crate::gfx::render_graph::CompiledGraph,
    )>,
    // Build-time `objects` count. Streamed chunks are appended past this, so a
    // draw index >= `n_objects` identifies a chunk -- which binds the shared
    // `chunk_object_set` rather than a per-object descriptor set.
    pub n_objects: usize,
    // Instanced-cluster instances folded into the GPU-driven bindless cull
    // buffers as per-object `GpuObjectData` records after the `n_objects`
    // static records (so the cull kernel tests each instance independently).
    // 0 when the world has no instanced props or the bindless pass is inactive.
    // `cull_count() == n_objects + n_instances`. See `gfx::render_types`.
    pub n_instances: usize,
    // Streamed-chunk record reserve folded into the GPU-driven bindless cull
    // buffers BETWEEN the instances and the skinned tail: the buffers reserve
    // `[n_objects + n_instances, +n_chunk)` at init (capacity = the worst-case
    // resident chunk window). Resident chunks pack into this region each frame
    // and are drawn by the static+instance prefix indirect draw (chunk geometry
    // already lives in the shared VB/IB); the unused tail is disabled. Fixed at
    // init, 0 for a non-voxel world. Mirrors `DxContext.n_chunk`.
    pub n_chunk: usize,
    // Skinned draw objects folded into the GPU-driven bindless cull buffers as
    // `GpuObjectData` / `GpuDrawArgs` records after the instance records (at
    // `n_objects + n_instances + k`), drawn as rigid deformed geometry by the
    // main pass's 2nd indirect draw against the per-frame deformed-vertex
    // buffer. The cull buffers reserve these slots at init (capacity threaded
    // through `new`); this count is set in `upload_skinned` once the skin fold
    // is built, so it stays 0 (and `cull_count()` excludes the reserved tail)
    // when no skinned mesh loads or the bindless pass is inactive.
    pub n_skinned: usize,
}

// The frame's view state, snapped from `FrameParams` at the top of `draw_frame`.
pub(super) struct ViewState {
    pub clear_color: [f32; 4],
    // Scene-transition fade to black in [0, 1], applied in the composite pass.
    // Backend-owned rather than a `PostProcessParams` field so a settings push
    // cannot reset an in-flight fade, and kept out of `clear_color` so it fades
    // the whole image, not just the pixels no geometry covers.
    pub scene_fade: f32,
    // The viewport view mode: the main passes read it for the wireframe
    // pipeline variant and the unlit shade flag, the composite for its channel
    // visualization.
    pub mode: concinnity_core::gfx::view_modes::ViewMode,
    // The frame's show flags; the graph-input seeding in draw.rs masks with both
    // these and `mode`.
    pub show: concinnity_core::gfx::view_modes::ShowFlags,
    // The frame's camera far plane, for the composite's depth-channel
    // normalization.
    pub far: f32,
    pub matrix: [[f32; 4]; 4],
}

// Scene-captured reflection probes and the staggered bake that fills them,
// driven each frame by `bake_pending_probes` (the shared
// `reflection_probe::next_bake_action` transition table). Mirrors DirectX /
// Metal.
pub(super) struct ProbeState {
    // Placements (declared `ReflectionProbe`s or an auto-seeded grid), supplied
    // once after construction via `set_reflection_probes`. The cube capture that
    // bakes one prefiltered cube per placement runs across later frames; held
    // here so that capture can walk them.
    #[allow(dead_code)] // consumed by the probe capture pass (next slice).
    pub placements: Vec<crate::gfx::reflection_probe::ProbePlacement>,
    // The probe set (count + per-probe parallax boxes) bound to the forward /
    // SSR / RT shaders. `EMPTY` (count 0 = sky reflection) until the staggered
    // capture bakes cubes and installs them; each install bumps the count.
    pub set: concinnity_render::uniforms::ProbeSet,
    // Baked prefilter cubes, one per installed probe, parallel to
    // `set.probes[..set.count]`. Distinct from `env_map`; sampled only by the
    // specular reflection term once the capture installs them. Grows as the
    // staggered bake installs each probe. Destroyed in `Drop`.
    pub maps: Vec<GpuImage>,
    // Hands out placements in order; at most one probe is `rendering` (six faces
    // submitting one per frame, on per-face fences) and one `converting` (its
    // faces read back, the prefilter convolution running off the render thread).
    pub bake_queue: crate::gfx::reflection_probe::ProbeBakeQueue,
    pub rendering: Option<super::probe::RenderingBake>,
    pub converting: Option<super::probe::ConvertingBake>,
}

// Stall-free texture streaming. A streamed slot swap replaces `textures[slot]`
// immediately but cannot rewrite the per-frame bindless pool descriptors while
// their frames are pending; `pool_rewrites` carries the slot to each frame
// slot's copy right after its fence wait (`apply_streamed_texture_rewrites`).
// The replaced image and the upload's transient resources are parked on
// `retires` against the monotonic `frame` tick and freed `frames_in_flight + 1`
// ticks later: by then every pool copy has been re-pointed, every frame recorded
// against the old view has retired, and the tick's fence wait covers the upload
// submission itself (a swap lands between frames, after the previous frame's
// submit, so a frame-slot-keyed drain would free it too soon).
pub(super) struct StreamState {
    pub pool_rewrites: crate::gfx::slot_rewrites::SlotRewriteQueue,
    pub frame: u64,
    pub retires: Vec<StreamedUploadRetire>,
}

// The swapchain and the per-image state derived from it. `last_present_index`
// is the most recently presented image, or `None` before the first present /
// right after a rebuild: the `screenshot` debug command reads that image back,
// and `None` makes a too-early capture a clean error rather than a read of an
// unrendered image.
pub(super) struct SwapchainState {
    pub loader: ash::khr::swapchain::Device,
    pub handle: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub last_present_index: Option<u32>,
}

//  Public struct

pub(crate) struct VkContext {
    // Vulkan core
    pub(super) instance: ash::Instance,
    // Owns the logical device: the instance and the entry above it stay alive
    // for as long as it does, and every Vulkan object the backend owns retires
    // through its queue. Derefs to `ash::Device`, so recording and querying
    // through it read the same as before.
    pub(super) device: super::owned::VkDevice,
    pub(super) physical_device: vk::PhysicalDevice,
    pub(super) surface: vk::SurfaceKHR,
    pub(super) surface_loader: ash::khr::surface::Instance,
    pub(super) graphics_queue: vk::Queue,
    pub(super) present_queue: vk::Queue,
    pub(super) graphics_family: u32,
    // The device allocator every pooled buffer / image is placed through. Ticked
    // once per frame in `draw_frame`; drained in Drop after every pooled holder
    // has been torn down.
    pub(super) alloc: super::allocator::DeviceAllocator,

    // Swapchain. See [`SwapchainState`].
    pub(super) swapchain: SwapchainState,
    // Resolution the 3D scene is rendered at. Equals `swapchain.extent` unless
    // temporal upscaling is active, in which case it is
    // `round(swapchain.extent * upscale_scale)` and an FSR pass reconstructs the
    // swapchain-resolution image. Every off-screen scene pass (main, velocity,
    // SSR, SSAO, decals, fog, raymarch, glass, particles, auto-exposure, Hi-Z)
    // sizes its targets + viewports to this; bloom / composite / swapchain stay
    // at `swapchain.extent` (display resolution).
    pub(super) render_extent: vk::Extent2D,

    // Render passes
    pub(super) main_render_pass: OwnedRenderPass,
    // Composite (post-process) pass. See [`CompositeState`].
    pub(super) composite: CompositeState,

    // Multisampling
    pub(super) msaa_samples: vk::SampleCountFlags,

    // Off-screen HDR attachments, one set per frame-in-flight slot (indexed by
    // `current_frame`). The main pass renders into these; the composite pass
    // samples `hdr_resolve_images`.
    pub(super) color_images: Vec<GpuImage>, // MSAA HDR colour; empty when msaa == 1
    pub(super) depth_images: Vec<GpuImage>, // MSAA depth
    pub(super) hdr_resolve_images: Vec<GpuImage>, // single-sample HDR resolve target

    // Main-pass framebuffers (one per frame-in-flight slot): HDR colour +
    // depth (+ resolve when multisampled).
    pub(super) framebuffers: Vec<OwnedFramebuffer>,

    // Cascaded shadow map + its pipelines, framebuffers, UBO, and sampler.
    pub(super) shadow: VkShadow,

    // Spot shadow map resources. See [`VkSpotShadow`].
    pub(super) spot_shadow: VkSpotShadow,

    // Rectangular area-light resources. See [`VkAreaLight`].
    pub(super) area_light: VkAreaLight,

    // Shared texture pool: every texture (albedo, normal map, emissive/ORM,
    // terrain secondary) lives here once at its handle, matching DX/Metal.
    pub(super) textures: Vec<GpuImage>,
    // Holds only the flat-normal fallback a normal-less draw samples (its pool
    // slot is one past the last real texture); real normal maps are in `textures`.
    pub(super) normal_map_textures: Vec<GpuImage>,

    // Samplers
    pub(super) linear_sampler: OwnedSampler,

    // Pipelines
    pub(super) main_pipeline: OwnedPipeline,
    pub(super) main_pipeline_layout: OwnedPipelineLayout,
    // GPU-driven cull + bindless static main pass + two-pass Hi-Z occlusion
    // (pyramid + temporal state). See `VkCull`.
    pub(super) cull: VkCull,
    // Clustered light binning: the compute pipeline (built only when the world
    // has local lights), the per-cluster light-index buffer, and the
    // `ClusterParams` UBOs. See `VkLightCull`.
    pub(super) light_cull: super::light_cull::VkLightCull,

    // HUD text pass. See [`TextState`].
    pub(super) text: TextState,

    // 3D colour-grading LUT sampled in the composite pass. Holds the declared
    // `ColorLut` payload, or a 2x2x2 identity LUT when the world declares none.
    // Resolution-independent, so it is never rebuilt on swapchain resize.
    pub(super) color_lut: GpuImage,

    // Bloom chain. See [`BloomState`].
    pub(super) bloom: BloomState,
    // Post-process tunables (bloom intensity / threshold / knee, exposure,
    // vignette). Drives whether the bloom passes run and feeds the composite
    // + bloom-prefilter push constants.
    pub(super) post_process: crate::gfx::render_types::PostProcessParams,

    // Temporal anti-aliasing resources. `Some` only when the world's
    // `PostProcessConfig` set `taa: true`; `None` skips the velocity pre-pass
    // and history resolve entirely (and the projection jitter with them).
    // Also forced `Some` when temporal upscaling is on (FSR consumes the
    // velocity pre-pass's motion + depth), in which case the TAA *resolve* is
    // dropped from the graph and `Upscale` runs in its slot.
    pub(super) taa: Option<TaaResources>,

    // Temporal upscaling (FSR / DLSS / XeSS, behind `VkUpscaleBackend`). `Some`
    // only when the world's `PostProcessConfig` set `temporal_upscaling: true`
    // AND a backend resolved + built; `None` renders at native resolution
    // (`render_extent == swapchain.extent`). When `Some`, the scene renders at
    // the reduced `render_extent` and this pass reconstructs the swapchain
    // resolution; bloom + composite sample its output.
    pub(super) upscale: Option<Box<dyn VkUpscaleBackend>>,

    // The upscaler backend the world requested (`PostProcessConfig.upscale_backend`).
    // Kept so a swapchain resize rebuilds the same backend via `build_upscaler`
    // (the DLSS / XeSS device extensions are fixed at device creation, so the
    // resize must re-resolve to the same first choice; it does, deterministically).
    pub(super) upscale_requested: crate::assets::UpscalerBackend,

    // Screen-space ambient occlusion (GTAO) resources. `Some` only when the
    // world's `PostProcessConfig` set `ssao: true`; `None` binds the
    // `ssao_white` 1×1 fallback at set 0 binding 6 so the main pass's SSAO
    // multiplier is a constant 1.0.
    pub(super) ssao: Option<SsaoResources>,
    // 1×1 white fallback bound at set 0 binding 6 when SSAO is off.
    pub(super) ssao_white: GpuImage,

    // Backing store for the render graph's transient images (the resources the
    // aliasing planner manages). Owns each managed transient's image + memory;
    // features read them back by label and the executor's barrier registry
    // resolves them the same way. Rebuilt on swapchain resize. Today it manages
    // `ao_output`; the set grows as more transients migrate off their feature
    // structs.
    pub(super) transient_pool: super::transient_pool::TransientImagePool,

    // Screen-space reflections. `Some` when the world's `PostProcessConfig`
    // set `ssr: true` *or* selected `indirect_lighting: ssgi` (SSGI reuses the
    // SSR depth + normal pre-pass G-buffer, so the pre-pass half is built
    // whenever either is on). When on, the bloom prefilter / composite / TAA
    // scene input descriptors are re-pointed at `SsrResources::output` only when
    // `ssr_resolve_active` is true (a SSGI-only build leaves the resolve off).
    pub(super) ssr: Option<SsrResources>,
    // True when the SSR *resolve* (the reflection compositing half) should run
    // and own the post-stack scene image. False for a SSGI-only build, where
    // `ssr` exists for the G-buffer but the post stack samples `hdr_resolve`
    // directly (SSGI has already composited into it). Mirrors DirectX's
    // `scene_srv_for_post` gating on `s.resolve.as_ref()`.
    pub(super) ssr_resolve_active: bool,

    // Roughness-aware reflection composite. `Some` whenever a reflection path owns
    // the post-stack scene image (the SSR resolve is active OR RT reflections are
    // active). Both resolves write reflected radiance + weight into their output
    // target, then this blurs by roughness and composites over the scene into
    // `reflection_composite.output` -- the scene image the post stack consumes in
    // place of the raw resolve output. Mirrors `DxContext::reflection_composite`.
    pub(super) reflection_composite: Option<ReflectionCompositeResources>,

    // Screen-space global illumination. `Some` only when the world's
    // `PostProcessConfig` selected `indirect_lighting: ssgi`. The gather +
    // composite run on the hdr_resolve RMW chain after the main pass, reusing
    // `ssr`'s pre-pass G-buffer.
    pub(super) ssgi: Option<SsgiResources>,

    // Unified geometry G-buffer pre-pass. `Some` whenever any screen-space
    // consumer of the merged buffer is on (SSR resolve OR SSGI OR RT OR SSAO OR
    // velocity for TAA / upscale): one jittered traversal rasterises the
    // normal+depth / roughness / velocity MRT every reader samples, replacing
    // the separate SSR / SSAO / velocity pre-passes (the `PassId::GBufferPrepass`
    // node). Mirrors `DxContext::gbuffer`.
    pub(super) gbuffer: Option<GbufferResources>,

    // Hardware ray-traced reflections (`VK_KHR_ray_query`). `rt_reflections` (the
    // fullscreen inline-`rayQueryEXT` pass + its output target) and `rt_accel`
    // (the scene BLAS / TLAS + geometry table) are both `Some` only when the
    // world set `ray_traced_reflections: true`, the GPU exposed the ray-query
    // extensions, and the acceleration-structure build succeeded; otherwise both
    // stay `None` and the graph falls back to `SsrResolve`. Like SSGI, RT reuses
    // the SSR depth + normal + roughness pre-pass G-buffer (so `ssr` is built
    // whenever RT is on), and it replaces the SSR *resolve* in the frame graph:
    // when `rt_reflections_active()` the post stack samples `rt_reflections.output`
    // (RT takes precedence over SSR, which stays the non-RT-GPU fallback).
    pub(super) rt_reflections: Option<RtReflectionsResources>,
    pub(super) rt_accel: Option<crate::vulkan::raytrace::RtAccelData>,
    // How the TLAS is kept current when props move (`CN_RT_DYNAMIC`); read by the
    // per-frame `rt_dynamic_update`. Inert when `rt_accel` is `None`.
    pub(super) rt_dynamic_mode: crate::vulkan::raytrace::RtDynamicMode,
    // Set when a runtime change altered the RT-relevant draw set (a cloned prop, a
    // streamed chunk added/removed) since the last update. Consumed once per frame
    // by `rt_dynamic_update`, which folds the change into the BLAS head
    // (`RtAccelData::refresh_topology`) -- reusing every unchanged BLAS and building
    // only the new ones -- rather than ignoring it (the `Auto` dirty check only
    // watches transforms of the prior set) or rebuilding every BLAS.
    pub(super) rt_topology_dirty: bool,
    // Whether the device is RT-capable (the ray-query extensions + features were
    // enabled at device creation, and XeSS is not active). Enabled whenever
    // capable -- independent of whether RT is on at launch -- so a live
    // `apply_quality_settings` toggle can bring RT up at runtime (a device
    // extension cannot be enabled after `create_device`). Read by `upload_skinned`
    // to add the AS-build / storage / device-address flags to the skinned VB/IB
    // whenever capable (mirroring how the static VB/IB gate their RT flags at
    // init), and by the RT toggle to reject an enable on an incapable device.
    pub(super) rt_capable: bool,
    // Whether `descriptorBindingSampledImageUpdateAfterBind` was enabled at device
    // creation, letting the bindless texture pool's set layout declare itself
    // update-after-bind and budget against the far larger update-after-bind
    // sampler limit. Only true on a sampler-constrained device (MoltenVK); every
    // desktop driver keeps the plain layout. Carried on the context so a live
    // editor reload rebuilds the same layout the running device was created for.
    pub(super) update_after_bind: bool,
    // Total static vertices uploaded at init (the shared VB element count). The
    // acceleration-structure build needs it to size the hit-shader vertex SSBO;
    // there is no separate count field, so it is captured here for a live RT
    // build. Static-geometry rebuilds are not reflected (a pre-existing RT
    // topology limitation).
    pub(super) rt_static_vertex_count: usize,

    // Projected decals. See [`DecalState`].
    pub(super) decal: DecalState,
    // World-space line pass state: the resources, built on the first frame
    // that publishes lines. See [`crate::vulkan::line::LineState`].
    pub(super) lines: crate::vulkan::line::LineState,

    // Volumetric fog. See [`FogState`].
    pub(super) fog: FogState,

    // Raymarched SDF volumes. `Some` only when the world declared at least one
    // `SdfVolume` whose `fragment_shader` is a `.glsl` payload; the `Raymarch`
    // pass is omitted from the frame graph otherwise. Built at init; the encoder
    // composites each visible volume into the scene between `AutoExposure` and
    // `Decals`. While present, the main pass switches to a STORE-colour render
    // pass (MSAA) so this pass can load + re-resolve the multisampled colour.
    pub(super) raymarch: Option<crate::vulkan::raymarch::RaymarchResources>,

    // Translucent glass panels: the generic producer for the shared
    // `PassId::Transparent` slot. `Some` only when the world declared any
    // `GlassPanel`; with none the field stays `None` and the transparent pass
    // is omitted from the frame graph (gated on `glass.any_visible()`). Built
    // at init; the encoder draws the panels back-to-front over the post-SSR
    // scene between `SsrResolve` and `TaaResolve`. Water is a separate
    // (Metal-only) producer not ported here. Mirrors `src/directx/glass.rs`.
    pub(super) glass: Option<crate::vulkan::glass::GlassResources>,

    // Planar reflections for flat glass panes: one render-resolution mirror render
    // per distinct reflector plane, sampled projectively by the glass pass. `Some`
    // only when the world declared glass panes assigned to a planar slot. Mirrors
    // `src/directx/planar.rs`.
    pub(super) planar_reflection: Option<crate::vulkan::planar::PlanarReflectionSet>,

    // Resolved swapchain colour-output mode, selected when the world's
    // `PostProcessConfig.hdr_display` was on AND the surface advertised a
    // matching HDR colour space via the `VK_EXT_swapchain_colorspace` instance
    // extension. Two HDR flavours: `HdrEncoding::ExtendedLinear` runs the
    // swapchain in `R16G16B16A16_SFLOAT` + `EXTENDED_SRGB_LINEAR_EXT` (scRGB
    // linear) and the composite emits linear extended-range values;
    // `HdrEncoding::Pq` (requested via `hdr_pq`, only when an `HDR10_ST2084_EXT`
    // pair is advertised) runs the swapchain in that colour space and the
    // composite PQ-encodes (SMPTE ST 2084) in-shader. On SDR the swapchain runs
    // in `BGRA8_UNORM` + sRGB-nonlinear and the ACES + gamma + FXAA + LUT path
    // runs unchanged. Mirrors `DxContext::hdr_mode`. Stored so the swapchain
    // rebuild path preserves the format + colour space on resize.
    pub(super) hdr_mode: crate::gfx::hdr_output::HdrOutputMode,

    // GPU-compute particle system. See [`ParticleState`].
    pub(super) particle: ParticleState,

    // Auto-exposure (EV adaptation). See [`AutoExposureState`].
    pub(super) auto_exposure: AutoExposureState,

    // Built-in shader hot reload. See [`HotReloadState`].
    pub(super) hot_reload: HotReloadState,

    // Per-frame draw-call / VRAM / GPU-time counters surfaced to the
    // profiler overlay via [`Self::render_stats`]. Lives in a `Cell` because
    // the `objects` / `gpu_frame_us` / `vram_bytes` fields are filled from
    // `&mut self` in `draw_frame`. Mirrors `DxContext::frame_stats`.
    pub(super) frame_stats: std::cell::Cell<crate::gfx::profile::RenderStats>,
    // Draw-call accumulator the pass encoders bump via `inc_draw_calls`. An
    // `AtomicU32` (not the `frame_stats` Cell) because the parallel
    // command-buffer recording fans the encoders onto rayon workers that bump
    // it concurrently; a `Cell` would be a data race. Reset to 0 at the top of
    // `draw_frame` and drained into `frame_stats.draw_calls` at the end of
    // `record_frame`. Mirrors `DxContext::draw_calls_accum`.
    pub(super) draw_calls_accum: std::sync::atomic::AtomicU32,

    // Timestamp query pool with `2 * frames_in_flight` slots (one start +
    // end pair per in-flight frame). `record_frame` issues
    // `cmd_write_timestamp` at the top and bottom of recording; the CPU
    // reads the previous trip's pair at the top of `draw_frame` after the
    // matching fence wait. `None` when the queue does not expose
    // timestamps; `gpu_frame_us` then stays 0.
    pub(super) timestamp_query_pool: Option<vk::QueryPool>,
    // `timestamp_period` from the physical device, in nanoseconds per tick.
    // Combined with the resolved tick delta to derive microseconds.
    pub(super) timestamp_period_ns: f32,

    // `VK_EXT_memory_budget` device-local heap indices summed for the
    // VRAM-residency chip. Empty when the extension is unavailable; the
    // chip then reports 0.
    pub(super) device_local_heaps: Vec<u32>,
    // `true` when [`Self::device_local_heaps`] should be queried via
    // `VK_EXT_memory_budget`.
    pub(super) memory_budget_supported: bool,

    // Main geometry-path descriptor layouts, shared pool, and per-frame sets.
    // See `VkDescriptors`.
    pub(super) descriptors: VkDescriptors,
    // Instanced-prop pipeline, per-cluster material sets, per-frame instance
    // buffers + sets, and the cluster list. See `VkInstanced`.
    pub(super) instanced: VkInstanced,

    // Shared static vertex/index buffers plus their byte-range sub-allocators.
    // See `VkGeometry`.
    pub(super) geometry: VkGeometry,

    // Streamed VoxelWorld chunk rendering: headroom sub-allocators, the chunk
    // draw-slot freelist, the shared chunk descriptor pool + set, and the
    // material slots it samples. See `VkChunkStream`.
    pub(super) chunk_stream: VkChunkStream,
    // Clone descriptor state. See [`CloneState`].
    pub(super) clone: CloneState,

    // Skinned (skeletally animated) mesh rendering. See `VkSkinned`.
    pub(super) skinned: VkSkinned,

    // Main-pass view (per-frame) + light (shared) uniform buffers. See
    // `VkUniforms`.
    pub(super) uniforms: VkUniforms,

    // Per-frame-in-flight synchronization primitives. See `VkFrameSync`.
    pub(super) frame_sync: VkFrameSync,
    pub(super) current_frame: usize,
    pub(super) frames_in_flight: usize,
    // Lock presentation to the display refresh. Captured so `rebuild_swapchain`
    // re-selects the same present mode (FIFO vsync vs MAILBOX uncapped) on resize.
    pub(super) vsync: bool,

    // Per-frame command pools + buffers (start / per-pass / end tiers + the
    // shared one-shot pool). See `VkCommands`.
    pub(super) commands: VkCommands,

    // Draw list + cull inputs + bindless record counts. See [`DrawState`].
    pub(super) draw: DrawState,
    // Per-frame view state. See [`ViewState`].
    pub(super) view: ViewState,
    // Lazily-built wireframe twins of the main-pass pipelines; empty until the
    // first Wireframe frame. See [`super::wireframe`].
    pub(super) wireframe: super::wireframe::VkWireframe,
    // Number of mip levels in the bound IBL prefilter cubemap. 0 = no
    // EnvironmentMap declared; the fragment shader uses this as the IBL
    // on/off signal and falls back to the legacy ambient path.
    pub(super) prefilter_mip_count: u32,
    // Cube sampler shared by the IBL irradiance + prefilter cube bindings.
    // Held here so Drop can destroy it after the device idles.
    pub(super) cube_sampler: OwnedSampler,
    // Owned IBL cube textures. Live for the lifetime of the context.
    pub(super) env_map: EnvironmentMapTextures,

    // Scene-captured reflection probes. See [`ProbeState`].
    pub(super) probe: ProbeState,

    // Stall-free texture streaming. See [`StreamState`].
    pub(super) stream: StreamState,

    // Window + input (native Win32 on Windows, GLFW on Linux). `Option` so a
    // `reload_world` can MOVE the live window (with its cursor / menu / keymap
    // state) into the successor context instead of opening a new OS window;
    // `None` only transiently on the outgoing context, which is dropped
    // immediately after (see `window` / `window_mut` accessors + `Drop`).
    pub(super) window: Option<super::PlatformWindow>,

    // Keep Entry alive for the lifetime of the instance
    pub(super) _entry: ash::Entry,

    // The swapchain-level config (frames-in-flight / HDR mode) this context was
    // built with, reported by `hot_swap_config` so a live editor reload
    // (`reload_world`) reuses this backend in place only when the new world's
    // `swapchain_config` still matches; a mismatch routes to a full rebuild.
    // Mirrors `DxContext::swapchain_config`.
    pub(super) swapchain_config: crate::gfx::backend_init::SwapchainConfig,
    // Set on the OUTGOING context of a `reload_world` right before its successor
    // replaces it: the successor inherits (shares) this context's instance,
    // device, surface, and swapchain, so this context's `Drop` must free only
    // this world's content and leave those four shared objects (and the moved
    // window / debug messenger / timestamp pool) alone. False for every normally
    // constructed context, so a plain shutdown still tears everything down.
    // Vulkan needs this because its handles are not refcounted (unlike DirectX's
    // COM device / swapchain, where the outgoing context's release just drops a
    // reference). See `apply_world_reload` + `destroy_swapchain_resources`.
    pub(super) reused_by_successor: bool,
    // Set once `destroy_world_content` has run, so `Drop` never runs the
    // content teardown twice. True early on the outgoing context of a
    // `reload_world`, which frees its world before the successor builds (the
    // reload then places into the blocks the old world's leases released).
    pub(super) world_content_destroyed: bool,
}

// SAFETY: The host-mapped uniform pointers and the RefCell device allocator behind
// every pooled resource are used only on this thread; see
// `debug_assert_main_thread` below.
unsafe impl Send for VkContext {}

// Thread id of the thread that built the context. `VkContext::new` runs on the
// main thread and records it here; `debug_assert_main_thread` checks every
// mutation entry point against it. Portable across platforms (unlike the Win32
// `GetCurrentThreadId` the DirectX backend uses) since Vulkan also targets Linux.
static MAIN_THREAD_ID: std::sync::OnceLock<std::thread::ThreadId> = std::sync::OnceLock::new();

// Record the calling thread as the main (render) thread. Called once from
// `VkContext::new`, which always runs on the main thread.
pub(super) fn record_main_thread() {
    let _ = MAIN_THREAD_ID.set(std::thread::current().id());
}

// Debug-only guard that the caller is on the main thread.
//
// The `unsafe impl Send for VkContext` above is sound only because the context
// is touched from one thread alone: the GLFW window/event pump is thread-affine,
// the host-mapped uniform pointers and the device allocator's RefCell are
// single-threaded, and the parallel-encoder fan-out only ever shares `&self`
// read-only. The `RenderBackend` mutation entry points (reached through the
// boxed trait object) had nothing proving this, so scheduling `GraphicsSystem`
// off the main thread would silently race the window + queue submission instead
// of failing. This makes that mistake panic loudly in debug builds and compiles
// to nothing in release. `entry` is the offending method name, for the message.
// Mirrors `directx/context.rs::debug_assert_main_thread`.
#[inline]
#[track_caller]
pub(super) fn debug_assert_main_thread(entry: &str) {
    debug_assert!(
        MAIN_THREAD_ID
            .get()
            .is_none_or(|main| *main == std::thread::current().id()),
        "{entry} must be called from the main thread: VkContext is main-thread-only \
         (see `unsafe impl Send for VkContext`); driving GraphicsSystem off the main \
         thread races the GLFW window + Vulkan queue submission",
    );
}

//  Public API

impl VkContext {
    pub(crate) fn draw_frame(
        &mut self,
        params: FrameParams<'_>,
    ) -> crate::gfx::error::RenderResult<()> {
        let FrameParams {
            elapsed,
            fov_y_radians,
            near,
            far,
            cam_pos,
            text_calls,
            lines,
            world_hidden,
            view_mode,
            show,
        } = params;
        // Snapped for the passes recorded below (the wireframe pipeline
        // variant, the unlit shade flag, the composite's channel visualization
        // + depth normalization) and for the graph-input mask in record_frame.
        self.view.mode = view_mode;
        self.view.show = show;
        self.view.far = far;
        // Vulkan polygon mode is pipeline state, so the wireframe view needs its
        // own main-pass pipelines; built here on the first wireframe frame.
        self.ensure_wireframe_pipelines();
        // Shader hot-reload: if either the filesystem watcher or the debug
        // `reload-shaders` command set the flag, rebuild every built-in
        // pipeline from disk-resident source before this frame's passes
        // start using them. The flag is cleared regardless of outcome so a
        // failed rebuild (typo in a shader edit) doesn't loop, and the
        // previous pipelines stay live so the session keeps rendering.
        // Wait for the GPU to drain first so swapping pipelines out from
        // under in-flight command buffers is safe. Mirrors the DirectX
        // path at the top of its `draw_frame`.
        if self.shader_reload_requested() {
            self.clear_shader_reload_flag();
            self.wait_idle();
            match self.reload_shaders() {
                Ok(()) => tracing::info!("hot-reload: shader pipelines rebuilt"),
                Err(e) => tracing::error!("hot-reload: shader rebuild failed: {}", e),
            }
        }

        // Minimised window: the client area is 0x0. Vulkan rejects every
        // zero-extent operation (swapchain, render area, viewport, image copy),
        // so park the whole frame (no acquire / record / submit / present) until
        // the window is restored. `window_closed` keeps pumping the message loop
        // each tick, so the restore is picked up. Mirrors the DirectX backend,
        // which skips its resize + present while minimised.
        if self.frame_is_parked() {
            return Ok(());
        }

        let frame = self.current_frame;
        // Cheap-cloneable handle (ash::Device is Arc-like). Holding a local
        // copy avoids tying the rest of the function to `&self.device` while
        // record_frame takes `&mut self`.
        let device = self.device.clone();
        let device = &device;

        // Wait for this frame's slot to finish.
        // SAFETY: the fence belongs to this frame slot and was created from this device; the slice
        // borrows it for the call.
        unsafe {
            device
                .wait_for_fences(
                    std::slice::from_ref(&self.frame_sync.in_flight[frame]),
                    true,
                    u64::MAX,
                )
                .map_err(|e| super::error::map_vk_result(e, "wait fences"))?;
        }

        // Streamed texture swaps: re-point this frame slot's bindless pool
        // copy at the swapped-in views (legal now -- the fence wait above
        // retired every command buffer that binds this slot's set), and free
        // the old images / upload transients this slot parked on its previous
        // trip (this slot's fence signalling also covers the older frames that
        // last sampled them, and every pool copy has been re-pointed since).
        self.apply_streamed_texture_rewrites(frame);

        // Tick the device allocator: destroy retired handles, reclaim retired
        // ranges, release empty blocks. Here because the fence wait above is
        // what guarantees a range freed `retire_depth` ticks ago is no longer
        // referenced.
        self.alloc.begin_frame();
        // Same tick for the owned pipeline / layout / render-pass handles a
        // rebuild displaced, on the same reasoning.
        self.device.begin_frame();

        // Periodic footprint readout, for measuring the pool under streaming
        // churn at scale. Inert unless debug logging is enabled.
        if self.stream.frame.is_multiple_of(1024) && tracing::enabled!(tracing::Level::DEBUG) {
            tracing::debug!("device allocator: {}", self.alloc.stats());
        }

        // Advance the staggered reflection-probe bake one step. Runs here -- after
        // this frame's slot fence wait, before `record_frame` -- so any cube it
        // installs (a binding-8 rewrite + `probe.set.count` bump) is picked up by this
        // frame's `record_frame` ProbeSet upload + rendering. Non-fatal.
        if let Err(e) = self.bake_pending_probes() {
            tracing::warn!("reflection probe bake step failed: {e}");
        }

        // Reset this frame's render stats. `record_frame` accumulates
        // `draw_calls` through `inc_draw_calls` (interior-mutability since
        // the encoders run through `&self`); `objects`, `gpu_frame_us`,
        // and `vram_bytes` are filled here from `&mut self` state. Mirrors
        // the DirectX `frame_stats` reset at the top of `draw_frame`.
        let instanced_total: usize = self
            .instanced
            .clusters
            .iter()
            .map(|c| c.instances.len())
            .sum();
        let objects =
            (self.draw.objects.len() + instanced_total + self.skinned.draw_objects.len()) as u32;
        // Live skinned count: authored meshes plus runtime-spawned instances,
        // excluding the hidden pre-reserved pool slots. `objects` above counts the
        // whole pool and so stays flat across skinned spawn/despawn; this tracks
        // the visible count, so a spawn bumps it and a despawn drops it.
        let skinned_visible = self
            .skinned
            .draw_objects
            .iter()
            .filter(|o| o.visible)
            .count() as u32;
        // Filled in by the engine, which owns the skinned instance pool.
        let skinned_pool_free = 0u32;
        // GPU timing for the most-recently completed block on this frame slot:
        // the whole-frame pair plus one (start, end) pair per render pass. The
        // fence wait above guarantees the previous trip's writes have retired, so
        // the available query results are committed. The block is read with
        // `WITH_AVAILABILITY` so a pass that did not run this trip (its slots were
        // reset but never written) reads back unavailable -> 0, without stalling
        // the host (no `WAIT`). Zero before a slot has been visited a second time.
        let empty_pass_times = [("", 0u32); crate::gfx::profile::MAX_PASS_TIMINGS];
        let (gpu_frame_us, pass_times_us) = if let Some(pool) = self.timestamp_query_pool {
            // One [value, availability] pair per query slot (TYPE_64 +
            // WITH_AVAILABILITY -> two u64 per query; ash uses the element size as
            // the stride and the slice length as the query count).
            let mut results = vec![[0u64; 2]; super::pass_timing::SLOTS_PER_FRAME];
            // SAFETY: a property query on a live handle; it only reads.
            let res = unsafe {
                device.get_query_pool_results(
                    pool,
                    super::pass_timing::frame_block_base(frame),
                    &mut results,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
                )
            };
            // WITH_AVAILABILITY fills the buffer + per-query availability bits and
            // returns SUCCESS; tolerate NOT_READY defensively (the buffer is still
            // written, and the availability bits gate every read).
            if matches!(res, Ok(()) | Err(vk::Result::NOT_READY)) {
                let period = self.timestamp_period_ns;
                let pair_micros = |start_slot: usize, end_slot: usize| -> u32 {
                    let [s_val, s_avail] = results[start_slot];
                    let [e_val, e_avail] = results[end_slot];
                    if s_avail != 0 && e_avail != 0 && e_val > s_val && period > 0.0 {
                        let nanos = (e_val - s_val) as f64 * period as f64;
                        ((nanos / 1000.0) as u64).min(u32::MAX as u64) as u32
                    } else {
                        0
                    }
                };
                let frame_us = pair_micros(0, 1);
                let mut times = empty_pass_times;
                for (i, name) in crate::gfx::render_graph::PASS_NAMES.iter().enumerate() {
                    if i >= crate::gfx::profile::MAX_PASS_TIMINGS {
                        break;
                    }
                    times[i] = (*name, pair_micros(2 + 2 * i, 3 + 2 * i));
                }
                (frame_us, times)
            } else {
                (0, empty_pass_times)
            }
        } else {
            (0, empty_pass_times)
        };
        let vram_bytes = self.query_vram_bytes();
        let transient_pool_bytes = self.transient_pool.allocated_bytes();
        // Reset the parallel-safe draw-call accumulator for this frame; the
        // encoders fetch_add into it during recording and `record_frame`
        // drains it back into `frame_stats.draw_calls` once recording is done.
        self.draw_calls_accum
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.frame_stats.set(crate::gfx::profile::RenderStats {
            draw_calls: 0,
            objects,
            skinned_visible,
            skinned_pool_free,
            gpu_frame_us,
            vram_bytes,
            transient_pool_bytes,
            pass_times_us,
            // Adapted auto-exposure EV for the StatHud `EV` chip. `Some` only
            // when the world opted into auto-exposure (the EMA state is then
            // live); the static-exposure path leaves it `None` so the chip
            // stays blank. The value is the EV the most recent
            // `update_auto_exposure` EMA step settled on (the multiplier the
            // post stack pushes is `2^ev`). Mirrors `DxContext` / `MtlContext`.
            auto_exposure_ev: self.auto_exposure.state.as_ref().map(|s| s.current_ev),
            // EDR headroom for the StatHud `EDR x.X` chip, taken from the
            // `HdrOutputMode` resolved at init. `Some` only on the HDR path
            // (Vulkan has no portable max-EDR query, so the value is the
            // synthesised placeholder set in `init`); `None` on SDR blanks the
            // chip. Mirrors `DxContext` / `MtlContext::render_stats`.
            max_edr: match self.hdr_mode {
                crate::gfx::hdr_output::HdrOutputMode::Hdr { max_edr, .. } => Some(max_edr),
                crate::gfx::hdr_output::HdrOutputMode::Sdr => None,
            },
        });

        // Acquire swapchain image.
        // SAFETY: `self.swapchain.handle` is the live swapchain and `image_available[frame]` is an
        // unsignalled semaphore from this device's own pool for this frame slot.
        let acquire = unsafe {
            self.swapchain.loader.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                self.frame_sync.image_available[frame],
                vk::Fence::null(),
            )
        };
        let image_index = match acquire {
            Ok((idx, suboptimal)) => {
                if suboptimal {
                    self.rebuild_swapchain()?;
                    return Ok(());
                }
                idx
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.rebuild_swapchain()?;
                return Ok(());
            }
            Err(e) => return Err(super::error::map_vk_result(e, "acquire swapchain image")),
        };

        // SAFETY: the fence belongs to this frame slot and was just waited on, so it is signalled
        // and not in use by a pending submission.
        unsafe { device.reset_fences(std::slice::from_ref(&self.frame_sync.in_flight[frame])) }
            .map_err(|e| format!("reset fences: {e}"))?;

        // Record the frame. `record_frame` records the leading timestamp into
        // the `start` buffer, fans each non-composite pass onto its own
        // per-pass command buffer, and records Composite + the post-graph work
        // into `cmd` (the outer "end" buffer begun here). It returns the
        // ordered `[start, ...pass buffers]` to submit before `end`.
        let cmd = self.commands.command_buffers[frame];
        // SAFETY: `cmd` belongs to this frame slot, whose fence was already waited on, so it is not
        // in flight; reset then begin puts it in the recording state, which is what the subsequent
        // recording requires.
        unsafe {
            device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("reset cmd buf: {e}"))?;
            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| format!("begin cmd buf: {e}"))?;
        }

        let mut submit_bufs = self.record_frame(
            RecordFrameTargets {
                cmd,
                image_index,
                frame_idx: frame,
            },
            RecordFrameView {
                elapsed,
                fov_y_radians,
                near,
                far,
                cam_pos,
                text_calls,
                lines,
            },
            world_hidden,
        )?;

        // SAFETY: `cmd` is in the recording state, which is what `end_command_buffer` requires.
        unsafe { device.end_command_buffer(cmd) }.map_err(|e| format!("end cmd buf: {e}"))?;
        // The outer "end" buffer (Composite + post-graph work + trailing
        // timestamp) submits last, after every per-pass buffer.
        submit_bufs.push(cmd);

        // Submit the whole batch in one call: submission order = GPU order on
        // the single graphics queue. The render-finished semaphore is indexed
        // by swapchain image (not frame slot) so present never reuses one still
        // in flight.
        let wait_sems = [self.frame_sync.image_available[frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_sems = [self.frame_sync.render_finished[image_index as usize]];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_sems)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&submit_bufs)
            .signal_semaphores(&signal_sems);
        // SAFETY: every command buffer in `submit_bufs` was ended and belongs to this frame slot,
        // the semaphores and fence were created from this device, and `submit_info` borrows all of
        // them for the call.
        unsafe {
            device
                .queue_submit(
                    self.graphics_queue,
                    std::slice::from_ref(&submit_info),
                    self.frame_sync.in_flight[frame],
                )
                .map_err(|e| super::error::map_vk_result(e, "queue submit"))?;
        }

        // Present.
        let swapchains = [self.swapchain.handle];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_sems)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        // SAFETY: `present_info` borrows the swapchain, image index, and wait semaphore for the
        // call; the semaphore is signalled by the submission above.
        let present_result = unsafe {
            self.swapchain
                .loader
                .queue_present(self.present_queue, &present_info)
        };
        if present_result == Err(vk::Result::ERROR_OUT_OF_DATE_KHR) || present_result == Ok(true) {
            self.rebuild_swapchain()?;
        } else {
            present_result.map_err(|e| super::error::map_vk_result(e, "present"))?;
            // Record which swapchain image now holds a complete, presented frame
            // so the `screenshot` debug command can read it back.
            self.swapchain.last_present_index = Some(image_index);
        }

        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;
        Ok(())
    }

    pub(crate) fn update_view(&mut self, matrix: [[f32; 4]; 4]) {
        self.view.matrix = matrix;
    }

    // Update the model matrices of the given draw objects, one
    // `(slot, matrix)` entry per changed object. Out-of-range slots have no
    // effect.
    pub(crate) fn update_models(&mut self, updates: &[(u32, [[f32; 4]; 4])]) {
        for &(index, model) in updates {
            if let Some(obj) = self.draw.objects.get_mut(index as usize) {
                obj.model = model;
            }
        }
    }

    pub(crate) fn update_visibility(&mut self, index: usize, visible: bool) {
        if let Some(obj) = self.draw.objects.get_mut(index) {
            obj.visible = visible;
        }
    }

    // Retire a draw object for a despawned entity: clear `visible` (drops it
    // from the main / shadow / velocity passes) and `resident` (drops it from
    // the ray-tracing BLAS / geometry-table rebuild), so it leaves no ghost in
    // any pass. The geometry buffers stay allocated; the engine's draw-slot
    // allocator recycles the index (only for the runtime-append region here:
    // `reuses_build_slots` is false because the init-time cull BVH and RT
    // `object_indices` key fixed build-time slots and cannot refit). If the
    // slot held a runtime clone, its descriptor-pool offset is freed too so a
    // steady spawn/despawn cadence does not exhaust the clone pool. No-op if
    // the index is out of range.
    pub(crate) fn retire_draw_object(&mut self, index: usize) {
        if let Some(obj) = self.draw.objects.get_mut(index) {
            obj.visible = false;
            obj.resident = false;
            if let Some(offset) = self.clone.slot_by_draw_idx.remove(&index) {
                self.clone.free_offsets.push(offset);
            }
        }
    }

    // Add a draw slot to `draw.always` if it is not already a member. Runtime
    // draws (chunks, spawned clones) are drawn unconditionally because the
    // init-time BVH cannot refit to admit them; a slot recycled from a culled
    // static prop is not yet in `draw.always` and must be added, while one
    // recycled from another chunk / clone already is.
    pub(super) fn ensure_always_draw(&mut self, slot: usize) {
        if !self.draw.always_member[slot] {
            self.draw.always.push(slot as u32);
            self.draw.always_member[slot] = true;
        }
    }

    // The frame's unlit flag for ViewUniforms, from the viewport view mode.
    pub(super) fn shade_mode(&self) -> f32 {
        if self.view.mode == concinnity_core::gfx::view_modes::ViewMode::Unlit {
            1.0
        } else {
            0.0
        }
    }

    pub(crate) fn set_fade(&mut self, fade: f32) {
        self.view.scene_fade = fade.clamp(0.0, 1.0);
    }

    // The live platform window. Present for every constructed context; `None`
    // only on the outgoing context of a `reload_world` (its window was moved
    // into the successor), which is dropped without any further window access.
    #[inline]
    pub(super) fn window(&self) -> &super::PlatformWindow {
        self.window
            .as_ref()
            .expect("VkContext window taken by reload_world")
    }

    #[inline]
    pub(super) fn window_mut(&mut self) -> &mut super::PlatformWindow {
        self.window
            .as_mut()
            .expect("VkContext window taken by reload_world")
    }

    // True while the window is minimised: the client area has collapsed to 0x0
    // (WM_SIZE reports zero on minimise; the GLFW / AppKit windows report the
    // same). Vulkan forbids a zero-extent swapchain, render area, viewport, or
    // image copy, so `draw_frame` and `rebuild_swapchain` park their work until
    // the window is restored. Mirrors the DirectX backend, whose
    // `maybe_handle_resize` skips the rebuild (and thus the frame) at 0x0.
    #[inline]
    pub(super) fn is_minimized(&self) -> bool {
        let (w, h) = self.window().framebuffer_size();
        extent_minimized(w, h)
    }

    // True while this frame cannot be presented, which is what `draw_frame`
    // parks on. The window's own size is not enough: it is tracked from WM_SIZE,
    // so a window that was already minimised when it was created never saw a
    // zero and reports its requested size for the whole run, while the surface
    // reports 0x0 from the start. Without the surface check that run builds a
    // zero-extent swapchain and then renders into it every frame -- a render
    // area, viewport and image copy all at zero, which validation rejects
    // individually and endlessly. `rebuild_swapchain` already gates on the
    // surface for the same reason.
    //
    // A failed capability query answers "presentable": a transient WSI error
    // must not wedge the renderer into a permanent park.
    pub(super) fn frame_is_parked(&self) -> bool {
        if self.is_minimized() {
            return true;
        }
        match self.surface_extent() {
            Ok(extent) => !super::swapchain::extent_is_presentable(extent),
            Err(_) => false,
        }
    }

    // Re-point the combined-image-sampler at `binding` of `set` to `view`.
    // Shared by the texture-streaming descriptor rewrites below.
    pub(crate) fn window_closed(&mut self) -> bool {
        self.window_mut().poll()
    }

    pub(crate) fn wait_idle(&self) {
        // SAFETY: a wait on this device's own queues; it takes no borrowed state.
        let _ = unsafe { self.device.device_wait_idle() };
    }

    // Render statistics for the most recent `draw_frame`, for the profiler
    // overlay. `gpu_frame_us` is filled at the top of each `draw_frame`
    // from the timestamp pair this slot resolved on its previous trip
    // through the ring (so the reading is `frames_in_flight`-stale by
    // construction, matching DirectX / Metal). Per-pass GPU timing is
    // still a follow-up.
    pub(crate) fn render_stats(&self) -> crate::gfx::profile::RenderStats {
        self.frame_stats.get()
    }

    // Current device-local memory residency in bytes, via
    // `VK_EXT_memory_budget`. Sums `heap_usage` on every DEVICE_LOCAL heap;
    // returns 0 when the extension is unavailable (so the chip degrades
    // gracefully on adapters that don't expose budgets, matching DirectX's
    // behaviour on pre-WDDM-2.0 adapters).
    pub(super) fn query_vram_bytes(&self) -> u64 {
        if !self.memory_budget_supported || self.device_local_heaps.is_empty() {
            return 0;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut props2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        // SAFETY: a property query on a live handle; it only reads.
        unsafe {
            self.instance
                .get_physical_device_memory_properties2(self.physical_device, &mut props2);
        }
        self.device_local_heaps
            .iter()
            .map(|&i| budget.heap_usage[i as usize])
            .sum()
    }

    // Bump this frame's CPU-issued draw-call counter. Called from each
    // draw site in the shadow, main, decal, and composite + text passes.
    // Mirrors `DxContext::inc_draw_calls`; fullscreen post-process passes
    // (SSAO, SSR, TAA, bloom, fog) are not counted per the `RenderStats`
    // doc comment.
    pub(super) fn inc_draw_calls(&self, n: u32) {
        // Bump the atomic accumulator (not the `frame_stats` Cell) so the
        // parallel-recording workers don't race. Drained into
        // `frame_stats.draw_calls` at the end of `record_frame`.
        self.draw_calls_accum
            .fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn capture_cursor(&mut self) {
        self.window_mut().capture_cursor();
    }

    // Symmetric with `capture_cursor`; reached only through `set_camera_capture`
    // today, kept public so the cursor API stays complete.
    #[allow(dead_code)]
    pub(crate) fn release_cursor(&mut self) {
        self.window_mut().release_cursor();
    }

    // Hide or show the OS cursor for an in-engine UI cursor (e.g. a MainMenu),
    // without engaging camera capture. Edge-triggered in the window helper.
    pub(crate) fn set_ui_cursor_hidden(&mut self, hidden: bool) {
        self.window_mut().set_ui_cursor_hidden(hidden);
    }

    // Whether the real cursor has left the window so the renderer should stop
    // drawing the in-engine UI cursor (windowed / borderless). Recomputed each
    // `poll` (in `window_closed`); false while captured or in fullscreen (which
    // confines the cursor instead).
    pub(crate) fn cursor_outside_window(&self) -> bool {
        self.window().cursor_outside_window()
    }

    // A togglable menu coexists with a captured camera; see
    // `RenderBackend::set_menu_mode`.
    pub(crate) fn set_menu_mode(&mut self, on: bool) {
        self.window_mut().set_menu_mode(on);
    }

    // Drive cursor capture from the menu state each frame: capture for camera
    // control, release while a menu is open. Edge-triggered in the window.
    pub(crate) fn set_camera_capture(&mut self, capture: bool) {
        self.window_mut().set_camera_capture(capture);
    }

    // Turn display sync (vsync) on or off at runtime. The present mode is fixed
    // at swapchain creation (FIFO for vsync, MAILBOX/IMMEDIATE for uncapped), so
    // a change recreates the swapchain, which re-selects the mode from
    // `self.vsync`. Edge-triggered: a redundant call (a swapchain rebuild is
    // expensive) is skipped.
    pub(crate) fn set_vsync(&mut self, on: bool) {
        if on == self.vsync {
            return;
        }
        self.vsync = on;
        if let Err(e) = self.rebuild_swapchain() {
            tracing::warn!("set_vsync: rebuild_swapchain failed: {}", e);
        }
    }

    // Switch window mode / resize at runtime. The GLFW work lives in window.rs;
    // the framebuffer-size change drives a swapchain rebuild via the present
    // path's OUT_OF_DATE handling.
    pub(crate) fn set_window_mode(&mut self, mode: crate::assets::WindowMode) {
        self.window_mut().set_window_mode(mode);
    }

    pub(crate) fn set_window_size(&mut self, width: u32, height: u32) {
        self.window_mut().set_window_size(width, height);
    }

    // The display modes feeding the Resolution settings row; enumeration,
    // the fullscreen mode hold, and the desktop-mode restore all live in
    // window.rs (GLFW owns the video-mode switching).
    pub(crate) fn display_modes(&self) -> Vec<crate::gfx::display_mode::DisplayMode> {
        self.window().display_modes()
    }

    pub(crate) fn current_display_mode(&self) -> Option<crate::gfx::display_mode::DisplayMode> {
        self.window().current_display_mode()
    }

    pub(crate) fn set_display_mode(&mut self, mode: crate::gfx::display_mode::DisplayMode) {
        self.window_mut().set_display_mode(mode);
    }

    // Replace the live post-process parameters, pushed to the bloom + composite
    // shaders each frame.
    pub(crate) fn update_post_process(
        &mut self,
        params: crate::gfx::render_types::PostProcessParams,
    ) {
        self.post_process = params;
    }

    // Set the live ambient (IBL) light scale (the Ambient slider). It lives in
    // `LightUniforms`, uploaded to a single (not per-frame) UBO, so unlike
    // `update_post_process` (push constants) it mutates the CPU-side copy and
    // re-uploads the buffer. Because the buffer is shared across frames-in-flight,
    // the device is drained first so the rewrite never races an in-flight read;
    // ambient changes only on a slider drag, so the stall is rare. Edge-triggered:
    // a no-op when the value is unchanged (e.g. an init push with no persisted
    // override).
    pub(crate) fn set_ambient_intensity(&mut self, value: f32) {
        if self.uniforms.light_uniforms.ambient_intensity == value {
            return;
        }
        self.uniforms.light_uniforms.ambient_intensity = value;
        self.wait_idle();
        super::draw::upload_light_uniforms(&self.uniforms.light_ubo, &self.uniforms.light_uniforms);
    }

    // Set the live shadow cascade re-render cadence. The per-frame cascade split
    // reads `shadow.update` at the start of each draw (see draw.rs), so a change
    // takes effect on the next frame with no rebuild or allocation.
    pub(crate) fn set_shadow_update(&mut self, update: crate::assets::ShadowUpdate) {
        self.shadow.update = update;
    }

    // Set the live shadow distance (world units). The per-frame cascade-split
    // computation reads `shadow.distance` each draw (capped at the camera far
    // plane), so a change takes effect on the next frame with no allocation (it
    // sizes no GPU resource).
    pub(crate) fn set_shadow_distance(&mut self, distance: u32) {
        self.shadow.distance = distance;
    }

    // Set the live shadow cascade count (1..=4). The per-frame split + schedule
    // read `shadow.cascades` each draw; only the first `count` of the four slots
    // are rendered + sampled, so a change takes effect on the next frame with no
    // resize (the shadow-map array stays sized for the 4-cascade capacity).
    pub(crate) fn set_shadow_cascades(&mut self, count: u32) {
        self.shadow.cascades = count;
    }

    // Update the live scalar sub-tunables of the SSAO / SSR / SSGI / auto-exposure
    // passes without rebuilding anything. Each pass rebuilds its per-frame uniform
    // from these stored `*Settings` every draw (`settings.params(...)`), so
    // mutating the stored struct here is picked up on the next frame. Only a
    // feature whose resources are currently live has settings to mutate; the rest
    // are skipped (the value still persists for the next launch). SSAO / SSR /
    // auto-exposure are fully scalar, so they are replaced wholesale; SSGI keeps
    // its gather resolution / ray / step counts (those size the gather target or
    // ride `apply_quality_settings`), so only its scalar intensity / distance are
    // updated. Auto-exposure settings live flat on the context here
    // (`auto_exposure.settings`), not inside a resources struct as on Metal.
    pub(crate) fn update_quality_params(&mut self, q: crate::gfx::backend::QualitySettings) {
        if let (Some(live), Some(cur)) = (q.ssao, self.ssao.as_mut().map(|s| &mut s.settings)) {
            *cur = live;
        }
        if let (Some(live), Some(cur)) = (q.ssr, self.ssr.as_mut().map(|s| &mut s.settings)) {
            *cur = live;
        }
        if let (Some(live), Some(cur)) = (q.ssgi, self.ssgi.as_mut().map(|s| &mut s.settings)) {
            cur.intensity = live.intensity;
            cur.max_distance = live.max_distance;
        }
        if let (Some(live), Some(cur)) = (q.auto_exposure, self.auto_exposure.settings.as_mut()) {
            *cur = live;
        }
    }

    // Public accessor for the shared shader-reload flag. Cloning the `Arc`
    // lets the debug WebSocket server flip it from a non-render thread.
    // `None` outside `cn debug`. Mirrors `DxContext::shader_reload_pending`.
    pub(crate) fn shader_reload_pending(
        &self,
    ) -> Option<std::sync::Arc<std::sync::atomic::AtomicBool>> {
        self.hot_reload
            .reload_pending
            .as_ref()
            .map(std::sync::Arc::clone)
    }

    pub(crate) fn take_input(&mut self) -> InputState {
        // Both platform windows snapshot straight into the shared RenderInput.
        self.window_mut().take_input()
    }

    // Replace the runtime movement key map. The window's key decode routes
    // through it, so a settings-menu rebind takes effect immediately.
    pub(crate) fn set_keymap(&mut self, keymap: &crate::gfx::keymap::KeyMap) {
        self.window_mut().set_keymap(keymap);
    }

    // Live window size for overlay (view-owned UI) scaling and cursor
    // hit-testing, in the window's logical units. Read from the platform window
    // rather than the swapchain extent so the overlay space matches the units
    // `poll()` reports the cursor in on every platform: points on macOS, client
    // pixels on Windows, window coordinates on Linux. Where the framebuffer is
    // larger (a retina drawable, a scaled Wayland surface) the difference is
    // absorbed by the UI shader's divide to NDC, and only the text scissor
    // converts back to pixels.
    pub(crate) fn logical_size(&self) -> (f32, f32) {
        self.window().logical_size()
    }

    // Device capability flags for the settings menu. RT reflects whether the
    // ray-query device extensions were enabled at device creation
    // (`rt_capable`).
    pub(crate) fn capabilities(&self) -> crate::gfx::backend::DeviceCapabilities {
        crate::gfx::backend::DeviceCapabilities {
            ray_tracing: self.rt_capable,
            selectable_upscaler: true,
            // The cull BVH + RT tables key fixed build-time slot indices and
            // cannot refit; only the runtime-append region recycles (tracked
            // as the RT incremental topology parity item).
            reuses_build_slots: false,
        }
    }

    // Coarse GPU performance profile for default-quality selection, read live
    // from the physical device: vendor id, discrete / integrated device type,
    // and the summed DEVICE_LOCAL heap size as the VRAM budget (the true heap
    // size, unlike the residency chip which sums live usage).
    pub(crate) fn gpu_profile(&self) -> crate::gfx::backend::GpuProfile {
        use crate::gfx::backend::{
            GpuClassInput, GpuProfile, GpuVendor, apple_family_from_device_name, classify_tier,
        };
        // SAFETY: a property query on a live handle; it only reads.
        let props = unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        };
        let vendor = match props.vendor_id {
            0x10DE => GpuVendor::Nvidia,
            0x1002 => GpuVendor::Amd,
            0x8086 => GpuVendor::Intel,
            0x106B => GpuVendor::Apple, // Apple / MoltenVK
            _ => GpuVendor::Other,
        };
        let discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
        let unified = props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU;
        // SAFETY: a property query on a live handle; it only reads.
        let mem = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        let budget: u64 = (0..mem.memory_heap_count as usize)
            .filter(|&i| {
                mem.memory_heaps[i]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            })
            .map(|i| mem.memory_heaps[i].size)
            .sum();
        let tier = classify_tier(&GpuClassInput {
            vendor,
            memory_budget_bytes: budget,
            discrete,
            apple_family: apple_family_from_device_name(&super::gpu_profile::device_name(&props)),
        });
        GpuProfile {
            vendor,
            tier,
            memory_budget_bytes: budget,
            unified_memory: unified,
            discrete,
        }
    }
}

impl crate::gfx::scene_flow::SceneControl for VkContext {
    fn update_visibility(&mut self, draw_idx: usize, visible: bool) {
        self.update_visibility(draw_idx, visible);
    }
    fn set_fade(&mut self, fade: f32) {
        self.set_fade(fade);
    }
}

impl VkContext {
    // Free every per-world resource: pipelines, descriptor pools, feature
    // states, and all pooled buffers / images (whose leases retire through the
    // allocator as their holders drop or clear). Shared hardware, the
    // allocator itself, and the surface / device / instance are NOT touched;
    // `Drop` owns those. Called from `Drop` on a normal shutdown, and early by
    // `apply_world_reload` so the successor build places into the blocks this
    // world releases instead of doubling the footprint. Guarded so the `Drop`
    // after an early call is a no-op; the caller has idled the device.
    pub(super) fn destroy_world_content(&mut self) {
        if self.world_content_destroyed {
            return;
        }
        self.world_content_destroyed = true;
        let device = self.device.clone();
        let device = &device;

        // Abandon any in-flight staggered probe bake: free its per-face command
        // buffers (before `self.commands` is destroyed below) + fences + bake target.
        // `wait_idle` above retired its GPU work. The converting slot holds only CPU
        // data (drops freely; its worker thread, if still running, touches no vk
        // handle, only the shared payload `OnceLock`).
        if let Some(rendering) = self.probe.rendering.take() {
            rendering.destroy(device, self.commands.command_pool);
        }

        // Parked streamed-texture retires (`wait_idle` above covered them).
        self.drain_stream_retires();

        // Sync (per-frame-in-flight semaphores + fences).
        self.frame_sync.destroy(device);

        // Command pools (each frees the buffers allocated from it).
        self.commands.destroy(device);

        // Framebuffers + attachments.
        self.destroy_swapchain_resources();

        // Shadow (framebuffers, pipelines, layouts, map, render pass, UBO,
        // sampler).
        self.shadow.destroy(device);
        self.spot_shadow.destroy(device);
        self.area_light.destroy(device);

        // IBL cubes + cube sampler.
        self.env_map.irradiance = GpuImage::null();
        self.env_map.prefilter = GpuImage::null();
        // SAFETY: the handle was created from this device and is destroyed exactly once; the caller
        // has already waited for the device to go idle, so no submission still references it.

        // Pipelines.
        self.wireframe.destroy();
        self.text.pipeline = None;
        // Instanced-prop pipeline + per-frame instance buffers (see
        // `VkInstanced::destroy`).
        self.instanced.destroy(device);
        // SAFETY: the handle was created from this device and is destroyed exactly once; the caller
        // has already waited for the device to go idle, so no submission still references it.
        // SAFETY: the handle was created from this device and is destroyed exactly once; the caller
        // has already waited for the device to go idle, so no submission still references it.

        // GPU-driven cull + bindless static pass resources, including the Hi-Z
        // pyramid (see `VkCull::destroy`). The bindless / cull / phase-2
        // descriptor sets are freed with the shared descriptor pool +
        // `two_pass_pool`.
        self.cull.destroy(device);
        self.light_cull.destroy(device);

        // Composite pass resources (the LUT retires through the allocator).
        self.color_lut = GpuImage::null();

        // Bloom resources (mips + framebuffers freed by
        // destroy_swapchain_resources above).

        // TAA resources (velocity + history passes, pipelines, targets, UBOs).
        if let Some(mut taa) = self.taa.take() {
            taa.destroy(device);
        }

        // SSAO resources (pre-pass + kernel + blur). The blur framebuffer
        // references the pool's `ao_output` view, so SSAO is torn down before
        // the transient pool below (framebuffers before their views).
        if let Some(mut ssao) = self.ssao.take() {
            ssao.destroy(device);
        }
        self.ssao_white = GpuImage::null();

        // Transient image pool (the graph-owned transients, e.g. `ao_output`).
        self.transient_pool.destroy(device);

        // SSR resources (pre-pass + resolve).
        if let Some(mut ssr) = self.ssr.take() {
            ssr.destroy(device);
        }

        // Reflection composite (roughness blur + composite of the SSR/RT output).
        if let Some(mut rc) = self.reflection_composite.take() {
            rc.destroy(device);
        }

        // SSGI resources (gather + composite).
        if let Some(mut ssgi) = self.ssgi.take() {
            ssgi.destroy(device);
        }

        // Unified G-buffer pre-pass resources (per-frame MRT + pipelines + UBOs).
        if let Some(mut gb) = self.gbuffer.take() {
            gb.destroy(device);
        }

        // Hardware ray-traced reflection resources (the pass + the acceleration
        // structures). The pass is destroyed first (its output + pipelines), then
        // the BLAS / TLAS / scratch / geometry table.
        if let Some(mut rt) = self.rt_reflections.take() {
            rt.destroy(device);
        }
        if let Some(mut accel) = self.rt_accel.take() {
            accel.destroy(device);
        }

        // Temporal upscaling (FSR / DLSS / XeSS): the vendor context + the
        // output texture, via the backend trait.
        if let Some(mut up) = self.upscale.take() {
            up.destroy(device);
        }

        // Decal resources (pipeline + per-frame uniforms + per-decal sets).
        if let Some(mut decals) = self.decal.resources.take() {
            decals.destroy(device);
        }

        // Line resources (pipeline + per-frame uniforms + vertex buffers +
        // framebuffers). Only present once a frame published lines.
        if let Some(mut lines) = self.lines.resources.take() {
            lines.destroy(device);
        }

        // Per-frame text-geometry upload buffers.
        self.text.upload.destroy();

        // Volumetric-fog resources (pipeline + per-frame uniforms).
        if let Some(mut fog) = self.fog.resources.take() {
            fog.destroy(device);
        }

        // Raymarched SDF volume resources (per-volume pipelines + UBOs, view
        // ring, descriptor pool, render passes, snapshot image).
        if let Some(mut rm) = self.raymarch.take() {
            rm.destroy(device);
        }

        // Planar reflection resources (mirror targets + framebuffers + per-(plane,
        // frame) view ring + global sets + descriptor pool). Destroyed before glass,
        // whose per-pane sets reference the planar target views.
        if let Some(mut planar) = self.planar_reflection.take() {
            planar.destroy(device);
        }

        // Glass / transparent-pass resources (pipeline, per-panel buffers +
        // UBOs, per-frame view ring, descriptor pool, render pass, framebuffers,
        // snapshot image).
        if let Some(mut glass) = self.glass.take() {
            glass.destroy(device);
        }

        // Auto-exposure resources (pipelines + histogram + per-frame readbacks).
        if let Some(mut ae) = self.auto_exposure.resources.take() {
            ae.destroy(device);
        }

        // Particle resources (compute + render pipelines, view UBO ring,
        // per-emitter descriptor pool, framebuffers). Per-emitter pool /
        // counter buffers are destroyed via the dedicated helper before
        // the shared pipeline state: Vulkan needs the per-emitter buffers
        // gone first so the upcoming pipeline destroys can't trip a
        // validation error on a still-referenced descriptor.
        self.destroy_particle_emitter_states(device);
        if let Some(mut p) = self.particle.resources.take() {
            p.destroy(device);
        }

        // Profiler-overlay timestamp pool.
        if let Some(pool) = self.timestamp_query_pool.take() {
            // SAFETY: the handle was created from this device and is destroyed exactly once; the
            // caller has already waited for the device to go idle, so no submission still
            // references it.
            unsafe { device.destroy_query_pool(pool, None) };
        }

        // Render passes.

        // Chunk-stream descriptor pool (frees `chunk_stream.object_set`); the
        // shared main pool + its set layouts are freed by
        // `self.descriptors.destroy` below.
        self.chunk_stream.destroy(device);

        // Skinned-mesh resources.
        self.skinned.destroy(device);

        // Main descriptor pool + the three set layouts (the pool frees the
        // global / object / text_atlas sets).
        self.descriptors.destroy(device);

        // Geometry.
        self.geometry.destroy();

        // UBOs (per-frame view + shared light).
        self.uniforms.destroy();

        // Samplers.

        // Scene textures + baked reflection-probe cubes: dropping them retires
        // them through the allocator.
        self.textures.clear();
        self.normal_map_textures.clear();
        self.text.atlas_textures.clear();
        self.probe.maps.clear();
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        self.wait_idle();

        // No-op on the outgoing context of a `reload_world`, which already
        // freed its world before the successor built.
        self.destroy_world_content();

        // Device allocator: destroy every handle its dropped leases queued and
        // free the blocks. After the content pass above so all leases have
        // dropped, before the device teardown below. On a reload the successor
        // shares this allocator, so only a context that still owns it drains.
        if !self.reused_by_successor {
            self.alloc.destroy();
        }

        // The surface is instance-level and not refcounted, so on a
        // `reload_world` the successor inherited it and the outgoing context
        // leaves it alone; its window / debug messenger were already moved out
        // (both `None` below). It goes before the instance, which the owning
        // device handle destroys once this context's fields have dropped. The
        // swapchain is likewise skipped inside `destroy_swapchain_resources`
        // (called above) when reused.
        if !self.reused_by_successor {
            // SAFETY: the handle was created from this device and is destroyed exactly once; the
            // caller has already waited for the device to go idle, so no submission still
            // references it.
            unsafe { self.surface_loader.destroy_surface(self.surface, None) };
        }

        // The device, the instance, the debug messenger and the pipeline cache
        // are the owning
        // device handle's to destroy, once every owned Vulkan object has
        // retired through it. That happens as this context's fields drop,
        // after this body returns, so nothing here has to order it -- and on a
        // live reload the successor's clone simply keeps them alive.
    }
}

// Whether a client-area extent counts as minimised (collapsed): a zero, or
// defensively negative, width or height. Split from `is_minimized` so the
// minimise gate is unit-testable without a live window / surface.
fn extent_minimized(width: i32, height: i32) -> bool {
    width <= 0 || height <= 0
}

#[cfg(test)]
mod tests {
    use super::extent_minimized;

    #[test]
    fn extent_minimized_gates_on_zero_or_negative_dimensions() {
        // A live window (both dimensions positive) is not minimised.
        assert!(!extent_minimized(1280, 720));
        assert!(!extent_minimized(1, 1));
        // Minimise collapses one or both dimensions to zero.
        assert!(extent_minimized(0, 0));
        assert!(extent_minimized(1280, 0));
        assert!(extent_minimized(0, 720));
        // Negative dimensions never form a valid extent either.
        assert!(extent_minimized(-1, 720));
        assert!(extent_minimized(1280, -1));
    }
}
