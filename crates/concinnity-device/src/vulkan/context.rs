// Vulkan rendering context. Owns all GPU resources, the GLFW window, and input state.
// Mirrors the public API of metal::MtlContext so GraphicsSystem can drive both
// backends identically.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::auto_exposure;
use concinnity_engine::gfx::frustum::Frustum;
use concinnity_engine::gfx::render_types::{
    InstancedCluster, LightUniforms, PostProcessParams, PostProcessTunables,
};
use concinnity_engine::input::keymap::KeyMap;
use concinnity_engine::input::snapshot::InputSnapshot;
use concinnity_engine::render::backend::FrameParams;
use concinnity_engine::render::buffers::retire_pool::RetirePool;
use concinnity_engine::render::frame::plan::drive_frame;
use concinnity_engine::render::light::lights::LightData;
use concinnity_engine::render::probe::book::ProbeBook;
use concinnity_engine::render::scene_state::SceneState;
use concinnity_engine::render::shader::reload::ShaderReload;
use concinnity_engine::render::{
    backend, backend::error, backend::init, decal, environment::volumetric_fog, output::hdr,
    shadow::spot, slots::rewrites,
};
use concinnity_engine::window::display_mode;
use concinnity_engine::{components, profile};

use super::allocator::PooledBuffer;
use super::geometry_upload::{CopySubmit, GeometryDest, GeometryUploads};
use super::post::{
    GbufferResources, PostSupport, RtReflectionsResources, SsaoResources, SsgiResources,
    SsrResources, TaaResources, VkBloomPass, VkReflectionCompositePass, VkUpscaleBackend, gbuffer,
};
use super::texture::{EnvironmentMapTextures, GpuImage, StreamedUploadRetire};
use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedFramebuffer, OwnedPipeline, OwnedPipelineLayout, OwnedRenderPass,
    OwnedSampler, OwnedSetLayout, VkDevice,
};

// Off-screen HDR render-target format. The main pass renders linear-light
// radiance into this; the composite pass tonemaps it down to the swapchain's
// 8-bit format. `R16G16B16A16_SFLOAT` is universally supported as a color
// attachment + sampled image on desktop GPUs.
pub(super) const HDR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

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
    pub(super) global_set_layout: OwnedSetLayout,
    pub(super) sampler: OwnedSampler,
    // Per-frame-in-flight `ShadowUniforms` ring, persistently mapped. One slot
    // per frame: a single buffer would let this frame's cascade VPs overwrite
    // memory an in-flight frame is still sampling, which under `Hybrid` pairs a
    // freshly-jumped far-cascade VP with depth rasterized from the old one.
    pub(super) ubos: Vec<PooledBuffer>,
}

impl VkShadow {
    // Whether the world renders cascades at all (`shadow_map_size > 0`).
    pub(super) fn enabled(&self) -> bool {
        !self.framebuffers.is_empty()
    }

    // Destroy every owned GPU object. Called from `VkContext::drop` after
    // `wait_idle`. The per-frame shadow global sets live in `VkDescriptors`.
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
// render pass, GPU-driven pipeline, and comparison sampler.
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
    // Bytes between two slices' uniforms in `ubo`.
    pub(super) ubo_stride: u64,

    pub(super) sets: Vec<vk::DescriptorSet>,
    pub(super) _descriptor_pool: OwnedDescriptorPool,
    // Each slice's light frustum, which its GPU cull keeps casters inside.
    pub(super) frusta: Vec<Frustum>,
    // Slices re-rendered this frame (bit `i` = slice `i`), as the frame's plan
    // scheduled them.
    pub(super) render_mask: u32,
}

impl VkSpotShadow {
    // Slices actually handed out; the array layers, the framebuffers, and the
    // data buffer all carry exactly this many entries.
    pub(super) fn count(&self) -> u32 {
        u32::saturating_from(self.framebuffers.len())
    }

    // Slices that re-render this frame.
    pub(super) fn refreshed_slices(&self) -> impl Iterator<Item = u32> {
        spot::refreshed_slices(self.render_mask, self.count())
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
// live in per-(frame, object) storage buffers the skin fold reads.
pub(super) struct VkSkinned {
    pub(super) vertex_buffer: PooledBuffer,
    pub(super) index_buffer: PooledBuffer,
    // Current byte sizes of the skinned VB / IB. Used by
    // `update_skinned_mesh_geometry` to bound-check the slot region the asset
    // hot-reload write lands in. Zero until `upload_skinned` runs.
    pub(super) vertex_buffer_bytes: u64,
    pub(super) index_buffer_bytes: u64,
    // Per-(frame, object) joint storage buffers (host-mapped). Indexed
    // [frame_idx][skinned_idx].
    pub(super) joint_buffers: Vec<Vec<PooledBuffer>>,
    // GPU-driven main-pass skinning fold. `skin` is the `rt_skin` compute pipeline
    // (reused independently of RT) + its per-(frame, object) descriptor sets,
    // written once in `build_main_skin`. `deformed` is one storage+vertex buffer
    // per frame-in-flight holding this frame's posed 56-byte `Vertex`s (global
    // skinned indexing, so the draw uses `base_vertex = 0`); `encode_skin` writes
    // it each frame and the bindless main pass's 2nd indirect draw reads it. Both
    // `None`/empty until `upload_skinned` runs with the bindless cull path active.
    pub(super) skin: Option<super::raytrace::SkinPipeline>,
    pub(super) deformed: Vec<super::raytrace::DeviceBuffer>,
    // Morph targets, parallel to the scene's skinned slots. `morph_delta_unique`
    // owns the per-mesh packed sparse morph device buffers
    // (`PayloadMorphs::packed_words`, deduped by source
    // `Arc`); `morph_delta_buffers[i]` is object `i`'s handle into them (null =
    // morphless). `morph_target_counts[i]` is its target count (0 = none). The
    // per-(frame, object) host-mapped `morph_weight_buffers`
    // ([frame_idx][skinned_idx], one f32 per target) are filled from
    // the scene's morph weights by `upload_morph_weights`, and are empty when no
    // skinned object carries morphs. The skin descriptor sets' morph bindings
    // (3 = deltas, 4 = weights) are re-pointed in `upload_skinned_morphs`.
    pub(super) morph_delta_unique: Vec<PooledBuffer>,
    pub(super) morph_delta_buffers: Vec<vk::Buffer>,
    pub(super) morph_target_counts: Vec<u32>,
    pub(super) morph_weight_buffers: Vec<Vec<PooledBuffer>>,
    // `false` until the deformed-vertex ring has been posed at least one full
    // frame. While false the GPU-driven G-buffer velocity binds the current
    // deformed buffer as the previous one (prev_pos == cur_pos), so an unposed
    // ring slot never feeds a garbage skinned motion vector on the first frame
    // (or after a runtime ring rebuild). Reset by `build_main_skin` / `upload_skinned`. Atomic, not `Cell`: the G-buffer
    // pass encodes on a `jobs::pool()` rayon worker thread (the parallel per-pass
    // encoder shares `&self` across workers), so any interior mutation reachable
    // from `encode_pass_into` must be atomic, like `draw_calls_accum`.
    pub(super) deformed_primed: std::sync::atomic::AtomicBool,
}

impl VkSkinned {
    // No skinned mesh uploaded yet: `upload_skinned` builds the rest.
    pub(super) fn new() -> Self {
        Self {
            vertex_buffer: PooledBuffer::null(),
            vertex_buffer_bytes: 0,
            index_buffer: PooledBuffer::null(),
            index_buffer_bytes: 0,
            joint_buffers: Vec::new(),
            skin: None,
            deformed: Vec::new(),
            morph_delta_unique: Vec::new(),
            morph_delta_buffers: Vec::new(),
            morph_target_counts: Vec::new(),
            morph_weight_buffers: Vec::new(),
            deformed_primed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    // Destroy every owned GPU object. Called from `VkContext::drop` after
    // `wait_idle`.
    pub(super) fn destroy(&mut self) {
        self.vertex_buffer = PooledBuffer::null();
        self.index_buffer = PooledBuffer::null();
        self.joint_buffers.clear();
        self.morph_delta_unique.clear();
        self.morph_weight_buffers.clear();
        // GPU-driven main-pass skinning resources.
        self.skin = None;
        self.deformed.clear();
    }
}

// Shared static vertex/index buffers, grouped off the flat `VkContext` field
// soup. Created at init and live for the context's lifetime; the
// streaming and geometry-rebuild paths swap the buffers in place.
pub(super) struct VkGeometry {
    pub(super) vertex_buffer: PooledBuffer,
    pub(super) index_buffer: PooledBuffer,
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

// The global set layout plus the shared pool the per-frame sets are allocated
// from, grouped off the flat `VkContext` field soup. Global set 0 (camera /
// lights / shadow / IBL / SSAO) and the cascade shadow pass's set 0; the
// `*_sets` are allocated from `descriptor_pool` at init and freed with it, as
// are the text, composite and cull sets their own states hold. Post and skinned
// descriptors live in their own pools, not here.
pub(super) struct VkDescriptors {
    pub(super) global_set_layout: OwnedSetLayout,
    pub(super) descriptor_pool: OwnedDescriptorPool,
    pub(super) global_sets: Vec<vk::DescriptorSet>,
    // The cascade shadow pass's set 0 per frame, over `VkShadow`'s layout and
    // uniform ring.
    pub(super) shadow_global_sets: Vec<vk::DescriptorSet>,
}

// Instanced-prop clusters. Every instance folds into the GPU-driven cull
// records, so no pass walks the instances on the CPU. Empty when the world
// declares no `InstancedProp` clusters. `clusters` holds the declared clusters
// (each with its per-instance transforms).
pub(super) struct VkInstanced {
    pub(super) clusters: Vec<InstancedCluster>,
    // Whether any cluster declares LOD alternates. False skips the per-frame
    // per-instance LOD patch: without alternates the base slice written into
    // every frame's draw-args buffer at init is right for the world's life.
    pub(super) any_lod: bool,
}

impl VkInstanced {
    pub(super) fn new(clusters: Vec<InstancedCluster>) -> Self {
        Self {
            any_lod: concinnity_engine::gfx::lod::any_cluster_has_lod(&clusters),
            clusters,
        }
    }
}

// The compute cull kernel and its layouts, plus two-pass occlusion's phase-2
// kernel over the same layout (`None` when two-pass is off).
pub(super) struct VkCullKernels {
    pub(super) pipeline: OwnedPipeline,
    pub(super) pipeline_phase2: Option<OwnedPipeline>,
    pub(super) pipeline_layout: OwnedPipelineLayout,
    pub(super) set_layout: OwnedSetLayout,
}

// GPU-driven cull + bindless static main pass (+ optional two-pass Hi-Z
// occlusion), grouped off the flat `VkContext` field soup. Mirrors the DirectX
// backend's `cull: CullState`. A compute kernel frustum/distance-tests the
// build-time static objects and writes one indirect draw per survivor; the
// bindless main pass issues the whole buffer with one indirect draw. All
// `Some` / non-empty only when the world has anything to GPU-drive. Field names
// are kept verbatim (heterogeneous prefixes, no single cluster prefix to drop).
// The two-pass Hi-Z pyramid + its temporal state live here too.
pub(super) struct VkCull {
    // Packs the static draws' records into `object_buffers` / `draw_args_buffers`.
    pub(super) packer: crate::record_pack::RecordPacker,
    // Bindless static main pass: bucket 0's pipeline, the world default Shader's
    // pair where the world declares one and the engine's otherwise. The bindless
    // descriptor sets are freed with the shared descriptor pool.
    pub(super) bindless_pipeline: Option<OwnedPipeline>,
    pub(super) bindless_pipeline_layout: Option<OwnedPipelineLayout>,
    pub(super) bindless_set_layout: Option<OwnedSetLayout>,
    // Descriptor count `bindless_set_layout`'s binding 1 was built with, and the
    // number of image infos the pool write pads to. 0 when the bindless path is
    // inactive. The shaders declare that array unsized, so this is the only place
    // the length lives.
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
    pub(super) world_pipelines: concinnity_engine::render::shader::world_pipelines::WorldPipelines<
        super::pipeline::BucketPipelines,
    >,
    // Commands reserved per single-stream region (a probe face, a spot slice),
    // fixed at init to the record capacity the buffers were sized for.
    pub(super) bucket_stride: usize,
    // The routes and the bucket regions of the indirect buffers, fixed at
    // init: each bucket's region holds the records it can take.
    pub(super) variants: concinnity_engine::gfx::render_types::GpuCullVariants,
    // Per-frame copies of `variants` with this frame's view, at set 0
    // binding 4 of every cull dispatch.
    pub(super) variant_buffers: Vec<PooledBuffer>,
    // Per frame, the buckets its view can pick, a bit each (see
    // `GpuCullVariants::live_buckets`); the others' regions skip drawing.
    pub(super) live_buckets: Vec<std::sync::atomic::AtomicU64>,
    // The engine's compiled bindless main-pass and pre-pass SPIR-V, retained so
    // a bucket that resolves to the engine default can build its pipelines
    // without recompiling.
    pub(super) bindless_main_spv: super::pipeline::BindlessSpv,
    // The layout every bucket's G-buffer pre-pass pipeline binds (`Some`
    // exactly when `bindless_pipeline` is), and bucket 0's pipeline, from the
    // same programs as `bindless_pipeline`, which exists only while a G-buffer
    // consumer is on and its build succeeded.
    pub(super) prepass_layout: Option<gbuffer::PrepassLayout>,
    pub(super) prepass_pipeline: Option<OwnedPipeline>,
    // One bindless descriptor set per frame-in-flight: binding 0 is that frame's
    // GpuObjectData storage buffer, binding 1 the shared texture pool, binding 2
    // that frame's material parameter table.
    pub(super) bindless_sets: Vec<vk::DescriptorSet>,
    // Per-frame GpuObjectData storage buffers, persistently mapped; rebuilt each
    // frame from `draw.objects[..draw.n_objects]`.
    pub(super) object_buffers: Vec<PooledBuffer>,
    // The material parameter table, one copy per frame at binding 2 of that
    // frame's bindless set. `None` when the bindless pass is inactive.
    pub(super) material_params: Option<super::material_params::VkMaterialParams>,
    // Compute cull kernels + their per-frame sets (bindings 0/1/2 = that frame's
    // object SSBO, draw-args SSBO, indirect-command SSBO). Sets are pool-freed.
    pub(super) cull_kernels: Option<VkCullKernels>,
    pub(super) cull_sets: Vec<vk::DescriptorSet>,
    // Per-frame `GpuDrawArgs` storage buffers, persistently mapped.
    pub(super) draw_args_buffers: Vec<PooledBuffer>,
    // Per-frame indirect draw-command buffers the cull kernel writes and the
    // main pass consumes (`INDIRECT_BUFFER`). Device-local.
    pub(super) indirect_buffers: Vec<PooledBuffer>,
    // Commands each indirect buffer holds ahead of its count slots: every
    // bucket region's, back to back.
    pub(super) indirect_commands: usize,
    // The scan and scatter that pack each cull's kept draws per stream, each
    // frame's group-count scratch (phase 2 reusing phase 1's once its packing
    // has read it), and each frame's phase-1 and phase-2 packing sets.
    pub(super) compact: Option<super::compact::VkCompactKernels>,
    pub(super) group_count_buffers: Vec<PooledBuffer>,
    pub(super) compact_sets: Vec<vk::DescriptorSet>,
    pub(super) compact_sets2: Vec<vk::DescriptorSet>,
    // Per-frame per-object cull-status buffers (one u32 each): phase-1 writes,
    // phase-2 reads. Device-local storage.
    pub(super) cull_status_buffers: Vec<PooledBuffer>,
    // Two-pass Hi-Z occlusion (HizBuild -> Cull2 -> Main2). `occlusion_two_pass`
    // records the world's request; the live resources below are `Some` /
    // non-empty only when it AND the bindless cull path are active.
    pub(super) occlusion_two_pass: bool,
    // Phase-2 cull sets (the pipeline is `VkCullKernels::pipeline_phase2`),
    // allocated from `two_pass_pool`.
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
    // set 1 (the Hi-Z image + per-frame `CullHizParams` UBO). `Some` exactly
    // when the GPU-cull pipeline is active (same gating as `cull_kernels`):
    // the next frame's `Cull` kernel projects each AABB through the previous
    // frame's un-jittered VP and discards objects fully behind the pyramid.
    pub(super) hiz: Option<crate::vulkan::hiz::HiZResources>,
    // False on the first frame and immediately after a swapchain resize (no
    // valid pyramid yet); drives the cull UBO's `hiz_enabled` so the cull
    // kernel falls back to frustum + distance only until a pyramid at the
    // current resolution exists. Set true at the end of `record_frame` once a
    // build has run.
    // GPU-driven shadow views. `shadow_cull_pipeline` is a frustum-only cull
    // kernel (`SHADOW_CULL`, no Hi-Z) over a lean 4-SSBO set (objects +
    // draw-args + the view's group counts + status); one dispatch per
    // re-rendered cascade or spot slice, then that view's packing, writes its
    // lists. `shadow_bindless_pipeline` is a depth-only graphics pipeline whose
    // VS reads `model` from the GpuObjectData SSBO (gl_InstanceIndex) and
    // projects through `light_vps[cascade_idx]` (a push constant); each cascade
    // is then issued with one indirect draw by count for the static + instance
    // stream and one for the skinned tail. `shadow_views` is indexed
    // [frame][cascade]. All `Some`/non-empty only when the bindless cull path is
    // active AND shadows are enabled.
    pub(super) shadow_cull_pipeline: Option<OwnedPipeline>,
    pub(super) shadow_cull_pipeline_layout: Option<OwnedPipelineLayout>,
    pub(super) _shadow_cull_set_layout: Option<OwnedSetLayout>,
    pub(super) shadow_views: Vec<Vec<super::cull::VkShadowView>>,
    pub(super) shadow_bindless_pipeline: Option<OwnedPipeline>,
    pub(super) shadow_bindless_pipeline_layout: Option<OwnedPipelineLayout>,
    // The spot shadow pass's views, indexed [frame][slice], written by the
    // same shadow kernel. Their own, so the spot pass shares no state with the
    // cascade pass. Empty when the world has no shadowed spot.
    pub(super) spot_views: Vec<Vec<super::cull::VkShadowView>>,
    // GPU-driven G-buffer pre-pass. Each bucket's pre-pass pipeline reads the
    // previous-frame model from `prev_model_buffers`; the velocity history for
    // the skinned tail rides the previous-frame deformed buffer. The pass reuses
    // the main pass's `indirect_buffers` (camera frustum, no extra cull).
    // `gbuffer_sets` is one `PrepassLayout` set 2 per frame (GbView UBO + the
    // PREVIOUS frame's history slot + this frame's draw args); the per-frame
    // `prev_model_*` buffers are device-local, written only by
    // `model_history`'s snapshot dispatch. All non-empty only when the bindless
    // cull path is active AND the G-buffer is enabled.
    pub(super) gbuffer_sets: Vec<vk::DescriptorSet>,
    pub(super) prev_model_buffers: Vec<PooledBuffer>,
    // The snapshot kernel that fills `prev_model_buffers`, and the per-frame
    // sets pairing each object buffer with its history slot.
    pub(super) model_history: Option<gbuffer::ModelHistoryPipeline>,
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
        // GPU-driven shadow views. The per-(frame, view) cascade and spot sets
        // are freed with the shared descriptor pool, so only the pipelines, the
        // set layout, and the per-view buffers are destroyed.
        // GPU-driven G-buffer pre-pass. The per-frame `gbuffer_sets` and the
        // snapshot kernel's sets are freed with the shared descriptor pool, so
        // only the pipelines, layouts, and the per-frame model-history buffers
        // are destroyed here.
        self.object_buffers.clear();
        self.material_params = None;
        self.draw_args_buffers.clear();
        self.indirect_buffers.clear();
        self.cull_status_buffers.clear();
        self.variant_buffers.clear();
        self.group_count_buffers.clear();
        self.compact = None;
        self.indirect_buffers2.clear();
        self.shadow_views.clear();
        self.spot_views.clear();
        self.prev_model_buffers.clear();
        self.model_history = None;
    }
}

// Per-frame-in-flight CPU/GPU synchronization primitives, grouped off the flat
// `VkContext` field soup. `image_available` + `in_flight` are one-per-frame-in-
// flight (`frames_in_flight` deep); `render_finished` is one-per-swapchain-image
// (its length tracks the swapchain, so a resize rebuilds it). The ring cursor
// (`current_frame`) and depth (`frames_in_flight`) stay flat on `VkContext`:
// they are read pervasively and are frame-pacing counters, not sync handles.
pub(super) struct VkFrameSync {
    // Signaled by `acquire_next_image`, waited on by that frame's submit.
    pub(super) image_available: Vec<vk::Semaphore>,
    // Signaled by the frame's submit, waited on by its present. Indexed by
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
        for &s in &self.image_available {
            // SAFETY: the semaphore was created from this device and is destroyed exactly once;
            // the caller has already idled the device, so no submission still waits on it.
            unsafe { device.destroy_semaphore(s, None) };
        }
        for &s in &self.render_finished {
            // SAFETY: owned by this frame-sync set like the ones above, and no present still
            // waits on it after the caller's idle wait.
            unsafe { device.destroy_semaphore(s, None) };
        }
        for &f in &self.in_flight {
            // SAFETY: the fence belongs to this device, and the idle wait means no queue
            // submission will still signal it.
            unsafe { device.destroy_fence(f, None) };
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
        // The shared pool (also frees the per-frame "end" buffers).
        // SAFETY: the pool was created from this device and is destroyed exactly once; the caller
        // has already idled the device, so none of its buffers is still executing.
        unsafe { device.destroy_command_pool(self.command_pool, None) };
        // The parallel-recording pools (each frees its own buffer).
        for &pool in self
            .start_command_pools
            .iter()
            .chain(self.pass_command_pools.iter())
        {
            // SAFETY: each per-frame pool is owned here alone, and after the idle wait no worker
            // records into it and no submission still runs its buffer.
            unsafe { device.destroy_command_pool(pool, None) };
        }
    }
}

// The main-pass view + light uniform buffers, grouped off the flat `VkContext`
// field soup. `view_ubo_*` and `light_ubo_*` are one host-mapped buffer per
// frame-in-flight, written by `record_frame`. NOTE the field names collide with
// the per-pass resource structs (decal / glass / raymarch / particle / gbuffer
// each own their own `view_ubo_*`), so accesses are always anchored on the
// `self.<field>` form, never a bare leading-dot.
pub(super) struct VkUniforms {
    // Per-frame-in-flight `ViewUniforms` UBO (camera + IBL params), persistently
    // mapped. `record_frame` memcpys this frame's view into its slot.
    pub(super) view_ubo_buffers: Vec<PooledBuffer>,
    // Per-frame-in-flight `ProbeSet` UBO (the live reflection-probe count),
    // bound at global set 0 binding 7, persistently mapped. `upload_probe_set`
    // writes this frame's slot; the count stays 0 (sky reflection) until a
    // probe bakes.
    pub(super) probe_set_ubo_buffers: Vec<PooledBuffer>,
    // Per-frame-in-flight `LightUniforms` UBO, persistently mapped. Every pass
    // that reads the light block binds this frame's slot: the global set, the
    // planar mirror sets, and the raymarch view sets are all built per frame
    // against the same ring.
    pub(super) light_ubo_buffers: Vec<PooledBuffer>,
    pub(super) local_light_buffer: PooledBuffer,
    // The values the ring carries, which slots of it are stale, and the local
    // lights' reach. `record_frame` writes the frame's own slot, so no
    // in-flight read is ever raced.
    pub(super) lights: concinnity_engine::render::light::state::LightState,
    // Per-frame-in-flight canopy block (global set 0 binding 21), persistently
    // mapped and rewritten each frame, and the block last written, which a
    // probe capture snapshots.
    pub(super) canopy_buffers: Vec<PooledBuffer>,
    pub(super) canopy: concinnity_engine::render::uniforms::canopy::CanopyParams,
    // The canopy's view tables (binding 22).
    pub(super) canopy_tables: super::canopy::VkCanopyTables,
}

impl VkUniforms {
    // Drop the per-frame view + light UBOs and the local-light SSBO (they
    // retire through the allocator).
    pub(super) fn destroy(&mut self) {
        self.view_ubo_buffers.clear();
        self.probe_set_ubo_buffers.clear();
        self.canopy_buffers.clear();
        self.canopy_tables.clear();
        self.light_ubo_buffers.clear();
        self.local_light_buffer = PooledBuffer::null();
    }
}

// Projected decals. `resources` (pipeline + unit-cube buffers + per-frame
// uniforms + per-decal albedo sets) is always built so runtime `add_decal`
// works from a world that started empty; the encoder simply skips when no slot
// is live or every live decal culls. `set` is the shared slot table, indexing
// the per-decal albedo sets and the per-frame params ring by decal id.
pub(super) struct DecalState {
    pub resources: Option<crate::vulkan::decal::DecalResources>,
    pub set: decal::DecalSet,
}

// Volumetric fog. `resources` is `Some` only when the world declared a
// `VolumetricFog` asset; with none it and `settings` both stay `None` and the
// fog pass is skipped entirely. The settings are cached so the per-frame
// encoder can build its `FogParams` without re-resolving the asset. The sun it
// lights with is the light state's.
pub(super) struct FogState {
    pub resources: Option<crate::vulkan::fog::FogResources>,
    pub settings: Option<volumetric_fog::FogSettings>,
}

// Auto-exposure (EV adaptation). `resources` is `Some` only when
// `Exposure.auto` is enabled; it holds the build + average
// compute pipelines, histogram + output buffers, and the per-frame readback
// buffers. `adaptation` carries the tunables, the authored EV bias and the EMA
// target, and `last_elapsed` the previous frame's elapsed time used to derive
// `dt` for the EMA.
pub(super) struct AutoExposureState {
    pub resources: Option<crate::vulkan::auto_exposure::AutoExposureResources>,
    pub adaptation: Option<auto_exposure::ExposureAdaptation>,
    pub last_elapsed: f32,
}

// Composite (post-process) pass: tonemaps the HDR resolve image onto the
// swapchain, with the text overlay drawn here too, post-tonemap. The
// framebuffers are one per swapchain image; `sets` is one per frame-in-flight
// slot, binding the matching HDR resolve image (binding 0), the bloom chain's
// top octave (binding 1), and the 3D color LUT (binding 2), read through the
// post sampler.
pub(super) struct CompositeState {
    pub render_pass: OwnedRenderPass,
    pub framebuffers: Vec<OwnedFramebuffer>,
    pub pipeline: OwnedPipeline,
    pub pipeline_layout: OwnedPipelineLayout,
    pub _set_layout: OwnedSetLayout,
    pub sets: Vec<vk::DescriptorSet>,
}

// HUD text pass: the glyph atlases with their set layout and one set per atlas,
// the pipeline (`None` when the world has no atlas), its layout, the sampler
// held for lifetime, and the
// per-frame-slot persistent upload buffers for transient text geometry. Each
// upload slot's cursor resets and its buffer grows inside the ring's `reserve`,
// which the composite pass calls once the frame fence confirms the GPU is done
// with that slot.
pub(super) struct TextState {
    pub atlas_textures: Vec<GpuImage>,
    pub _set_layout: OwnedSetLayout,
    pub atlas_sets: Vec<vk::DescriptorSet>,
    pub pipeline: Option<OwnedPipeline>,
    pub pipeline_layout: OwnedPipelineLayout,
    pub _sampler: OwnedSampler,
    pub upload: super::upload_ring::UploadRing,
}

// Scene-captured reflection probes and the staggered bake that fills them,
// driven each frame by `bake_pending_probes` through the shared `ProbeBake`.
pub(super) struct ProbeState {
    // Placements (declared `ReflectionProbe`s or an auto-seeded grid), supplied
    // once after construction via `set_reflection_probes`, the record of every
    // installed probe (the live count the forward / SSR / RT / transparent
    // shaders read) and the queue handing placements to the bake.
    pub book: ProbeBook,
    // The cube array the bake writes a cube of per placement, and the per-frame
    // record buffers. Distinct from `env_map`; sampled only by the specular
    // reflection term.
    pub gpu: super::probe_set::ProbeSetGpu,
    // At most one probe capturing (six faces submitting one per frame, on
    // per-face fences) and one convolving into its cube on the GPU, one
    // destination mip per frame.
    pub bake: super::probe::VkProbeBake,
    // The three convolution kernels and the layouts they bind, built at init under
    // the same gate the bake needs. `None` disables baking.
    pub prefilter: Option<super::probe_prefilter::ProbePrefilterPipelines>,
}

impl ProbeState {
    // No placements and nothing baked, so reflections read the sky until
    // `set_reflection_probes` supplies placements and the bake installs cubes.
    pub(super) fn new(
        prefilter: Option<super::probe_prefilter::ProbePrefilterPipelines>,
        gpu: super::probe_set::ProbeSetGpu,
    ) -> Self {
        Self {
            book: ProbeBook::new(),
            gpu,
            bake: super::probe::VkProbeBake::default(),
            prefilter,
        }
    }
}

// Stall-free texture streaming. A streamed slot swap replaces `textures[slot]`
// immediately but cannot rewrite the per-frame bindless pool descriptors while
// their frames are pending; `pool_rewrites` carries the slot to each frame
// slot's copy right after its fence wait (`apply_streamed_texture_rewrites`).
// The replaced image and the upload's transient resources are parked on
// `retires` against the monotonic `frame` tick and freed `retire_depth`
// (`frames_in_flight + 1`) ticks later: by then every pool copy has been
// re-pointed, every frame recorded against the old view has retired, and the
// tick's fence wait covers the upload submission itself (a swap lands between
// frames, after the previous frame's submit, so the first frame fence that
// covers it is the one signaled by the NEXT draw, and a frame-slot-keyed drain
// would free it too soon).
pub(super) struct StreamState {
    pub pool_rewrites: rewrites::SlotRewriteQueue,
    pub frame: u64,
    pub retires: RetirePool<StreamedUploadRetire>,
    pub retire_depth: u64,
}

impl StreamState {
    pub(super) fn new(frames: usize) -> Self {
        Self {
            pool_rewrites: rewrites::SlotRewriteQueue::new(frames),
            frame: 0,
            retires: RetirePool::new(),
            retire_depth: frames as u64 + 1,
        }
    }
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

impl SwapchainState {
    // The swapchain a live world reload hands its successor: the same handle
    // and images, without the views each context creates and frees itself.
    pub(super) fn share(&self) -> Self {
        Self {
            loader: self.loader.clone(),
            handle: self.handle,
            images: self.images.clone(),
            image_views: Vec::new(),
            format: self.format,
            extent: self.extent,
            last_present_index: None,
        }
    }
}

// The render-resolution scene targets: the main render pass with its
// per-frame-in-flight HDR attachments and framebuffers, and the transient image
// pool. `rebuild_swapchain` rebuilds everything but the render pass.
pub(super) struct VkTargets {
    // Resolution the 3D scene is rendered at. Equals `swapchain.extent` unless
    // temporal upscaling is active, in which case it is
    // `round(swapchain.extent * upscale_scale)` and an FSR pass reconstructs the
    // swapchain-resolution image. Every off-screen scene pass (main, velocity,
    // SSR, SSAO, decals, fog, raymarch, glass, particles, auto-exposure, Hi-Z)
    // sizes its targets + viewports to this; bloom / composite / swapchain stay
    // at `swapchain.extent` (display resolution).
    pub(super) render_extent: vk::Extent2D,
    pub(super) main_render_pass: OwnedRenderPass,
    pub(super) msaa_samples: vk::SampleCountFlags,
    // Off-screen HDR attachments, one set per frame-in-flight slot (indexed by
    // `current_frame`). The main pass renders into these; the composite pass
    // samples `hdr_resolve_images`.
    pub(super) color_images: Vec<GpuImage>, // MSAA HDR color; empty when msaa == 1
    pub(super) depth_images: Vec<GpuImage>, // MSAA depth
    pub(super) hdr_resolve_images: Vec<GpuImage>, // single-sample HDR resolve target
    // The reactive mask the particle and transparent passes write, one per
    // frame-in-flight slot (see `vulkan/reactive_mask.rs`).
    pub(super) reactive_mask_images: Vec<GpuImage>,
    // Main-pass framebuffers (one per frame-in-flight slot): HDR color +
    // depth (+ resolve when multisampled).
    pub(super) framebuffers: Vec<OwnedFramebuffer>,
    // Backing store for the render graph's transient images (the resources the
    // aliasing planner manages). Owns each managed transient's image + memory;
    // features read them back by label and the executor's barrier registry
    // resolves them the same way. It manages the transients the planner in
    // crates/concinnity-engine/src/render/render_graph/transient.rs declares.
    pub(super) transient_pool: super::transient_pool::TransientImagePool,
}

// The world's sampled scene assets: the shared texture pool and its fallbacks,
// the IBL cubes and color-grading LUT, the samplers they are read with, and the
// white stand-in for disabled SSAO. None of it depends on the swapchain.
pub(super) struct VkSceneAssets {
    // Shared texture pool: every texture (albedo, normal map, emissive/ORM,
    // terrain secondary) lives here once at its handle, matching DX/Metal.
    pub(super) textures: Vec<GpuImage>,
    // The reserved fallbacks a draw without a normal map or albedo samples; their
    // pool slots follow the last real texture.
    pub(super) fallback_textures: Vec<GpuImage>,
    pub(super) linear_sampler: OwnedSampler,
    // The anisotropic degree `linear_sampler` filters at, which the ocean's
    // sampler shares.
    pub(super) anisotropy: f32,
    // Trilinear clamp sampler the IBL cubes, the probe cubes and the LTC tables
    // are read through.
    pub(super) cube_sampler: OwnedSampler,
    // Owned IBL cube textures.
    pub(super) env_map: EnvironmentMapTextures,
    // Number of mip levels in the bound IBL prefilter cubemap. 0 = no
    // EnvironmentMap declared; the fragment shader then uses the flat albedo
    // ambient term instead of IBL.
    pub(super) prefilter_mip_count: u32,
    // 3D color-grading LUT sampled in the composite pass. Holds the declared
    // `ColorLut` payload, or a 2x2x2 identity LUT when the world declares none.
    pub(super) color_lut: GpuImage,
    // 1x1 white fallback bound at set 0 binding 6 when SSAO is off.
    pub(super) ssao_white: GpuImage,
}

// The hardware ray-tracing scene: the acceleration structures and the policy
// that keeps them current as the draw set changes.
pub(super) struct VkRayTracing {
    // The scene BLAS / TLAS + geometry table. `Some` only while the RT pass is
    // live and the scene has participating geometry: it is seeded when the
    // first geometry appears and dropped once nothing can be traced.
    pub(super) accel: Option<crate::vulkan::raytrace::RtAccelData>,
    // Dropped BVHs, held until the frames in flight have finished tracing them,
    // timed against `retire_tick`.
    pub(super) retired: RetirePool<crate::vulkan::raytrace::RtAccelData>,
    // Advanced once per frame by `rt_dynamic_update`.
    pub(super) retire_tick: u64,
    // How the TLAS is kept current when props move (the launch's `--rt-dynamic`
    // request); read by the per-frame `rt_dynamic_update`. Inert when `accel`
    // is `None`.
    pub(super) dynamic_mode: concinnity_engine::render::rt_accel::geom::RtDynamicMode,
    // Whether skinned meshes join the TLAS (the launch's `--rt-skinned-geometry`
    // request; in by default). Clear it and the BVH covers static + instanced
    // geometry only, isolating the skinned trace path.
    pub(super) skinned_geometry: bool,
    // Whether the per-frame BVH update is failing, so a failure is logged once
    // per streak rather than every frame.
    pub(super) update_streak: concinnity_engine::render::rt_accel::FailureStreak,
    // The compute-skinning pipeline (`rt_skin`) skinned geometry joins the BVH
    // through, built with the RT pass. `None` when the kernel failed to compile.
    pub(super) skin: Option<crate::vulkan::raytrace::SkinPipeline>,
}

// The device layer every per-world resource is built on: instance, device,
// surface, queues, allocator and window, plus the capabilities and display
// settings negotiated with them. Declared so the device and allocator handles
// drop before the window closes.
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent device features and the present mode"
)]
pub(super) struct VkHardware {
    pub(super) instance: ash::Instance,
    // Owns the logical device: the instance and the entry stay alive for as
    // long as it does, and every Vulkan object the backend owns retires through
    // its queue. Derefs to `ash::Device`.
    pub(super) device: VkDevice,
    pub(super) physical_device: vk::PhysicalDevice,
    pub(super) surface: vk::SurfaceKHR,
    pub(super) surface_loader: ash::khr::surface::Instance,
    pub(super) graphics_queue: vk::Queue,
    pub(super) present_queue: vk::Queue,
    pub(super) graphics_family: u32,
    // The device allocator every pooled buffer / image is placed through. Ticked
    // once per frame in `draw_frame`; drained in Drop after every pooled holder
    // has been torn down. A live reload shares it with the successor, so the
    // rebuilt world places into the blocks the old world's leases released.
    pub(super) alloc: super::allocator::DeviceAllocator,
    // Timestamp query pool with `SLOTS_PER_FRAME * frames_in_flight` slots.
    // `record_frame` writes the frame and per-pass pairs; the CPU reads the
    // previous trip's block at the top of `draw_frame` after the matching fence
    // wait. `None` when the queue does not expose timestamps; `gpu_frame_us`
    // then stays 0.
    pub(super) timestamp_query_pool: Option<vk::QueryPool>,
    // `timestamp_period` from the physical device, in nanoseconds per tick.
    pub(super) timestamp_period_ns: f32,
    // Reads the GPU's timestamp counter now, to place a frame's timestamps on
    // the host clock. `None` without `VK_EXT_calibrated_timestamps`.
    pub(super) timestamp_calibration: Option<ash::ext::calibrated_timestamps::Device>,
    // Draws a cull's lists by their GPU-written counts; `None` without
    // `VK_KHR_draw_indirect_count`, where each list draws at full capacity
    // over commands zeroed before the scatter.
    pub(super) draw_count: Option<ash::khr::draw_indirect_count::Device>,
    // `VK_EXT_memory_budget` device-local heap indices summed for the
    // VRAM-residency chip. Empty when the extension is unavailable; the chip
    // then reports 0.
    pub(super) device_local_heaps: Vec<u32>,
    // `true` when `device_local_heaps` should be queried via
    // `VK_EXT_memory_budget`.
    pub(super) memory_budget_supported: bool,
    // Whether the device is RT-capable (the ray-query extensions + features were
    // enabled at device creation, and XeSS is not active). Enabled whenever
    // capable -- independent of whether RT is on at launch -- so a live
    // settings rebuild can bring RT up at runtime (a device
    // extension cannot be enabled after `create_device`). Read by `upload_skinned`
    // to add the AS-build / storage / device-address flags to the skinned VB/IB
    // whenever capable, and by the RT toggle to reject an enable on an incapable
    // device.
    pub(super) rt_capable: bool,
    // Whether `descriptorBindingSampledImageUpdateAfterBind` was enabled at device
    // creation, letting the bindless texture pool's set layout declare itself
    // update-after-bind and budget against the far larger update-after-bind
    // sampler limit. Only true on a sampler-constrained device (MoltenVK); every
    // desktop driver keeps the plain layout.
    pub(super) update_after_bind: bool,
    // Resolved swapchain color-output mode, selected when the world's
    // `Display.hdr` was on AND the surface advertised a
    // matching HDR color space via the `VK_EXT_swapchain_colorspace` instance
    // extension. Two HDR flavors: `HdrEncoding::ExtendedLinear` runs the
    // swapchain in `R16G16B16A16_SFLOAT` + `EXTENDED_SRGB_LINEAR_EXT` (scRGB
    // linear) and the composite emits linear extended-range values;
    // `HdrEncoding::Pq` (requested via `hdr_pq`, only when an `HDR10_ST2084_EXT`
    // pair is advertised) runs the swapchain in that color space and the
    // composite PQ-encodes (SMPTE ST 2084) in-shader. On SDR the swapchain runs
    // in `BGRA8_UNORM` + sRGB-nonlinear and the ACES + gamma + FXAA + LUT path
    // runs unchanged. Mirrors `DxHardware::hdr_mode`. Stored so the swapchain
    // rebuild path preserves the format + color space on resize.
    pub(super) hdr_mode: hdr::HdrOutputMode,
    // Lock presentation to the display refresh. Captured so `rebuild_swapchain`
    // re-selects the same present mode (FIFO vsync vs MAILBOX uncapped) on resize.
    pub(super) vsync: bool,
    // The swapchain-level config (frames-in-flight / HDR mode) this context was
    // built with, reported by `hot_swap_config` so a live editor reload
    // (`reload_world`) reuses this backend in place only when the new world's
    // `swapchain_config` still matches; a mismatch routes to a full rebuild.
    // Mirrors `DxHardware::swapchain_config`.
    pub(super) swapchain_config: init::SwapchainConfig,
    // Window + input (native Win32 on Windows, AppKit on macOS, GLFW on Linux).
    // `Option` so a `reload_world` can MOVE the live window (with its cursor /
    // menu / keymap state) into the successor context instead of opening a new
    // OS window; `None` only transiently on the outgoing context, which is
    // dropped immediately after (see the `window` / `window_mut` accessors).
    pub(super) window: Option<super::window::PlatformWindow>,
    // Keep Entry alive for the lifetime of the instance
    pub(super) entry: ash::Entry,
}

impl VkHardware {
    // The hardware an outgoing context hands its successor on a live editor
    // `reload_world`: dispatch-table and allocator clones over the same
    // underlying objects, with the window and the timestamp pool moved out so
    // the outgoing `Drop` leaves them alone. Vulkan handles are not refcounted,
    // so that `Drop` also skips the shared surface and swapchain (gated on
    // `reused_by_successor`).
    pub(super) fn hand_over(&mut self) -> error::RenderResult<Self> {
        Ok(Self {
            window: Some(self.window.take().ok_or_else(|| {
                error::RenderError::Other("apply_world_reload: window already taken".into())
            })?),
            instance: self.instance.clone(),
            device: self.device.clone(),
            physical_device: self.physical_device,
            surface: self.surface,
            surface_loader: self.surface_loader.clone(),
            graphics_queue: self.graphics_queue,
            present_queue: self.present_queue,
            graphics_family: self.graphics_family,
            alloc: self.alloc.clone(),
            timestamp_query_pool: self.timestamp_query_pool.take(),
            timestamp_period_ns: self.timestamp_period_ns,
            timestamp_calibration: self.timestamp_calibration.clone(),
            draw_count: self.draw_count.clone(),
            device_local_heaps: self.device_local_heaps.clone(),
            memory_budget_supported: self.memory_budget_supported,
            rt_capable: self.rt_capable,
            update_after_bind: self.update_after_bind,
            hdr_mode: self.hdr_mode,
            vsync: self.vsync,
            swapchain_config: self.swapchain_config,
            entry: self.entry.clone(),
        })
    }
}

pub(crate) struct VkContext {
    // Swapchain. See [`SwapchainState`].
    pub(super) swapchain: SwapchainState,
    // Render-resolution scene targets. See [`VkTargets`].
    pub(super) targets: VkTargets,

    // Composite (post-process) pass. See [`CompositeState`].
    pub(super) composite: CompositeState,
    // What each frame slot's sets and framebuffers naming a rebuildable
    // feature's output were last wired with. See `feature_wires.rs`.
    pub(super) feature_wires: super::feature_wires::FeatureWires,

    // Cascaded shadow map + its pipelines, framebuffers, UBO, and sampler.
    pub(super) shadow: VkShadow,

    // Spot shadow map resources. See [`VkSpotShadow`].
    pub(super) spot_shadow: VkSpotShadow,

    // Rectangular area-light resources. See [`VkAreaLight`].
    pub(super) area_light: VkAreaLight,

    // Sampled scene assets. See [`VkSceneAssets`].
    pub(super) scene: VkSceneAssets,

    // Pipelines
    // GPU-driven cull + bindless static main pass + two-pass Hi-Z occlusion
    // (pyramid + temporal state). See `VkCull`.
    pub(super) cull: VkCull,
    // Clustered light binning: the compute pipeline (built only when the world
    // has local lights), the per-cluster light-index buffer, and the
    // `ClusterParams` UBOs. See `VkLightCull`.
    pub(super) light_cull: super::light_cull::VkLightCull,

    // HUD text pass. See [`TextState`].
    pub(super) text: TextState,

    // The shared bloom chain. `None` only once torn down.
    pub(super) bloom: Option<VkBloomPass>,
    // Post-process tunables (bloom intensity / threshold / knee, exposure,
    // vignette). Drives whether the bloom passes run and feeds the composite
    // + bloom-prefilter push constants.
    pub(super) post_process: PostProcessParams,
    // The scale the frame's HDR chain is written at, advanced once per frame.
    pub(super) pre_exposure: auto_exposure::PreExposure,

    // Temporal anti-aliasing resources. `Some` only when the world's
    // `AntiAliasing` set `mode: taa`; `None` skips the velocity pre-pass
    // and history resolve entirely (and the projection jitter with them).
    // Also forced `Some` when temporal upscaling is on (FSR consumes the
    // velocity pre-pass's motion + depth), in which case the TAA *resolve* is
    // dropped from the graph and `Upscale` runs in its slot.
    pub(super) taa: Option<TaaResources>,

    // What every shared fullscreen post pass draws through: the cached render
    // passes and framebuffers, and the per-frame descriptor arena. Held once for
    // the backend rather than once per effect, which is the point of the seam.
    pub(super) post: PostSupport,

    // Temporal upscaling (FSR / DLSS / XeSS, behind `VkUpscaleBackend`). `Some`
    // only when the world's `AntiAliasing` set `upscaling: true`
    // AND a backend resolved + built; `None` renders at native resolution
    // (`render_extent == swapchain.extent`). When `Some`, the scene renders at
    // the reduced `render_extent` and this pass reconstructs the swapchain
    // resolution; bloom + composite sample its output.
    pub(super) upscale: Option<Box<dyn VkUpscaleBackend>>,
    // The pre-exposure the upscaler divides its input by, stepped between
    // frames so its history stays in FP16's normal range.
    pub(super) upscale_exposure: auto_exposure::UpscalerExposure,

    // The upscaler and model the world requested. Kept so a swapchain resize
    // rebuilds the same backend via `build_upscaler` (the DLSS / XeSS device
    // extensions are fixed at device creation, so the resize must re-resolve to
    // the same first choice; it does, deterministically).
    pub(super) upscale_requested: crate::upscale_sdk::UpscaleRequest,

    // Screen-space ambient occlusion (GTAO) resources. `Some` only when the
    // world's `AmbientOcclusion` set `enabled: true`; `None` binds the
    // `ssao_white` 1×1 fallback at set 0 binding 6 so the main pass's SSAO
    // multiplier is a constant 1.0.
    pub(super) ssao: Option<SsaoResources>,

    // Screen-space reflections. `Some` whenever SSR, SSGI or RT reflections are on,
    // since all three share its pre-pass G-buffer; its settings are `Some` only
    // when SSR itself is authored (`ssr_resolve_active` says whether it runs).
    pub(super) ssr: Option<SsrResources>,

    // Roughness-aware reflection composite. `Some` whenever a reflection path owns
    // the post-stack scene image (the SSR resolve is active OR RT reflections are
    // active). Both resolves write reflected radiance + weight into their output
    // target, then this blurs by roughness and composites over the scene into
    // `reflection_composite.output` -- the scene image the post stack consumes in
    // place of the raw resolve output. Mirrors `DxContext::reflection_composite`.
    pub(super) reflection_composite: Option<VkReflectionCompositePass>,

    // Screen-space global illumination. `Some` only when the world's
    // `IndirectLighting` selected `technique: ssgi`. The pyramid,
    // trace and composite run on the hdr_resolve RMW chain after the main pass, reusing
    // `ssr`'s pre-pass G-buffer.
    pub(super) ssgi: Option<SsgiResources>,

    // The reactive mask's disocclusion mark. `Some` while an upscaler that
    // reads the mask runs over the G-buffer pre-pass.
    pub(super) disocclusion: Option<super::post::VkDisocclusionPass>,

    // Unified geometry G-buffer pre-pass. `Some` whenever any screen-space
    // consumer of the merged buffer is on (SSR resolve OR SSGI OR RT OR SSAO OR
    // velocity for TAA / upscale): one jittered traversal rasterizes the
    // normal+depth / roughness / velocity MRT every reader samples, replacing
    // the separate SSR / SSAO / velocity pre-passes (the `PassId::GBufferPrepass`
    // node). Mirrors `DxContext::gbuffer`.
    pub(super) gbuffer: Option<GbufferResources>,

    // Hardware ray-traced reflections (`VK_KHR_ray_query`). `rt_reflections` (the
    // fullscreen inline-`rayQueryEXT` pass + its output target) is `Some` when
    // the world set `ray_traced_reflections: true`, the GPU exposed the
    // ray-query extensions, and the pass built; otherwise the graph falls back
    // to `SsrResolve`. `rt.accel` (the scene BLAS / TLAS + geometry table) lives
    // under it only while there is geometry to trace. Like SSGI, RT reuses
    // the SSR depth + normal + roughness pre-pass G-buffer (so `ssr` is built
    // whenever RT is on), and it replaces the SSR *resolve* in the frame graph:
    // when `rt_reflections_active()` the reflection composite blends
    // `rt_reflections.output` over the scene (RT takes precedence over SSR, which
    // stays the non-RT-GPU fallback).
    pub(super) rt_reflections: Option<RtReflectionsResources>,
    // Ray-tracing acceleration state. See [`VkRayTracing`].
    pub(super) rt: VkRayTracing,

    // Projected decals. See [`DecalState`].
    pub(super) decal: DecalState,
    // World-space line pass state: the resources, built on the first frame
    // that publishes lines. See [`crate::vulkan::line::LineState`].
    pub(super) lines: crate::vulkan::line::LineState,

    // Volumetric fog. See [`FogState`].
    pub(super) fog: FogState,

    // Raymarched SDF volumes. `Some` only when the world declared at least one
    // `SdfVolume`; the `Raymarch` pass is omitted from the frame graph otherwise. Built at init; the encoder
    // composites each visible volume into the scene between `AutoExposure` and
    // `Decals`. While present, the main pass switches to a STORE-color render
    // pass (MSAA) so this pass can load + re-resolve the multisampled color.
    pub(super) raymarch: Option<crate::vulkan::raymarch::RaymarchResources>,

    // The environment drawn behind the opaque scene. See `vulkan/sky.rs`.
    pub(super) sky: crate::vulkan::sky::VkSky,
    // The world's atmosphere, when it declares one: the sky, the haze and the
    // ambient light. See `vulkan/atmosphere.rs`.
    pub(super) atmosphere: Option<crate::vulkan::atmosphere::VkAtmosphere>,

    // The shared `PassId::Transparent` slot and its two producers, translucent
    // glass panes and water surfaces. `Some` only when the world declared a
    // `GlassPanel` or a `WaterSurface`; with neither the field stays `None` and
    // the pass is omitted from the frame graph (gated on
    // `transparent.any_visible()`). Built at init; the encoder draws every record
    // back-to-front over the post-SSR scene between `SsrResolve` and
    // `TaaResolve`. Mirrors `src/directx/transparent.rs`.
    pub(super) transparent: Option<crate::vulkan::transparent::TransparentResources>,
    // The simulated oceans' cascade arrays, which every water draw binds.
    // `Some` whenever the world declared a `WaterSurface`.
    pub(super) water_sim: Option<crate::vulkan::water_sim::OceanSim>,
    // The wake's history, and this frame's block and lane once a frame that
    // runs it has advanced it.
    pub(super) wake: concinnity_engine::render::environment::water::wake::WakeHistory,
    pub(super) wake_frame: Option<concinnity_engine::render::environment::water::wake::WakeFrame>,

    // Planar reflections for the transparent pass's flat reflectors: one
    // render-resolution mirror render per distinct reflector plane (a water
    // surface's rest plane, a glass pane's plane), sampled by the shaders at
    // screen UV. `Some` only when the world declared reflectors assigned to a
    // planar slot. Mirrors `src/directx/planar.rs`.
    pub(super) planar_reflection: Option<crate::vulkan::planar::PlanarReflectionSet>,

    // GPU-compute particle system. See [`ParticleState`].
    pub(super) particle: crate::vulkan::particle::ParticleState,
    // The grass field, when the world grows one and the GPU-driven main pass
    // exists to draw it in. See [`super::grass`].
    pub(super) grass: Option<super::grass::GrassResources>,
    // How this frame's view resolves the grass: where it coarsens, thins and
    // hands over to the canopy. `None` without grass.
    pub(super) grass_lod: Option<concinnity_engine::render::environment::grass::GrassLodView>,
    // What the canopy block is filled from: the grass's far fields and the
    // planet's ground. `None` when the world has neither.
    pub(super) canopy_field: Option<concinnity_engine::render::environment::grass::CanopyField>,

    // Auto-exposure (EV adaptation). See [`AutoExposureState`].
    pub(super) auto_exposure: AutoExposureState,

    // Built-in shader hot reload. See [`ShaderReload`].
    pub(super) hot_reload: ShaderReload,
    // Closed before teardown, so no pipeline builder handed out by this
    // context builds against a handle it has destroyed.
    pub(super) pipeline_gate: crate::vulkan::pipeline_builder::PipelineGate,
    // The world default Shader's compiled programs, `None` for the engine's
    // own. Kept past init so the built-in shader reload rebuilds bucket 0 from
    // the world's pair rather than the engine's.
    pub(super) world_shader: Option<components::ShaderPrograms>,

    // Per-frame draw-call / VRAM / GPU-time counters surfaced to the
    // profiler overlay via [`Self::render_stats`]. Lives in a `Cell` because
    // the `objects` / `gpu_frame_us` / `vram_bytes` fields are filled from
    // `&mut self` in `draw_frame`. Mirrors `DxContext::frame_stats`.
    pub(super) frame_stats: std::cell::Cell<profile::RenderStats>,
    // Draw-call accumulator the pass encoders bump via `inc_draw_calls`. An
    // `AtomicU32` (not the `frame_stats` Cell) because the parallel
    // command-buffer recording fans the encoders onto rayon workers that bump
    // it concurrently; a `Cell` would be a data race. Reset to 0 at the top of
    // `draw_frame` and drained into `frame_stats.draw_calls` at the end of
    // `record_frame`. Mirrors `DxContext::draw_calls_accum`.
    pub(super) draw_calls_accum: std::sync::atomic::AtomicU32,

    // Main geometry-path descriptor layouts, shared pool, and per-frame sets.
    // See `VkDescriptors`.
    pub(super) descriptors: VkDescriptors,
    // Instanced-prop pipeline, per-cluster material sets, per-frame instance
    // buffers + sets, and the cluster list. See `VkInstanced`.
    pub(super) instanced: VkInstanced,

    // Shared static vertex/index buffers. See `VkGeometry`.
    pub(super) geometry: VkGeometry,

    // Geometry writes staged for the next copy submit. See [`GeometryUploads`].
    pub(super) geometry_uploads: core::cell::RefCell<GeometryUploads>,

    // Skinned (skeletally animated) mesh rendering. See `VkSkinned`.
    pub(super) skinned: VkSkinned,

    // Main-pass view (per-frame) + light (shared) uniform buffers. See
    // `VkUniforms`.
    pub(super) uniforms: VkUniforms,

    // Per-frame-in-flight synchronization primitives. See `VkFrameSync`.
    pub(super) frame_sync: VkFrameSync,
    // Frame captures asked of `draw_frame`: their read-back buffers, the
    // captures awaiting their frame slot's fence, and the writer.
    pub(super) frame_capture: super::screenshot::VkFrameCaptures,
    pub(super) current_frame: usize,
    // When the input each frame slot's last frame showed was sampled, on the
    // host clock, read back beside that frame's timestamps.
    pub(super) frame_input_sampled_us: Vec<Option<u64>>,
    pub(super) frames_in_flight: usize,

    // Per-frame command pools + buffers (start / per-pass / end tiers + the
    // shared one-shot pool). See `VkCommands`.
    pub(super) commands: VkCommands,

    // The CPU-side scene: draw list, view, skinned slots, model history and
    // streamed-geometry placement. See [`SceneState`].
    pub(super) state: SceneState,
    // Scratch the graph executor refills each frame: the per-resource barrier
    // targets, the per-pass aliasing barriers and the per-pass recording slots. Their contents are derived from
    // live state every frame (so no handle can go stale here); only the
    // allocations are carried over, which is what the per-frame `Vec` builds were
    // actually costing. Taken during `execute_graph` and put back after, like
    // `graph_cache` above.
    pub(super) graph_scratch: Option<super::graph_exec::VkGraphScratch>,
    // The frame's submit list, `[start, ...pass buffers, end]`, kept across
    // frames so building it allocates nothing.
    pub(super) submit_scratch: Vec<vk::CommandBuffer>,
    // Lazily-built wireframe twins of the main-pass pipelines; empty until the
    // first Wireframe frame. See [`super::wireframe`].
    pub(super) wireframe: super::wireframe::VkWireframe,

    // Scene-captured reflection probes. See [`ProbeState`].
    pub(super) probe: ProbeState,

    // Stall-free texture streaming. See [`StreamState`].
    pub(super) stream: StreamState,

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

    // The device layer. Declared last so every retiring field above has
    // dropped before the device and the window go. See [`VkHardware`].
    pub(super) hw: VkHardware,
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
#[expect(
    clippy::let_underscore_must_use,
    reason = "the main thread is the same for every context a process makes"
)]
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

impl VkContext {
    #[expect(
        clippy::large_types_passed_by_value,
        reason = "keeps the `RenderBackend::draw_frame` signature it serves"
    )]
    pub(crate) fn draw_frame(&mut self, params: FrameParams<'_>) -> error::RenderResult<()> {
        drive_frame(self, params, concinnity_host::clock::monotonic_micros)
    }

    // The live platform window. Present for every constructed context; `None`
    // only on the outgoing context of a `reload_world` (its window was moved
    // into the successor), which is dropped without any further window access.
    #[inline]
    pub(super) fn window(&self) -> &super::window::PlatformWindow {
        self.hw
            .window
            .as_ref()
            .expect("VkContext window taken by reload_world")
    }

    #[inline]
    pub(super) fn window_mut(&mut self) -> &mut super::window::PlatformWindow {
        self.hw
            .window
            .as_mut()
            .expect("VkContext window taken by reload_world")
    }

    // True while the window is minimized: the client area has collapsed to 0x0
    // (WM_SIZE reports zero on minimize; the GLFW / AppKit windows report the
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
    // so a window that was already minimized when it was created never saw a
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

    pub(crate) fn window_closed(&mut self) -> bool {
        self.window_mut().poll()
    }

    // Submit every staged geometry write. Submission order puts the copies
    // ahead of any later submission, so nothing waits on them.
    pub(super) fn flush_geometry_uploads(&self) -> error::RenderResult<()> {
        self.geometry_uploads.borrow_mut().flush(
            &self.hw.device,
            CopySubmit {
                command_pool: self.commands.command_pool,
                queue: self.hw.graphics_queue,
            },
            GeometryDest {
                vertex: self.geometry.vertex_buffer.buffer(),
                index: self.geometry.index_buffer.buffer(),
            },
        )
    }

    pub(crate) fn wait_idle(&self) {
        // Staged geometry writes count as submitted work, so an idle device has
        // applied them.
        if let Err(e) = self.flush_geometry_uploads() {
            tracing::error!("geometry upload submit failed: {e}");
        }
        // SAFETY: a wait on this device's own queues; it takes no borrowed state.
        if let Err(e) = unsafe { self.hw.device.device_wait_idle() } {
            tracing::error!("device wait idle failed: {e}");
        }
    }

    // Render statistics for the most recent `draw_frame`, for the profiler
    // overlay. `gpu_frame_us` is filled at the top of each `draw_frame`
    // from the timestamp pair this slot resolved on its previous trip
    // through the ring (so the reading is `frames_in_flight`-stale by
    // construction, matching DirectX / Metal). Per-pass GPU timing is
    // still a follow-up.
    pub(crate) fn render_stats(&self) -> profile::RenderStats {
        profile::RenderStats {
            spans: self.state.spans.get(),
            ..self.frame_stats.get()
        }
    }

    // Current device-local memory residency in bytes, via
    // `VK_EXT_memory_budget`. Sums `heap_usage` on every DEVICE_LOCAL heap;
    // returns 0 when the extension is unavailable (so the chip degrades
    // gracefully on adapters that don't expose budgets, matching DirectX's
    // behavior on pre-WDDM-2.0 adapters).
    pub(super) fn query_vram_bytes(&self) -> u64 {
        if !self.hw.memory_budget_supported || self.hw.device_local_heaps.is_empty() {
            return 0;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut props2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        // SAFETY: a property query on a live handle; it only reads.
        unsafe {
            self.hw
                .instance
                .get_physical_device_memory_properties2(self.hw.physical_device, &mut props2);
        }
        self.hw
            .device_local_heaps
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

    pub(crate) fn request_cursor_capture(&mut self) {
        self.window_mut().request_cursor_capture();
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

    pub(crate) fn window_minimized(&self) -> bool {
        self.window().window_minimized()
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
    // `self.hw.vsync`. Edge-triggered: a redundant call (a swapchain rebuild is
    // expensive) is skipped.
    pub(crate) fn set_vsync(&mut self, on: bool) {
        if on == self.hw.vsync {
            return;
        }
        self.hw.vsync = on;
        if let Err(e) = self.rebuild_swapchain() {
            tracing::warn!("set_vsync: rebuild_swapchain failed: {}", e);
        }
    }

    // Switch window mode / resize at runtime. The GLFW work lives in window.rs;
    // the framebuffer-size change drives a swapchain rebuild via the present
    // path's OUT_OF_DATE handling.
    pub(crate) fn set_window_mode(&mut self, mode: components::WindowMode) {
        self.window_mut().set_window_mode(mode);
    }

    pub(crate) fn set_window_size(&mut self, width: u32, height: u32) {
        self.window_mut().set_window_size(width, height);
    }

    // The display modes feeding the Resolution settings row; enumeration,
    // the fullscreen mode hold, and the desktop-mode restore all live in
    // window.rs (GLFW owns the video-mode switching).
    pub(crate) fn display_modes(&self) -> Vec<display_mode::DisplayMode> {
        self.window().display_modes()
    }

    pub(crate) fn current_display_mode(&self) -> Option<display_mode::DisplayMode> {
        self.window().current_display_mode()
    }

    pub(crate) fn display_refresh(&self) -> Option<display_mode::DisplayRefresh> {
        self.window().display_refresh()
    }

    pub(crate) fn set_display_mode(&mut self, mode: display_mode::DisplayMode) {
        self.window_mut().set_display_mode(mode);
    }

    // Replace the live post-process tunables, pushed to the bloom + composite
    // shaders each frame. The composite's display-output flags are not part of
    // the payload, so the EDR path negotiated at init survives every push.
    pub(crate) fn update_post_process(&mut self, tunables: PostProcessTunables) {
        self.post_process.set_tunables(tunables);
    }

    // Set the live ambient (IBL) light scale (the Ambient slider). It lives in
    // `LightUniforms`, which rides a per-frame-in-flight UBO ring, so this
    // mutates the CPU-side copy and re-arms every slot; `record_frame` writes
    // the frame's own slot. Edge-triggered: a no-op when the value is unchanged
    // (e.g. an init push with no persisted override).
    pub(crate) fn set_ambient_intensity(&mut self, value: f32) {
        self.uniforms.lights.set_ambient(value);
    }

    // Replace the live directional lights. The cascade shadow direction and the
    // fog sun are derived from the first light on the CPU, so both are
    // re-derived here. Edge-triggered: an unchanged set touches nothing, and a
    // changed one only re-arms the light UBO ring.
    pub(crate) fn update_directional_lights(&mut self, lights: &[components::DirectionalLight]) {
        self.uniforms.lights.set_directional(lights);
    }

    // Move the local lights in place. The light SSBO, the area table and the
    // spot projections are single host-visible buffers every in-flight frame
    // reads, so the GPU is drained before they are rewritten; the cluster
    // reach and the spot frusta are re-derived, and the light UBO ring is
    // re-armed for the point array.
    pub(crate) fn move_local_lights(
        &mut self,
        data: &LightData,
        uniforms: &LightUniforms,
    ) -> error::RenderResult<()> {
        if !self.uniforms.lights.same_local_count(&data.lights)
            || data.spot_shadows.len() != self.spot_shadow.count() as usize
        {
            return Err(error::RenderError::Other(
                "move_local_lights: the light count changed since init".into(),
            ));
        }
        self.wait_idle();
        super::draw::upload_static_records(&self.uniforms.local_light_buffer, &data.lights);
        super::draw::upload_static_records(&self.area_light.buffer, &data.area_lights);
        self.spot_shadow.rewrite(&data.spot_shadows);
        self.state.planner.shadow.reset_spots();
        self.uniforms.lights.move_local(&data.lights, uniforms);
        Ok(())
    }

    pub(crate) fn set_shadow_cadence(&mut self, cadence: init::ShadowCadence) {
        self.state.planner.shadow.cadence = cadence;
    }

    // Update the live scalar sub-tunables of the SSAO / SSR / SSGI / auto-exposure
    // passes without rebuilding anything. Each pass rebuilds its per-frame uniform
    // from these stored `*Settings` every draw (`settings.params(...)`), so
    // mutating the stored struct here is picked up on the next frame. Only a
    // feature whose resources are currently live has settings to mutate; the rest
    // are skipped (the value still persists for the next launch). SSAO / SSR /
    // auto-exposure are fully scalar, so they are replaced wholesale; SSGI's trace
    // resolution sizes its targets and rides its rebuild with its ray
    // count, so only its scalar intensity / distance are updated.
    pub(crate) fn update_quality_params(
        &mut self,
        g: &concinnity_engine::settings::ResolvedGraphics,
    ) {
        if let (Some(live), Some(cur)) = (g.ssao(), self.ssao.as_mut().map(|s| &mut s.settings)) {
            *cur = live;
        }
        if let (Some(live), Some(cur)) =
            (g.ssr(), self.ssr.as_mut().and_then(|s| s.settings.as_mut()))
        {
            *cur = live;
        }
        if let (Some(live), Some(cur)) = (g.ssgi(), self.ssgi.as_mut().map(|s| &mut s.settings)) {
            cur.intensity = live.intensity;
            cur.max_distance = live.max_distance;
        }
        if let (Some(live), Some(cur)) = (g.auto_exposure(), self.auto_exposure.adaptation.as_mut())
        {
            cur.settings = live;
        }
        self.refresh_gates(g);
    }

    // Public accessor for the shared shader-reload flag. Cloning the `Arc`
    // lets the debug server flip it from a non-render thread.
    // `None` outside `cn debug`. Mirrors `DxContext::shader_reload_pending`.
    pub(crate) fn take_input(&mut self) -> InputSnapshot {
        // Both platform windows snapshot straight into the shared InputSnapshot.
        self.window_mut().take_input()
    }

    // Replace the runtime movement key map. The window's key decode routes
    // through it, so a settings-menu rebind takes effect immediately.
    pub(crate) fn set_keymap(&mut self, keymap: KeyMap) {
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

    // The window chrome overlapping the top of the frame, in the same units.
    // Non-zero only on the macOS window, whose content view runs under a
    // transparent title bar.
    pub(crate) fn top_content_inset(&self) -> f32 {
        self.window().top_content_inset()
    }

    // Device capability flags for the settings menu. RT reflects whether the
    // ray-query device extensions were enabled at device creation
    // (`rt_capable`), which XeSS rules out.
    pub(crate) fn capabilities(&self) -> backend::DeviceCapabilities {
        super::gpu_profile::capabilities(self.hw.rt_capable)
    }

    // Coarse GPU performance profile for default-quality selection, read live
    // from the physical device: vendor id, discrete / integrated device type,
    // and the summed DEVICE_LOCAL heap size as the VRAM budget (the true heap
    // size, unlike the residency chip which sums live usage).
    pub(crate) fn gpu_profile(&self) -> backend::GpuProfile {
        super::gpu_profile::device_profile(&self.hw.instance, self.hw.physical_device)
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
        self.grass = None;
        let device = self.hw.device.clone();
        let device = &device;

        // Abandon any in-flight staggered probe bake: free both slots' command
        // buffers (before `self.commands` is destroyed below) + fences + targets.
        // `wait_idle` above retired their GPU work.
        let (rendering, prefiltering) = self.probe.bake.take();
        if let Some(rendering) = rendering {
            rendering.destroy(device, self.commands.command_pool);
        }
        if let Some(prefiltering) = prefiltering {
            prefiltering.destroy(device, self.commands.command_pool);
        }

        // Parked streamed-texture retires (`wait_idle` above covered them).
        self.drain_stream_retires();

        // Sync (per-frame-in-flight semaphores + fences).
        self.frame_sync.destroy(device);

        // Staged geometry writes: the ring retires through the allocator, and
        // the copy command buffers go with the pool below.
        self.geometry_uploads.get_mut().destroy();

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
        self.scene.env_map.irradiance = GpuImage::null();
        self.scene.env_map.prefilter = GpuImage::null();

        // Pipelines.
        self.wireframe.destroy();
        self.text.pipeline = None;

        // GPU-driven cull + bindless static pass resources, including the Hi-Z
        // pyramid (see `VkCull::destroy`). The bindless / cull / phase-2
        // descriptor sets are freed with the shared descriptor pool +
        // `two_pass_pool`.
        self.cull.destroy(device);
        self.light_cull.destroy(device);

        // Composite pass resources (the LUT retires through the allocator).
        self.scene.color_lut = GpuImage::null();

        // The shared post passes' framebuffers, before the targets whose views
        // they name, then the TAA resolve and the bloom chain themselves
        // (pipelines + targets).
        self.post.cache.destroy();
        self.taa = None;
        self.bloom = None;

        // SSAO resources (kernel + blur pipelines and the raw occlusion).
        self.ssao = None;
        self.scene.ssao_white = GpuImage::null();

        // Transient image pool (the graph-owned transients, e.g. `ao_output`).
        self.targets.transient_pool.destroy(device);

        // SSR resolve (pipeline + reflection target).
        self.ssr = None;

        // Reflection composite (roughness blur + composite of the SSR/RT output).
        self.reflection_composite = None;

        // SSGI resources (pyramids, accumulation and pipelines).
        self.ssgi = None;

        // The disocclusion mark's snapshots and pipelines.
        self.disocclusion = None;

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
        self.rt.destroy_accels();
        self.rt.skin = None;

        // Temporal upscaling (FSR / DLSS / XeSS): the vendor context + the
        // output texture, via the backend trait.
        if let Some(mut up) = self.upscale.take() {
            up.destroy();
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

        // The atmosphere's tables, uniforms and framebuffers.
        if let Some(mut atmosphere) = self.atmosphere.take() {
            atmosphere.destroy();
        }

        // Raymarched SDF volume resources (per-volume pipelines + UBOs, view
        // ring, descriptor pool, render passes, snapshot image).
        if let Some(mut rm) = self.raymarch.take() {
            rm.destroy(device);
        }

        // Planar reflection resources (mirror targets + framebuffers + per-(plane,
        // frame) view ring + global sets + descriptor pool). Destroyed before the
        // transparent pass, whose per-record sets reference the planar target views.
        if let Some(mut planar) = self.planar_reflection.take() {
            planar.destroy(device);
        }

        // Transparent-pass resources (both producers' pipelines, their per-record
        // buffers + UBOs, the per-frame view ring, descriptor pool, render pass,
        // framebuffers, snapshot image).
        if let Some(mut transparent) = self.transparent.take() {
            transparent.destroy(device);
        }

        // The ocean cascades and FFT kernels the water draws sampled; their
        // images and buffers retire through the allocator.
        self.water_sim = None;

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
        if let Some(pool) = self.hw.timestamp_query_pool.take() {
            // SAFETY: the handle was created from this device and is destroyed exactly once; the
            // caller has already waited for the device to go idle, so no submission still
            // references it.
            unsafe { device.destroy_query_pool(pool, None) };
        }

        // Render passes.

        // Skinned-mesh resources.
        self.skinned.destroy();

        // Geometry.
        self.geometry.destroy();

        // UBOs (per-frame view + shared light).
        self.uniforms.destroy();

        // Samplers.

        // Scene textures + the reflection-probe cube array: dropping them retires
        // them through the allocator.
        self.scene.textures.clear();
        self.scene.fallback_textures.clear();
        self.text.atlas_textures.clear();
        self.probe.gpu.release();
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        self.pipeline_gate.close();
        self.wait_idle();

        // No-op on the outgoing context of a `reload_world`, which already
        // freed its world before the successor built.
        self.destroy_world_content();

        // Device allocator: destroy every handle its dropped leases queued and
        // free the blocks. After the content pass above so all leases have
        // dropped, before the device teardown below. On a reload the successor
        // shares this allocator, so only a context that still owns it drains.
        if !self.reused_by_successor {
            // SAFETY: `wait_idle` above left the GPU idle.
            unsafe { self.frame_capture.close() };
            self.hw.alloc.destroy();
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
            unsafe {
                self.hw
                    .surface_loader
                    .destroy_surface(self.hw.surface, None);
            };
        }

        // The device, the instance, the debug messenger and the pipeline cache
        // are the owning
        // device handle's to destroy, once every owned Vulkan object has
        // retired through it. That happens as this context's fields drop,
        // after this body returns, so nothing here has to order it -- and on a
        // live reload the successor's clone simply keeps them alive.
    }
}

// Whether a client-area extent counts as minimized (collapsed): a zero, or
// defensively negative, width or height. Split from `is_minimized` so the
// minimize gate is unit-testable without a live window / surface.
fn extent_minimized(width: i32, height: i32) -> bool {
    width <= 0 || height <= 0
}

#[cfg(test)]
mod tests {
    use super::extent_minimized;

    #[test]
    fn extent_minimized_gates_on_zero_or_negative_dimensions() {
        // A live window (both dimensions positive) is not minimized.
        assert!(!extent_minimized(1280, 720));
        assert!(!extent_minimized(1, 1));
        // Minimize collapses one or both dimensions to zero.
        assert!(extent_minimized(0, 0));
        assert!(extent_minimized(1280, 0));
        assert!(extent_minimized(0, 720));
        // Negative dimensions never form a valid extent either.
        assert!(extent_minimized(-1, 720));
        assert!(extent_minimized(1280, -1));
    }
}
