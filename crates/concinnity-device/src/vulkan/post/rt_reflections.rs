//! Hardware ray-traced reflection pass for the Vulkan backend, screen space
//! first. A classify kernel walks each glossy pixel's reflection ray through
//! SSGI's depth pyramid, writes what the screen and the probe settle into the
//! output, and lists the pixels still owed a hardware ray; a trace kernel
//! dispatched indirectly over that list traces them against the scene's
//! top-level acceleration structure ([`crate::vulkan::raytrace`]) with inline
//! `rayQueryEXT`, shading the hit (sun + IBL split-sum, optionally textured) or
//! the probe / prefilter cube on a miss, and adds their share. The reflection
//! composite then blends the output over the scene.
//!
//! It occupies the `SsrResolve` slot in the frame graph (reads the HDR scene,
//! writes its own `output` target) and is mutually exclusive with the SSR
//! resolve. Like SSGI it reuses the SSR depth + normal + roughness pre-pass
//! G-buffer, so that pre-pass is forced on whenever RT reflections are enabled.
//! While there is no BVH to trace, an authored SSR resolve takes the slot, or
//! the pass clears its output so the reflection composite leaves the scene as
//! it was.
//! Mirrors `src/directx/post/rt_reflections.rs` (DXR inline `RayQuery`).
//!
//! Unlike DirectX (which binds the TLAS + geometry table as root SRVs by GPU
//! virtual address each frame), Vulkan binds them through a descriptor set, so
//! `VkContext::rt_dynamic_update` re-points the current frame's set at the
//! live TLAS + geometry-table handles every frame (they change on a dynamic
//! rebuild; see `crate::vulkan::raytrace`), and at this frame's SSGI pyramid.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types::RtParams;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::post::rt_reflections::{
    RAY_ARGS_RESET, RtReflectionSettings, classify_groups, ray_list_len,
};
use concinnity_engine::render::rt_accel::seed_wanted;

use super::super::allocator::{DeviceAllocator, PooledBuffer};
use super::super::context::{HDR_FORMAT, VkContext};
use super::super::descriptor_layout::{Binding, PoolSizes};
use super::super::pipeline_desc::compute_pipeline;
use super::super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::super::set_writes::SetWrites;
use super::super::texture::{
    GpuImage, ImageSpec, create_image, create_image_view, create_sampler_linear_clamp,
};
use crate::vulkan::builtin_shaders::CompileProgram;
use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedPipeline, OwnedPipelineLayout, OwnedSampler, OwnedSetLayout, VkDevice,
};
use crate::vulkan::record::Recorder;
use crate::vulkan::wire_cache::WireCache;

// Set-0 bindings of the classify's pyramid, the output both kernels write, and
// the ray list and arguments between them.
const PYRAMID_BINDING: u32 = 16;
const OUTPUT_BINDING: u32 = 17;
const RAY_LIST_BINDING: u32 = 18;
const RAY_ARGS_BINDING: u32 = 19;

// SPIR-V blobs for the RT kernels. Produced by [`compile_rt_shaders`];
// consumed by `RtReflectionsResources::new` at init and by
// `rebuild_rt_pipelines` during shader hot-reload.
pub(in crate::vulkan) struct RtShaders {
    pub classify: Vec<u8>,
    pub flat: Vec<u8>,
    // Textured trace SPIR-V; `None` when the bindless texture pool is absent
    // (the flat-tint trace is then the only one built).
    pub textured: Option<Vec<u8>>,
}

// Compile the classify and the flat trace, plus the textured trace when
// `pool_size > 0` (the bindless pool is live). dxc emits `SPV_KHR_ray_query`
// for the traversal, which the device already advertises wherever this pass is
// built.
pub(in crate::vulkan) fn compile_rt_shaders(
    hot_reload: bool,
    pool_size: usize,
) -> RenderResult<RtShaders> {
    use super::super::builtin_shaders;
    let classify = builtin_shaders::RT_REFLECTIONS_CLASSIFY.compile(hot_reload)?;
    let flat = builtin_shaders::RT_REFLECTIONS_TRACE.compile(hot_reload)?;
    let textured = if pool_size > 0 {
        Some(builtin_shaders::RT_REFLECTIONS_TRACE_TEXTURED.compile(hot_reload)?)
    } else {
        None
    };
    Ok(RtShaders {
        classify,
        flat,
        textured,
    })
}

// RT-reflection resources held by `VkContext` when `ray_traced_reflections` is
// on AND the GPU exposes the ray-query extensions AND the pass built; otherwise
// the context leaves this `None` and the graph falls back to `SsrResolve`. The
// pass outlives the scene acceleration structure, which comes and goes with the
// geometry it covers. All `vk::*` handles are owned here and freed on `destroy`.
pub(in crate::vulkan) struct RtReflectionsResources {
    // Resolved authored tunables; turned into a per-frame `RtParams` push.
    pub(in crate::vulkan) settings: RtReflectionSettings,

    // Reflection output: reflected radiance + composite weight, at the trace
    // resolution, which the reflection composite upsamples over the scene. Owns
    // its own slot because RT can be authored with the SSR resolve off.
    pub(in crate::vulkan) output: GpuImage,
    // The trace resolution: render resolution reduced by `settings.divisor`.
    extent: vk::Extent2D,
    // The pixels the classify owes a ray (one entry per output texel) and the
    // trace's indirect dispatch arguments.
    ray_list: PooledBuffer,
    ray_args: PooledBuffer,

    _set_layout: OwnedSetLayout,
    // Flat (material-tint) layout = [set 0, global set]; textured layout adds
    // the bindless pool. The textured layout/PSO are `Some` only when the
    // bindless pool is live (same gate as the bindless static pass).
    layout_flat: OwnedPipelineLayout,
    layout_textured: Option<OwnedPipelineLayout>,
    classify_pso: OwnedPipeline,
    flat_pso: OwnedPipeline,
    textured_pso: Option<OwnedPipeline>,

    // Per-frame `RtParams` UBO (144 B), host-mapped.
    params_buffers: Vec<PooledBuffer>,

    _descriptor_pool: OwnedDescriptorPool,
    // Per-frame resolve sets: scene = that frame's HDR resolve, plus the shared
    // gbuffer / roughness / verts / indices / output / ray list. The TLAS +
    // geometry table (bindings 1/2) and the pyramid (16) are re-pointed every
    // frame by `wire_dynamic` / `wire_pyramid`.
    resolve_sets: Vec<vk::DescriptorSet>,

    // Linear-clamp sampler the pass reads scene / G-buffer / roughness through,
    // written into every resolve set once at construction.
    _sampler: OwnedSampler,

    // A 1-element dummy storage buffer bound to the skinned-index SSBO (binding
    // 10) when the scene carries no skinned geometry (the accel data's skinned
    // index handle is then `vk::Buffer::null()`). Keeps the descriptor always
    // valid; the deformed-verts SSBO (binding 9) needs no dummy because the accel
    // data always holds a valid 1-element deformed buffer.
    dummy_ssbo: PooledBuffer,

    // Bindless texture-pool length, kept for the hot-reload recompile of the
    // textured variant.
    pool_size: usize,

    // What each frame's dynamic bindings (1/2/9/10) already point at, so a frame
    // whose acceleration structures did not move skips four descriptor writes,
    // one of them an acceleration-structure write.
    wired_accel: WireCache<RtAccelHandles>,
    // The pyramid view each frame's set binds.
    wired_pyramid: WireCache<vk::ImageView>,
}

// SAFETY: The params UBOs' mapped pointers are host-mapped, render-thread-only; the
// whole struct lives inside `VkContext`, which is already `unsafe impl Send`.
unsafe impl Send for RtReflectionsResources {}

// Set 0 of the RT kernels: the RtParams UBO (0), TLAS (1), geometry table (2),
// static verts and indices (3, 4), scene, G-buffer and roughness (5-7), the
// deformed skinned verts and skinned indices (9, 10), for skinned hits, the
// three images' samplers (11-13), and the classify's pyramid, the output and
// the ray list and arguments (16-19).
fn resolve_set_bindings() -> [Binding; 17] {
    use vk::DescriptorType as T;
    let cs = vk::ShaderStageFlags::COMPUTE;
    [
        (0, T::UNIFORM_BUFFER, cs),
        (1, T::ACCELERATION_STRUCTURE_KHR, cs),
        (2, T::STORAGE_BUFFER, cs),
        (3, T::STORAGE_BUFFER, cs),
        (4, T::STORAGE_BUFFER, cs),
        (5, T::SAMPLED_IMAGE, cs),
        (6, T::SAMPLED_IMAGE, cs),
        (7, T::SAMPLED_IMAGE, cs),
        (9, T::STORAGE_BUFFER, cs),
        (10, T::STORAGE_BUFFER, cs),
        (11, T::SAMPLER, cs),
        (12, T::SAMPLER, cs),
        (13, T::SAMPLER, cs),
        (PYRAMID_BINDING, T::SAMPLED_IMAGE, cs),
        (OUTPUT_BINDING, T::STORAGE_IMAGE, cs),
        (RAY_LIST_BINDING, T::STORAGE_BUFFER, cs),
        (RAY_ARGS_BINDING, T::STORAGE_BUFFER, cs),
    ]
}

// The classify and trace pipelines over their layouts.
struct RtPipelines {
    classify: OwnedPipeline,
    flat: OwnedPipeline,
    textured: Option<OwnedPipeline>,
}

fn create_rt_pipelines(
    device: &VkDevice,
    layout_flat: &OwnedPipelineLayout,
    layout_textured: Option<&OwnedPipelineLayout>,
    shaders: &RtShaders,
) -> RenderResult<RtPipelines> {
    Ok(RtPipelines {
        classify: compute_pipeline(
            device,
            layout_flat.handle(),
            &shaders.classify,
            "rt reflections classify",
        )?,
        flat: compute_pipeline(
            device,
            layout_flat.handle(),
            &shaders.flat,
            "rt reflections trace",
        )?,
        textured: match (layout_textured, &shaders.textured) {
            (Some(layout), Some(spv)) => Some(compute_pipeline(
                device,
                layout.handle(),
                spv,
                "rt reflections trace textured",
            )?),
            _ => None,
        },
    })
}

// Replacement RT pipelines built by the hot-reload pass.
pub(in crate::vulkan) struct RebuiltRtPipelines(RtPipelines);

// Rebuild the RT pipelines from disk-resident source against the existing
// layouts.
pub(in crate::vulkan) fn rebuild_rt_pipelines(
    device: &VkDevice,
    rt: &RtReflectionsResources,
    hot_reload: bool,
) -> RenderResult<RebuiltRtPipelines> {
    let shaders = compile_rt_shaders(hot_reload, rt.pool_size)?;
    create_rt_pipelines(
        device,
        &rt.layout_flat,
        rt.layout_textured.as_ref(),
        &shaders,
    )
    .map(RebuiltRtPipelines)
}

// Allocation context for building RT resources: the device to allocate on, the
// output-target extent, and the number of frames in flight (per-frame UBOs +
// descriptor sets). Everything `create_buffer` / `create_image` / `build_targets`
// need to size and place the pass's GPU memory.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct RtBuild<'a> {
    pub alloc: &'a DeviceAllocator,
    pub device: &'a VkDevice,
    pub width: u32,
    pub height: u32,
    pub frames: usize,
}

// The resolution-independent static resolve inputs the pass samples every frame:
// the scene vertex/index SSBOs and the per-frame HDR scene / G-buffer /
// roughness views. Wired by `wire_static` (at init and on resize) into every
// frame's set. The IBL prefilter cube is the global set's.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct RtStaticInputs<'a> {
    pub vertex_buffer: vk::Buffer,
    pub index_buffer: vk::Buffer,
    pub hdr_resolve_views: &'a [vk::ImageView],
    pub gbuffer_views: &'a [vk::ImageView],
    pub roughness_views: &'a [vk::ImageView],
}

// The live acceleration-structure handles the trace binds per frame: the TLAS,
// the geometry table (buffer + byte size), the deformed skinned vertex buffer,
// and the skinned index buffer. All re-pointed each frame by `wire_dynamic`
// because a dynamic rebuild fresh-allocates them. Compared frame to frame so a
// frame that rebuilt nothing rewrites nothing.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::vulkan) struct RtAccelHandles {
    pub tlas: vk::AccelerationStructureKHR,
    pub geom_buffer: vk::Buffer,
    pub geom_size: vk::DeviceSize,
    pub deformed_verts: vk::Buffer,
    pub skinned_indices: vk::Buffer,
}

// Pipeline-layout / bindless configuration for building the RT pipelines: the
// optional bindless texture-pool layout + its length (enable the textured
// variant), the forward global set's layout (bound as set 1 for the
// reflection-probe miss fallback), and whether this is a hot-reload recompile.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct RtLayoutConfig {
    pub bindless_set_layout: Option<vk::DescriptorSetLayout>,
    // The forward global set's layout, bound as set 1 so a missed ray can fall
    // back to the reflection-probe set (bindings 7, 8 and 17) or the sky
    // prefilter cube (5), read through the set's cube sampler (19).
    pub global_set_layout: vk::DescriptorSetLayout,
    pub pool_size: usize,
    pub hot_reload: bool,
}

impl RtReflectionsResources {
    // Build every RT-reflection resource. Returns `Err` when a kernel fails to
    // compile (the caller then falls back to SSR). The acceleration-structure
    // bindings are wired by `wire_dynamic` on each frame that has a BVH to
    // trace; `layout.bindless_set_layout` + `layout.pool_size` enable the
    // textured variant.
    pub(in crate::vulkan) fn new(
        build: RtBuild<'_>,
        settings: RtReflectionSettings,
        static_inputs: RtStaticInputs<'_>,
        layout: RtLayoutConfig,
    ) -> RenderResult<Self> {
        let RtBuild {
            alloc,
            device,
            width,
            height,
            frames,
        } = build;
        let RtLayoutConfig {
            bindless_set_layout,
            global_set_layout,
            pool_size,
            hot_reload,
        } = layout;

        let set_layout = create_descriptor_set_layout(device, &resolve_set_bindings())?;

        // set 0 = the RT set; set 1 = the global set (probe set/cubes). The
        // textured variant adds the bindless pool as set 2 (kept past the global set
        // so probe_common's set index stays a fixed 1 across both variants).
        let flat_layouts = [set_layout.handle(), global_set_layout];
        let layout_flat = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&flat_layouts),
            )
            .map_err(|e| crate::vulkan::error::map_vk_result(e, "rt flat pipeline layout"))?;
        let layout_textured = if let Some(bsl) = bindless_set_layout {
            let layouts = [set_layout.handle(), global_set_layout, bsl];
            Some(
                device
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                    )
                    .map_err(|e| {
                        crate::vulkan::error::map_vk_result(e, "rt textured pipeline layout")
                    })?,
            )
        } else {
            None
        };

        let shaders = compile_rt_shaders(hot_reload, pool_size)?;
        let RtPipelines {
            classify: classify_pso,
            flat: flat_pso,
            textured: textured_pso,
        } = create_rt_pipelines(device, &layout_flat, layout_textured.as_ref(), &shaders)?;

        // Per-frame RtParams UBO.
        let params_size = size_of::<RtParams>() as vk::DeviceSize;
        let mut params_buffers = Vec::with_capacity(frames);
        for _ in 0..frames {
            let buf = alloc.create_buffer(
                params_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            params_buffers.push(buf);
        }

        // Pool: one resolve set per frame.
        let f = u32::saturating_from(frames);
        let pool_sizes = PoolSizes::default()
            .sets(&resolve_set_bindings(), f)
            .build();
        let descriptor_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(f),
            )
            .map_err(|e| crate::vulkan::error::map_vk_result(e, "rt descriptor pool"))?;
        let layouts: Vec<_> = (0..frames).map(|_| set_layout.handle()).collect();
        let resolve_sets = alloc_descriptor_sets(device, descriptor_pool.handle(), &layouts)?;

        let sampler = create_sampler_linear_clamp(device)?;
        let screen = sampler.handle();
        for &set in &resolve_sets {
            SetWrites::new(set)
                .sampler(11, screen)
                .sampler(12, screen)
                .sampler(13, screen)
                .apply(device);
        }

        // 1-element dummy storage buffer for the skinned-index binding when there
        // is no skinned geometry.
        let dummy_ssbo = alloc.create_buffer(
            16,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let ray_args = alloc.create_buffer(
            size_of_val(&RAY_ARGS_RESET) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let mut me = Self {
            settings,
            output: GpuImage::null(),
            extent: vk::Extent2D::default(),
            ray_list: PooledBuffer::null(),
            ray_args,
            _set_layout: set_layout,
            layout_flat,
            layout_textured,
            classify_pso,
            flat_pso,
            textured_pso,
            params_buffers,
            _descriptor_pool: descriptor_pool,
            resolve_sets,
            _sampler: sampler,
            dummy_ssbo,
            pool_size,
            wired_accel: WireCache::new(frames),
            wired_pyramid: WireCache::new(frames),
        };
        me.build_targets(alloc, device, width, height)?;
        me.wire_static(device, static_inputs);
        Ok(me)
    }

    // Allocate / re-allocate the output target and the ray list at the trace
    // resolution for a `width` x `height` render, and point every frame's set
    // at them.
    fn build_targets(
        &mut self,
        alloc: &DeviceAllocator,
        device: &VkDevice,
        width: u32,
        height: u32,
    ) -> RenderResult<()> {
        let (w, h) = self.settings.trace_extent(width, height);
        self.extent = vk::Extent2D {
            width: w,
            height: h,
        };
        let pooled = create_image(
            alloc,
            &ImageSpec {
                width: w,
                height: h,
                format: HDR_FORMAT,
                tiling: vk::ImageTiling::OPTIMAL,
                // TRANSFER_SRC so the transparent (glass) pass can snapshot the
                // post-RT scene for its refraction tap, the same usage the SSR output
                // carries when SSR owns the scene image; TRANSFER_DST for the clear
                // while there is no BVH.
                usage: vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
                mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                samples: vk::SampleCountFlags::TYPE_1,
            },
        )?;
        let image = pooled.image();
        let view = create_image_view(device, image, HDR_FORMAT, vk::ImageAspectFlags::COLOR)?;
        self.output = GpuImage::from_pooled(pooled, view);
        self.ray_list = alloc.create_buffer(
            ray_list_len(w, h) * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        for &set in &self.resolve_sets {
            SetWrites::new(set)
                .storage_image(OUTPUT_BINDING, self.output.view)
                .storage_buffer(RAY_LIST_BINDING, self.ray_list.buffer(), vk::WHOLE_SIZE)
                .storage_buffer(RAY_ARGS_BINDING, self.ray_args.buffer(), vk::WHOLE_SIZE)
                .apply(device);
        }
        Ok(())
    }

    // Re-point every frame's shared static verts (3) + u32 indices (4) at the
    // given buffers. Called by `wire_static`, and again on its own when an asset
    // hot-reload replaces the shared geometry buffers under the pass (the sets
    // would otherwise keep descriptors on destroyed buffers).
    pub(in crate::vulkan) fn rewire_geometry(
        &self,
        device: &VkDevice,
        vertex_buffer: vk::Buffer,
        index_buffer: vk::Buffer,
    ) {
        for &set in &self.resolve_sets {
            SetWrites::new(set)
                .storage_buffer(3, vertex_buffer, vk::WHOLE_SIZE)
                .storage_buffer(4, index_buffer, vk::WHOLE_SIZE)
                .apply(device);
        }
    }

    // Wire the static per-frame bindings (UBO, verts, indices, scene, gbuffer,
    // roughness). The TLAS + geom table (bindings 1/2) are wired by
    // `wire_dynamic`. Called at init + on swapchain resize.
    //
    // `gbuffer_views` / `roughness_views` carry the unified G-buffer pre-pass's
    // per-frame normal+depth / roughness views; resolve set `i` binds slot `i`.
    pub(in crate::vulkan) fn wire_static(&self, device: &VkDevice, inputs: RtStaticInputs<'_>) {
        let RtStaticInputs {
            vertex_buffer,
            index_buffer,
            hdr_resolve_views,
            gbuffer_views,
            roughness_views,
        } = inputs;
        self.rewire_geometry(device, vertex_buffer, index_buffer);
        let frames = self.resolve_sets.len();
        debug_assert_eq!(hdr_resolve_views.len(), frames);
        debug_assert_eq!(gbuffer_views.len(), frames);
        debug_assert_eq!(roughness_views.len(), frames);
        for (i, &set) in self.resolve_sets.iter().enumerate() {
            SetWrites::new(set)
                .uniform_buffer(
                    0,
                    self.params_buffers[i].buffer(),
                    size_of::<RtParams>() as vk::DeviceSize,
                )
                .sampled_image(5, hdr_resolve_views[i])
                .sampled_image(6, gbuffer_views[i])
                .sampled_image(7, roughness_views[i])
                .apply(device);
        }
    }

    // Re-point one frame's TLAS (binding 1), geometry-table (binding 2), deformed
    // skinned verts (binding 9), and skinned indices (binding 10) descriptors at
    // the live handles. Called every frame because a dynamic rebuild
    // fresh-allocates the TLAS / geom table / deformed buffer; the current frame's
    // set is fence-gated (its previous submission completed at the top of
    // `draw_frame`), so the update is safe. A frame that rebuilt nothing hands
    // over the same five handles this slot already holds and writes nothing:
    // the comparison is over the handles themselves, so any that moved still
    // fires. `deformed` is always a valid handle
    // (the accel data holds a 1-element dummy when there is no skinned geometry);
    // `skinned_indices` is `vk::Buffer::null()` until the first skinned rebuild,
    // in which case the 1-element dummy SSBO is bound so the descriptor stays
    // valid.
    pub(in crate::vulkan) fn wire_dynamic(
        &mut self,
        device: &VkDevice,
        frame_idx: usize,
        accel: RtAccelHandles,
    ) {
        if !self.wired_accel.changed(frame_idx, accel) {
            return;
        }
        let RtAccelHandles {
            tlas,
            geom_buffer,
            geom_size,
            deformed_verts: deformed,
            skinned_indices,
        } = accel;
        let sidx_buffer = if skinned_indices == vk::Buffer::null() {
            self.dummy_ssbo.buffer()
        } else {
            skinned_indices
        };
        SetWrites::new(self.resolve_sets[frame_idx])
            .acceleration_structure(1, tlas)
            .storage_buffer(2, geom_buffer, geom_size)
            .storage_buffer(9, deformed, vk::WHOLE_SIZE)
            .storage_buffer(10, sidx_buffer, vk::WHOLE_SIZE)
            .apply(device);
    }

    // Point one frame's set at the pyramid its classify walks: SSGI's for the
    // frame, or the G-buffer standing in while there is none (the classify then
    // reads no pyramid). The current frame's set is fence-gated, so the update
    // is safe.
    pub(in crate::vulkan) fn wire_pyramid(
        &mut self,
        device: &VkDevice,
        frame_idx: usize,
        view: vk::ImageView,
    ) {
        if !self.wired_pyramid.changed(frame_idx, view) {
            return;
        }
        SetWrites::new(self.resolve_sets[frame_idx])
            .sampled_image(PYRAMID_BINDING, view)
            .apply(device);
    }

    fn destroy_targets(&mut self) {
        if self.output.image != vk::Image::null() {
            self.output = GpuImage::null();
        }
        self.ray_list = PooledBuffer::null();
    }

    // Rebuild the resolution-dependent output target at a new extent and re-wire
    // the static descriptors (the gbuffer / roughness / scene views all moved).
    // The TLAS + geom table are resolution-independent; the caller re-points them
    // per frame as usual.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        alloc: &DeviceAllocator,
        device: &VkDevice,
        width: u32,
        height: u32,
        inputs: RtStaticInputs<'_>,
    ) -> RenderResult<()> {
        self.destroy_targets();
        self.build_targets(alloc, device, width, height)?;
        self.wire_static(device, inputs);
        // `wire_static` rewrites the set; drop the dynamic memos so the next
        // frame re-points bindings 1/2/9/10 and 16 unconditionally.
        self.wired_accel.reset();
        self.wired_pyramid.reset();
        Ok(())
    }

    // Take `settings`, rebuilding the output at its trace resolution for a
    // `width` x `height` render when that moved. Nothing in the resolve sets
    // names the output, and the replaced image and framebuffer wait out the
    // frames in flight on their own.
    pub(in crate::vulkan) fn retrace(
        &mut self,
        alloc: &DeviceAllocator,
        device: &VkDevice,
        settings: RtReflectionSettings,
        (width, height): (u32, u32),
    ) -> RenderResult<()> {
        let (w, h) = settings.trace_extent(width, height);
        self.settings = settings;
        if (w, h) == (self.extent.width, self.extent.height) {
            return Ok(());
        }
        self.build_targets(alloc, device, width, height)
    }

    // Forget what each frame's acceleration-structure bindings point at, so the
    // next frame rewires them. A replaced BVH can reuse a destroyed one's handles.
    pub(in crate::vulkan) fn forget_accel(&mut self) {
        self.wired_accel.reset();
    }

    // Swap freshly-built pipelines into the live resources after a hot-reload.
    pub(in crate::vulkan) fn swap_pipelines(&mut self, rebuilt: RebuiltRtPipelines) {
        let RebuiltRtPipelines(RtPipelines {
            classify,
            flat,
            textured,
        }) = rebuilt;
        self.classify_pso = classify;
        self.flat_pso = flat;
        self.textured_pso = textured;
    }

    // Destroy every RT resource. The caller has already idled the device.
    pub(in crate::vulkan) fn destroy(&mut self, _device: &VkDevice) {
        self.destroy_targets();
        self.dummy_ssbo = PooledBuffer::null();
        self.ray_args = PooledBuffer::null();
        self.params_buffers.clear();
    }
}

impl VkContext {
    // True when hardware ray-traced reflections are live (the pass is built and
    // a BVH exists to trace). Gates the transparent pass's trace and the planar
    // mirrors; `ReflectionPath` settles the graph's resolve slot from the same
    // state. Mirrors `DxContext::rt_reflections_active`.
    pub(in crate::vulkan) fn rt_reflections_active(&self) -> bool {
        self.rt_reflections.is_some() && self.rt.accel.is_some()
    }

    // True when the transparent pass should trace per-pixel RT reflections this
    // frame: RT is live (the scene TLAS is built) AND every live producer's RT
    // pipelines compiled at init. Single-sources the transparent encoder's
    // RT-vs-base selection and the `graph_exec` planar skip, so the two always
    // agree -- gating the skip on `rt_reflections_active()` alone would drop the
    // planar re-render even when a producer's RT pipelines failed to build,
    // leaving its fallback sampling a stale resolve. Mirrors
    // `DxContext::rt_transparent_active`.
    pub(in crate::vulkan) fn rt_transparent_active(&self) -> bool {
        self.rt_reflections_active()
            && self
                .transparent
                .as_ref()
                .is_some_and(super::super::transparent::TransparentResources::rt_pipelines_ready)
    }

    // Run the per-frame dynamic acceleration-structure update on `cmd` (the
    // frame's "start" command buffer, submitted before every per-pass trace),
    // then re-point this frame's RT descriptor set at the live TLAS + geometry
    // table. A no-op when RT reflections are off. The descriptor rewrite happens
    // every frame (not only on a rebuild) so a frame that did not rebuild still
    // binds the current handles rather than a stale / retired one.
    //
    // Follows the shared BVH lifetime: a pass with no BVH seeds one once
    // participating geometry appears or skinned geometry is present
    // (`seed_wanted`), and this frame's update then runs over it. A BVH with
    // nothing left to trace and nothing that could rejoin it is dropped
    // (`is_spent`), held until the frames in flight have finished tracing it.
    // The trace skips while there is none.
    pub(in crate::vulkan) fn rt_dynamic_update(
        &mut self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
    ) {
        // Consumed by the accel's `dynamic_update`, which folds a runtime draw-set
        // change (cloned prop, streamed chunk added/removed) into the BLAS head,
        // or by the seed of a scene that had nothing to trace.
        let mut topology_dirty = std::mem::take(&mut self.state.gpu_dirty.rt_topology);
        let device = self.hw.device.clone();
        self.rt.collect_retired(self.frames_in_flight);
        if self.rt_reflections.is_none() {
            return;
        }
        if self.rt.accel.is_none() {
            // The seed is fence-waited internally (a rare stall).
            let skinned_present = self.rt_skinned_present();
            if !seed_wanted(self.rt.dynamic_mode, topology_dirty, skinned_present) {
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
            self.forget_wired_accel();
            // The seed already covers this frame's draw set.
            topology_dirty = false;
        }
        self.update_live_accel(cmd, frame_idx, topology_dirty);

        let Some(accel) = self.rt.accel.as_ref() else {
            return;
        };
        let (geom_buffer, geom_size) = accel.geom_table();
        let tlas = accel.tlas();
        let deformed = accel.deformed_verts();
        let skinned_indices = accel.skinned_indices();
        let pyramid = self
            .ssgi
            .as_ref()
            .and_then(|ssgi| ssgi.pyramid().ok())
            .map(|image| image.view)
            .or_else(|| {
                self.gbuffer_targets()
                    .and_then(|gb| gb.frame(frame_idx))
                    .map(|frame| frame.normal_depth)
            });
        let Some(rt) = self.rt_reflections.as_mut() else {
            return;
        };
        if let Some(view) = pyramid {
            rt.wire_pyramid(&device, frame_idx, view);
        }
        rt.wire_dynamic(
            &device,
            frame_idx,
            RtAccelHandles {
                tlas,
                geom_buffer,
                geom_size,
                deformed_verts: deformed,
                skinned_indices,
            },
        );
        // Re-point the transparent pass's RT descriptor ring at the same live
        // handles, so a trace this frame samples the current TLAS / geometry table.
        // A no-op when the world has no transparent content or the RT pipelines are
        // absent.
        if let Some(transparent) = self.transparent.as_mut() {
            transparent.wire_rt_dynamic(
                &device,
                frame_idx,
                super::super::transparent::TransparentRtDynamic {
                    tlas,
                    geom_buffer,
                    geom_size,
                    deformed,
                    skinned_indices,
                },
            );
        }
    }

    // Run the live BVH's dynamic update, then drop it when nothing is left to
    // trace and no skinned geometry can rejoin it.
    fn update_live_accel(
        &mut self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        topology_dirty: bool,
    ) {
        // Assemble this frame's skinned-geometry inputs while `self` is still
        // fully borrowable: the shared skinned VB/IB handles. `None` when skinned
        // geometry cannot join (the static path runs).
        let skinned_inputs = self.rt_skinned_buffers();

        // Read before `rt_accel` is taken: `seethrough_meshes_enabled` borrows
        // `self.transparent`.
        let exclude_seethrough = self.seethrough_meshes_enabled();
        let shared = super::super::raytrace::SharedGeometry::of(&self.geometry);

        // Take `rt_accel` out so its `&mut` borrow does not overlap the shared
        // `&self` reads (`skinned_draw_objects` / `draw_objects`) the inputs need;
        // put it back immediately after.
        let Some(mut accel) = self.rt.accel.take() else {
            return;
        };
        let joint_buffers = self
            .skinned
            .joint_buffers
            .get(frame_idx)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let skinned = skinned_inputs
            .zip(self.rt.skin.as_mut())
            .map(|((vb, ib), skin)| super::super::raytrace::SkinnedRtInputs {
                objects: &self.state.skinned.draw_objects,
                vertex_buffer: vb,
                index_buffer: ib,
                joint_buffers,
                skin,
            });
        let updated = accel.dynamic_update(
            super::super::raytrace::RtDeviceCtx {
                alloc: &self.hw.alloc,
                instance: &self.hw.instance,
                device: &self.hw.device,
                pd: self.hw.physical_device,
            },
            cmd,
            &self.state.draw.objects,
            super::super::raytrace::RtDynamicInputs {
                policy: super::super::raytrace::RtRebuildPolicy {
                    mode: self.rt.dynamic_mode,
                    topology_dirty,
                    exclude_seethrough,
                },
                frame_idx,
                shared,
                skinned,
                spans: &self.state.spans,
            },
        );
        crate::rt_report::report_rt_update(&mut self.rt.update_streak, updated);
        let spent = accel.is_spent(skinned_inputs.is_some());
        self.rt.accel = Some(accel);
        if spent {
            self.rt.retire_accel();
        }
    }

    // Encode the RT-reflection resolve: reset the ray arguments, classify every
    // output texel, trace the listed ones, then blend the output over the scene
    // through the reflection composite. Without a BVH the output is cleared to
    // zero weight instead, which the composite passes the scene through
    // unchanged for. No-op when the pass is absent (the graph only schedules it
    // when the pass exists, so the guard is defensive).
    pub(in crate::vulkan) fn encode_rt_reflections(
        &self,
        rec: &Recorder<'_>,
        frame_idx: usize,
        params: Option<&RtParams>,
    ) {
        let Some(rt) = &self.rt_reflections else {
            return;
        };
        let to_general = output_barrier(
            rt.output.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );
        let to_sampled = output_barrier(
            rt.output.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        let compute = vk::PipelineStageFlags::COMPUTE_SHADER;
        let fragment = vk::PipelineStageFlags::FRAGMENT_SHADER;
        let transfer = vk::PipelineStageFlags::TRANSFER;
        let (Some(params), Some(_)) = (params, self.rt.accel.as_ref()) else {
            // The composite's earlier sampling is ordered before the clear, and
            // the clear before this frame's composite.
            rec.pipeline_barrier(
                fragment,
                transfer,
                &[],
                &[],
                &[to_general
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
            );
            rec.clear_color_image(
                rt.output.image,
                vk::ImageLayout::GENERAL,
                &vk::ClearColorValue::default(),
                color_range(),
            );
            rec.pipeline_barrier(
                transfer,
                fragment,
                &[],
                &[],
                &[to_sampled.src_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
            );
            self.encode_reflection_composite(rec.raw(), rt.output.view, frame_idx);
            return;
        };
        rt.params_buffers[frame_idx].write_val(0, params);

        // Textured hit shading needs the bindless albedo/normal pool, which only
        // the bindless static path populates; otherwise fall back to the
        // flat-tint variant. Mirrors DirectX's bindless gate.
        let textured = self.cull.bindless_pipeline.is_some() && rt.textured_pso.is_some();
        let (trace_pso, trace_layout) = match (
            textured,
            rt.textured_pso.as_ref(),
            rt.layout_textured.as_ref(),
        ) {
            (true, Some(pso), Some(layout)) => (pso, layout),
            _ => (&rt.flat_pso, &rt.layout_flat),
        };
        let screen_sets = [
            rt.resolve_sets[frame_idx],
            self.descriptors.global_sets[frame_idx],
        ];

        // Last frame's trace read the arguments as indirect parameters and
        // through its shader; the reset waits for both.
        rec.pipeline_barrier(
            compute | vk::PipelineStageFlags::DRAW_INDIRECT,
            transfer,
            &[],
            &[],
            &[],
        );
        rec.update_buffer(rt.ray_args.buffer(), 0, &RAY_ARGS_RESET);
        // The reset and SSGI's pyramid, drawn earlier in the frame, become
        // visible to the classify; the output's old contents are discarded.
        let ready = vk::MemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        rec.pipeline_barrier(
            transfer | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | fragment | compute,
            compute,
            &[ready],
            &[],
            &[to_general],
        );
        rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &rt.classify_pso);
        rec.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            &rt.layout_flat,
            0,
            &screen_sets,
            &[],
        );
        let (groups_x, groups_y) = classify_groups(rt.extent.width, rt.extent.height);
        rec.dispatch(groups_x, groups_y, 1);

        // The list and the arguments the classify wrote, for the indirect
        // dispatch and the trace that reads them, and its output texels.
        let listed = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::INDIRECT_COMMAND_READ,
            );
        rec.pipeline_barrier(
            compute,
            compute | vk::PipelineStageFlags::DRAW_INDIRECT,
            &[listed],
            &[],
            &[],
        );
        rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, trace_pso);
        rec.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            trace_layout,
            0,
            &screen_sets,
            &[],
        );
        if textured {
            rec.bind_descriptor_sets(
                vk::PipelineBindPoint::COMPUTE,
                trace_layout,
                2,
                &[self.cull.bindless_sets[frame_idx]],
                &[],
            );
        }
        rec.dispatch_indirect(rt.ray_args.buffer(), 0);
        rec.pipeline_barrier(compute, fragment, &[], &[], &[to_sampled]);

        // Blur the output's radiance+weight by roughness and composite it over
        // the scene into the reflection composite's output (the scene image the
        // post stack consumes). No-op when the composite is absent.
        self.encode_reflection_composite(rec.raw(), rt.output.view, frame_idx);
    }
}

// The output's single color level.
fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

// A layout transition of the output between the kernels' storage writes and
// the composite's sampling.
fn output_barrier(
    image: vk::Image,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) -> vk::ImageMemoryBarrier<'static> {
    let writes = vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE;
    vk::ImageMemoryBarrier::default()
        .old_layout(from)
        .new_layout(to)
        .src_access_mask(if from == vk::ImageLayout::GENERAL {
            vk::AccessFlags::SHADER_WRITE
        } else {
            vk::AccessFlags::empty()
        })
        .dst_access_mask(if to == vk::ImageLayout::GENERAL {
            writes
        } else {
            vk::AccessFlags::SHADER_READ
        })
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(color_range())
}

#[cfg(test)]
mod tests {
    // The RT classify and both trace variants compile to SPIR-V (ray query
    // target). Guards the `RT_TEXTURED` split. The CPU<->GPU `RtParams` /
    // `RtGeomEntry` layouts are guarded by the shader layout checks.
    #[test]
    fn rt_reflections_shaders_compile() {
        concinnity_shader::require_dxc!();
        let shaders = super::compile_rt_shaders(false, 4).expect("rt shaders compile");
        assert!(crate::vulkan::pipeline::is_spirv(&shaders.classify));
        // No kernel may need compute derivatives, which the device does not
        // enable.
        for spv in [&shaders.classify, &shaders.flat] {
            assert!(!declares_compute_derivatives(spv));
        }
        assert!(!declares_compute_derivatives(
            shaders.textured.as_ref().expect("textured")
        ));
        assert!(crate::vulkan::pipeline::is_spirv(&shaders.flat));
        assert!(shaders.textured.is_some(), "pool_size>0 builds textured");
        // pool_size 0 builds only the flat variant.
        let flat_only = super::compile_rt_shaders(false, 0).expect("rt flat compiles");
        assert!(flat_only.textured.is_none());
    }

    // Whether `spv` declares either compute-derivative-group capability
    // (`OpCapability ComputeDerivativeGroupQuadsKHR` / `...LinearKHR`).
    fn declares_compute_derivatives(spv: &[u8]) -> bool {
        const OP_CAPABILITY: u32 = 0x0002_0011;
        const QUADS: u32 = 5288;
        const LINEAR: u32 = 5350;
        let words: Vec<u32> = spv
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        words
            .windows(2)
            .any(|w| w[0] == OP_CAPABILITY && (w[1] == QUADS || w[1] == LINEAR))
    }
}
