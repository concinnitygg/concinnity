//! Hardware ray-traced reflection pass for the Vulkan backend. A fullscreen
//! fragment pass that, per glossy pixel, rebuilds a world-space surface point +
//! normal from the SSR pre-pass G-buffer, traces a reflection ray against the
//! scene's top-level acceleration structure ([`crate::vulkan::raytrace`]) with
//! inline `rayQueryEXT`, shades the hit (sun + IBL split-sum, optionally textured)
//! or the IBL prefilter cube on a miss, and writes the reflected radiance with the
//! same Fresnel/gloss weight SSR uses for the reflection composite to blend.
//!
//! It occupies the `SsrResolve` slot in the frame graph (reads the HDR scene,
//! writes its own `output` target) and is mutually exclusive with the SSR
//! resolve. Like SSGI it reuses the SSR depth + normal + roughness pre-pass
//! G-buffer, so that pre-pass is forced on whenever RT reflections are enabled.
//! While there is no BVH to trace, an authored SSR resolve takes the slot, or
//! the pass clears its output so the reflection composite leaves the scene as
//! it was.
//! Mirrors src/directx/post/rt_reflections.rs (DXR inline `RayQuery`); the GLSL
//! is compiled with the Vulkan-1.2 / SPIR-V-1.4 target ray query needs.
//!
//! Unlike DirectX (which binds the TLAS + geometry table as root SRVs by GPU
//! virtual address each frame), Vulkan binds them through a descriptor set, so
//! `VkContext::rt_dynamic_update` re-points the current frame's set at the
//! live TLAS + geometry-table handles every frame (they change on a dynamic
//! rebuild; see `crate::vulkan::raytrace`).

use ash::vk;
use concinnity_core::gfx::render_types::RtParams;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::planar_reflection;
use concinnity_core::render::post::rt_reflections::{RtParamsInputs, RtReflectionSettings};
use concinnity_core::render::rt_accel::seed_wanted;

use super::super::allocator::{DeviceAllocator, PooledBuffer};
use super::super::context::{HDR_FORMAT, VkContext};
use super::super::descriptor_layout::{Binding, PoolSizes};
use super::super::pipeline_desc::{Blend, GraphicsPipelineDesc};
use super::super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::super::set_writes::SetWrites;
use super::super::texture::*;
use crate::vulkan::builtin_shaders::CompileProgram;
use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedFramebuffer, OwnedPipeline, OwnedPipelineLayout, OwnedRenderPass,
    OwnedSampler, OwnedSetLayout, VkDevice,
};
use crate::vulkan::wire_cache::WireCache;

// SPIR-V blobs for the RT pipelines. Produced by [`compile_rt_shaders`];
// consumed by `RtReflectionsResources::new` at init and by
// `rebuild_rt_pipelines` during shader hot-reload.
pub(in crate::vulkan) struct RtShaders {
    pub vs: Vec<u8>,
    pub flat_fs: Vec<u8>,
    // Textured fragment SPIR-V; `None` when the bindless texture pool is absent
    // (the flat-tint variant is then the only one built).
    pub textured_fs: Option<Vec<u8>>,
}

// Compile the shared fullscreen vertex stage + the flat fragment, plus the
// textured fragment when `pool_size > 0` (the bindless pool is live). dxc
// emits `SPV_KHR_ray_query` for the traversal, which the device already
// advertises wherever this pass is built.
pub(in crate::vulkan) fn compile_rt_shaders(
    hot_reload: bool,
    pool_size: usize,
) -> RenderResult<RtShaders> {
    use super::super::builtin_shaders;
    let vs = builtin_shaders::FULLSCREEN_VERT.compile(hot_reload)?;
    let flat_fs = builtin_shaders::RT_REFLECTIONS_FRAG.compile(hot_reload)?;
    let textured_fs = if pool_size > 0 {
        Some(builtin_shaders::RT_REFLECTIONS_FRAG_TEXTURED.compile(hot_reload)?)
    } else {
        None
    };
    Ok(RtShaders {
        vs,
        flat_fs,
        textured_fs,
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
    render_pass: OwnedRenderPass,
    framebuffer: OwnedFramebuffer,

    _set_layout: OwnedSetLayout,
    // Flat (material-tint) layout = [set 0]; textured layout = [set 0, bindless
    // pool]. The textured layout/PSO are `Some` only when the bindless pool is
    // live (same gate as the bindless static pass).
    layout_flat: OwnedPipelineLayout,
    layout_textured: Option<OwnedPipelineLayout>,
    flat_pso: OwnedPipeline,
    textured_pso: Option<OwnedPipeline>,

    // Per-frame `RtParams` UBO (144 B), host-mapped.
    params_buffers: Vec<PooledBuffer>,

    _descriptor_pool: OwnedDescriptorPool,
    // Per-frame resolve sets: scene = that frame's HDR resolve, plus the shared
    // gbuffer / roughness / verts / indices. The TLAS + geometry
    // table (bindings 1/2) are re-pointed every frame by `wire_dynamic`.
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
}

// SAFETY: The params UBOs' mapped pointers are host-mapped, render-thread-only; the
// whole struct lives inside `VkContext`, which is already `unsafe impl Send`.
unsafe impl Send for RtReflectionsResources {}

// Set 0 of the RT resolve: the RtParams UBO (0), TLAS (1), geometry table (2),
// static verts and indices (3, 4), scene, G-buffer and roughness (5-7), the
// deformed skinned verts and skinned indices (9, 10), for skinned hits, and
// the three images' samplers (11-13).
fn resolve_set_bindings() -> [Binding; 13] {
    use vk::DescriptorType as T;
    let frag = vk::ShaderStageFlags::FRAGMENT;
    [
        (0, T::UNIFORM_BUFFER, frag),
        (1, T::ACCELERATION_STRUCTURE_KHR, frag),
        (2, T::STORAGE_BUFFER, frag),
        (3, T::STORAGE_BUFFER, frag),
        (4, T::STORAGE_BUFFER, frag),
        (5, T::SAMPLED_IMAGE, frag),
        (6, T::SAMPLED_IMAGE, frag),
        (7, T::SAMPLED_IMAGE, frag),
        (9, T::STORAGE_BUFFER, frag),
        (10, T::STORAGE_BUFFER, frag),
        (11, T::SAMPLER, frag),
        (12, T::SAMPLER, frag),
        (13, T::SAMPLER, frag),
    ]
}

// RT render pass: one HDR-format color attachment (`output`), no depth. The
// fullscreen triangle (or the clear while there is no BVH) overwrites every
// pixel, so `DONT_CARE` is safe on load.
// Ends shader-readable for the bloom + composite passes. Mirrors the SSR resolve
// render pass.
fn create_rt_render_pass(device: &VkDevice) -> RenderResult<OwnedRenderPass> {
    let attachment = vk::AttachmentDescription::default()
        .format(HDR_FORMAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref));
    // Broad SUBPASS_EXTERNAL dep: synchronize every prior color write + shader
    // read (the main pass's hdr_resolve, the SSR pre-pass G-buffer / roughness,
    // and the start-buffer's acceleration-structure build) against this pass's
    // reads + write. Same shape as the SSR resolve dep.
    let dep = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_READ)
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_READ);
    let info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&dep));
    device
        .create_render_pass(&info)
        .map_err(|e| crate::vulkan::error::map_vk_result(e, "RT reflections render pass"))
}

// Build one fullscreen RT pipeline. No vertex input (procedural fullscreen
// triangle); no depth; no blend; writes the HDR output. Mirrors the SSR resolve
// pipeline.
fn create_rt_pipeline(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> RenderResult<OwnedPipeline> {
    GraphicsPipelineDesc::fullscreen(vert_spv, frag_spv, layout, render_pass, &[Blend::Opaque])
        .build(device, "rt reflections")
}

// Replacement RT pipelines built by the hot-reload pass.
pub(in crate::vulkan) struct RebuiltRtPipelines {
    flat: OwnedPipeline,
    textured: Option<OwnedPipeline>,
}

// Rebuild the RT pipelines from disk-resident GLSL against the existing layouts +
// render pass.
pub(in crate::vulkan) fn rebuild_rt_pipelines(
    device: &VkDevice,
    rt: &RtReflectionsResources,
    hot_reload: bool,
) -> RenderResult<RebuiltRtPipelines> {
    let shaders = compile_rt_shaders(hot_reload, rt.pool_size)?;
    let flat = create_rt_pipeline(
        device,
        rt.render_pass.handle(),
        rt.layout_flat.handle(),
        &shaders.vs,
        &shaders.flat_fs,
    )?;
    let textured = match (rt.layout_textured.as_ref(), &shaders.textured_fs) {
        (Some(layout), Some(fs)) => Some(create_rt_pipeline(
            device,
            rt.render_pass.handle(),
            layout.handle(),
            &shaders.vs,
            fs,
        )?),
        _ => None,
    };
    Ok(RebuiltRtPipelines { flat, textured })
}

// Allocation context for building RT resources: the device to allocate on, the
// output-target extent, and the number of frames in flight (per-frame UBOs +
// descriptor sets). Everything `create_buffer` / `create_image` / `build_targets`
// need to size and place the pass's GPU memory.
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
    // Build every RT-reflection resource. Returns `Err` when the GLSL fails to
    // compile (the caller then falls back to SSR). The acceleration-structure
    // bindings are wired by `wire_dynamic` on each frame that has a BVH to
    // trace; `layout.bindless_set_layout` + `layout.pool_size` enable the
    // textured variant.
    pub(in crate::vulkan) fn new(
        build: RtBuild,
        settings: RtReflectionSettings,
        static_inputs: RtStaticInputs,
        layout: RtLayoutConfig,
    ) -> RenderResult<Self> {
        let RtBuild {
            alloc,
            device,
            width,
            height,
            frames,
        } = build;
        let RtStaticInputs {
            vertex_buffer,
            index_buffer,
            hdr_resolve_views,
            gbuffer_views,
            roughness_views,
        } = static_inputs;
        let RtLayoutConfig {
            bindless_set_layout,
            global_set_layout,
            pool_size,
            hot_reload,
        } = layout;
        let render_pass = create_rt_render_pass(device)?;

        let set_layout = create_descriptor_set_layout(device, &resolve_set_bindings())?;

        // set 0 = the RT resolve set; set 1 = the global set (probe set/cubes). The
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
        let flat_pso = create_rt_pipeline(
            device,
            render_pass.handle(),
            layout_flat.handle(),
            &shaders.vs,
            &shaders.flat_fs,
        )?;
        let textured_pso = match (layout_textured.as_ref(), &shaders.textured_fs) {
            (Some(layout), Some(fs)) => Some(create_rt_pipeline(
                device,
                render_pass.handle(),
                layout.handle(),
                &shaders.vs,
                fs,
            )?),
            _ => None,
        };

        // Per-frame RtParams UBO.
        let params_size = std::mem::size_of::<RtParams>() as vk::DeviceSize;
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
        let f = frames as u32;
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

        let mut me = Self {
            settings,
            output: GpuImage::null(),
            extent: vk::Extent2D::default(),
            render_pass,
            framebuffer: OwnedFramebuffer::null(),
            _set_layout: set_layout,
            layout_flat,
            layout_textured,
            flat_pso,
            textured_pso,
            params_buffers,
            _descriptor_pool: descriptor_pool,
            resolve_sets,
            _sampler: sampler,
            dummy_ssbo,
            pool_size,
            wired_accel: WireCache::new(frames),
        };
        me.build_targets(alloc, device, width, height)?;
        me.wire_static(
            device,
            RtStaticInputs {
                vertex_buffer,
                index_buffer,
                hdr_resolve_views,
                gbuffer_views,
                roughness_views,
            },
        );
        Ok(me)
    }

    // Allocate / re-allocate the output target + framebuffer at the trace
    // resolution for a `width` x `height` render.
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
                // carries when SSR owns the scene image.
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC,
                mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                samples: vk::SampleCountFlags::TYPE_1,
            },
        )?;
        let image = pooled.image();
        let view = create_image_view(device, image, HDR_FORMAT, vk::ImageAspectFlags::COLOR)?;
        self.output = GpuImage::from_pooled(pooled, view);
        self.framebuffer = device
            .create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(self.render_pass.handle())
                    .attachments(std::slice::from_ref(&self.output.view))
                    .width(w)
                    .height(h)
                    .layers(1),
            )
            .map_err(|e| crate::vulkan::error::map_vk_result(e, "rt framebuffer"))?;
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
    pub(in crate::vulkan) fn wire_static(&self, device: &VkDevice, inputs: RtStaticInputs) {
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
        let sidx_buffer = if skinned_indices != vk::Buffer::null() {
            skinned_indices
        } else {
            self.dummy_ssbo.buffer()
        };
        SetWrites::new(self.resolve_sets[frame_idx])
            .acceleration_structure(1, tlas)
            .storage_buffer(2, geom_buffer, geom_size)
            .storage_buffer(9, deformed, vk::WHOLE_SIZE)
            .storage_buffer(10, sidx_buffer, vk::WHOLE_SIZE)
            .apply(device);
    }

    fn destroy_targets(&mut self, _device: &VkDevice) {
        if !self.framebuffer.is_null() {
            // SAFETY: the handle was created from this device and is destroyed exactly once; the
            // caller has already waited for the device to go idle, so no submission still
            // references it.
            self.framebuffer = OwnedFramebuffer::null();
        }
        if self.output.image != vk::Image::null() {
            self.output = GpuImage::null();
        }
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
        inputs: RtStaticInputs,
    ) -> RenderResult<()> {
        self.destroy_targets(device);
        self.build_targets(alloc, device, width, height)?;
        self.wire_static(device, inputs);
        // `wire_static` rewrites the set; drop the dynamic memo so the next frame
        // re-points bindings 1/2/9/10 unconditionally.
        self.wired_accel.reset();
        Ok(())
    }

    // Forget what each frame's acceleration-structure bindings point at, so the
    // next frame rewires them. A replaced BVH can reuse a destroyed one's handles.
    pub(in crate::vulkan) fn forget_accel(&mut self) {
        self.wired_accel.reset();
    }

    // Swap freshly-built pipelines into the live resources after a hot-reload.
    pub(in crate::vulkan) fn swap_pipelines(&mut self, rebuilt: RebuiltRtPipelines) {
        self.flat_pso = rebuilt.flat;
        self.textured_pso = rebuilt.textured;
    }

    // Destroy every RT resource. The caller has already idled the device.
    pub(in crate::vulkan) fn destroy(&mut self, device: &VkDevice) {
        self.destroy_targets(device);
        self.dummy_ssbo = PooledBuffer::null();
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
    // True when the transparent pass samples planar mirrors, which puts the
    // `PlanarReflection` node in the graph. Water takes the mirror over its own
    // trace wherever it holds a slot (see `water.hlsl`), so a visible water
    // surface keeps the mirrors alive even while the trace is live; a glass-only
    // world under a live trace skips them. Shared with the other backends through
    // `planar_reflection::planar_pass_needed`.
    pub(in crate::vulkan) fn planar_pass_needed(&self) -> bool {
        planar_reflection::planar_pass_needed(
            self.planar_reflection.is_some(),
            self.transparent
                .as_ref()
                .is_some_and(|t| t.water_planar_slot_live()),
            self.rt_transparent_active(),
        )
    }

    pub(in crate::vulkan) fn rt_transparent_active(&self) -> bool {
        self.rt_reflections_active()
            && self
                .transparent
                .as_ref()
                .is_some_and(|t| t.rt_pipelines_ready())
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
        let Some(rt) = self.rt_reflections.as_mut() else {
            return;
        };
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
            .map(|b| b.as_slice())
            .unwrap_or(&[]);
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
            },
        );
        crate::rt_report::report_rt_update(&mut self.rt.update_streak, updated);
        let spent = accel.is_spent(skinned_inputs.is_some());
        self.rt.accel = Some(accel);
        if spent {
            self.rt.retire_accel();
        }
    }

    // Encode the RT-reflection resolve: a fullscreen triangle that traces each
    // glossy pixel's reflection ray against the scene TLAS and writes radiance +
    // weight into `rt_reflections.output`, which the reflection composite then
    // blends over the scene. Without a BVH the output is cleared to zero weight
    // instead, which the composite passes the scene through unchanged for. No-op
    // when the pass is absent (the graph only schedules it when the pass exists,
    // so the guard is defensive).
    pub(in crate::vulkan) fn encode_rt_reflections(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        fov_y_radians: f32,
        aspect: f32,
        cam_pos: [f32; 3],
    ) {
        let rt = match &self.rt_reflections {
            Some(r) => r,
            None => return,
        };
        let device = &self.hw.device;
        let extent = rt.extent;
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(rt.render_pass.handle())
            .framebuffer(rt.framebuffer.handle())
            .render_area(vk::Rect2D::default().extent(extent));
        if self.rt.accel.is_none() {
            let clear = vk::ClearAttachment {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                color_attachment: 0,
                clear_value: vk::ClearValue::default(),
            };
            let rect = vk::ClearRect {
                rect: vk::Rect2D::default().extent(extent),
                base_array_layer: 0,
                layer_count: 1,
            };
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and
            // slice these commands name is live for the call.
            unsafe {
                device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
                device.cmd_clear_attachments(cmd, &[clear], &[rect]);
                device.cmd_end_render_pass(cmd);
            }
            self.encode_reflection_composite(cmd, rt.output.view, frame_idx);
            return;
        }

        // The view->world rotation is the transpose of the view matrix's
        // orthonormal 3x3; `params` fills in the camera-position translation
        // column to complete the camera-to-world transform.
        let v = self.state.view.matrix;
        let inv_view_rot = [
            [v[0][0], v[1][0], v[2][0], 0.0],
            [v[0][1], v[1][1], v[2][1], 0.0],
            [v[0][2], v[1][2], v[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let params = rt.settings.params(RtParamsInputs {
            fov_y_radians,
            aspect,
            inv_view_rot,
            cam_pos,
            sun_dir: self.fog.sun_dir,
            sun_color: self.fog.sun_color,
            prefilter_mip_count: self.scene.prefilter_mip_count as f32,
            sky_rot: self.state.view.sky_rot,
        });
        rt.params_buffers[frame_idx].write_val(0, &params);

        // Textured hit shading needs the bindless albedo/normal pool, which only
        // the bindless static path populates; otherwise fall back to the
        // flat-tint variant. Mirrors DirectX's bindless gate.
        let textured = self.cull.bindless_pipeline.is_some() && rt.textured_pso.is_some();
        let (pso, layout) = match (
            textured,
            rt.textured_pso.as_ref(),
            rt.layout_textured.as_ref(),
        ) {
            (true, Some(pso), Some(layout)) => (pso, layout),
            _ => (&rt.flat_pso, &rt.layout_flat),
        };

        let vp = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(extent);
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pso.handle());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout.handle(),
                0,
                std::slice::from_ref(&rt.resolve_sets[frame_idx]),
                &[],
            );
            // set 1: the global set, for the reflection-probe miss fallback.
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout.handle(),
                1,
                std::slice::from_ref(&self.descriptors.global_sets[frame_idx]),
                &[],
            );
            if textured {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout.handle(),
                    2,
                    std::slice::from_ref(&self.cull.bindless_sets[frame_idx]),
                    &[],
                );
            }
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_render_pass(cmd);
        }
        // Blur the trace's radiance+weight by roughness and composite it over the
        // scene into the reflection composite's output (the scene image the post
        // stack consumes). No-op when the composite is absent.
        self.encode_reflection_composite(cmd, rt.output.view, frame_idx);
    }
}

#[cfg(test)]
mod tests {
    // The RT fullscreen vert + both fragment variants compile to SPIR-V (ray
    // query target). Guards the `GL_EXT_ray_query` GLSL + the `RT_TEXTURED`
    // split. The CPU<->GPU `RtParams` / `RtGeomEntry` layouts are guarded by the
    // `rt_params_layout_*` / `rt_geom_entry_*` tests in gfx::render_types.
    #[test]
    fn rt_reflections_shaders_compile() {
        concinnity_shader::require_dxc!();
        let shaders = super::compile_rt_shaders(false, 4).expect("rt shaders compile");
        assert!(crate::vulkan::pipeline::is_spirv(&shaders.vs));
        assert!(crate::vulkan::pipeline::is_spirv(&shaders.flat_fs));
        assert!(shaders.textured_fs.is_some(), "pool_size>0 builds textured");
        // pool_size 0 builds only the flat variant.
        let flat_only = super::compile_rt_shaders(false, 0).expect("rt flat compiles");
        assert!(flat_only.textured_fs.is_none());
    }
}
