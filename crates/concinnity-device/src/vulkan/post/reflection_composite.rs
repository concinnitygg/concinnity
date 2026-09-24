//! Roughness-aware reflection composite for the Vulkan backend. The SSR and RT
//! resolves now write reflected radiance (.rgb) + a Fresnel/gloss weight (.a) into
//! their output target instead of compositing inline; this two-pass effect blurs
//! that reflection by surface roughness and composites it over the base HDR scene
//! into `output`, the scene-with-reflections the TAA / bloom / composite / glass
//! passes consume. Mirrors src/metal/post/ssr.rs (the composite half) +
//! src/directx/post/reflection_composite.rs.
//!
//!   pass 1 (blur, reduced resolution): weight-averages the reflection over a
//!       roughness-scaled cone into `blur`. The expensive multi-tap part, run at a
//!       fraction of the pixels.
//!   pass 2 (composite, full resolution): lerps the sharp full-res reflection
//!       against the upsampled half-res blur by roughness, then composites over the
//!       scene into `output`.
//!
//! Both reflection paths feed one composite: `encode_ssr_resolve` /
//! `encode_rt_reflections` each render their resolve target, then call
//! `encode_reflection_composite` with that target's view; the composite's reflection
//! binding is re-pointed to it per encode (the two paths are mutually exclusive).

use ash::vk;
use concinnity_core::render::error::RenderResult;

use super::super::context::{HDR_FORMAT, VkContext};
use super::super::descriptor_layout::PoolSizes;
use super::super::pipeline::*;
use super::super::resources::{
    alloc_descriptor_sets, create_descriptor_set_layout, source_set_bindings, write_source_set,
};
use super::super::texture::*;
use super::gbuffer::GbufferResources;
use crate::vulkan::builtin_shaders::CompileProgram;
use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedFramebuffer, OwnedPipeline, OwnedPipelineLayout, OwnedRenderPass,
    OwnedSampler, OwnedSetLayout, VkDevice,
};
use crate::vulkan::wire_cache::WireCache;

// Reflection-composite resources, held by `VkContext` when the SSR resolve or RT
// reflections are active (both feed this composite). All `vk::*` handles are owned
// here and freed on `destroy`.
pub(in crate::vulkan) struct ReflectionCompositeResources {
    // Composited scene (full render resolution): the scene image the post stack
    // consumes in place of the raw SSR / RT resolve output.
    pub(in crate::vulkan) output: GpuImage,
    output_framebuffer: OwnedFramebuffer,

    // Reduced-resolution roughness blur of the reflection target (pass 1 writes it,
    // the composite upsamples it). Sized at render / `blur_scale`.
    blur: GpuImage,
    blur_framebuffer: OwnedFramebuffer,
    blur_extent: vk::Extent2D,

    // One render pass (RGBA16F color, DONT_CARE load, ends shader-readable) shared
    // by both passes; the framebuffer selects the target.
    render_pass: OwnedRenderPass,

    // Pass 1 (blur) reads reflection + roughness; pass 2 (composite) reads those
    // plus scene + G-buffer normal+depth + blur.
    _blur_set_layout: OwnedSetLayout,
    _composite_set_layout: OwnedSetLayout,
    blur_pipeline_layout: OwnedPipelineLayout,
    composite_pipeline_layout: OwnedPipelineLayout,
    blur_pso: OwnedPipeline,
    composite_pso: OwnedPipeline,

    _descriptor_pool: OwnedDescriptorPool,
    // Per-frame sets. Binding 0 (the reflection target) is re-pointed at the top
    // of a frame whose resolve target moved; the rest are wired at init / resize.
    blur_sets: Vec<vk::DescriptorSet>,
    composite_sets: Vec<vk::DescriptorSet>,

    // Which reflection view each frame's binding 0 already names. The view only
    // moves on a resize / quality rebuild, so the steady state writes nothing.
    wired_reflection: WireCache<vk::ImageView>,

    sampler: OwnedSampler,

    // Per-axis divisor the blur target is sized by (from the world's
    // `reflection_blur_resolution`). Held so `rebuild` reuses the same scale.
    blur_scale: u32,
}

// SAFETY: Raw per-frame sets are render-thread-only; the struct lives inside `VkContext`,
// already `unsafe impl Send`.
unsafe impl Send for ReflectionCompositeResources {}

// SPIR-V blobs for the composite pipelines.
pub(in crate::vulkan) struct ReflectionCompositeShaders {
    pub vs: Vec<u8>,
    pub blur_fs: Vec<u8>,
    pub composite_fs: Vec<u8>,
}

// Compile the shared fullscreen vertex shader + the blur + composite fragments.
// The vertex shader is the SSR resolve's fullscreen triangle (same no-flip
// [0,1] UV convention, so the composite taps line up with the resolve's).
pub(in crate::vulkan) fn compile_reflection_composite_shaders(
    hot_reload: bool,
) -> RenderResult<ReflectionCompositeShaders> {
    use super::super::builtin_shaders;
    Ok(ReflectionCompositeShaders {
        vs: builtin_shaders::FULLSCREEN_VERT.compile(hot_reload)?,
        blur_fs: builtin_shaders::REFLECTION_BLUR.compile(hot_reload)?,
        composite_fs: builtin_shaders::REFLECTION_COMPOSITE.compile(hot_reload)?,
    })
}

// Replacement composite pipelines from a shader hot-reload.
pub(in crate::vulkan) struct RebuiltReflectionComposite {
    pub blur: OwnedPipeline,
    pub composite: OwnedPipeline,
}

pub(in crate::vulkan) fn rebuild_reflection_composite_pipelines(
    device: &VkDevice,
    rc: &ReflectionCompositeResources,
    hot_reload: bool,
) -> RenderResult<RebuiltReflectionComposite> {
    let shaders = compile_reflection_composite_shaders(hot_reload)?;
    let blur = create_composite_pipeline(
        device,
        rc.render_pass.handle(),
        rc.blur_pipeline_layout.handle(),
        &shaders.vs,
        &shaders.blur_fs,
    )?;
    let composite = create_composite_pipeline(
        device,
        rc.render_pass.handle(),
        rc.composite_pipeline_layout.handle(),
        &shaders.vs,
        &shaders.composite_fs,
    )?;
    Ok(RebuiltReflectionComposite { blur, composite })
}

// Composite render pass: one HDR-format color attachment, no depth. The fullscreen
// triangle overwrites every pixel so DONT_CARE is safe on load. Ends shader-readable
// for the next pass (composite -> bloom/TAA; blur -> composite). Mirrors the SSR
// resolve render pass.
// Sources the reflection blur and composite fragments sample, each through a
// sampler of its own.
const BLUR_SOURCES: u32 = 2;
const COMPOSITE_SOURCES: u32 = 5;

fn create_composite_render_pass(device: &VkDevice) -> RenderResult<OwnedRenderPass> {
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
    // Synchronize every prior color write + shader read (the resolve output, the
    // pass-1 blur write, the scene + G-buffer) against this pass's reads + write.
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
        .map_err(|e| crate::vulkan::error::map_vk_result(e, "reflection composite render pass"))
}

// Per-frame views feeding the composite's static bindings: the scene HDR resolve
// and the unified pre-pass normal+depth and roughness. Wired at init / resize; the
// reflection binding is re-pointed per encode.
pub(in crate::vulkan) struct CompositeInputs {
    hdr_resolve_views: Vec<vk::ImageView>,
    normal_depth_views: Vec<vk::ImageView>,
    roughness_views: Vec<vk::ImageView>,
}

impl CompositeInputs {
    pub(in crate::vulkan) fn new(
        hdr_resolve_images: &[GpuImage],
        gbuffer: &GbufferResources,
    ) -> Self {
        Self {
            hdr_resolve_views: hdr_resolve_images.iter().map(|img| img.view).collect(),
            normal_depth_views: gbuffer.normal_depth_views(),
            roughness_views: gbuffer.roughness_views(),
        }
    }
}

// One full-screen color target pre-transitioned to SHADER_READ_ONLY_OPTIMAL so the
// descriptor sets bound to it at init see a valid layout before the first encode.
fn create_target(ctx: &GpuUploadContext, width: u32, height: u32) -> RenderResult<GpuImage> {
    let &GpuUploadContext {
        alloc,
        device,
        command_pool,
        queue,
    } = ctx;
    let pooled = create_image(
        alloc,
        &super::super::texture::ImageSpec {
            width,
            height,
            format: HDR_FORMAT,
            tiling: vk::ImageTiling::OPTIMAL,
            // TRANSFER_SRC so the transparent (glass) pass can snapshot the
            // post-reflection scene for its refraction tap, the same usage the SSR
            // output carried before the composite owned the scene image.
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC,
            mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            samples: vk::SampleCountFlags::TYPE_1,
        },
    )?;
    let image = pooled.image();
    one_shot_submit(device, command_pool, queue, |cmd| {
        transition_image_layout(
            device,
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
        );
    })?;
    let view = create_image_view(device, image, HDR_FORMAT, vk::ImageAspectFlags::COLOR)?;
    Ok(GpuImage::from_pooled(pooled, view))
}

// Fullscreen pipeline: no vertex input, no depth, no blend; writes one HDR target.
// Mirrors the SSR resolve pipeline.
fn create_composite_pipeline(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> RenderResult<OwnedPipeline> {
    let modules = GraphicsStages::new(device, vert_spv, frag_spv)?;
    let stages = modules.infos();
    let vert_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::ALWAYS);
    let blend_attach = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(false);
    let blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(std::slice::from_ref(&blend_attach));
    let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);

    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vert_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);
    let pipeline = crate::vulkan::pipeline_cache::create_graphics_pipeline(device, &info)
        .map_err(|e| crate::vulkan::error::map_vk_result(e, "create reflection composite pso"))?;
    Ok(pipeline)
}

impl ReflectionCompositeResources {
    // Build every composite resource, wiring `inputs` into its static bindings.
    // `blur_scale` is the per-axis blur divisor.
    pub(in crate::vulkan) fn new(
        ctx: &GpuUploadContext,
        width: u32,
        height: u32,
        frames: usize,
        blur_scale: u32,
        inputs: &CompositeInputs,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let device = ctx.device;
        let blur_scale = blur_scale.max(1);
        let render_pass = create_composite_render_pass(device)?;

        // Blur set 0: reflection + roughness. Composite set 0: reflection + scene +
        // gbuffer normal+depth + roughness + blur. Each source's sampler follows
        // the images, in the same order.
        let blur_bindings = source_set_bindings(BLUR_SOURCES);
        let composite_bindings = source_set_bindings(COMPOSITE_SOURCES);
        let blur_set_layout = create_descriptor_set_layout(device, &blur_bindings)?;
        let composite_set_layout = create_descriptor_set_layout(device, &composite_bindings)?;

        let make_layout = |set_layout: vk::DescriptorSetLayout, name: &str| -> RenderResult<_> {
            let layouts = [set_layout];
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                )
                .map_err(|e| crate::vulkan::error::map_vk_result(e, name))
        };
        let blur_pipeline_layout = make_layout(blur_set_layout.handle(), "reflection blur layout")?;
        let composite_pipeline_layout =
            make_layout(composite_set_layout.handle(), "reflection composite layout")?;

        let shaders = compile_reflection_composite_shaders(hot_reload)?;
        let blur_pso = create_composite_pipeline(
            device,
            render_pass.handle(),
            blur_pipeline_layout.handle(),
            &shaders.vs,
            &shaders.blur_fs,
        )?;
        let composite_pso = create_composite_pipeline(
            device,
            render_pass.handle(),
            composite_pipeline_layout.handle(),
            &shaders.vs,
            &shaders.composite_fs,
        )?;

        // Pool: per-frame blur sets + composite sets.
        let f = frames as u32;
        let pool_sizes = PoolSizes::default()
            .sets(&blur_bindings, f)
            .sets(&composite_bindings, f)
            .build();
        let descriptor_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(f * 2),
            )
            .map_err(|e| {
                crate::vulkan::error::map_vk_result(e, "reflection composite descriptor pool")
            })?;
        let blur_layouts: Vec<_> = (0..frames).map(|_| blur_set_layout.handle()).collect();
        let blur_sets = alloc_descriptor_sets(device, descriptor_pool.handle(), &blur_layouts)?;
        let composite_layouts: Vec<_> =
            (0..frames).map(|_| composite_set_layout.handle()).collect();
        let composite_sets =
            alloc_descriptor_sets(device, descriptor_pool.handle(), &composite_layouts)?;

        let sampler = create_sampler_linear_clamp(device)?;

        let mut me = Self {
            output: GpuImage::null(),
            output_framebuffer: OwnedFramebuffer::null(),
            blur: GpuImage::null(),
            blur_framebuffer: OwnedFramebuffer::null(),
            blur_extent: vk::Extent2D::default(),
            render_pass,
            _blur_set_layout: blur_set_layout,
            _composite_set_layout: composite_set_layout,
            blur_pipeline_layout,
            composite_pipeline_layout,
            blur_pso,
            composite_pso,
            _descriptor_pool: descriptor_pool,
            blur_sets,
            composite_sets,
            sampler,
            blur_scale,
            wired_reflection: WireCache::new(frames),
        };
        me.build_targets(ctx, width, height)?;
        me.wire_sets(device, inputs);
        Ok(me)
    }

    // Allocate / re-allocate the resolution-dependent output + blur targets and
    // their framebuffers at the given extent.
    fn build_targets(
        &mut self,
        ctx: &GpuUploadContext,
        width: u32,
        height: u32,
    ) -> RenderResult<()> {
        let device = ctx.device;
        let w = width.max(1);
        let h = height.max(1);
        let bw = (w / self.blur_scale).max(1);
        let bh = (h / self.blur_scale).max(1);
        self.output = create_target(ctx, w, h)?;
        self.blur = create_target(ctx, bw, bh)?;
        self.blur_extent = vk::Extent2D {
            width: bw,
            height: bh,
        };

        let make_fb = |view: vk::ImageView, fw: u32, fh: u32| -> RenderResult<OwnedFramebuffer> {
            device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(self.render_pass.handle())
                        .attachments(std::slice::from_ref(&view))
                        .width(fw)
                        .height(fh)
                        .layers(1),
                )
                .map_err(|e| {
                    crate::vulkan::error::map_vk_result(e, "reflection composite framebuffer")
                })
        };
        self.output_framebuffer = make_fb(self.output.view, w, h)?;
        self.blur_framebuffer = make_fb(self.blur.view, bw, bh)?;
        Ok(())
    }

    // Wire the per-frame static bindings: blur set binding 1 = roughness; composite
    // set bindings 1..4 = scene / normal+depth / roughness / blur. Binding 0 (the
    // reflection target) is left at a valid placeholder and re-pointed per encode.
    // Every view slice holds one view per frame in flight.
    fn wire_sets(&self, device: &VkDevice, inputs: &CompositeInputs) {
        let hdr_resolve_views = inputs.hdr_resolve_views.as_slice();
        let normal_depth_views = inputs.normal_depth_views.as_slice();
        let roughness_views = inputs.roughness_views.as_slice();
        let frames = self.blur_sets.len();
        debug_assert_eq!(hdr_resolve_views.len(), frames);
        debug_assert_eq!(normal_depth_views.len(), frames);
        debug_assert_eq!(roughness_views.len(), frames);
        let sampler = self.sampler.handle();
        for i in 0..frames {
            let placeholder = hdr_resolve_views[i];
            let rough = roughness_views[i];
            // Blur set: 0 = reflection placeholder, 1 = roughness.
            write_source_set(
                device,
                self.blur_sets[i],
                &[(placeholder, sampler), (rough, sampler)],
            );
            // Composite set: 0 = reflection placeholder, 1 = scene, 2 = normal+depth,
            // 3 = roughness, 4 = blur.
            write_source_set(
                device,
                self.composite_sets[i],
                &[
                    (placeholder, sampler),
                    (hdr_resolve_views[i], sampler),
                    (normal_depth_views[i], sampler),
                    (rough, sampler),
                    (self.blur.view, sampler),
                ],
            );
        }
    }

    fn destroy_targets(&mut self, _device: &VkDevice) {
        self.output_framebuffer = OwnedFramebuffer::null();
        self.blur_framebuffer = OwnedFramebuffer::null();
        self.output = GpuImage::null();
        self.blur = GpuImage::null();
    }

    // Rebuild the resolution-dependent targets at a new extent and re-wire the
    // per-frame static bindings (the scene / G-buffer / blur views all moved). The
    // caller has already idled the device.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        ctx: &GpuUploadContext,
        width: u32,
        height: u32,
        inputs: &CompositeInputs,
    ) -> RenderResult<()> {
        self.destroy_targets(ctx.device);
        self.build_targets(ctx, width, height)?;
        self.wire_sets(ctx.device, inputs);
        // The resolves moved with everything else; drop the memo so the next
        // frame re-points binding 0 unconditionally.
        self.wired_reflection.reset();
        Ok(())
    }

    // Point every frame's binding 0 (blur + composite) at `view`, the resolve
    // target that feeds the composite this frame, skipping the write when this
    // slot already names it. Called on `&mut self` before any pass records, so
    // the update lands ahead of the workers that bind these sets; the slot is
    // fence-gated (its previous submission completed at the top of the frame).
    pub(in crate::vulkan) fn repoint_reflection(
        &mut self,
        device: &VkDevice,
        frame_idx: usize,
        view: vk::ImageView,
    ) {
        if !self.wired_reflection.changed(frame_idx, view) {
            return;
        }
        let refl = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view);
        let write = |set: vk::DescriptorSet| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(std::slice::from_ref(&refl))
        };
        let writes = [
            write(self.blur_sets[frame_idx]),
            write(self.composite_sets[frame_idx]),
        ];
        // SAFETY: `writes` and the image info it borrows are live for the call, and every set and
        // resource it names belongs to this device.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    // Swap freshly-built pipelines into the live resources after a hot-reload.
    pub(in crate::vulkan) fn swap_pipelines(&mut self, rebuilt: RebuiltReflectionComposite) {
        self.blur_pso = rebuilt.blur;
        self.composite_pso = rebuilt.composite;
    }

    // Destroy every composite resource. The caller has already idled the device.
    pub(in crate::vulkan) fn destroy(&mut self, device: &VkDevice) {
        self.destroy_targets(device);
    }
}

// Which reflection stages exist. RT takes the resolve slot when it is live, the
// SSR resolve runs only when SSR is authored and RT did not take the slot, and
// the composite exists exactly when one of the two feeds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::vulkan) struct ReflectionPath {
    pub(in crate::vulkan) ssr_resolve: bool,
    pub(in crate::vulkan) composite: bool,
}

impl ReflectionPath {
    pub(in crate::vulkan) fn new(ssr_authored: bool, rt_active: bool) -> Self {
        let ssr_resolve = ssr_authored && !rt_active;
        Self {
            ssr_resolve,
            composite: rt_active || ssr_resolve,
        }
    }
}

impl VkContext {
    fn reflection_path(&self) -> ReflectionPath {
        ReflectionPath::new(self.ssr_authored(), self.rt_reflections_active())
    }

    // Whether the SSR resolve runs: SSR is authored and RT did not take its slot.
    pub(in crate::vulkan) fn ssr_resolve_active(&self) -> bool {
        self.reflection_path().ssr_resolve
    }

    // The pre-TAA scene image for frame slot `frame`: the composite's output when
    // one exists, which is exactly when a resolve feeds it.
    pub(in crate::vulkan) fn post_scene_image(&self, frame: usize) -> &GpuImage {
        match self.reflection_composite.as_ref() {
            Some(rc) => &rc.output,
            None => &self.targets.hdr_resolve_images[frame % self.targets.hdr_resolve_images.len()],
        }
    }

    // Destroy the composite once no resolve feeds it, reporting whether it went.
    // The caller rebuilds the swapchain when it did, which re-points every scene
    // reader at the HDR resolve.
    pub(in crate::vulkan) fn release_unfed_reflection_composite(&mut self) -> bool {
        if self.reflection_path().composite {
            return false;
        }
        let Some(mut rc) = self.reflection_composite.take() else {
            return false;
        };
        rc.destroy(&self.hw.device);
        true
    }

    // Bring the composite in line with the authored SSR and the live RT state,
    // building it at `blur_scale` when a resolve feeds it. The caller idles the
    // device first and rebuilds the swapchain after.
    pub(in crate::vulkan) fn reconcile_reflection_composite(
        &mut self,
        blur_scale: u32,
    ) -> RenderResult<()> {
        self.release_unfed_reflection_composite();
        if !self.reflection_path().composite || self.reflection_composite.is_some() {
            return Ok(());
        }
        let gb = self
            .gbuffer
            .as_ref()
            .expect("a reflection path forces the unified G-buffer pre-pass");
        let rc = ReflectionCompositeResources::new(
            &GpuUploadContext {
                alloc: &self.hw.alloc,
                device: &self.hw.device,
                command_pool: self.commands.command_pool,
                queue: self.hw.graphics_queue,
            },
            self.targets.render_extent.width,
            self.targets.render_extent.height,
            self.frames_in_flight,
            blur_scale,
            &CompositeInputs::new(&self.targets.hdr_resolve_images, gb),
            self.hot_reload.enabled,
        )?;
        self.reflection_composite = Some(rc);
        Ok(())
    }

    // Point this frame's composite sets at the resolve target that will feed
    // them: the RT output when the trace is live (RT takes the `SsrResolve`
    // slot), the SSR output otherwise. Reads the same `rt_reflections_active`
    // the frame graph gates `rt_reflections_enabled` on, so the wiring and the
    // pass that encodes always agree. Runs on `&mut self` ahead of the parallel
    // recording, and skips the write unless the view actually moved.
    pub(in crate::vulkan) fn prepare_reflection_composite(&mut self, frame_idx: usize) {
        debug_assert_eq!(
            self.reflection_composite.is_some(),
            self.reflection_path().composite,
            "the reflection composite is out of step with the resolves that feed it"
        );
        let view = if self.rt_reflections_active() {
            self.rt_reflections.as_ref().map(|rt| rt.output.view)
        } else {
            self.ssr.as_ref().map(|ssr| ssr.output.view())
        };
        let Some(view) = view else {
            return;
        };
        let device = self.hw.device.clone();
        if let Some(rc) = self.reflection_composite.as_mut() {
            rc.repoint_reflection(&device, frame_idx, view);
        }
    }

    // Blur the reflection target by surface roughness and composite it over the base
    // HDR scene into `reflection_composite.output`. `reflection_view` is the resolve
    // target the SSR / RT pass just wrote (radiance + weight), in
    // SHADER_READ_ONLY_OPTIMAL after its render pass. Encoded inline at the tail of
    // `encode_ssr_resolve` / `encode_rt_reflections`. No-op when the composite is
    // absent (no reflection path active). Binding 0 was pointed at this view by
    // `prepare_reflection_composite` before any pass recorded; the assert catches
    // a resolve encoding against a set wired for the other one.
    pub(in crate::vulkan) fn encode_reflection_composite(
        &self,
        cmd: vk::CommandBuffer,
        reflection_view: vk::ImageView,
        frame_idx: usize,
    ) {
        let Some(rc) = &self.reflection_composite else {
            return;
        };
        debug_assert_eq!(
            rc.wired_reflection.current(frame_idx),
            Some(reflection_view),
            "reflection composite set wired for a different resolve target"
        );
        let device = &self.hw.device;
        // Pass 1: roughness blur into the reduced-resolution blur target.
        self.begin_fullscreen_pass_sized(
            cmd,
            rc.render_pass.handle(),
            rc.blur_framebuffer.handle(),
            rc.blur_extent,
        );
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, rc.blur_pso.handle());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                rc.blur_pipeline_layout.handle(),
                0,
                std::slice::from_ref(&rc.blur_sets[frame_idx]),
                &[],
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
        }
        self.end_fullscreen_pass(cmd);

        // Pass 2: lerp sharp vs upsampled blur by roughness, composite over scene.
        self.begin_fullscreen_pass(cmd, rc.render_pass.handle(), rc.output_framebuffer.handle());
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                rc.composite_pso.handle(),
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                rc.composite_pipeline_layout.handle(),
                0,
                std::slice::from_ref(&rc.composite_sets[frame_idx]),
                &[],
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
        }
        self.end_fullscreen_pass(cmd);
    }
}

#[cfg(test)]
mod tests {
    use super::ReflectionPath;

    #[test]
    fn rt_takes_the_resolve_slot_from_authored_ssr() {
        let path = ReflectionPath::new(true, true);
        assert!(!path.ssr_resolve);
        assert!(path.composite);
    }

    #[test]
    fn authored_ssr_resolves_without_rt() {
        let path = ReflectionPath::new(true, false);
        assert!(path.ssr_resolve);
        assert!(path.composite);
    }

    #[test]
    fn rt_alone_keeps_the_composite() {
        let path = ReflectionPath::new(false, true);
        assert!(!path.ssr_resolve);
        assert!(path.composite);
    }

    #[test]
    fn no_resolve_leaves_no_composite() {
        // RT lost without authored SSR, or a SSGI-only world.
        let path = ReflectionPath::new(false, false);
        assert!(!path.ssr_resolve);
        assert!(!path.composite);
    }

    // The composite vert + blur + composite fragments compile to SPIR-V. Guards the
    // GLSL so a shader error fails a test instead of only an init failure on the GPU
    // host. The composite passes carry no push constant / UBO, so there is no
    // CPU<->GPU layout to assert.
    #[test]
    fn reflection_composite_shaders_compile() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        let shaders = super::compile_reflection_composite_shaders(false)
            .expect("reflection composite shaders compile");
        assert!(super::is_spirv(&shaders.vs));
        assert!(super::is_spirv(&shaders.blur_fs));
        assert!(super::is_spirv(&shaders.composite_fs));
    }
}
