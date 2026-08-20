// src/vulkan/post/ssr.rs
//
// Screen-space reflections for the Vulkan backend. Owns the fullscreen
// ray-march resolve render pass and pipeline, the per-frame resolve descriptor
// sets, and the `encode_ssr_resolve` per-frame encoder. The depth + normal +
// roughness the resolve samples come from the unified G-buffer pre-pass; the
// resolve sets are wired to its per-frame views by `init.rs`. Mirrors
// src/directx/post/ssr.rs and src/metal/post/ssr.rs.
//
// When SSR is on the bloom prefilter + composite + (optional) TAA scene input
// is re-pointed at `SsrResources::output` so the post-process stack consumes
// the HDR scene with reflections composited in.

use ash::{Device, vk};

use crate::gfx::fullscreen::{FullscreenPass, encode_fullscreen};
use crate::gfx::render_types::SsrParams;
use crate::vulkan::allocator::DeviceAllocator;

use super::super::context::VkContext;
use super::super::pipeline::*;
use super::super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::super::texture::*;

// HDR-format SSR resolve output. Replaces the raw HDR resolve as the scene
// input the TAA / bloom / composite passes consume when SSR is on.
pub(in crate::vulkan) const SSR_OUTPUT_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

// SSR resources held by `VkContext` when `PostProcessConfig.ssr` is on.
// All `vk::*` handles are owned by this struct and freed on `destroy`.
pub(in crate::vulkan) struct SsrResources {
    // Resolved authored tunables; turned into a per-frame `SsrParams` push.
    pub(in crate::vulkan) settings: crate::gfx::ssr::SsrSettings,

    // Render pass.
    pub(in crate::vulkan) resolve_render_pass: vk::RenderPass,

    // Resolve pipeline: fullscreen triangle reading the scene + G-buffer +
    // roughness + prefilter cubemap, writing `output`.
    pub(in crate::vulkan) resolve_set_layout: vk::DescriptorSetLayout,
    pub(in crate::vulkan) resolve_layout: vk::PipelineLayout,
    pub(in crate::vulkan) resolve_pso: vk::Pipeline,

    // Probe cube-array length the resolve fragment was built against, kept so a
    // hot-reload recompile sizes its `probe_cubes[]` to the same global set
    // layout the pipeline layout already references.
    probe_cube_count: u32,

    // Per-frame resolve sets: each binds that frame's HDR resolve as the
    // scene SRV. The G-buffer / roughness come from the unified pre-pass's
    // per-frame views; the prefilter cube is shared across all frames.
    pub(in crate::vulkan) resolve_sets: Vec<vk::DescriptorSet>,
    pub(in crate::vulkan) descriptor_pool: vk::DescriptorPool,

    // Linear-clamp sampler the resolve reads scene / G-buffer / roughness
    // through. The cubemap fallback uses VkContext's `cube_sampler`.
    pub(in crate::vulkan) sampler: vk::Sampler,

    // Resolution-dependent target (rebuilt on swapchain resize).
    pub(in crate::vulkan) output: GpuImage,
    pub(in crate::vulkan) resolve_framebuffer: vk::Framebuffer,
}

// SPIR-V blobs for every SSR pipeline. Produced by [`compile_ssr_shaders`];
// consumed by `SsrResources::new` at init and by `rebuild_ssr_pipelines`
// during shader hot-reload. Mirrors the matching SSAO struct.
pub(in crate::vulkan) struct SsrShaders {
    pub fullscreen_vs: Vec<u8>,
    pub resolve_fs: Vec<u8>,
}

// Compile every SSR GLSL source. `hot_reload` routes each source resolve
// through the builtins' disk-first path. The resolve fragment's shared
// reflection-probe sampling ({PROBE_COMMON} / {MAX_PROBES} / {PROBE_DESC_SET}
// = 1, the global set carrying the probe set/cubes here) is substituted by
// the builtins assembly, sized by `probe_cube_count` (that global set layout's
// binding-8 descriptor count). Mirrors `compile_rt_shaders`.
pub(in crate::vulkan) fn compile_ssr_shaders(
    hot_reload: bool,
    probe_cube_count: u32,
) -> Result<SsrShaders, String> {
    use super::super::{builtins, slang_builtins};
    let ctx = builtins::Ctx {
        probe_count: probe_cube_count as usize,
        ..builtins::Ctx::plain(hot_reload)
    };
    Ok(SsrShaders {
        fullscreen_vs: slang_builtins::FULLSCREEN_VERT.compile(&ctx)?,
        resolve_fs: slang_builtins::SSR_RESOLVE.compile(&ctx)?,
    })
}

// Replacement SSR pipeline built by the hot-reload pass.
pub(in crate::vulkan) struct RebuiltSsrPipelines {
    pub resolve: vk::Pipeline,
}

// Rebuild the live SSR resolve pipeline from disk-resident GLSL source against
// the existing layout + render pass. Same shape as [`rebuild_ssao_pipelines`].
pub(in crate::vulkan) fn rebuild_ssr_pipelines(
    device: &Device,
    ssr: &SsrResources,
    hot_reload: bool,
) -> Result<RebuiltSsrPipelines, String> {
    let shaders = compile_ssr_shaders(hot_reload, ssr.probe_cube_count)?;
    let resolve = create_resolve_pipeline(
        device,
        ssr.resolve_render_pass,
        ssr.resolve_layout,
        &shaders.fullscreen_vs,
        &shaders.resolve_fs,
    )?;
    Ok(RebuiltSsrPipelines { resolve })
}

impl SsrResources {
    // Swap the freshly-built pipeline into the live resources. The caller
    // has already `device_wait_idle`'d so the old pipeline is not in
    // flight. Driven by the Vulkan shader hot-reload pass after the
    // replacement successfully compiled.
    pub(in crate::vulkan) fn swap_pipelines(
        &mut self,
        device: &Device,
        rebuilt: RebuiltSsrPipelines,
    ) {
        // SAFETY: the handle was created from this device and is destroyed exactly once; the caller
        // has already waited for the device to go idle, so no submission still references it.
        unsafe {
            device.destroy_pipeline(self.resolve_pso, None);
        }
        self.resolve_pso = rebuilt.resolve;
    }
}

// Resolve render pass: one HDR-format colour attachment, no depth. The
// fullscreen triangle overwrites every pixel so `DONT_CARE` is safe on load.
// Ends shader-readable for the bloom + composite passes.
fn create_resolve_render_pass(device: &Device) -> Result<vk::RenderPass, String> {
    let attachment = vk::AttachmentDescription::default()
        .format(SSR_OUTPUT_FORMAT)
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
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    unsafe { device.create_render_pass(&info, None) }
        .map_err(|e| format!("SSR resolve render pass: {e}"))
}

// Shared device/queue handles every SSR target build needs. Borrowed by
// `create_output_target`, `build_targets`, `new`, and `rebuild`.
pub(in crate::vulkan) struct SsrGpuContext<'a> {
    pub alloc: &'a DeviceAllocator,
    pub device: &'a Device,
    pub command_pool: vk::CommandPool,
    pub queue: vk::Queue,
}

// Resolution of the resolution-dependent SSR targets.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct SsrExtent {
    pub width: u32,
    pub height: u32,
}

// Output target: pre-transitioned to `SHADER_READ_ONLY_OPTIMAL` so the
// composite / bloom / TAA descriptor sets bound to it at init see a
// validly-laid-out image even before the first SSR resolve runs (e.g. the
// first frame's composite reads ssr.output before SSR has fired this slot).
fn create_output_target(gpu: &SsrGpuContext, extent: SsrExtent) -> Result<GpuImage, String> {
    let &SsrGpuContext {
        alloc,
        device,
        command_pool,
        queue,
    } = gpu;
    let SsrExtent { width, height } = extent;
    let pooled = create_image(
        alloc,
        &ImageSpec {
            width,
            height,
            format: SSR_OUTPUT_FORMAT,
            tiling: vk::ImageTiling::OPTIMAL,
            // TRANSFER_SRC so the transparent pass can snapshot this scene image for
            // its refraction tap (the glass pass copies the post-SSR scene into its
            // own snapshot, mirroring how `hdr_resolve` is the copy source when SSR
            // is off). Inert for the SSR resolve itself.
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
    let view = create_image_view(
        device,
        pooled.image(),
        SSR_OUTPUT_FORMAT,
        vk::ImageAspectFlags::COLOR,
    )?;
    Ok(GpuImage::from_pooled(pooled, view))
}

// Build the fullscreen resolve pipeline. No vertex input (the fullscreen
// triangle is procedural in the VS); no depth; no blend; writes the HDR
// resolve output target.
fn create_resolve_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> Result<vk::Pipeline, String> {
    let vert_mod = spv_module(device, vert_spv)?;
    let frag_mod = spv_module(device, frag_spv)?;
    let entry = std::ffi::CString::new("main").unwrap();

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_mod)
            .name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_mod)
            .name(&entry),
    ];
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
    // SAFETY: the create-infos and every slice they borrow are live for the call, and each handle
    // they name belongs to this device.
    let pipeline = unsafe {
        crate::vulkan::pipeline_cache::create_graphics_pipelines(
            device,
            std::slice::from_ref(&info),
        )
    }
    .map_err(|(_, e)| format!("create ssr resolve pso: {e}"))?[0];
    // SAFETY: the shader module was created from this device, and a module may be destroyed as soon
    // as the pipelines that consumed it exist.
    unsafe {
        device.destroy_shader_module(vert_mod, None);
        device.destroy_shader_module(frag_mod, None);
    }
    Ok(pipeline)
}

// Content + wiring inputs `SsrResources::new` builds the resolve from, apart
// from the device/queue infra (`SsrGpuContext`), the extent, frame count, and
// the hot-reload flag.
pub(in crate::vulkan) struct SsrInitInputs<'a> {
    // Resolved authored tunables.
    pub settings: crate::gfx::ssr::SsrSettings,
    // Per-frame HDR resolve scene views feeding the resolve sets.
    pub hdr_resolve_views: &'a [vk::ImageView],
    // Prefilter cubemap + its sampler for the IBL fallback all resolve sets bind.
    pub prefilter_view: vk::ImageView,
    pub cube_sampler: vk::Sampler,
    // The forward global set's layout, bound as set 1 so the resolve can sample
    // the reflection-probe set + cube array (binding 7/8) on a missed ray, and
    // that layout's binding-8 descriptor count (sizes the fragment's array).
    pub global_set_layout: vk::DescriptorSetLayout,
    pub probe_cube_count: u32,
}

// The per-frame view + sampler wiring the resolve descriptor sets bind on a
// rebuild. `gbuffer_views` / `roughness_views` carry the unified pre-pass's
// per-frame views (empty falls back to the scene view); `prefilter_view` +
// `cube_sampler` feed the IBL fallback.
pub(in crate::vulkan) struct SsrResolveInputs<'a> {
    pub hdr_resolve_views: &'a [vk::ImageView],
    pub gbuffer_views: &'a [vk::ImageView],
    pub roughness_views: &'a [vk::ImageView],
    pub prefilter_view: vk::ImageView,
    pub cube_sampler: vk::Sampler,
}

impl SsrResources {
    // Build every SSR resource. `hdr_resolve_views` feeds the per-frame
    // resolve descriptor sets; `prefilter_view` + `cube_sampler` feed the IBL
    // fallback all resolve sets bind. The G-buffer / roughness the resolve
    // samples come from the unified pre-pass; the caller re-points those
    // bindings at its per-frame views after construction.
    pub(in crate::vulkan) fn new(
        gpu: &SsrGpuContext,
        extent: SsrExtent,
        frames: usize,
        inputs: SsrInitInputs,
        hot_reload: bool,
    ) -> Result<Self, String> {
        let device = gpu.device;
        let SsrInitInputs {
            settings,
            hdr_resolve_views,
            prefilter_view,
            cube_sampler,
            global_set_layout,
            probe_cube_count,
        } = inputs;
        let resolve_render_pass = create_resolve_render_pass(device)?;

        // Resolve set 0: scene + gbuffer + roughness + prefilter cube samplers.
        let resolve_set_layout = create_descriptor_set_layout(
            device,
            &[
                (
                    0,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
                (
                    1,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
                (
                    2,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
                (
                    3,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
            ],
        )?;

        let params_push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<SsrParams>() as u32);
        // set 0 = the resolve set (scene/gbuffer/roughness/prefilter); set 1 = the
        // global set (probe set/cubes) for the missed-ray probe fallback.
        let resolve_set_layouts = [resolve_set_layout, global_set_layout];
        // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
        // it names belongs to this device.
        let resolve_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&resolve_set_layouts)
                    .push_constant_ranges(std::slice::from_ref(&params_push)),
                None,
            )
        }
        .map_err(|e| format!("ssr resolve layout: {e}"))?;

        // Pipeline.
        let shaders = compile_ssr_shaders(hot_reload, probe_cube_count)?;
        let resolve_pso = create_resolve_pipeline(
            device,
            resolve_render_pass,
            resolve_layout,
            &shaders.fullscreen_vs,
            &shaders.resolve_fs,
        )?;

        // Descriptor pool: `frames` resolve sets (4 samplers each).
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(frames as u32 * 4)];
        // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
        // it names belongs to this device.
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(frames as u32),
                None,
            )
        }
        .map_err(|e| format!("ssr descriptor pool: {e}"))?;

        let resolve_layouts_vec: Vec<_> = (0..frames).map(|_| resolve_set_layout).collect();
        let resolve_sets = alloc_descriptor_sets(device, descriptor_pool, &resolve_layouts_vec)?;

        // Dedicated linear-clamp sampler the resolve reads scene / G-buffer /
        // roughness through.
        let sampler = create_sampler_linear_clamp(device)?;

        let mut me = Self {
            settings,
            resolve_render_pass,
            resolve_set_layout,
            resolve_layout,
            resolve_pso,
            probe_cube_count,
            resolve_sets,
            descriptor_pool,
            sampler,
            // Placeholder GpuImage; replaced by build_targets below.
            output: GpuImage::null(),
            resolve_framebuffer: vk::Framebuffer::null(),
        };
        me.build_targets(gpu, extent)?;
        // Bind scene + prefilter cube; the G-buffer / roughness slots are
        // re-pointed at the unified pre-pass per-frame views by the caller
        // (the live path) before the first frame. Until then they fall back to
        // the scene view so every binding is a valid `SHADER_READ_ONLY` image.
        me.wire_resolve_sets(
            device,
            hdr_resolve_views,
            &[],
            &[],
            prefilter_view,
            cube_sampler,
        );
        Ok(me)
    }

    // Allocate or re-allocate the resolution-dependent output target +
    // framebuffer at the given extent.
    fn build_targets(&mut self, gpu: &SsrGpuContext, extent: SsrExtent) -> Result<(), String> {
        let device = gpu.device;
        let w = extent.width.max(1);
        let h = extent.height.max(1);
        self.output = create_output_target(
            gpu,
            SsrExtent {
                width: w,
                height: h,
            },
        )?;

        // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
        // it names belongs to this device.
        self.resolve_framebuffer = unsafe {
            device.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(self.resolve_render_pass)
                    .attachments(std::slice::from_ref(&self.output.view))
                    .width(w)
                    .height(h)
                    .layers(1),
                None,
            )
        }
        .map_err(|e| format!("ssr resolve framebuffer: {e}"))?;
        Ok(())
    }

    // Wire the per-frame resolve descriptor sets to the current scene
    // (HDR resolve) / G-buffer / roughness / prefilter cubemap. Called after
    // `build_targets` (init or resize) so the resolve sees the current images.
    //
    // `gbuffer_views` / `roughness_views` carry the unified pre-pass's per-frame
    // normal+depth / roughness views; resolve set `i` binds slot `i`. When empty
    // (the init pre-wire before the caller re-points them) the scene view stands
    // in so every binding is a valid `SHADER_READ_ONLY` image.
    pub(in crate::vulkan) fn wire_resolve_sets(
        &self,
        device: &Device,
        hdr_resolve_views: &[vk::ImageView],
        gbuffer_views: &[vk::ImageView],
        roughness_views: &[vk::ImageView],
        prefilter_view: vk::ImageView,
        cube_sampler: vk::Sampler,
    ) {
        let cube_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(prefilter_view)
            .sampler(cube_sampler);
        for (i, &set) in self.resolve_sets.iter().enumerate() {
            let scene_placeholder = hdr_resolve_views[i % hdr_resolve_views.len().max(1)];
            // Unified G-buffer per-frame views once re-pointed; the scene view
            // stands in for the init pre-wire so the binding is always valid.
            let gb_view = if gbuffer_views.is_empty() {
                scene_placeholder
            } else {
                gbuffer_views[i % gbuffer_views.len()]
            };
            let rough_view = if roughness_views.is_empty() {
                scene_placeholder
            } else {
                roughness_views[i % roughness_views.len()]
            };
            let gb_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(gb_view)
                .sampler(self.sampler);
            let rough_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(rough_view)
                .sampler(self.sampler);
            // Each resolve set binds its frame's HDR resolve as the scene
            // input. With fewer HDR resolves than frames (shouldn't happen)
            // we wrap.
            let scene_view = hdr_resolve_views[i % hdr_resolve_views.len().max(1)];
            let scene_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(scene_view)
                .sampler(self.sampler);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&scene_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&gb_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&rough_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&cube_info)),
            ];
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }

    fn destroy_targets(&mut self, device: &Device) {
        if self.resolve_framebuffer != vk::Framebuffer::null() {
            // SAFETY: the handle was created from this device and is destroyed exactly once; the
            // caller has already waited for the device to go idle, so no submission still
            // references it.
            unsafe {
                device.destroy_framebuffer(self.resolve_framebuffer, None);
            }
            self.resolve_framebuffer = vk::Framebuffer::null();
            self.output = GpuImage::null();
        }
    }

    // Rebuild the resolution-dependent targets at a new swapchain extent and
    // re-wire the resolve descriptor sets. The caller has already idled the
    // device.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        gpu: &SsrGpuContext,
        extent: SsrExtent,
        inputs: SsrResolveInputs,
    ) -> Result<(), String> {
        let device = gpu.device;
        let SsrResolveInputs {
            hdr_resolve_views,
            gbuffer_views,
            roughness_views,
            prefilter_view,
            cube_sampler,
        } = inputs;
        self.destroy_targets(device);
        self.build_targets(gpu, extent)?;
        self.wire_resolve_sets(
            device,
            hdr_resolve_views,
            gbuffer_views,
            roughness_views,
            prefilter_view,
            cube_sampler,
        );
        Ok(())
    }

    // Destroy every SSR resource. The caller has already idled the device.
    pub(in crate::vulkan) fn destroy(&mut self, device: &Device) {
        self.destroy_targets(device);
        // SAFETY: the handle was created from this device and is destroyed exactly once; the caller
        // has already waited for the device to go idle, so no submission still references it.
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline(self.resolve_pso, None);
            device.destroy_pipeline_layout(self.resolve_layout, None);
            device.destroy_descriptor_set_layout(self.resolve_set_layout, None);
            device.destroy_render_pass(self.resolve_render_pass, None);
        }
    }
}

// Encoders
//
// The downstream "scene image" the post stack should sample (SSR output when
// SSR is on, HDR resolve otherwise) is wired statically into the bloom /
// composite / TAA descriptor sets at init (and on swapchain resize), so this
// module exposes no runtime accessor; mirrors DirectX's `scene_srv_for_post`
// only in spirit.
impl VkContext {
    // Encode the SSR resolve: a fullscreen triangle that ray-marches the
    // reflection through the HDR scene (already in `SHADER_READ_ONLY_OPTIMAL`
    // after the main pass) and composites the result into `ssr.output`. The
    // output then becomes the scene image the bloom / composite / TAA
    // consume; see `scene_view_for_post`. No-op when SSR is disabled.
    pub(in crate::vulkan) fn encode_ssr_resolve(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        fov_y_radians: f32,
        aspect: f32,
        cam_pos: [f32; 3],
    ) {
        let Some(ssr) = &self.ssr else { return };
        encode_fullscreen(
            &SsrResolvePass {
                ctx: self,
                ssr,
                frame_idx,
                fov_y_radians,
                aspect,
                cam_pos,
            },
            &cmd,
        );
        // Blur the resolve's radiance+weight by roughness and composite it over the
        // scene into the reflection composite's output (the scene image the post
        // stack consumes). No-op when the composite is absent.
        self.encode_reflection_composite(cmd, ssr.output.view, frame_idx);
    }
}

// Encoder for the SSR resolve fullscreen pass: the resolved resources + the
// per-call view params. The render-pass bracket + viewport / scissor live in
// `VkContext::begin/end_fullscreen_pass` (post/fullscreen.rs); only the
// SSR-specific bind + draw is here. Constructed + driven by `encode_ssr_resolve`
// through `gfx::fullscreen::encode_fullscreen`.
struct SsrResolvePass<'a> {
    ctx: &'a VkContext,
    ssr: &'a SsrResources,
    frame_idx: usize,
    fov_y_radians: f32,
    aspect: f32,
    cam_pos: [f32; 3],
}

impl FullscreenPass for SsrResolvePass<'_> {
    type Rec = vk::CommandBuffer;

    fn begin(&self, cmd: &Self::Rec) {
        self.ctx.begin_fullscreen_pass(
            *cmd,
            self.ssr.resolve_render_pass,
            self.ssr.resolve_framebuffer,
        );
    }

    fn draw(&self, cmd: &Self::Rec) {
        let cmd = *cmd;
        let device = &self.ctx.device;
        // The view→world rotation is the transpose of the view matrix's
        // orthonormal 3×3, embedded in a 4×4.
        let v = self.ctx.view_matrix;
        let inv_view_rot = [
            [v[0][0], v[1][0], v[2][0], 0.0],
            [v[0][1], v[1][1], v[2][1], 0.0],
            [v[0][2], v[1][2], v[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let params = self.ssr.settings.params(
            self.fov_y_radians,
            self.aspect,
            inv_view_rot,
            self.cam_pos,
            self.ctx.prefilter_mip_count as f32,
        );
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.ssr.resolve_pso);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.ssr.resolve_layout,
                0,
                std::slice::from_ref(&self.ssr.resolve_sets[self.frame_idx]),
                &[],
            );
            // set 1: the global set, for the reflection-probe miss fallback.
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.ssr.resolve_layout,
                1,
                std::slice::from_ref(&self.ctx.descriptors.global_sets[self.frame_idx]),
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.ssr.resolve_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                std::slice::from_raw_parts(
                    &params as *const SsrParams as *const u8,
                    std::mem::size_of::<SsrParams>(),
                ),
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
        }
    }

    fn end(&self, cmd: &Self::Rec) {
        self.ctx.end_fullscreen_pass(*cmd);
    }
}

#[cfg(test)]
mod tests {
    // The SSR fullscreen vert + resolve fragment compile to SPIR-V. Guards the
    // shared probe sampling injected at the resolve's PROBE_COMMON marker: the
    // {MAX_PROBES} / {PROBE_DESC_SET} substitution and the comment-token trap (a
    // brace token left in a comment would be substituted and break the GLSL). The
    // CPU<->GPU `SsrParams` layout is guarded by the `ssr_params_*` tests in
    // gfx::render_types.
    #[test]
    fn ssr_shaders_compile() {
        // Both the ceiling and a device-shortened probe cube array must compile.
        for probes in [1, concinnity_render::uniforms::MAX_PROBES as u32] {
            let shaders = super::compile_ssr_shaders(false, probes).expect("ssr shaders compile");
            assert!(super::is_spirv(&shaders.fullscreen_vs));
            assert!(super::is_spirv(&shaders.resolve_fs));
        }
    }
}
