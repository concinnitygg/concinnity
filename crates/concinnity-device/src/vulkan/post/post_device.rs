// src/vulkan/post/post_device.rs
//
// Vulkan's implementation of the shared fullscreen post-pass seam
// (`gfx::post::PostPassDevice`).
//
// Everything a pass used to own per effect is derived here from what the single
// source declares: the descriptor set layout is N combined image samplers, the
// pipeline layout adds a fragment push-constant range of the declared size, and
// the render pass comes from the target's format and load action. The sets
// themselves are allocated per frame (post/set_arena.rs) rather than pre-wired
// per effect, which is what removes the `rewire_*` a pass needed for every input
// another effect might own.

use ash::vk;

use concinnity_core::render::post::device::{
    PostBlend, PostDraw, PostExtent, PostLoadOp, PostPassDevice, PostSampler, resolved_texture,
};
use concinnity_core::render::post::program::PostProgram;
use concinnity_core::render::render_graph::{PixelFormat, TextureDesc};

use crate::vulkan::allocator::DeviceAllocator;
use crate::vulkan::owned::{OwnedPipeline, OwnedPipelineLayout, OwnedSetLayout, VkDevice};
use crate::vulkan::pipeline::spv_module;
use crate::vulkan::post::pass_cache::PostPassCache;
use crate::vulkan::post::set_arena::PostSetArena;
use crate::vulkan::slang_builtins::{self, SlangCompile};
use crate::vulkan::texture::{
    GpuImage, ImageSpec, create_image, create_image_view, one_shot_submit, transition_image_layout,
};
use crate::vulkan::transient_pool::{image_format, image_usage, sample_count};

// A built fullscreen post pipeline plus the layouts a draw binds it through.
// The two layouts travel with the pipeline because they are derived from the
// same program declaration, so nothing else has to know the binding count.
pub(in crate::vulkan) struct PostPipeline {
    pipeline: OwnedPipeline,
    layout: OwnedPipelineLayout,
    set_layout: OwnedSetLayout,
    // What the program declares, so a draw can check what it was handed.
    textures: usize,
    constants: usize,
}

// A persistent post target: the image, its view, and the extent its
// framebuffers were sized at.
pub(in crate::vulkan) struct PostTarget {
    image: GpuImage,
    extent: vk::Extent2D,
    format: PixelFormat,
}

impl PostTarget {
    /// The sampled view of this target, for a consumer that binds it directly.
    pub(in crate::vulkan) fn view(&self) -> vk::ImageView {
        self.image.view
    }
}

// The one-shot submit a freshly created target's initial layout transition
// needs. Held by the device value because target creation is the only operation
// that touches a queue.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct PostQueue {
    pub command_pool: vk::CommandPool,
    pub queue: vk::Queue,
}

// The Vulkan handles a shared post pass builds and encodes through. Borrowed,
// so the same value shape serves init (where no context exists yet) and each
// frame's encode.
pub(in crate::vulkan) struct VkPostDevice<'a> {
    pub device: &'a VkDevice,
    pub alloc: &'a DeviceAllocator,
    pub queue: PostQueue,
    // Render passes + framebuffers, cached across frames and passes.
    pub cache: &'a PostPassCache,
    // Per-frame descriptor sets.
    pub arena: &'a PostSetArena,
    // The linear clamp-to-edge state every screen-space source is sampled
    // through.
    pub sampler: vk::Sampler,
    // Which frame in flight is recording.
    pub frame: usize,
    pub hot_reload: bool,
}

// The SPIR-V a post program's two stages compile to: the one shared fullscreen
// triangle vertex plus the program's own fragment.
fn compile(program: PostProgram, hot_reload: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let ctx = crate::vulkan::builtins::Ctx::plain(hot_reload);
    let frag = match program {
        PostProgram::TaaResolve => &slang_builtins::TAA_FRAG,
    };
    let vert = slang_builtins::FULLSCREEN_VERT.compile(&ctx)?;
    Ok((vert, frag.compile(&ctx)?))
}

fn blend_attachment(blend: PostBlend) -> vk::PipelineColorBlendAttachmentState {
    let base = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    match blend {
        PostBlend::Replace => base.blend_enable(false),
        PostBlend::Additive => base
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE),
        PostBlend::PremultipliedOver => base
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA),
    }
}

impl VkPostDevice<'_> {
    // The descriptor set layout for `n` combined image samplers at bindings
    // `0..n`, fragment-visible. Derived from the program's declared count rather
    // than written per pass, which is what keeps the single source the contract.
    fn set_layout(&self, n: usize) -> Result<OwnedSetLayout, String> {
        let bindings: Vec<_> = (0..n as u32)
            .map(|b| {
                (
                    b,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ShaderStageFlags::FRAGMENT,
                )
            })
            .collect();
        crate::vulkan::resources::create_descriptor_set_layout(self.device, &bindings)
    }
}

impl VkPostDevice<'_> {
    // The graphics pipeline itself: a vertex-buffer-less fullscreen triangle
    // into one colour attachment, no depth, dynamic viewport and scissor. Every
    // fullscreen post pass has this shape, which is why it is built once here
    // instead of once per effect.
    fn build_pipeline(
        &self,
        program: PostProgram,
        render_pass: vk::RenderPass,
        layout: vk::PipelineLayout,
        blend: PostBlend,
    ) -> Result<OwnedPipeline, String> {
        let (vert_spv, frag_spv) = compile(program, self.hot_reload)?;
        let vert_mod = spv_module(self.device, &vert_spv)?;
        let frag_mod = spv_module(self.device, &frag_spv)?;
        let entry = std::ffi::CString::new("main").expect("the entry point name has no NUL");
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_mod.handle())
                .name(&entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_mod.handle())
                .name(&entry),
        ];
        let vert_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(false)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::ALWAYS);
        let attach = blend_attachment(blend);
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&attach));
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vert_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);
        crate::vulkan::pipeline_cache::create_graphics_pipeline(self.device, &info)
            .map_err(|e| format!("create post pipeline: {e}"))
    }
}

impl PostPassDevice for VkPostDevice<'_> {
    type Recorder = vk::CommandBuffer;
    type Pipeline = PostPipeline;
    type Target = PostTarget;
    type TextureRef<'a> = vk::ImageView;

    fn create_pipeline(
        &self,
        program: PostProgram,
        format: PixelFormat,
        blend: PostBlend,
    ) -> Result<Self::Pipeline, String> {
        let bindings = program.bindings();
        let set_layout = self.set_layout(bindings.textures)?;
        let set_layouts = [set_layout.handle()];
        let push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(bindings.constants as u32);
        let mut layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        if bindings.constants > 0 {
            layout_info = layout_info.push_constant_ranges(std::slice::from_ref(&push));
        }
        let layout = self
            .device
            .create_pipeline_layout(&layout_info)
            .map_err(|e| format!("post pipeline layout: {e}"))?;

        // A pipeline is created against a render pass but is compatible with any
        // pass of the same attachment shape, so the cached one for this format
        // serves both the build and every draw.
        let render_pass = self
            .cache
            .render_pass(self.device, format, PostLoadOp::DontCare)?;
        let pipeline = self.build_pipeline(program, render_pass, layout.handle(), blend)?;
        Ok(PostPipeline {
            pipeline,
            layout,
            set_layout,
            textures: bindings.textures,
            constants: bindings.constants,
        })
    }

    fn create_target(
        &self,
        label: &'static str,
        desc: &TextureDesc,
        extent: PostExtent,
    ) -> Result<Self::Target, String> {
        let spec = resolved_texture(label, desc, extent);
        let pooled = create_image(
            self.alloc,
            &ImageSpec {
                width: spec.width,
                height: spec.height,
                format: image_format(spec.format),
                tiling: vk::ImageTiling::OPTIMAL,
                usage: image_usage(spec.usage),
                mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                samples: sample_count(spec.sample_count),
            },
        )
        .map_err(|e| format!("{label} post target: {e}"))?;
        let image = pooled.image();
        // Pre-transitioned so the first frame can sample a slot before anything
        // has rendered into it: a temporal pass binds its history on the very
        // first draw, gated to ignore what it reads.
        one_shot_submit(
            self.device,
            self.queue.command_pool,
            self.queue.queue,
            |cmd| {
                transition_image_layout(
                    self.device,
                    cmd,
                    image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageAspectFlags::COLOR,
                );
            },
        )?;
        let view = create_image_view(
            self.device,
            image,
            image_format(spec.format),
            vk::ImageAspectFlags::COLOR,
        )?;
        Ok(PostTarget {
            image: GpuImage::from_pooled(pooled, view),
            extent: vk::Extent2D {
                width: spec.width,
                height: spec.height,
            },
            format: spec.format,
        })
    }

    fn target_ref<'a>(&self, target: &'a Self::Target) -> Self::TextureRef<'a> {
        target.image.view
    }

    fn encode(&self, rec: &Self::Recorder, draw: &PostDraw<'_, '_, Self>) -> Result<(), String> {
        let cmd = *rec;
        let pipe = draw.pipeline;
        if draw.binds.len() != pipe.textures || draw.constants.len() != pipe.constants {
            return Err(format!(
                "{}: the draw binds {} texture(s) and {} constant byte(s) where the program \
                 declares {} and {}",
                draw.label,
                draw.binds.len(),
                draw.constants.len(),
                pipe.textures,
                pipe.constants,
            ));
        }
        let render_pass = self
            .cache
            .render_pass(self.device, draw.target.format, draw.load)?;
        let framebuffer = self.cache.framebuffer(
            self.device,
            render_pass,
            draw.target.image.view,
            draw.target.extent,
        )?;

        // One set per draw from this frame's arena, written from what the pass
        // holds now. Nothing here is cached across frames, so no input needs a
        // re-point when another effect takes over the scene image.
        let set = self
            .arena
            .alloc(self.device, self.frame, pipe.set_layout.handle())?;
        let infos: Vec<vk::DescriptorImageInfo> = draw
            .binds
            .iter()
            .map(|b| {
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(b.texture)
                    .sampler(match b.sampler {
                        PostSampler::LinearClamp => self.sampler,
                    })
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(info))
            })
            .collect();
        // SAFETY: `writes` and the image infos it borrows are live for the call, and every set and
        // resource it names belongs to this device.
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        let extent = draw.target.extent;
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D::default().extent(extent));
        let vp = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(extent);
        let device = self.device;
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipe.pipeline.handle());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipe.layout.handle(),
                0,
                std::slice::from_ref(&set),
                &[],
            );
            if !draw.constants.is_empty() {
                device.cmd_push_constants(
                    cmd,
                    pipe.layout.handle(),
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    draw.constants,
                );
            }
            // The vertex stage builds the fullscreen triangle from the vertex id,
            // so the draw reads no vertex buffer.
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_render_pass(cmd);
        }
        Ok(())
    }
}

impl crate::vulkan::context::VkContext {
    // The post-pass device over this context, recording into frame slot `frame`.
    pub(in crate::vulkan) fn post_device(&self, frame: usize) -> VkPostDevice<'_> {
        VkPostDevice {
            device: &self.device,
            alloc: &self.alloc,
            queue: PostQueue {
                command_pool: self.commands.command_pool,
                queue: self.graphics_queue,
            },
            cache: &self.post.cache,
            arena: &self.post.arena,
            sampler: self.composite.sampler.handle(),
            frame,
            hot_reload: self.hot_reload.enabled,
        }
    }
}
