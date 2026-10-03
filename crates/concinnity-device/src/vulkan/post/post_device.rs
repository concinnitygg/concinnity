//! Vulkan's implementation of the shared fullscreen post-pass seam
//! (`render::post::device::PostPassDevice`).
//!
//! Each pass's objects are derived here from what the single source declares:
//! the descriptor set layout is N sampled images and their N samplers, the
//! pipeline layout adds a fragment push-constant range of the declared size and,
//! for a probe-reading program, the forward global set as set 1, and the render
//! pass comes from the target's format and load action. The sets are allocated
//! per frame (post/set_arena.rs) rather than pre-wired per effect, so no pass
//! has to rewire an input another effect owns.

use ash::vk;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::post::device::{
    PostBlend, PostDraw, PostExtent, PostLoadOp, PostPassDevice, PostSampler, check_level,
    level_extent, resolved_texture,
};
use concinnity_core::render::post::program::{PostProgram, PostProgramBindings};
use concinnity_core::render::render_graph::{PixelFormat, TextureDesc};

use crate::vulkan::allocator::DeviceAllocator;
use crate::vulkan::builtin_shaders::{self, CompileProgram};
use crate::vulkan::error::map_vk_result;
use crate::vulkan::owned::{OwnedPipeline, OwnedPipelineLayout, OwnedSetLayout, VkDevice};
use crate::vulkan::pipeline_desc::{Blend, GraphicsPipelineDesc};
use crate::vulkan::post::pass_cache::PostPassCache;
use crate::vulkan::post::set_arena::PostSetArena;
use crate::vulkan::set_writes::SetWrites;
use crate::vulkan::texture::{
    GpuImage, LayoutTransition, SubresourceRange, one_shot_submit, transition_image_layout_range,
};
use crate::vulkan::transient_pool::{image_format, image_usage, sample_count};

// The set index a probe-reading program declares the forward global set at.
const PROBE_SET_INDEX: u32 = 1;

// A built fullscreen post pipeline plus the layouts a draw binds it through.
// The two layouts travel with the pipeline because they are derived from the
// same program declaration, so nothing else has to know the binding count.
pub(in crate::vulkan) struct PostPipeline {
    pipeline: OwnedPipeline,
    layout: OwnedPipelineLayout,
    set_layout: OwnedSetLayout,
    // What the program declares, so a draw can check what it was handed.
    bindings: PostProgramBindings,
}

// A persistent post target: the image, its view of every mip level, a view of
// each level alone when it has more than one, and the extent of its top level.
pub(in crate::vulkan) struct PostTarget {
    label: &'static str,
    image: GpuImage,
    extent: vk::Extent2D,
    format: PixelFormat,
    levels: u32,
}

impl PostTarget {
    /// The sampled view of this target, for a consumer that binds it directly.
    pub(in crate::vulkan) fn view(&self) -> vk::ImageView {
        self.image.view
    }

    // Mip `level` alone: its own view when the target has several, else the
    // one view.
    fn level_view(&self, level: u32) -> RenderResult<vk::ImageView> {
        check_level(self.label, level, self.levels)?;
        Ok(self
            .image
            .aux_views
            .get(level as usize)
            .copied()
            .unwrap_or(self.image.view))
    }
}

// A 2D color view of `count` mip levels of `image` from `base`.
fn level_range_view(
    device: &VkDevice,
    image: vk::Image,
    format: vk::Format,
    base: u32,
    count: u32,
) -> RenderResult<vk::ImageView> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(base)
                .level_count(count)
                .base_array_layer(0)
                .layer_count(1),
        );
    // SAFETY: the create-info is live for the call and names an image of this
    // device holding at least `base + count` levels.
    unsafe { device.create_image_view(&info, None) }
        .map_err(|e| map_vk_result(e, "post target view"))
}

// A draw's color target: the view a framebuffer binds, the extent it is sized
// at, and the format its render pass is built for. A created target names
// itself this way through `target_attachment`; an image another subsystem owns
// (the HDR scene) is described directly.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkAttachment {
    pub view: vk::ImageView,
    pub extent: vk::Extent2D,
    pub format: PixelFormat,
}

// The forward global set, as a probe-reading program binds it for the probe
// count (binding 7), the cube array (binding 8) and the records (binding 17).
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkPostProbes<'a> {
    pub layout: vk::DescriptorSetLayout,
    // One set per frame in flight. Empty at init, where no draw is encoded.
    pub sets: &'a [vk::DescriptorSet],
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
    // The trilinear clamp-to-edge state an environment cube is sampled through.
    pub cube_sampler: vk::Sampler,
    // The global set a probe-reading program binds.
    pub probes: VkPostProbes<'a>,
    // Which frame in flight is recording.
    pub frame: usize,
    pub hot_reload: bool,
}

// The SPIR-V a post program's two stages compile to: the one shared fullscreen
// triangle vertex plus the program's own fragment.
fn compile(program: PostProgram, hot_reload: bool) -> RenderResult<(Vec<u8>, Vec<u8>)> {
    let vert = builtin_shaders::FULLSCREEN_VERT.compile(hot_reload)?;
    Ok((vert, program.program().compile(hot_reload)?))
}

impl VkPostDevice<'_> {
    // The descriptor set layout for `n` sources, fragment-visible: their
    // sampled images at bindings `0..n` and their samplers at `n..2n`. Derived
    // from the program's declared count rather than written per pass, which is
    // what keeps the single source the contract.
    fn set_layout(&self, n: usize) -> RenderResult<OwnedSetLayout> {
        crate::vulkan::resources::create_descriptor_set_layout(
            self.device,
            &crate::vulkan::resources::source_set_bindings(n as u32),
        )
    }

    // The probe set a program that declares one binds.
    fn probes_for(&self, bindings: PostProgramBindings) -> Option<VkPostProbes<'_>> {
        bindings.probes.then_some(self.probes)
    }

    fn sampler_for(&self, sampler: PostSampler) -> vk::Sampler {
        match sampler {
            PostSampler::LinearClamp => self.sampler,
            PostSampler::LinearCube => self.cube_sampler,
        }
    }
}

impl PostPassDevice for VkPostDevice<'_> {
    type Recorder = vk::CommandBuffer;
    type Pipeline = PostPipeline;
    type Target = PostTarget;
    type TextureRef<'a> = vk::ImageView;
    type Attachment<'a> = VkAttachment;

    fn create_pipeline(
        &self,
        program: PostProgram,
        format: PixelFormat,
        blend: PostBlend,
    ) -> RenderResult<Self::Pipeline> {
        let bindings = program.bindings();
        let probes = self.probes_for(bindings);
        let set_layout = self.set_layout(bindings.textures)?;
        let mut set_layouts = vec![set_layout.handle()];
        set_layouts.extend(probes.map(|p| p.layout));
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
            .map_err(|e| map_vk_result(e, "post pipeline layout"))?;

        // A pipeline is created against a render pass but is compatible with any
        // pass of the same attachment shape, so the cached one for this format
        // serves both the build and every draw.
        let render_pass = self
            .cache
            .render_pass(self.device, format, PostLoadOp::DontCare)?;
        // Every fullscreen post pass has the same pipeline shape, which is why it
        // is built once here instead of once per effect.
        let (vert, frag) = compile(program, self.hot_reload)?;
        let pipeline = GraphicsPipelineDesc::fullscreen(
            &vert,
            &frag,
            layout.handle(),
            render_pass,
            &[Blend::from(blend)],
        )
        .build(self.device, "post")?;
        Ok(PostPipeline {
            pipeline,
            layout,
            set_layout,
            bindings,
        })
    }

    fn create_target(
        &self,
        label: &'static str,
        desc: &TextureDesc,
        extent: PostExtent,
    ) -> RenderResult<Self::Target> {
        let spec = resolved_texture(label, desc, extent);
        let format = image_format(spec.format);
        let levels = spec.mip_levels.max(1);
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: spec.width,
                height: spec.height,
                depth: 1,
            })
            .mip_levels(levels)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(image_usage(spec.usage))
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(sample_count(spec.sample_count));
        let pooled = self
            .alloc
            .create_image(&info, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .map_err(|e| e.context(format_args!("{label} post target")))?;
        let image = pooled.image();
        // Pre-transitioned so the first frame can sample a slot before anything
        // has rendered into it: a temporal pass binds its history on the very
        // first draw, gated to ignore what it reads. Every level rests readable,
        // and a draw's render pass moves the one level it writes in and out.
        one_shot_submit(
            self.device,
            self.queue.command_pool,
            self.queue.queue,
            |cmd| {
                transition_image_layout_range(
                    self.device,
                    cmd,
                    image,
                    LayoutTransition {
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        aspect: vk::ImageAspectFlags::COLOR,
                    },
                    SubresourceRange {
                        base_layer: 0,
                        layer_count: 1,
                        base_mip: 0,
                        mip_count: levels,
                    },
                );
            },
        )?;
        let view = level_range_view(self.device, image, format, 0, levels)?;
        let mut gpu = GpuImage::from_pooled(pooled, view);
        if levels > 1 {
            for level in 0..levels {
                gpu.push_aux_view(level_range_view(self.device, image, format, level, 1)?);
            }
        }
        Ok(PostTarget {
            label,
            image: gpu,
            extent: vk::Extent2D {
                width: spec.width,
                height: spec.height,
            },
            format: spec.format,
            levels,
        })
    }

    fn target_ref<'a>(&self, target: &'a Self::Target) -> Self::TextureRef<'a> {
        target.image.view
    }

    fn target_attachment<'a>(&self, target: &'a Self::Target) -> Self::Attachment<'a> {
        VkAttachment {
            // The top level alone: a framebuffer attachment is one level.
            view: target
                .image
                .aux_views
                .first()
                .copied()
                .unwrap_or(target.image.view),
            extent: target.extent,
            format: target.format,
        }
    }

    fn target_level_ref<'a>(
        &self,
        target: &'a Self::Target,
        level: u32,
    ) -> RenderResult<Self::TextureRef<'a>> {
        target.level_view(level)
    }

    fn target_level_attachment<'a>(
        &self,
        target: &'a Self::Target,
        level: u32,
    ) -> RenderResult<Self::Attachment<'a>> {
        let extent = level_extent(post_extent(target.extent), level);
        Ok(VkAttachment {
            view: target.level_view(level)?,
            extent: vk::Extent2D {
                width: extent.width,
                height: extent.height,
            },
            format: target.format,
        })
    }

    fn encode(&self, rec: &Self::Recorder, draw: &PostDraw<'_, '_, Self>) -> RenderResult<()> {
        let cmd = *rec;
        let pipe = draw.pipeline;
        draw.check(pipe.bindings)?;
        // A render pass writes its attachment's layout in and out itself, so
        // who owns the target's state between passes changes nothing here.
        let probe_set = match self.probes_for(pipe.bindings) {
            None => None,
            Some(probes) => Some(*probes.sets.get(self.frame).ok_or_else(|| {
                RenderError::Other(format!(
                    "{}: no global set for frame {}",
                    draw.label, self.frame
                ))
            })?),
        };
        let target = draw.target;
        let render_pass = self
            .cache
            .render_pass(self.device, target.format, draw.load)?;
        let framebuffer =
            self.cache
                .framebuffer(self.device, render_pass, target.view, target.extent)?;

        // One set per draw from this frame's arena, written from what the pass
        // holds now. Nothing here is cached across frames, so no input needs a
        // re-point when another effect takes over the scene image.
        let set = self
            .arena
            .alloc(self.device, self.frame, pipe.set_layout.handle())?;
        let n = draw.binds.len() as u32;
        draw.binds
            .iter()
            .zip(0..)
            .fold(SetWrites::new(set), |w, (b, i)| {
                w.sampled_image(i, b.texture)
                    .sampler(n + i, self.sampler_for(b.sampler))
            })
            .apply(self.device);

        let extent = target.extent;
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
            if let Some(global) = probe_set {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipe.layout.handle(),
                    PROBE_SET_INDEX,
                    std::slice::from_ref(&global),
                    &[],
                );
            }
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

// A post-pass extent from a Vulkan one.
pub(in crate::vulkan) fn post_extent(extent: vk::Extent2D) -> PostExtent {
    PostExtent {
        width: extent.width,
        height: extent.height,
    }
}

impl crate::vulkan::context::VkContext {
    // The post-pass device over this context, recording into frame slot `frame`.
    pub(in crate::vulkan) fn post_device(&self, frame: usize) -> VkPostDevice<'_> {
        VkPostDevice {
            device: &self.hw.device,
            alloc: &self.hw.alloc,
            queue: PostQueue {
                command_pool: self.commands.command_pool,
                queue: self.hw.graphics_queue,
            },
            cache: &self.post.cache,
            arena: &self.post.arena,
            sampler: self.post.sampler.handle(),
            cube_sampler: self.scene.cube_sampler.handle(),
            probes: VkPostProbes {
                layout: self.descriptors.global_set_layout.handle(),
                sets: &self.descriptors.global_sets,
            },
            frame,
            hot_reload: self.hot_reload.enabled,
        }
    }

    // This frame slot's HDR scene, as a post draw's target.
    pub(in crate::vulkan) fn hdr_scene_attachment(&self, frame: usize) -> VkAttachment {
        VkAttachment {
            view: self.targets.hdr_resolve_images[frame % self.targets.hdr_resolve_images.len()]
                .view,
            extent: self.targets.render_extent,
            format: PixelFormat::Rgba16Float,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every post program's fragment, and the shared vertex, compile to SPIR-V.
    #[test]
    fn every_post_program_compiles() {
        concinnity_shader::require_dxc!();
        for program in PostProgram::ALL {
            let (vert, frag) =
                compile(program, false).unwrap_or_else(|e| panic!("{program:?}: {e}"));
            assert!(crate::vulkan::pipeline::is_spirv(&vert));
            assert!(crate::vulkan::pipeline::is_spirv(&frag));
        }
    }
}
