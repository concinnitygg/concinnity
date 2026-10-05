// The environment drawn as the background (see `concinnity_core::render::sky`):
// a fullscreen draw at the tail of each opaque scene pass (the main camera, a
// reflection-probe face, a planar mirror), inside that pass's own render pass so
// the sky costs no extra load or store of its targets, and one at the tail of
// the G-buffer pre-pass for the sky's motion (`post/gbuffer_sky.rs`).
//
// The sky reads set 0 of the pass it completes: the view block, the prefilter
// cube and the cube sampler of the global set the main pass, a probe face or a
// mirror already bound for its own viewpoint.

use ash::vk;
use concinnity_core::gfx::view_modes::ViewMode;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::pass_timing;
use concinnity_core::render::render_graph::PassId;
use concinnity_core::render::sky;

use super::builtin_shaders::CompileProgram;
use super::context::VkContext;
use super::owned::{OwnedPipeline, OwnedPipelineLayout, VkDevice};
use super::pipeline_desc::{Blend, Depth, GraphicsPipelineDesc};

pub(in crate::vulkan) struct VkSky {
    // Set 0 only: the global set layout every scene pass binds.
    layout: OwnedPipelineLayout,
    pipeline: OwnedPipeline,
    // The world draws its environment map as the background.
    background: bool,
}

impl VkSky {
    pub(in crate::vulkan) fn build(
        device: &VkDevice,
        global_set_layout: vk::DescriptorSetLayout,
        main_render_pass: vk::RenderPass,
        samples: vk::SampleCountFlags,
        background: bool,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let layouts = [global_set_layout];
        let layout = device
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts))
            .map_err(|e| super::error::map_vk_result(e, "sky pipeline layout"))?;
        let pipeline = build_sky_pipeline(
            device,
            layout.handle(),
            main_render_pass,
            samples,
            hot_reload,
        )?;
        Ok(Self {
            layout,
            pipeline,
            background,
        })
    }

    pub(in crate::vulkan) fn layout(&self) -> vk::PipelineLayout {
        self.layout.handle()
    }

    pub(in crate::vulkan) fn swap_pipeline(&mut self, pipeline: OwnedPipeline) {
        self.pipeline = pipeline;
    }
}

// The sky over the HDR color and the main depth, at the main pass's sample
// count: no blending, no vertex input, and the inclusive read-only depth test
// that passes only where the depth still holds the clear.
pub(in crate::vulkan) fn build_sky_pipeline(
    device: &VkDevice,
    layout: vk::PipelineLayout,
    main_render_pass: vk::RenderPass,
    samples: vk::SampleCountFlags,
    hot_reload: bool,
) -> RenderResult<OwnedPipeline> {
    let vs = super::builtin_shaders::SKY_VERT.compile(hot_reload)?;
    let fs = super::builtin_shaders::SKY_FRAG.compile(hot_reload)?;
    GraphicsPipelineDesc {
        depth: Depth::camera_read_only(),
        samples,
        ..GraphicsPipelineDesc::fullscreen(&vs, &fs, layout, main_render_pass, &[Blend::Opaque])
    }
    .build(device, "sky")
}

impl VkContext {
    // Whether a view rendered in `mode` draws the sky.
    pub(in crate::vulkan) fn draws_sky(&self, mode: ViewMode) -> bool {
        sky::draws_sky(
            self.scene.prefilter_mip_count > 0,
            self.sky.background,
            mode,
        )
    }

    // Draw the environment behind everything drawn so far in the open render
    // pass, which must be `main_render_pass`-compatible, from the viewpoint
    // `global_set` describes.
    pub(in crate::vulkan) fn encode_sky(
        &self,
        cmd: vk::CommandBuffer,
        global_set: vk::DescriptorSet,
    ) {
        let device = &self.hw.device;
        // SAFETY: `cmd` is a command buffer in the recording state inside a render pass the sky
        // pipeline is compatible with, and every handle these commands name is live for the call.
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.sky.pipeline.handle(),
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.sky.layout(),
                0,
                &[global_set],
                &[],
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
        }
        self.inc_draw_calls(1);
    }

    // The main camera's sky, bracketed by its own timestamps inside the main
    // pass. The draw is the last in the pass, so the bottom-of-pipe stamp
    // before it lands when the opaque geometry finishes.
    pub(in crate::vulkan) fn encode_main_sky(&self, cmd: vk::CommandBuffer, frame_idx: usize) {
        let device = &self.hw.device;
        let stamps = self
            .hw
            .timestamp_query_pool
            .map(|pool| (pool, pass_timing::pass_pair(frame_idx, PassId::Sky)));
        if let Some((pool, (start, _))) = stamps {
            // SAFETY: `cmd` is a command buffer in the recording state and the query pool is live;
            // the frame's query block was reset at the frame's start.
            unsafe {
                device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, start)
            };
        }
        self.encode_sky(cmd, self.descriptors.global_sets[frame_idx]);
        if let Some((pool, (_, end))) = stamps {
            // SAFETY: as above.
            unsafe {
                device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, end)
            };
        }
    }
}
