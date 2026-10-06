//! The sky's motion in the G-buffer pre-pass: a fullscreen triangle behind the
//! geometry that writes the camera's rotation into the velocity target. It
//! reads only the pre-pass's view block, through its own one-binding set per
//! frame, so it draws whether or not the world has cull records.

use ash::vk;
use concinnity_core::render::error::RenderResult;

use super::gbuffer::{GBUFFER_VIEW_UBO_SIZE, PREPASS_TARGETS};
use crate::vulkan::allocator::PooledBuffer;
use crate::vulkan::builtin_shaders::{self, CompileProgram};
use crate::vulkan::descriptor_layout::{Binding, PoolSizes};
use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedPipeline, OwnedPipelineLayout, OwnedSetLayout, VkDevice,
};
use crate::vulkan::pipeline_desc::{Depth, GraphicsPipelineDesc};
use crate::vulkan::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use crate::vulkan::set_writes::SetWrites;

const VIEW_SET: [Binding; 1] = [(
    0,
    vk::DescriptorType::UNIFORM_BUFFER,
    vk::ShaderStageFlags::VERTEX,
)];

pub(in crate::vulkan) struct GbufferSky {
    // Owners only: the sets are written once at build.
    _pool: OwnedDescriptorPool,
    _set_layout: OwnedSetLayout,
    layout: OwnedPipelineLayout,
    pipeline: OwnedPipeline,
    sets: Vec<vk::DescriptorSet>,
}

impl GbufferSky {
    // One set per frame over that frame's view UBO, and the pipeline against
    // the pre-pass render pass.
    pub(in crate::vulkan) fn build(
        device: &VkDevice,
        render_pass: vk::RenderPass,
        view_ubos: &[PooledBuffer],
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let frames = view_ubos.len() as u32;
        let sizes = PoolSizes::default().sets(&VIEW_SET, frames).build();
        let pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(frames)
                    .pool_sizes(&sizes),
            )
            .map_err(|e| crate::vulkan::error::map_vk_result(e, "gbuffer sky descriptor pool"))?;
        let set_layout = create_descriptor_set_layout(device, &VIEW_SET)?;
        let layouts = [set_layout.handle()];
        let layout = device
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts))
            .map_err(|e| crate::vulkan::error::map_vk_result(e, "gbuffer sky pipeline layout"))?;
        let set_layouts = vec![set_layout.handle(); view_ubos.len()];
        let sets = alloc_descriptor_sets(device, pool.handle(), &set_layouts)?;
        for (&set, ubo) in sets.iter().zip(view_ubos) {
            SetWrites::new(set)
                .uniform_buffer(0, ubo.buffer(), GBUFFER_VIEW_UBO_SIZE)
                .apply(device);
        }
        let pipeline = Self::build_pipeline(device, layout.handle(), render_pass, hot_reload)?;
        Ok(Self {
            _pool: pool,
            _set_layout: set_layout,
            layout,
            pipeline,
            sets,
        })
    }

    // Behind the geometry: the inclusive read-only depth test passes only
    // where the pre-pass depth still holds the clear.
    fn build_pipeline(
        device: &VkDevice,
        layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
        hot_reload: bool,
    ) -> RenderResult<OwnedPipeline> {
        let vs = builtin_shaders::GBUFFER_SKY_VERT.compile(hot_reload)?;
        let fs = builtin_shaders::GBUFFER_PREPASS_FRAG_BINDLESS.compile(hot_reload)?;
        GraphicsPipelineDesc {
            depth: Depth::read_only(),
            ..GraphicsPipelineDesc::fullscreen(&vs, &fs, layout, render_pass, &PREPASS_TARGETS)
        }
        .build(device, "gbuffer sky")
    }

    // A pipeline rebuilt from the current shader sources, for a hot reload.
    pub(in crate::vulkan) fn rebuild_pipeline(
        &self,
        device: &VkDevice,
        render_pass: vk::RenderPass,
    ) -> RenderResult<OwnedPipeline> {
        Self::build_pipeline(device, self.layout.handle(), render_pass, true)
    }

    pub(in crate::vulkan) fn swap_pipeline(&mut self, pipeline: OwnedPipeline) {
        self.pipeline = pipeline;
    }

    // Draw the sky's motion inside the open pre-pass render pass.
    pub(in crate::vulkan) fn encode(
        &self,
        device: &VkDevice,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
    ) {
        let Some(&set) = self.sets.get(frame_idx) else {
            return;
        };
        // SAFETY: `cmd` is a command buffer in the recording state inside the pre-pass render pass
        // the pipeline was built against, and every handle these commands name is live for the
        // call.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline.handle());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout.handle(),
                0,
                &[set],
                &[],
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
        }
    }
}
