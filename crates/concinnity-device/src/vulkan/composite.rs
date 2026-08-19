// src/vulkan/composite.rs
//
// Composite (post-process) pass + text overlay. The post-process pipeline
// reads the post-stack scene texture (TAA output > SSR output > HDR resolve,
// wired to `composite_sets` at init / on resize), the bloom mip-0 target, and
// the 3D colour-grading LUT, then writes ACES tonemap + gamma + FXAA into the
// swapchain image. Text is drawn after in the same render pass so it sits on
// top of the tonemapped image in display-referred LDR space.
//
// The shape mirrors `metal/draw/composite.rs::encode_composite_and_text`;
// the graph executor in [`graph_exec.rs`](graph_exec.rs) dispatches
// `PassId::Composite` here.

use ash::vk;

use crate::gfx::render_types::{CompositeParams, TextDrawCall, TextVertex};
use concinnity_core::gfx::render_types::TextUniforms;

use super::context::VkContext;

// Per-invocation binding context for the composite pass. `pub` because it is the
// `Args` associated type of the (cross-crate) `render::fullscreen::CompositeEncoder`
// impl below, so it cannot be more private than that public trait's interface.
pub struct VkCompositeArgs {
    image_index: usize,
    frame_idx: usize,
}

// The composite + text orchestration lives once in `gfx::fullscreen`; this impl
// drives each step in Vulkan. The composite pipeline samples the post-stack scene
// texture via `composite_sets[frame_idx]` (wired at init / on resize) and writes
// the ACES + gamma + FXAA tonemap into `composite_framebuffers[image_index]`;
// text is drawn after in the same render pass so it sits on top in LDR space.
impl crate::gfx::fullscreen::CompositeEncoder for VkContext {
    type Rec = vk::CommandBuffer;
    type Args = VkCompositeArgs;

    fn begin_composite(&self, cmd: &Self::Rec, args: &Self::Args) {
        let device = &self.device;
        let extent = self.swapchain_extent;
        let composite_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.composite_render_pass)
            .framebuffer(self.composite_framebuffers[args.image_index])
            .render_area(vk::Rect2D::default().extent(extent));
        // The composite pass uses a standard positive-height viewport: the HDR
        // image is already upright, so it is a plain copy + post.
        let composite_vp = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(extent);
        unsafe {
            device.cmd_begin_render_pass(*cmd, &composite_begin, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(*cmd, 0, std::slice::from_ref(&composite_vp));
            device.cmd_set_scissor(*cmd, 0, std::slice::from_ref(&scissor));
        }
    }

    fn composite_draw(&self, cmd: &Self::Rec, args: &Self::Args) {
        let device = &self.device;
        unsafe {
            device.cmd_bind_pipeline(
                *cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.composite_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.composite_pipeline_layout,
                0,
                std::slice::from_ref(&self.composite_sets[args.frame_idx]),
                &[],
            );
            // Post-process tunables (bloom intensity, exposure, vignette) plus
            // the scene-transition fade.
            // Non-zero only for the G-buffer channel views, which take the
            // fragment's visualization branch; Lit / Unlit / Wireframe all
            // composite the scene.
            let composite = CompositeParams {
                post: self.post_process,
                fade: self.scene_fade,
                view_mode: if self.view_mode.is_gbuffer_channel() {
                    self.view_mode as u32
                } else {
                    0
                },
                far: self.view_far,
            };
            device.cmd_push_constants(
                *cmd,
                self.composite_pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                std::slice::from_raw_parts(
                    &composite as *const CompositeParams as *const u8,
                    std::mem::size_of::<CompositeParams>(),
                ),
            );
            // Fullscreen triangle: three vertices, no vertex buffer.
            device.cmd_draw(*cmd, 3, 1, 0, 0);
        }
        self.inc_draw_calls(1);
    }

    fn begin_text(&self, cmd: &Self::Rec, _args: &Self::Args) -> bool {
        let Some(text_pipeline) = self.text_pipeline else {
            return false;
        };
        if self.text_atlas_textures.is_empty() {
            return false;
        }
        unsafe {
            self.device
                .cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::GRAPHICS, text_pipeline);
        }
        true
    }

    fn text_draw(
        &self,
        cmd: &Self::Rec,
        _args: &Self::Args,
        call: &TextDrawCall,
    ) -> Result<(), String> {
        if call.vertices.is_empty() || self.descriptors.text_atlas_sets.is_empty() {
            return Ok(());
        }
        let device = &self.device;
        let extent = self.swapchain_extent;
        // The text vertices are in overlay units (mapped to NDC by the shader's
        // divide by win_width/height); the scissor is in attachment pixels, so a
        // per-call clip rect scales between the two.
        let ui = self.logical_size();

        // Scissor a clipped (scrollable-panel) call to its band, restoring the
        // full-window scissor for an unclipped call so chrome is never cropped.
        // Resolved first so a fully-scrolled-out row skips before allocating its
        // transient buffers.
        let scissor = match call.clip_rect {
            Some(clip) => {
                match crate::gfx::fullscreen::clip_rect_to_scissor(
                    clip,
                    ui,
                    (extent.width, extent.height),
                ) {
                    None => return Ok(()),
                    Some((x, y, w, h)) => vk::Rect2D {
                        offset: vk::Offset2D { x, y },
                        extent: vk::Extent2D {
                            width: w,
                            height: h,
                        },
                    },
                }
            }
            None => vk::Rect2D::default().extent(extent),
        };

        let text_push = TextUniforms {
            win_width: ui.0,
            win_height: ui.1,
            _pad: [0.0; 2],
        };
        let atlas_idx = call
            .atlas_slot
            .min(self.descriptors.text_atlas_sets.len() - 1);

        // Transient vertex + index buffers for this text call.
        let vert_size = (call.vertices.len() * std::mem::size_of::<TextVertex>()) as u64;
        let idx_size = (call.indices.len() * std::mem::size_of::<u16>()) as u64;

        let tvbuf = self
            .alloc
            .create_buffer(
                vert_size,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .map_err(|e| format!("text vtx buf: {e}"))?;
        let tibuf = self
            .alloc
            .create_buffer(
                idx_size,
                vk::BufferUsageFlags::INDEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .map_err(|e| format!("text idx buf: {e}"))?;

        unsafe {
            std::ptr::copy_nonoverlapping(
                call.vertices.as_ptr() as *const u8,
                tvbuf.mapped_ptr(),
                vert_size as usize,
            );
            std::ptr::copy_nonoverlapping(
                call.indices.as_ptr() as *const u8,
                tibuf.mapped_ptr(),
                idx_size as usize,
            );

            device.cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.text_pipeline_layout,
                0,
                std::slice::from_ref(&self.descriptors.text_atlas_sets[atlas_idx]),
                &[],
            );
            device.cmd_push_constants(
                *cmd,
                self.text_pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                std::slice::from_raw_parts(
                    &text_push as *const TextUniforms as *const u8,
                    std::mem::size_of::<TextUniforms>(),
                ),
            );
            device.cmd_set_scissor(*cmd, 0, std::slice::from_ref(&scissor));
            device.cmd_bind_vertex_buffers(*cmd, 0, &[tvbuf.buffer()], &[0]);
            device.cmd_bind_index_buffer(*cmd, tibuf.buffer(), 0, vk::IndexType::UINT16);
            device.cmd_draw_indexed(*cmd, call.indices.len() as u32, 1, 0, 0, 0);
        }
        self.inc_draw_calls(1);

        // Dropping the buffers here is safe: the allocator withholds their
        // ranges until every in-flight frame has retired.
        Ok(())
    }

    fn end_composite(&self, cmd: &Self::Rec, _args: &Self::Args) {
        unsafe { self.device.cmd_end_render_pass(*cmd) };
    }
}

impl VkContext {
    // Encode the composite tonemap pass and text overlay for frame slot
    // `frame_idx`, targeting the swapchain image at `image_index`, via the shared
    // `gfx::fullscreen` driver.
    pub(in crate::vulkan) fn encode_composite_and_text(
        &self,
        cmd: vk::CommandBuffer,
        image_index: u32,
        frame_idx: usize,
        text_calls: &[TextDrawCall],
    ) -> Result<(), String> {
        let args = VkCompositeArgs {
            image_index: image_index as usize,
            frame_idx,
        };
        crate::gfx::fullscreen::encode_composite_chain(self, &cmd, &args, text_calls)
    }
}
