//! Composite (post-process) pass + text overlay. The post-process pipeline
//! reads the post-stack scene texture (TAA output > SSR output > HDR resolve,
//! wired to `composite.sets` at init / on resize), the bloom top octave, and
//! the 3D color-grading LUT, then writes ACES tonemap + gamma + FXAA into the
//! swapchain image. Text is drawn after in the same render pass so it sits on
//! top of the tonemapped image in display-referred LDR space.
//!
//! The shape mirrors `metal/draw/composite.rs::encode_composite_and_text`;
//! the graph executor in [`graph_exec.rs`](graph_exec.rs) dispatches
//! `PassId::Composite` here. Text geometry is appended into this frame slot's
//! persistent upload buffer (see [`super::super::upload_ring::UploadRing`]) and drawn
//! from sub-ranges of it, so no GPU buffer is created per label per frame.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::auto_exposure;
use concinnity_engine::gfx::render_types::{CompositeParams, TextDrawCall, TextUniforms};
use concinnity_engine::render::backend::WindowControl;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::fullscreen;
use concinnity_engine::render::fullscreen::TextBindCache;

use super::super::context::VkContext;
use super::super::record::Recorder;
use super::super::upload_ring::UPLOAD_ALIGN;

// Per-invocation binding context for the composite pass.
pub(in crate::vulkan) struct CompositeArgs {
    image_index: usize,
    frame_idx: usize,
    // Whether a particle or transparent pass wrote the reactive mask this frame.
    reactive_valid: bool,
}

// The context, recording the composite through a `Recorder` borrowed from it.
pub(in crate::vulkan) struct CompositePass<'a> {
    ctx: &'a VkContext,
}

impl core::ops::Deref for CompositePass<'_> {
    type Target = VkContext;

    fn deref(&self) -> &VkContext {
        self.ctx
    }
}

// The composite + text orchestration lives once in `gfx::fullscreen`; this impl
// drives each step in Vulkan. The composite pipeline samples the post-stack scene
// texture via `composite.sets[frame_idx]` (wired at init / on resize) and writes
// the ACES + gamma + FXAA tonemap into `composite.framebuffers[image_index]`;
// text is drawn after in the same render pass so it sits on top in LDR space.
impl<'a> fullscreen::CompositeEncoder for CompositePass<'a> {
    type Rec = Recorder<'a>;
    type Args = CompositeArgs;

    fn begin_composite(&self, rec: &Self::Rec, args: &Self::Args) {
        let extent = self.swapchain.extent;
        let area = vk::Rect2D::default().extent(extent);
        rec.begin_render_pass(
            &self.composite.render_pass,
            &self.composite.framebuffers[args.image_index],
            area,
            &[],
        );
        // The composite pass uses a standard positive-height viewport: the HDR
        // image is already upright, so it is a plain copy + post.
        rec.set_viewport(&vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        });
        rec.set_scissor(&area);
    }

    fn composite_draw(&self, rec: &Self::Rec, args: &Self::Args) {
        rec.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, &self.composite.pipeline);
        rec.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            &self.composite.pipeline_layout,
            0,
            std::slice::from_ref(&self.composite.sets[args.frame_idx]),
            &[],
        );
        // Post-process tunables (bloom intensity, exposure, vignette) plus
        // the scene-transition fade.
        // Non-zero only for the G-buffer channel views, which take the
        // fragment's visualization branch; Lit / Unlit / Wireframe all
        // composite the scene.
        let (depth_near, depth_far) =
            CompositeParams::depth_view_range(self.state.view.near, self.state.view.view_distance);
        let composite = CompositeParams {
            post: self.pre_exposure.post_params(&self.post_process),
            fade: self.state.view.scene_fade,
            view_mode: if self.state.view.mode.is_gbuffer_channel() {
                self.state.view.mode as u32
            } else {
                0
            },
            depth_near,
            depth_far,
            reactive_valid: if args.reactive_valid { 1.0 } else { 0.0 },
            night_vision: self
                .auto_exposure
                .adaptation
                .as_ref()
                .map_or(0.0, |a| a.night_vision(self.state.view.sunlight)),
            night_nits: auto_exposure::nits_per_exposed(
                self.state.view.sunlight,
                self.post_process.exposure,
            ),
        };
        rec.push_constants(
            &self.composite.pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            0,
            &composite,
        );
        // Fullscreen triangle: three vertices, no vertex buffer.
        rec.draw(3, 1, 0, 0);
        self.inc_draw_calls(1);
    }

    fn begin_text(&self, rec: &Self::Rec, _args: &Self::Args) -> bool {
        let Some(text_pipeline) = self.text.pipeline.as_ref() else {
            return false;
        };
        if self.text.atlas_textures.is_empty() {
            return false;
        }
        // The overlay size every call divides by: one value for the frame, so it
        // is pushed here rather than per label. The push follows the pipeline
        // bind, and the loop only ever binds descriptor sets with this same
        // layout, so nothing below invalidates it.
        let ui = self.logical_size();
        let text_push = TextUniforms {
            win_width: ui.0,
            win_height: ui.1,
            _pad: [0.0; 2],
        };
        rec.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, text_pipeline);
        rec.push_constants(
            &self.text.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            &text_push,
        );
        true
    }

    fn text_draw(
        &self,
        rec: &Self::Rec,
        args: &Self::Args,
        // This backend appends each call's geometry here, so its position in the
        // frame's list carries nothing it needs.
        _idx: usize,
        call: &TextDrawCall,
        binds: &mut TextBindCache,
    ) -> RenderResult<()> {
        if call.vertices.is_empty() || self.text.atlas_sets.is_empty() {
            return Ok(());
        }
        let extent = self.swapchain.extent;
        // The text vertices are in overlay units (mapped to NDC by the shader's
        // divide by win_width/height); the scissor is in attachment pixels, so a
        // per-call clip rect scales between the two.
        let ui = self.logical_size();

        // Scissor a clipped (scrollable-panel) call to its band, restoring the
        // full-window scissor for an unclipped call so chrome is never cropped.
        // Resolved first so a fully-scrolled-out row skips before it takes any
        // room in the frame's upload buffer.
        let scissor = match call.clip_rect {
            Some(clip) => {
                match fullscreen::clip_rect_to_scissor(clip, ui, (extent.width, extent.height)) {
                    None => return Ok(()),
                    Some(rect) => rect,
                }
            }
            None => (0, 0, extent.width, extent.height),
        };

        let atlas_idx = call.atlas_slot.min(self.text.atlas_sets.len() - 1);

        // Append this label's vertex + index geometry into the frame slot's
        // persistent upload buffer (sized up front by `reserve` in
        // `encode_composite_and_text`) and bind sub-ranges of it.
        let (vert_buf, vert_offset) = self
            .text
            .upload
            .push(args.frame_idx, bytemuck::cast_slice(&call.vertices))?;
        let (idx_buf, idx_offset) = self
            .text
            .upload
            .push(args.frame_idx, bytemuck::cast_slice(&call.indices))?;

        if binds.atlas_changed(atlas_idx) {
            rec.bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                &self.text.pipeline_layout,
                0,
                std::slice::from_ref(&self.text.atlas_sets[atlas_idx]),
                &[],
            );
        }
        if binds.scissor_changed(scissor) {
            let (x, y, width, height) = scissor;
            let rect = vk::Rect2D {
                offset: vk::Offset2D { x, y },
                extent: vk::Extent2D { width, height },
            };
            rec.set_scissor(&rect);
        }
        // The upload buffer was reserved up front, so it is not reallocated
        // while bound.
        rec.bind_vertex_buffers(0, &[vert_buf], &[vert_offset]);
        rec.bind_index_buffer(idx_buf, idx_offset, vk::IndexType::UINT16);
        rec.draw_indexed(u32::saturating_from(call.indices.len()), 1, 0, 0, 0);
        self.inc_draw_calls(1);
        Ok(())
    }

    fn end_composite(&self, rec: &Self::Rec, _args: &Self::Args) {
        rec.end_render_pass();
    }
}

impl VkContext {
    // Encode the composite tonemap pass and text overlay for frame slot
    // `frame_idx`, targeting the swapchain image at `image_index`, via the shared
    // `gfx::fullscreen` driver.
    pub(in crate::vulkan) fn encode_composite_and_text(
        &self,
        rec: &Recorder<'_>,
        image_index: u32,
        frame_idx: usize,
        text_calls: &[TextDrawCall],
        reactive_valid: bool,
    ) -> RenderResult<()> {
        // Reset this slot's text-upload cursor and ensure its buffer holds the
        // whole frame's text up front, so each `text_draw` only appends (and
        // never reallocates out from under an already-bound sub-range). The
        // frame fence waited before this frame's recording has already confirmed
        // the GPU is done with this slot, so resetting / growing it is race-free.
        let text_bytes = fullscreen::text_upload_bytes(text_calls, UPLOAD_ALIGN);
        self.text
            .upload
            .reserve(&self.hw.alloc, frame_idx, text_bytes)?;

        let args = CompositeArgs {
            image_index: image_index as usize,
            frame_idx,
            reactive_valid,
        };
        fullscreen::encode_composite_chain(&CompositePass { ctx: self }, rec, &args, text_calls)
    }
}
