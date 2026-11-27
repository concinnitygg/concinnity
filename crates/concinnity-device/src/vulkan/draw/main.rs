//! Main scene pass for the Vulkan backend: linear-light HDR off-screen
//! render that draws every static / instanced / skinned object into the
//! multisampled HDR color + depth attachments (resolved into
//! `hdr_resolve` for the post stack). One rendering scope, two indirect
//! draws over the GPU-cull-written indirect buffer per shader bucket: the
//! static + instance + runtime prefix against the shared VB/IB, then the skinned
//! tail against this frame's deformed vertices.
//!
//! The shape mirrors `metal/draw/main.rs::encode_main_pass`; the graph
//! executor in [`graph_exec.rs`](graph_exec.rs) dispatches
//! `PassId::Main` here. The Shadow → Main `shadow_map` read edge in
//! the frame graph pins Shadow before Main via toposort; the encoder
//! itself only deals with the HDR pass.

use ash::vk;
use concinnity_engine::gfx::render_types::SKINNED_STREAM;

use super::super::context::VkContext;
use crate::vulkan::graph_exec::GraphFrameParams;
use crate::vulkan::main_pass::{self, MainClose, MainOpen, SceneResumers};
use crate::vulkan::record::Recorder;
use crate::vulkan::rendering::ImageUse;

impl VkContext {
    // Encode the main HDR scene pass for frame slot `frame_idx`, resolving into
    // `hdr_resolve`. Every draw comes from the GPU-culled indirect buffer.
    pub(in crate::vulkan) fn encode_main_pass(
        &self,
        rec: &Recorder<'_>,
        params: &GraphFrameParams<'_>,
        grass: Option<&crate::vulkan::grass::GrassFrame>,
    ) {
        let (frame_idx, world_hidden) = (params.frame_idx, params.frame.world_hidden);
        let extent = self.targets.render_extent;

        // Behind an opaque menu on the MSAA path, skip the pass entirely: its
        // resolve of an undrawn frame would be the paused frame's main GPU cost. A
        // lone barrier leaves hdr_resolve ready to sample. The single-sample path has
        // no resolve, so it keeps its clear-only pass.
        if world_hidden && self.targets.msaa_samples != vk::SampleCountFlags::TYPE_1 {
            super::super::texture::transition_image_layout(
                rec,
                self.targets.hdr_resolve_images[frame_idx].image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
            );
            return;
        }

        let [r, g, b, a] = self.pre_exposure.clear_color(self.state.view.clear_color);
        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [r, g, b, a],
            },
        };
        // The scene was last sampled by the previous frame's readers.
        let attachments = self.main_attachments(frame_idx, ImageUse::Sampled);
        let close = params.scene.main_close(self.main_msaa());
        main_pass::begin(rec, &attachments, MainOpen::Clear(clear_color), close);

        // Behind an opaque menu on the single-sample path the clear above is all there
        // is.
        if world_hidden {
            main_pass::end(rec, &attachments, close);
            return;
        }

        // Viewport: negative height flips Y to match Metal coordinate system.
        let vp = vk::Viewport {
            x: 0.0,
            y: extent.height as f32,
            width: extent.width as f32,
            height: -(extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(extent);
        rec.set_viewport(&vp);
        rec.set_scissor(&scissor);

        // Geometry buffers for the static + instance + runtime prefix.
        rec.bind_vertex_buffers(0, &[self.geometry.vertex_buffer.buffer()], &[0]);
        rec.bind_index_buffer(
            self.geometry.index_buffer.buffer(),
            0,
            vk::IndexType::UINT32,
        );

        // Build-time static objects draw through the bindless pipeline from the
        // GPU-culled indirect buffer; culled draws have `instance_count = 0`. The
        // object id rides `first_instance`, and everything else comes from the object
        // SSBO and the texture pool.
        let use_bindless =
            self.cull.bindless_pipeline.is_some() && self.state.draw.record_count() > 0;
        if use_bindless {
            let pipeline = self.wireframe_or(
                self.cull
                    .bindless_pipeline
                    .as_ref()
                    .expect("bindless pipeline is live"),
                self.wireframe.bindless.as_ref(),
            );
            let layout = self
                .cull
                .bindless_pipeline_layout
                .as_ref()
                .expect("bindless pipeline layout is live alongside its pipeline");
            rec.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline);
            rec.bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                &[
                    self.descriptors.global_sets[frame_idx],
                    self.cull.bindless_sets[frame_idx],
                ],
                &[],
            );
            // Indirect draw 1: the static + instance prefix, once per shader bucket under
            // its pipeline. The cull puts every record in exactly one bucket's list. The
            // skinned tail is draw 2 below.
            self.draw_main_stream(rec, self.cull.indirect_buffers[frame_idx].buffer(), 0);
            // GPU expands the indirect buffer to up to `draw.n_objects` draw
            // commands inside, but the call count surfaced to the profiler is
            // the host-side draw.
            self.inc_draw_calls(1);
            self.inc_draw_calls(self.draw_bucket_regions(
                rec,
                self.cull.indirect_buffers[frame_idx].buffer(),
                frame_idx,
            ));
            // Restore the default bindless pipeline for the sub-paths below.
            if self.shader_bucket_count() > 1 {
                rec.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline);
            }
        }

        // Indirect draw 2: the skinned tail over this frame's deformed VB and the
        // skinned IB, already posed by `encode_skin`.
        if use_bindless
            && self.state.draw.n_skinned > 0
            && let (Some(bindless_pipeline), Some(bindless_layout), Some(deformed)) = (
                self.cull.bindless_pipeline.as_ref(),
                self.cull.bindless_pipeline_layout.as_ref(),
                self.skinned.deformed.get(frame_idx),
            )
        {
            let skinned_pipeline =
                self.wireframe_or(bindless_pipeline, self.wireframe.bindless.as_ref());
            rec.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, skinned_pipeline);
            rec.bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                bindless_layout,
                0,
                &[
                    self.descriptors.global_sets[frame_idx],
                    self.cull.bindless_sets[frame_idx],
                ],
                &[],
            );
            // Bind the deformed verts (base_vertex = 0, global skinned
            // indexing) + the skinned IB.
            // The deformed VB was posed by the skin compute pass ahead of this one.
            rec.bind_vertex_buffers(0, std::slice::from_ref(&deformed.buffer), &[0]);
            rec.bind_index_buffer(self.skinned.index_buffer.buffer(), 0, vk::IndexType::UINT32);
            // Indirect draw #2: the skinned tail's list, from
            // `skinned_record_base()` in the same indirect command buffer.
            self.draw_main_stream(
                rec,
                self.cull.indirect_buffers[frame_idx].buffer(),
                SKINNED_STREAM,
            );
            self.inc_draw_calls(1);
        }

        // The blades shade on the lights, shadows and environment the surfaces
        // just bound in sets 0 and 1.
        if let (Some(frame), true) = (grass, use_bindless) {
            self.encode_grass_main(rec, frame_idx, frame);
        }

        // The sky lands behind the early pass's geometry. Under two-pass occlusion
        // `MainLate` then draws the disoccluded rest over it, which wins the depth
        // test because the sky writes none.
        if self.draws_sky(self.state.view.mode) {
            self.encode_main_sky(rec, frame_idx);
        }

        // End the main scene pass, leaving the HDR resolve sampled for the
        // post stack unless a later scope resumes it.
        main_pass::end(rec, &attachments, close);
    }

    // Late main pass for two-pass occlusion: load the early color and depth, draw
    // what `CullLate` kept, and resolve into `hdr_resolve`. Skinned geometry is
    // not Hi-Z-culled, so it drew fully in the early pass. Uses the main pipeline.
    pub(in crate::vulkan) fn encode_main_pass_late(
        &self,
        rec: &Recorder<'_>,
        frame_idx: usize,
        scene: SceneResumers,
    ) {
        let msaa = self.main_msaa();
        let close = scene.late_close(msaa);
        // The early pass left a multisampled scene's resolve sampled, and held a
        // single-sample scene as an attachment.
        let scene_from = if msaa {
            ImageUse::Sampled
        } else {
            ImageUse::ColorAttachment
        };
        let attachments = self.main_attachments(frame_idx, scene_from);
        // With nothing to draw the late pass still leaves the scene where it
        // would have.
        let skip = || {
            if close == MainClose::Sampled {
                main_pass::release(rec, &attachments);
            }
        };
        let (true, Some(pipeline), Some(layout)) = (
            self.two_pass_occlusion_built(),
            self.cull.bindless_pipeline.as_ref(),
            self.cull.bindless_pipeline_layout.as_ref(),
        ) else {
            skip();
            return;
        };
        let pipeline = self.wireframe_or(pipeline, self.wireframe.bindless.as_ref());
        if self.state.draw.n_objects == 0 || self.cull.indirect_buffers_late.is_empty() {
            skip();
            return;
        }
        let extent = self.targets.render_extent;

        // Resume the early pass's color and depth.
        main_pass::begin(rec, &attachments, MainOpen::Resume, close);

        // Same negative-height viewport flip as the early main pass so the
        // disoccluded geometry rasterizes into identical pixels.
        let vp = vk::Viewport {
            x: 0.0,
            y: extent.height as f32,
            width: extent.width as f32,
            height: -(extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(extent);
        rec.set_viewport(&vp);
        rec.set_scissor(&scissor);
        rec.bind_vertex_buffers(0, &[self.geometry.vertex_buffer.buffer()], &[0]);
        rec.bind_index_buffer(
            self.geometry.index_buffer.buffer(),
            0,
            vk::IndexType::UINT32,
        );

        rec.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline);
        rec.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            layout,
            0,
            &[
                self.descriptors.global_sets[frame_idx],
                self.cull.bindless_sets[frame_idx],
            ],
            &[],
        );
        // Indirect draw #1: the static + instance prefix against the static
        // VB/IB bound above, once per shader bucket.
        self.draw_main_stream(rec, self.cull.indirect_buffers_late[frame_idx].buffer(), 0);
        self.inc_draw_calls(1);
        self.inc_draw_calls(self.draw_bucket_regions(
            rec,
            self.cull.indirect_buffers_late[frame_idx].buffer(),
            frame_idx,
        ));

        // Indirect draw #2: the skinned tail against the deformed VB + skinned IB.
        // The descriptor sets bound above persist, so only the pipeline (a bucket
        // may have replaced it) and the vertex/index buffers rebind. Skinned draws
        // always render bucket 0.
        if self.state.draw.n_skinned > 0
            && let Some(deformed) = self.skinned.deformed.get(frame_idx)
        {
            rec.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline);
            // The deformed VB was posed by the skin compute pass ahead of this one.
            rec.bind_vertex_buffers(0, std::slice::from_ref(&deformed.buffer), &[0]);
            rec.bind_index_buffer(self.skinned.index_buffer.buffer(), 0, vk::IndexType::UINT32);
            self.draw_main_stream(
                rec,
                self.cull.indirect_buffers_late[frame_idx].buffer(),
                SKINNED_STREAM,
            );
            self.inc_draw_calls(1);
        }

        // End the late pass, which resolves the combined early + late MSAA color
        // into `hdr_resolve` and leaves it sampled for the post stack.
        main_pass::end(rec, &attachments, close);
    }
}
