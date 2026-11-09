//! Main scene pass for the Vulkan backend: linear-light HDR off-screen
//! render that draws every static / instanced / skinned object into the
//! multisampled HDR color + depth attachments (resolved into
//! `hdr_resolve` for the post stack). One Vulkan render pass, two indirect
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
use crate::vulkan::depth;
use crate::vulkan::record::Recorder;

impl VkContext {
    // Encode the main HDR scene pass for frame slot `frame_idx` into the
    // multisampled color + depth attachments of `framebuffers[frame_idx]`; the
    // render pass resolves into `hdr_resolve` (the post-stack input) on
    // `cmd_end_render_pass`. Every draw comes from the GPU-culled indirect
    // buffer the cull compute kernel wrote earlier this frame.
    pub(in crate::vulkan) fn encode_main_pass(
        &self,
        rec: &Recorder<'_>,
        frame_idx: usize,
        world_hidden: bool,
        grass: Option<&crate::vulkan::grass::GrassFrame>,
    ) {
        let extent = self.targets.render_extent;

        // Opaque menu backdrop, MSAA path: skip the main render pass entirely.
        // Beginning it would clear the MSAA color+depth and, on
        // `end_render_pass`, resolve the (undrawn) MSAA color into hdr_resolve:
        // a full-render-resolution resolve of a frame nothing presents (the
        // composite samples the post-stack scene + the opaque overlay on top).
        // On an immediate-mode GPU that resolve is the bulk of the paused
        // frame's GPU cost. Skipping it drops the main pass to a lone layout
        // barrier (~0us, so it falls off the passes HUD like Metal / DirectX),
        // and the barrier leaves hdr_resolve sampled-ready for the plain-world
        // composite path (no TAA / reflections sample it directly). Its
        // contents are irrelevant under the opaque overlay, matching the
        // DirectX paused path (which likewise skips the resolve). The
        // single-sample path below has no resolve to skip (hdr_resolve is the
        // color attachment), so it keeps its cheap clear-only render pass.
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

        // Clears for the main HDR attachments. With MSAA on, the
        // resolve attachment doesn't need a clear (resolve overwrites);
        // a `ClearValue::default()` placeholder keeps the slice index
        // aligned with `main_render_pass`'s attachment count.
        let [r, g, b, a] = self.pre_exposure.clear_color(self.state.view.clear_color);
        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [r, g, b, a],
            },
        };
        let clear_depth = depth::CLEAR_VALUE;
        let clears: &[vk::ClearValue] = if self.targets.msaa_samples == vk::SampleCountFlags::TYPE_1
        {
            &[clear_color, clear_depth]
        } else {
            &[clear_color, clear_depth, vk::ClearValue::default()]
        };

        // Under two-pass occlusion, the early pass renders into a variant render pass
        // that STORE's the MSAA color (so `MainLate` can load + composite onto
        // it) and leaves the color in COLOR_ATTACHMENT_OPTIMAL. The clears are
        // identical (the early pass still CLEAR's), so only the render pass differs.
        // The raymarch pass needs the same STORE-color treatment (it loads +
        // re-resolves the MSAA color after AutoExposure), so when raymarch is
        // active but two-pass occlusion is not, switch to its store-color pass.
        let render_pass = if self.two_pass_occlusion_active() {
            self.cull
                .main_render_pass_early
                .as_ref()
                .unwrap_or(&self.targets.main_render_pass)
        } else if let Some(rp) = self
            .raymarch
            .as_ref()
            .and_then(|r| r.main_store_color_pass.as_ref())
        {
            rp
        } else {
            &self.targets.main_render_pass
        };
        rec.begin_render_pass(
            render_pass,
            &self.targets.framebuffers[frame_idx],
            vk::Rect2D::default().extent(extent),
            clears,
        );

        // Opaque menu backdrop, single-sample path: the render pass above
        // already cleared hdr_resolve (the color attachment) and there is no
        // resolve to skip, so just end immediately; nothing of the world draws
        // behind the menu. (The MSAA path returned earlier without beginning a
        // render pass at all.)
        if world_hidden {
            rec.end_render_pass();
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

        // Build-time static objects render through the bindless pipeline
        // driven by the GPU-culled indirect command buffer the cull
        // compute pass wrote above. One `cmd_draw_indexed_indirect`
        // issues every build-time object's draw; culled / disabled objects
        // were written with `instance_count = 0` (a no-op). Each draw is
        // stateless apart from the object id, which rides `first_instance`
        // (Vulkan's `gl_InstanceIndex` includes it); model/material/textures
        // are fetched from the per-frame GpuObjectData SSBO + the bindless
        // texture pool. Instances, streamed chunks and runtime clones are
        // records of their own in the same buffer.
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
            // Indirect draw #1: the static + instance prefix
            // `[0, skinned_record_base())` against the static VB/IB bound above.
            // The skinned tail is drawn by a second indirect draw below (deformed
            // VB + skinned IB), reading the same indirect buffer from
            // `skinned_record_base()` on.
            //
            // Once per shader bucket: bucket 0 runs under the bindless pipeline
            // bound above, each later bucket under its material shader's own
            // pipeline. The cull packed every kept record into exactly one
            // stream's list, so the lists never double-draw.
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

        // Skinned meshes main pass. Skinned objects ride the same cull buffers as
        // static + instances and are drawn (as rigid deformed geometry) by a 2nd
        // `cmd_draw_indexed_indirect` over this frame's deformed-vertex buffer +
        // the skinned index buffer, reading the cull-written indirect buffer from
        // `skinned_record_base()`. The `encode_skin` compute pass (Cull graph arm)
        // has already posed the deformed buffer.
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

        // End the main scene pass. The render pass leaves the HDR resolve
        // image in SHADER_READ_ONLY_OPTIMAL for the composite pass to sample.
        rec.end_render_pass();
    }

    // Late main pass for two-pass occlusion. Loads (does not clear) the
    // early HDR color + depth, re-runs the bindless indirect draw (static
    // objects + merged instances, both Hi-Z-tested through the cull buffer) over
    // the late indirect buffer `CullLate` wrote, and resolves the combined scene
    // into `hdr_resolve` for the post stack. Skinned geometry is not Hi-Z-culled,
    // so it was fully drawn in the early pass and is not repeated here. A no-op unless
    // the bindless path is active with build-time geometry. Mirrors
    // `directx/draw/main.rs::encode_main_pass_late`.
    //
    // `bindless_pipeline` was created against `main_render_pass` but is used
    // here with `main_render_pass_late`; that is valid because the two passes
    // are render-pass-compatible (identical attachment count / formats / sample
    // counts, only load/store ops + layouts differ). Keep them so if the
    // late attachment set ever diverges, build a late-specific pipeline.
    pub(in crate::vulkan) fn encode_main_pass_late(&self, rec: &Recorder<'_>, frame_idx: usize) {
        let (Some(render_pass), Some(pipeline), Some(layout)) = (
            self.cull.main_render_pass_late.as_ref(),
            self.cull.bindless_pipeline.as_ref(),
            self.cull.bindless_pipeline_layout.as_ref(),
        ) else {
            return;
        };
        let pipeline = self.wireframe_or(pipeline, self.wireframe.bindless.as_ref());
        if self.state.draw.n_objects == 0 || self.cull.indirect_buffers_late.is_empty() {
            return;
        }
        let extent = self.targets.render_extent;

        // The late render pass LOADs the early color + depth. Because it
        // shares the main render pass's subpass dependency (so the shared
        // framebuffer + bindless pipeline stay render-pass-compatible), that
        // dependency does not order the early pass writes before this LOAD. Emit
        // the ordering explicitly here: early Main's color + depth writes
        // (and HizEarly's depth restore) -> this pass's attachment load. Spans
        // the command-buffer boundary via submission order under parallel
        // recording.
        let load_barrier = vk::MemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );
        rec.pipeline_barrier(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            std::slice::from_ref(&load_barrier),
            &[],
            &[],
        );

        // LOAD render pass: no clears (loadOp = LOAD / DONT_CARE), so no clear
        // values are required.
        rec.begin_render_pass(
            render_pass,
            &self.targets.framebuffers[frame_idx],
            vk::Rect2D::default().extent(extent),
            &[],
        );

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

        // End the late pass. The render pass resolves the combined early +
        // late MSAA color into `hdr_resolve` and leaves it
        // SHADER_READ_ONLY_OPTIMAL, so the post stack reads the combined scene.
        rec.end_render_pass();
    }
}
