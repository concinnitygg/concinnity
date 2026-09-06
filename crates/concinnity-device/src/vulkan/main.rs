// src/vulkan/main.rs
//
// Main scene pass for the Vulkan backend: linear-light HDR off-screen
// render that draws every static / instanced / skinned object into the
// multisampled HDR colour + depth attachments (resolved into
// `hdr_resolve` for the post stack). One Vulkan render pass, two indirect
// draws over the GPU-cull-written indirect buffer per shader bucket: the
// static + instance + runtime prefix against the shared VB/IB, then the skinned
// tail against this frame's deformed vertices.
//
// The shape mirrors `metal/draw/main.rs::encode_main_pass`; the graph
// executor in [`graph_exec.rs`](graph_exec.rs) dispatches
// `PassId::Main` here. The Shadow → Main `shadow_map` read edge in
// the frame graph pins Shadow before Main via toposort; the encoder
// itself only deals with the HDR pass.

use ash::vk;

use super::context::VkContext;

impl VkContext {
    // Recompute every instanced cluster's per-LOD-bucket partition for the
    // current camera into `instanced.lod_buckets`, which the spot shadow pass
    // reads. Run on `&mut self` from `execute_graph` before the render-graph
    // fan-out, mirroring `prepare_particle_pass`. Mirrors
    // `DxContext::build_instance_upload`.
    pub(in crate::vulkan) fn prepare_instanced_clusters(&mut self, cam_pos: [f32; 3]) {
        if self.instanced.clusters.is_empty() {
            return;
        }
        // Re-shape on a runtime cluster-count change (asset hot-reload), then
        // clear each row in place to reuse its heap allocation.
        if self.instanced.lod_buckets.len() != self.instanced.clusters.len() {
            self.instanced
                .lod_buckets
                .resize(self.instanced.clusters.len(), Vec::new());
        }
        for (cluster_idx, cluster) in self.instanced.clusters.iter().enumerate() {
            let row = &mut self.instanced.lod_buckets[cluster_idx];
            row.clear();
            if cluster.instances.is_empty() {
                continue;
            }
            row.extend(cluster.lod_buckets(cam_pos));
        }
    }

    // Encode the main HDR scene pass for frame slot `frame_idx` into the
    // multisampled colour + depth attachments of `framebuffers[frame_idx]`; the
    // render pass resolves into `hdr_resolve` (the post-stack input) on
    // `cmd_end_render_pass`. Every draw comes from the GPU-culled indirect
    // buffer the cull compute kernel wrote earlier this frame.
    pub(in crate::vulkan) fn encode_main_pass(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        world_hidden: bool,
    ) {
        let device = self.device.clone();
        let device = &device;
        let extent = self.render_extent;

        // Opaque menu backdrop, MSAA path: skip the main render pass entirely.
        // Beginning it would clear the MSAA colour+depth and, on
        // `end_render_pass`, resolve the (undrawn) MSAA colour into hdr_resolve:
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
        // colour attachment), so it keeps its cheap clear-only render pass.
        if world_hidden && self.msaa_samples != vk::SampleCountFlags::TYPE_1 {
            super::texture::transition_image_layout(
                device,
                cmd,
                self.hdr_resolve_images[frame_idx].image,
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
        let [r, g, b, a] = self.view.clear_color;
        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [r, g, b, a],
            },
        };
        let clear_depth = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };
        let clears: &[vk::ClearValue] = if self.msaa_samples != vk::SampleCountFlags::TYPE_1 {
            &[clear_color, clear_depth, vk::ClearValue::default()]
        } else {
            &[clear_color, clear_depth]
        };

        // Under two-pass occlusion, phase 1 renders into a variant render pass
        // that STORE's the MSAA colour (so `Main2` can load + composite onto
        // it) and leaves the colour in COLOR_ATTACHMENT_OPTIMAL. The clears are
        // identical (phase 1 still CLEAR's), so only the render pass differs.
        // The raymarch pass needs the same STORE-colour treatment (it loads +
        // re-resolves the MSAA colour after AutoExposure), so when raymarch is
        // active but two-pass occlusion is not, switch to its store-colour pass.
        let render_pass = if self.two_pass_occlusion_active() {
            self.cull
                .main_render_pass_phase1
                .as_ref()
                .unwrap_or(&self.main_render_pass)
        } else if let Some(rp) = self
            .raymarch
            .as_ref()
            .and_then(|r| r.main_store_color_pass.as_ref())
        {
            rp
        } else {
            &self.main_render_pass
        };
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass.handle())
            .framebuffer(self.framebuffers[frame_idx].handle())
            .render_area(vk::Rect2D::default().extent(extent))
            .clear_values(clears);

        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe { device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE) };

        // Opaque menu backdrop, single-sample path: the render pass above
        // already cleared hdr_resolve (the colour attachment) and there is no
        // resolve to skip, so just end immediately; nothing of the world draws
        // behind the menu. (The MSAA path returned earlier without beginning a
        // render pass at all.)
        if world_hidden {
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe { device.cmd_end_render_pass(cmd) };
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
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
        }

        // Geometry buffers for the static + instance + runtime prefix.
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.geometry.vertex_buffer.buffer()], &[0]);
            device.cmd_bind_index_buffer(
                cmd,
                self.geometry.index_buffer.buffer(),
                0,
                vk::IndexType::UINT32,
            );
        }

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
        let use_bindless = self.cull.bindless_pipeline.is_some() && self.cull_count() > 0;
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
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.handle());
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout.handle(),
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
                // pipeline. The cull kernel wrote every record's command into
                // exactly one bucket's region, so the regions never double-draw.
                device.cmd_draw_indexed_indirect(
                    cmd,
                    self.cull.indirect_buffers[frame_idx].buffer(),
                    0,
                    self.skinned_record_base() as u32,
                    std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                );
            }
            // GPU expands the indirect buffer to up to `draw.n_objects` draw
            // commands inside, but the call count surfaced to the profiler
            // is the host-side draw. Mirrors DirectX / Metal.
            self.inc_draw_calls(1);
            self.inc_draw_calls(self.draw_bucket_regions(
                cmd,
                self.cull.indirect_buffers[frame_idx].buffer(),
                self.skinned_record_base() as u32,
            ));
            // Restore the default bindless pipeline for the sub-paths below.
            if self.shader_bucket_count() > 1 {
                // SAFETY: `cmd` is a command buffer in the recording state, and every handle and
                // slice these commands name is live for the call.
                unsafe {
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.handle(),
                    )
                };
            }
        }

        // Skinned meshes main pass. Skinned objects ride the same cull buffers as
        // static + instances and are drawn (as rigid deformed geometry) by a 2nd
        // `cmd_draw_indexed_indirect` over this frame's deformed-vertex buffer +
        // the skinned index buffer, reading the cull-written indirect buffer from
        // `skinned_record_base()`. The `encode_skin` compute pass (Cull graph arm)
        // has already posed the deformed buffer.
        if use_bindless
            && self.draw.n_skinned > 0
            && let (Some(bindless_pipeline), Some(bindless_layout), Some(deformed)) = (
                self.cull.bindless_pipeline.as_ref(),
                self.cull.bindless_pipeline_layout.as_ref(),
                self.skinned.deformed.get(frame_idx),
            )
        {
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and
            // slice these commands name is live for the call.
            unsafe {
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.wireframe_or(bindless_pipeline, self.wireframe.bindless.as_ref())
                        .handle(),
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    bindless_layout.handle(),
                    0,
                    &[
                        self.descriptors.global_sets[frame_idx],
                        self.cull.bindless_sets[frame_idx],
                    ],
                    &[],
                );
                // Bind the deformed verts (base_vertex = 0, global skinned
                // indexing) + the skinned IB.
                device.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    std::slice::from_ref(&deformed.buffer),
                    &[0],
                );
                device.cmd_bind_index_buffer(
                    cmd,
                    self.skinned.index_buffer.buffer(),
                    0,
                    vk::IndexType::UINT32,
                );
                // Indirect draw #2: the skinned tail
                // `[skinned_record_base(), cull_count())`, byte-offset into the
                // same indirect command buffer.
                let cmd_stride = std::mem::size_of::<vk::DrawIndexedIndirectCommand>();
                device.cmd_draw_indexed_indirect(
                    cmd,
                    self.cull.indirect_buffers[frame_idx].buffer(),
                    (self.skinned_record_base() * cmd_stride) as u64,
                    self.draw.n_skinned as u32,
                    cmd_stride as u32,
                );
            }
            self.inc_draw_calls(1);
        }

        // End the main scene pass. The render pass leaves the HDR resolve
        // image in SHADER_READ_ONLY_OPTIMAL for the composite pass to sample.
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe { device.cmd_end_render_pass(cmd) };
    }

    // Phase-2 main pass for two-pass occlusion. Loads (does not clear) the
    // phase-1 HDR colour + depth, re-runs the bindless indirect draw (static
    // objects + merged instances, both Hi-Z-tested through the cull buffer) over
    // the phase-2 indirect buffer `Cull2` wrote, and resolves the combined scene
    // into `hdr_resolve` for the post stack. Skinned geometry is not Hi-Z-culled,
    // so it was fully drawn in phase 1 and is not repeated here. A no-op unless
    // the bindless path is active with build-time geometry. Mirrors
    // `directx/draw/main.rs::encode_main_pass_phase2`.
    //
    // `bindless_pipeline` was created against `main_render_pass` but is used
    // here with `main_render_pass_phase2`; that is valid because the two passes
    // are render-pass-compatible (identical attachment count / formats / sample
    // counts, only load/store ops + layouts differ). Keep them so if the
    // phase-2 attachment set ever diverges, build a phase-2-specific pipeline.
    pub(in crate::vulkan) fn encode_main_pass_phase2(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
    ) {
        let (Some(render_pass), Some(pipeline), Some(layout)) = (
            self.cull.main_render_pass_phase2.as_ref(),
            self.cull.bindless_pipeline.as_ref(),
            self.cull.bindless_pipeline_layout.as_ref(),
        ) else {
            return;
        };
        let pipeline = self.wireframe_or(pipeline, self.wireframe.bindless.as_ref());
        if self.draw.n_objects == 0 || self.cull.indirect_buffers2.is_empty() {
            return;
        }
        let device = self.device.clone();
        let device = &device;
        let extent = self.render_extent;

        // The phase-2 render pass LOADs the phase-1 colour + depth. Because it
        // shares the main render pass's subpass dependency (so the shared
        // framebuffer + bindless pipeline stay render-pass-compatible), that
        // dependency does not order the phase-1 writes before this LOAD. Emit
        // the ordering explicitly here: phase-1 Main's colour + depth writes
        // (and HizBuild's depth restore) -> this pass's attachment load. Spans
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
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&load_barrier),
                &[],
                &[],
            );
        }

        // LOAD render pass: no clears (loadOp = LOAD / DONT_CARE), so no clear
        // values are required.
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass.handle())
            .framebuffer(self.framebuffers[frame_idx].handle())
            .render_area(vk::Rect2D::default().extent(extent));
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe { device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE) };

        // Same negative-height viewport flip as the phase-1 main pass so the
        // disoccluded geometry rasterises into identical pixels.
        let vp = vk::Viewport {
            x: 0.0,
            y: extent.height as f32,
            width: extent.width as f32,
            height: -(extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(extent);
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.geometry.vertex_buffer.buffer()], &[0]);
            device.cmd_bind_index_buffer(
                cmd,
                self.geometry.index_buffer.buffer(),
                0,
                vk::IndexType::UINT32,
            );

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.handle());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout.handle(),
                0,
                &[
                    self.descriptors.global_sets[frame_idx],
                    self.cull.bindless_sets[frame_idx],
                ],
                &[],
            );
            // Indirect draw #1: the static + instance prefix against the static
            // VB/IB bound above, once per shader bucket.
            device.cmd_draw_indexed_indirect(
                cmd,
                self.cull.indirect_buffers2[frame_idx].buffer(),
                0,
                self.skinned_record_base() as u32,
                std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
        }
        self.inc_draw_calls(1);
        self.inc_draw_calls(self.draw_bucket_regions(
            cmd,
            self.cull.indirect_buffers2[frame_idx].buffer(),
            self.skinned_record_base() as u32,
        ));

        // Indirect draw #2: the skinned tail against the deformed VB + skinned IB.
        // The descriptor sets bound above persist, so only the pipeline (a bucket
        // may have replaced it) and the vertex/index buffers rebind. Skinned draws
        // always render bucket 0.
        if self.draw.n_skinned > 0
            && let Some(deformed) = self.skinned.deformed.get(frame_idx)
        {
            let cmd_stride = std::mem::size_of::<vk::DrawIndexedIndirectCommand>();
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.handle());
                device.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    std::slice::from_ref(&deformed.buffer),
                    &[0],
                );
                device.cmd_bind_index_buffer(
                    cmd,
                    self.skinned.index_buffer.buffer(),
                    0,
                    vk::IndexType::UINT32,
                );
                device.cmd_draw_indexed_indirect(
                    cmd,
                    self.cull.indirect_buffers2[frame_idx].buffer(),
                    (self.skinned_record_base() * cmd_stride) as u64,
                    self.draw.n_skinned as u32,
                    cmd_stride as u32,
                );
            }
            self.inc_draw_calls(1);
        }

        // End the phase-2 pass. The render pass resolves the combined phase-1 +
        // phase-2 MSAA colour into `hdr_resolve` and leaves it
        // SHADER_READ_ONLY_OPTIMAL, so the post stack reads the combined scene.
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe { device.cmd_end_render_pass(cmd) };
    }
}
