// src/vulkan/shadow.rs
//
// Shadow pass for the Vulkan backend: one depth-only render pass per
// cascade slice of the shadow-map array. Both of shadow_map's transitions are
// graph-driven: shadow_map is the render graph's `shadow_map` resource, so the
// executor emits (over every cascade layer) the Shadow producer barrier
// (`SHADER_READ_ONLY_OPTIMAL` -> `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`, the
// cross-frame reset for this frame's shadow loop) before this pass and the Main
// consumer barrier (`DEPTH_STENCIL_ATTACHMENT_OPTIMAL` -> `SHADER_READ_ONLY_OPTIMAL`,
// letting the main pass sample the cascades) before the Main pass. The map rests
// sampled between frames, so there is no inline reset.
//
// The cascades are GPU-driven: a per-cascade cull dispatch writes one indirect
// buffer per cascade and each cascade is issued with one
// `cmd_draw_indexed_indirect` (static + instance prefix) + one for the skinned
// tail. Streamed chunks and runtime clones ride the same records, so the CPU
// never walks a caster list here. Spot slices keep their own per-object encoder
// in [`spot_shadow.rs`](spot_shadow.rs): the indirect buffer is laid out per
// cascade and has no slots for them.
//
// The shape mirrors `metal/draw/shadow.rs::encode_shadow_pass`; the
// graph executor in [`graph_exec.rs`](graph_exec.rs) dispatches
// `PassId::Shadow` here.

use ash::vk;

use crate::vulkan::owned::VkDevice;

use super::context::VkContext;

impl VkContext {
    // Encode the cascaded-shadow-map render passes for frame slot
    // `frame_idx`: one render pass per cascade slice, drawing every
    // visible static / instanced / skinned caster into the array layer
    // for that cascade. Ends with a single barrier transitioning every
    // cascade slice from depth-attachment to shader-read so the main
    // pass can sample them.
    //
    // A no-op when no shadow pipeline is built (geometry-less worlds
    // or a world that opted out of CSM). The caller must compute +
    // upload `shadow_uniforms` and `upload_joint_matrices` before this
    // runs so the shadow vertex shader sees the current cascade VPs
    // and the skinned caster pass sees the current joint matrices.
    pub(in crate::vulkan) fn encode_shadow_pass(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        cam_pos: [f32; 3],
        elapsed: f32,
    ) {
        // The depth-only pipeline is the spot pass's; its absence still means
        // shadows are not configured, so there is nothing to render here either.
        if self.shadow.pipeline.is_none() {
            return;
        }

        // Raymarched SDF shadow casters share these cascade DSVs: upload this
        // frame's animation time once (no-op without casters) so the from-light
        // SDF march lines up with the lit-side surface.
        self.upload_raymarch_shadow_view(frame_idx, elapsed);
        let device = self.device.clone();
        let device = &device;

        let sm = self.shadow.map_size;
        let shadow_extent = vk::Extent2D {
            width: sm,
            height: sm,
        };

        let clear_depth = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };

        // Cascades to re-render this frame; draw_frame computed the mask from the
        // update policy. A skipped cascade's render pass is omitted entirely, so
        // its slice keeps the depth + VP from when it was last rendered (the
        // graph-driven producer/consumer barriers still round-trip every layer,
        // preserving the contents). The 0 sentinel falls back to all cascades.
        let all_cascades = (1u32 << crate::gfx::render_types::NUM_SHADOW_CASCADES) - 1;
        let render_mask = if self.shadow.render_mask == 0 {
            all_cascades
        } else {
            self.shadow.render_mask
        };

        // Nothing to cull means nothing to draw: the render passes below still
        // run, so every re-rendered cascade is cleared for the raymarched
        // casters that follow the rasterised ones.
        let gpu_driven = self.cull.shadow_bindless_pipeline.is_some() && self.cull_count() > 0;

        // GPU-driven cull prologue: dispatch every re-rendered cascade's cull
        // before opening any render pass (Vulkan disallows compute inside a
        // render pass). Each writes that cascade's indirect buffer.
        if gpu_driven {
            self.encode_shadow_culls(cmd, frame_idx, render_mask, cam_pos);
        }

        for (cascade_idx, shadow_fb) in self.shadow.framebuffers.iter().enumerate() {
            if render_mask & (1u32 << cascade_idx) == 0 {
                continue;
            }
            let rp_begin = vk::RenderPassBeginInfo::default()
                .render_pass(self.shadow.render_pass.handle())
                .framebuffer(shadow_fb.handle())
                .render_area(vk::Rect2D::default().extent(shadow_extent))
                .clear_values(std::slice::from_ref(&clear_depth));

            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);

                // Negative-height viewport: Y-flips NDC so Y-up matches Metal.
                let vp = vk::Viewport {
                    x: 0.0,
                    y: sm as f32,
                    width: sm as f32,
                    height: -(sm as f32),
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
                let scissor = vk::Rect2D::default().extent(shadow_extent);
                device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            }

            if gpu_driven {
                self.encode_shadow_cascade_indirect(device, cmd, frame_idx, cascade_idx);
            }

            // Raymarched SDF shadow casters into this cascade's DSV, after the
            // rasterised casters and within the same render pass (no re-clear);
            // the LESS depth test keeps the nearer occluder.
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                self.encode_sdf_shadow_cascade(cmd, frame_idx, cascade_idx);
                device.cmd_end_render_pass(cmd);
            }
        }

        // shadow_map's transitions are fully graph-driven (over every cascade
        // layer): the Shadow producer barrier (SHADER_READ_ONLY ->
        // DEPTH_STENCIL_ATTACHMENT, the cross-frame reset) runs before this pass
        // and the Main consumer barrier (DEPTH_STENCIL_ATTACHMENT ->
        // SHADER_READ_ONLY) before the Main pass. Neither is emitted here, and
        // the map rests sampled between frames (no inline reset).
    }

    // GPU-driven cascade body (inside the cascade's render pass): the depth-only
    // bindless pipeline issues this cascade's static + instance prefix and the
    // skinned tail with two `cmd_draw_indexed_indirect` calls over the cascade's
    // cull-written indirect buffer. The CPU never walks the caster lists.
    fn encode_shadow_cascade_indirect(
        &self,
        device: &VkDevice,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        cascade_idx: usize,
    ) {
        let (Some(sb_pipeline), Some(sb_layout)) = (
            self.cull.shadow_bindless_pipeline.as_ref(),
            self.cull.shadow_bindless_pipeline_layout.as_ref(),
        ) else {
            return;
        };
        let Some(indirect) = self
            .cull
            .shadow_indirect_buffers
            .get(frame_idx)
            .and_then(|c| c.get(cascade_idx).map(|b| b.buffer()))
        else {
            return;
        };
        let stride = std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32;
        let prefix = self.skinned_record_base() as u32;
        let cascade = cascade_idx as u32;

        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, sb_pipeline.handle());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                sb_layout.handle(),
                0,
                &[
                    self.shadow.global_sets[frame_idx],
                    self.cull.bindless_sets[frame_idx],
                ],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                sb_layout.handle(),
                vk::ShaderStageFlags::VERTEX,
                0,
                &cascade.to_ne_bytes(),
            );

            // Static + instance prefix against the static VB/IB.
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.geometry.vertex_buffer.buffer()], &[0]);
            device.cmd_bind_index_buffer(
                cmd,
                self.geometry.index_buffer.buffer(),
                0,
                vk::IndexType::UINT32,
            );
            if prefix > 0 {
                device.cmd_draw_indexed_indirect(cmd, indirect, 0, prefix, stride);
                self.inc_draw_calls(1);
            }

            // Skinned tail against the deformed VB + skinned IB.
            if self.draw.n_skinned > 0
                && let Some(deformed) = self.skinned.deformed.get(frame_idx)
            {
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
                    indirect,
                    (self.skinned_record_base() * stride as usize) as u64,
                    self.draw.n_skinned as u32,
                    stride,
                );
                self.inc_draw_calls(1);
            }
        }
    }
}
