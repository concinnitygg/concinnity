//! Shadow pass for the Vulkan backend: one depth-only render pass per
//! cascade slice of the shadow-map array. Both of shadow_map's transitions are
//! graph-driven: shadow_map is the render graph's `shadow_map` resource, so the
//! executor emits (over every cascade layer) the Shadow producer barrier
//! (`SHADER_READ_ONLY_OPTIMAL` -> `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`, the
//! cross-frame reset for this frame's shadow loop) before this pass and the Main
//! consumer barrier (`DEPTH_STENCIL_ATTACHMENT_OPTIMAL` -> `SHADER_READ_ONLY_OPTIMAL`,
//! letting the main pass sample the cascades) before the Main pass. The map rests
//! sampled between frames, so there is no inline reset.
//!
//! The cascades are GPU-driven: a per-cascade cull dispatch writes one indirect
//! buffer per cascade and each cascade is issued with one
//! `cmd_draw_indexed_indirect` (static + instance prefix) + one for the skinned
//! tail. Streamed chunks and runtime clones ride the same records, so the CPU
//! never walks a caster list here. The spot slices in
//! [`spot_shadow.rs`](spot_shadow.rs) draw through the same `draw_shadow_view`
//! from their own indirect buffers.
//!
//! The shape mirrors `metal/draw/shadow.rs::encode_shadow_pass`; the
//! graph executor in [`graph_exec.rs`](graph_exec.rs) dispatches
//! `PassId::Shadow` here.

use ash::vk;
use concinnity_core::gfx::render_types;

use super::super::context::VkContext;
use crate::vulkan::depth;
use crate::vulkan::owned::VkDevice;
use crate::vulkan::record::cmd_push_constants;

// One depth-only shadow view to draw from a cull-written indirect buffer: the
// set 0 holding its `ShadowUniforms`, which of their `light_vps` the vertex
// shader projects through, and the buffer its commands live in. A cascade and a
// spot slice differ only in these.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct ShadowView {
    pub uniforms_set: vk::DescriptorSet,
    pub vp_index: u32,
    pub indirect: vk::Buffer,
}

impl VkContext {
    // Encode the cascaded-shadow-map render passes for frame slot
    // `frame_idx`: one render pass per cascade slice, drawing every
    // visible static / instanced / skinned caster into the array layer
    // for that cascade.
    //
    // A no-op when shadows are off (`shadow_map_size == 0`). The caller must
    // compute + upload `shadow_uniforms` before this runs so the shadow
    // vertex shader sees the current cascade VPs.
    pub(in crate::vulkan) fn encode_shadow_pass(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        cam_pos: [f32; 3],
        elapsed: f32,
        // This frame's grass, whose cascade blades draw into the nearest
        // cascade after its rasterized casters, when it casts this frame.
        grass: Option<&crate::vulkan::grass::GrassFrame>,
    ) {
        if !self.shadow.enabled() {
            return;
        }

        // Raymarched SDF shadow casters share these cascade DSVs: upload this
        // frame's animation time once (no-op without casters) so the from-light
        // SDF march lines up with the lit-side surface.
        self.upload_raymarch_shadow_view(frame_idx, elapsed);
        let device = self.hw.device.clone();
        let device = &device;

        // Cascades to re-render this frame; draw_frame computed the mask from the
        // update policy. A skipped cascade's render pass is omitted entirely, so
        // its slice keeps the depth + VP from when it was last rendered (the
        // graph-driven producer/consumer barriers still round-trip every layer,
        // preserving the contents). The 0 sentinel falls back to all cascades.
        let all_cascades = (1u32 << render_types::NUM_SHADOW_CASCADES) - 1;
        let render_mask = if self.shadow.render_mask == 0 {
            all_cascades
        } else {
            self.shadow.render_mask
        };

        // Nothing to cull means nothing to draw: the render passes below still
        // run, so every re-rendered cascade is cleared for the raymarched
        // casters that follow the rasterized ones.
        let gpu_driven = self.shadow_views_drawable();

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
            self.begin_shadow_slice(cmd, shadow_fb.handle(), self.shadow.map_size);

            if gpu_driven
                && let Some(indirect) = self
                    .cull
                    .shadow_indirect_buffers
                    .get(frame_idx)
                    .and_then(|c| c.get(cascade_idx))
            {
                self.draw_shadow_view(
                    device,
                    cmd,
                    frame_idx,
                    ShadowView {
                        uniforms_set: self.descriptors.shadow_global_sets[frame_idx],
                        vp_index: cascade_idx as u32,
                        indirect: indirect.buffer(),
                    },
                );
            }

            if cascade_idx == 0
                && let Some(frame) = grass
            {
                self.encode_grass_shadow_draw(cmd, frame_idx, frame);
            }

            // Raymarched SDF shadow casters into this cascade's DSV, after the
            // rasterized casters and within the same render pass (no re-clear);
            // the depth write test keeps the nearer occluder.
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

    // Whether the GPU-driven shadow path has a pipeline and records to draw.
    pub(in crate::vulkan) fn shadow_views_drawable(&self) -> bool {
        self.cull.shadow_bindless_pipeline.is_some() && self.cull_count() > 0
    }

    // Open the depth-only shadow render pass on one square `size` slice, which
    // clears it, with the viewport and scissor over it.
    pub(in crate::vulkan) fn begin_shadow_slice(
        &self,
        cmd: vk::CommandBuffer,
        framebuffer: vk::Framebuffer,
        size: u32,
    ) {
        let device = &self.hw.device;
        let extent = vk::Extent2D {
            width: size,
            height: size,
        };
        let clear_depth = depth::CLEAR_VALUE;
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.shadow.render_pass.handle())
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D::default().extent(extent))
            .clear_values(std::slice::from_ref(&clear_depth));
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            // Negative-height viewport: Y-flips NDC so Y-up matches Metal, and
            // matches the `-ndc.y` the forward sampler applies.
            let vp = vk::Viewport {
                x: 0.0,
                y: size as f32,
                width: size as f32,
                height: -(size as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
            let scissor = vk::Rect2D::default().extent(extent);
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
        }
    }

    // GPU-driven body of one shadow view (inside the render pass the caller
    // opened): the depth-only bindless pipeline issues the view's static +
    // instance prefix and the skinned tail with two `cmd_draw_indexed_indirect`
    // calls over its cull-written indirect buffer, projected through
    // `light_vps[view.vp_index]`. The CPU never walks the caster lists.
    pub(in crate::vulkan) fn draw_shadow_view(
        &self,
        device: &VkDevice,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        view: ShadowView,
    ) {
        let (Some(sb_pipeline), Some(sb_layout)) = (
            self.cull.shadow_bindless_pipeline.as_ref(),
            self.cull.shadow_bindless_pipeline_layout.as_ref(),
        ) else {
            return;
        };
        let stride = std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32;
        let prefix = self.skinned_record_base() as u32;

        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, sb_pipeline.handle());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                sb_layout.handle(),
                0,
                &[view.uniforms_set, self.cull.bindless_sets[frame_idx]],
                &[],
            );
            cmd_push_constants(
                device,
                cmd,
                sb_layout.handle(),
                vk::ShaderStageFlags::VERTEX,
                &view.vp_index,
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
                device.cmd_draw_indexed_indirect(cmd, view.indirect, 0, prefix, stride);
                self.inc_draw_calls(1);
            }

            // Skinned tail against the deformed VB + skinned IB.
            if self.state.draw.n_skinned > 0
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
                    view.indirect,
                    (self.skinned_record_base() * stride as usize) as u64,
                    self.state.draw.n_skinned as u32,
                    stride,
                );
                self.inc_draw_calls(1);
            }
        }
    }
}
