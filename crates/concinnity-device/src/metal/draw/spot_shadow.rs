// src/metal/draw/spot_shadow.rs
//
// Spot shadow pass: one depth-only render per shadow-casting spot light into its
// slice of `spot_shadow_map`. Structurally the cascade pass with a different
// projection source -- each slice reuses the same depth-only shadow pipeline and
// the same static / instanced / skinned caster sub-encoders, driven by a
// `ShadowPassBinding` whose uniforms hold that spot's light-space matrix in slot
// 0 rather than the CSM cascade set.
//
// Local lights are static, so the matrices are built once at init and only the
// depth contents refresh here. `spot_shadow_render_mask` (from
// `SpotShadowScheduler`) picks which slices redraw; a skipped slice keeps the
// depth it last rendered, which stays correct until a caster moves.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer as _, MTLLoadAction, MTLRenderPassDescriptor, MTLStoreAction,
};

use crate::gfx::render_types::{ShadowPassPush, ShadowUniforms, SpotShadowData};
use crate::metal::context::MtlContext;
use crate::metal::scoped_encoder::ScopedEncoder;

use super::shadow::ShadowPassBinding;

impl MtlContext {
    // Choose which spot shadow slices to re-render this frame and advance the
    // round-robin clock. Called once per frame from draw_frame; the result is
    // stashed in `spot_shadow_render_mask` for encode_spot_shadow_pass.
    pub(in crate::metal) fn next_spot_shadow_mask(&mut self) -> u32 {
        let every_frame = matches!(self.shadow_update, crate::assets::ShadowUpdate::EveryFrame);
        self.spot_shadow_scheduler
            .next_mask(every_frame, self.spot_shadow_count as usize)
    }

    // pub(in crate::metal) so the render-graph executor can dispatch this pass.
    pub(in crate::metal) fn encode_spot_shadow_pass(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        skinned_joint_bufs: &[Retained<ProtocolObject<dyn MTLBuffer>>],
        cam_pos: [f32; 3],
    ) -> Result<u32, String> {
        let Some(shadow_pipeline) = self.shadow_pipeline_state.clone() else {
            return Ok(0);
        };
        if self.spot_shadow_count == 0 {
            return Ok(0);
        }

        let all = if self.spot_shadow_count >= 32 {
            u32::MAX
        } else {
            (1_u32 << self.spot_shadow_count) - 1
        };
        // Defensive fallback to every slice if no mask was set this frame.
        let mask = if self.spot_shadow_render_mask == 0 {
            all
        } else {
            self.spot_shadow_render_mask
        };
        let rendered: Vec<u32> = (0..self.spot_shadow_count)
            .filter(|i| mask & (1u32 << i) != 0)
            .collect();
        let first_rendered = rendered.first().copied();
        let last_rendered = rendered.last().copied();

        let mut total_draws: u32 = 0;
        for &slice in &rendered {
            let pass_desc = MTLRenderPassDescriptor::new();
            let depth_attach = pass_desc.depthAttachment();
            depth_attach.setTexture(Some(self.spot_shadow_map.as_ref()));
            depth_attach.setSlice(slice as usize);
            depth_attach.setLoadAction(MTLLoadAction::Clear);
            depth_attach.setStoreAction(MTLStoreAction::Store);
            depth_attach.setClearDepth(1.0);

            // Timing spans the first to the last slice actually rendered, the
            // same shape the cascade pass uses.
            if let Some(t) = &self.pass_timing {
                let id = super::super::pass_timing::PassId::SpotShadow;
                let is_first = Some(slice) == first_rendered;
                let is_last = Some(slice) == last_rendered;
                if is_first && is_last {
                    t.attach_render(&pass_desc, id);
                } else if is_first {
                    t.attach_render_first(&pass_desc, id);
                } else if is_last {
                    t.attach_render_last(&pass_desc, id);
                }
            }

            let enc = ScopedEncoder::new(
                cmd_buf
                    .renderCommandEncoderWithDescriptor(&pass_desc)
                    .ok_or("failed to get spot shadow render encoder")?,
                "spot shadow slice",
            );

            let uniforms = self.spot_slice_uniforms(slice);
            let bind = ShadowPassBinding {
                pipeline: &shadow_pipeline,
                uniforms: &uniforms,
                // The shadow VS indexes `light_vps` by this; the spot slice's
                // matrix lives in slot 0.
                push: ShadowPassPush {
                    cascade_idx: 0,
                    _pad: [0; 3],
                },
                slope_bias: 1.0,
            };

            // Spot casters go through the legacy CPU sub-encoders: the shadow
            // ICB the bindless cull fills is laid out per CSM cascade, so it has
            // no slots for these slices.
            total_draws += self.encode_shadow_static_into(&enc, &bind, cam_pos);
            total_draws += self.encode_shadow_instanced_into(&enc, &bind, cam_pos);
            total_draws +=
                self.encode_shadow_skinned_into(&enc, &bind, cam_pos, skinned_joint_bufs);
        }

        Ok(total_draws)
    }

    // A one-matrix `ShadowUniforms` carrying `slice`'s light-space projection in
    // slot 0, so the shared shadow vertex shader can render a spot slice without
    // a second pipeline or a second uniform layout.
    fn spot_slice_uniforms(&self, slice: u32) -> ShadowUniforms {
        let data = self.spot_shadow_data(slice);
        let mut uniforms = crate::gfx::csm::empty_shadow_uniforms();
        uniforms.light_vps[0] = data.light_vp;
        uniforms.active_cascades = 1;
        uniforms
    }

    // Read slice `slice`'s projection back from the uploaded buffer. The buffer
    // is Shared storage and written once at init, so this is a plain read of
    // memory the GPU only ever reads.
    fn spot_shadow_data(&self, slice: u32) -> SpotShadowData {
        debug_assert!(slice < self.spot_shadow_count);
        // SAFETY: the buffer was created from a `&[SpotShadowData]` of exactly
        // `spot_shadow_count` elements and is never resized; `slice` is bounded
        // by that count above.
        unsafe {
            let base = self.spot_shadow_buffer.contents().as_ptr() as *const SpotShadowData;
            *base.add(slice as usize)
        }
    }
}
