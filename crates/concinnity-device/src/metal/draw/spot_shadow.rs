//! Spot shadow pass: one depth-only render per shadow-casting spot light into its
//! slice of `spot_shadow.map`. GPU-driven like the cascades: the shadow cull
//! fills one region of the spot ICB per refreshed slice against that spot's
//! light frustum, and each slice draws its region through the shared bindless
//! shadow pipeline (`draw_shadow_view`) with a one-matrix `ShadowUniforms`.
//!
//! Local lights are static, so the matrices are built once at init and only the
//! depth contents refresh here. `spot_shadow.render_mask` (from
//! `SpotShadowScheduler`) picks which slices redraw; a skipped slice keeps the
//! depth it last rendered.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::components;
use concinnity_core::gfx::render_types::{ShadowUniforms, SpotShadowData};
use concinnity_core::render::csm;
use concinnity_core::render::error::{RenderError, RenderResult};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer as _, MTLLoadAction, MTLRenderPassDescriptor, MTLStoreAction,
};

use super::shadow::ShadowView;
use crate::metal::context::MtlContext;
use crate::metal::depth::CLEAR_DEPTH;
use crate::metal::scoped_encoder::ScopedEncoder;

impl MtlContext {
    // Choose which spot shadow slices to re-render this frame and advance the
    // round-robin clock. Called once per frame from draw_frame; the result is
    // stashed in `spot_shadow.render_mask` for the cull and the pass.
    pub(in crate::metal) fn next_spot_shadow_mask(&mut self) -> u32 {
        let every_frame = matches!(
            self.shadow.cadence.update,
            components::ShadowUpdate::EveryFrame
        );
        self.spot_shadow
            .scheduler
            .next_mask(every_frame, self.spot_shadow.count as usize)
    }

    // pub(in crate::metal) so the render-graph executor can dispatch this pass.
    pub(in crate::metal) fn encode_spot_shadow_pass(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        // The per-frame `GpuObjectData` buffer the bindless shadow VS reads each
        // record's model from. `None` when nothing is in the cull records, which
        // leaves every refreshed slice a bare depth clear.
        object_buffer: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        // This frame's skinned deformed-vertex buffer, for the skinned tail.
        deformed_skinned: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
    ) -> RenderResult<u32> {
        if !self.shadow.enabled || self.spot_shadow.count == 0 {
            return Ok(0);
        }
        let first_rendered = self.spot_shadow.refreshed_slices().next();
        let last_rendered = self.spot_shadow.refreshed_slices().last();

        let mut total_draws: u32 = 0;
        for slice in self.spot_shadow.refreshed_slices() {
            let pass_desc = MTLRenderPassDescriptor::new();
            let depth_attach = pass_desc.depthAttachment();
            depth_attach.setTexture(Some(self.spot_shadow.map.as_ref()));
            depth_attach.setSlice(slice as usize);
            depth_attach.setLoadAction(MTLLoadAction::Clear);
            depth_attach.setStoreAction(MTLStoreAction::Store);
            depth_attach.setClearDepth(CLEAR_DEPTH);

            // Timing spans the first to the last slice actually rendered, the
            // same shape the cascade pass uses.
            if let Some(t) = &self.diagnostics.pass_timing {
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
                    .ok_or_else(|| {
                        RenderError::Other("failed to get spot shadow render encoder".to_string())
                    })?,
                ns_string!("spot shadow slice"),
            );

            if let Some(object_buffer) = object_buffer {
                let uniforms = self.spot_slice_uniforms(slice);
                total_draws += self.draw_shadow_view(
                    &enc,
                    ShadowView {
                        uniforms: &uniforms,
                        // A spot slice's matrix sits in slot 0 of its uniforms.
                        vp_index: 0,
                        set: &self.cull.spot_views,
                        region: slice as usize,
                    },
                    object_buffer,
                    deformed_skinned,
                );
            }
        }

        Ok(total_draws)
    }

    // A one-matrix `ShadowUniforms` carrying `slice`'s light-space projection in
    // slot 0, so the shared shadow vertex shader can render a spot slice without
    // a second pipeline or a second uniform layout.
    fn spot_slice_uniforms(&self, slice: u32) -> ShadowUniforms {
        let data = self.spot_shadow_data(slice);
        let mut uniforms = csm::empty_shadow_uniforms();
        uniforms.light_vps[0] = data.light_vp;
        uniforms.active_cascades = 1;
        uniforms
    }

    // Read slice `slice`'s projection back from the uploaded buffer. The buffer
    // is Shared storage and written once at init, so this is a plain read of
    // memory the GPU only ever reads.
    fn spot_shadow_data(&self, slice: u32) -> SpotShadowData {
        debug_assert!(slice < self.spot_shadow.count);
        // SAFETY: the buffer was created from a `&[SpotShadowData]` of exactly
        // `spot_shadow.count` elements and is never resized; `slice` is bounded
        // by that count above.
        unsafe {
            let base = self.spot_shadow.buffer.contents().as_ptr() as *const SpotShadowData;
            *base.add(slice as usize)
        }
    }
}
