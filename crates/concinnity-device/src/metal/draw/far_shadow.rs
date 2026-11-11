//! Far shadow pass: one depth-only render of the large casters around the
//! camera into the far shadow's `Depth16Unorm` map, on the frames its schedule
//! redraws it. GPU-driven like the cascades: the shadow cull packs the far
//! view's casters, those whose bounds span enough of its texels and the planet
//! terrain at any size, into its own ICB lists, which draw through the
//! `Depth16Unorm` twin of the bindless shadow pipeline with a one-matrix
//! `ShadowUniforms`. A frame that keeps the map
//! encodes nothing, so the shaded passes read what it last drew.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::shadow::bias;
use concinnity_engine::render::shadow::far::far_view_uniforms;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer as _, MTLLoadAction, MTLRenderPassDescriptor, MTLStoreAction,
};

use super::shadow::{ShadowTarget, ShadowView};
use crate::metal::context::MtlContext;
use crate::metal::depth::CLEAR_DEPTH;
use crate::metal::scoped_encoder::ScopedEncoder;

impl MtlContext {
    // pub(in crate::metal) so the render-graph executor can dispatch this pass.
    pub(in crate::metal) fn encode_far_shadow_pass(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        // The per-frame `GpuObjectData` buffer the bindless shadow VS reads each
        // record's model from. `None` when nothing is in the cull records, which
        // leaves the redrawn map a bare depth clear.
        object_buffer: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        // This frame's skinned deformed-vertex buffer, for the skinned tail.
        deformed_skinned: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
    ) -> RenderResult<u32> {
        let shadow = &self.state.planner.shadow;
        let Some(far) = self.shadow.far.as_ref() else {
            return Ok(0);
        };
        if !shadow.far_draw {
            return Ok(0);
        }

        let pass_desc = MTLRenderPassDescriptor::new();
        let depth_attach = pass_desc.depthAttachment();
        depth_attach.setTexture(Some(far.map.as_ref()));
        depth_attach.setLoadAction(MTLLoadAction::Clear);
        depth_attach.setStoreAction(MTLStoreAction::Store);
        depth_attach.setClearDepth(CLEAR_DEPTH);
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_render(&pass_desc, super::super::pass_timing::PassId::FarShadow);
        }

        let enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&pass_desc)
                .ok_or_else(|| {
                    RenderError::Other("failed to get far shadow render encoder".to_string())
                })?,
            ns_string!("far shadow"),
        );
        let Some(object_buffer) = object_buffer else {
            return Ok(0);
        };
        let uniforms = far_view_uniforms(&shadow.uniforms.far);
        Ok(self.draw_shadow_view(
            &enc,
            ShadowView {
                uniforms: &uniforms,
                // The far projection sits in slot 0 of its uniforms.
                vp_index: 0,
                set: &self.cull.far_view,
                region: 0,
                target: ShadowTarget {
                    pipeline: self.cull.far_shadow_bindless_pipeline.as_deref(),
                    raster_clamp: bias::FAR_RASTER_CLAMP,
                },
            },
            object_buffer,
            deformed_skinned,
        ))
    }
}
