//! Far shadow pass: one depth-only render of the large casters around the
//! camera into the far shadow's D16 map, on the frames its schedule redraws it.
//! GPU-driven like the cascades: the shadow kernel culls the far view against
//! its projection, keeping the casters whose bounds span enough of its texels
//! and the planet terrain at any size, and the view draws them through the
//! bindless shadow pipeline built against the far map's own render pass, with
//! this frame's one-matrix uniforms at set 0. A frame that keeps the map
//! records nothing, so the shaded passes read what it last drew.

use concinnity_engine::render::shadow::far::far_view_uniforms;

use super::shadow::ShadowView;
use super::upload_shadow_uniforms;
use crate::vulkan::context::VkContext;
use crate::vulkan::record::Recorder;

impl VkContext {
    // pub(in crate::vulkan) so the render-graph executor can dispatch it.
    pub(in crate::vulkan) fn encode_far_shadow_pass(&self, rec: &Recorder<'_>, frame_idx: usize) {
        let shadow = &self.state.planner.shadow;
        let Some(far) = self.shadow.far.as_ref() else {
            return;
        };
        if !shadow.far_draw {
            return;
        }
        let (Some(ubo), Some(&set)) = (far.ubos.get(frame_idx), far.sets.get(frame_idx)) else {
            return;
        };
        upload_shadow_uniforms(ubo, &far_view_uniforms(&shadow.uniforms.far));

        // Cull before opening the render pass: Vulkan allows no compute inside
        // one.
        let views = self.cull.far_views.get(frame_idx);
        let gpu_driven = self.shadow_views_drawable() && views.is_some();
        if gpu_driven {
            self.encode_far_cull(rec, frame_idx);
        }
        Self::begin_shadow_slice(rec, &far.render_pass, &far.framebuffer, far.size);
        if gpu_driven && let Some(view) = views.and_then(|v| v.first()) {
            self.draw_shadow_view(
                rec,
                frame_idx,
                ShadowView {
                    uniforms_set: set,
                    // The far projection sits in slot 0 of its uniforms.
                    vp_index: 0,
                    view,
                    pipeline: self.cull.far_shadow_bindless_pipeline.as_ref(),
                },
            );
        }
        rec.end_render_pass();
    }
}
