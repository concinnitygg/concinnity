//! Far shadow pass: one depth-only render of the large casters around the
//! camera into the far shadow's D16 map, on the frames its schedule redraws it.
//! GPU-driven like the cascades: the shadow kernel culls the far view against
//! its projection, keeping the casters whose bounds span enough of its texels
//! and the planet terrain at any size, into the view's own lists, which draw
//! through the D16 twin of the bindless shadow pipeline with this frame's
//! one-matrix uniforms. A frame that keeps the map records nothing, so the
//! shaded passes read what it last drew.

use concinnity_engine::render::shadow::far::far_view_uniforms;
use windows::Win32::Graphics::Direct3D12::*;

use super::shadow::ShadowView;
use crate::directx::context::DxContext;
use crate::directx::mapped_blocks::MappedBlocks;

// The far shadow's D16 map and its DSV, and the per-frame-in-flight
// one-matrix uniforms its view renders through.
pub(in crate::directx) struct FarShadowState {
    pub map: ID3D12Resource,
    pub dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub size: u32,
    pub uniforms: MappedBlocks,
}

impl DxContext {
    // pub(in crate::directx) so the render-graph executor can dispatch this pass.
    pub(in crate::directx) fn encode_far_shadow_pass(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
    ) {
        let shadow = &self.state.planner.shadow;
        let Some(far) = self.shadow.far.as_ref() else {
            return;
        };
        if !shadow.far_draw {
            return;
        }
        let Some(ubo_gva) = far
            .uniforms
            .write(frame_idx, &far_view_uniforms(&shadow.uniforms.far))
        else {
            return;
        };
        Self::set_shadow_raster_state(cmd, far.size);
        Self::clear_shadow_slice(cmd, far.dsv);

        let Some(views) = self.cull.far_views.get(frame_idx) else {
            return;
        };
        if !self.shadow_views_drawable() {
            return;
        }
        self.encode_far_cull(cmd, frame_idx);
        self.bind_shadow_views(cmd, frame_idx, self.cull.far_shadow_bindless_pso.as_ref());
        self.draw_shadow_view(
            cmd,
            frame_idx,
            ShadowView {
                dsv: far.dsv,
                ubo_gva,
                // The far projection sits in slot 0 of its uniforms.
                vp_index: 0,
                lists: self.shadow_lists(views, 0),
            },
        );
    }
}
