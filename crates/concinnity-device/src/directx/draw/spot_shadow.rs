//! Spot shadow pass: one depth-only render per shadow-casting spot light into
//! its slice of the spot shadow array. Structurally the cascade pass with a
//! different projection source -- each slice is a `ShadowView` drawn by the
//! same GPU-driven shadow pipeline, whose uniforms hold that spot's
//! light-space matrix in slot 0 rather than the CSM cascade set.
//!
//! Local lights are static, so the matrices are built once at init and only the
//! depth contents refresh here. `spot_shadow.render_mask` (from
//! `SpotShadowScheduler`) picks which slices redraw; a skipped slice keeps the
//! depth it last rendered, which stays correct until a caster moves.

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::frustum::Frustum;
use concinnity_engine::render::shadow::spot;
use windows::Win32::Graphics::Direct3D12::*;

use super::shadow::ShadowView;
use crate::directx::allocator::PooledBuffer;
use crate::directx::com;
use crate::directx::context::DxContext;
use crate::directx::descriptor_slot::SrvSlot;
use crate::directx::texture::GpuResource;

// Spot shadows: one depth slice per shadow-casting spot plus each slice's
// projection, fixed at init since local lights are static. Without one a 1x1
// fallback keeps the main pass's descriptors valid.
pub(in crate::directx) struct SpotShadowState {
    pub resource: Option<GpuResource<ID3D12Resource>>,
    // One DSV per shadowed spot; empty when the world has none.
    pub dsvs: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub srv_gpu: SrvSlot,
    // `SpotShadowData` per slice, uploaded once at init.
    pub buffer: PooledBuffer,
    // One `ShadowUniforms` per slice with its matrix in `light_vps[0]`, so the
    // shared shadow pipeline renders it. Written once at init.
    pub ubo: PooledBuffer,
    // 256-byte-aligned distance between consecutive slices in `ubo`.
    pub ubo_stride: u64,
    pub slice_size: u32,
    // Each slice's light frustum, which its GPU cull keeps casters inside.
    pub frusta: Vec<Frustum>,
    // Slices re-rendered this frame (bit `i` = slice `i`), as the frame's plan
    // scheduled them; read by encode_spot_shadow_pass.
    pub render_mask: u32,
}

impl SpotShadowState {
    // Slices actually handed out; the array, the DSV list, and the data buffer
    // all carry exactly this many entries.
    pub(crate) fn count(&self) -> u32 {
        u32::saturating_from(self.dsvs.len())
    }

    // Slices that re-render this frame.
    pub(crate) fn refreshed_slices(&self) -> impl Iterator<Item = u32> {
        spot::refreshed_slices(self.render_mask, self.count())
    }

    // GPU address of slice `slice`'s baked `ShadowUniforms`.
    pub(crate) fn slice_ubo_gva(&self, slice: u32) -> u64 {
        debug_assert!(slice < self.count());
        let base = com::gpu_va(&self.ubo);
        base + u64::from(slice) * self.ubo_stride
    }
}

impl DxContext {
    // One depth-only render per scheduled spot slice, culled against its
    // frustum like the cascades. Empty slices still clear.
    pub(in crate::directx) fn encode_spot_shadow_pass(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
    ) {
        if self.spot_shadow.count() == 0 {
            return;
        }
        Self::set_shadow_raster_state(cmd, self.spot_shadow.slice_size);
        for slice in self.spot_shadow.refreshed_slices() {
            Self::clear_shadow_slice(cmd, self.spot_shadow.dsvs[slice as usize]);
        }

        let Some(views) = self.cull.spot_views.get(frame_idx) else {
            return;
        };
        if !self.shadow_views_drawable() {
            return;
        }
        self.encode_spot_culls(cmd, frame_idx);
        self.bind_shadow_views(cmd, frame_idx, self.cull.shadow_bindless_pso.as_ref());
        for slice in self.spot_shadow.refreshed_slices() {
            self.draw_shadow_view(
                cmd,
                frame_idx,
                ShadowView {
                    dsv: self.spot_shadow.dsvs[slice as usize],
                    ubo_gva: self.spot_shadow.slice_ubo_gva(slice),
                    // Every spot slice carries its own matrix in `light_vps[0]`.
                    vp_index: 0,
                    lists: self.shadow_lists(views, slice as usize),
                },
            );
        }
    }
}
