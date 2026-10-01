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

use concinnity_core::gfx::frustum::Frustum;
use concinnity_core::render::spot_shadow;
use windows::Win32::Graphics::Direct3D12::*;

use super::shadow::ShadowView;
use crate::directx::allocator::PooledBuffer;
use crate::directx::com;
use crate::directx::context::DxContext;
use crate::directx::descriptor_slot::SrvSlot;
use crate::directx::texture::GpuResource;

// Spot shadow map resources: one depth array slice per shadow-casting spot
// light, plus the `SpotShadowData` buffer holding each slice's light-space
// projection. Local lights are static, so the slice assignment and every matrix
// are decided once at init and only the depth contents refresh. A world with no
// shadowed spot still gets a 1x1 fallback array and a one-element buffer, so
// the main pass's descriptors are always valid.
pub(in crate::directx) struct SpotShadowState {
    pub resource: Option<GpuResource<ID3D12Resource>>,
    // One DSV per shadowed spot; empty when the world has none.
    pub dsvs: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub srv_gpu: SrvSlot,
    // `SpotShadowData` per slice, uploaded once at init.
    pub buffer: PooledBuffer,
    // One `ShadowUniforms` per slice, carrying that spot's matrix in
    // `light_vps[0]` so the shared shadow vertex shader can render a spot slice
    // without a second pipeline or a second uniform layout. Written once at
    // init: the projections are fixed for the world's lifetime, so unlike the
    // cascade UBO this needs no per-frame copy.
    pub ubo: PooledBuffer,
    // 256-byte-aligned distance between consecutive slices in `ubo`.
    pub ubo_stride: u64,
    pub slice_size: u32,
    // Each slice's light frustum, which its GPU cull keeps casters inside.
    pub frusta: Vec<Frustum>,
    // Round-robin clock + primed set, advanced once per frame in record_frame.
    pub scheduler: spot_shadow::SpotShadowScheduler,
    // Slices re-rendered this frame (bit `i` = slice `i`). Set in record_frame
    // and read by encode_spot_shadow_pass.
    pub render_mask: u32,
}

impl SpotShadowState {
    // Slices actually handed out; the array, the DSV list, and the data buffer
    // all carry exactly this many entries.
    pub(crate) fn count(&self) -> u32 {
        self.dsvs.len() as u32
    }

    // Advance the round-robin clock and record which slices re-render this
    // frame. A no-op (mask stays 0) when the world has no shadowed spot.
    pub(crate) fn advance(&mut self, every_frame: bool) {
        let count = self.dsvs.len();
        self.render_mask = self.scheduler.next_mask(every_frame, count);
    }

    // Slices that re-render this frame.
    pub(crate) fn refreshed_slices(&self) -> impl Iterator<Item = u32> {
        spot_shadow::refreshed_slices(self.render_mask, self.count())
    }

    // GPU address of slice `slice`'s baked `ShadowUniforms`.
    pub(crate) fn slice_ubo_gva(&self, slice: u32) -> u64 {
        debug_assert!(slice < self.count());
        let base = com::gpu_va(&self.ubo);
        base + slice as u64 * self.ubo_stride
    }
}

impl DxContext {
    // One depth-only render per scheduled spot slice, GPU-driven like the
    // cascades: a per-slice cull against the spot's light frustum writes the
    // slice's region of this frame's spot indirect buffer, and the slice draws
    // that region through the shared bindless shadow pipeline. Slices with no
    // records still clear, so the main pass samples valid depth.
    // pub(in crate::directx) so the render-graph executor can dispatch this pass.
    pub(in crate::directx) fn encode_spot_shadow_pass(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        cam_pos: [f32; 3],
    ) {
        if self.spot_shadow.count() == 0 {
            return;
        }
        self.set_shadow_raster_state(cmd, self.spot_shadow.slice_size);
        for slice in self.spot_shadow.refreshed_slices() {
            self.clear_shadow_slice(cmd, self.spot_shadow.dsvs[slice as usize]);
        }

        let Some(indirect) = self.cull.spot_indirect_buffers.get(frame_idx) else {
            return;
        };
        if !self.shadow_views_drawable() {
            return;
        }
        self.encode_spot_culls(cmd, frame_idx, cam_pos);
        self.bind_shadow_views(cmd, frame_idx);
        for slice in self.spot_shadow.refreshed_slices() {
            self.draw_shadow_view(
                cmd,
                frame_idx,
                ShadowView {
                    dsv: self.spot_shadow.dsvs[slice as usize],
                    ubo_gva: self.spot_shadow.slice_ubo_gva(slice),
                    // Every spot slice carries its own matrix in `light_vps[0]`.
                    vp_index: 0,
                    indirect,
                    first_command: slice as usize * self.cull.bucket_stride,
                },
            );
        }
    }
}
