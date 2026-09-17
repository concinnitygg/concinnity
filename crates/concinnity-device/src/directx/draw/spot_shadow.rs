//! Spot shadow pass: one depth-only render per shadow-casting spot light into
//! its slice of the spot shadow array. Structurally the cascade pass with a
//! different projection source -- each slice reuses the same depth-only shadow
//! pipeline and the same static / instanced / skinned caster sub-encoders,
//! driven by a `ShadowPassBinding` whose uniforms hold that spot's light-space
//! matrix in slot 0 rather than the CSM cascade set.
//!
//! Local lights are static, so the matrices are built once at init and only the
//! depth contents refresh here. `spot_shadow.render_mask` (from
//! `SpotShadowScheduler`) picks which slices redraw; a skipped slice keeps the
//! depth it last rendered, which stays correct until a caster moves.

use concinnity_core::render::spot_shadow;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;

use super::shadow::ShadowPassBinding;
use crate::directx::allocator::PooledBuffer;
use crate::directx::com;
use crate::directx::context::DxContext;
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
    pub srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
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

    // GPU address of slice `slice`'s baked `ShadowUniforms`.
    pub(crate) fn slice_ubo_gva(&self, slice: u32) -> u64 {
        debug_assert!(slice < self.count());
        let base = com::gpu_va(&self.ubo);
        base + slice as u64 * self.ubo_stride
    }
}

impl DxContext {
    // pub(in crate::directx) so the render-graph executor can dispatch this pass.
    pub(in crate::directx) fn encode_spot_shadow_pass(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        cam_pos: [f32; 3],
    ) {
        let (Some(shadow_pso), Some(shadow_root_sig)) =
            (self.shadow.pso.as_ref(), self.shadow.root_sig.as_ref())
        else {
            return;
        };
        let count = self.spot_shadow.count();
        if count == 0 {
            return;
        }

        let all = if count >= 32 {
            u32::MAX
        } else {
            (1u32 << count) - 1
        };
        // Defensive fallback to every slice if no mask was set this frame.
        let mask = if self.spot_shadow.render_mask == 0 {
            all
        } else {
            self.spot_shadow.render_mask
        };

        let sz = self.spot_shadow.slice_size;
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: sz as f32,
                Height: sz as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            cmd.RSSetViewports(&[vp]);
            let scissor = RECT {
                left: 0,
                top: 0,
                right: sz as i32,
                bottom: sz as i32,
            };
            cmd.RSSetScissorRects(&[scissor]);
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
        }

        for slice in 0..count {
            if mask & (1u32 << slice) == 0 {
                continue;
            }
            // Spot casters go through the per-draw CPU encoders
            // (`encode_shadow_casters_into` / `encode_shadow_skinned_into`): the
            // shadow ICB the bindless cull fills is laid out per CSM cascade, so it
            // has no slots for these slices.
            let ubo_gva = self.spot_shadow.slice_ubo_gva(slice);
            let dsv = self.spot_shadow.dsvs[slice as usize];
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                cmd.OMSetRenderTargets(0, None, false, Some(&dsv));
                cmd.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
            }
            self.encode_shadow_casters_into(
                cmd,
                ShadowPassBinding {
                    pso: shadow_pso,
                    root_sig: shadow_root_sig,
                    ubo_gva,
                    // The shadow VS indexes `light_vps` by this; the spot
                    // slice's matrix lives in slot 0.
                },
                cam_pos,
            );
            self.encode_shadow_skinned_into(cmd, ubo_gva, frame_idx, cam_pos);
        }
    }
}
