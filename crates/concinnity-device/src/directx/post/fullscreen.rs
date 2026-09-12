// src/directx/post/fullscreen.rs
//
// The PIXEL_SHADER_RESOURCE <-> RENDER_TARGET barrier bracket, render-target
// bind, and viewport / scissor that a fullscreen pass not yet drawn through the
// shared post seam (`render::post`) writes by hand: the reflection composite's
// blur and composite.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::context::DxContext;
use crate::directx::texture::transition_barrier;

// Pixel dimensions of the target a fullscreen pass writes, which are its
// viewport and scissor. The caller owns the target and so already knows them;
// reading them back off the resource would be a COM round trip per pass.
#[derive(Clone, Copy)]
pub(in crate::directx) struct FullscreenExtent {
    pub width: u32,
    pub height: u32,
}

impl DxContext {
    // Begin a fullscreen render-target pass: transition `output` from its sampled
    // resting state into RENDER_TARGET, bind it as the sole RTV, set the
    // viewport / scissor to `extent`, and bind the SRV heap the pass's root tables
    // index. Paired with `end_fullscreen_rt`.
    //
    // The viewport is the target size (not a fixed render resolution) so a pass
    // writing a reduced-resolution target -- the reflection blur -- rasterizes
    // the full fullscreen triangle across its smaller target.
    pub(in crate::directx) fn begin_fullscreen_rt(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        output: &ID3D12Resource,
        output_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        extent: FullscreenExtent,
    ) {
        let to_rt = transition_barrier(
            output,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe { cmd.ResourceBarrier(&[to_rt]) };
        self.bind_fullscreen_rt(cmd, output_rtv, extent);
    }

    // The bind half of `begin_fullscreen_rt`, without the transition: for a
    // target the render graph drives, which the executor has already put in
    // RENDER_TARGET before this pass's command list. Such a pass has no `end`
    // half either -- the next consumer's graph barrier takes the target back out.
    pub(in crate::directx) fn bind_fullscreen_rt(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        output_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        extent: FullscreenExtent,
    ) {
        let FullscreenExtent {
            width: w,
            height: h,
        } = extent;
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.OMSetRenderTargets(1, Some(&output_rtv), false, None);
            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: w as f32,
                Height: h as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            cmd.RSSetViewports(&[vp]);
            let scissor = RECT {
                left: 0,
                top: 0,
                right: w as i32,
                bottom: h as i32,
            };
            cmd.RSSetScissorRects(&[scissor]);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
        }
    }

    // End a fullscreen render-target pass: transition `output` back to its sampled
    // resting state for the downstream consumer. Paired with `begin_fullscreen_rt`.
    pub(in crate::directx) fn end_fullscreen_rt(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        output: &ID3D12Resource,
    ) {
        let to_psr = transition_barrier(
            output,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe { cmd.ResourceBarrier(&[to_psr]) };
    }
}
