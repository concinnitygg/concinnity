// The environment drawn as the background (see `concinnity_core::render::sky`):
// a fullscreen draw at the tail of each opaque scene pass (the main camera, a
// reflection-probe face, a planar mirror), on the targets that pass already has
// bound, and one at the tail of the G-buffer pre-pass for the sky's motion
// (`post/gbuffer_sky.rs`).
//
// Root signature:
//   [0] root CBV b0   ViewUniforms of the pass it completes
//   [1] table  t0     the environment's prefilter cube
// Static linear-clamp sampler s0, the cube sampler the main pass reads through.

use concinnity_core::gfx::view_modes::ViewMode;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::pass_timing;
use concinnity_core::render::render_graph::PassId;
use concinnity_core::render::sky;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_QUERY_TYPE_TIMESTAMP, ID3D12Device, ID3D12GraphicsCommandList, ID3D12PipelineState,
    ID3D12RootSignature,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT;

use super::builtin_shaders::{self, CompileProgram};
use super::context::DxContext;
use super::descriptor_slot::DescriptorTables;
use super::pso::{Blend, Depth, GraphicsPso, Raster};
use super::root_sig::{RootSig, SamplerState, Visibility};
use super::texture::HDR_FORMAT;

pub(in crate::directx) struct DxSky {
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    // The world draws its environment map as the background.
    background: bool,
}

impl DxSky {
    pub(in crate::directx) fn build(
        device: &ID3D12Device,
        msaa_samples: u32,
        background: bool,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let root_sig = RootSig::new()
            .cbv(0, Visibility::All)
            .srv_table(0, 1, Visibility::Pixel)
            .static_sampler(SamplerState::LinearClamp, 0, Visibility::Pixel)
            .build(device, "sky root sig")?;
        let pso = build_sky_pso(device, &root_sig, msaa_samples, hot_reload)?;
        Ok(Self {
            root_sig,
            pso,
            background,
        })
    }

    pub(in crate::directx) fn root_sig(&self) -> &ID3D12RootSignature {
        &self.root_sig
    }

    pub(in crate::directx) fn swap_pso(&mut self, pso: ID3D12PipelineState) {
        self.pso = pso;
    }
}

// The sky over the HDR color and the main depth, at the main pass's sample
// count: no blending, no input layout, and the inclusive read-only depth test
// that passes only where the depth still holds the clear.
pub(in crate::directx) fn build_sky_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    msaa_samples: u32,
    hot_reload: bool,
) -> RenderResult<ID3D12PipelineState> {
    let vs = builtin_shaders::SKY_VERT.compile(hot_reload)?;
    let ps = builtin_shaders::SKY_FRAG.compile(hot_reload)?;
    GraphicsPso::fullscreen(root_sig, &vs, &ps, HDR_FORMAT, Blend::Opaque)
        .depth(DXGI_FORMAT_D32_FLOAT, Depth::read_only())
        .samples(msaa_samples.max(1))
        .raster(Raster {
            multisample: msaa_samples > 1,
            ..Raster::default()
        })
        .build(device, "sky")
}

impl DxContext {
    // Whether a view rendered in `mode` draws the sky.
    pub(in crate::directx) fn draws_sky(&self, mode: ViewMode) -> bool {
        sky::draws_sky(
            self.scene.env_map.prefilter_mip_count > 0,
            self.sky.background,
            mode,
        )
    }

    // Draw the environment behind everything drawn so far into the bound HDR
    // color and depth, from the viewpoint the ViewUniforms at `view_gva`
    // describe. The shader-visible heaps must be set on `cmd`.
    pub(in crate::directx) fn encode_sky(&self, cmd: &ID3D12GraphicsCommandList, view_gva: u64) {
        // SAFETY: the command list is in the recording state with the CBV/SRV/UAV heap set, and the
        // root signature bound here declares the CBV and the one-entry table these commands set;
        // every resource and descriptor they name is live for the call.
        unsafe {
            cmd.SetPipelineState(&self.sky.pso);
            cmd.SetGraphicsRootSignature(&self.sky.root_sig);
            cmd.SetGraphicsRootConstantBufferView(0, view_gva);
            cmd.set_graphics_srv_table(1, self.scene.env_map.prefilter.srv_gpu);
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd.DrawInstanced(3, 1, 0, 0);
        }
        self.inc_draw_calls(1);
    }

    // The main camera's sky, bracketed by its own timestamps inside the main
    // pass's command list.
    pub(in crate::directx) fn encode_main_sky(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        view_gva: u64,
    ) {
        let (start, end) = pass_timing::pass_pair(frame_idx, PassId::Sky);
        let heap = self.timestamps.query_heap.as_ref();
        if let Some(heap) = heap {
            // SAFETY: the command list is in the recording state and the query heap is live.
            unsafe { cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, start) };
        }
        self.encode_sky(cmd, view_gva);
        if let Some(heap) = heap {
            // SAFETY: as above.
            unsafe { cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, end) };
        }
    }
}
