//! The sky's motion in the G-buffer pre-pass: a fullscreen triangle behind the
//! geometry that writes the camera's rotation into the velocity target. It
//! reads only the pre-pass's view block, through its own root signature, so it
//! draws whether or not the world has cull records.
//!
//! Root signature: [0] root CBV b1, the pre-pass `GbView` (the DirectX register
//! `gbuffer_prepass.hlsl` gives it on every entry).

use concinnity_core::render::error::RenderResult;
use windows::Win32::Graphics::Direct3D12::{
    ID3D12Device, ID3D12GraphicsCommandList, ID3D12InfoQueue, ID3D12PipelineState,
    ID3D12RootSignature,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT;

use super::gbuffer::gbuffer_targets;
use crate::directx::builtin_shaders::{self, CompileProgram};
use crate::directx::context::dump_on_err;
use crate::directx::pso::{Depth, GraphicsPso};
use crate::directx::root_sig::{RootSig, Visibility};

pub(in crate::directx) struct GbufferSky {
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

impl GbufferSky {
    pub(in crate::directx) fn build(
        device: &ID3D12Device,
        info_queue: Option<&ID3D12InfoQueue>,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let root_sig = dump_on_err(
            info_queue,
            RootSig::new()
                .cbv(1, Visibility::Vertex)
                .build(device, "gbuffer sky root sig"),
        )?;
        let pso = Self::build_pso(device, &root_sig, info_queue, hot_reload)?;
        Ok(Self { root_sig, pso })
    }

    // Into the pre-pass's three targets, tested inclusively against its depth
    // without writing it, so it lands only where no geometry did.
    fn build_pso(
        device: &ID3D12Device,
        root_sig: &ID3D12RootSignature,
        info_queue: Option<&ID3D12InfoQueue>,
        hot_reload: bool,
    ) -> RenderResult<ID3D12PipelineState> {
        let vs = builtin_shaders::GBUFFER_SKY_VERT.compile(hot_reload)?;
        let ps = builtin_shaders::GBUFFER_PREPASS_FRAG_BINDLESS.compile(hot_reload)?;
        dump_on_err(
            info_queue,
            gbuffer_targets(GraphicsPso::new(root_sig, &vs, &ps))
                .depth(DXGI_FORMAT_D32_FLOAT, Depth::read_only())
                .build(device, "gbuffer sky"),
        )
    }

    // A pipeline state rebuilt from the current shader sources, for a hot reload.
    pub(in crate::directx) fn rebuild_pso(
        &self,
        device: &ID3D12Device,
        info_queue: Option<&ID3D12InfoQueue>,
    ) -> RenderResult<ID3D12PipelineState> {
        Self::build_pso(device, &self.root_sig, info_queue, true)
    }

    pub(in crate::directx) fn swap_pso(&mut self, pso: ID3D12PipelineState) {
        self.pso = pso;
    }

    // Draw the sky's motion over the bound pre-pass targets, reading the view
    // block at `view_gva`.
    pub(in crate::directx) fn encode(&self, cmd: &ID3D12GraphicsCommandList, view_gva: u64) {
        // SAFETY: the command list is in the recording state with the pre-pass targets bound, and
        // the root signature bound here declares the CBV at parameter 0; every resource these
        // commands name is live for the call.
        unsafe {
            cmd.SetPipelineState(&self.pso);
            cmd.SetGraphicsRootSignature(&self.root_sig);
            cmd.SetGraphicsRootConstantBufferView(0, view_gva);
            cmd.DrawInstanced(3, 1, 0, 0);
        }
    }
}
