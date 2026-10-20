// The environment drawn as the background (see `concinnity_engine::render::environment::sky`):
// a fullscreen draw at the tail of each opaque scene pass (the main camera, a
// reflection-probe face, a planar mirror), on the targets that pass already has
// bound, and one at the tail of the G-buffer pre-pass for the sky's motion
// (`post/gbuffer_sky.rs`).
//
// Root signature:
//   [0] root CBV b0   ViewUniforms of the pass it completes
//   [1] table  t0     the environment's prefilter cube
// Static linear-clamp sampler s0, the cube sampler the main pass reads through.

use concinnity_engine::gfx::view_modes::ViewMode;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::render_graph::PassId;
use concinnity_engine::render::{backend::pass_timing, environment::sky};
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
        let pso = build_sky_pso(
            device,
            &builtin_shaders::SKY_FRAG,
            &root_sig,
            msaa_samples,
            hot_reload,
        )?;
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
    fragment: &concinnity_engine::render::shader::programs::ShaderProgram,
    root_sig: &ID3D12RootSignature,
    msaa_samples: u32,
    hot_reload: bool,
) -> RenderResult<ID3D12PipelineState> {
    let vs = builtin_shaders::SKY_VERT.compile(hot_reload)?;
    let ps = fragment.compile(hot_reload)?;
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
    // Whether a view rendered in `mode` draws the sky: the atmosphere's once it
    // has a frame's inputs, else the environment map's.
    pub(in crate::directx) fn draws_sky(&self, mode: ViewMode) -> bool {
        if self
            .atmosphere
            .as_ref()
            .is_some_and(|a| a.params.get().is_some())
        {
            return sky::draws_sky(true, true, mode);
        }
        sky::draws_sky(
            self.scene.env_map.prefilter_mip_count > 0,
            self.sky.background,
            mode,
        )
    }

    // Draw the environment behind everything drawn so far into the bound HDR
    // color and depth, from the viewpoint the ViewUniforms at `view_gva`
    // describe; `frame_idx` picks the atmosphere's constants. The
    // shader-visible heaps must be set on `cmd`.
    pub(in crate::directx) fn encode_sky(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        view_gva: u64,
        frame_idx: usize,
    ) {
        if self.encode_atmosphere_sky(cmd, view_gva, frame_idx) {
            return;
        }
        // SAFETY: the list is recording, and the sky PSO is owned by `self.sky`.
        unsafe { cmd.SetPipelineState(&self.sky.pso) };
        // SAFETY: the list is recording, and the sky root signature is owned by `self.sky`.
        unsafe { cmd.SetGraphicsRootSignature(&self.sky.root_sig) };
        // SAFETY: the sky signature declares the view CBV at 0; the caller's view block is live.
        unsafe { cmd.SetGraphicsRootConstantBufferView(0, view_gva) };
        // SAFETY: the sky signature declares a one-entry table at 1, the CBV/SRV/UAV heap is set,
        // and the prefiltered environment's descriptor is live.
        unsafe { cmd.set_graphics_srv_table(1, self.scene.env_map.prefilter.srv_gpu) };
        // SAFETY: the list is recording; setting the topology names no resource.
        unsafe { cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST) };
        // SAFETY: the pipeline and every binding the sky reads are set above.
        unsafe { cmd.DrawInstanced(3, 1, 0, 0) };
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
        self.encode_sky(cmd, view_gva, frame_idx);
        if let Some(heap) = heap {
            // SAFETY: as above.
            unsafe { cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, end) };
        }
    }
}
