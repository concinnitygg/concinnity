//! Cross-cutting D3D12 pipeline helpers shared by every pass:
//!   * Vertex input layouts referenced by main + shadow + velocity + SSAO
//!     pre-pass + text pipelines (`main_input_layout`, `text_input_layout`).
//!   * The text overlay pipeline (`create_text_root_signature`,
//!     `create_text_pso`) and the composite (post-process) pipeline
//!     (`create_composite_root_signature`, `create_composite_pso`).
//!
//! Mirrors src/metal/pipeline.rs (shared helpers + text + composite).
//! Per-effect pipelines live in their
//! own files: bloom/TAA/SSAO in directx/post/, cull at directx/cull.rs,
//! main + shadow in directx/init/pipelines.rs.

use concinnity_engine::gfx::render_types;
use concinnity_engine::render::backend::error::RenderResult;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::directx::builtin_shaders::CompileProgram;
use crate::directx::pso::{Blend, GraphicsPso};
use crate::directx::root_sig::{RootSig, SamplerState, Visibility};

// Shared vertex input layouts
//
// Used by main + shadow + velocity + SSAO pre-pass + text pipelines. Kept
// here because multiple per-effect pipelines reference them.

// Vertex input elements for the main pass (56-byte Vertex struct).
pub(super) fn main_input_layout() -> Vec<D3D12_INPUT_ELEMENT_DESC> {
    // SAFETY: the PSTR literals live for 'static; this is standard D3D12 usage.
    vec![
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("POSITION"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 0,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("NORMAL"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 12,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("TANGENT"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 24,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("COLOR"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 36,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("TEXCOORD"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 48,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
    ]
}

// Vertex input elements for the text pass (32-byte TextVertex struct), asserted
// by `text_vertex_layout_matches_shaders`.
//
// `mode` takes a semantic of its own rather than a second TEXCOORD, so every
// element below matches the semantic `text.hlsl` spells at index 0.
fn text_input_layout() -> Vec<D3D12_INPUT_ELEMENT_DESC> {
    vec![
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("POSITION"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 0,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("TEXCOORD"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 8,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("COLOR"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 16,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("MODE"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 28,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
    ]
}

// Composite (post-process) pipeline
//
// A vertex-buffer-less fullscreen triangle samples the off-screen FP16 HDR
// scene target, composites the bloom top octave, applies an exposure multiplier, the
// Narkowicz ACES tonemap + gamma 2.2 encode, a single FXAA 3.11-style edge
// pass, a 3D-LUT color grade, and a radial vignette, then writes the
// swapchain backbuffer. Ships from `src/render/shader/hlsl/composite.hlsl`, paired with
// the shared single-source fullscreen-triangle vertex every post pass uses.

// Compile the composite (post-process) pass shaders. Returns (vs, ps).
pub(super) fn compile_composite_shaders(hot_reload: bool) -> RenderResult<(Vec<u8>, Vec<u8>)> {
    let vs = super::builtin_shaders::FULLSCREEN_VERT.compile(hot_reload)?;
    let ps = super::builtin_shaders::COMPOSITE_FRAG.compile(hot_reload)?;
    Ok((vs, ps))
}

// Root signature for the composite pass: a 1-SRV descriptor table for the scene
// target (the HDR resolve, or the TAA output when TAA is on), one for the bloom
// top octave, `CompositeParams` as 32-bit root constants, one for the 3D
// color-grading LUT, one each for the G-buffer normal+depth, roughness, SSAO and
// motion channels and the reactive mask the debug view modes visualize, and a
// static linear-clamp sampler per source, each the sampler half of a source's
// texture/sampler pair in the single source. The
// scene SRV is its own table (separate from the bloom top octave) so the runtime can
// re-point it at the per-frame TAA output without the two needing to be
// heap-contiguous. Clamp keeps the FXAA neighbor taps from wrapping at screen
// edges and the LUT taps inside the cube.
pub(super) fn create_composite_root_signature(
    device: &ID3D12Device,
) -> RenderResult<ID3D12RootSignature> {
    use concinnity_engine::render::shader::programs::abi::composite::dx as composite;
    // [0] the scene, [1] the bloom top octave.
    let sig = RootSig::new()
        .srv_table(composite::HDR, 1, Visibility::Pixel)
        .srv_table(composite::BLOOM, 1, Visibility::Pixel)
        // [2] CompositeParams (the 9 PostProcessParams tunables, the
        // scene-transition fade, the view mode and the depth view's range). The
        // count must cover the whole struct: constants past `Num32BitValues`
        // read as zero in the shader, which silently disabled the `fxaa` flag
        // while this was 8.
        .constants::<render_types::CompositeParams>(composite::PARAMS, Visibility::Pixel)
        // [3] The 3D color-grading LUT is a separate, non-contiguous heap
        // slot, so it needs its own table.
        .srv_table(composite::LUT, 1, Visibility::Pixel);
    // [4..7] The G-buffer channel sources the debug view modes visualize
    // (normal + depth, roughness, the blurred SSAO occlusion, motion). Each is
    // a separate non-contiguous heap slot, so each needs its own table. The
    // fragment references all four statically, so they are bound every frame
    // (the SSAO white 1x1 stands in when no G-buffer was built).
    // [8] The reactive mask, the reactive view's source.
    let sig = [
        composite::GBUFFER_NORMAL_DEPTH,
        composite::GBUFFER_ROUGHNESS,
        composite::AO,
        composite::GBUFFER_MOTION,
        composite::REACTIVE,
    ]
    .into_iter()
    .fold(sig, |sig, reg| sig.srv_table(reg, 1, Visibility::Pixel));
    // One sampler per source. Identical descriptors; the split is the
    // shader's, not the pass's.
    [
        composite::HDR_SAMPLER,
        composite::BLOOM_SAMPLER,
        composite::LUT_SAMPLER,
        composite::GBUFFER_NORMAL_DEPTH_SAMPLER,
        composite::GBUFFER_ROUGHNESS_SAMPLER,
        composite::AO_SAMPLER,
        composite::GBUFFER_MOTION_SAMPLER,
        composite::REACTIVE_SAMPLER,
    ]
    .into_iter()
    .fold(sig, |sig, reg| {
        sig.static_sampler(SamplerState::LinearClamp, reg, Visibility::Pixel)
    })
    .build(device, "composite root sig")
}

// PSO for the composite pass: a vertex-buffer-less fullscreen triangle that
// samples the HDR scene target and writes the single-sample swapchain
// backbuffer. No input layout, no depth.
pub(super) fn create_composite_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
    rtv_format: DXGI_FORMAT,
) -> RenderResult<ID3D12PipelineState> {
    GraphicsPso::fullscreen(root_sig, vs, ps, rtv_format, Blend::Opaque).build(device, "composite")
}

// Text overlay pipeline
//
// Drawn after the composite into the single-sample swapchain backbuffer with
// straight alpha-blending. Per-call vertex + index buffers are uploaded
// dynamically by `encode_composite_and_text`.

// Compile the text overlay shaders.
pub(super) fn compile_text_shaders(hot_reload: bool) -> RenderResult<(Vec<u8>, Vec<u8>)> {
    let text_vs = super::builtin_shaders::TEXT_VERT.compile(hot_reload)?;
    let text_ps = super::builtin_shaders::TEXT_FRAG.compile(hot_reload)?;
    Ok((text_vs, text_ps))
}

pub(super) fn create_text_root_signature(
    device: &ID3D12Device,
) -> RenderResult<ID3D12RootSignature> {
    use concinnity_engine::render::shader::programs::abi::text::dx as text;
    RootSig::new()
        .constants::<render_types::TextUniforms>(text::UNIFORMS, Visibility::Vertex) // [0]
        .srv_table(text::ATLAS, 1, Visibility::Pixel) // [1]
        .sampler_table(text::ATLAS_SAMPLER, 1, Visibility::Pixel) // [2]
        .input_layout()
        .build(device, "text root sig")
}

pub(super) fn create_text_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
    rtv_format: DXGI_FORMAT,
    sample_count: u32,
) -> RenderResult<ID3D12PipelineState> {
    let layout = text_input_layout();
    GraphicsPso::new(root_sig, vs, ps)
        .input_layout(&layout)
        .target(rtv_format, Blend::AlphaOver)
        .samples(sample_count)
        .build(device, "text")
}
