//! Shader-source helpers shared across every Metal pipeline builder plus the
//! two genuinely cross-effect pipelines: the text overlay and the post-process
//! composite. Per-effect pipeline builders (bloom, TAA, velocity, SSAO, SSR,
//! decal, fog, auto-exposure, cull) live next to their encoders in the
//! matching `post/*.rs` / `decal.rs` / `fog.rs` / `auto_exposure.rs` /
//! `cull.rs` files so each effect is a single unit.
#![deny(unsafe_op_in_unsafe_fn)]

use std::borrow::Cow;

use concinnity_core::render::error::{RenderError, RenderResult};
use dispatch2::DispatchData;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLDevice as _, MTLPixelFormat, MTLRenderPipelineDescriptor, MTLRenderPipelineState,
};

use crate::metal::descriptors::{VertexAttr, VertexLayout, vertex_descriptor};
use crate::metal::post::fullscreen::{FullscreenBlend, build_fullscreen_pipeline};

pub(super) fn ns_str(s: &str) -> Retained<NSString> {
    NSString::from_str(s)
}

// The hand-written cull encode kernel, the one built-in shader authored in MSL.
pub(super) const CULL_ENCODE: &str = "cull_encode.metal";

// The cull encode kernel's MSL. Under hot-reload the checkout's copy wins when
// it is readable, so a saved edit is picked up on the next build; otherwise, or
// when the read fails, the embedded copy.
pub(super) fn cull_encode_source(hot_reload: bool) -> Cow<'static, str> {
    if hot_reload {
        let path = format!(
            "{}/src/metal/shaders/{CULL_ENCODE}",
            env!("CARGO_MANIFEST_DIR")
        );
        match std::fs::read_to_string(&path) {
            Ok(s) => return Cow::Owned(s),
            Err(e) => {
                tracing::debug!(
                    "hot-reload: falling back to embedded source for {CULL_ENCODE} ({e})"
                );
            }
        }
    }
    Cow::Borrowed(include_str!("shaders/cull_encode.metal"))
}

// The cull encode kernel's MTLLibrary, from the metallib the build script
// embedded when its source is unedited, else a cached or fresh compile.
pub(super) fn cull_encode_library(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn objc2_metal::MTLLibrary>>> {
    let msl = cull_encode_source(hot_reload);
    super::msl_cache::compiled_library(
        device,
        &msl,
        CULL_ENCODE,
        super::metallib::embedded_metallib(CULL_ENCODE),
    )
}

// The world Shader's function for `entry`: the cook's MSL text, or a compile
// of the current templates when it predates them. Each entry is its own
// translation, so a pipeline takes its two stages from two of these.
pub(super) fn world_function(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
    programs: &concinnity_core::components::ShaderPrograms,
    entry: &str,
) -> RenderResult<Retained<ProtocolObject<dyn objc2_metal::MTLFunction>>> {
    let req = crate::shader::surface_source::Request {
        platform: concinnity_core::platform::Platform::Metal,
        hot_reload,
    };
    let msl = crate::shader::surface_source::artifact(
        programs,
        entry,
        &req,
        crate::shader::compile::cooked,
    )?;
    super::msl_cache::cooked_function(device, &msl, entry, &format!("world shader {entry}"))
}

// Load a MTLLibrary from raw .metallib bytes via a DispatchData.
pub(super) fn load_library(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    bytes: &[u8],
) -> RenderResult<Retained<ProtocolObject<dyn objc2_metal::MTLLibrary>>> {
    let data = DispatchData::from_bytes(bytes);
    device
        .newLibraryWithData_error(&data)
        .map_err(|e| RenderError::ShaderCompile(format!("{e:?}")))
}

// Build the text overlay render pipeline from the single-source `text.hlsl`
// pair. Renders screen-space quads with alpha blending and no depth test.
pub(super) fn build_text_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    swap_pixel_format: MTLPixelFormat,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    use objc2_metal::{MTLBlendFactor, MTLVertexFormat, MTLVertexStepFunction};

    // Each entry compiles to its own metallib, so the two stages come from
    // separate libraries and pair by semantic.
    let vert_fn = crate::metal::builtin_shaders::entry_function(
        device,
        &crate::metal::builtin_shaders::TEXT_VERT,
        hot_reload,
    )?;
    let frag_fn = crate::metal::builtin_shaders::entry_function(
        device,
        &crate::metal::builtin_shaders::TEXT_FRAG,
        hot_reload,
    )?;

    // Vertex layout: pos (float2) @ 0, uv (float2) @ 8, color (float3) @ 16,
    // mode (float) @ 28; buffer(1). Mirrors TextVertex in render_types.rs.
    let vert_desc = vertex_descriptor(
        &[
            VertexAttr {
                index: 0,
                format: MTLVertexFormat::Float2,
                offset: 0,
                buffer_index: 1,
            },
            VertexAttr {
                index: 1,
                format: MTLVertexFormat::Float2,
                offset: 8,
                buffer_index: 1,
            },
            VertexAttr {
                index: 2,
                format: MTLVertexFormat::Float3,
                offset: 16,
                buffer_index: 1,
            },
            VertexAttr {
                index: 3,
                format: MTLVertexFormat::Float,
                offset: 28,
                buffer_index: 1,
            },
        ],
        &[VertexLayout {
            buffer_index: 1,
            stride: 32,
            step: MTLVertexStepFunction::PerVertex,
        }],
    );

    let pipeline_desc = MTLRenderPipelineDescriptor::new();
    pipeline_desc.setVertexDescriptor(Some(&vert_desc));
    pipeline_desc.setVertexFunction(Some(&vert_fn));
    pipeline_desc.setFragmentFunction(Some(&frag_fn));
    pipeline_desc.setRasterSampleCount(1);
    // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
    // declares.
    unsafe {
        let ca = pipeline_desc.colorAttachments().objectAtIndexedSubscript(0);
        // The composite pass already chose the swapchain format (BGRA8Unorm
        // for SDR; RGBA16Float for HDR EDR output): match it so text quads
        // can be drawn straight into the drawable in either mode.
        ca.setPixelFormat(swap_pixel_format);
        // Standard premultiplied-alpha blend so text sits on the tonemapped image.
        ca.setBlendingEnabled(true);
        ca.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
        ca.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        ca.setSourceAlphaBlendFactor(MTLBlendFactor::One);
        ca.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    }

    device
        .newRenderPipelineStateWithDescriptor_error(&pipeline_desc)
        .map_err(|e| RenderError::ShaderCompile(format!("text pipeline state: {e:?}")))
}

// Build the post-process pipeline: a fullscreen triangle that samples the
// resolved HDR target, applies ACES (Narkowicz fit) tonemap + gamma 2.2
// encode (SDR) or passes the exposed HDR scene through linearly (HDR EDR
// output), then runs FXAA + ColorLut grading on the SDR path. Renders into
// the drawable's single-sample swapchain attachment (`BGRA8Unorm` for SDR,
// `RGBA16Float` for HDR EDR).
pub(super) fn build_post_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    swap_pixel_format: MTLPixelFormat,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    // Single color attachment matches the swapchain format chosen by
    // `configure_mtk_view` (`BGRA8Unorm` for SDR, `RGBA16Float` for HDR EDR).
    build_fullscreen_pipeline(
        device,
        &super::builtin_shaders::COMPOSITE_FRAG,
        swap_pixel_format,
        FullscreenBlend::Replace,
        hot_reload,
    )
}

#[cfg(test)]
mod tests {
    use super::cull_encode_source;

    #[test]
    fn embedded_cull_encode_holds_its_kernel() {
        assert!(cull_encode_source(false).contains("kernel void cull_encode("));
    }

    #[test]
    fn hot_reload_reads_the_checkout_copy() {
        assert!(cull_encode_source(true).contains("kernel void cull_encode("));
    }
}
