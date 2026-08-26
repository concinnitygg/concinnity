// src/metal/pipeline.rs
//
// Shader-source helpers shared across every Metal pipeline builder plus the
// two genuinely cross-effect pipelines: the text overlay and the post-process
// composite. Per-effect pipeline builders (bloom, TAA, velocity, SSAO, SSR,
// decal, fog, auto-exposure, cull) live next to their encoders in the
// matching `post/*.rs` / `decal.rs` / `fog.rs` / `auto_exposure.rs` /
// `cull.rs` files so each effect is a single unit.
#![deny(unsafe_op_in_unsafe_fn)]

use dispatch2::DispatchData;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLDevice as _, MTLPixelFormat, MTLRenderPipelineDescriptor, MTLRenderPipelineState,
};

use crate::metal::descriptors::{VertexAttr, VertexLayout, vertex_descriptor};
use crate::metal::post::fullscreen::{FullscreenBlend, build_slang_fullscreen_pipeline};

pub(super) fn ns_str(s: &str) -> Retained<NSString> {
    NSString::from_str(s)
}

// Resolve the MSL source for one of the built-in renderer shaders. With
// `hot_reload` off this is just the `include_str!`-baked source -- same byte
// stream the binary has always compiled. With `hot_reload` on (set by
// `cn debug` via the `hot_reload` flag on `BackendInit`) the helper first tries
// `<CARGO_MANIFEST_DIR>/src/metal/shaders/<name>` so a saved edit to the
// `.metal` file in this checkout is picked up on the next call; if the disk
// read fails (binary moved, file removed, IO error) it transparently falls
// back to the embedded source. The embedded fallback means a shipped binary
// keeps working no matter where it is run from.
//
// Returning `Cow` keeps the no-hot-reload case allocation-free.
//
// Panics on an unregistered `name`. Every caller passes a compile-time string
// literal, so an unknown name is strictly a registration bug (a new
// `shaders/*.metal` file that was never added to the match below) -- never a
// runtime condition. Failing loudly here pins the blame at the source; the old
// silent `""` fall-through instead "compiled" an empty library and surfaced as
// a baffling `<entry-point> not found in metallib` at pipeline build. The
// registration is required even with `hot_reload` on -- the disk read is keyed
// off the same `name`, so an unregistered shader is never loaded from disk
// either. Locked by `unknown_name_panics` /
// `unknown_name_panics_even_with_hot_reload`.
// The shared bindless per-object record, substituted into every pass that
// strides the per-frame object buffer at its `{OBJECT_DATA}` marker. Sole MSL
// declaration of `GpuObjectData`: the legacy per-draw main pass and the cull
// kernel read the same buffer the single-source passes do, so a per-shader copy
// is a silent layout-drift hazard. `newLibraryWithSource` resolves no include paths, hence
// the marker; the build script's precompile substitutes the same fragment
// before handing each shader to `xcrun metal`.
const OBJECT_COMMON_MSL: &str = include_str!("shaders/object_common.msl");

// Resolve the shared fragment the same disk-first way as its consumer, so a
// hot-reload edit to the record reaches every pass that strides the buffer.
fn object_common(hot_reload: bool) -> std::borrow::Cow<'static, str> {
    if hot_reload {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/metal/shaders/object_common.msl"
        );
        match std::fs::read_to_string(path) {
            Ok(s) => return std::borrow::Cow::Owned(s),
            Err(e) => {
                tracing::debug!("hot-reload: falling back to embedded object_common.msl ({e})");
            }
        }
    }
    std::borrow::Cow::Borrowed(OBJECT_COMMON_MSL)
}

pub(super) fn shader_source(hot_reload: bool, name: &str) -> std::borrow::Cow<'static, str> {
    let embedded: &'static str = match name {
        "cull.metal" => include_str!("shaders/cull.metal"),
        "main.metal" => include_str!("shaders/main.metal"),
        _ => panic!(
            "shader_source: '{name}' is not a registered Metal shader. Add an \
             `include_str!(\"shaders/{name}\")` arm to shader_source in \
             metal/pipeline.rs -- every shipped shader must be registered."
        ),
    };
    let src = if hot_reload {
        let path = format!("{}/src/metal/shaders/{}", env!("CARGO_MANIFEST_DIR"), name);
        match std::fs::read_to_string(&path) {
            Ok(s) => std::borrow::Cow::Owned(s),
            Err(e) => {
                tracing::debug!(
                    "hot-reload: falling back to embedded source for {} ({})",
                    name,
                    e
                );
                std::borrow::Cow::Borrowed(embedded)
            }
        }
    } else {
        std::borrow::Cow::Borrowed(embedded)
    };
    if src.contains("{OBJECT_DATA}") {
        return std::borrow::Cow::Owned(src.replace("{OBJECT_DATA}", &object_common(hot_reload)));
    }
    src
}

// Produce the MTLLibrary for a built-in renderer shader. The fast path loads
// the metallib precompiled by the build script; source compilation remains for
// hot-reload (disk edits must win) and for binaries built without the Metal
// toolchain, whose embedded lookup is empty.
pub(super) fn shader_library(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
    name: &str,
) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLLibrary>>, String> {
    if !hot_reload && let Some(bytes) = crate::metal::metallib::embedded_metallib(name) {
        return load_library(device, bytes)
            .map_err(|e| format!("{name}: failed to load precompiled metallib: {e}"));
    }
    let msl = shader_source(hot_reload, name);
    let options = objc2_metal::MTLCompileOptions::new();
    device
        .newLibraryWithSource_options_error(&ns_str(msl.as_ref()), Some(&options))
        .map_err(|e| format!("{name}: shader compile error: {e:?}"))
}

// The library a main-pass stage renders with: a world-authored Shader supplies
// its own compiled metallib, and an empty slice means the world declared none,
// so the engine's own `main.metal` program runs instead.
pub(super) fn stage_library(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
    bytes: &[u8],
) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLLibrary>>, String> {
    if bytes.is_empty() {
        return shader_library(device, hot_reload, "main.metal");
    }
    load_library(device, bytes)
}

// Load a MTLLibrary from raw .metallib bytes via a DispatchData.
pub(super) fn load_library(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    bytes: &[u8],
) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLLibrary>>, String> {
    let data = DispatchData::from_bytes(bytes);
    device
        .newLibraryWithData_error(&data)
        .map_err(|e| format!("{:?}", e))
}

// Build the text overlay render pipeline from the single-source `text.slang`
// pair. Renders screen-space quads with alpha blending and no depth test.
pub(super) fn build_text_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    swap_pixel_format: MTLPixelFormat,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    use objc2_metal::{MTLBlendFactor, MTLVertexFormat, MTLVertexStepFunction};

    // Each entry compiles to its own metallib, so the two stages come from
    // separate libraries and pair by semantic.
    let vert_fn = crate::metal::slang_shaders::entry_function(
        device,
        &crate::metal::slang_shaders::TEXT_VERT,
        hot_reload,
    )?;
    let frag_fn = crate::metal::slang_shaders::entry_function(
        device,
        &crate::metal::slang_shaders::TEXT_FRAG,
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
        .map_err(|e| format!("failed to create text pipeline state: {:?}", e))
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
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    // Single colour attachment matches the swapchain format chosen by
    // `configure_mtk_view` (`BGRA8Unorm` for SDR, `RGBA16Float` for HDR EDR).
    build_slang_fullscreen_pipeline(
        device,
        &super::slang_shaders::COMPOSITE_FRAG,
        swap_pixel_format,
        FullscreenBlend::Replace,
        hot_reload,
    )
}

#[cfg(test)]
mod shader_source_tests {
    use super::shader_source;

    #[test]
    fn embedded_path_splices_object_data() {
        // Both registered shaders carry the marker, so the embedded source is
        // spliced rather than handed back verbatim.
        let s = shader_source(false, "main.metal");
        assert!(s.contains("vertex VertexOut vertex_main("));
        assert!(!s.contains("{OBJECT_DATA}"));
    }

    #[test]
    #[should_panic(expected = "not a registered Metal shader")]
    fn unknown_name_panics() {
        // An unregistered shader name is a registration bug, not a runtime
        // condition -- the loader hard-errors instead of silently returning an
        // empty source that "compiles" to an empty library.
        let _ = shader_source(false, "nope.metal");
    }

    #[test]
    #[should_panic(expected = "not a registered Metal shader")]
    fn unknown_name_panics_even_with_hot_reload() {
        // Registration is required even with hot-reload on: the disk read is
        // keyed off the same `name`, so an unregistered shader is never loaded
        // from disk either.
        let _ = shader_source(true, "nope.metal");
    }

    #[test]
    fn hot_reload_prefers_disk_when_present() {
        // The shader files live in this checkout, so the disk-load path
        // succeeds and produces the same content (or a newer edit).
        let s = shader_source(true, "main.metal");
        assert!(s.contains("vertex VertexOut vertex_main("));
    }

    #[test]
    fn main_metal_reflection_cut_matches_canonical() {
        // main.metal is precompiled to a metallib at build time, so it keeps
        // its own `constant float REFL_RESOLVE_CUT` instead of the
        // runtime-injected shared constant the resolve shaders use. Lock it to
        // the canonical value so the forward double-count fade can never drift
        // from the SSR / RT resolve gates. Expects a clean `= <value>;` decl.
        let src = shader_source(false, "main.metal");
        let decl = src
            .lines()
            .find(|l| l.contains("constant float REFL_RESOLVE_CUT"))
            .expect("REFL_RESOLVE_CUT declaration in main.metal");
        let value: f32 = decl
            .split(';')
            .next()
            .and_then(|head| head.split('=').nth(1))
            .map(str::trim)
            .and_then(|s| s.parse().ok())
            .expect("parse REFL_RESOLVE_CUT value from main.metal");
        assert_eq!(
            value,
            concinnity_core::gfx::ssr::REFLECTION_ROUGHNESS_CUT,
            "main.metal REFL_RESOLVE_CUT must equal REFLECTION_ROUGHNESS_CUT"
        );
    }

    // Every pass that strides the per-frame object buffer resolves its marker
    // to the shared record, in both hot-reload modes -- the embedded fragment
    // and the disk read must agree, since a `cn debug` session compiles from
    // source while a shipped binary loads the build-time metallib.
    #[test]
    fn object_data_shaders_splice_the_shared_record() {
        for name in ["main.metal", "cull.metal"] {
            for hot_reload in [false, true] {
                let src = shader_source(hot_reload, name);
                assert!(
                    src.contains("struct GpuObjectData"),
                    "{name}: object record missing (hot_reload = {hot_reload})"
                );
                assert!(
                    !src.contains("{OBJECT_DATA}"),
                    "{name}: left the OBJECT_DATA marker (hot_reload = {hot_reload})"
                );
            }
        }
    }

    #[test]
    fn shipped_shaders_are_registered() {
        // Every `.metal` under src/metal/shaders/ must resolve to non-empty
        // source through `shader_source` in BOTH hot-reload modes -- i.e. it is
        // registered in the match (an unregistered name now panics) and, with
        // hot_reload on, readable from disk. This is the guard that would have
        // caught the unregistered `gbuffer_prepass.metal` at test time instead
        // of as a baffling `<entry> not found in metallib` at init.
        //
        // The raymarch SDF templates/helpers are deliberately excluded: they
        // are not standalone libraries loaded by name but text fragments
        // assembled with the user's `SdfVolume` source at runtime (see
        // metal/raymarch.rs, which `include_str!`s them directly). They never
        // pass through `shader_source`, so registering them would be wrong.
        const ASSEMBLED_ELSEWHERE: &[&str] = &[
            "raymarch_helpers.metal",
            "raymarch_shadow.metal",
            "raymarch_template.metal",
            "raymarch_volumetric_template.metal",
        ];

        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/metal/shaders");
        let mut checked = 0usize;
        for entry in std::fs::read_dir(dir).expect("read shaders dir") {
            let file_name = entry.expect("dir entry").file_name();
            let name = file_name.to_str().expect("utf8 shader filename");
            if !name.ends_with(".metal") || ASSEMBLED_ELSEWHERE.contains(&name) {
                continue;
            }
            // Both arms must return non-empty. An unregistered name panics here
            // (with the missing-arm message), which is the failure we want.
            assert!(
                !shader_source(false, name).trim().is_empty(),
                "{name}: shader_source(false) returned empty source",
            );
            assert!(
                !shader_source(true, name).trim().is_empty(),
                "{name}: shader_source(true) returned empty source",
            );
            checked += 1;
        }
        assert!(checked > 0, "no .metal shaders found under {dir}");
    }
}
