// src/metal/raymarch.rs
//
// Per-frame encoder for the raymarched SDF volume pass. Runs at
// `PassId::Raymarch`, between `AutoExposure` and `Decals` on the
// hdr_resolve RMW chain. Each `SdfVolume` rasterises the back faces of
// its world-space bounding box and runs a user-authored fragment
// shader that sphere-traces the SDF inside the box.
//
// Architecture:
//   * One MTLRenderPipelineState per `SdfVolume` (built lazily at init
//     from the engine-shipped helpers + the user's source bytes + the
//     engine-shipped template). The wrap order is helpers → user →
//     template so the template's `fragment_main` can call the user's
//     `map` and `shade` functions through the forward declarations the
//     helpers expose.
//   * One shared unit-cube VB+IB for the proxy geometry; 8 corners /
//     36 indices, allocated once at init. The encoder draws back faces
//     only (cull mode = Front) so we get exactly one fragment per pixel
//     inside the box regardless of whether the camera is outside or
//     inside it.
//   * Color attachment = `hdr_resolve` (LoadAction::Load, opaque write).
//     No depth attachment, matching the projected-decal pass. Depth
//     compositing is shader-side via the early-out against
//     `main_depth` (texture(0)).

#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLBlitCommandEncoder as _, MTLBuffer,
    MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCullMode, MTLDevice, MTLIndexType,
    MTLLibrary, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder as _,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLRenderPipelineState,
    MTLResourceOptions, MTLStoreAction, MTLVertexFormat, MTLVertexStepFunction,
};

use concinnity_core::components::sdf_programs::SdfPrograms;
use concinnity_core::platform::Platform;
use concinnity_core::render::slang_programs::raymarch::{self, Family};
use concinnity_slang::SlangTarget;

use crate::components::sdf_volume::SdfVolume;
use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::LightUniforms;

use super::context::MtlContext;
use super::descriptors::{VertexAttr, VertexLayout, vertex_descriptor};
use super::encode::RenderEncode;
use super::pipeline::ns_str;
use super::scoped_encoder::ScopedEncoder;
// One declaration for all three backends, in `core::render::uniforms`.
// Re-exported at `pub(in crate::metal)` so the graph executor, the shadow pass
// and this file keep their existing paths.
pub(in crate::metal) use concinnity_core::render::uniforms::{
    RaymarchShadowCascade, RaymarchView, RaymarchVolumeUniforms,
};

// Metal buffer index for the proxy cube's vertex stream.
//
// Vertex streams and uniform buffers share one index space on Metal, and every
// entry point compiled from the single source receives every global the file
// declares -- so the vertex stage sees the light and cascade blocks it never
// reads, at their own slots. The stream therefore sits past all of them. The
// hand-written MSL could use a low index because its vertex stage declared only
// the two buffers it read.
const RAYMARCH_VERTEX_BUFFER: usize = 5;

// `RaymarchLights` mirror of `crate::gfx::render_types::LightUniforms`.
// The Rust struct already has the right layout; we just hand the
// buffer over to the shader at buffer(2). Kept as a type alias so the
// raymarch encoder can reference it without re-defining the layout.
type RaymarchLightsGpu = LightUniforms;

// Per-`SdfVolume` GPU state: the compiled render pipeline (one PSO per
// volume) plus the static per-volume uniforms.
pub(in crate::metal) struct RaymarchVolumeRecord {
    // The volume's draw pipeline. Compiled as the opaque surface variant
    // (cone-marched SDF, depth write) for a normal volume, or the
    // alpha-blended volumetric variant (Beer-Lambert march, no depth write)
    // when the asset's `volumetric` flag is set. A volume is one or the
    // other, never both: a volumetric shader provides `sampleVolume`
    // instead of `map`/`shade`, so the surface template would not link
    // against it. Mirrors DirectX's single per-volume `pso`.
    pub(in crate::metal) pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    // Depth-only shadow-caster pipeline. `Some` exactly when the asset's
    // `cast_shadows` is set; the shadow pass draws this volume into each CSM
    // cascade when it is `Some` AND `visible` AND `cast_shadows`.
    pub(in crate::metal) shadow_pipeline:
        Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub(in crate::metal) uniforms: RaymarchVolumeUniforms,
    pub(in crate::metal) visible: bool,
    // Whether this volume is volumetric (participating medium). Mirrors the
    // asset flag; the draw loop reads it to bind the read-only depth state
    // (no write) instead of the write-on state. The PSO variant is already
    // baked into `pipeline` at build time.
    pub(in crate::metal) volumetric: bool,
    // Whether this volume casts SDF shadows into the CSM cascades. Mirrors the
    // asset flag; paired with `shadow_pipeline` so the shadow encoder can skip
    // non-casters without inspecting the pipeline option.
    pub(in crate::metal) cast_shadows: bool,
    // Whether the authored field samples the scene behind the surface. The
    // pass copies `hdr_resolve` into `hdr_resolve_copy` only when some volume
    // drawing this frame does; the binding at fragment texture(4) stands
    // either way, so this selects no pipeline variant.
    pub(in crate::metal) refractive: bool,
    // Asset-side AABB centre / half-widths. `encode_raymarch` derives the
    // world-space AABB from these to frustum-cull the volume each frame.
    pub(in crate::metal) world_centre: [f32; 3],
    pub(in crate::metal) world_extent: [f32; 3],
}

// True when a volume at `centre` with half-widths `extent` is not entirely
// outside the camera frustum. Factored out of the draw loop so the cull
// predicate can be unit-tested without a GPU-backed `RaymarchVolumeRecord`.
pub(in crate::metal) fn volume_in_frustum(
    centre: [f32; 3],
    extent: [f32; 3],
    frustum: &crate::gfx::frustum::Frustum,
) -> bool {
    let min = [
        centre[0] - extent[0],
        centre[1] - extent[1],
        centre[2] - extent[2],
    ];
    let max = [
        centre[0] + extent[0],
        centre[1] + extent[1],
        centre[2] + extent[2],
    ];
    frustum.intersects_aabb(min, max)
}

// The MTLLibrary for one family of a volume's field.
//
// slangc emits one MSL translation unit per family, so both stages come out of
// one library and the cook stores it as one artifact. `msl_cache` turns that
// text into a metallib where a Metal toolchain exists and hands it to
// `newLibraryWithSource` where none does, which is every player machine.
fn family_library(
    device: &ProtocolObject<dyn MTLDevice>,
    programs: &SdfPrograms,
    family: Family,
    hot_reload: bool,
    asset_label: &str,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
    let entries: Vec<&str> = raymarch::ALL
        .iter()
        .filter(|p| p.family == family)
        .map(|p| p.entry)
        .collect();
    let msl = crate::raymarch_source::artifact(
        programs,
        &crate::raymarch_source::Request {
            family,
            platform: Platform::Metal,
            entries: &entries,
            target: SlangTarget::Metal,
            hot_reload,
            label: asset_label,
        },
    )?;
    let text = std::str::from_utf8(&msl)
        .map_err(|e| format!("SdfVolume '{asset_label}': compiled field is not MSL text: {e}"))?;
    super::msl_cache::compiled_library(device, text, asset_label)
        .map_err(|e| format!("raymarch shader compile error for SdfVolume '{asset_label}': {e}"))
}

// One entry point out of a family's library, named so a missing one points at
// the volume rather than at the engine.
fn entry_function(
    library: &ProtocolObject<dyn MTLLibrary>,
    entry: &str,
    asset_label: &str,
) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLFunction>>, String> {
    library.newFunctionWithName(&ns_str(entry)).ok_or_else(|| {
        format!("{entry} entry not found in compiled library for SdfVolume '{asset_label}'")
    })
}

// Compile + link a per-volume raymarch pipeline. Wraps the user
// fragment source bytes between the engine-shipped helpers and the
// engine-shipped fragment_main template, then compiles with
// `newLibraryWithSource_options_error` (same path the water / fog /
// decal / particle passes use for their built-in MSL).
//
// `asset_label` is included in error messages so a malformed user
// shader points at the right SdfVolume in the world.jsonl.
pub(in crate::metal) fn build_raymarch_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    programs: &SdfPrograms,
    hot_reload: bool,
    asset_label: &str,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let library = family_library(device, programs, Family::Surface, hot_reload, asset_label)?;
    let vert_fn = entry_function(&library, "raymarch_vertex", asset_label)?;
    let frag_fn = entry_function(&library, "raymarch_fragment", asset_label)?;

    // The proxy cube is stored as the engine's 56-byte `Vertex`, but the
    // shared vertex entry reads position alone, so the descriptor declares that
    // one attribute and strides over the rest.
    let vert_desc = vertex_descriptor(
        &[VertexAttr {
            index: 0,
            format: MTLVertexFormat::Float3,
            offset: 0,
            buffer_index: RAYMARCH_VERTEX_BUFFER,
        }],
        &[VertexLayout {
            buffer_index: RAYMARCH_VERTEX_BUFFER,
            stride: std::mem::size_of::<Vertex>(),
            step: MTLVertexStepFunction::PerVertex,
        }],
    );

    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexDescriptor(Some(&vert_desc));
    desc.setVertexFunction(Some(&vert_fn));
    desc.setFragmentFunction(Some(&frag_fn));
    desc.setRasterSampleCount(1);
    // SAFETY: plain descriptor property setters; attachment 0 is the only colour attachment this
    // pipeline declares.
    unsafe {
        let ca = desc.colorAttachments().objectAtIndexedSubscript(0);
        ca.setPixelFormat(MTLPixelFormat::RGBA16Float);
        // This variant writes opaque colour (the surface is opaque);
        // blending off keeps the per-pixel cost low. Volumetric
        // raymarching is the case that needs blend.
        ca.setBlendingEnabled(false);
    }
    // The fragment writes `[[depth(less)]]` into the bound
    // `depth_resolve` attachment so downstream passes that sample it
    // see raymarched-surface depth. Pipeline must declare the same
    // depth format the attachment uses.
    desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);

    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| {
            format!(
                "failed to create raymarch pipeline state for SdfVolume '{}': {:?}",
                asset_label, e
            )
        })
}

// Compile a per-volume depth-only shadow-caster pipeline. Wraps the user
// source between the helpers and the shadow template (the main template is
// *not* included: this library defines only `raymarch_shadow_vertex` /
// `raymarch_shadow_fragment`), then builds a render pipeline with no colour
// attachment and a `Depth32Float` depth attachment matching the CSM
// `shadow_map`. Mirrors `directx/raymarch.rs::compile_volume_shadow_pso`.
pub(in crate::metal) fn build_raymarch_shadow_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    programs: &SdfPrograms,
    hot_reload: bool,
    asset_label: &str,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let library = family_library(device, programs, Family::Shadow, hot_reload, asset_label)?;
    let vert_fn = entry_function(&library, "raymarch_shadow_vertex", asset_label)?;
    let frag_fn = entry_function(&library, "raymarch_shadow_fragment", asset_label)?;

    // Same proxy-cube vertex layout as the main pass.
    let vert_desc = vertex_descriptor(
        &[VertexAttr {
            index: 0,
            format: MTLVertexFormat::Float3,
            offset: 0,
            buffer_index: RAYMARCH_VERTEX_BUFFER,
        }],
        &[VertexLayout {
            buffer_index: RAYMARCH_VERTEX_BUFFER,
            stride: std::mem::size_of::<Vertex>(),
            step: MTLVertexStepFunction::PerVertex,
        }],
    );

    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexDescriptor(Some(&vert_desc));
    desc.setVertexFunction(Some(&vert_fn));
    desc.setFragmentFunction(Some(&frag_fn));
    // Shadow map is single-sample; no colour attachment is bound in the
    // shadow pass (depth-only). Only the depth format is declared.
    desc.setRasterSampleCount(1);
    desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);

    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| {
            format!(
                "failed to create raymarch shadow pipeline state for SdfVolume '{}': {:?}",
                asset_label, e
            )
        })
}

// Compile a per-volume volumetric raymarching pipeline: alpha blended, no depth
// write. The field provides `sampleVolume(p, params, time)` returning density,
// scattering and emission rather than `map` and `shade`.
pub(in crate::metal) fn build_raymarch_volumetric_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    programs: &SdfPrograms,
    hot_reload: bool,
    asset_label: &str,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let library = family_library(
        device,
        programs,
        Family::Volumetric,
        hot_reload,
        asset_label,
    )?;
    let vert_fn = entry_function(&library, "raymarch_volumetric_vertex", asset_label)?;
    let frag_fn = entry_function(&library, "raymarch_volumetric_fragment", asset_label)?;

    // Same proxy-cube vertex layout as the main pass.
    let vert_desc = vertex_descriptor(
        &[VertexAttr {
            index: 0,
            format: MTLVertexFormat::Float3,
            offset: 0,
            buffer_index: RAYMARCH_VERTEX_BUFFER,
        }],
        &[VertexLayout {
            buffer_index: RAYMARCH_VERTEX_BUFFER,
            stride: std::mem::size_of::<Vertex>(),
            step: MTLVertexStepFunction::PerVertex,
        }],
    );

    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexDescriptor(Some(&vert_desc));
    desc.setVertexFunction(Some(&vert_fn));
    desc.setFragmentFunction(Some(&frag_fn));
    desc.setRasterSampleCount(1);
    // SAFETY: plain descriptor property setters; attachment 0 is the only colour attachment this
    // pipeline declares.
    unsafe {
        let ca = desc.colorAttachments().objectAtIndexedSubscript(0);
        ca.setPixelFormat(MTLPixelFormat::RGBA16Float);
        // Volumetrics alpha-blend over the scene: output is translucent
        // so we use SRC_ALPHA / ONE_MINUS_SRC_ALPHA at the blend level.
        ca.setBlendingEnabled(true);
        ca.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
        ca.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        ca.setRgbBlendOperation(MTLBlendOperation::Add);
        ca.setSourceAlphaBlendFactor(MTLBlendFactor::One);
        ca.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        ca.setAlphaBlendOperation(MTLBlendOperation::Add);
    }
    // No depth write for volumetrics: they're translucent and don't update depth.
    // Depth test is still enabled to early-out against rasterised geometry.
    desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);

    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| {
            format!(
                "failed to create raymarch volumetric pipeline state for SdfVolume '{}': {:?}",
                asset_label, e
            )
        })
}

// Build the per-volume record (PSO + per-volume uniforms) from one declared
// `SdfVolume` and the compiled field the build packed into its payload.
pub(in crate::metal) fn build_raymarch_volume_record(
    device: &ProtocolObject<dyn MTLDevice>,
    volume: &SdfVolume,
    payload: &[u8],
    hot_reload: bool,
    asset_label: &str,
) -> Result<RaymarchVolumeRecord, String> {
    let programs = crate::raymarch_source::decode(payload, asset_label)?;
    // A medium is integrated rather than surfaced, so it builds the blended
    // pipeline and nothing else: its field defines `sampleVolume` and no
    // `map`, which the surface entries would fail to link against.
    let pipeline = if volume.volumetric {
        build_raymarch_volumetric_pipeline(device, &programs, hot_reload, asset_label)?
    } else {
        build_raymarch_pipeline(device, &programs, hot_reload, asset_label)?
    };
    let shadow_pipeline = if volume.cast_shadows {
        Some(build_raymarch_shadow_pipeline(
            device,
            &programs,
            hot_reload,
            asset_label,
        )?)
    } else {
        None
    };
    Ok(RaymarchVolumeRecord {
        pipeline,
        shadow_pipeline,
        uniforms: volume_uniforms_from(volume),
        visible: volume.visible,
        volumetric: volume.volumetric,
        cast_shadows: volume.cast_shadows,
        refractive: crate::raymarch_source::taps_scene(&programs),
        world_centre: volume.centre,
        world_extent: volume.extent,
    })
}

fn volume_uniforms_from(volume: &SdfVolume) -> RaymarchVolumeUniforms {
    RaymarchVolumeUniforms {
        centre: volume.centre,
        _pad0: 0.0,
        extent: volume.extent,
        _pad1: 0.0,
        cone_ratio: volume.cone_ratio(),
        max_distance: volume.max_distance,
        max_steps: volume.max_steps as i32,
        receive_shadows: if volume.receive_shadows { 1 } else { 0 },
        params: volume.params,
    }
}

// Build the shared unit-cube proxy geometry buffers. The vertices ride
// in `Vertex` shape so the proxy works through the engine's standard
// vertex descriptor (the same five-attribute layout the main pass and
// every custom mesh shader expect). 8 corners in `[-0.5, 0.5]^3`; the
// vertex shader scales by `vol.extent` and translates by `vol.centre`.
// Indices wind 36 CCW triangles (the encoder culls front faces so the
// rasteriser only fires for back faces).
type RaymarchCubeBuffers = (
    Retained<ProtocolObject<dyn MTLBuffer>>,
    Retained<ProtocolObject<dyn MTLBuffer>>,
);

pub(in crate::metal) fn build_raymarch_cube_buffers(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<RaymarchCubeBuffers, String> {
    // `extent` in SdfVolume is the AABB half-widths: the box spans
    // `centre ± extent`. The vertex shader computes `pos * extent +
    // centre`, so the proxy corners must be at `±1.0` for the scaled
    // corners to land at `centre ± extent`.
    #[rustfmt::skip]
    let corners: [Vertex; 8] = [
        v([-1.0, -1.0, -1.0]),
        v([ 1.0, -1.0, -1.0]),
        v([ 1.0,  1.0, -1.0]),
        v([-1.0,  1.0, -1.0]),
        v([-1.0, -1.0,  1.0]),
        v([ 1.0, -1.0,  1.0]),
        v([ 1.0,  1.0,  1.0]),
        v([-1.0,  1.0,  1.0]),
    ];
    // 36 CCW indices (outward winding when viewed from +x / +y / +z
    // halfspaces). Front-face cull will render back faces only at
    // encode time.
    #[rustfmt::skip]
    let indices: [u16; 36] = [
        // -Z
        0, 2, 1,  0, 3, 2,
        // +Z
        4, 5, 6,  4, 6, 7,
        // -X
        0, 4, 7,  0, 7, 3,
        // +X
        1, 2, 6,  1, 6, 5,
        // -Y
        0, 1, 5,  0, 5, 4,
        // +Y
        3, 7, 6,  3, 6, 2,
    ];

    let vb_bytes = std::mem::size_of_val(&corners);
    let ib_bytes = std::mem::size_of_val(&indices);

    // SAFETY: `ptr`/`vb_bytes` describe the live `corners` array, and Metal copies those bytes into
    // the new buffer before the call returns.
    let vb = unsafe {
        let ptr = std::ptr::NonNull::new(corners.as_ptr() as *mut _)
            .ok_or("raymarch cube vertex pointer null")?;
        device
            .newBufferWithBytes_length_options(ptr, vb_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or("failed to allocate raymarch cube vertex buffer")?
    };
    // SAFETY: as above -- `ptr`/`ib_bytes` describe the live `indices` array.
    let ib = unsafe {
        let ptr = std::ptr::NonNull::new(indices.as_ptr() as *mut _)
            .ok_or("raymarch cube index pointer null")?;
        device
            .newBufferWithBytes_length_options(ptr, ib_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or("failed to allocate raymarch cube index buffer")?
    };
    Ok((vb, ib))
}

fn v(pos: [f32; 3]) -> Vertex {
    Vertex {
        pos,
        normal: [0.0, 0.0, 0.0],
        tangent: [0.0, 0.0, 0.0],
        color: [0.0, 0.0, 0.0],
        uv: [0.0, 0.0],
    }
}

impl MtlContext {
    // Encode the raymarched SDF volume pass. Caller has ended the main
    // pass (so `hdr_targets.depth` carries scene depth) and the
    // post-Main hdr_resolve writes from Decals / Fog / ParticlesDraw
    // have not yet fired. Each visible `SdfVolume` issues one indexed
    // draw of the proxy cube; the user's `map` + `shade` run per
    // fragment.
    pub(in crate::metal) fn encode_raymarch(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        view: &RaymarchView,
        frustum: &crate::gfx::frustum::Frustum,
    ) -> Result<u32, String> {
        if self.raymarch.volumes.is_empty() {
            return Ok(0);
        }
        // Frustum-cull each volume's world-space AABB up front. A volume that
        // is hidden or entirely off-screen costs nothing this frame -- not even
        // the scene-copy blit + render encoder below. Mirrors the decal /
        // particle passes' pre-pass visibility mask.
        let visible: Vec<bool> = self
            .raymarch
            .volumes
            .iter()
            .map(|v| v.visible && volume_in_frustum(v.world_centre, v.world_extent, frustum))
            .collect();
        if !visible.iter().any(|&v| v) {
            return Ok(0);
        }
        let vbuf = self
            .raymarch
            .cube_vertex_buffer
            .as_ref()
            .ok_or("raymarch cube vertex buffer missing")?;
        let ibuf = self
            .raymarch
            .cube_index_buffer
            .as_ref()
            .ok_or("raymarch cube index buffer missing")?;
        let depth_sampler = self.post_sampler.as_ref();

        let lights_gpu: RaymarchLightsGpu = self.light_uniforms;
        let shadow_uniforms = self.shadow.uniforms;

        // Refraction support: snapshot the pre-raymarch
        // `hdr_resolve` into `hdr_resolve_copy` so user SDF shaders can
        // sample the scene below the surface without violating Metal's
        // attachment-aliasing rule (the same `hdr_resolve` we'd want to
        // read is also the colour attachment we're about to write).
        // Single full-screen blit per frame; AutoExposure has already
        // sampled `hdr_resolve_v1`, so this captures the same un-
        // decorated scene the next post-Main pass starts with.
        //
        // Only a volume whose field calls the scene tap reads the result, so a
        // frame drawing none skips the blit entirely and leaves the copy
        // holding whatever an earlier frame put there. Nothing samples it.
        if self
            .raymarch
            .volumes
            .iter()
            .zip(&visible)
            .any(|(v, &vis)| vis && v.refractive)
        {
            let blit = cmd_buf
                .blitCommandEncoder()
                .ok_or("failed to get raymarch scene-copy blit encoder")?;
            blit.pushDebugGroup(&NSString::from_str("raymarch_scene_copy"));
            // SAFETY: both textures are `hdr_targets`-owned and were created with the same format
            // and dimensions, which is what a whole-texture blit copy requires.
            unsafe {
                blit.copyFromTexture_toTexture(
                    self.hdr_targets.hdr_resolve.as_ref(),
                    self.hdr_targets.hdr_resolve_copy.as_ref(),
                );
            }
            blit.popDebugGroup();
            blit.endEncoding();
        }

        let pass_desc = MTLRenderPassDescriptor::new();
        // SAFETY: plain descriptor property setters; attachment 0 is the only colour attachment
        // this pass declares, and every texture set is owned by `self`.
        unsafe {
            let ca = pass_desc.colorAttachments().objectAtIndexedSubscript(0);
            ca.setTexture(Some(self.hdr_targets.hdr_resolve.as_ref()));
            ca.setLoadAction(MTLLoadAction::Load);
            ca.setStoreAction(MTLStoreAction::Store);
            // Bind the single-sample depth resolve as the
            // writable depth attachment. `Load` keeps the rasterised
            // depth that the Main pass resolved into it, so the
            // hardware depth test rejects raymarched fragments behind
            // existing geometry (and behind earlier raymarch volumes
            // in this pass). `Store` keeps the new depth (the min of
            // rasterised and raymarched per pixel) alive for
            // water / decal / fog to consume.
            let da = pass_desc.depthAttachment();
            da.setTexture(Some(self.hdr_targets.depth_resolve.as_ref()));
            da.setLoadAction(MTLLoadAction::Load);
            da.setStoreAction(MTLStoreAction::Store);
        }
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_render(&pass_desc, super::pass_timing::PassId::Raymarch);
        }
        // Any blit above is ended explicitly; this render encoder spans to the
        // end of the function, so the guard ends it on drop.
        let enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&pass_desc)
                .ok_or("failed to get raymarch render encoder")?,
            "raymarch",
        );
        // Front-face cull so each pixel inside the box receives exactly
        // one fragment shader invocation regardless of whether the
        // camera is outside or inside the bounding box. (Outside →
        // back faces visible; inside → front faces behind camera, only
        // back faces in view.)
        enc.setCullMode(MTLCullMode::Front);
        // Standard forward-render depth state: compare = less, write
        // = on. The fragment shader's `[[depth(less)]]` output further
        // gates: even if the rasterised proxy fragment passes the
        // depth test, the actual raymarch hit depth has to be < the
        // existing value to commit.
        enc.set_depth_stencil(self.depth_state.as_ref());

        // Per-frame view at buffer(0); same value for vertex + fragment.
        enc.set_vertex_value(view, 0);
        enc.set_fragment_value(view, 0);
        // Lights at buffer(2); rebound once.
        enc.set_fragment_value(&lights_gpu, 2);
        // Cascade-shadow uniforms at buffer(3).
        // Always bound; the helper falls back to `shadow = 1.0`
        // when `vol.receive_shadows == 0` or when the world has no
        // shadow stage (in which case `shadow.map` is the 1×1
        // fallback texture and the cascade compare returns full
        // light).
        enc.set_fragment_value(&shadow_uniforms, 3);
        // Proxy-cube vertices at vertex buffer(2); index buffer
        // bound per-draw. The vertex descriptor declares the full
        // 56-byte Vertex layout at this binding.
        enc.set_vertex_buffer(vbuf, 0, RAYMARCH_VERTEX_BUFFER);
        // Main pass MSAA depth at fragment texture(0); sampled by
        // `main_depth.read(px, 0)` in the template fragment for
        // the shader-side cone-march early-out (separate texture
        // from the writable `depth_resolve` attachment so no
        // aliasing).
        enc.set_fragment_texture(self.hdr_targets.depth.as_ref(), 0);
        // CSM shadow map array + IBL cubes.
        // Always bound (1×1 fallback when the world has no shadow
        // stage / no EnvironmentMap), matching the Main pass.
        enc.set_fragment_texture(self.shadow.map.as_ref(), 1);
        enc.set_fragment_texture(self.env_map.irradiance.as_ref(), 2);
        enc.set_fragment_texture(self.env_map.prefilter.as_ref(), 3);
        // Pre-raymarch scene snapshot for refraction
        // sampling. The blit at the top of this function populated
        // `hdr_resolve_copy` from `hdr_resolve` when some volume
        // drawing this frame calls `sampleSceneRefracted`. Always
        // bound (even when no shader uses it, and even when the blit
        // was skipped) so the per-volume PSO doesn't need a
        // "refraction enabled" variant.
        enc.set_fragment_texture(self.hdr_targets.hdr_resolve_copy.as_ref(), 4);
        // Samplers in the order the single source declares them. The depth
        // read needs none (`read` takes integer pixels), so there is no
        // sampler(0) standing in for one, which is what the hand-written MSL
        // used to leave bound and unused.
        enc.set_fragment_sampler(self.shadow.sampler.as_ref(), 0);
        enc.set_fragment_sampler(self.cube_sampler.as_ref(), 1);
        // The linear-clamp post sampler for the scene-copy tap: the same
        // filter the water and bloom passes use.
        enc.set_fragment_sampler(depth_sampler, 2);

        let mut draws: u32 = 0;
        for (i, vol) in self.raymarch.volumes.iter().enumerate() {
            if !visible[i] {
                continue;
            }
            // `pipeline` is already the right variant for this volume: the
            // volumetric (alpha-blended) PSO when `volumetric`, the opaque
            // surface PSO otherwise (selected at build time).
            enc.set_pipeline(&vol.pipeline);
            // Volumetric media are translucent and must not write depth, but
            // they should still be occluded by nearer opaque geometry. Bind the
            // read-only `LessEqual` state (no write): matching the DirectX
            // volumetric PSO's `DepthFunc=LESS_EQUAL, WriteMask=ZERO`. Passing
            // `None` here would trip Metal's validation layer
            // (`setDepthStencilState(nil)` is illegal). Opaque SDF surfaces keep
            // the write-on state so they composite into the depth buffer
            // downstream passes sample.
            if vol.volumetric {
                enc.set_depth_stencil(self.depth_state_read_only.as_ref());
            } else {
                enc.set_depth_stencil(self.depth_state.as_ref());
            }
            enc.set_vertex_value(&vol.uniforms, 1);
            enc.set_fragment_value(&vol.uniforms, 1);
            // SAFETY: the 36 indices in `ibuf` address the 8-vertex proxy cube
            // bound at vertex buffer(2).
            unsafe {
                enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                    MTLPrimitiveType::Triangle,
                    36,
                    MTLIndexType::UInt16,
                    ibuf,
                    0,
                );
            }
            draws += 1;
        }

        Ok(draws)
    }

    // `true` when at least one visible `SdfVolume` opted into `cast_shadows`
    // and built a shadow pipeline. The shadow pass builds the per-frame
    // `RaymarchView` and dispatches `encode_sdf_shadow_casters` only when this
    // holds. Mirrors `directx::raymarch`'s `any_shadow_casters`.
    pub(in crate::metal) fn any_raymarch_shadow_casters(&self) -> bool {
        self.raymarch
            .volumes
            .iter()
            .any(|v| v.visible && v.cast_shadows && v.shadow_pipeline.is_some())
    }

    // Encode raymarched SDF shadow casters into the CSM cascades. Called from
    // `encode_shadow_pass` after the rasterised + skinned casters, on the same
    // command buffer so the writes land before the Main pass samples the
    // shadow map. For each cascade this opens a depth-only render pass on that
    // `shadow.map` slice with `Load` / `Store` (keeping the rasterised depth
    // already written into the slice), then draws each caster's proxy cube
    // with front faces culled. The depth-only fragment cone-marches the SDF
    // from the light side and writes the hit's NDC.z via `[[depth(less)]]`;
    // the slice's LESS depth test keeps the nearest caster (rasterised or
    // raymarched) per texel. A no-op (returns 0) when no volume casts.
    pub(in crate::metal) fn encode_sdf_shadow_casters(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        view: &RaymarchView,
    ) -> Result<u32, String> {
        use crate::gfx::render_types::NUM_SHADOW_CASCADES;
        if !self.any_raymarch_shadow_casters() {
            return Ok(0);
        }
        let vbuf = self
            .raymarch
            .cube_vertex_buffer
            .as_ref()
            .ok_or("raymarch shadow: cube vertex buffer missing")?;
        let ibuf = self
            .raymarch
            .cube_index_buffer
            .as_ref()
            .ok_or("raymarch shadow: cube index buffer missing")?;
        let lights_gpu: RaymarchLightsGpu = self.light_uniforms;
        let shadow_uniforms = self.shadow.uniforms;

        let mut draws: u32 = 0;
        // Only cast into cascades the rasterised shadow pass re-rendered this
        // frame: a skipped cascade's slice must stay exactly as it was last
        // fully rendered (raster + SDF), so we neither clear nor add to it.
        let render_mask = if self.shadow.render_mask == 0 {
            (1u32 << NUM_SHADOW_CASCADES) - 1
        } else {
            self.shadow.render_mask
        };
        for cascade_idx in 0..NUM_SHADOW_CASCADES {
            if render_mask & (1u32 << cascade_idx) == 0 {
                continue;
            }
            let pass_desc = MTLRenderPassDescriptor::new();
            let da = pass_desc.depthAttachment();
            da.setTexture(Some(self.shadow.map.as_ref()));
            da.setSlice(cascade_idx);
            // Load the rasterised depth this cascade already holds, draw the
            // SDF casters on top, and keep the merged depth for the Main pass.
            da.setLoadAction(MTLLoadAction::Load);
            da.setStoreAction(MTLStoreAction::Store);

            // Loop-local guard: each cascade's encoder ends on drop at the end
            // of this iteration, before the next cascade opens one.
            let enc = ScopedEncoder::new(
                cmd_buf
                    .renderCommandEncoderWithDescriptor(&pass_desc)
                    .ok_or("failed to get raymarch shadow render encoder")?,
                "raymarch shadow",
            );
            // Front-face cull → exactly one fragment per texel inside the box's
            // light-space projection. Same depth state (compare = less, write
            // on) as the rasterised casters so the two layers composite.
            enc.setCullMode(MTLCullMode::Front);
            enc.set_depth_stencil(self.depth_state.as_ref());

            let cascade = RaymarchShadowCascade {
                cascade_idx: cascade_idx as u32,
                _pad: [0; 3],
            };
            // Per-cascade shared bindings: view@0 (fragment reads
            // view.time), lights@2 (fragment), shadow uniforms@3 (vertex
            // projection + fragment reprojection), cascade selector@4
            // (both stages), proxy-cube vertices@2 (vertex).
            enc.set_fragment_value(view, 0);
            enc.set_fragment_value(&lights_gpu, 2);
            enc.set_vertex_value(&shadow_uniforms, 3);
            enc.set_fragment_value(&shadow_uniforms, 3);
            enc.set_vertex_value(&cascade, 4);
            enc.set_fragment_value(&cascade, 4);
            enc.set_vertex_buffer(vbuf, 0, RAYMARCH_VERTEX_BUFFER);

            for vol in &self.raymarch.volumes {
                if !vol.visible || !vol.cast_shadows {
                    continue;
                }
                let Some(pso) = vol.shadow_pipeline.as_ref() else {
                    continue;
                };
                enc.set_pipeline(pso);
                enc.set_vertex_value(&vol.uniforms, 1);
                enc.set_fragment_value(&vol.uniforms, 1);
                // SAFETY: the 36 indices in `ibuf` address the 8-vertex proxy
                // cube bound at vertex buffer(2).
                unsafe {
                    enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                        MTLPrimitiveType::Triangle,
                        36,
                        MTLIndexType::UInt16,
                        ibuf,
                        0,
                    );
                }
                draws += 1;
            }
        }
        Ok(draws)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volume_in_frustum_culls_offscreen_boxes() {
        use crate::gfx::frustum::Frustum;
        // Identity view-projection -> the visible region is the [-1, 1]^3 clip
        // cube. A unit box at the origin overlaps it; a box far to the right is
        // entirely past the right clip plane and is culled.
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let f = Frustum::from_view_projection(identity);
        assert!(volume_in_frustum([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], &f));
        assert!(!volume_in_frustum([10.0, 0.0, 0.0], [0.5, 0.5, 0.5], &f));
        // A box the camera sits inside (origin within its extent) still
        // overlaps the frustum.
        assert!(volume_in_frustum(
            [0.0, 0.0, 0.0],
            [100.0, 100.0, 100.0],
            &f
        ));
    }
}
