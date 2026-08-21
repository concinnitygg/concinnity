// src/metal/line.rs
//
// Per-frame encoder for the world-space line pass. Runs at the tail of
// the hdr_resolve decoration chain, after the main pass resolved colour into
// `hdr_targets.hdr_resolve` and depth into `hdr_targets.depth_resolve`, so the
// lines layer over the lit scene and SSR / TAA treat them like any other scene
// content.
//
// The ribbons arrive already expanded (`gfx::lines::build_vertices`):
// world-space quads whose width was sized off each corner's depth, so a line
// holds its pixel thickness at any distance. Like the decal pass this one
// attaches no depth buffer and instead samples the resolved depth, so an
// occluded line fades to `OCCLUDED_ALPHA` rather than being clipped by
// hardware.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlendFactor, MTLCommandBuffer, MTLDevice as _, MTLLoadAction, MTLPixelFormat,
    MTLPrimitiveType, MTLRenderCommandEncoder as _, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLResourceOptions, MTLStoreAction,
    MTLVertexFormat, MTLVertexStepFunction,
};

use super::context::MtlContext;
use super::descriptors::{VertexAttr, VertexLayout, vertex_descriptor};

use super::scoped_encoder::ScopedEncoder;
use crate::gfx::render_types::LineVertex;

// How much of a line still shows where scene geometry is in front of it. A
// faint trace keeps the axes readable inside a dense scene without letting
// them pretend to be unoccluded.
const OCCLUDED_ALPHA: f32 = 0.12;

// Vertex buffer index the ribbon geometry binds to, clear of buffer(0) (the
// view uniforms).
const VERTEX_BUFFER_INDEX: usize = 1;

// Line-pass state: the pipeline, built on the first frame that submits lines
// so a world that never draws any pays nothing, plus the build-failure latch
// that keeps a broken build from re-reporting every frame.
pub(crate) struct LineState {
    pub pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub build_failed: bool,
}

impl MtlContext {
    // Build the line pipeline if this frame has lines to draw and it is
    // not built yet. A failed build latches, so the error is reported once and
    // the pass stays skipped for the rest of the run.
    pub(in crate::metal) fn ensure_line_pipeline(&mut self, has_lines: bool) {
        if !has_lines || self.lines.pipeline.is_some() || self.lines.build_failed {
            return;
        }
        match build_line_pipeline(&self.device, self.hot_reload) {
            Ok(ps) => self.lines.pipeline = Some(ps),
            Err(e) => {
                self.lines.build_failed = true;
                tracing::error!("line pipeline: {}", e);
            }
        }
    }

    // Encode the line pass: one unindexed triangle list covering every
    // expanded ribbon, alpha-blended into `hdr_resolve`. `vp` is the same
    // view-projection the main pass rasterised with (jittered under TAA), so a
    // line sits on the pixel its geometry did. Returns the draw-call count.
    // pub(in crate::metal) so the render-graph executor in metal/graph_exec.rs
    // can dispatch this pass from a CompiledGraph.
    pub(in crate::metal) fn encode_lines(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        vp: [[f32; 4]; 4],
        vertices: &[LineVertex],
    ) -> Result<u32, String> {
        let pipeline = match &self.lines.pipeline {
            Some(p) => p,
            None => return Ok(0),
        };
        if vertices.is_empty() {
            return Ok(0);
        }
        let view = concinnity_render::uniforms::LineView {
            vp,
            occluded_alpha: OCCLUDED_ALPHA,
            _pad: [0.0; 3],
        };
        let bytes = std::mem::size_of_val(vertices);
        // SAFETY: the pointer and length describe the live `vertices` allocation, and Metal copies
        // those bytes into the new buffer before the call returns.
        let vbuf = unsafe {
            self.device
                .newBufferWithBytes_length_options(
                    std::ptr::NonNull::from(vertices).cast(),
                    bytes,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or("failed to create line vertex buffer")?
        };

        let pass_desc = MTLRenderPassDescriptor::new();
        // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
        // declares.
        unsafe {
            let ca = pass_desc.colorAttachments().objectAtIndexedSubscript(0);
            ca.setTexture(Some(self.hdr_targets.hdr_resolve.as_ref()));
            ca.setLoadAction(MTLLoadAction::Load);
            ca.setStoreAction(MTLStoreAction::Store);
        }
        if let Some(t) = &self.pass_timing {
            t.attach_render(&pass_desc, super::pass_timing::PassId::Lines);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&pass_desc)
                .ok_or("failed to get line render encoder")?,
            "lines",
        );
        enc.setRenderPipelineState(pipeline);
        // SAFETY: each pointer is derived from the live `view` with its `size_of` as the length,
        // and `vbuf` outlives the encoder; the indices are the slots the line shaders declare.
        unsafe {
            enc.setVertexBytes_length_atIndex(
                std::ptr::NonNull::from(&view).cast(),
                std::mem::size_of::<concinnity_render::uniforms::LineView>(),
                0,
            );
            enc.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(&view).cast(),
                std::mem::size_of::<concinnity_render::uniforms::LineView>(),
                0,
            );
            enc.setVertexBuffer_offset_atIndex(Some(&vbuf), 0, VERTEX_BUFFER_INDEX);
            // Resolved scene depth at texture(0) for the manual depth test.
            enc.setFragmentTexture_atIndex(Some(self.hdr_targets.depth_resolve.as_ref()), 0);
            enc.drawPrimitives_vertexStart_vertexCount(
                MTLPrimitiveType::Triangle,
                0,
                vertices.len(),
            );
        }
        Ok(1)
    }
}

// Build the line pipeline: world-space ribbon corners transformed by the
// camera VP and alpha-blended into the resolved HDR target. No depth
// attachment; the fragment shader tests the resolved depth itself so an
// occluded line can fade instead of vanishing.
fn build_line_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    // Each entry compiles to its own metallib, so the two stages come from
    // separate libraries and pair by semantic.
    let vert_fn =
        super::slang_shaders::entry_function(device, &super::slang_shaders::LINE_VERT, hot_reload)?;
    let frag_fn =
        super::slang_shaders::entry_function(device, &super::slang_shaders::LINE_FRAG, hot_reload)?;

    // Vertex layout: `LineVertex` (position, edge, colour) at 32 bytes,
    // asserted by `line_vertex_layout_matches_shaders`.
    let vert_desc = vertex_descriptor(
        &[
            VertexAttr {
                index: 0,
                format: MTLVertexFormat::Float3,
                offset: 0,
                buffer_index: VERTEX_BUFFER_INDEX,
            },
            VertexAttr {
                index: 1,
                format: MTLVertexFormat::Float,
                offset: 12,
                buffer_index: VERTEX_BUFFER_INDEX,
            },
            VertexAttr {
                index: 2,
                format: MTLVertexFormat::Float4,
                offset: 16,
                buffer_index: VERTEX_BUFFER_INDEX,
            },
        ],
        &[VertexLayout {
            buffer_index: VERTEX_BUFFER_INDEX,
            stride: std::mem::size_of::<LineVertex>(),
            step: MTLVertexStepFunction::PerVertex,
        }],
    );

    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexDescriptor(Some(&vert_desc));
    desc.setVertexFunction(Some(&vert_fn));
    desc.setFragmentFunction(Some(&frag_fn));
    desc.setRasterSampleCount(1);
    // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
    // declares.
    unsafe {
        let ca = desc.colorAttachments().objectAtIndexedSubscript(0);
        ca.setPixelFormat(MTLPixelFormat::RGBA16Float);
        ca.setBlendingEnabled(true);
        ca.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
        ca.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        ca.setSourceAlphaBlendFactor(MTLBlendFactor::SourceAlpha);
        ca.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    }

    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| format!("failed to create line pipeline state: {:?}", e))
}
