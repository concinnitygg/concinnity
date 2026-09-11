// src/metal/line.rs
//
// Per-frame encoder for the world-space line pass. Runs at the tail of
// the hdr_resolve decoration chain, after the main pass resolved color into
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

use concinnity_core::gfx::render_types::LineVertex;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLBlendFactor, MTLBuffer, MTLCommandBuffer, MTLDevice as _, MTLLoadAction, MTLPixelFormat,
    MTLPrimitiveType, MTLRenderCommandEncoder as _, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLStoreAction, MTLVertexFormat,
    MTLVertexStepFunction,
};

use super::context::{MtlContext, bytes_of_slice};
use super::descriptors::{VertexAttr, VertexLayout, vertex_descriptor};
use super::encode::RenderEncode;
use super::scoped_encoder::ScopedEncoder;
use super::transient::TransientRing;

// How much of a line still shows where scene geometry is in front of it. A
// faint trace keeps the axes readable inside a dense scene without letting
// them pretend to be unoccluded.
const OCCLUDED_ALPHA: f32 = 0.12;

// Vertex buffer index the ribbon geometry binds to, clear of buffer(0) (the
// view uniforms).
const VERTEX_BUFFER_INDEX: usize = 1;

// Line-pass state: the pipeline, built on the first frame that submits lines
// so a world that never draws any pays nothing, the build-failure latch that
// keeps a broken build from re-reporting every frame, and the per-frame ribbon
// vertex upload.
pub(crate) struct LineState {
    pub pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub build_failed: bool,
    // Ring of per-frame ribbon vertex buffers, one slot per frame-in-flight.
    // Written by [`MtlContext::upload_lines`] before the graph runs.
    pub upload: TransientRing,
    // This frame's slot handle and its vertex count, `None` on a frame that
    // publishes no lines. Set by `upload_lines`, bound by `encode_lines`.
    pub frame: Option<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>,
}

impl MtlContext {
    // Build the line pipeline if this frame has lines to draw and it is
    // not built yet. A failed build latches, so the error is reported once and
    // the pass stays skipped for the rest of the run.
    pub(in crate::metal) fn ensure_line_pipeline(&mut self, has_lines: bool) {
        if !has_lines || self.lines.pipeline.is_some() || self.lines.build_failed {
            return;
        }
        match build_line_pipeline(&self.device, self.hot_reload.enabled) {
            Ok(ps) => self.lines.pipeline = Some(ps),
            Err(e) => {
                self.lines.build_failed = true;
                tracing::error!("line pipeline: {}", e);
            }
        }
    }

    // Copy this frame's expanded ribbons into this frame's ring slot. Call once
    // per frame, past the frames-in-flight fence and before the graph runs, so
    // overwriting the slot cannot race a GPU read of the frame that last used
    // it. `encode_lines` takes `&self` and so cannot upload for itself.
    pub(in crate::metal) fn upload_lines(
        &mut self,
        slot: usize,
        vertices: &[LineVertex],
    ) -> Result<(), String> {
        self.lines.frame = None;
        if self.lines.pipeline.is_none() || vertices.is_empty() {
            return Ok(());
        }
        let buf = self
            .lines
            .upload
            .write(&self.device, slot, bytes_of_slice(vertices))?;
        self.lines.frame = Some((buf, vertices.len()));
        Ok(())
    }

    // Encode the line pass: one unindexed triangle list covering every
    // expanded ribbon, alpha-blended into `hdr_resolve`. `vp` is the same
    // view-projection the main pass rasterized with (jittered under TAA), so a
    // line sits on the pixel its geometry did. Returns the draw-call count.
    // pub(in crate::metal) so the render-graph executor in metal/graph_exec.rs
    // can dispatch this pass from a CompiledGraph.
    pub(in crate::metal) fn encode_lines(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        vp: [[f32; 4]; 4],
    ) -> Result<u32, String> {
        let (Some(pipeline), Some((vbuf, vertex_count))) =
            (self.lines.pipeline.as_ref(), self.lines.frame.as_ref())
        else {
            return Ok(0);
        };
        let view = concinnity_core::render::uniforms::LineView {
            vp,
            occluded_alpha: OCCLUDED_ALPHA,
            _pad: [0.0; 3],
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
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_render(&pass_desc, super::pass_timing::PassId::Lines);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&pass_desc)
                .ok_or("failed to get line render encoder")?,
            ns_string!("lines"),
        );
        enc.set_pipeline(pipeline);
        enc.set_vertex_value(&view, 0);
        enc.set_fragment_value(&view, 0);
        enc.set_vertex_buffer(vbuf, 0, VERTEX_BUFFER_INDEX);
        // Resolved scene depth at texture(0) for the manual depth test.
        enc.set_fragment_texture(self.hdr_targets.depth_resolve.as_ref(), 0);
        // SAFETY: the draw covers exactly the vertices uploaded into `vbuf`.
        unsafe {
            enc.drawPrimitives_vertexStart_vertexCount(
                MTLPrimitiveType::Triangle,
                0,
                *vertex_count,
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
    let vert_fn = super::slang_builtins::entry_function(
        device,
        &super::slang_builtins::LINE_VERT,
        hot_reload,
    )?;
    let frag_fn = super::slang_builtins::entry_function(
        device,
        &super::slang_builtins::LINE_FRAG,
        hot_reload,
    )?;

    // Vertex layout: `LineVertex` (position, edge, color) at 32 bytes,
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
