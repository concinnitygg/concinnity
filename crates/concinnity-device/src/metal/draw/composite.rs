// src/metal/draw/composite.rs
//
// Composite (post-process) pass + text overlay. The post-process pipeline
// reads `scene_color`, the bloom mip-0 target, and the 3D color-grading LUT,
// then writes ACES tonemap + gamma + FXAA into the drawable. Text is drawn
// after in the same render pass so it sits on top of the tonemapped image in
// display-referred LDR space; its geometry comes from sub-ranges of this
// frame's text-upload slot, filled by `draw_frame` before the graph ran (see
// [`crate::metal::text_upload::TextUploadRing`]).
//
// The order of the two halves lives once in `gfx::fullscreen`; this file is
// Metal's implementation of each step. Metal's recorder is the render encoder
// itself, which is why `encode_composite_and_text` opens it before handing the
// chain over: the encoder is the open pass, so beginning and ending the pass
// are its construction and its drop rather than driver steps.
#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::Cell;

use concinnity_core::gfx::render_types;
use concinnity_core::gfx::render_types::TextDrawCall;
use concinnity_core::render::fullscreen;
use concinnity_core::render::fullscreen::TextBindCache;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLCommandBuffer as _, MTLIndexType, MTLLoadAction, MTLPrimitiveType,
    MTLRenderCommandEncoder as _, MTLScissorRect, MTLStoreAction, MTLTexture,
};

use crate::metal::context::MtlContext;
use crate::metal::encode::RenderEncode;
use crate::metal::scoped_encoder::ScopedEncoder;

// The composite + text steps in Metal, over the state the whole chain shares.
// `draws` counts what the chain encoded, since the driver returns only success
// or failure and the frame reports a draw-call count. It is a `Cell` because
// every step takes `&self`, and the pass is encoded on one thread.
struct CompositePass<'a> {
    ctx: &'a MtlContext,
    scene_color: &'a ProtocolObject<dyn MTLTexture>,
    // The window's content size in logical points, which the text vertices are
    // in and the text shader divides by. Read once for the pass: every label
    // scales against it.
    logical: (f32, f32),
    // The drawable's pixel size, which a per-call clip rect scales into.
    framebuffer: (u32, u32),
    channel_view: u32,
    draws: Cell<u32>,
}

impl fullscreen::CompositeEncoder for CompositePass<'_> {
    type Rec = ScopedEncoder<dyn objc2_metal::MTLRenderCommandEncoder>;
    type Args = ();

    // Nothing to do: the recorder this chain records into is the open pass,
    // created by `encode_composite_and_text` below.
    fn begin_composite(&self, _enc: &Self::Rec, _args: &()) {}

    fn composite_draw(&self, enc: &Self::Rec, _args: &()) {
        enc.set_pipeline(&self.ctx.post_pipeline_state);
        enc.set_fragment_texture(self.scene_color, 0);
        // Bloom mip 0 at texture(1). Always bound so the binding resolves;
        // the shader skips the sample when bloom_intensity == 0.
        enc.set_fragment_texture(self.ctx.bloom_targets.mips[0].as_ref(), 1);
        // 3D color-grading LUT at texture(2). Always bound -- an identity
        // LUT stands in when the world declares no ColorLut.
        enc.set_fragment_texture(self.ctx.color_lut.as_ref(), 2);
        // The channel sources at texture(3..5), bound only while a channel
        // view will sample them. The SSAO white 1x1 stands in when a
        // G-buffer was never built.
        if self.channel_view != 0 {
            let nd = self
                .ctx
                .gbuffer_normal_depth()
                .unwrap_or_else(|| self.ctx.ssao.white.as_ref());
            let rough = self
                .ctx
                .gbuffer_roughness()
                .unwrap_or_else(|| self.ctx.ssao.white.as_ref());
            enc.set_fragment_texture(nd, 3);
            enc.set_fragment_texture(rough, 4);
            enc.set_fragment_texture(self.ctx.ao_output_texture(), 5);
        }
        crate::metal::post::fullscreen::set_fragment_sampler_range(
            enc,
            &self.ctx.post_sampler,
            0,
            6,
        );
        // Post-process tunables (bloom intensity) plus the scene-transition
        // fade at buffer(0).
        let composite = render_types::CompositeParams {
            post: self.ctx.post_process,
            fade: self.ctx.view.scene_fade,
            view_mode: self.channel_view,
            far: self.ctx.view.far,
        };
        enc.set_fragment_value(&composite, 0);
        // Fullscreen triangle: 3 vertices, no vertex buffer (the shared
        // fullscreen_vertex synthesizes position + UV from SV_VertexID).
        // SAFETY: the vertex shader generates all three vertices, so the draw reads no bound
        // vertex buffer.
        unsafe {
            enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
        self.draws.set(self.draws.get() + 1);
    }

    fn begin_text(&self, enc: &Self::Rec, _args: &()) -> bool {
        let Some(text_ps) = self.ctx.text.pipeline_state.clone() else {
            return false;
        };
        if self.ctx.text.upload.binding().is_none() || self.ctx.text.atlas_textures.is_empty() {
            return false;
        }
        let (win_w, win_h) = self.logical;
        let text_uniforms = render_types::TextUniforms {
            win_width: win_w,
            win_height: win_h,
            _pad: [0.0; 2],
        };
        enc.set_pipeline(&text_ps);
        // Frame-invariant text state, set once for the whole run of labels:
        // the encoder keeps it bound across the draws below, and nothing in
        // the loop switches pipeline.
        enc.set_fragment_sampler(&self.ctx.text.sampler, 0);
        enc.set_vertex_value(&text_uniforms, 0);
        true
    }

    fn text_draw(
        &self,
        enc: &Self::Rec,
        _args: &(),
        idx: usize,
        call: &TextDrawCall,
        binds: &mut TextBindCache,
    ) -> Result<(), String> {
        if call.vertices.is_empty() {
            return Ok(());
        }
        // This frame's geometry was uploaded before the graph ran, one range per
        // call in list order, so the driver's index is what addresses it. A list
        // longer than the ranges uploaded for it draws only what was uploaded.
        let Some((text_buffer, ranges)) = self.ctx.text.upload.binding() else {
            return Ok(());
        };
        let Some(range) = ranges.get(idx) else {
            return Ok(());
        };
        let (win_w, win_h) = self.logical;
        let (fb_w, fb_h) = self.framebuffer;
        // Clip this call to its band (scrollable panel content) or reset
        // to the full drawable (chrome / HUD). A clip rect that scales to
        // an empty rectangle means the element scrolled fully out of its
        // band: skip the draw entirely.
        let scissor = match call.clip_rect {
            Some(clip) => {
                let Some(rect) =
                    fullscreen::clip_rect_to_scissor(clip, (win_w, win_h), (fb_w, fb_h))
                else {
                    return Ok(());
                };
                rect
            }
            None => (0, 0, fb_w, fb_h),
        };
        if binds.scissor_changed(scissor) {
            let (x, y, w, h) = scissor;
            enc.setScissorRect(MTLScissorRect {
                x: x as usize,
                y: y as usize,
                width: w as usize,
                height: h as usize,
            });
        }
        let atlas_idx = call.atlas_slot.min(self.ctx.text.atlas_textures.len() - 1);
        if binds.atlas_changed(atlas_idx) {
            enc.set_fragment_texture(self.ctx.text.atlas_textures[atlas_idx].as_ref(), 0);
        }

        enc.set_vertex_buffer(text_buffer, range.vertex_offset, 1);
        // SAFETY: `text_buffer` is owned by the ring and outlives the encoder, and this
        // call's blocks were written at `range`'s offsets from this same call's vertex /
        // index data, so the index count is exactly the range they cover.
        unsafe {
            enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                MTLPrimitiveType::Triangle,
                call.indices.len(),
                MTLIndexType::UInt16,
                text_buffer,
                range.index_offset,
            );
        }
        self.draws.set(self.draws.get() + 1);
        Ok(())
    }

    // Nothing to do: the recorder ends its own pass when the caller's guard
    // drops, which is what makes a failed text draw safe to propagate here.
    fn end_composite(&self, _enc: &Self::Rec, _args: &()) {}
}

impl MtlContext {
    // pub(in crate::metal) so the render-graph executor in
    // `metal/graph_exec.rs` can dispatch to this from outside `metal/draw/`.
    pub(in crate::metal) fn encode_composite_and_text(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        scene_color: &Retained<ProtocolObject<dyn MTLTexture>>,
        text_calls: &[TextDrawCall],
    ) -> Result<u32, String> {
        let composite_pass_desc = self
            .window
            .view
            .currentRenderPassDescriptor()
            .ok_or("no current render pass descriptor")?;
        // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
        // declares.
        unsafe {
            let ca = composite_pass_desc
                .colorAttachments()
                .objectAtIndexedSubscript(0);
            ca.setLoadAction(MTLLoadAction::DontCare);
            ca.setStoreAction(MTLStoreAction::Store);
        }

        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_render(
                &composite_pass_desc,
                super::super::pass_timing::PassId::Composite,
            );
        }
        // The text vertices are in logical points (mapped to NDC by the shader's
        // divide by win_width/height); the scissor is in framebuffer pixels.
        // Recover the drawable's pixel size from the composite color attachment
        // so a per-call clip rect scales from points to pixels.
        let size = self.window.view.bounds().size;
        let logical = (size.width as f32, size.height as f32);
        // SAFETY: attachment 0 is the only color attachment this pass declares, and the
        // accessors only read it.
        let framebuffer = unsafe {
            match composite_pass_desc
                .colorAttachments()
                .objectAtIndexedSubscript(0)
                .texture()
            {
                Some(t) => (t.width() as u32, t.height() as u32),
                None => (logical.0.max(0.0) as u32, logical.1.max(0.0) as u32),
            }
        };
        // Opening the encoder is what begins the pass, and the guard ends it
        // however this function leaves, including on a failed text draw: a
        // render encoder left open crashes at commit.
        let post_encoder = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&composite_pass_desc)
                .ok_or("failed to get post-process render encoder")?,
            ns_string!("composite"),
        );

        let pass = CompositePass {
            ctx: self,
            scene_color: scene_color.as_ref(),
            logical,
            framebuffer,
            // A G-buffer channel view swaps the fragment onto its visualization
            // branch; Lit / Unlit / Wireframe all take the normal scene path.
            channel_view: if self.view.mode.is_gbuffer_channel() {
                self.view.mode as u32
            } else {
                0
            },
            draws: Cell::new(0),
        };
        fullscreen::encode_composite_chain(&pass, &post_encoder, &(), text_calls)?;
        Ok(pass.draws.get())
    }
}
