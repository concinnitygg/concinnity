// src/metal/draw/composite.rs
//
// Composite (post-process) pass + text overlay. The post-process pipeline
// reads `scene_color`, the bloom mip-0 target, and the 3D colour-grading LUT,
// then writes ACES tonemap + gamma + FXAA into the drawable. Text is drawn
// after in the same render pass so it sits on top of the tonemapped image in
// display-referred LDR space; its geometry comes from sub-ranges of this
// frame's text-upload slot, filled by `draw_frame` before the graph ran (see
// [`crate::metal::text_upload::TextUploadRing`]).
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer as _, MTLIndexType, MTLLoadAction, MTLPrimitiveType,
    MTLRenderCommandEncoder as _, MTLScissorRect, MTLStoreAction, MTLTexture,
};

use crate::gfx::render_types::TextDrawCall;
use crate::metal::context::MtlContext;
use crate::metal::encode::RenderEncode;
use crate::metal::scoped_encoder::ScopedEncoder;

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
            .mtk_view
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

        if let Some(t) = &self.pass_timing {
            t.attach_render(
                &composite_pass_desc,
                super::super::pass_timing::PassId::Composite,
            );
        }
        // ScopedEncoder ends the pass however this function leaves, including on
        // an `?` mid-encode: a render encoder left open crashes at commit.
        let post_encoder = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&composite_pass_desc)
                .ok_or("failed to get post-process render encoder")?,
            "composite",
        );

        post_encoder.set_pipeline(&self.post_pipeline_state);
        // A G-buffer channel view swaps the fragment onto its visualization
        // branch; Lit / Unlit / Wireframe all take the normal scene path.
        let channel_view = if self.view_mode.is_gbuffer_channel() {
            self.view_mode as u32
        } else {
            0
        };
        post_encoder.set_fragment_texture(scene_color.as_ref(), 0);
        // Bloom mip 0 at texture(1). Always bound so the binding resolves;
        // the shader skips the sample when bloom_intensity == 0.
        post_encoder.set_fragment_texture(self.bloom_targets.mips[0].as_ref(), 1);
        // 3D colour-grading LUT at texture(2). Always bound -- an identity
        // LUT stands in when the world declares no ColorLut.
        post_encoder.set_fragment_texture(self.color_lut.as_ref(), 2);
        // The channel sources at texture(3..5), bound only while a channel
        // view will sample them. The SSAO white 1x1 stands in when a
        // G-buffer was never built.
        if channel_view != 0 {
            let nd = self
                .gbuffer_normal_depth()
                .unwrap_or_else(|| self.ssao.white.as_ref());
            let rough = self
                .gbuffer_roughness()
                .unwrap_or_else(|| self.ssao.white.as_ref());
            post_encoder.set_fragment_texture(nd, 3);
            post_encoder.set_fragment_texture(rough, 4);
            post_encoder.set_fragment_texture(self.ao_output_texture(), 5);
        }
        crate::metal::post::fullscreen::set_fragment_sampler_range(
            &post_encoder,
            &self.post_sampler,
            0,
            6,
        );
        // Post-process tunables (bloom intensity) plus the scene-transition
        // fade at buffer(0).
        let composite = crate::gfx::render_types::CompositeParams {
            post: self.post_process,
            fade: self.scene_fade,
            view_mode: channel_view,
            far: self.view_far,
        };
        post_encoder.set_fragment_value(&composite, 0);
        // Fullscreen triangle: 3 vertices, no vertex buffer (the shared
        // fullscreen_vertex synthesises position + UV from SV_VertexID).
        // SAFETY: the vertex shader generates all three vertices, so the draw reads no bound
        // vertex buffer.
        unsafe {
            post_encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
        let mut draw_calls: u32 = 1;

        // Text overlay: rendered in the same composite pass so it sits on
        // top of the tonemapped image in display-referred LDR space.
        if let Some(text_ps) = self.text_pipeline_state.clone()
            && let Some((text_buffer, text_ranges)) = self.text_upload.binding()
            && !self.text_atlas_textures.is_empty()
        {
            let logical = self.mtk_view.bounds().size;
            let win_w = logical.width as f32;
            let win_h = logical.height as f32;
            let text_uniforms = crate::gfx::render_types::TextUniforms {
                win_width: win_w,
                win_height: win_h,
                _pad: [0.0; 2],
            };

            // The text vertices are in logical points (mapped to NDC by the
            // shader's divide by win_width/height); the scissor is in framebuffer
            // pixels. Recover the drawable's pixel size from the composite color
            // attachment so a per-call clip rect scales from points to pixels.
            // SAFETY: attachment 0 is the only colour attachment this pass declares, and the
            // accessors only read it.
            let (fb_w, fb_h) = unsafe {
                match composite_pass_desc
                    .colorAttachments()
                    .objectAtIndexedSubscript(0)
                    .texture()
                {
                    Some(t) => (t.width(), t.height()),
                    None => (win_w.max(0.0) as usize, win_h.max(0.0) as usize),
                }
            };

            post_encoder.set_pipeline(&text_ps);

            for (call, range) in text_calls.iter().zip(text_ranges) {
                if call.vertices.is_empty() {
                    continue;
                }
                // Clip this call to its band (scrollable panel content) or reset
                // to the full drawable (chrome / HUD). A clip rect that scales to
                // an empty rectangle means the element scrolled fully out of its
                // band: skip the draw entirely.
                match call.clip_rect {
                    Some(clip) => {
                        let scissor = crate::gfx::fullscreen::clip_rect_to_scissor(
                            clip,
                            (win_w, win_h),
                            (fb_w as u32, fb_h as u32),
                        );
                        let Some((x, y, w, h)) = scissor else {
                            continue;
                        };
                        post_encoder.setScissorRect(MTLScissorRect {
                            x: x as usize,
                            y: y as usize,
                            width: w as usize,
                            height: h as usize,
                        });
                    }
                    None => {
                        post_encoder.setScissorRect(MTLScissorRect {
                            x: 0,
                            y: 0,
                            width: fb_w,
                            height: fb_h,
                        });
                    }
                }
                let atlas_idx = call.atlas_slot.min(self.text_atlas_textures.len() - 1);
                post_encoder.set_fragment_texture(self.text_atlas_textures[atlas_idx].as_ref(), 0);
                post_encoder.set_fragment_sampler(&self.text_sampler, 0);

                post_encoder.set_vertex_value(&text_uniforms, 0);
                post_encoder.set_vertex_buffer(text_buffer, range.vertex_offset, 1);
                // SAFETY: `text_buffer` is owned by the ring and outlives the encoder, and this
                // call's blocks were written at `range`'s offsets from this same call's vertex /
                // index data, so the index count is exactly the range they cover.
                unsafe {
                    post_encoder
                        .drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                            MTLPrimitiveType::Triangle,
                            call.indices.len(),
                            MTLIndexType::UInt16,
                            text_buffer,
                            range.index_offset,
                        );
                }
                draw_calls += 1;
            }
        }

        Ok(draw_calls)
    }
}
