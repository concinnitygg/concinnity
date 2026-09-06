// src/metal/draw/spot_shadow.rs
//
// Spot shadow pass: one depth-only render per shadow-casting spot light into its
// slice of `spot_shadow.map`, plus the per-draw caster body it is the only
// caller of. The shadow ICB the bindless cull fills is laid out per CSM cascade,
// so it has no slots for these slices; the casters are walked on the CPU here,
// once per slice, through the same depth-only pipelines the cascade pass binds.
//
// Local lights are static, so the matrices are built once at init and only the
// depth contents refresh here. `spot_shadow.render_mask` (from
// `SpotShadowScheduler`) picks which slices redraw; a skipped slice keeps the
// depth it last rendered, which stays correct until a caster moves.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLIndexType, MTLLoadAction,
    MTLPrimitiveType, MTLRenderCommandEncoder as _, MTLRenderPassDescriptor, MTLStoreAction,
};

use crate::gfx::render_types::{ShadowPassPush, ShadowUniforms, SpotShadowData};
use crate::gfx::shadow_bias;
use crate::metal::context::MtlContext;
use crate::metal::encode::RenderEncode;
use crate::metal::scoped_encoder::ScopedEncoder;
use crate::metal::uniforms::ModelUniforms;

// A spot slice's matrix always lands in slot 0 of its one-matrix
// `ShadowUniforms`, so the shadow VS's cascade index is constant here.
const SPOT_SLICE_IDX: u32 = 0;

// The per-slice state every caster sub-encoder binds before drawing. The static
// and instanced casters take the depth-only shadow pipeline; the skinned ones
// swap in the skinned variant against the same uniforms.
struct SpotSliceBinding<'a> {
    pipeline: &'a ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>,
    uniforms: &'a ShadowUniforms,
}

impl MtlContext {
    // Choose which spot shadow slices to re-render this frame and advance the
    // round-robin clock. Called once per frame from draw_frame; the result is
    // stashed in `spot_shadow.render_mask` for encode_spot_shadow_pass.
    pub(in crate::metal) fn next_spot_shadow_mask(&mut self) -> u32 {
        let every_frame = matches!(
            self.shadow.update,
            crate::components::ShadowUpdate::EveryFrame
        );
        self.spot_shadow
            .scheduler
            .next_mask(every_frame, self.spot_shadow.count as usize)
    }

    // pub(in crate::metal) so the render-graph executor can dispatch this pass.
    pub(in crate::metal) fn encode_spot_shadow_pass(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        skinned_joint_bufs: &[Retained<ProtocolObject<dyn MTLBuffer>>],
        cam_pos: [f32; 3],
    ) -> Result<u32, String> {
        let Some(shadow_pipeline) = self.shadow.pipeline_state.clone() else {
            return Ok(0);
        };
        if self.spot_shadow.count == 0 {
            return Ok(0);
        }

        let all = if self.spot_shadow.count >= 32 {
            u32::MAX
        } else {
            (1_u32 << self.spot_shadow.count) - 1
        };
        // Defensive fallback to every slice if no mask was set this frame.
        let mask = if self.spot_shadow.render_mask == 0 {
            all
        } else {
            self.spot_shadow.render_mask
        };
        let rendered: Vec<u32> = (0..self.spot_shadow.count)
            .filter(|i| mask & (1u32 << i) != 0)
            .collect();
        let first_rendered = rendered.first().copied();
        let last_rendered = rendered.last().copied();

        let mut total_draws: u32 = 0;
        for &slice in &rendered {
            let pass_desc = MTLRenderPassDescriptor::new();
            let depth_attach = pass_desc.depthAttachment();
            depth_attach.setTexture(Some(self.spot_shadow.map.as_ref()));
            depth_attach.setSlice(slice as usize);
            depth_attach.setLoadAction(MTLLoadAction::Clear);
            depth_attach.setStoreAction(MTLStoreAction::Store);
            depth_attach.setClearDepth(1.0);

            // Timing spans the first to the last slice actually rendered, the
            // same shape the cascade pass uses.
            if let Some(t) = &self.diagnostics.pass_timing {
                let id = super::super::pass_timing::PassId::SpotShadow;
                let is_first = Some(slice) == first_rendered;
                let is_last = Some(slice) == last_rendered;
                if is_first && is_last {
                    t.attach_render(&pass_desc, id);
                } else if is_first {
                    t.attach_render_first(&pass_desc, id);
                } else if is_last {
                    t.attach_render_last(&pass_desc, id);
                }
            }

            let enc = ScopedEncoder::new(
                cmd_buf
                    .renderCommandEncoderWithDescriptor(&pass_desc)
                    .ok_or("failed to get spot shadow render encoder")?,
                "spot shadow slice",
            );

            let uniforms = self.spot_slice_uniforms(slice);
            let bind = SpotSliceBinding {
                pipeline: &shadow_pipeline,
                uniforms: &uniforms,
            };
            total_draws += self.encode_spot_casters(&enc, &bind, cam_pos, skinned_joint_bufs);
        }

        Ok(total_draws)
    }

    // Every caster in the scene, drawn into the slice `enc` targets: static
    // objects, then each cluster's instances, then the skinned meshes.
    fn encode_spot_casters(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        bind: &SpotSliceBinding,
        cam_pos: [f32; 3],
        skinned_joint_bufs: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    ) -> u32 {
        self.encode_spot_static_into(enc, bind, cam_pos)
            + self.encode_spot_instanced_into(enc, bind, cam_pos)
            + self.encode_spot_skinned_into(enc, bind, cam_pos, skinned_joint_bufs)
    }

    // Apply the bindings every caster sub-path needs (shadow pipeline, depth
    // state, ShadowUniforms at vertex buffer 0, the slice index at vertex buffer
    // 7, and the shared vertex buffer at binding 1).
    fn bind_spot_slice(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        bind: &SpotSliceBinding,
    ) {
        enc.set_pipeline(bind.pipeline);
        enc.set_depth_stencil(&self.depth_state);
        enc.setDepthBias_slopeScale_clamp(
            shadow_bias::RASTER_CONSTANT,
            shadow_bias::RASTER_SLOPE,
            shadow_bias::RASTER_CLAMP,
        );
        enc.set_vertex_value(bind.uniforms, 0);
        enc.set_vertex_value(
            &ShadowPassPush {
                cascade_idx: SPOT_SLICE_IDX,
                _pad: [0; 3],
            },
            7,
        );
        enc.set_vertex_buffer(&self.vertex_buffer, 0, 1);
    }

    // Encode the static-geometry caster draws.
    fn encode_spot_static_into(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        bind: &SpotSliceBinding,
        cam_pos: [f32; 3],
    ) -> u32 {
        enc.pushDebugGroup(&objc2_foundation::NSString::from_str("spot shadow static"));
        self.bind_spot_slice(enc, bind);

        let mut draw_calls: u32 = 0;
        for obj in &self.draw.objects {
            if !obj.visible || !obj.resident {
                continue;
            }
            let model_uniforms = ModelUniforms { model: obj.model };
            enc.set_vertex_value(&model_uniforms, 2);
            // Pick the LOD by camera distance -- the shadow pass uses the
            // same slice the main pass will, so silhouettes track when the
            // runtime swaps to a coarser LOD.
            let d = crate::gfx::lod::camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            let index_byte_offset = index_offset * std::mem::size_of::<u32>();
            // SAFETY: `index_byte_offset` and `index_count` come from `active_lod`, which returns a
            // range inside this object's own slice of `self.index_buffer`, and `base_vertex` is
            // that object's own base.
            unsafe {
                enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                    MTLPrimitiveType::Triangle,
                    index_count,
                    MTLIndexType::UInt32,
                    &self.index_buffer,
                    index_byte_offset,
                    1,
                    obj.base_vertex as isize,
                    0,
                );
            }
            draw_calls += 1;
        }
        enc.popDebugGroup();
        draw_calls
    }

    // Encode caster draws for instanced clusters by iterating per-instance using
    // the (non-instanced) shadow pipeline. Cheap to ship and visually identical
    // to an instanced shadow shader. Off-screen instances can still cast shadows
    // onto visible surfaces, so no cluster-level cull here.
    fn encode_spot_instanced_into(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        bind: &SpotSliceBinding,
        cam_pos: [f32; 3],
    ) -> u32 {
        if self.instanced.clusters.is_empty() {
            return 0;
        }
        enc.pushDebugGroup(&objc2_foundation::NSString::from_str(
            "spot shadow instanced",
        ));
        self.bind_spot_slice(enc, bind);

        let mut draw_calls: u32 = 0;
        for cluster in &self.instanced.clusters {
            // Shadows only read each bucket's matrices (per-instance vertex
            // bytes), so borrow them: no LOD-bucket clone, which otherwise
            // recurred once per slice.
            cluster.for_each_lod_bucket(cam_pos, |index_offset, index_count, instances| {
                let index_byte_offset = index_offset * std::mem::size_of::<u32>();
                for &model in instances {
                    let model_uniforms = ModelUniforms { model };
                    enc.set_vertex_value(&model_uniforms, 2);
                    // SAFETY: the index range comes from `active_lod` on this object's own slice of
                    // `self.index_buffer`.
                    unsafe {
                        enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                            MTLPrimitiveType::Triangle,
                            index_count,
                            MTLIndexType::UInt32,
                            &self.index_buffer,
                            index_byte_offset,
                        );
                    }
                    draw_calls += 1;
                }
            });
        }
        enc.popDebugGroup();
        draw_calls
    }

    // Encode caster draws for skinned meshes (deformed depth, drawn last in the
    // slice).
    fn encode_spot_skinned_into(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        bind: &SpotSliceBinding,
        cam_pos: [f32; 3],
        skinned_joint_bufs: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    ) -> u32 {
        let mut draw_calls: u32 = 0;
        let (Some(skinned_shadow_ps), Some(svb), Some(sib)) = (
            &self.skinned.shadow_pipeline_state,
            &self.skinned.vertex_buffer,
            &self.skinned.index_buffer,
        ) else {
            return draw_calls;
        };
        if self.skinned.draw_objects.is_empty() {
            return draw_calls;
        }
        enc.pushDebugGroup(&objc2_foundation::NSString::from_str("spot shadow skinned"));
        // The skinned path needs the same uniforms and depth state as the others
        // but its own pipeline, so it binds the shared state with the skinned
        // pipeline swapped in.
        self.bind_spot_slice(
            enc,
            &SpotSliceBinding {
                pipeline: skinned_shadow_ps,
                uniforms: bind.uniforms,
            },
        );
        enc.set_vertex_buffer(svb, 0, 1);
        for (i, obj) in self.skinned.draw_objects.iter().enumerate() {
            if !obj.visible {
                continue;
            }
            let model_uniforms = ModelUniforms { model: obj.model };
            let d = crate::gfx::lod::skinned_camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            let index_byte_offset = index_offset * std::mem::size_of::<u32>();
            enc.set_vertex_value(&model_uniforms, 2);
            enc.set_vertex_buffer(&skinned_joint_bufs[i], 0, 8);
            // SAFETY: the index range is this object's own slice of the skinned index buffer.
            unsafe {
                enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                    MTLPrimitiveType::Triangle,
                    index_count,
                    MTLIndexType::UInt32,
                    sib,
                    index_byte_offset,
                );
            }
            draw_calls += 1;
        }
        enc.popDebugGroup();
        draw_calls
    }

    // A one-matrix `ShadowUniforms` carrying `slice`'s light-space projection in
    // slot 0, so the shared shadow vertex shader can render a spot slice without
    // a second pipeline or a second uniform layout.
    fn spot_slice_uniforms(&self, slice: u32) -> ShadowUniforms {
        let data = self.spot_shadow_data(slice);
        let mut uniforms = crate::gfx::csm::empty_shadow_uniforms();
        uniforms.light_vps[SPOT_SLICE_IDX as usize] = data.light_vp;
        uniforms.active_cascades = 1;
        uniforms
    }

    // Read slice `slice`'s projection back from the uploaded buffer. The buffer
    // is Shared storage and written once at init, so this is a plain read of
    // memory the GPU only ever reads.
    fn spot_shadow_data(&self, slice: u32) -> SpotShadowData {
        debug_assert!(slice < self.spot_shadow.count);
        // SAFETY: the buffer was created from a `&[SpotShadowData]` of exactly
        // `spot_shadow.count` elements and is never resized; `slice` is bounded
        // by that count above.
        unsafe {
            let base = self.spot_shadow.buffer.contents().as_ptr() as *const SpotShadowData;
            *base.add(slice as usize)
        }
    }
}
