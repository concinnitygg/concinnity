// src/metal/draw/shadow.rs
//
// Cascaded shadow-map pass: one depth-only render per CSM cascade slice. Every
// caster (static, instanced, and the folded skinned tail) draws through that
// cascade's slice of the shadow ICB the cull's encode dispatch filled, so the
// pass issues at most two indirect draws per cascade and walks no draw list.
// Skipped entirely when no shadow pipeline is configured.
//
// Each cascade is its own render pass (`setSlice` targets a different
// shadow.map array slice) on a single `MTLRenderCommandEncoder`; see
// [`encode_main_pass`](../draw/main.rs) for why the earlier
// `MTLParallelRenderCommandEncoder` landing was reverted.
//
// Spot shadows cannot share the ICB (its slots are laid out per cascade), so
// their per-draw caster body lives in [`spot_shadow`](spot_shadow.rs).
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLLoadAction,
    MTLRenderCommandEncoder as _, MTLRenderPassDescriptor, MTLStoreAction,
};

use crate::gfx::render_types::{NUM_SHADOW_CASCADES, ShadowPassPush};
use crate::gfx::shadow_bias;
use crate::metal::context::MtlContext;
use crate::metal::encode::RenderEncode;
use crate::metal::scoped_encoder::ScopedEncoder;

impl MtlContext {
    // Choose which shadow cascades to re-render this frame and advance the
    // round-robin clock. Delegates to the shared `ShadowCascadeScheduler`
    // (`gfx::shadow_schedule`, unit-tested there). Called once per frame from
    // draw_frame; the result is stashed in `shadow.render_mask` for
    // encode_shadow_pass and used to gate which cascade VPs refresh.
    pub(in crate::metal) fn next_shadow_cascade_mask(&mut self) -> u32 {
        self.shadow
            .scheduler
            .next_mask(self.shadow.update, self.shadow.cascades)
    }

    // pub(in crate::metal) so the render-graph executor in
    // metal/graph_exec.rs can dispatch this pass from a CompiledGraph.
    pub(in crate::metal) fn encode_shadow_pass(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        // The per-frame `GpuObjectData` buffer the bindless shadow VS reads each
        // cascade's model matrix from (by the `[[base_instance]]` record id the
        // shadow cull baked). `Some` exactly when the cull ran this frame; a
        // world with nothing in the cull records leaves it `None` and every
        // cascade is a bare depth clear.
        object_buffer: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        // This frame's skinned deformed-vertex buffer (skinned fold). Bound as
        // the vertex buffer for the skinned tail of each cascade's indirect
        // draw. `Some` only when the skinned fold is active.
        deformed_skinned: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        // When `Some`, raymarched SDF casters draw into each cascade after the
        // rasterised + skinned draws (and before the Main pass samples the
        // shadow map). Built by the graph executor: same matrix / time /
        // camera the main raymarch pass will use later this frame, so the
        // shadow cast and the live surface agree. `None` when no volume opts
        // into `cast_shadows`. Mirrors the DirectX shadow pass.
        raymarch_view: Option<&crate::metal::raymarch::RaymarchView>,
    ) -> Result<u32, String> {
        // The shadow map and its cascade set stand up together with the
        // depth-only pipeline; without it there is nothing to render into.
        if self.shadow.pipeline_state.is_none() {
            return Ok(0);
        }
        let mut total_draws: u32 = 0;

        // Cascades to re-render this frame; draw_frame computed the mask from
        // the update policy. A skipped cascade keeps the depth and VP from when
        // it was last rendered, so the Main pass still samples it consistently.
        // Defensive fallback to all cascades if no mask was set this frame.
        let all = (1u32 << NUM_SHADOW_CASCADES) - 1;
        let mask = if self.shadow.render_mask == 0 {
            all
        } else {
            self.shadow.render_mask
        };
        let rendered: Vec<usize> = (0..NUM_SHADOW_CASCADES)
            .filter(|i| mask & (1u32 << i) != 0)
            .collect();
        let first_rendered = rendered.first().copied();
        let last_rendered = rendered.last().copied();

        for &cascade_idx in &rendered {
            let shadow_pass_desc = MTLRenderPassDescriptor::new();
            let depth_attach = shadow_pass_desc.depthAttachment();
            depth_attach.setTexture(Some(self.shadow.map.as_ref()));
            depth_attach.setSlice(cascade_idx);
            depth_attach.setLoadAction(MTLLoadAction::Clear);
            depth_attach.setStoreAction(MTLStoreAction::Store);
            depth_attach.setClearDepth(1.0);

            // Per-pass GPU timing spans the first to the last cascade actually
            // rendered this frame (the set varies with the update policy):
            // attach the start sample to the first and the end to the last.
            if let Some(t) = &self.diagnostics.pass_timing {
                let id = super::super::pass_timing::PassId::Shadow;
                let is_first = Some(cascade_idx) == first_rendered;
                let is_last = Some(cascade_idx) == last_rendered;
                if is_first && is_last {
                    t.attach_render(&shadow_pass_desc, id);
                } else if is_first {
                    t.attach_render_first(&shadow_pass_desc, id);
                } else if is_last {
                    t.attach_render_last(&shadow_pass_desc, id);
                }
            }

            // Loop-local guard: each cascade's encoder ends when the guard drops
            // at the end of this iteration, before the next cascade opens one.
            // The descriptor clears the slice, so a cascade with no rasterised
            // casters still leaves a cleared depth for the SDF pass below.
            let shadow_enc = ScopedEncoder::new(
                cmd_buf
                    .renderCommandEncoderWithDescriptor(&shadow_pass_desc)
                    .ok_or("failed to get shadow render encoder")?,
                "shadow cascade",
            );

            if let Some(object_buffer) = object_buffer {
                let push = ShadowPassPush {
                    cascade_idx: cascade_idx as u32,
                    _pad: [0; 3],
                };
                // One (static+instance prefix) or two (+ skinned tail) indirect
                // draws over this cascade's slice of the shadow ICB.
                total_draws += self.encode_shadow_cascade_indirect(
                    &shadow_enc,
                    &push,
                    cascade_idx,
                    object_buffer,
                    deformed_skinned,
                );
            }
        }

        // Raymarched SDF shadow casters: depth-only draws into the same
        // per-cascade slices, run after the rasterised + skinned casters so
        // both layers compete via the slice's LESS depth test (nearest caster
        // wins per texel). No-op when no volume opts into `cast_shadows` or the
        // executor passed no view.
        if let Some(view) = raymarch_view {
            total_draws += self.encode_sdf_shadow_casters(cmd_buf, view)?;
        }

        Ok(total_draws)
    }

    // GPU-driven shadow draws for one cascade: execute this cascade's
    // slice of the shadow ICB the shadow cull's encode dispatch filled. Mirrors
    // the main pass's two-range split (`execute_bindless_static_icb`): one
    // indirect draw for the static + instance prefix (static VB bound at 1,
    // static u32 IB resident), then one for the folded skinned tail (deformed VB
    // rebound at 1, skinned IB resident). The depth-only bindless shadow VS
    // reads each record's model from the object buffer at vbuf 9 by
    // `[[base_instance]]`. Returns the indirect-draw count (1 or 2).
    fn encode_shadow_cascade_indirect(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        push: &ShadowPassPush,
        cascade_idx: usize,
        object_buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        deformed_skinned: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
    ) -> u32 {
        use objc2_metal::{MTLRenderStages, MTLResourceUsage};
        let (Some(pipeline), Some(icb)) = (
            self.cull.shadow_bindless_pipeline.as_ref(),
            self.cull.shadow_icb.as_ref(),
        ) else {
            return 0;
        };
        enc.pushDebugGroup(&objc2_foundation::NSString::from_str(
            "shadow cascade indirect",
        ));
        enc.set_pipeline(pipeline);
        enc.set_depth_stencil(&self.depth_state);
        enc.setDepthBias_slopeScale_clamp(
            shadow_bias::RASTER_CONSTANT,
            shadow_bias::RASTER_SLOPE,
            shadow_bias::RASTER_CLAMP,
        );
        // ShadowUniforms (vbuf 0), cascade push (vbuf 7), object buffer
        // (vbuf 9), static vertex buffer (vbuf 1). The ICB commands inherit
        // these bindings; the cull baked base_instance = record id, so the VS
        // reads `objects[id].model`.
        enc.set_vertex_value(&self.shadow.uniforms, 0);
        enc.set_vertex_value(push, 7);
        enc.set_vertex_buffer(object_buffer, 0, 9);
        enc.set_vertex_buffer(&self.vertex_buffer, 0, 1);

        // This cascade's command slots live at `[c*stride, c*stride + stride)`
        // in the shared shadow ICB (stride = the live record count, the same
        // value `encode_shadow_culls` used as `cascade_base`).
        let counts = self.draw_record_counts();
        let cascade_off = cascade_idx * counts.total;
        let mut draw_calls = 0u32;

        // Static + instance prefix.
        if let Some(prefix) = counts.prefix(cascade_off) {
            enc.useResource_usage_stages(
                ProtocolObject::from_ref(&*self.index_buffer),
                MTLResourceUsage::Read,
                MTLRenderStages::Vertex,
            );
            // SAFETY: the prefix spans this cascade's static + instance command
            // slots (ensure_shadow_icb_capacity sized the ICB for
            // NUM_SHADOW_CASCADES * cull_count).
            unsafe {
                enc.executeCommandsInBuffer_withRange(
                    icb.as_ref(),
                    crate::metal::context::ns_range(prefix),
                );
            }
            draw_calls += 1;
        }

        // Folded skinned tail: deformed VB at binding 1, skinned IB resident.
        if let (Some(deformed), Some(tail)) = (deformed_skinned, counts.skinned_tail(cascade_off)) {
            enc.set_vertex_buffer(deformed, 0, 1);
            if let Some(skinned_ib) = self.skinned.index_buffer.as_ref() {
                enc.useResource_usage_stages(
                    ProtocolObject::from_ref(&**skinned_ib),
                    MTLResourceUsage::Read,
                    MTLRenderStages::Vertex,
                );
            }
            // SAFETY: the tail spans this cascade's folded skinned command slots.
            unsafe {
                enc.executeCommandsInBuffer_withRange(
                    icb.as_ref(),
                    crate::metal::context::ns_range(tail),
                );
            }
            draw_calls += 1;
        }
        enc.popDebugGroup();
        draw_calls
    }
}
