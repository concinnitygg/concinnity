//! Shadow pass for the Vulkan backend: one depth-only render pass per
//! cascade slice of the shadow-map array. Both of `shadow_map`'s transitions are
//! graph-driven: `shadow_map` is the render graph's `shadow_map` resource, so the
//! executor emits (over every cascade layer) the Shadow producer barrier
//! (`SHADER_READ_ONLY_OPTIMAL` -> `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`, the
//! cross-frame reset for this frame's shadow loop) before this pass and the Main
//! consumer barrier (`DEPTH_STENCIL_ATTACHMENT_OPTIMAL` -> `SHADER_READ_ONLY_OPTIMAL`,
//! letting the main pass sample the cascades) before the Main pass. The map rests
//! sampled between frames, so there is no inline reset.
//!
//! The cascades are GPU-driven: a per-cascade cull dispatch and its packing
//! write each cascade's lists, and each cascade is issued with one indirect
//! draw of the kept static + instance records and one of the kept skinned tail.
//! Streamed chunks and runtime clones ride the same records, so the CPU never
//! walks a caster list here. The spot slices in
//! [`spot_shadow.rs`](spot_shadow.rs) draw through the same `draw_shadow_view`
//! from their own lists.
//!
//! The shape mirrors `metal/draw/shadow.rs::encode_shadow_pass`; the
//! graph executor in [`graph_exec.rs`](graph_exec.rs) dispatches
//! `PassId::Shadow` here.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types::{self, SKINNED_STREAM, ShadowPassPush};
use concinnity_engine::render::uniforms::grass::GrassCascadeGpu;

use super::super::context::VkContext;
use crate::vulkan::depth;
use crate::vulkan::owned::{OwnedFramebuffer, OwnedPipeline, OwnedRenderPass};
use crate::vulkan::record::Recorder;

// One depth-only shadow view to draw from its packed lists: the set 0 holding
// its `ShadowUniforms`, which of their `light_vps` the vertex shader projects
// through, and the view its lists live in. A cascade and a spot slice differ
// only in these.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct ShadowView<'a> {
    pub uniforms_set: vk::DescriptorSet,
    pub vp_index: u32,
    pub view: &'a crate::vulkan::cull::ShadowCullView,
    // The depth-only pipeline built against the render pass the view draws in.
    pub pipeline: Option<&'a OwnedPipeline>,
}

impl VkContext {
    // Encode the cascade shadow passes for frame slot `frame_idx`, one render pass
    // per cascade. A no-op with shadows off; the caller uploaded the cascade VPs.
    pub(in crate::vulkan) fn encode_shadow_pass(
        &self,
        rec: &Recorder<'_>,
        frame_idx: usize,
        elapsed: f32,
        // This frame's grass, whose cascade blades draw into the nearest
        // cascade after its rasterized casters, when it casts this frame.
        grass: Option<&crate::vulkan::grass::GrassFrame>,
    ) {
        if !self.shadow.enabled() {
            return;
        }

        // Raymarched SDF shadow casters share these cascade DSVs: upload this
        // frame's animation time once (no-op without casters) so the from-light
        // SDF march lines up with the lit-side surface.
        self.upload_raymarch_shadow_view(frame_idx, elapsed);

        // Cascades to re-render this frame; a skipped one keeps its depth and VP. The 0
        // sentinel means all.
        let all_cascades = (1u32 << render_types::NUM_SHADOW_CASCADES) - 1;
        let render_mask = if self.state.planner.shadow.cascade_mask == 0 {
            all_cascades
        } else {
            self.state.planner.shadow.cascade_mask
        };

        // Nothing to cull means nothing to draw: the render passes below still
        // run, so every re-rendered cascade is cleared for the raymarched
        // casters that follow the rasterized ones.
        let gpu_driven = self.shadow_views_drawable();

        // GPU-driven cull prologue: dispatch every re-rendered cascade's cull
        // before opening any render pass (Vulkan disallows compute inside a
        // render pass). Each packs that cascade's lists.
        if gpu_driven {
            self.encode_shadow_culls(rec, frame_idx, render_mask);
        }

        for (cascade_idx, shadow_fb) in self.shadow.framebuffers.iter().enumerate() {
            if render_mask & (1u32 << cascade_idx) == 0 {
                continue;
            }
            Self::begin_shadow_slice(
                rec,
                &self.shadow.render_pass,
                shadow_fb,
                self.shadow.map_size,
            );

            if gpu_driven
                && let Some(view) = self
                    .cull
                    .shadow_views
                    .get(frame_idx)
                    .and_then(|c| c.get(cascade_idx))
            {
                self.draw_shadow_view(
                    rec,
                    frame_idx,
                    ShadowView {
                        uniforms_set: self.descriptors.shadow_global_sets[frame_idx],
                        vp_index: u32::saturating_from(cascade_idx),
                        view,
                        pipeline: self.cull.shadow_bindless_pipeline.as_ref(),
                    },
                );
            }

            let grass_mask = self.state.planner.shadow.grass_mask;
            if grass_mask & (1u32 << cascade_idx) != 0
                && let Some(frame) = grass
            {
                let cascade = GrassCascadeGpu {
                    light_vp: self.state.planner.shadow.uniforms.light_vps[cascade_idx],
                };
                let first = grass_mask.trailing_zeros() as usize == cascade_idx;
                self.encode_grass_shadow_draw(rec, frame_idx, frame, &cascade, first);
            }

            // Raymarched SDF shadow casters into this cascade's DSV, after the
            // rasterized casters and within the same render pass (no re-clear);
            // the depth write test keeps the nearer occluder.
            self.encode_sdf_shadow_cascade(rec, frame_idx, cascade_idx);
            rec.end_render_pass();
        }

        // The shadow map's transitions are graph-driven; it rests sampled between
        // frames.
    }

    // Whether the GPU-driven shadow path has a pipeline and records to draw.
    pub(in crate::vulkan) fn shadow_views_drawable(&self) -> bool {
        self.cull.shadow_bindless_pipeline.is_some() && self.state.draw.record_count() > 0
    }

    // Open the depth-only shadow render pass `render_pass` on one square `size`
    // slice, which clears it, with the viewport and scissor over it.
    pub(in crate::vulkan) fn begin_shadow_slice(
        rec: &Recorder<'_>,
        render_pass: &OwnedRenderPass,
        framebuffer: &OwnedFramebuffer,
        size: u32,
    ) {
        let extent = vk::Extent2D {
            width: size,
            height: size,
        };
        let clear_depth = depth::CLEAR_VALUE;
        rec.begin_render_pass(
            render_pass,
            framebuffer,
            vk::Rect2D::default().extent(extent),
            std::slice::from_ref(&clear_depth),
        );
        // Negative-height viewport: Y-flips NDC so Y-up matches Metal, and
        // matches the `-ndc.y` the forward sampler applies.
        let vp = vk::Viewport {
            x: 0.0,
            y: size as f32,
            width: size as f32,
            height: -(size as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        rec.set_viewport(&vp);
        let scissor = vk::Rect2D::default().extent(extent);
        rec.set_scissor(&scissor);
    }

    // One shadow view inside the caller's render pass: the depth-only bindless
    // pipeline draws the view's kept records and skinned tail by count, through
    // `light_vps[view.vp_index]`.
    pub(in crate::vulkan) fn draw_shadow_view(
        &self,
        rec: &Recorder<'_>,
        frame_idx: usize,
        view: ShadowView<'_>,
    ) {
        let (Some(sb_pipeline), Some(sb_layout)) = (
            view.pipeline,
            self.cull.shadow_bindless_pipeline_layout.as_ref(),
        ) else {
            return;
        };
        let layout = self.shadow_layout(view.view);
        let lists = view.view.lists();

        rec.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, sb_pipeline);
        rec.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            sb_layout,
            0,
            &[view.uniforms_set, self.cull.bindless_sets[frame_idx]],
            &[],
        );
        let push = ShadowPassPush {
            cascade_idx: view.vp_index,
            _pad: [0; 3],
            tile_morph: self.state.tile_morph_view(),
        };
        rec.push_constants(sb_layout, vk::ShaderStageFlags::VERTEX, 0, &push);

        // Static + instance prefix against the static VB/IB.
        rec.bind_vertex_buffers(0, &[self.geometry.vertex_buffer.buffer()], &[0]);
        rec.bind_index_buffer(
            self.geometry.index_buffer.buffer(),
            0,
            vk::IndexType::UINT32,
        );
        let statics = layout.slots(0);
        if statics.capacity > 0 {
            self.draw_stream(rec, lists, 0, statics);
            self.inc_draw_calls(1);
        }

        // Skinned tail against the deformed VB + skinned IB.
        let skinned = layout.slots(SKINNED_STREAM);
        if skinned.capacity > 0
            && let Some(deformed) = self.skinned.deformed.get(frame_idx)
        {
            // The deformed VB was posed by the skin compute pass ahead of this one.
            rec.bind_vertex_buffers(0, std::slice::from_ref(&deformed.buffer), &[0]);
            rec.bind_index_buffer(self.skinned.index_buffer.buffer(), 0, vk::IndexType::UINT32);
            self.draw_stream(rec, lists, SKINNED_STREAM, skinned);
            self.inc_draw_calls(1);
        }
    }
}
