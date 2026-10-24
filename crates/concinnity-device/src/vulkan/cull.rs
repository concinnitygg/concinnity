//! Compute-driven cull pass for the Vulkan backend. One compute
//! invocation per build-time `DrawObject` frustum / distance-tests the
//! object, and the packing (`vulkan/compact.rs`) writes the kept ones'
//! `VkDrawIndexedIndirectCommand`s into one list per stream of this frame's
//! indirect buffer. The bindless Main pass then draws each list with one
//! indirect draw. Must run outside any render pass
//! (Vulkan disallows compute dispatch inside a render pass), which is
//! why the graph dispatch site in `vulkan/draw.rs::record_frame` sits
//! before `cmd_begin_render_pass` for Shadow / Main.
//!
//! The shape mirrors `metal/cull.rs::encode_cull`; the graph executor
//! in [`graph_exec.rs`](graph_exec.rs) dispatches `PassId::Cull` here.
//!
//! CPU-side per-frame buffer rebuilds (`build_record_buffers`) stay in
//! `record_frame`: they're host
//! writes to mapped GPU memory, not part of the GPU command stream the
//! graph orders.

use ash::vk;
use concinnity_core::math::{SaturatingFrom, saturating_u32};
use concinnity_engine::gfx::cull_status::CullStatus;
use concinnity_engine::gfx::frustum::Frustum;
use concinnity_engine::gfx::render_types::{self, GpuCompactParams, StreamLayout};
use concinnity_engine::render::uniforms::vulkan::{CullHizParams, CullParams};

use crate::vulkan::record::cmd_push_constants;

use super::allocator::PooledBuffer;
use super::compact::{CompactPass, DrawLists};
use super::context::VkContext;

// Byte stride of one `VkDrawIndexedIndirectCommand` in the cull kernel's output.
pub(in crate::vulkan) const INDIRECT_COMMAND_STRIDE: u32 =
    saturating_u32(size_of::<vk::DrawIndexedIndirectCommand>());

// The most views one shadow pass culls: its cascades or its spot slices.
const MAX_SHADOW_VIEWS: usize =
    if render_types::NUM_SHADOW_CASCADES > render_types::MAX_SHADOWED_SPOTS {
        render_types::NUM_SHADOW_CASCADES
    } else {
        render_types::MAX_SHADOWED_SPOTS
    };

// One shadow view of one frame: the shadow kernel's set over the view's group
// counts and status, the packing's set over them and the view's lists, and
// the lists its draws read.
pub(in crate::vulkan) struct VkShadowView {
    pub(in crate::vulkan) cull_set: vk::DescriptorSet,
    pub(in crate::vulkan) compact_set: vk::DescriptorSet,
    pub(in crate::vulkan) lists: PooledBuffer,
    pub(in crate::vulkan) _status: PooledBuffer,
    pub(in crate::vulkan) _group_counts: PooledBuffer,
    // Commands `lists` holds ahead of its count slots.
    pub(in crate::vulkan) commands: usize,
}

impl VkShadowView {
    pub(in crate::vulkan) fn lists(&self) -> DrawLists {
        DrawLists {
            buffer: self.lists.buffer(),
            commands: self.commands,
        }
    }
}

impl VkContext {
    // Shader-bucket regions the cull kernel routes between: the world default
    // program plus one per material-referenced world shader. 1 when the world
    // declares no extra shaders, which collapses the indirect buffer to the
    // single region every pass used before buckets existed.
    pub(in crate::vulkan) fn shader_bucket_count(&self) -> usize {
        self.cull.world_pipelines.bucket_count()
    }

    // The main pipeline a probe or planar face draws its one command stream
    // under: the default's fullest grass variant when the world has one, so a
    // far band seen in a face keeps its canopy, else the default's.
    pub(in crate::vulkan) fn face_bindless_pipeline(
        &self,
    ) -> Option<&crate::vulkan::owned::OwnedPipeline> {
        let superset = self
            .cull
            .world_pipelines
            .superset_bucket()
            .and_then(|bucket| self.cull.world_pipelines.get(bucket as usize));
        match superset {
            Some(pipelines) => Some(&pipelines.main),
            None => self.cull.bindless_pipeline.as_ref(),
        }
    }

    // True when two-pass Hi-Z occlusion runs this frame: the world requested
    // `occlusion_two_pass`, the phase-2 cull pipeline + Hi-Z + second indirect
    // buffers are built, and the bindless GPU-cull path is active with
    // build-time geometry. This is the exact condition under which the shared
    // graph inserts the HizBuild / Cull2 / Main2 chain, so the frame-graph seed
    // (`record_frame`), the phase-1 render-pass selection (`encode_main_pass`),
    // and the executor's phase-2 arms all gate on it identically. Mirrors
    // `directx/cull.rs::two_pass_occlusion_active`.
    pub(in crate::vulkan) fn two_pass_occlusion_active(&self) -> bool {
        self.two_pass_occlusion_built() && self.state.draw.record_count() > 0
    }

    // Whether launch built two-pass occlusion: requested, with the phase-two
    // cull, the Hi-Z pyramid and the second indirect buffers.
    pub(in crate::vulkan) fn two_pass_occlusion_built(&self) -> bool {
        self.cull.occlusion_two_pass
            && self
                .cull
                .cull_kernels
                .as_ref()
                .is_some_and(|k| k.pipeline_phase2.is_some())
            && self.cull.hiz.is_some()
            && !self.cull.indirect_buffers2.is_empty()
    }

    // Dispatch the compute-driven cull pass for frame slot
    // `frame_idx`. Ends with a memory barrier ordering the kernel's
    // SSBO writes (`SHADER_WRITE`) before the bindless main pass's
    // `cmd_draw_indexed_indirect` reads (`INDIRECT_COMMAND_READ`).
    // A no-op when the cull pipeline isn't built (geometry-less
    // worlds or a world that opted out of bindless cull).
    //
    // The caller (`record_frame`) must rebuild this frame's
    // `object_buffer` + `draw_args_buffer` host-side before this runs;
    // those are mapped-memory writes that don't belong in the GPU
    // command stream.
    pub(in crate::vulkan) fn encode_cull(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        frustum: &Frustum,
        cam_pos: [f32; 3],
    ) {
        let Some(kernels) = self.cull.cull_kernels.as_ref() else {
            return;
        };
        let (pipeline, layout) = (&kernels.pipeline, &kernels.pipeline_layout);
        let device = &self.hw.device;

        // Pack the six already-normalized frustum planes for the kernel.
        let mut params = CullParams {
            planes: [[0.0; 4]; 6],
            cam_pos,
            object_count: u32::saturating_from(self.state.draw.record_count()),
            bucket_count: u32::saturating_from(self.shader_bucket_count()),
            skinned_base: u32::saturating_from(self.state.draw.skinned_record_base()),
        };
        params.planes = frustum.packed_planes();
        // SAFETY: `cmd` is in the recording state, and the cull pipeline is owned by
        // `self.cull.cull_kernels`, which outlives this recording.
        unsafe { device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.handle()) };
        // SAFETY: the frame's cull set is allocated from the live cull descriptor pool and matches
        // the kernel layout bound with the pipeline above.
        unsafe {
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout.handle(),
                0,
                std::slice::from_ref(&self.cull.cull_sets[frame_idx]),
                &[],
            );
        }
        // Hi-Z occlusion set (set 1): the depth pyramid sampler + this
        // frame's CullHizParams (previous-frame VP + pyramid dims +
        // validity gate). `Some` whenever the cull pipeline is (same
        // gating). `hiz_enabled` is 0 until a pyramid at the current
        // resolution exists, so the kernel falls back to frustum + distance.
        if let Some(hiz) = self.cull.hiz.as_ref() {
            let params = CullHizParams {
                prev_view_proj: self.state.planner.temporal().occlusion().prev_view_proj(),
                hiz_size: [hiz.width as f32, hiz.height as f32],
                hiz_mip_count: hiz.mip_count,
                hiz_enabled: u32::from(self.state.planner.temporal().occlusion().valid()),
            };
            hiz.cull_ubos[frame_idx].write_val(0, &params);
            // SAFETY: the Hi-Z read set is owned by `self.cull.hiz`, which lives for the frame,
            // and its UBO was just written on the host before submission.
            unsafe {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout.handle(),
                    1,
                    std::slice::from_ref(&hiz.read_sets[frame_idx]),
                    &[],
                );
            }
        }
        // SAFETY: `layout` belongs to the same live kernel set as the bound pipeline, and `params`
        // is borrowed for the call.
        unsafe {
            cmd_push_constants(
                device,
                cmd,
                layout.handle(),
                vk::ShaderStageFlags::COMPUTE,
                &params,
            );
        }
        // One invocation per build-time object, 64-wide local groups.
        let groups = u32::saturating_from(self.state.draw.record_count()).div_ceil(64);
        // SAFETY: the pipeline, both sets and the push constants above are bound on this recording
        // command buffer.
        unsafe { device.cmd_dispatch(cmd, groups, 1, 1) };
        if let (Some(&set), Some(lists)) = (
            self.cull.compact_sets.get(frame_idx),
            self.cull.indirect_buffers.get(frame_idx),
        ) {
            self.encode_compaction(
                cmd,
                CompactPass {
                    set,
                    lists: self.main_lists(lists.buffer()),
                    params: self.compact_params(CullStatus::DRAWN),
                },
            );
        }
    }

    // The main view's lists in `buffer`, a frame's phase-1 or phase-2
    // indirect buffer.
    pub(in crate::vulkan) fn main_lists(&self, buffer: vk::Buffer) -> DrawLists {
        DrawLists {
            buffer,
            commands: self.cull.indirect_commands,
        }
    }

    // What the main view's packing packs: this frame's records, keeping those
    // with outcome `drawn`.
    fn compact_params(&self, drawn: u32) -> GpuCompactParams {
        GpuCompactParams {
            object_count: u32::saturating_from(self.state.draw.record_count()),
            bucket_count: u32::saturating_from(self.shader_bucket_count()),
            skinned_base: u32::saturating_from(self.state.draw.skinned_record_base()),
            drawn,
        }
    }

    // How the main view's lists sit in its indirect buffers.
    pub(in crate::vulkan) fn stream_layout(&self) -> StreamLayout<'_> {
        StreamLayout {
            regions: &self.cull.variants.regions,
            bucket_count: u32::saturating_from(self.shader_bucket_count()),
            skinned_base: u32::saturating_from(self.state.draw.skinned_record_base()),
            records: u32::saturating_from(self.state.draw.record_count()),
        }
    }

    // Per-cascade GPU cull for the GPU-driven shadow pass: each re-rendered
    // cascade's frustum test of every record (static + instances + skinned)
    // against that cascade's light frustum (extracted from `light_vps[c]`; no
    // Hi-Z), packed into the cascade's lists. Must run outside any render pass,
    // so the caller dispatches it at the top of `encode_shadow_pass`. Mirrors
    // `directx/cull.rs::encode_shadow_culls`.
    pub(in crate::vulkan) fn encode_shadow_culls(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        render_mask: u32,
        cam_pos: [f32; 3],
    ) {
        let Some(views) = self.cull.shadow_views.get(frame_idx) else {
            return;
        };
        let mut culls = [None; render_types::NUM_SHADOW_CASCADES];
        let mut kept = 0;
        for (c, view) in views.iter().enumerate() {
            if render_mask & (1u32 << c) == 0 {
                continue;
            }
            let light_vp = self.state.planner.shadow.uniforms.light_vps[c];
            culls[kept] = Some((view, Frustum::from_shadow(light_vp)));
            kept += 1;
        }
        self.encode_view_culls(cmd, &culls[..kept], cam_pos);
    }

    // Per-slice GPU cull for the spot shadow pass: the same shadow kernel, each
    // refreshed slice's casters kept against that spot's light frustum into the
    // slice's own lists. The kernel picks no LOD: each caster draws the
    // camera-keyed LOD the frame's draw args already carry.
    pub(in crate::vulkan) fn encode_spot_culls(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        cam_pos: [f32; 3],
    ) {
        let Some(views) = self.cull.spot_views.get(frame_idx) else {
            return;
        };
        let mut culls = [None; render_types::MAX_SHADOWED_SPOTS];
        let mut kept = 0;
        for slice in self.spot_shadow.refreshed_slices() {
            let slice = slice as usize;
            culls[kept] = views
                .get(slice)
                .map(|view| (view, self.spot_shadow.frusta[slice]));
            kept += 1;
        }
        self.encode_view_culls(cmd, &culls[..kept], cam_pos);
    }

    // What a shadow view's cull and packing cover: this frame's records, at
    // most as many as the view holds, one stream and the skinned tail.
    pub(in crate::vulkan) fn shadow_layout(&self, view: &VkShadowView) -> StreamLayout<'_> {
        StreamLayout {
            regions: &self.cull.variants.regions,
            bucket_count: 1,
            skinned_base: u32::saturating_from(self.state.draw.skinned_record_base()),
            records: render_types::culled_records(self.state.draw.record_count(), view.commands),
        }
    }

    // One shadow-kernel dispatch per view, each testing every record against
    // the view's frustum, then every view's packing into its lists. Ends with
    // one memory barrier ordering the lists' writes before the shadow draws'
    // indirect reads. A no-op when the GPU-driven shadow resources are absent,
    // `draw.record_count() == 0`, or no view asked for.
    fn encode_view_culls(
        &self,
        cmd: vk::CommandBuffer,
        views: &[Option<(&VkShadowView, Frustum)>],
        cam_pos: [f32; 3],
    ) {
        let (Some(pipeline), Some(layout)) = (
            self.cull.shadow_cull_pipeline.as_ref(),
            self.cull.shadow_cull_pipeline_layout.as_ref(),
        ) else {
            return;
        };
        if self.state.draw.record_count() == 0 || views.is_empty() {
            return;
        }
        let device = &self.hw.device;

        // SAFETY: `cmd` is in the recording state, and the shadow cull pipeline is owned by
        // `self.cull`, which outlives this recording.
        unsafe { device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.handle()) };
        let mut passes = [CompactPass::default(); MAX_SHADOW_VIEWS];
        let mut packed_views = 0;
        for (pass, (view, frustum)) in passes.iter_mut().zip(views.iter().flatten()) {
            let packed = self.shadow_layout(view);
            let mut params = CullParams {
                planes: [[0.0; 4]; 6],
                cam_pos,
                object_count: packed.records,
                bucket_count: 1,
                skinned_base: packed.skinned_base,
            };
            params.planes = frustum.packed_planes();
            // SAFETY: the view's cull set was built against the shadow cull layout.
            unsafe {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout.handle(),
                    0,
                    std::slice::from_ref(&view.cull_set),
                    &[],
                );
            }
            // SAFETY: `layout` is the live shadow cull layout bound with the pipeline above, and
            // `params` is borrowed for the call.
            unsafe {
                cmd_push_constants(
                    device,
                    cmd,
                    layout.handle(),
                    vk::ShaderStageFlags::COMPUTE,
                    &params,
                );
            }
            // SAFETY: the pipeline, this view's set and its push constants are bound on the
            // recording command buffer.
            unsafe { device.cmd_dispatch(cmd, packed.records.div_ceil(64), 1, 1) };
            *pass = CompactPass {
                set: view.compact_set,
                lists: view.lists(),
                params: GpuCompactParams {
                    object_count: packed.records,
                    bucket_count: 1,
                    skinned_base: packed.skinned_base,
                    drawn: CullStatus::DRAWN,
                },
            };
            packed_views += 1;
        }
        self.encode_compactions(cmd, &passes[..packed_views]);
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ);
        // SAFETY: a global memory barrier on the recording command buffer; `barrier` is borrowed
        // for the call.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }

    // Dispatch the phase-2 (two-pass occlusion) cull for frame slot
    // `frame_idx`. Runs after `HizBuild` has rebuilt the Hi-Z pyramid from this
    // frame's phase-1 depth; re-tests only the objects phase-1 cull marked
    // `STATUS_HIZ_CANDIDATE` against the fresh pyramid (projected through this
    // frame's un-jittered VP) and writes a draw for any that turn out visible
    // into the phase-2 indirect buffer `Main2` consumes. A no-op unless the
    // phase-2 pipeline + sets are built (two-pass occlusion active). Mirrors
    // `directx/cull.rs::encode_cull_phase2`.
    pub(in crate::vulkan) fn encode_cull_phase2(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        frustum: &Frustum,
        cam_pos: [f32; 3],
        cur_vp: [[f32; 4]; 4],
    ) {
        let (Some(kernels), Some(hiz)) = (self.cull.cull_kernels.as_ref(), self.cull.hiz.as_ref())
        else {
            return;
        };
        let Some(pipeline) = kernels.pipeline_phase2.as_ref() else {
            return;
        };
        let layout = &kernels.pipeline_layout;
        if self.cull.cull_sets2.is_empty() || self.state.draw.record_count() == 0 {
            return;
        }
        let device = &self.hw.device;

        // Frustum planes + camera position are unused by the phase-2 kernel
        // (candidates already passed those in phase 1) but the push-constant
        // block layout is shared with phase 1, so pack them anyway.
        let mut params = CullParams {
            planes: [[0.0; 4]; 6],
            cam_pos,
            object_count: u32::saturating_from(self.state.draw.record_count()),
            bucket_count: u32::saturating_from(self.shader_bucket_count()),
            skinned_base: u32::saturating_from(self.state.draw.skinned_record_base()),
        };
        params.planes = frustum.packed_planes();

        // Project AABBs through this frame's un-jittered VP against the pyramid
        // `HizBuild` just rebuilt from this frame's depth. `hiz_enabled = 1`:
        // HizBuild always precedes this dispatch, so a valid pyramid is
        // guaranteed (the kernel still guards defensively).
        let hiz_params = CullHizParams {
            prev_view_proj: cur_vp,
            hiz_size: [hiz.width as f32, hiz.height as f32],
            hiz_mip_count: hiz.mip_count,
            hiz_enabled: 1,
        };
        hiz.cull_ubos2[frame_idx].write_val(0, &hiz_params);

        // SAFETY: `cmd` is in the recording state, and the cull pipeline is owned by `self.cull`,
        // which outlives this recording.
        unsafe { device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.handle()) };
        // SAFETY: the second-pass cull set for this frame is allocated from the live cull pool and
        // matches `layout`.
        unsafe {
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout.handle(),
                0,
                std::slice::from_ref(&self.cull.cull_sets2[frame_idx]),
                &[],
            );
        }
        // SAFETY: the Hi-Z read set is owned by `hiz`, which outlives the frame, and its UBO was
        // written on the host above.
        unsafe {
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout.handle(),
                1,
                std::slice::from_ref(&hiz.read_sets2[frame_idx]),
                &[],
            );
        }
        // SAFETY: `layout` is the live kernel layout bound with the pipeline, and `params` is
        // borrowed for the call.
        unsafe {
            cmd_push_constants(
                device,
                cmd,
                layout.handle(),
                vk::ShaderStageFlags::COMPUTE,
                &params,
            );
        }
        let groups = u32::saturating_from(self.state.draw.record_count()).div_ceil(64);
        // SAFETY: the pipeline, both sets and the push constants above are bound on this recording
        // command buffer.
        unsafe { device.cmd_dispatch(cmd, groups, 1, 1) };
        if let (Some(&set), Some(lists)) = (
            self.cull.compact_sets2.get(frame_idx),
            self.cull.indirect_buffers2.get(frame_idx),
        ) {
            self.encode_compaction(
                cmd,
                CompactPass {
                    set,
                    lists: self.main_lists(lists.buffer()),
                    params: self.compact_params(CullStatus::REDRAW),
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use concinnity_core::math::SaturatingFrom;
    // The `CullParams` layout test lives with the struct in
    // `concinnity_engine::render::uniforms::vulkan`. The struct-size == push-range
    // cross-check stays here, where `CULL_PUSH_CONSTANT_BYTES` is defined.
    #[test]
    fn cull_params_size_matches_push_range() {
        assert_eq!(
            u32::saturating_from(size_of::<super::CullParams>()),
            super::super::pipeline::CULL_PUSH_CONSTANT_BYTES
        );
    }
}
