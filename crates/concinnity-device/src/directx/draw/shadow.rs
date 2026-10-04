//! Cascaded shadow-map pass: one depth-only render per CSM cascade slice.
//! Draws static objects, instanced clusters, and (when present) skinned
//! meshes into each slice of the shadow map array. Caller has already
//! uploaded this frame's `ShadowUniforms` into `shadow_ubo_gva`; this pass
//! binds it and sets the cascade index per cascade. Skipped entirely when the
//! fallback 1x1 shadow array is bound.
//!
//! The cascades are GPU-driven: a per-cascade cull dispatch writes one
//! `ExecuteIndirect` region per cascade and each cascade is issued with a single
//! `ExecuteIndirect` over the static + skinned records (the same cull buffers the
//! main pass uses). Everything appearing after init -- streamed chunks and
//! spawned clones -- folds into those same records, so the whole scene is
//! covered. The spot slices draw through the same `draw_shadow_view` from their
//! own indirect buffer.

use concinnity_core::gfx::render_types::{NUM_SHADOW_CASCADES, ShadowUniforms};
use concinnity_core::render::backend_init::ShadowCadence;
use concinnity_core::render::depth::DepthConvention;
use concinnity_core::render::shadow_schedule;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::com;
use crate::directx::context::DxContext;
use crate::directx::descriptor_slot::SrvSlot;
use crate::directx::root_constants::RootConstants;
use crate::directx::texture::GpuResource;

// Shadow map resources. `resource` / `dsvs` are `None` / empty when the shadow
// pass is disabled (a 1x1 array fallback SRV is still bound at `srv_gpu`).
// `dsvs` is one DSV per cascade slice. `light_dir` is the world-space unit
// vector pointing toward the first directional light, captured at init from
// `light_uniforms` and used by per-frame CSM updates.
pub(in crate::directx) struct ShadowState {
    pub resource: Option<GpuResource<ID3D12Resource>>,
    pub dsvs: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub map_size: u32,
    pub srv_gpu: SrvSlot,
    pub light_dir: [f32; 3],
    pub cadence: ShadowCadence,
    // Round-robin clock + primed-set for the cascade schedule; advanced once per
    // frame in record_frame.
    pub scheduler: shadow_schedule::ShadowCascadeScheduler,
    // Cascades re-rendered this frame (bit `i` = cascade `i`). Set in
    // record_frame and read by encode_shadow_pass so the two agree on which
    // slices to refresh and which to leave intact.
    pub render_mask: u32,
    // Carried CSM uniforms: skipped cascades keep the VP their slice was last
    // rendered with, so the Main pass samples each slice consistently. Splits
    // refresh every frame; per-cascade light VPs only when the mask includes
    // that cascade. Uploaded to the per-frame shadow UBO each frame.
    pub uniforms: ShadowUniforms,
}

// One depth-only shadow view to draw from a cull-written indirect region: the
// target slice, the uniforms holding its light-space matrix, which of their
// `light_vps` the vertex shader projects through, and where its commands start.
// A cascade and a spot slice differ only in these.
#[derive(Clone, Copy)]
pub(in crate::directx) struct ShadowView<'a> {
    pub dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub ubo_gva: u64,
    pub vp_index: u32,
    pub indirect: &'a ID3D12Resource,
    // First command of this view's region, in commands.
    pub first_command: usize,
}

impl DxContext {
    pub(in crate::directx) fn encode_shadow_pass(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        shadow_ubo_gva: u64,
        cam_pos: [f32; 3],
        // When `Some`, raymarched SDF casters draw into each cascade
        // after the rasterized + skinned draws and before the
        // depth-write → pixel-shader-resource transition. Constructed
        // by the graph executor: same matrix / time / camera the main
        // raymarch pass will use later this frame, so the shadow cast
        // and the live pass agree on the SDF surface.
        raymarch_view: Option<&crate::directx::raymarch::RaymarchView>,
    ) {
        // No cascade DSVs means shadows are not configured.
        if self.shadow.dsvs.is_empty() {
            return;
        }

        let sm = self.shadow.map_size;

        // Cascades to re-render this frame; draw_frame computed the mask from the
        // update policy. A skipped cascade keeps the depth + VP from when it was
        // last rendered, so the Main pass still samples it consistently. The 0
        // sentinel (mask not yet set) falls back to all cascades.
        let all_cascades = (1u32 << NUM_SHADOW_CASCADES) - 1;
        let render_mask = if self.shadow.render_mask == 0 {
            all_cascades
        } else {
            self.shadow.render_mask
        };

        self.set_shadow_raster_state(cmd, sm);

        // Clear every re-rendered cascade first: with nothing to rasterize the
        // pass is these clears, which is what the raymarched casters below and
        // the main pass's sampler both expect.
        for cascade_idx in 0..NUM_SHADOW_CASCADES {
            if render_mask & (1u32 << cascade_idx) == 0 {
                continue;
            }
            self.clear_shadow_slice(cmd, self.shadow.dsvs[cascade_idx]);
        }

        if let Some(indirect) = self.cull.shadow_indirect_buffers.get(frame_idx)
            && self.shadow_views_drawable()
        {
            // Per-cascade GPU cull -> per-cascade indirect command regions. Runs
            // as a compute prologue in this (shadow) command list, before any draw.
            self.encode_shadow_culls(cmd, frame_idx, render_mask, cam_pos);
            self.bind_shadow_views(cmd, frame_idx);
            let n_cull = self.cull_count();
            for cascade_idx in 0..NUM_SHADOW_CASCADES {
                if render_mask & (1u32 << cascade_idx) == 0 {
                    continue;
                }
                self.draw_shadow_view(
                    cmd,
                    frame_idx,
                    ShadowView {
                        dsv: self.shadow.dsvs[cascade_idx],
                        ubo_gva: shadow_ubo_gva,
                        vp_index: cascade_idx as u32,
                        indirect,
                        first_command: cascade_idx * n_cull,
                    },
                );
            }
        }

        // Raymarched SDF shadow casters: depth-only draws into the same
        // per-cascade DSVs. Run after the rasterized + skinned draws so
        // both layers compete via the cascade's LESS depth test: the
        // nearer caster wins per texel. No-op when no volume opts into
        // `cast_shadows` or when no `raymarch_view` was supplied by the
        // executor.
        if let Some(view) = raymarch_view
            && let Err(e) = self.encode_sdf_shadow_casters(cmd, frame_idx, shadow_ubo_gva, view)
        {
            tracing::error!("encode_sdf_shadow_casters: {}", e);
        }

        // shadow_map's transitions are fully graph-driven: the Shadow producer
        // barrier (PIXEL_SHADER_RESOURCE -> DEPTH_WRITE) runs before this pass
        // and Main's consumer barrier (DEPTH_WRITE -> PIXEL_SHADER_RESOURCE)
        // before the main pass. Neither is emitted here, and there is no inline
        // cross-frame reset (the map rests sampled between frames).
    }

    // Viewport + scissor over a square `size` slice, and triangle-list topology.
    pub(in crate::directx) fn set_shadow_raster_state(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        size: u32,
    ) {
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: size as f32,
                Height: size as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            cmd.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: size as i32,
                bottom: size as i32,
            }]);
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
        }
    }

    pub(in crate::directx) fn clear_shadow_slice(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
    ) {
        // SAFETY: the command list is in the recording state, and the DSV names a live slice.
        unsafe {
            cmd.OMSetRenderTargets(0, None, false, Some(&dsv));
            cmd.ClearDepthStencilView(
                dsv,
                D3D12_CLEAR_FLAG_DEPTH,
                DepthConvention::Shadow.clear(),
                0,
                None,
            );
        }
    }

    // Whether the GPU-driven shadow path has a pipeline and records to draw.
    pub(in crate::directx) fn shadow_views_drawable(&self) -> bool {
        self.cull.shadow_bindless_pso.is_some()
            && self.cull.shadow_bindless_cmd_sig.is_some()
            && self.cull_count() > 0
    }

    // Bind the depth-only bindless shadow pipeline and this frame's object
    // records, the state every `draw_shadow_view` after it shares.
    pub(in crate::directx) fn bind_shadow_views(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
    ) {
        let (Some(pso), Some(root_sig)) = (
            self.cull.shadow_bindless_pso.as_ref(),
            self.cull.shadow_bindless_root_sig.as_ref(),
        ) else {
            return;
        };
        let object_gva = com::gpu_va(&self.cull.object_buffer_resources[frame_idx]);
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.SetPipelineState(pso);
            cmd.SetGraphicsRootSignature(root_sig);
            // [3] this frame's GpuObjectData.
            cmd.SetGraphicsRootShaderResourceView(3, object_gva);
        }
    }

    // GPU-driven depth-only raster of one shadow view: one `ExecuteIndirect` for
    // the static + instance prefix of its region and, when present, a second for
    // the skinned tail -- the same two-region split the bindless main pass uses,
    // projected through `light_vps[view.vp_index]`. The CPU never walks the
    // static / instanced / skinned draw lists. The caller has cleared the depth
    // and bound the pipeline with `bind_shadow_views`.
    pub(in crate::directx) fn draw_shadow_view(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        view: ShadowView<'_>,
    ) {
        let Some(cmd_sig) = self.cull.shadow_bindless_cmd_sig.as_ref() else {
            return;
        };
        let prefix = self.skinned_record_base();
        let stride = crate::directx::cull::INDIRECT_COMMAND_STRIDE as u64;
        let byte_off = |first: usize| first as u64 * stride;
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.OMSetRenderTargets(0, None, false, Some(&view.dsv));
            // [1] the view's light_vps, [2] the slot of it this view projects through.
            cmd.SetGraphicsRootConstantBufferView(1, view.ubo_gva);
            cmd.set_graphics_root_constants(2, &view.vp_index);
            cmd.IASetVertexBuffers(0, Some(&[self.scene.geometry.vertex_buffer_view]));
            cmd.IASetIndexBuffer(Some(&self.scene.geometry.index_buffer_view));
            cmd.ExecuteIndirect(
                cmd_sig,
                prefix as u32,
                view.indirect,
                byte_off(view.first_command),
                None::<&ID3D12Resource>,
                0,
            );
        }
        self.inc_draw_calls(1);

        // Skinned tail over the deformed VB + skinned IB. No depth clear -- it
        // appends to the static depth via the LESS test.
        if self.state.draw.n_skinned > 0
            && let Some(deformed_vbv) = self.skinned.deformed_vbvs.get(frame_idx)
        {
            // SAFETY: the command list is in the recording state, and every resource,
            // descriptor and slice these commands name is live for the call.
            unsafe {
                cmd.IASetVertexBuffers(0, Some(&[*deformed_vbv]));
                cmd.IASetIndexBuffer(Some(&self.skinned.index_buffer_view));
                cmd.ExecuteIndirect(
                    cmd_sig,
                    self.state.draw.n_skinned as u32,
                    view.indirect,
                    byte_off(view.first_command + prefix),
                    None::<&ID3D12Resource>,
                    0,
                );
            }
            self.inc_draw_calls(1);
        }
    }
}
