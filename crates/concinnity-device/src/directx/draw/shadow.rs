//! Cascaded shadow-map pass: one depth-only render per CSM cascade slice.
//! Draws static objects, instanced clusters, and (when present) skinned
//! meshes into each slice of the shadow map array. Caller has already
//! uploaded this frame's `ShadowUniforms` into `shadow_ubo_gva`; this pass
//! binds it and sets the cascade index per cascade. Skipped entirely when the
//! fallback 1x1 shadow array is bound.
//!
//! The cascades are GPU-driven: a per-cascade cull dispatch and its packing
//! write one region of lists per cascade, and each cascade is issued with one
//! `ExecuteIndirect` by count over its kept static + instance records and one
//! over its kept skinned tail. Everything appearing after init -- streamed chunks and
//! spawned clones -- folds into those same records, so the whole scene is
//! covered. The spot slices draw through the same `draw_shadow_view` from their
//! own indirect buffer.

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types::{NUM_SHADOW_CASCADES, SKINNED_STREAM};
use concinnity_engine::render::depth::DEPTH_CLEAR;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::com;
use crate::directx::compact::{DrawLists, execute_stream};
use crate::directx::context::DxContext;
use crate::directx::descriptor_slot::SrvSlot;
use crate::directx::root_constants::RootConstants;
use crate::directx::texture::GpuResource;

// Shadow map resources. `resource` / `dsvs` are `None` / empty when the shadow
// pass is disabled (a 1x1 array fallback SRV is still bound at `srv_gpu`).
// `dsvs` is one DSV per cascade slice.
pub(in crate::directx) struct ShadowState {
    pub resource: Option<GpuResource<ID3D12Resource>>,
    pub dsvs: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub map_size: u32,
    // Slices the bound resource holds: the cascades, or the fallback's one.
    pub layers: u32,
    pub srv_gpu: SrvSlot,
}

// One depth-only shadow view to draw from its packed lists: the target slice,
// the uniforms holding its light-space matrix, which of their `light_vps` the
// vertex shader projects through, and its lists. A cascade and a spot slice
// differ only in these.
#[derive(Clone, Copy)]
pub(in crate::directx) struct ShadowView<'a> {
    pub dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub ubo_gva: u64,
    pub vp_index: u32,
    pub lists: DrawLists<'a>,
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
        // This frame's grass, whose cascade blades draw into the nearest
        // cascade after its rasterized casters, when it casts this frame.
        grass: Option<&crate::directx::grass::GrassFrame>,
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
        let render_mask = if self.state.planner.shadow.cascade_mask == 0 {
            all_cascades
        } else {
            self.state.planner.shadow.cascade_mask
        };

        Self::set_shadow_raster_state(cmd, sm);

        // Clear every re-rendered cascade first: with nothing to rasterize the
        // pass is these clears, which is what the raymarched casters below and
        // the main pass's sampler both expect.
        for cascade_idx in 0..NUM_SHADOW_CASCADES {
            if render_mask & (1u32 << cascade_idx) == 0 {
                continue;
            }
            Self::clear_shadow_slice(cmd, self.shadow.dsvs[cascade_idx]);
        }

        if let Some(views) = self.cull.shadow_views.get(frame_idx)
            && self.shadow_views_drawable()
        {
            // Per-cascade GPU cull -> per-cascade packed lists. Runs as a
            // compute prologue in this (shadow) command list, before any draw.
            self.encode_shadow_culls(cmd, frame_idx, render_mask, cam_pos);
            self.bind_shadow_views(cmd, frame_idx);
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
                        vp_index: u32::saturating_from(cascade_idx),
                        lists: self.shadow_lists(views, cascade_idx),
                    },
                );
            }
        }

        if render_mask & 1 != 0
            && let Some(frame) = grass
        {
            self.encode_grass_shadow_draw(cmd, frame_idx, self.shadow.dsvs[0], frame);
        }

        // Raymarched SDF shadow casters: depth-only draws into the same
        // per-cascade DSVs. Run after the rasterized + skinned draws so
        // both layers compete via the cascade's depth write test: the
        // caster nearer the light wins per texel. No-op when no volume opts into
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
    pub(in crate::directx) fn set_shadow_raster_state(cmd: &ID3D12GraphicsCommandList, size: u32) {
        let viewport = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: size as f32,
            Height: size as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let scissor = RECT {
            left: 0,
            top: 0,
            right: i32::saturating_from(size),
            bottom: i32::saturating_from(size),
        };
        // SAFETY: the command list is recording, and the viewport slice is a local that outlives
        // the call.
        unsafe { cmd.RSSetViewports(&[viewport]) };
        // SAFETY: the scissor slice is a local that outlives the call.
        unsafe { cmd.RSSetScissorRects(&[scissor]) };
        // SAFETY: setting the topology only needs a recording command list.
        unsafe {
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
        }
    }

    pub(in crate::directx) fn clear_shadow_slice(
        cmd: &ID3D12GraphicsCommandList,
        dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
    ) {
        // SAFETY: the command list is in the recording state, and the DSV names a live slice.
        unsafe { cmd.OMSetRenderTargets(0, None, false, Some(&raw const dsv)) };
        // SAFETY: `dsv` is the live slice just bound as the depth target, and no rects are passed.
        unsafe { cmd.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, DEPTH_CLEAR, 0, None) };
    }

    // Whether the GPU-driven shadow path has a pipeline and records to draw.
    pub(in crate::directx) fn shadow_views_drawable(&self) -> bool {
        self.cull.shadow_bindless_pso.is_some()
            && self.cull.shadow_bindless_cmd_sig.is_some()
            && self.state.draw.record_count() > 0
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
        // SAFETY: the command list is recording, and the shadow PSO is owned by `self.cull`.
        unsafe { cmd.SetPipelineState(pso) };
        // SAFETY: the shadow root signature is owned by `self.cull` alongside its PSO.
        unsafe { cmd.SetGraphicsRootSignature(root_sig) };
        // [3] this frame's GpuObjectData, [4] the shared vertex stream.
        // SAFETY: `object_gva` addresses this frame's live object buffer.
        unsafe { cmd.SetGraphicsRootShaderResourceView(3, object_gva) };
        let vertices_gva = com::gpu_va(&self.scene.geometry.vertex_buffer);
        // SAFETY: the shared vertex stream is owned by `self.scene` and outlives the frame.
        unsafe { cmd.SetGraphicsRootShaderResourceView(4, vertices_gva) };
    }

    // GPU-driven depth-only raster of one shadow view: one `ExecuteIndirect` by
    // count for its kept static + instance records and, when present, a second
    // for its kept skinned tail -- the same two-stream split the bindless main
    // pass uses, projected through `light_vps[view.vp_index]`. The CPU never walks the
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
        let layout = self.shadow_layout();
        // SAFETY: the command list is recording, and the caller cleared the live slice `view.dsv`
        // names.
        unsafe { cmd.OMSetRenderTargets(0, None, false, Some(&raw const view.dsv)) };
        // [1] the view's light_vps, [2] the slot of it this view projects through.
        // SAFETY: `view.ubo_gva` addresses the view's live light_vps buffer.
        unsafe { cmd.SetGraphicsRootConstantBufferView(1, view.ubo_gva) };
        // SAFETY: root param 2 of the shadow root signature is one root-constant DWORD.
        unsafe { cmd.set_graphics_root_constants(2, &view.vp_index) };
        // SAFETY: the static vertex buffer is owned by `self.scene` and outlives the frame.
        unsafe { cmd.IASetVertexBuffers(0, Some(&[self.scene.geometry.vertex_buffer_view])) };
        // SAFETY: the static index buffer is owned by `self.scene`; the view is read in the call.
        unsafe { cmd.IASetIndexBuffer(Some(&raw const self.scene.geometry.index_buffer_view)) };
        // `bind_shadow_views` bound the pipeline the command signature matches.
        self.inc_draw_calls(execute_stream(cmd, cmd_sig, view.lists, 0, layout.slots(0)));

        // Skinned tail over the deformed VB + skinned IB. No depth clear -- it
        // appends to the static depth via the write test.
        let skinned = layout.slots(SKINNED_STREAM);
        if skinned.capacity > 0
            && let Some(deformed_vbv) = self.skinned.deformed_vbvs.get(frame_idx)
        {
            // SAFETY: this frame's deformed buffer is owned by `self.skinned` and was posed by
            // the skin pass.
            unsafe { cmd.IASetVertexBuffers(0, Some(&[*deformed_vbv])) };
            // SAFETY: the skinned index buffer is owned by `self.skinned` and outlives the frame.
            unsafe { cmd.IASetIndexBuffer(Some(&raw const self.skinned.index_buffer_view)) };
            self.inc_draw_calls(execute_stream(
                cmd,
                cmd_sig,
                view.lists,
                SKINNED_STREAM,
                skinned,
            ));
        }
    }
}
