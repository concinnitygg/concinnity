// src/directx/draw/shadow.rs
//
// Cascaded shadow-map pass: one depth-only render per CSM cascade slice.
// Draws static objects, instanced clusters, and (when present) skinned
// meshes into each slice of the shadow map array. Caller has already
// uploaded this frame's `ShadowUniforms` into `shadow_ubo_gva`; this pass
// just binds it once per shadow pipeline and pushes the cascade index per
// draw. Skipped entirely when no shadow pipeline is configured or the
// fallback 1x1 shadow array is bound.
//
// When the bindless GPU-cull path is active (`shadow_bindless_pso` built +
// build-time geometry present) the pass is GPU-driven: a per-cascade cull
// dispatch writes one `ExecuteIndirect` region per cascade and each cascade is
// issued with a single `ExecuteIndirect` over the static + skinned records (the
// same cull buffers the main pass uses), instead of the CPU per-object loop.
// Streamed chunks / runtime clones (records past `draw.n_objects`) keep the legacy
// per-object loop. A non-bindless world (custom shader) keeps the legacy path
// entirely.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;

use crate::gfx::render_types::NUM_SHADOW_CASCADES;

use crate::directx::com;
use crate::directx::context::DxContext;

// Root constants for the legacy shadow pass (80 bytes = 20 DWORDs): model matrix
// + cascade_idx + padding. cascade_idx selects which `ShadowUniforms.light_vps[i]`
// the shadow vertex shader projects through; iterated 0..NUM_SHADOW_CASCADES
// across the per-cascade shadow passes.
#[derive(Copy, Clone)]
#[repr(C)]
struct ShadowPush {
    model: [[f32; 4]; 4],
    cascade_idx: u32,
    _pad: [u32; 3],
}

// The depth-only shadow pipeline: its PSO + root signature, always used together.
#[derive(Clone, Copy)]
struct ShadowPipeline<'a> {
    pso: &'a ID3D12PipelineState,
    root_sig: &'a ID3D12RootSignature,
}

// One shadow slice's draw state: the depth-only pipeline to draw with, the
// uniform buffer holding its light-space matrices, and which `light_vps` entry
// the vertex shader projects through. Grouping them keeps the caster
// sub-encoders' argument lists short and lets the cascade pass and the spot
// shadow pass share those encoders: a spot slice is just a binding whose
// uniforms carry its own matrix in slot 0.
#[derive(Clone, Copy)]
pub(in crate::directx) struct ShadowPassBinding<'a> {
    pub pso: &'a ID3D12PipelineState,
    pub root_sig: &'a ID3D12RootSignature,
    pub ubo_gva: u64,
    pub slice_idx: u32,
}

impl DxContext {
    pub(in crate::directx) fn encode_shadow_pass(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        shadow_ubo_gva: u64,
        cam_pos: [f32; 3],
        // When `Some`, raymarched SDF casters draw into each cascade
        // after the rasterised + skinned draws and before the
        // depth-write → pixel-shader-resource transition. Constructed
        // by the graph executor: same matrix / time / camera the main
        // raymarch pass will use later this frame, so the shadow cast
        // and the live pass agree on the SDF surface.
        raymarch_view: Option<&crate::directx::raymarch::RaymarchView>,
    ) {
        let (Some(shadow_pso), Some(shadow_root_sig)) =
            (self.shadow_pso.as_ref(), self.shadow_root_sig.as_ref())
        else {
            return;
        };
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

        // Viewport + scissor + topology are common to both the GPU-driven and the
        // legacy raster paths.
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: sm as f32,
                Height: sm as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            cmd.RSSetViewports(&[vp]);
            let scissor = RECT {
                left: 0,
                top: 0,
                right: sm as i32,
                bottom: sm as i32,
            };
            cmd.RSSetScissorRects(&[scissor]);
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
        }

        if self.cull.shadow_bindless_pso.is_some() && self.cull_count() > 0 {
            self.encode_shadow_pass_gpu_driven(
                cmd,
                frame_idx,
                shadow_ubo_gva,
                cam_pos,
                render_mask,
                ShadowPipeline {
                    pso: shadow_pso,
                    root_sig: shadow_root_sig,
                },
            );
        } else {
            self.encode_shadow_pass_legacy(
                cmd,
                frame_idx,
                shadow_ubo_gva,
                cam_pos,
                render_mask,
                ShadowPipeline {
                    pso: shadow_pso,
                    root_sig: shadow_root_sig,
                },
            );
        }

        // Raymarched SDF shadow casters: depth-only draws into the same
        // per-cascade DSVs. Run after the rasterised + skinned draws so
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

    // GPU-driven shadow raster: per-cascade GPU cull writes one indirect region
    // per cascade, then each re-rendered cascade is issued with one
    // `ExecuteIndirect` for the static + instance prefix and (when present) a
    // second for the skinned tail -- the same two-region split the bindless main
    // pass uses, but depth-only and through `light_vps[cascade_idx]`. The CPU
    // never walks the static / instanced / skinned draw lists.
    fn encode_shadow_pass_gpu_driven(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        shadow_ubo_gva: u64,
        cam_pos: [f32; 3],
        render_mask: u32,
        legacy_pipeline: ShadowPipeline<'_>,
    ) {
        let ShadowPipeline {
            pso: legacy_shadow_pso,
            root_sig: legacy_shadow_root_sig,
        } = legacy_pipeline;
        let (Some(sb_pso), Some(sb_root), Some(sb_sig), Some(indirect)) = (
            self.cull.shadow_bindless_pso.as_ref(),
            self.cull.shadow_bindless_root_sig.as_ref(),
            self.cull.shadow_bindless_cmd_sig.as_ref(),
            self.cull.shadow_indirect_buffers.get(frame_idx),
        ) else {
            return;
        };
        let n_cull = self.cull_count();
        let prefix = self.skinned_record_base();
        let stride = crate::directx::cull::INDIRECT_COMMAND_STRIDE as usize;
        let object_gva = com::gpu_va(&self.cull.object_buffer_resources[frame_idx]);

        // Per-cascade GPU cull -> per-cascade indirect command regions. Runs as a
        // compute prologue in this (shadow) command list, before any render pass.
        self.encode_shadow_culls(cmd, frame_idx, render_mask, cam_pos);

        // Static + instance prefix: clear each re-rendered cascade's depth then
        // issue its `[0, skinned_record_base())` region against the static VB/IB.
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.SetPipelineState(sb_pso);
            cmd.SetGraphicsRootSignature(sb_root);
            cmd.IASetVertexBuffers(0, Some(&[self.geometry.vertex_buffer_view]));
            cmd.IASetIndexBuffer(Some(&self.geometry.index_buffer_view));
            // [1] shadow UBO (light_vps), [3] this frame's GpuObjectData.
            cmd.SetGraphicsRootConstantBufferView(1, shadow_ubo_gva);
            cmd.SetGraphicsRootShaderResourceView(3, object_gva);
        }
        for cascade_idx in 0..NUM_SHADOW_CASCADES {
            if render_mask & (1u32 << cascade_idx) == 0 {
                continue;
            }
            let dsv = self.shadow.dsvs[cascade_idx];
            let c = cascade_idx as u32;
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                cmd.OMSetRenderTargets(0, None, false, Some(&dsv));
                cmd.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
                // [2] cascade index, constant across this cascade's ExecuteIndirect.
                cmd.SetGraphicsRoot32BitConstants(
                    2,
                    1,
                    &c as *const u32 as *const std::ffi::c_void,
                    0,
                );
                let byte_off = ((cascade_idx * n_cull) * stride) as u64;
                cmd.ExecuteIndirect(
                    sb_sig,
                    prefix as u32,
                    indirect,
                    byte_off,
                    None::<&ID3D12Resource>,
                    0,
                );
            }
            self.inc_draw_calls(1);
        }

        // Skinned tail: a second `ExecuteIndirect` per cascade over the deformed
        // VB + skinned u16 IB, reading each cascade region from `skinned_record_base()`
        // on. No depth clear -- appends to the static depth via the LESS test.
        if self.draw.n_skinned > 0
            && let Some(deformed_vbv) = self.skinned.deformed_vbvs.get(frame_idx)
        {
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                cmd.IASetVertexBuffers(0, Some(&[*deformed_vbv]));
                cmd.IASetIndexBuffer(Some(&self.skinned.index_buffer_view));
            }
            for cascade_idx in 0..NUM_SHADOW_CASCADES {
                if render_mask & (1u32 << cascade_idx) == 0 {
                    continue;
                }
                let dsv = self.shadow.dsvs[cascade_idx];
                let c = cascade_idx as u32;
                // SAFETY: the command list is in the recording state, and every resource,
                // descriptor and slice these commands name is live for the call.
                unsafe {
                    cmd.OMSetRenderTargets(0, None, false, Some(&dsv));
                    cmd.SetGraphicsRoot32BitConstants(
                        2,
                        1,
                        &c as *const u32 as *const std::ffi::c_void,
                        0,
                    );
                    let byte_off = ((cascade_idx * n_cull + prefix) * stride) as u64;
                    cmd.ExecuteIndirect(
                        sb_sig,
                        self.draw.n_skinned as u32,
                        indirect,
                        byte_off,
                        None::<&ID3D12Resource>,
                        0,
                    );
                }
                self.inc_draw_calls(1);
            }
        }

        // Legacy depth-only casters for draws past the bindless record range
        // (streamed VoxelWorld chunks + runtime clones, which are not in the
        // GpuObjectData buffer the cull kernel walks). A no-op for worlds with no
        // such draws (the common case). Converged into the unified records by the
        // chunk phase.
        self.encode_shadow_legacy_extra(
            cmd,
            render_mask,
            cam_pos,
            legacy_shadow_pso,
            legacy_shadow_root_sig,
            shadow_ubo_gva,
        );
    }

    // Legacy per-object casters for runtime clones past the bindless record range
    // (`i >= draw.n_objects` AND in `clone.slot_by_draw_idx`). Streamed VoxelWorld chunks
    // now fold into the GPU-driven cull records (drawn by the per-cascade indirect
    // draw), so they are skipped here. Mirrors the legacy static loop, appending
    // into each re-rendered cascade's depth (no re-clear). A no-op for worlds with
    // no clones (the common case, incl. pure-voxel worlds).
    fn encode_shadow_legacy_extra(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        render_mask: u32,
        cam_pos: [f32; 3],
        shadow_pso: &ID3D12PipelineState,
        shadow_root_sig: &ID3D12RootSignature,
        shadow_ubo_gva: u64,
    ) {
        if self.clone.slot_by_draw_idx.is_empty() {
            return;
        }
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.SetPipelineState(shadow_pso);
            cmd.SetGraphicsRootSignature(shadow_root_sig);
            cmd.IASetVertexBuffers(0, Some(&[self.geometry.vertex_buffer_view]));
            cmd.IASetIndexBuffer(Some(&self.geometry.index_buffer_view));
            cmd.SetGraphicsRootConstantBufferView(1, shadow_ubo_gva);
        }
        for cascade_idx in 0..NUM_SHADOW_CASCADES {
            if render_mask & (1u32 << cascade_idx) == 0 {
                continue;
            }
            let dsv = self.shadow.dsvs[cascade_idx];
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                // Append to the GPU-driven cascade depth (no re-clear).
                cmd.OMSetRenderTargets(0, None, false, Some(&dsv));
            }
            for (i, obj) in self.draw.objects.iter().enumerate() {
                if i < self.draw.n_objects || !obj.visible || !obj.resident {
                    continue;
                }
                if !self.clone.slot_by_draw_idx.contains_key(&i) {
                    continue; // streamed chunk -> folded into the cull records
                }
                let push = ShadowPush {
                    model: obj.model,
                    cascade_idx: cascade_idx as u32,
                    _pad: [0; 3],
                };
                let d = crate::gfx::lod::camera_distance(obj, cam_pos);
                let (index_offset, index_count) = obj.active_lod(d);
                // SAFETY: the command list is in the recording state, and every resource,
                // descriptor and slice these commands name is live for the call.
                unsafe {
                    cmd.SetGraphicsRoot32BitConstants(
                        0,
                        20,
                        &push as *const ShadowPush as *const std::ffi::c_void,
                        0,
                    );
                    cmd.DrawIndexedInstanced(
                        index_count as u32,
                        1,
                        index_offset as u32,
                        obj.base_vertex,
                        0,
                    );
                }
                self.inc_draw_calls(1);
            }
        }
    }

    // Static + instanced depth-only casters for one shadow slice, drawn into
    // whichever DSV the caller bound. Binds the pipeline, the shared geometry
    // buffers, and `bind.ubo_gva`; the caller owns the render target and any
    // depth clear. Shared by the cascade pass and the spot shadow pass, which
    // differ only in which matrix `bind.slice_idx` selects.
    pub(in crate::directx) fn encode_shadow_casters_into(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        bind: ShadowPassBinding<'_>,
        cam_pos: [f32; 3],
    ) {
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.SetPipelineState(bind.pso);
            cmd.SetGraphicsRootSignature(bind.root_sig);
            cmd.IASetVertexBuffers(0, Some(&[self.geometry.vertex_buffer_view]));
            cmd.IASetIndexBuffer(Some(&self.geometry.index_buffer_view));
            cmd.SetGraphicsRootConstantBufferView(1, bind.ubo_gva);

            for obj in &self.draw.objects {
                // A non-resident streamed mesh has no geometry in the shared
                // buffers yet -- skip it everywhere.
                if !obj.visible || !obj.resident {
                    continue;
                }
                let push = ShadowPush {
                    model: obj.model,
                    cascade_idx: bind.slice_idx,
                    _pad: [0; 3],
                };
                // Pick the LOD by camera distance; the shadow pass uses the same
                // slice the main pass will, so silhouettes track when the runtime
                // swaps to a coarser LOD.
                let d = crate::gfx::lod::camera_distance(obj, cam_pos);
                let (index_offset, index_count) = obj.active_lod(d);
                cmd.SetGraphicsRoot32BitConstants(
                    0,
                    20,
                    &push as *const ShadowPush as *const std::ffi::c_void,
                    0,
                );
                cmd.DrawIndexedInstanced(
                    index_count as u32,
                    1,
                    index_offset as u32,
                    obj.base_vertex,
                    0,
                );
                self.inc_draw_calls(1);
            }

            // Instanced clusters: iterate instances individually (cheap, visually
            // identical to an instanced shadow shader). Reads the per-cluster LOD
            // bucket layout cached at the top of record_frame by
            // `build_instance_upload`, so the shadow pass picks the exact same LOD
            // slice the main pass is about to draw; cascade-seam silhouettes stay
            // coherent when the runtime swaps to a coarser LOD.
            let layouts = self.instanced.bucket_layouts.read().unwrap();
            for buckets in layouts.iter() {
                for bucket in buckets.iter() {
                    for &model in &bucket.instances {
                        let push = ShadowPush {
                            model,
                            cascade_idx: bind.slice_idx,
                            _pad: [0; 3],
                        };
                        cmd.SetGraphicsRoot32BitConstants(
                            0,
                            20,
                            &push as *const ShadowPush as *const std::ffi::c_void,
                            0,
                        );
                        cmd.DrawIndexedInstanced(
                            bucket.index_count as u32,
                            1,
                            bucket.index_offset as u32,
                            0,
                            0,
                        );
                        self.inc_draw_calls(1);
                    }
                }
            }
        }
    }

    // Skinned depth-only casters for one shadow slice, drawn into whichever DSV
    // the caller bound. Binds the skinned shadow pipeline and the deformed
    // geometry; no depth clear, so skinned depth appends to whatever
    // `encode_shadow_casters_into` already laid down. A no-op when the world has
    // no skinned mesh.
    pub(in crate::directx) fn encode_shadow_skinned_into(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        ubo_gva: u64,
        slice_idx: u32,
        frame_idx: usize,
        cam_pos: [f32; 3],
    ) {
        let (Some(pso), Some(root_sig)) = (
            self.skinned.shadow_pso.as_ref(),
            self.skinned.shadow_root_sig.as_ref(),
        ) else {
            return;
        };
        if self.skinned.draw_objects.is_empty() {
            return;
        }
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.SetPipelineState(pso);
            cmd.SetGraphicsRootSignature(root_sig);
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            cmd.IASetVertexBuffers(0, Some(&[self.skinned.vertex_buffer_view]));
            cmd.IASetIndexBuffer(Some(&self.skinned.index_buffer_view));
            cmd.SetGraphicsRootConstantBufferView(1, ubo_gva);

            for (i, obj) in self.skinned.draw_objects.iter().enumerate() {
                if !obj.visible {
                    continue;
                }
                // Skinned-mesh LOD: pick by camera distance (not light
                // direction) so the shadow casts match the triangles main will
                // rasterise. Per-cascade LOD would technically be cheaper for
                // distant cascades, but matching main keeps cascade seams free
                // of silhouette swaps. Mirrors Metal.
                let d = crate::gfx::lod::skinned_camera_distance(obj, cam_pos);
                let (index_offset, index_count) = obj.active_lod(d);
                let push = ShadowPush {
                    model: obj.model,
                    cascade_idx: slice_idx,
                    _pad: [0; 3],
                };
                cmd.SetGraphicsRoot32BitConstants(
                    0,
                    20,
                    &push as *const ShadowPush as *const std::ffi::c_void,
                    0,
                );
                cmd.SetGraphicsRootShaderResourceView(2, self.skinned_joint_gva(frame_idx, i));
                cmd.DrawIndexedInstanced(index_count as u32, 1, index_offset as u32, 0, 0);
                self.inc_draw_calls(1);
            }
        }
    }

    // Legacy CPU-driven shadow raster: per-cascade per-object `DrawIndexed` for
    // static + instanced (iterated per instance) + skinned casters. Used for
    // non-bindless worlds (custom shader) or worlds with no build-time geometry.
    // Static + instanced run in one cascade sweep and skinned in a second, so
    // each pipeline is set once; the skinned sweep does not re-clear, letting
    // deformed depth append to the static depth via the LESS test.
    fn encode_shadow_pass_legacy(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        shadow_ubo_gva: u64,
        cam_pos: [f32; 3],
        render_mask: u32,
        pipeline: ShadowPipeline<'_>,
    ) {
        for cascade_idx in 0..NUM_SHADOW_CASCADES {
            if render_mask & (1u32 << cascade_idx) == 0 {
                continue;
            }
            let dsv = self.shadow.dsvs[cascade_idx];
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                cmd.OMSetRenderTargets(0, None, false, Some(&dsv));
                cmd.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
            }
            self.encode_shadow_casters_into(
                cmd,
                ShadowPassBinding {
                    pso: pipeline.pso,
                    root_sig: pipeline.root_sig,
                    ubo_gva: shadow_ubo_gva,
                    slice_idx: cascade_idx as u32,
                },
                cam_pos,
            );
        }

        for cascade_idx in 0..NUM_SHADOW_CASCADES {
            if render_mask & (1u32 << cascade_idx) == 0 {
                continue;
            }
            let dsv = self.shadow.dsvs[cascade_idx];
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                cmd.OMSetRenderTargets(0, None, false, Some(&dsv));
            }
            self.encode_shadow_skinned_into(
                cmd,
                shadow_ubo_gva,
                cascade_idx as u32,
                frame_idx,
                cam_pos,
            );
        }
    }
}
