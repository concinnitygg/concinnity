// src/directx/draw/main.rs
//
// Main HDR scene pass: two GPU-driven `ExecuteIndirect`s per shader bucket, the
// static + instance + runtime prefix and the skinned tail. Renders linear-light
// HDR into `hdr_color`; the composite pass tonemaps that down
// onto the swapchain backbuffer. Ends by transitioning (or MSAA-resolving)
// `hdr_color` to `PIXEL_SHADER_RESOURCE` so post-process passes can sample
// it.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::com;
use crate::directx::context::DxContext;
use crate::directx::graph_exec::{FrameGpuBuffers, MainPassExtent};
use crate::directx::texture::{HDR_FORMAT, transition_barrier};

impl DxContext {
    // Bind the per-scene local-light side tables on whichever root signature is
    // current: the spot shadow projections + depth array, and the area-light
    // table + its two LTC lookups. All are static for the world's lifetime, so
    // every main-pass site binds the same four and only the parameter indices
    // differ. Kept as one call so a new side table reaches every site at once.
    pub(in crate::directx) fn bind_local_light_tables(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        params: super::LocalLightParams,
    ) {
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.SetGraphicsRootShaderResourceView(
                params.spot_buffer,
                com::gpu_va(&self.spot_shadow.buffer),
            );
            cmd.SetGraphicsRootDescriptorTable(params.spot_table, self.spot_shadow.srv_gpu);
            cmd.SetGraphicsRootShaderResourceView(
                params.area_buffer,
                com::gpu_va(&self.area_light.buffer),
            );
            cmd.SetGraphicsRootDescriptorTable(params.ltc_table, self.area_light.ltc_table_gpu);
        }
    }

    // Rebuild this frame's `StructuredBuffer<GpuObjectData>` for the bindless
    // static pass: one record per `DrawObject`, indexed by object id. Everything
    // past `draw.n_objects` -- streamed `VoxelWorld` chunks and spawned clones --
    // folds into the runtime reserve below. Rebuilt every frame so
    // `update_model` / `update_visibility` edits are reflected; a no-op when
    // the bindless pass is inactive.
    pub(in crate::directx) fn build_object_buffer(&self, frame_idx: usize) {
        use crate::gfx::render_types::{
            GpuObjectData, albedo_pool_index, normal_pool_index, pack_object_record,
            pack_skinned_record,
        };
        let Some(&ptr) = self.cull.object_buffer_ptrs.get(frame_idx) else {
            return;
        };
        let stride = std::mem::size_of::<GpuObjectData>();
        // Shared handle-indexed pool indices, identical to Vulkan/Metal: albedo =
        // texture_slot, normal = the normal map's own handle (or the flat-normal
        // fallback slot for a normal-less draw). The bindless main pass + RT hit
        // shader bind the pool base, so a shared texture resolves to one descriptor.
        let texture_count = self.descriptors.textures.len() as u32;
        for (i, obj) in self
            .draw
            .objects
            .iter()
            .take(self.draw.n_objects)
            .enumerate()
        {
            let albedo = albedo_pool_index(obj.texture_slot, texture_count);
            let normal = normal_pool_index(obj.normal_map_slot, texture_count);
            let rec = pack_object_record(obj, albedo, normal);
            // SAFETY: the buffer was sized for `draw.n_objects` records and the
            // loop is bounded by `take(draw.n_objects)`, so `i * stride` is in range.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &rec as *const GpuObjectData as *const u8,
                    ptr.add(i * stride),
                    stride,
                );
            }
        }

        // Runtime objects -- streamed chunks and spawned clones -- one record each
        // in the reserved region at `[runtime_record_base() + k]`, packed exactly
        // like a static object (their geometry already lives in the shared VB/IB
        // with their own `base_vertex`, so they ride the static + instance prefix
        // `ExecuteIndirect`). Flat-pool texture indices give each its own
        // material. A non-resident (freed) slot's stale record here is never
        // read -- `build_draw_args_buffer` disables it (ENABLED clear), and the
        // cull kernel skips `objects[i]` for a disabled record. The unused
        // reserve tail is likewise never read.
        let runtime_base = self.runtime_record_base();
        self.for_each_runtime_record(|k, _, obj| {
            let albedo = albedo_pool_index(obj.texture_slot, texture_count);
            let normal = normal_pool_index(obj.normal_map_slot, texture_count);
            let rec = pack_object_record(obj, albedo, normal);
            // SAFETY: the reserve is `[runtime_base, runtime_base + draw.n_runtime)`
            // and `for_each_runtime_record` caps `k < draw.n_runtime`, so the
            // write is in range.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &rec as *const GpuObjectData as *const u8,
                    ptr.add((runtime_base + k) * stride),
                    stride,
                );
            }
        });

        // Skinned objects: one record each in the reserved tail at
        // `[skinned_record_base(), cull_count())`. `model = obj.model` (applied
        // after the per-frame skin deform), flat-pool texture indices like a
        // static object, and a padded bind-pose AABB so the cull kernel can
        // frustum/Hi-Z test them. Drawn by the main pass's 2nd `ExecuteIndirect`.
        let skinned_base = self.skinned_record_base();
        for (k, obj) in self
            .skinned
            .draw_objects
            .iter()
            .take(self.draw.n_skinned)
            .enumerate()
        {
            let albedo = albedo_pool_index(obj.texture_slot, texture_count);
            let normal = normal_pool_index(obj.normal_map_slot, texture_count);
            let rec = pack_skinned_record(obj, albedo, normal);
            // SAFETY: the buffer reserved `draw.n_skinned` records past
            // `skinned_record_base()` at init; the loop is bounded by
            // `self.skinned.draw_objects.len() == self.draw.n_skinned`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &rec as *const GpuObjectData as *const u8,
                    ptr.add((skinned_base + k) * stride),
                    stride,
                );
            }
        }
    }

    // Recompute every instanced cluster's per-LOD-bucket partition for the
    // current camera into `instanced.bucket_layouts`, which the spot shadow
    // pass reads. Called once per frame from `record_frame` before
    // `execute_graph` dispatches any pass.
    pub(super) fn build_instance_upload(&self, cam_pos: [f32; 3]) {
        let mut layouts = self.instanced.bucket_layouts.write().unwrap();
        // Re-shape the outer Vec when cluster count changed (runtime asset
        // hot-reload), then clear every row in place to reuse heap.
        if layouts.len() != self.instanced.clusters.len() {
            layouts.clear();
            layouts.resize(self.instanced.clusters.len(), Vec::new());
        } else {
            for row in layouts.iter_mut() {
                row.clear();
            }
        }
        for (cluster_idx, cluster) in self.instanced.clusters.iter().enumerate() {
            if cluster.instances.is_empty() {
                continue;
            }
            let buckets = cluster.lod_buckets(cam_pos);
            let row = &mut layouts[cluster_idx];
            row.reserve(buckets.len());
            for bucket in buckets {
                row.push(crate::directx::context::InstanceBucketLayout {
                    index_offset: bucket.index_offset,
                    index_count: bucket.index_count,
                    instances: bucket.instances,
                });
            }
        }
    }

    // Encode the GPU-driven main pass into `cmd`: the static + instance +
    // runtime prefix, then the skinned tail, per shader bucket. Finishes by
    // resolving (MSAA) or transitioning (no MSAA) the HDR target
    // to `PIXEL_SHADER_RESOURCE` so the velocity / TAA / bloom / composite
    // passes can sample it.
    pub(in crate::directx) fn encode_main_pass(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        extent: MainPassExtent,
        gpu: FrameGpuBuffers,
        world_hidden: bool,
    ) {
        let MainPassExtent { width, height } = extent;
        let FrameGpuBuffers {
            view_gva,
            light_gva,
            local_lights_gva,
            shadow_ubo_gva,
        } = gpu;
        let depth_dsv = self.depth.dsv;

        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.OMSetRenderTargets(1, Some(&self.hdr.color_rtv), false, Some(&depth_dsv));
            cmd.ClearRenderTargetView(self.hdr.color_rtv, &self.view.clear_color, None);
            cmd.ClearDepthStencilView(depth_dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);

            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            cmd.RSSetViewports(&[vp]);
            let scissor = RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            };
            cmd.RSSetScissorRects(&[scissor]);
        }

        // Opaque menu backdrop: the render target was just cleared; skip every
        // draw so nothing of the world renders behind the menu (the bindless
        // ExecuteIndirect below would otherwise consume a stale indirect-command
        // buffer, since the per-frame rebuild + cull were skipped this frame).
        if world_hidden {
            return;
        }

        // Pipeline-independent main-pass state: topology, geometry buffers,
        // and the shader-visible descriptor heaps. Survives root-signature
        // changes, so it is set once before either sub-pass binds its pipeline.
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            cmd.IASetVertexBuffers(0, Some(&[self.geometry.vertex_buffer_view]));
            cmd.IASetIndexBuffer(Some(&self.geometry.index_buffer_view));
            cmd.SetDescriptorHeaps(&[
                Some(self.descriptors.srv_heap.clone()),
                Some(self.descriptors.sampler_heap.clone()),
            ]);
        }

        // SSAO (GTAO) ran ahead of this pass via the pre-graph
        // (`PassId::SsaoBlur` dispatches the bundled prepass + kernel +
        // blur). The RAW edge `ao_output` → Main pins SsaoBlur → Main
        // in the toposort, so by the time we get here the blurred
        // occlusion target is filled. The main fragment shaders sample
        // it via the standard SRV binding at the bindless slot the
        // SSAO encoder updates.

        // Build-time static objects render through the bindless pipeline
        // driven by a GPU-culled indirect command buffer.
        // A compute kernel frustum/distance-tests every build-time object and
        // writes one ExecuteIndirect command per object (survivors get a real
        // draw, culled / disabled objects an instance_count-0 no-op); the
        // bindless main pass then issues the whole buffer with a single
        // ExecuteIndirect; the CPU never walks the static draw list. Each
        // draw is stateless apart from the per-command object-id b0 root
        // constant, with model/material/textures fetched from the per-frame
        // GpuObjectData buffer + the bindless texture pool. Instances, streamed
        // chunks and runtime clones are records of their own in the same buffer.
        let use_bindless = self.cull.main_bindless_pso.is_some() && self.cull_count() > 0;
        if use_bindless {
            // The per-frame per-object SRV-pool record buffer
            // (`build_object_buffer`) and the cull compute dispatch
            // (`encode_cull`) both ran ahead of this pass via the
            // pre-graph: `build_object_buffer` as inline CPU prep
            // before the executor dispatch, and `encode_cull` through
            // the executor's `PassId::Cull` arm. The toposort's
            // RAW edge from `draw_args` to Main pins Cull → Main, so
            // by the time we get here `indirect_cmd_buffers[frame_idx]`
            // is filled with this frame's ExecuteIndirect commands.

            let bindless_pso = self.wireframe_or(
                self.cull
                    .main_bindless_pso
                    .as_ref()
                    .expect("bindless PSO is live"),
                self.wireframe.bindless.as_ref(),
            );
            let bindless_root = self
                .cull
                .main_bindless_root_sig
                .as_ref()
                .expect("bindless root signature is live alongside its PSO");
            let cull_sig = self
                .cull
                .cull_command_signature
                .as_ref()
                .expect("cull command signature is live alongside the bindless PSO");
            let indirect = &self.cull.indirect_cmd_buffers[frame_idx];
            let object_gva = com::gpu_va(&self.cull.object_buffer_resources[frame_idx]);

            // Main bindless pass: issue the GPU-culled command buffer. The b0
            // object-id root constant is set per command by the command
            // signature, so it is not bound here.
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                cmd.SetPipelineState(bindless_pso);
                cmd.SetGraphicsRootSignature(bindless_root);
                cmd.SetGraphicsRootConstantBufferView(1, view_gva);
                cmd.SetGraphicsRootConstantBufferView(2, light_gva);
                // [12] root SRV: per-scene GpuLight storage buffer (t1).
                cmd.SetGraphicsRootShaderResourceView(12, local_lights_gva);
                // [13] ClusterParams (b5) + [14] the per-cluster light lists
                // (t2). The main camera shades from its cluster's binned lights.
                cmd.SetGraphicsRootConstantBufferView(13, self.cluster_params_gva(frame_idx, true));
                cmd.SetGraphicsRootShaderResourceView(14, self.cluster_list_gva());
                self.bind_local_light_tables(cmd, super::LocalLightParams::BINDLESS);
                cmd.SetGraphicsRootConstantBufferView(3, shadow_ubo_gva);
                cmd.SetGraphicsRootDescriptorTable(4, self.shadow.srv_gpu);
                // [5] is the bindless texture pool (per-object SRV region base).
                cmd.SetGraphicsRootDescriptorTable(
                    5,
                    self.cull.bindless_pool_gpu[self.current_frame],
                );
                cmd.SetGraphicsRootDescriptorTable(6, self.descriptors.shadow_sampler_gpu);
                cmd.SetGraphicsRootDescriptorTable(7, self.descriptors.linear_sampler_gpu);
                // [8] root SRV: this frame's StructuredBuffer<GpuObjectData>.
                cmd.SetGraphicsRootShaderResourceView(8, object_gva);
                // [9] descriptor table: blurred SSAO occlusion (or 1x1 white
                // fallback when SSAO is disabled).
                cmd.SetGraphicsRootDescriptorTable(9, self.ssao_ao_srv_gpu());
                // [10] reflection-probe cube array + [11] the live ProbeSet (this
                // frame's boxes + count). The forward shader box-projects + blends
                // them for the specular reflection; count 0 keeps the sky.
                cmd.SetGraphicsRootDescriptorTable(10, self.probe_cube_table_gpu());
                cmd.SetGraphicsRootConstantBufferView(
                    11,
                    com::gpu_va(&self.probe.set_cbvs[frame_idx]),
                );
                // ExecuteIndirect #1: the static + instance prefix
                // `[0, skinned_record_base())` against the static VB/IB (bound
                // above). The skinned tail is drawn by a second ExecuteIndirect
                // below (different bound VB/IB), reading the same indirect buffer
                // from `skinned_record_base()` on.
                //
                // Once per shader bucket: bucket 0 runs under the bindless
                // pipeline bound above, each later bucket under its material
                // shader's own pipeline. The cull kernel wrote every record's
                // command into exactly one bucket's region, so the regions never
                // double-draw.
                cmd.ExecuteIndirect(
                    cull_sig,
                    self.skinned_record_base() as u32,
                    indirect,
                    0,
                    None::<&ID3D12Resource>,
                    0,
                );
            }
            // One CPU draw issued; the kernel-written ICB runs N indirect
            // commands inside, but the call count surfaced to the profiler
            // is the host-side draw. Mirrors Metal's bindless main pass.
            self.inc_draw_calls(1);
            self.inc_draw_calls(self.execute_bucket_regions(
                cmd,
                cull_sig,
                indirect,
                self.skinned_record_base() as u32,
            ));
            // Restore the default bindless pipeline for the sub-paths below.
            if self.shader_bucket_count() > 1 {
                // SAFETY: the command list is in the recording state, and every resource,
                // descriptor and slice these commands name is live for the call.
                unsafe { cmd.SetPipelineState(bindless_pso) };
            }
        }

        // Skinned meshes main pass. Skinned objects ride the same cull buffers as
        // static + instances and are drawn (as rigid deformed geometry) by a 2nd
        // `ExecuteIndirect` over this frame's deformed-vertex buffer + the skinned
        // index buffer, reading the cull-written indirect buffer from
        // `skinned_record_base()`. The `encode_skin` compute pass (Cull graph arm)
        // has already posed the deformed buffer and left it in
        // VERTEX_AND_CONSTANT_BUFFER.
        if use_bindless
            && self.draw.n_skinned > 0
            && let (Some(bindless_pso), Some(bindless_root), Some(cull_sig), Some(deformed_vbv)) = (
                self.cull.main_bindless_pso.as_ref(),
                self.cull.main_bindless_root_sig.as_ref(),
                self.cull.cull_command_signature.as_ref(),
                self.skinned.deformed_vbvs.get(frame_idx),
            )
        {
            let indirect = &self.cull.indirect_cmd_buffers[frame_idx];
            let bindless_pso = self.wireframe_or(bindless_pso, self.wireframe.bindless.as_ref());
            let object_gva = com::gpu_va(&self.cull.object_buffer_resources[frame_idx]);
            // SAFETY: the command list is in the recording state, and every resource,
            // descriptor and slice these commands name is live for the call.
            unsafe {
                cmd.SetPipelineState(bindless_pso);
                cmd.SetGraphicsRootSignature(bindless_root);
                cmd.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                );
                // Bind the deformed verts + skinned IB; the records carry
                // base_vertex = 0 (the deformed buffer mirrors global skinned
                // indexing) and index offsets into the skinned IB.
                cmd.IASetVertexBuffers(0, Some(&[*deformed_vbv]));
                cmd.IASetIndexBuffer(Some(&self.skinned.index_buffer_view));
                cmd.SetDescriptorHeaps(&[
                    Some(self.descriptors.srv_heap.clone()),
                    Some(self.descriptors.sampler_heap.clone()),
                ]);
                cmd.SetGraphicsRootConstantBufferView(1, view_gva);
                cmd.SetGraphicsRootConstantBufferView(2, light_gva);
                // [12] root SRV: per-scene GpuLight storage buffer (t1).
                cmd.SetGraphicsRootShaderResourceView(12, local_lights_gva);
                // [13] ClusterParams (b5) + [14] the per-cluster light lists
                // (t2). The main camera shades from its cluster's lights.
                cmd.SetGraphicsRootConstantBufferView(13, self.cluster_params_gva(frame_idx, true));
                cmd.SetGraphicsRootShaderResourceView(14, self.cluster_list_gva());
                self.bind_local_light_tables(cmd, super::LocalLightParams::BINDLESS);
                cmd.SetGraphicsRootConstantBufferView(3, shadow_ubo_gva);
                cmd.SetGraphicsRootDescriptorTable(4, self.shadow.srv_gpu);
                cmd.SetGraphicsRootDescriptorTable(
                    5,
                    self.cull.bindless_pool_gpu[self.current_frame],
                );
                cmd.SetGraphicsRootDescriptorTable(6, self.descriptors.shadow_sampler_gpu);
                cmd.SetGraphicsRootDescriptorTable(7, self.descriptors.linear_sampler_gpu);
                cmd.SetGraphicsRootShaderResourceView(8, object_gva);
                cmd.SetGraphicsRootDescriptorTable(9, self.ssao_ao_srv_gpu());
                cmd.SetGraphicsRootDescriptorTable(10, self.probe_cube_table_gpu());
                cmd.SetGraphicsRootConstantBufferView(
                    11,
                    com::gpu_va(&self.probe.set_cbvs[frame_idx]),
                );
                // ExecuteIndirect #2: skinned tail
                // `[skinned_record_base(), cull_count())`, byte-offset into the
                // same indirect command buffer.
                cmd.ExecuteIndirect(
                    cull_sig,
                    self.draw.n_skinned as u32,
                    indirect,
                    (self.skinned_record_base()
                        * crate::directx::cull::INDIRECT_COMMAND_STRIDE as usize)
                        as u64,
                    None::<&ID3D12Resource>,
                    0,
                );
            }
            self.inc_draw_calls(1);
        }

        // Resolve the HDR scene target so the post stack can sample it. Under
        // two-pass occlusion the resolve is deferred to `Main2` (which re-runs
        // the disoccluded geometry on top of this pass's colour + depth), so the
        // post stack sees the combined phase-1 + phase-2 scene. Skipping it here
        // leaves both targets in RENDER_TARGET, exactly as `Main2` expects to
        // load them.
        if !self.two_pass_occlusion_active() {
            self.finish_hdr_target(cmd);
        }
    }

    // Resolve the multisampled `hdr_color` into the single-sample `hdr_resolve`
    // that every later pass reads. A no-op with MSAA off, where there is no
    // separate resolve target and `hdr_color` *is* the spine. Shared by the
    // phase-1 main pass and the phase-2 `Main2` pass.
    //
    // Both targets are graph resources sitting in RENDER_TARGET here -- their
    // writes are what this pass declares -- so the transitions below are the
    // resolve step's own, finer than the one-state-per-resource the graph
    // models, and each returns its target to RENDER_TARGET before the pass ends.
    pub(in crate::directx) fn finish_hdr_target(&self, cmd: &ID3D12GraphicsCommandList) {
        let Some(hdr_resolve) = &self.hdr.resolve else {
            return;
        };
        let color_to_src = transition_barrier(
            &self.hdr.color,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
        );
        let resolve_to_dst = transition_barrier(
            self.hdr_scene_target(),
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RESOLVE_DEST,
        );
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe { cmd.ResourceBarrier(&[color_to_src, resolve_to_dst]) };
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe { cmd.ResolveSubresource(hdr_resolve, 0, &self.hdr.color, 0, HDR_FORMAT) };
        let resolve_to_rt = transition_barrier(
            self.hdr_scene_target(),
            D3D12_RESOURCE_STATE_RESOLVE_DEST,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        let color_to_rt = transition_barrier(
            &self.hdr.color,
            D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe { cmd.ResourceBarrier(&[resolve_to_rt, color_to_rt]) };
    }

    // Phase-2 main pass for two-pass occlusion (`Main2`). Loads (does not clear)
    // the HDR colour + depth `encode_main_pass` (phase 1) wrote and re-runs the
    // bindless indirect draw through this frame's second indirect buffer (the
    // phase-2 cull's output), depth-compositing the disoccluded geometry with
    // phase 1. Static + instances + skinned all ride the shared cull buffers, so
    // any of them that were occlusion candidates are re-tested by the phase-2 cull
    // and redrawn here with the same two-`ExecuteIndirect` split as phase 1 (the
    // static+instance prefix against the static VB/IB, the skinned tail against the
    // deformed VB + skinned IB). Finishes by resolving the HDR target (the resolve
    // phase 1 deferred), so the post-decoration stack reads the combined result.
    // Only dispatched when `two_pass_occlusion_active`
    // (the graph gates the Main2 node on it), so all phase-2 resources are
    // present; the resolve still runs even if there is nothing to redraw.
    // Mirrors `metal/draw/main.rs::encode_main_pass_phase2`.
    pub(in crate::directx) fn encode_main_pass_phase2(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        width: u32,
        height: u32,
        gpu: FrameGpuBuffers,
    ) {
        let FrameGpuBuffers {
            view_gva,
            light_gva,
            local_lights_gva,
            shadow_ubo_gva,
        } = gpu;
        let depth_dsv = self.depth.dsv;

        // Load (do not clear) the phase-1 colour + depth: Main2 composites the
        // disoccluded geometry on top.
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.OMSetRenderTargets(1, Some(&self.hdr.color_rtv), false, Some(&depth_dsv));

            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            cmd.RSSetViewports(&[vp]);
            let scissor = RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            };
            cmd.RSSetScissorRects(&[scissor]);

            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            cmd.IASetVertexBuffers(0, Some(&[self.geometry.vertex_buffer_view]));
            cmd.IASetIndexBuffer(Some(&self.geometry.index_buffer_view));
            cmd.SetDescriptorHeaps(&[
                Some(self.descriptors.srv_heap.clone()),
                Some(self.descriptors.sampler_heap.clone()),
            ]);
        }

        // Re-issue the bindless pass over the phase-2 indirect buffer. Same
        // bindings + the same static-prefix / skinned-tail split as the phase-1
        // bindless branch.
        if let (Some(bindless_pso), Some(bindless_root), Some(cull_sig), Some(indirect)) = (
            self.cull.main_bindless_pso.as_ref(),
            self.cull.main_bindless_root_sig.as_ref(),
            self.cull.cull_command_signature.as_ref(),
            self.cull.indirect_cmd_buffers_2.get(frame_idx),
        ) && self.cull_count() > 0
        {
            let bindless_pso = self.wireframe_or(bindless_pso, self.wireframe.bindless.as_ref());
            let object_gva = com::gpu_va(&self.cull.object_buffer_resources[frame_idx]);
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            unsafe {
                cmd.SetPipelineState(bindless_pso);
                cmd.SetGraphicsRootSignature(bindless_root);
                cmd.SetGraphicsRootConstantBufferView(1, view_gva);
                cmd.SetGraphicsRootConstantBufferView(2, light_gva);
                // [12] root SRV: per-scene GpuLight storage buffer (t1).
                cmd.SetGraphicsRootShaderResourceView(12, local_lights_gva);
                // [13] ClusterParams (b5) + [14] the per-cluster light lists
                // (t2). The main camera shades from its cluster's binned lights.
                cmd.SetGraphicsRootConstantBufferView(13, self.cluster_params_gva(frame_idx, true));
                cmd.SetGraphicsRootShaderResourceView(14, self.cluster_list_gva());
                self.bind_local_light_tables(cmd, super::LocalLightParams::BINDLESS);
                cmd.SetGraphicsRootConstantBufferView(3, shadow_ubo_gva);
                cmd.SetGraphicsRootDescriptorTable(4, self.shadow.srv_gpu);
                cmd.SetGraphicsRootDescriptorTable(
                    5,
                    self.cull.bindless_pool_gpu[self.current_frame],
                );
                cmd.SetGraphicsRootDescriptorTable(6, self.descriptors.shadow_sampler_gpu);
                cmd.SetGraphicsRootDescriptorTable(7, self.descriptors.linear_sampler_gpu);
                cmd.SetGraphicsRootShaderResourceView(8, object_gva);
                cmd.SetGraphicsRootDescriptorTable(9, self.ssao_ao_srv_gpu());
                cmd.SetGraphicsRootDescriptorTable(10, self.probe_cube_table_gpu());
                cmd.SetGraphicsRootConstantBufferView(
                    11,
                    com::gpu_va(&self.probe.set_cbvs[frame_idx]),
                );
                // ExecuteIndirect #1: static + instance prefix against the static
                // VB/IB (bound above), once per shader bucket.
                cmd.ExecuteIndirect(
                    cull_sig,
                    self.skinned_record_base() as u32,
                    indirect,
                    0,
                    None::<&ID3D12Resource>,
                    0,
                );
            }
            self.inc_draw_calls(1);
            self.inc_draw_calls(self.execute_bucket_regions(
                cmd,
                cull_sig,
                indirect,
                self.skinned_record_base() as u32,
            ));

            // ExecuteIndirect #2: skinned tail against the deformed VB + skinned
            // IB. The root signature + root descriptors set above persist, so only
            // the pipeline (a bucket may have replaced it) and the vertex/index
            // buffers rebind. Skinned draws always render bucket 0.
            if self.draw.n_skinned > 0
                && let Some(deformed_vbv) = self.skinned.deformed_vbvs.get(frame_idx)
            {
                // SAFETY: the command list is in the recording state, and every resource,
                // descriptor and slice these commands name is live for the call.
                unsafe {
                    cmd.SetPipelineState(bindless_pso);
                    cmd.IASetVertexBuffers(0, Some(&[*deformed_vbv]));
                    cmd.IASetIndexBuffer(Some(&self.skinned.index_buffer_view));
                    cmd.ExecuteIndirect(
                        cull_sig,
                        self.draw.n_skinned as u32,
                        indirect,
                        (self.skinned_record_base()
                            * crate::directx::cull::INDIRECT_COMMAND_STRIDE as usize)
                            as u64,
                        None::<&ID3D12Resource>,
                        0,
                    );
                }
                self.inc_draw_calls(1);
            }
        }

        // Resolve the combined phase-1 + phase-2 scene (the resolve phase 1
        // deferred). Always runs, even with nothing disoccluded, so the post
        // stack always reads a resolved target.
        self.finish_hdr_target(cmd);
    }
}
