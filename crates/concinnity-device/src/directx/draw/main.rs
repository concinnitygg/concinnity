//! Main HDR scene pass: two GPU-driven `ExecuteIndirect`s per shader bucket, the
//! static + instance + runtime prefix and the skinned tail. Renders linear-light
//! HDR into `hdr_color`; the composite pass tonemaps that down
//! onto the swapchain backbuffer. Ends by transitioning (or MSAA-resolving)
//! `hdr_color` to `PIXEL_SHADER_RESOURCE` so post-process passes can sample
//! it.

use concinnity_core::math::SaturatingFrom;
use concinnity_core::render::depth::DEPTH_CLEAR;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::com;
use crate::directx::context::DxContext;
use crate::directx::descriptor_slot::DescriptorTables;
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
        let spot_gva = com::gpu_va(&self.spot_shadow.buffer);
        // SAFETY: the command list is recording, and the spot projection buffer is owned by
        // `self.spot_shadow` for the world's lifetime.
        unsafe { cmd.SetGraphicsRootShaderResourceView(params.spot_buffer, spot_gva) };
        // SAFETY: the spot depth array's SRV is a static slot in the bound CBV/SRV/UAV heap.
        unsafe { cmd.set_graphics_srv_table(params.spot_table, self.spot_shadow.srv_gpu) };
        let area_gva = com::gpu_va(&self.scene.area_light.buffer);
        // SAFETY: the area-light table is owned by `self.scene` for the world's lifetime.
        unsafe { cmd.SetGraphicsRootShaderResourceView(params.area_buffer, area_gva) };
        let ltc = self.scene.area_light.ltc_table_gpu;
        // SAFETY: the LTC lookups' SRVs are static slots in the bound CBV/SRV/UAV heap.
        unsafe { cmd.set_graphics_srv_table(params.ltc_table, ltc) };
    }

    // Size the viewport and scissor to the full `width` x `height` target.
    fn set_full_viewport(cmd: &ID3D12GraphicsCommandList, width: u32, height: u32) {
        let vp = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: width as f32,
            Height: height as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let scissor = RECT {
            left: 0,
            top: 0,
            right: i32::saturating_from(width),
            bottom: i32::saturating_from(height),
        };
        // SAFETY: the caller's command list is recording, and the viewport slice outlives the call.
        unsafe { cmd.RSSetViewports(&[vp]) };
        // SAFETY: the scissor slice is a local that outlives the call.
        unsafe { cmd.RSSetScissorRects(&[scissor]) };
    }

    // Pipeline-independent main-pass state: topology, the static geometry
    // buffers and the shader-visible descriptor heaps.
    fn bind_main_geometry(&self, cmd: &ID3D12GraphicsCommandList) {
        // SAFETY: setting the topology only needs the caller's recording command list.
        unsafe {
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
        }
        let geometry = &self.scene.geometry;
        // SAFETY: the static vertex buffer is owned by `self.scene` and outlives the frame.
        unsafe { cmd.IASetVertexBuffers(0, Some(&[geometry.vertex_buffer_view])) };
        // SAFETY: the static index buffer is owned by `self.scene`; the view is read in the call.
        unsafe { cmd.IASetIndexBuffer(Some(&raw const geometry.index_buffer_view)) };
        // SAFETY: both heaps are owned by `self.descriptors` and outlive the frame.
        unsafe {
            cmd.SetDescriptorHeaps(&[
                Some(self.descriptors.srv_heap.clone()),
                Some(self.descriptors.sampler_heap.clone()),
            ]);
        }
    }

    // Bind every root argument the bindless main root signature declares, on a
    // list that already has that root signature and the descriptor heaps set.
    fn bind_main_root_args(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        gpu: FrameGpuBuffers,
        object_gva: u64,
    ) {
        // SAFETY: `view_gva` addresses this frame's live view constant buffer.
        unsafe { cmd.SetGraphicsRootConstantBufferView(1, gpu.view_gva) };
        // SAFETY: `light_gva` addresses this frame's live light constant buffer.
        unsafe { cmd.SetGraphicsRootConstantBufferView(2, gpu.light_gva) };
        // [12] root SRV: per-scene GpuLight storage buffer (t1).
        // SAFETY: the per-scene light list is static for the world's lifetime.
        unsafe { cmd.SetGraphicsRootShaderResourceView(12, gpu.local_lights_gva) };
        // [13] ClusterParams (b5) + [14] the per-cluster light lists
        // (t2). The main camera shades from its cluster's binned lights.
        let cluster_params = self.cluster_params_gva(frame_idx, true);
        // SAFETY: the cluster params for `frame_idx` live in a buffer the light cull owns.
        unsafe { cmd.SetGraphicsRootConstantBufferView(13, cluster_params) };
        let cluster_list = self.cluster_list_gva();
        // SAFETY: the per-cluster light lists live in a buffer the light cull owns.
        unsafe { cmd.SetGraphicsRootShaderResourceView(14, cluster_list) };
        self.bind_local_light_tables(cmd, super::LocalLightParams::BINDLESS);
        // SAFETY: `shadow_ubo_gva` addresses this frame's live shadow constant buffer.
        unsafe { cmd.SetGraphicsRootConstantBufferView(3, gpu.shadow_ubo_gva) };
        // SAFETY: the shadow map SRV is a slot in the bound CBV/SRV/UAV heap.
        unsafe { cmd.set_graphics_srv_table(4, self.shadow.srv_gpu) };
        // [5] is the bindless texture pool (per-object SRV region base).
        let pool = self.cull.bindless_pool_gpu[self.current_frame];
        // SAFETY: the current frame's bindless pool region lives in the bound heap.
        unsafe { cmd.set_graphics_srv_table(5, pool) };
        // SAFETY: the shadow comparison sampler lives in the bound sampler heap.
        unsafe { cmd.set_graphics_sampler_table(6, self.descriptors.shadow_sampler_gpu) };
        // SAFETY: the linear sampler lives in the bound sampler heap.
        unsafe { cmd.set_graphics_sampler_table(7, self.descriptors.linear_sampler_gpu) };
        // [8] root SRV: this frame's StructuredBuffer<GpuObjectData>.
        // SAFETY: `object_gva` addresses this frame's live object buffer.
        unsafe { cmd.SetGraphicsRootShaderResourceView(8, object_gva) };
        let vertices_gva = com::gpu_va(&self.scene.geometry.vertex_buffer);
        // SAFETY: the shared vertex stream is owned by `self.scene` and outlives the frame.
        unsafe {
            cmd.SetGraphicsRootShaderResourceView(super::MAIN_MESH_VERTICES_PARAM, vertices_gva);
        }
        let material_gva = self.material_params_gva(frame_idx);
        // SAFETY: the material params for `frame_idx` live in a buffer the context owns.
        unsafe {
            cmd.SetGraphicsRootShaderResourceView(
                crate::directx::material_params::MATERIAL_PARAMS_ROOT_PARAM,
                material_gva,
            );
        }
        let canopy_gva = self.canopy.gpu_va(frame_idx);
        // SAFETY: the canopy constants for `frame_idx` live in a buffer `self.canopy` owns.
        unsafe {
            cmd.SetGraphicsRootConstantBufferView(
                crate::directx::canopy::CANOPY_ROOT_PARAM,
                canopy_gva,
            );
        }
        let canopy_tables_gva = self.canopy.tables_gpu_va(frame_idx);
        // SAFETY: the canopy tables for `frame_idx` live in a buffer `self.canopy` owns.
        unsafe {
            cmd.SetGraphicsRootShaderResourceView(
                crate::directx::canopy::CANOPY_TABLES_ROOT_PARAM,
                canopy_tables_gva,
            );
        }
        // [9] descriptor table: the screen-space indirect light (SSGI's
        // visibility and bounce, SSAO's occlusion, or 1x1 white).
        let indirect_light = self.screen_indirect_srv_gpu();
        // SAFETY: the screen-space indirect light, or its white fallback, is a slot in the heap.
        unsafe { cmd.set_graphics_srv_table(9, indirect_light) };
        // [10] reflection-probe cube array + [11] the live count + [19]
        // this frame's boxes. The forward shader box-projects + blends them
        // for the specular reflection; count 0 keeps the sky.
        let probes = self.probe_bindings(frame_idx);
        self.bind_main_probe_set(cmd, probes.set_cbv, probes.records);
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
        grass: Option<&crate::directx::grass::GrassFrame>,
    ) {
        let MainPassExtent { width, height } = extent;
        let view_gva = gpu.view_gva;
        let depth_dsv = self.targets.depth.dsv;

        // SAFETY: the command list is recording, and the HDR color and depth views are owned by
        // `self.targets` and sit in their writable states for this pass.
        unsafe {
            cmd.OMSetRenderTargets(
                1,
                Some(&raw const self.targets.hdr.color_rtv),
                false,
                Some(&raw const depth_dsv),
            );
        }
        let clear_color = self.pre_exposure.clear_color(self.state.view.clear_color);
        // SAFETY: the HDR color RTV is live and bound above; the clear color is a local.
        unsafe { cmd.ClearRenderTargetView(self.targets.hdr.color_rtv, &clear_color, None) };
        // SAFETY: the depth DSV is live and bound above, and no rects are passed.
        unsafe {
            cmd.ClearDepthStencilView(depth_dsv, D3D12_CLEAR_FLAG_DEPTH, DEPTH_CLEAR, 0, None);
        }
        Self::set_full_viewport(cmd, width, height);

        // Opaque menu backdrop: the render target was just cleared; skip every
        // draw so nothing of the world renders behind the menu (the bindless
        // ExecuteIndirect below would otherwise consume a stale indirect-command
        // buffer, since the per-frame rebuild + cull were skipped this frame).
        if world_hidden {
            return;
        }

        // Pipeline-independent main-pass state survives root-signature
        // changes, so it is set once before either sub-pass binds its pipeline.
        self.bind_main_geometry(cmd);

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
        let use_bindless =
            self.cull.main_bindless_pso.is_some() && self.state.draw.record_count() > 0;
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
            // SAFETY: the command list is recording, and the bindless PSO is owned by `self.cull`.
            unsafe { cmd.SetPipelineState(bindless_pso) };
            // SAFETY: the bindless root signature is owned by `self.cull` alongside its PSO.
            unsafe { cmd.SetGraphicsRootSignature(bindless_root) };
            self.bind_main_root_args(cmd, frame_idx, gpu, object_gva);
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
            let prefix_count = u32::saturating_from(self.state.draw.skinned_record_base());
            // SAFETY: the cull pass filled this frame's indirect buffer ahead of this pass, and
            // the command signature matches the bindless root signature bound above.
            unsafe {
                cmd.ExecuteIndirect(
                    cull_sig,
                    prefix_count,
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
                u32::saturating_from(self.state.draw.skinned_record_base()),
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
            && self.state.draw.n_skinned > 0
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
            // SAFETY: the command list is recording, and the bindless PSO is owned by `self.cull`.
            unsafe { cmd.SetPipelineState(bindless_pso) };
            // SAFETY: the bindless root signature is owned by `self.cull` alongside its PSO.
            unsafe { cmd.SetGraphicsRootSignature(bindless_root) };
            // SAFETY: setting the topology only needs a recording command list.
            unsafe {
                cmd.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                );
            }
            // Bind the deformed verts + skinned IB; the records carry
            // base_vertex = 0 (the deformed buffer mirrors global skinned
            // indexing) and index offsets into the skinned IB.
            // SAFETY: the skin pass left this frame's deformed buffer in
            // VERTEX_AND_CONSTANT_BUFFER, and `self.skinned` owns it.
            unsafe { cmd.IASetVertexBuffers(0, Some(&[*deformed_vbv])) };
            // SAFETY: the skinned index buffer is owned by `self.skinned` and outlives the frame.
            unsafe { cmd.IASetIndexBuffer(Some(&raw const self.skinned.index_buffer_view)) };
            // SAFETY: both heaps are owned by `self.descriptors` and outlive the frame.
            unsafe {
                cmd.SetDescriptorHeaps(&[
                    Some(self.descriptors.srv_heap.clone()),
                    Some(self.descriptors.sampler_heap.clone()),
                ]);
            }
            self.bind_main_root_args(cmd, frame_idx, gpu, object_gva);
            // ExecuteIndirect #2: skinned tail
            // `[skinned_record_base(), record_count())`, byte-offset into the
            // same indirect command buffer.
            let skinned_count = u32::saturating_from(self.state.draw.n_skinned);
            let skinned_offset = (self.state.draw.skinned_record_base()
                * crate::directx::cull::INDIRECT_COMMAND_STRIDE as usize)
                as u64;
            // SAFETY: the cull pass wrote the skinned tail at `skinned_offset` inside this frame's
            // indirect buffer, and the bindless root signature is bound above.
            unsafe {
                cmd.ExecuteIndirect(
                    cull_sig,
                    skinned_count,
                    indirect,
                    skinned_offset,
                    None::<&ID3D12Resource>,
                    0,
                );
            }
            self.inc_draw_calls(1);
        }

        // The blades shade on the lights, shadows and environment the surfaces
        // just bound under the main root signature.
        if let (Some(frame), true) = (grass, use_bindless) {
            self.encode_grass_main(cmd, frame_idx, frame);
        }

        // The sky lands behind phase 1's geometry. Under two-pass occlusion
        // `Main2` then draws the disoccluded rest over it, which wins the depth
        // test because the sky writes none.
        if self.draws_sky(self.state.view.mode) {
            self.encode_main_sky(cmd, frame_idx, view_gva);
        }

        // Resolve the HDR scene target so the post stack can sample it. Under
        // two-pass occlusion the resolve is deferred to `Main2` (which re-runs
        // the disoccluded geometry on top of this pass's color + depth), so the
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
        let Some(hdr_resolve) = &self.targets.hdr.resolve else {
            return;
        };
        let color_to_src = transition_barrier(
            &self.targets.hdr.color,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
        );
        let resolve_to_dst = transition_barrier(
            self.hdr_scene_target(),
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RESOLVE_DEST,
        );
        // SAFETY: the command list is recording, and both targets sit in RENDER_TARGET here.
        unsafe { cmd.ResourceBarrier(&[color_to_src, resolve_to_dst]) };
        // SAFETY: the barrier above put the color in RESOLVE_SOURCE and the resolve target in
        // RESOLVE_DEST; both share `HDR_FORMAT`.
        unsafe { cmd.ResolveSubresource(hdr_resolve, 0, &self.targets.hdr.color, 0, HDR_FORMAT) };
        let resolve_to_rt = transition_barrier(
            self.hdr_scene_target(),
            D3D12_RESOURCE_STATE_RESOLVE_DEST,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        let color_to_rt = transition_barrier(
            &self.targets.hdr.color,
            D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe { cmd.ResourceBarrier(&[resolve_to_rt, color_to_rt]) };
    }

    // Phase-2 main pass for two-pass occlusion (`Main2`). Loads (does not clear)
    // the HDR color + depth `encode_main_pass` (phase 1) wrote and re-runs the
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
        let depth_dsv = self.targets.depth.dsv;

        // Load (do not clear) the phase-1 color + depth: Main2 composites the
        // disoccluded geometry on top.
        // SAFETY: the command list is recording, and phase 1 left the HDR color and depth these
        // views name in their writable states.
        unsafe {
            cmd.OMSetRenderTargets(
                1,
                Some(&raw const self.targets.hdr.color_rtv),
                false,
                Some(&raw const depth_dsv),
            );
        }
        Self::set_full_viewport(cmd, width, height);
        self.bind_main_geometry(cmd);

        // Re-issue the bindless pass over the phase-2 indirect buffer. Same
        // bindings + the same static-prefix / skinned-tail split as the phase-1
        // bindless branch.
        if let (Some(bindless_pso), Some(bindless_root), Some(cull_sig), Some(indirect)) = (
            self.cull.main_bindless_pso.as_ref(),
            self.cull.main_bindless_root_sig.as_ref(),
            self.cull.cull_command_signature.as_ref(),
            self.cull.indirect_cmd_buffers_2.get(frame_idx),
        ) && self.state.draw.record_count() > 0
        {
            let bindless_pso = self.wireframe_or(bindless_pso, self.wireframe.bindless.as_ref());
            let object_gva = com::gpu_va(&self.cull.object_buffer_resources[frame_idx]);
            // SAFETY: the command list is recording, and the bindless PSO is owned by `self.cull`.
            unsafe { cmd.SetPipelineState(bindless_pso) };
            // SAFETY: the bindless root signature is owned by `self.cull` alongside its PSO.
            unsafe { cmd.SetGraphicsRootSignature(bindless_root) };
            self.bind_main_root_args(cmd, frame_idx, gpu, object_gva);
            // ExecuteIndirect #1: static + instance prefix against the static
            // VB/IB (bound above), once per shader bucket.
            let prefix_count = u32::saturating_from(self.state.draw.skinned_record_base());
            // SAFETY: the phase-2 cull filled this frame's second indirect buffer ahead of this
            // pass, and the command signature matches the root signature bound above.
            unsafe {
                cmd.ExecuteIndirect(
                    cull_sig,
                    prefix_count,
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
                u32::saturating_from(self.state.draw.skinned_record_base()),
            ));

            // ExecuteIndirect #2: skinned tail against the deformed VB + skinned
            // IB. The root signature + root descriptors set above persist, so only
            // the pipeline (a bucket may have replaced it) and the vertex/index
            // buffers rebind. Skinned draws always render bucket 0.
            if self.state.draw.n_skinned > 0
                && let Some(deformed_vbv) = self.skinned.deformed_vbvs.get(frame_idx)
            {
                // SAFETY: the command list is recording, and the bindless PSO is owned by
                // `self.cull`.
                unsafe { cmd.SetPipelineState(bindless_pso) };
                // SAFETY: the skin pass left this frame's deformed buffer in
                // VERTEX_AND_CONSTANT_BUFFER, and `self.skinned` owns it.
                unsafe { cmd.IASetVertexBuffers(0, Some(&[*deformed_vbv])) };
                // SAFETY: the skinned index buffer is owned by `self.skinned` and outlives the
                // frame.
                unsafe { cmd.IASetIndexBuffer(Some(&raw const self.skinned.index_buffer_view)) };
                let skinned_count = u32::saturating_from(self.state.draw.n_skinned);
                let skinned_offset = (self.state.draw.skinned_record_base()
                    * crate::directx::cull::INDIRECT_COMMAND_STRIDE as usize)
                    as u64;
                // SAFETY: the phase-2 cull wrote the skinned tail at `skinned_offset` inside the
                // second indirect buffer, and the root arguments set above persist.
                unsafe {
                    cmd.ExecuteIndirect(
                        cull_sig,
                        skinned_count,
                        indirect,
                        skinned_offset,
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
