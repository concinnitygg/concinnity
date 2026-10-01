//! Skinned-mesh resources for DxContext: the skinned geometry upload (built
//! lazily by `upload_skinned` the first time a SkinnedMesh is uploaded), and
//! the per-frame joint / morph-weight uploads.
//! The per-slot CPU records these uploads read live in `gfx::skinned_slots`.

use concinnity_core::gfx::mesh_payload;
use concinnity_core::gfx::mesh_payload::{SkinnedVertex, Vertex};
use concinnity_core::gfx::render_types::*;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::rt_geom;
use concinnity_core::render::skinned_slots;
use concinnity_core::transform::IDENTITY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::super::allocator::PooledBuffer;
use super::super::com;
use super::super::context::*;
use super::super::error::map_hresult;
use super::super::texture::*;

// Skinned (skeletally animated) mesh rendering. All `None` / empty until
// `upload_skinned` runs; with no `SkinnedMesh` in the world every skinned pass
// is skipped. Every pass draws the skin fold's deformed vertices.
pub(in crate::directx) struct SkinnedState {
    // Shared skinned vertex/index buffers. Kept alive for the GPU; referenced
    // through `vertex_buffer_view` / `index_buffer_view`.
    pub vertex_buffer: Option<PooledBuffer>,
    pub index_buffer: Option<PooledBuffer>,
    pub vertex_buffer_view: D3D12_VERTEX_BUFFER_VIEW,
    pub index_buffer_view: D3D12_INDEX_BUFFER_VIEW,
    // Per-slot draw objects, joint palettes, and morph weights: the CPU-side
    // records this backend shares with Metal and Vulkan.
    pub slots: skinned_slots::SkinnedSlots,
    // Per-frame, per-object joint-matrix upload buffers, indexed
    // [frame_idx][skinned_idx]. Each holds MAX_JOINTS float4x4 matrices,
    // persistently mapped; rewritten each frame from `slots.joint_matrices`.
    pub joint_buffers: Vec<Vec<PooledBuffer>>,
    pub joint_ptrs: Vec<Vec<*mut u8>>,
    // GPU-driven main-pass skinning. `skin_pipeline` is the `rt_skin` compute
    // kernel reused to deform the bind-pose verts into a per-frame buffer for the
    // bindless main pass (independent of RT, which keeps its own skin dispatch);
    // built in `upload_skinned`. `deformed_buffers` is one UAV-writable buffer per
    // frame-in-flight holding this frame's posed 56-byte `Vertex`s (global skinned
    // indexing, so the draw uses `base_vertex = 0`); rests in
    // VERTEX_AND_CONSTANT_BUFFER, flipped to UNORDERED_ACCESS for the skin
    // dispatch each frame. `deformed_vbvs` is the parallel vertex-buffer view the
    // main pass's 2nd `ExecuteIndirect` binds. All empty / `None` until
    // `upload_skinned` runs.
    pub skin_pipeline: Option<crate::directx::raytrace::SkinPipeline>,
    pub deformed_buffers: Vec<ID3D12Resource>,
    pub deformed_vbvs: Vec<D3D12_VERTEX_BUFFER_VIEW>,
    // Morph targets, parallel to `slots.draw_objects`. `morph_delta_buffers[i]`
    // is the per-mesh packed sparse morph buffer
    // (`PayloadMorphs::packed_words`; instance copies share the
    // template's resource) or `None` for a mesh without morph targets;
    // `morph_target_counts[i]` is its target count (0 = none). The per-frame
    // `morph_weight_buffers` ([frame_idx][skinned_idx], one f32 per target,
    // persistently mapped) are filled from `slots.morph_weights` by
    // `upload_morph_weights`, and are empty when no skinned object carries
    // morphs.
    pub morph_delta_buffers: Vec<Option<PooledBuffer>>,
    pub morph_target_counts: Vec<u32>,
    pub morph_weight_buffers: Vec<Vec<PooledBuffer>>,
    pub morph_weight_ptrs: Vec<Vec<*mut u8>>,
    // `false` until the deformed-vertex ring has been posed at least one full
    // frame; `true` once a prior frame's `encode_skin` has filled the slot the
    // next frame reads as its velocity history. While false the GPU-driven
    // G-buffer velocity binds the current deformed buffer as the previous one
    // (prev_pos == cur_pos), so an unposed ring slot never feeds a garbage
    // skinned motion vector on the first frame (or after a runtime ring rebuild).
    // Matches Metal's `deformed_primed` gate. Reset by `upload_skinned`. Atomic, not `Cell`: the G-buffer pass
    // encodes on a `jobs::pool()` rayon worker thread (the parallel per-pass
    // encoder shares `&self` across workers), so any interior mutation reachable
    // from `encode_pass_into` must be atomic, like `draw_calls_accum`.
    pub deformed_primed: std::sync::atomic::AtomicBool,
}

impl SkinnedState {
    pub(in crate::directx) fn new() -> Self {
        Self {
            vertex_buffer: None,
            index_buffer: None,
            vertex_buffer_view: D3D12_VERTEX_BUFFER_VIEW::default(),
            index_buffer_view: D3D12_INDEX_BUFFER_VIEW::default(),
            slots: skinned_slots::SkinnedSlots::new(),
            joint_buffers: Vec::new(),
            joint_ptrs: Vec::new(),
            skin_pipeline: None,
            deformed_primed: std::sync::atomic::AtomicBool::new(false),
            deformed_buffers: Vec::new(),
            deformed_vbvs: Vec::new(),
            morph_delta_buffers: Vec::new(),
            morph_target_counts: Vec::new(),
            morph_weight_buffers: Vec::new(),
            morph_weight_ptrs: Vec::new(),
        }
    }
}
impl DxContext {
    // Upload skinned-mesh geometry and build the skin fold.
    //
    // Called once at init by `GraphicsSystem` when the world declares at least
    // one `SkinnedMesh`. The joint matrices live in per-(frame, object) upload
    // buffers the skinned passes bind as a root SRV. With no skinned meshes
    // this is never called and every skinned pass is skipped.
    pub(crate) fn upload_skinned(
        &mut self,
        vertices: &[SkinnedVertex],
        indices: &[u32],
        draw_objects: Vec<SkinnedDrawObject>,
    ) -> RenderResult<()> {
        if draw_objects.is_empty() || vertices.is_empty() || indices.is_empty() {
            return Ok(());
        }
        if draw_objects.len() > MAX_SKINNED_OBJECTS {
            return Err(RenderError::Other(format!(
                "skinned: {} skinned meshes exceeds MAX_SKINNED_OBJECTS ({})",
                draw_objects.len(),
                MAX_SKINNED_OBJECTS
            )));
        }
        self.wait_idle();

        // Shared skinned vertex/index buffers (DEFAULT heap, GPU-copied once).
        let vtx_bytes = bytemuck::cast_slice(vertices);
        let idx_bytes = bytemuck::cast_slice(indices);
        // GENERIC_READ (rather than the narrower VERTEX_AND_CONSTANT_BUFFER /
        // INDEX_BUFFER) so these stay both vertex/index-bindable for the skinned
        // main + shadow passes AND shader-readable as raw root SRVs for the RT
        // skin compute dispatch (bind-pose VB) and the RT reflection trace (u32
        // IB). GENERIC_READ is a superset of both, so no per-frame transition on
        // these shared resources is needed.
        let skinned_vertex_buffer =
            upload_buffer(&self.hw.alloc, vtx_bytes, D3D12_RESOURCE_STATE_GENERIC_READ)?;
        // Never zero-length: the ray-traced hit path binds this buffer as a raw
        // word array and no backend accepts a zero-length binding.
        let skinned_index_buffer = upload_buffer_padded(
            &self.hw.alloc,
            idx_bytes,
            rt_geom::skinned_index_buffer_bytes(indices.len()) as u64,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        self.skinned.vertex_buffer_view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: com::gpu_va(&skinned_vertex_buffer),
            SizeInBytes: vtx_bytes.len() as u32,
            StrideInBytes: std::mem::size_of::<SkinnedVertex>() as u32,
        };
        self.skinned.index_buffer_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: com::gpu_va(&skinned_index_buffer),
            SizeInBytes: idx_bytes.len() as u32,
            Format: DXGI_FORMAT_R32_UINT,
        };

        // Per-(frame, object) joint-matrix upload buffers, each MAX_JOINTS
        // float4x4 matrices, persistently mapped.
        //
        // The buffer is seeded with `MAX_JOINTS` identity matrices once at
        // creation. `upload_joint_matrices` later overwrites only the first
        // `mats.len()` slots each frame; anything past the live pose count
        // keeps the identity seed, so a vertex whose `joints.{xyzw}` indexes
        // past the live range degenerates into an LBS of identity matrices
        // (i.e. its bind-pose position) instead of reading uninitialized
        // UPLOAD-heap memory and producing an arbitrary spike. The seed is
        // also what the renderer wants on frame 0 before the first pose
        // arrives: every joint is identity, so the mesh shows in bind pose.
        let joint_buf_bytes = (MAX_JOINTS * std::mem::size_of::<[[f32; 4]; 4]>()) as u64;
        let identity_seed: Vec<[[f32; 4]; 4]> = vec![IDENTITY; MAX_JOINTS];
        let mut joint_buffers: Vec<Vec<PooledBuffer>> = Vec::with_capacity(FRAMES);
        let mut joint_ptrs: Vec<Vec<*mut u8>> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let mut frame_bufs: Vec<PooledBuffer> = Vec::with_capacity(draw_objects.len());
            let mut frame_ptrs: Vec<*mut u8> = Vec::with_capacity(draw_objects.len());
            for _ in 0..draw_objects.len() {
                let buf = self
                    .hw
                    .alloc
                    .alloc_buffer(
                        joint_buf_bytes,
                        D3D12_HEAP_TYPE_UPLOAD,
                        D3D12_RESOURCE_STATE_GENERIC_READ,
                    )
                    .map_err(|e| e.context("skinned joint buf"))?;
                let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
                // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload,
                // and the source is a separate allocation, so the ranges cannot overlap.
                unsafe {
                    buf.Map(0, None, Some(&mut ptr))
                        .map_err(|e| map_hresult(e.code(), "map skinned joint buf"))?;
                    std::ptr::copy_nonoverlapping(
                        identity_seed.as_ptr() as *const u8,
                        ptr as *mut u8,
                        joint_buf_bytes as usize,
                    );
                }
                frame_bufs.push(buf);
                frame_ptrs.push(ptr as *mut u8);
            }
            joint_buffers.push(frame_bufs);
            joint_ptrs.push(frame_ptrs);
        }

        // Seed each object's joint matrices to identity (bind pose) so the mesh
        // renders undeformed until the first `update_skinned_pose`.
        self.skinned.slots.joint_matrices = draw_objects
            .iter()
            .map(|o| vec![IDENTITY; o.joint_count.max(1)])
            .collect();

        self.skinned.vertex_buffer = Some(skinned_vertex_buffer);
        self.skinned.index_buffer = Some(skinned_index_buffer);
        self.skinned.joint_buffers = joint_buffers;
        self.skinned.joint_ptrs = joint_ptrs;
        self.skinned.slots.draw_objects = draw_objects;
        // A whole new skinned set: nothing in the model-history ring was written
        // for these records.
        self.model_history.borrow_mut().reset(self.cull_count());

        // Morph targets are attached by a later `upload_skinned_morphs`; until
        // then every object is morphless (a re-upload / hot-reload resets here).
        let n_objects = self.skinned.slots.draw_objects.len();
        self.skinned.morph_delta_buffers = (0..n_objects).map(|_| None).collect();
        self.skinned.morph_target_counts = vec![0; n_objects];
        self.skinned.slots.morph_weights = vec![Vec::new(); n_objects];
        self.skinned.morph_weight_buffers = Vec::new();
        self.skinned.morph_weight_ptrs = Vec::new();

        // GPU-driven main-pass skinning: build the `rt_skin` compute pipeline
        // (reused independently of RT) + one UAV-writable deformed-vertex buffer
        // per frame-in-flight, sized to all skinned verts. Each frame
        // `encode_skin` poses the bind-pose verts into this frame's buffer and
        // the main pass's 2nd ExecuteIndirect draws the skinned records the cull
        // buffers reserved at init via the threaded `n_skinned` capacity.
        // Setting `self.draw.n_skinned` here engages the fold. Every skinned draw
        // rides the GPU-driven pass, so a build failure is a startup error, as
        // on Metal.
        {
            let stride = std::mem::size_of::<Vertex>();
            let deformed_bytes = (vertices.len() * stride).max(stride) as u64;
            let mut deformed_buffers: Vec<ID3D12Resource> = Vec::with_capacity(FRAMES);
            let mut deformed_vbvs: Vec<D3D12_VERTEX_BUFFER_VIEW> = Vec::with_capacity(FRAMES);
            for _ in 0..FRAMES {
                let buf = create_uav_buffer(
                    &self.hw.device,
                    deformed_bytes,
                    D3D12_RESOURCE_STATE_COMMON,
                )?;
                let vbv = D3D12_VERTEX_BUFFER_VIEW {
                    BufferLocation: com::gpu_va(&buf),
                    SizeInBytes: deformed_bytes as u32,
                    StrideInBytes: stride as u32,
                };
                deformed_buffers.push(buf);
                deformed_vbvs.push(vbv);
            }
            // Move COMMON -> VERTEX_AND_CONSTANT_BUFFER so the per-frame skin
            // pass's VERTEX -> UAV -> VERTEX transition cycle is valid from frame 0.
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            one_shot_submit(&self.hw.device, &self.hw.command_queue, |cmd| unsafe {
                let barriers: Vec<D3D12_RESOURCE_BARRIER> = deformed_buffers
                    .iter()
                    .map(|b| {
                        transition_barrier(
                            b,
                            D3D12_RESOURCE_STATE_COMMON,
                            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                        )
                    })
                    .collect();
                cmd.ResourceBarrier(&barriers);
            })?;
            let skin = super::super::raytrace::build_rt_skin_pipeline(
                &self.hw.device,
                self.hot_reload.enabled,
            )
            .map_err(|e| e.context("skinned: main-pass skin fold build failed"))?;
            self.skinned.skin_pipeline = Some(skin);
            self.skinned.deformed_buffers = deformed_buffers;
            self.skinned.deformed_vbvs = deformed_vbvs;
            // Fresh ring: no slot has been posed yet, so the G-buffer velocity
            // must treat the previous deformed buffer as the current one until a
            // full frame has primed it.
            self.skinned
                .deformed_primed
                .store(false, std::sync::atomic::Ordering::Relaxed);
            self.draw.n_skinned = self.skinned.slots.draw_objects.len();
        }

        Ok(())
    }

    // Overwrite a `SkinnedMesh` draw slot's vertex + index data in the shared
    // skinned vertex / index buffers in place. Driven by asset hot-reload
    // (`cn debug` only). The slot's vertex region starts at
    // `vertex_base * size_of::<SkinnedVertex>()` and is `vertices.len()`
    // vertices wide; the index region lives at the slot's init-time
    // `index_offset` / `index_count`. Indices are rebased onto `vertex_base`
    // before writing (matching the init-time `upload_skinned` rebasing).
    // `indices.len()` must match init-time; size-changing reloads route
    // through `rebuild_skinned_geometry`. Joint-count
    // changes resize the per-slot joint-matrix buffers via
    // `update_skinned_skeleton`. Pipelines stay untouched.
    // Mirrors `MtlContext::update_skinned_mesh_geometry`.
    pub(crate) fn update_skinned_mesh_geometry(
        &mut self,
        skinned_index: SkinnedIndex,
        vertex_base: u32,
        vertices: &[SkinnedVertex],
        indices: &[u16],
    ) -> RenderResult<()> {
        let obj = self
            .skinned
            .slots
            .draw_objects
            .get(skinned_index.index())
            .ok_or_else(|| {
                RenderError::Other(format!(
                    "update_skinned_mesh_geometry: skinned object {} out of range",
                    skinned_index
                ))
            })?;
        if indices.len() != obj.index_count {
            return Err(RenderError::Other(format!(
                "update_skinned_mesh_geometry: skinned {} expects {} indices, got {} \
                 (in-place path is size-matched only; size changes route through \
                 rebuild_skinned_geometry)",
                skinned_index,
                obj.index_count,
                indices.len()
            )));
        }
        let v_buf = self.skinned.vertex_buffer.clone().ok_or_else(|| {
            RenderError::Other(
                "update_skinned_mesh_geometry: no skinned vertex buffer (was upload_skinned called?)"
                    .into(),
            )
        })?;
        let i_buf = self.skinned.index_buffer.clone().ok_or_else(|| {
            RenderError::Other(
                "update_skinned_mesh_geometry: no skinned index buffer (was upload_skinned called?)"
                    .into(),
            )
        })?;
        // Check the vertex region fits inside the live buffer. The shared
        // buffer was sized once at `upload_skinned` to hold every skinned
        // mesh's vertices; vertex_base + vertices.len() must stay within that
        // region or a neighboring slot would be overwritten.
        let v_byte_off = (vertex_base as usize) * std::mem::size_of::<SkinnedVertex>();
        let v_byte_len = std::mem::size_of_val(vertices);
        let v_buf_len = self.skinned.vertex_buffer_view.SizeInBytes as usize;
        if v_byte_off + v_byte_len > v_buf_len {
            return Err(RenderError::Other(format!(
                "update_skinned_mesh_geometry: vertex region [{}, {}) overruns skinned \
                 vertex buffer length {}",
                v_byte_off,
                v_byte_off + v_byte_len,
                v_buf_len
            )));
        }
        let i_byte_off = (obj.index_offset * std::mem::size_of::<u32>()) as u64;
        let rebased: Vec<u32> = indices
            .iter()
            .map(|&i| u32::from(i) + vertex_base)
            .collect();

        self.wait_idle();

        let vert_bytes = bytemuck::cast_slice(vertices);
        self.write_geometry_region(
            &v_buf,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            v_byte_off as u64,
            vert_bytes,
        )?;
        let idx_bytes = bytemuck::cast_slice(&rebased);
        self.write_geometry_region(
            &i_buf,
            D3D12_RESOURCE_STATE_INDEX_BUFFER,
            i_byte_off,
            idx_bytes,
        )?;
        Ok(())
    }

    // The CPU-side skinned entry points the `RenderBackend` impl forwards to.
    // Each is the `SkinnedSlots` operation of the same name; the behavior and
    // its contract are documented there, once for all three backends.
    pub(crate) fn update_skinned_skeleton(
        &mut self,
        skinned_index: SkinnedIndex,
        new_joint_count: usize,
    ) -> RenderResult<()> {
        self.skinned
            .slots
            .update_skeleton(skinned_index, new_joint_count)
            .map_err(RenderError::Other)
    }

    pub(crate) fn update_skinned_pose(
        &mut self,
        skinned_index: SkinnedIndex,
        matrices: &[[[f32; 4]; 4]],
    ) {
        self.skinned.slots.update_pose(skinned_index, matrices);
    }

    pub(crate) fn reveal_skinned_instance(
        &mut self,
        instance_index: SkinnedIndex,
        model: [[f32; 4]; 4],
    ) {
        self.skinned
            .slots
            .reveal(instance_index, model, &mut self.model_history.borrow_mut());
    }

    pub(crate) fn retire_skinned_draw_object(&mut self, skinned_index: SkinnedIndex) {
        self.skinned.slots.retire(skinned_index);
    }

    pub(crate) fn update_skinned_models(&mut self, updates: &[(SkinnedIndex, [[f32; 4]; 4])]) {
        self.skinned.slots.update_models(updates);
    }

    // Copy this frame's skinning matrices into the per-frame joint buffers.
    // Called from `record_frame` before the skin fold reads them.
    pub(in crate::directx) fn upload_joint_matrices(&self, frame_idx: usize) {
        let Some(frame_ptrs) = self.skinned.joint_ptrs.get(frame_idx) else {
            return;
        };
        for (i, mats) in self.skinned.slots.joint_matrices.iter().enumerate() {
            let Some(&dst) = frame_ptrs.get(i) else {
                continue;
            };
            let n = mats.len().min(MAX_JOINTS);
            // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and
            // the source is a separate allocation, so the ranges cannot overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    mats.as_ptr() as *const u8,
                    dst,
                    n * std::mem::size_of::<[[f32; 4]; 4]>(),
                );
            }
        }
    }

    // GPU virtual address of skinned object `i`'s joint buffer for `frame_idx`.
    pub(in crate::directx) fn skinned_joint_gva(&self, frame_idx: usize, i: usize) -> u64 {
        com::gpu_va(&self.skinned.joint_buffers[frame_idx][i])
    }

    // Attach morph-target buffers (`PayloadMorphs::packed_words`) to the skinned
    // draw objects. `morphs[i]` pairs with draw object `i`; instance copies share
    // their template's `Arc`, so each unique entry set becomes one GPU buffer. Allocates the per-frame
    // weight upload buffers (one f32 per target per object) when any object
    // carries morphs. Called once after `upload_skinned`.
    pub(in crate::directx) fn upload_skinned_morphs(
        &mut self,
        morphs: Vec<Option<std::sync::Arc<mesh_payload::PayloadMorphs>>>,
    ) -> RenderResult<()> {
        use std::collections::HashMap;

        let n = self.skinned.slots.draw_objects.len();
        let mut delta_buffers: Vec<Option<PooledBuffer>> = Vec::with_capacity(n);
        let mut target_counts: Vec<u32> = Vec::with_capacity(n);
        let mut weights: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut by_source: HashMap<usize, (PooledBuffer, u32)> = HashMap::new();

        for m in morphs.iter().take(n) {
            match m {
                None => {
                    delta_buffers.push(None);
                    target_counts.push(0);
                    weights.push(Vec::new());
                }
                Some(data) => {
                    let key = std::sync::Arc::as_ptr(data) as usize;
                    let (buf, count) = match by_source.get(&key) {
                        Some(entry) => entry.clone(),
                        None => {
                            let words = data.packed_words();
                            let bytes: &[u8] = bytemuck::cast_slice(&words);
                            let buf = upload_buffer(
                                &self.hw.alloc,
                                bytes,
                                D3D12_RESOURCE_STATE_GENERIC_READ,
                            )?;
                            let count = data.target_count() as u32;
                            by_source.insert(key, (buf.clone(), count));
                            (buf, count)
                        }
                    };
                    delta_buffers.push(Some(buf));
                    weights.push(vec![0.0; count as usize]);
                    target_counts.push(count);
                }
            }
        }
        // Pad the tail morphless if `morphs` was shorter than `draw.objects`.
        while delta_buffers.len() < n {
            delta_buffers.push(None);
            target_counts.push(0);
            weights.push(Vec::new());
        }

        // Per-(frame, object) weight upload buffers, one f32 per target (>= 1 so
        // every slot has a valid GVA), persistently mapped and zero-seeded. Only
        // allocated when some object carries morphs.
        let (mut weight_buffers, mut weight_ptrs) = (Vec::new(), Vec::new());
        if target_counts.iter().any(|&c| c > 0) {
            for _ in 0..FRAMES {
                let mut frame_bufs: Vec<PooledBuffer> = Vec::with_capacity(n);
                let mut frame_ptrs: Vec<*mut u8> = Vec::with_capacity(n);
                for count in &target_counts {
                    let bytes = ((*count).max(1) as u64) * std::mem::size_of::<f32>() as u64;
                    let buf = self
                        .hw
                        .alloc
                        .alloc_buffer(
                            bytes,
                            D3D12_HEAP_TYPE_UPLOAD,
                            D3D12_RESOURCE_STATE_GENERIC_READ,
                        )
                        .map_err(|e| e.context("morph weight buf"))?;
                    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
                    // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this
                    // payload, and the source is a separate allocation, so the ranges cannot
                    // overlap.
                    unsafe {
                        buf.Map(0, None, Some(&mut ptr))
                            .map_err(|e| map_hresult(e.code(), "map morph weight buf"))?;
                        std::ptr::write_bytes(ptr as *mut u8, 0, bytes as usize);
                    }
                    frame_bufs.push(buf);
                    frame_ptrs.push(ptr as *mut u8);
                }
                weight_buffers.push(frame_bufs);
                weight_ptrs.push(frame_ptrs);
            }
        }

        self.skinned.morph_delta_buffers = delta_buffers;
        self.skinned.morph_target_counts = target_counts;
        self.skinned.slots.morph_weights = weights;
        self.skinned.morph_weight_buffers = weight_buffers;
        self.skinned.morph_weight_ptrs = weight_ptrs;
        Ok(())
    }

    pub(in crate::directx) fn update_morph_weights(
        &mut self,
        skinned_index: SkinnedIndex,
        weights: &[f32],
    ) {
        self.skinned
            .slots
            .update_morph_weights(skinned_index, weights);
    }

    // Copy this frame's morph weights into the per-frame weight buffers. Called
    // from `record_frame` alongside `upload_joint_matrices`. A no-op when no
    // object carries morphs (the buffers are empty).
    pub(in crate::directx) fn upload_morph_weights(&self, frame_idx: usize) {
        let Some(frame_ptrs) = self.skinned.morph_weight_ptrs.get(frame_idx) else {
            return;
        };
        for (i, w) in self.skinned.slots.morph_weights.iter().enumerate() {
            let (Some(&dst), false) = (frame_ptrs.get(i), w.is_empty()) else {
                continue;
            };
            // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and
            // the source is a separate allocation, so the ranges cannot overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    w.as_ptr() as *const u8,
                    dst,
                    w.len() * std::mem::size_of::<f32>(),
                );
            }
        }
    }

    // GPU virtual address of skinned object `i`'s morph weight buffer for
    // `frame_idx`, or `None` when no weight buffers are allocated.
    pub(in crate::directx) fn morph_weight_gva(&self, frame_idx: usize, i: usize) -> Option<u64> {
        let buf = self.skinned.morph_weight_buffers.get(frame_idx)?.get(i)?;
        Some(com::gpu_va(buf))
    }
}
