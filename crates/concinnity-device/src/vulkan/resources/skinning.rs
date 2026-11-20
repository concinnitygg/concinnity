//! Skinned-mesh upload, per-frame joint upload, and helpers for `VkContext`.
//! Builds the per-(frame, object) joint storage buffers once at init;
//! per-frame `update_skinned_pose` + `upload_joint_matrices` keep the matrices
//! fresh from the gameplay-side pose update.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::mesh_payload;
use concinnity_engine::gfx::mesh_payload::SkinnedVertex;
use concinnity_engine::gfx::render_types::{MAX_JOINTS, SkinnedDrawObject, SkinnedIndex};
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::backend::{RenderBackend, SkinnedDraws};
use concinnity_engine::render::{rt_accel::geom, slots::geometry_repack};
use concinnity_engine::transform::IDENTITY;

use super::super::context::VkContext;
use super::super::set_writes::SetWrites;

impl VkContext {
    // Copy this frame's skinning matrices into the per-frame joint buffers.
    pub(in crate::vulkan) fn upload_joint_matrices(&self, frame_idx: usize) {
        let Some(frame_bufs) = self.skinned.joint_buffers.get(frame_idx) else {
            return;
        };
        for (i, mats) in self.state.skinned.joint_matrices.iter().enumerate() {
            let Some(dst) = frame_bufs.get(i) else {
                continue;
            };
            let count = mats.len().min(MAX_JOINTS);
            dst.write_slice(0, &mats[..count]);
        }
    }

    // Copy this frame's morph weights into the per-frame weight buffers the skin
    // fold reads. Called alongside `upload_joint_matrices`. A no-op when no
    // object carries morphs (the buffers are empty).
    pub(in crate::vulkan) fn upload_morph_weights(&self, frame_idx: usize) {
        let Some(frame_bufs) = self.skinned.morph_weight_buffers.get(frame_idx) else {
            return;
        };
        for (i, w) in self.state.skinned.morph_weights.iter().enumerate() {
            let (Some(dst), false) = (frame_bufs.get(i), w.is_empty()) else {
                continue;
            };
            dst.write_slice(0, w);
        }
    }
}

impl SkinnedDraws for VkContext {
    // Upload skinned-mesh geometry and the per-(frame, object) joint buffers.
    fn upload_skinned(
        &mut self,
        vertices: &[SkinnedVertex],
        indices: &[u32],
        draw_objects: Vec<SkinnedDrawObject>,
    ) -> RenderResult<()> {
        if draw_objects.is_empty() || vertices.is_empty() || indices.is_empty() {
            return Ok(());
        }
        self.wait_idle();
        let frames = self.frames_in_flight.max(1);
        let n = draw_objects.len();

        let vtx_bytes = bytemuck::cast_slice(vertices);
        let idx_bytes = bytemuck::cast_slice(indices);
        // The skin kernel reads the bind-pose VB as storage, so STORAGE_BUFFER is
        // unconditional. The IB's BLAS-input and SSBO flags are RT-only, added whenever
        // the device can trace so a later toggle finds it usable.
        let skinned_ib_rt = if self.hw.rt_capable {
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
        } else {
            vk::BufferUsageFlags::empty()
        };
        let skinned_vbuf = self.hw.alloc.create_buffer(
            vtx_bytes.len() as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        // Never zero-length: the ray-traced hit path binds this as a storage
        // buffer of index words and its descriptor takes the whole size.
        let skinned_ibuf = self.hw.alloc.create_buffer(
            geom::skinned_index_buffer_bytes(indices.len()) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | skinned_ib_rt,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        self.write_geometry_region(skinned_vbuf.buffer(), 0, vtx_bytes)?;
        self.write_geometry_region(skinned_ibuf.buffer(), 0, idx_bytes)?;

        // Per-(frame, object) joint storage buffers seeded with identity
        // matrices so any not-yet-overwritten slot reads as identity.
        let joint_buf_bytes = (MAX_JOINTS * size_of::<[[f32; 4]; 4]>()) as u64;
        let identity_seed: Vec<[[f32; 4]; 4]> = vec![IDENTITY; MAX_JOINTS];
        let mut joint_buffers: Vec<Vec<super::super::allocator::PooledBuffer>> =
            Vec::with_capacity(frames);
        for _ in 0..frames {
            let mut bufs: Vec<super::super::allocator::PooledBuffer> = Vec::with_capacity(n);
            for _ in 0..n {
                let buf = self.hw.alloc.create_buffer(
                    joint_buf_bytes,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )?;
                buf.write_slice(0, &identity_seed);
                bufs.push(buf);
            }
            joint_buffers.push(bufs);
        }

        self.state.skinned.joint_matrices = draw_objects
            .iter()
            .map(|o| vec![IDENTITY; o.joint_count.max(1)])
            .collect();

        self.skinned.vertex_buffer = skinned_vbuf;
        self.skinned.vertex_buffer_bytes = vtx_bytes.len() as u64;
        self.skinned.index_buffer = skinned_ibuf;
        self.skinned.index_buffer_bytes = idx_bytes.len() as u64;
        self.skinned.joint_buffers = joint_buffers;
        self.state.skinned.draw_objects = draw_objects;
        // A whole new skinned set: nothing in the model-history ring was written
        // for these records.
        let n_cull = self.state.draw.record_count();
        self.state.model_history.get_mut().reset(n_cull);

        // Morph targets are attached by a later `upload_skinned_morphs`; until
        // then every object is morphless (a re-upload resets here).
        self.skinned.morph_delta_unique = Vec::new();
        self.skinned.morph_delta_buffers = vec![vk::Buffer::null(); n];
        self.skinned.morph_target_counts = vec![0; n];
        self.state.skinned.morph_weights = vec![Vec::new(); n];
        self.skinned.morph_weight_buffers = Vec::new();

        // GPU-driven skinning, which every skinned draw rides, so a failure is a
        // startup error.
        self.build_main_skin(vertices.len())
            .map_err(|e| e.context("skinned: main-pass skin fold build failed"))?;
        Ok(())
    }

    // Attach morph-target buffers to the skinned draws, one device buffer per
    // unique set, plus per-frame weight buffers, and re-point the skin sets' morph
    // bindings. Called once after `upload_skinned`.
    fn upload_skinned_morphs(
        &mut self,
        morphs: &[Option<std::sync::Arc<mesh_payload::PayloadMorphs>>],
    ) -> RenderResult<()> {
        use std::collections::HashMap;

        let n = self.state.skinned.draw_objects.len();
        let frames = self.frames_in_flight.max(1);

        let mut delta_unique: Vec<super::super::allocator::PooledBuffer> = Vec::new();
        let mut delta_buffers: Vec<vk::Buffer> = vec![vk::Buffer::null(); n];
        let mut target_counts: Vec<u32> = vec![0; n];
        let mut weights: Vec<Vec<f32>> = vec![Vec::new(); n];
        let mut by_source: HashMap<usize, (vk::Buffer, u32)> = HashMap::new();

        for (i, m) in morphs.iter().take(n).enumerate() {
            let Some(data) = m else { continue };
            let key = std::sync::Arc::as_ptr(data) as usize;
            let (buf, count) = if let Some(e) = by_source.get(&key) {
                *e
            } else {
                let words = data.packed_words();
                let bytes: &[u8] = bytemuck::cast_slice(&words);
                let pooled = self.hw.alloc.create_buffer(
                    bytes.len().max(4) as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )?;
                let buf = pooled.buffer();
                self.write_geometry_region(buf, 0, bytes)?;
                let count = u32::saturating_from(data.target_count());
                delta_unique.push(pooled);
                by_source.insert(key, (buf, count));
                (buf, count)
            };
            delta_buffers[i] = buf;
            target_counts[i] = count;
            weights[i] = vec![0.0; count as usize];
        }

        // Per-(frame, object) host-mapped weight buffers, one f32 per target
        // (>= 1 so every binding has a valid buffer), zero-seeded. Only allocated
        // when some object carries morphs.
        let mut weight_buffers: Vec<Vec<super::super::allocator::PooledBuffer>> = Vec::new();
        if target_counts.iter().any(|&c| c > 0) {
            for _ in 0..frames {
                let mut bufs = Vec::with_capacity(n);
                for &count in &target_counts {
                    let size = u64::from(count.max(1)) * size_of::<f32>() as u64;
                    let buf = self.hw.alloc.create_buffer(
                        size,
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )?;
                    buf.zero_bytes(0, usize::saturating_from(size));
                    bufs.push(buf);
                }
                weight_buffers.push(bufs);
            }
        }

        // Re-point the main fold's skin descriptor sets' morph bindings (3 =
        // deltas, 4 = weights). Morphless objects keep the dummy SSBO the
        // `build_main_skin` write left. A no-op when the fold is inactive.
        if let Some(skin) = self.skinned.skin.as_ref() {
            let dummy = skin.morph_dummy;
            for (f, frame_sets) in skin.sets.iter().enumerate() {
                for (o, &set) in frame_sets.iter().enumerate() {
                    let delta_buf = match delta_buffers.get(o) {
                        Some(&b) if b != vk::Buffer::null() => b,
                        _ => dummy,
                    };
                    let weight_buf = match weight_buffers.get(f).and_then(|fb| fb.get(o)) {
                        Some(b) => b.buffer(),
                        None => dummy,
                    };
                    SetWrites::new(set)
                        .storage_buffer(3, delta_buf, vk::WHOLE_SIZE)
                        .storage_buffer(4, weight_buf, vk::WHOLE_SIZE)
                        .apply(&self.hw.device);
                }
            }
        }

        self.skinned.morph_delta_unique = delta_unique;
        self.skinned.morph_delta_buffers = delta_buffers;
        self.skinned.morph_target_counts = target_counts;
        self.state.skinned.morph_weights = weights;
        self.skinned.morph_weight_buffers = weight_buffers;
        Ok(())
    }

    // Replace a `SkinnedMesh` slot's vertices and indices in place on hot reload.
    // The payload must fit the slot's region (resizes go through
    // `rebuild_skinned_geometry`); indices are rebased by `vertex_base`.
    fn update_skinned_mesh_geometry(
        &mut self,
        skinned_index: SkinnedIndex,
        vertex_base: u32,
        vertices: &[SkinnedVertex],
        indices: &[u16],
    ) -> RenderResult<()> {
        if self.skinned.vertex_buffer.is_null() || self.skinned.index_buffer.is_null() {
            return Err(RenderError::Other(
                "update_skinned_mesh_geometry: no skinned vertex/index buffer (was \
                 upload_skinned called?)"
                    .to_string(),
            ));
        }
        let write = geometry_repack::place_skinned_update(
            &self.state.skinned.draw_objects,
            skinned_index,
            vertex_base,
            vertices.len(),
            indices,
            usize::saturating_from(self.skinned.vertex_buffer_bytes),
        )?;

        self.wait_idle();

        self.write_geometry_region(
            self.skinned.vertex_buffer.buffer(),
            write.vertex_offset,
            bytemuck::cast_slice(vertices),
        )?;
        self.write_geometry_region(
            self.skinned.index_buffer.buffer(),
            write.index_offset,
            bytemuck::cast_slice(&write.indices),
        )?;
        Ok(())
    }
}
