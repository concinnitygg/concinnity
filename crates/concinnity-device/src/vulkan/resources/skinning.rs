// src/vulkan/resources/skinning.rs
//
// Skinned-mesh upload, per-frame joint upload, and helpers for VkContext.
// Builds the skinned pipelines + per-(frame, object) joint storage buffers
// once at init; per-frame `update_skinned_pose` + `upload_joint_matrices`
// keep the matrices fresh from the gameplay-side pose update.

use ash::vk;
use concinnity_core::gfx::mesh_payload;
use concinnity_core::gfx::mesh_payload::SkinnedVertex;
use concinnity_core::gfx::render_types::*;
use concinnity_core::gfx::transform::IDENTITY;
use concinnity_core::render::rt_geom;

use super::super::context::*;
use super::super::pipeline::{compile_skinned_shadow_shader, create_skinned_shadow_pipeline};
use super::{alloc_descriptor_sets, create_descriptor_set_layout};

impl VkContext {
    // Upload skinned-mesh geometry and build the skinned render pipelines.
    pub(crate) fn upload_skinned(
        &mut self,
        vertices: &[SkinnedVertex],
        indices: &[u32],
        draw_objects: Vec<SkinnedDrawObject>,
    ) -> Result<(), String> {
        if draw_objects.is_empty() || vertices.is_empty() || indices.is_empty() {
            return Ok(());
        }
        self.wait_idle();
        let frames = self.frames_in_flight.max(1);
        let n = draw_objects.len();

        let skinned_shadow_vs = compile_skinned_shadow_shader(self.hot_reload.enabled)?;

        let joint_set_layout = create_descriptor_set_layout(
            &self.device,
            &[(
                0,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::VERTEX,
            )],
        )?;

        let (skinned_shadow_pipeline, skinned_shadow_pipeline_layout) =
            if let (Some(_), Some(shadow_global)) = (
                self.shadow.pipeline.as_ref(),
                self.shadow.global_set_layout.as_ref(),
            ) {
                let shadow_pc = vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::VERTEX)
                    .offset(0)
                    .size(80);
                let shadow_set_layouts = [shadow_global.handle(), joint_set_layout.handle()];
                let layout = self
                    .device
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default()
                            .set_layouts(&shadow_set_layouts)
                            .push_constant_ranges(std::slice::from_ref(&shadow_pc)),
                    )
                    .map_err(|e| format!("skinned shadow pipeline layout: {e}"))?;
                let pipeline = create_skinned_shadow_pipeline(
                    &self.device,
                    self.shadow.render_pass.handle(),
                    layout.handle(),
                    &skinned_shadow_vs,
                )?;
                (Some(pipeline), Some(layout))
            } else {
                (None, None)
            };

        let vtx_bytes = bytemuck::cast_slice(vertices);
        let idx_bytes = bytemuck::cast_slice(indices);
        // The skin compute kernel reads the bind-pose VB as a storage buffer, so
        // STORAGE_BUFFER is unconditional: the main-pass skinning fold runs
        // whether or not the device is RT-capable.
        //
        // The IB's extra flags are genuinely RT-only: it is the skinned BLAS
        // index input (device-addressed) and the hit shader's index SSBO,
        // and nothing outside the RT path binds it as a buffer. Added whenever
        // the device is RT-capable (not only when RT is on at launch) so a later
        // live toggle finds the skinned IB already usable, mirroring how the
        // static VB/IB gate their RT flags at init. Inert when RT is never built.
        let skinned_ib_rt = if self.rt_capable {
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
        } else {
            vk::BufferUsageFlags::empty()
        };
        let skinned_vbuf = self.alloc.create_buffer(
            vtx_bytes.len() as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        // Never zero-length: the ray-traced hit path binds this as a storage
        // buffer of index words and its descriptor takes the whole size.
        let skinned_ibuf = self.alloc.create_buffer(
            rt_geom::skinned_index_buffer_bytes(indices.len()) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | skinned_ib_rt,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        self.write_geometry_region(skinned_vbuf.buffer(), 0, vtx_bytes)?;
        self.write_geometry_region(skinned_ibuf.buffer(), 0, idx_bytes)?;

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count((n * frames) as u32)];
        let pool = self
            .device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets((n * frames) as u32)
                    .pool_sizes(&pool_sizes),
            )
            .map_err(|e| format!("skinned descriptor pool: {e}"))?;

        // Per-(frame, object) joint storage buffers seeded with identity
        // matrices so any not-yet-overwritten slot reads as identity.
        let joint_buf_bytes = (MAX_JOINTS * std::mem::size_of::<[[f32; 4]; 4]>()) as u64;
        let identity_seed: Vec<[[f32; 4]; 4]> = vec![IDENTITY; MAX_JOINTS];
        let mut joint_buffers: Vec<Vec<super::super::allocator::PooledBuffer>> =
            Vec::with_capacity(frames);
        let mut joint_sets: Vec<Vec<vk::DescriptorSet>> = Vec::with_capacity(frames);
        for _ in 0..frames {
            let mut bufs: Vec<super::super::allocator::PooledBuffer> = Vec::with_capacity(n);
            for _ in 0..n {
                let buf = self.alloc.create_buffer(
                    joint_buf_bytes,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )?;
                buf.write_slice(0, &identity_seed);
                bufs.push(buf);
            }
            let layouts: Vec<_> = (0..n).map(|_| joint_set_layout.handle()).collect();
            let sets = alloc_descriptor_sets(&self.device, pool.handle(), &layouts)?;
            for (i, &set) in sets.iter().enumerate() {
                let info = vk::DescriptorBufferInfo::default()
                    .buffer(bufs[i].buffer())
                    .offset(0)
                    .range(vk::WHOLE_SIZE);
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&info));
                // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
                // every set and resource it names belongs to this device.
                unsafe {
                    self.device
                        .update_descriptor_sets(std::slice::from_ref(&write), &[])
                };
            }
            joint_buffers.push(bufs);
            joint_sets.push(sets);
        }

        self.skinned.slots.joint_matrices = draw_objects
            .iter()
            .map(|o| vec![IDENTITY; o.joint_count.max(1)])
            .collect();

        self.shadow.skinned_pipeline = skinned_shadow_pipeline;
        self.shadow.skinned_pipeline_layout = skinned_shadow_pipeline_layout;
        self.skinned.joint_set_layout = Some(joint_set_layout);
        self.skinned.descriptor_pool = Some(pool);
        self.skinned.vertex_buffer = skinned_vbuf;
        self.skinned.vertex_buffer_bytes = vtx_bytes.len() as u64;
        self.skinned.index_buffer = skinned_ibuf;
        self.skinned.index_buffer_bytes = idx_bytes.len() as u64;
        self.skinned.joint_buffers = joint_buffers;
        self.skinned.joint_sets = joint_sets;
        self.skinned.slots.draw_objects = draw_objects;
        // A whole new skinned set: nothing in the model-history ring was written
        // for these records.
        self.model_history.borrow_mut().reset(self.cull_count());

        // Morph targets are attached by a later `upload_skinned_morphs`; until
        // then every object is morphless (a re-upload resets here).
        self.skinned.morph_delta_unique = Vec::new();
        self.skinned.morph_delta_buffers = vec![vk::Buffer::null(); n];
        self.skinned.morph_target_counts = vec![0; n];
        self.skinned.slots.morph_weights = vec![Vec::new(); n];
        self.skinned.morph_weight_buffers = Vec::new();

        // GPU-driven main-pass skinning: build the `rt_skin` compute pipeline +
        // per-frame deformed-vertex buffers + their descriptor sets, and set
        // `self.draw.n_skinned` so `cull_count()` covers the skinned tail. Every
        // skinned draw rides the GPU-driven pass, so a build failure is a
        // startup error, as on Metal. Mirrors the DirectX `upload_skinned`.
        self.build_main_skin(vertices.len())
            .map_err(|e| format!("skinned: main-pass skin fold build failed: {e}"))?;
        Ok(())
    }

    // Replace a `SkinnedMesh` draw slot's vertex + index data in place.
    // Driven by asset hot-reload (`cn debug` only). The shared skinned VB
    // / IB were sized once at `upload_skinned` to hold every skinned
    // mesh's geometry, so the new payload must fit within this slot's
    // existing region (size-changing reloads route through
    // `rebuild_skinned_geometry`). `vertex_base` is the slot's vertex
    // offset *in vertices*; `indices` are mesh-relative and get rebased
    // by `vertex_base` before being written into the shared IB.
    // Mirrors `DxContext::update_skinned_mesh_geometry`. Reached only through
    // the bin's `cn debug` runtime-mutation path (dead in the FFI lib, live in
    // the bin).
    pub(crate) fn update_skinned_mesh_geometry(
        &mut self,
        skinned_index: usize,
        vertex_base: u32,
        vertices: &[SkinnedVertex],
        indices: &[u16],
    ) -> Result<(), String> {
        let obj = self
            .skinned
            .slots
            .draw_objects
            .get(skinned_index)
            .ok_or_else(|| {
                format!(
                    "update_skinned_mesh_geometry: skinned object {} out of range",
                    skinned_index
                )
            })?;
        if indices.len() != obj.index_count {
            return Err(format!(
                "update_skinned_mesh_geometry: skinned {} expects {} indices, got {} \
                 (in-place path is size-matched only; size changes route through \
                 rebuild_skinned_geometry)",
                skinned_index,
                obj.index_count,
                indices.len()
            ));
        }
        if self.skinned.vertex_buffer.is_null() || self.skinned.index_buffer.is_null() {
            return Err(
                "update_skinned_mesh_geometry: no skinned vertex/index buffer (was \
                 upload_skinned called?)"
                    .to_string(),
            );
        }
        let v_byte_off =
            (vertex_base as usize).saturating_mul(std::mem::size_of::<SkinnedVertex>());
        let v_byte_len = std::mem::size_of_val(vertices);
        let v_buf_len = self.skinned.vertex_buffer_bytes as usize;
        if v_byte_off + v_byte_len > v_buf_len {
            return Err(format!(
                "update_skinned_mesh_geometry: vertex region [{}, {}) overruns skinned \
                 vertex buffer length {}",
                v_byte_off,
                v_byte_off + v_byte_len,
                v_buf_len
            ));
        }
        let i_byte_off = (obj.index_offset * std::mem::size_of::<u32>()) as u64;
        let rebased: Vec<u32> = indices
            .iter()
            .map(|&i| u32::from(i) + vertex_base)
            .collect();

        self.wait_idle();

        let vert_bytes = bytemuck::cast_slice(vertices);
        self.write_geometry_region(
            self.skinned.vertex_buffer.buffer(),
            v_byte_off as u64,
            vert_bytes,
        )?;
        let idx_bytes = bytemuck::cast_slice(&rebased);
        self.write_geometry_region(self.skinned.index_buffer.buffer(), i_byte_off, idx_bytes)?;
        Ok(())
    }

    // The CPU-side skinned entry points the `RenderBackend` impl forwards to.
    // Each is the `SkinnedSlots` operation of the same name; the behavior and
    // its contract are documented there, once for all three backends.
    pub(crate) fn update_skinned_skeleton(
        &mut self,
        skinned_index: usize,
        new_joint_count: usize,
    ) -> Result<(), String> {
        self.skinned
            .slots
            .update_skeleton(skinned_index, new_joint_count)
    }

    pub(crate) fn update_skinned_pose(&mut self, skinned_index: usize, matrices: &[[[f32; 4]; 4]]) {
        self.skinned.slots.update_pose(skinned_index, matrices);
    }

    pub(crate) fn reveal_skinned_instance(&mut self, instance_index: usize, model: [[f32; 4]; 4]) {
        self.skinned
            .slots
            .reveal(instance_index, model, &mut self.model_history.borrow_mut());
    }

    pub(crate) fn retire_skinned_draw_object(&mut self, skinned_index: usize) {
        self.skinned.slots.retire(skinned_index);
    }

    pub(crate) fn update_skinned_models(&mut self, updates: &[(u32, [[f32; 4]; 4])]) {
        self.skinned.slots.update_models(updates);
    }

    // Copy this frame's skinning matrices into the per-frame joint buffers.
    pub(in crate::vulkan) fn upload_joint_matrices(&self, frame_idx: usize) {
        let Some(frame_bufs) = self.skinned.joint_buffers.get(frame_idx) else {
            return;
        };
        for (i, mats) in self.skinned.slots.joint_matrices.iter().enumerate() {
            let Some(dst) = frame_bufs.get(i) else {
                continue;
            };
            let count = mats.len().min(MAX_JOINTS);
            dst.write_slice(0, &mats[..count]);
        }
    }

    // Attach morph-target buffers (`PayloadMorphs::packed_words`) to the skinned
    // draw objects. `morphs[i]` pairs with draw object `i`; instance copies share
    // their template's `Arc`, so each unique entry set becomes one device buffer. Allocates the per-frame
    // weight buffers (one f32 per target per object) and re-points the main fold's
    // skin descriptor-set morph bindings when any object carries morphs. Called
    // once after `upload_skinned`. Mirrors the DirectX `upload_skinned_morphs`.
    pub(in crate::vulkan) fn upload_skinned_morphs(
        &mut self,
        morphs: Vec<Option<std::sync::Arc<mesh_payload::PayloadMorphs>>>,
    ) -> Result<(), String> {
        use std::collections::HashMap;

        let n = self.skinned.slots.draw_objects.len();
        let device = self.device.clone();
        let frames = self.frames_in_flight.max(1);

        let mut delta_unique: Vec<super::super::allocator::PooledBuffer> = Vec::new();
        let mut delta_buffers: Vec<vk::Buffer> = vec![vk::Buffer::null(); n];
        let mut target_counts: Vec<u32> = vec![0; n];
        let mut weights: Vec<Vec<f32>> = vec![Vec::new(); n];
        let mut by_source: HashMap<usize, (vk::Buffer, u32)> = HashMap::new();

        for (i, m) in morphs.iter().take(n).enumerate() {
            let Some(data) = m else { continue };
            let key = std::sync::Arc::as_ptr(data) as usize;
            let (buf, count) = match by_source.get(&key) {
                Some(e) => *e,
                None => {
                    let words = data.packed_words();
                    let bytes: &[u8] = bytemuck::cast_slice(&words);
                    let pooled = self.alloc.create_buffer(
                        bytes.len().max(4) as u64,
                        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    )?;
                    let buf = pooled.buffer();
                    self.write_geometry_region(buf, 0, bytes)?;
                    let count = data.target_count() as u32;
                    delta_unique.push(pooled);
                    by_source.insert(key, (buf, count));
                    (buf, count)
                }
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
                    let size = (count.max(1) as u64) * std::mem::size_of::<f32>() as u64;
                    let buf = self.alloc.create_buffer(
                        size,
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )?;
                    buf.zero_bytes(0, size as usize);
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
                    let delta_info = vk::DescriptorBufferInfo::default()
                        .buffer(delta_buf)
                        .offset(0)
                        .range(vk::WHOLE_SIZE);
                    let weight_info = vk::DescriptorBufferInfo::default()
                        .buffer(weight_buf)
                        .offset(0)
                        .range(vk::WHOLE_SIZE);
                    let writes = [
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(3)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(std::slice::from_ref(&delta_info)),
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(4)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(std::slice::from_ref(&weight_info)),
                    ];
                    // SAFETY: `writes` and the buffer/image infos it borrows are live for the call,
                    // and every set and resource it names belongs to this device.
                    unsafe { device.update_descriptor_sets(&writes, &[]) };
                }
            }
        }

        self.skinned.morph_delta_unique = delta_unique;
        self.skinned.morph_delta_buffers = delta_buffers;
        self.skinned.morph_target_counts = target_counts;
        self.skinned.slots.morph_weights = weights;
        self.skinned.morph_weight_buffers = weight_buffers;
        Ok(())
    }

    pub(crate) fn update_morph_weights(&mut self, skinned_index: usize, weights: &[f32]) {
        self.skinned
            .slots
            .update_morph_weights(skinned_index, weights);
    }

    // Copy this frame's morph weights into the per-frame weight buffers the skin
    // fold reads. Called alongside `upload_joint_matrices`. A no-op when no
    // object carries morphs (the buffers are empty).
    pub(in crate::vulkan) fn upload_morph_weights(&self, frame_idx: usize) {
        let Some(frame_bufs) = self.skinned.morph_weight_buffers.get(frame_idx) else {
            return;
        };
        for (i, w) in self.skinned.slots.morph_weights.iter().enumerate() {
            let (Some(dst), false) = (frame_bufs.get(i), w.is_empty()) else {
                continue;
            };
            dst.write_slice(0, w);
        }
    }

    // Bind the skinned vertex + index buffers for the skinned passes.
    pub(in crate::vulkan) fn skinned_geometry(&self) -> (vk::Buffer, vk::Buffer) {
        (
            self.skinned.vertex_buffer.buffer(),
            self.skinned.index_buffer.buffer(),
        )
    }
}
