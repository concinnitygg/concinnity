//! Hot-reload rebuild of the shared static + skinned vertex / index buffers
//! when a re-imported `.glb` no longer fits in its init-time slot. Mirrors
//! `directx/resources/geometry_rebuild.rs` and `metal/resources/geometry_rebuild.rs +
//! metal/resources/skinning.rs`'s rebuild paths.
//!
//! Vulkan's DEVICE_LOCAL buffers are not host-visible, so the rebuild forces
//! a CPU round-trip: a one-shot `cmd_copy_buffer` reads the live VB/IB into
//! HOST_VISIBLE staging, the new contents are spliced CPU-side, fresh
//! DEVICE_LOCAL buffers are allocated at the post-rebuild size, and the
//! rebuilt data is uploaded via the existing `write_geometry_region` helper.
//! Streamed-mesh sub-allocators (`mesh_vtx_alloc`, `mesh_idx_alloc`) are
//! **not** preserved: the rebuilt buffer is sized exactly for the current
//! draws, so any subsequent `upload_mesh` will fail allocation. `cn debug`-
//! only by design; matches DirectX + Metal.

use ash::vk;
use concinnity_core::gfx::mesh_payload::{SkinnedVertex, Vertex};
use concinnity_core::render::backend::{
    DrawGeometryUpdate, SkinnedDrawGeometryUpdate, SkinnedSlotLayout,
};
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::geometry_repack;
use concinnity_core::render::rt_geom;
use std::collections::HashMap;

use super::super::context::VkContext;
use super::super::texture::one_shot_submit;

impl VkContext {
    // Swap in rebuilt shared static-mesh buffers for the draws named in
    // `changes`, when a reloaded mesh no longer fits its slot. The live buffers
    // are read back for the repack, the result is uploaded into fresh
    // DEVICE_LOCAL buffers, and the streamed-mesh sub-allocators reset, since
    // the rebuilt buffers leave no headroom.
    pub(crate) fn rebuild_static_geometry(
        &mut self,
        changes: Vec<DrawGeometryUpdate>,
    ) -> RenderResult<()> {
        // Stop GPU + CPU pipelines so the readback + swap can run safely.
        self.wait_idle();

        // Read the live VB / IB back to CPU memory through a HOST_VISIBLE
        // staging buffer (DEVICE_LOCAL is not host-mappable).
        let old_v_bytes = self.geometry.vertex_buffer_bytes;
        let old_i_bytes = self.geometry.index_buffer_bytes;
        let old_vertices: Vec<Vertex> =
            readback_typed(self, self.geometry.vertex_buffer.buffer(), old_v_bytes)?;
        let old_indices: Vec<u32> =
            readback_typed(self, self.geometry.index_buffer.buffer(), old_i_bytes)?;

        let repacked = geometry_repack::repack_static_geometry(
            &self.draw.objects,
            &old_vertices,
            &old_indices,
            changes,
        )?;
        if repacked.ignored_changes > 0 {
            tracing::warn!(
                "rebuild_static_geometry: {} change(s) targeted draw indices not in \
                 draw_objects (ignored)",
                repacked.ignored_changes
            );
        }
        let new_vertices = repacked.vertices;
        let new_indices = repacked.indices;

        // Allocate new DEVICE_LOCAL buffers + ship the rebuilt contents
        // through staging (write_geometry_region's one-shot pattern).
        let new_vertex_count = new_vertices.len();
        let new_v_bytes = std::mem::size_of_val(new_vertices.as_slice()) as u64;
        let new_i_bytes = std::mem::size_of_val(new_indices.as_slice()) as u64;
        let shared = super::shared_geometry_usage(self.hw.rt_capable);
        let new_vbuf = self.hw.alloc.create_buffer(
            new_v_bytes,
            vk::BufferUsageFlags::VERTEX_BUFFER | shared,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let new_ibuf = self.hw.alloc.create_buffer(
            new_i_bytes,
            vk::BufferUsageFlags::INDEX_BUFFER | shared,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let vert_bytes = bytemuck::cast_slice(&new_vertices);
        let idx_bytes = bytemuck::cast_slice(&new_indices);
        self.write_geometry_region(new_vbuf.buffer(), 0, vert_bytes)?;
        self.write_geometry_region(new_ibuf.buffer(), 0, idx_bytes)?;

        // Commit the swap: the replaced buffers retire through the allocator.
        // `wait_idle` above gated every in-flight read.
        self.geometry.vertex_buffer = new_vbuf;
        self.geometry.vertex_buffer_bytes = new_v_bytes;
        self.geometry.index_buffer = new_ibuf;
        self.geometry.index_buffer_bytes = new_i_bytes;
        self.geometry.mesh_vtx_alloc = crate::suballoc::range_alloc::RangeAllocator::new();
        self.geometry.mesh_idx_alloc = crate::suballoc::range_alloc::RangeAllocator::new();
        for (layout, obj) in repacked.layouts.into_iter().zip(&mut self.draw.objects) {
            layout.apply_to(obj);
        }

        // The RT acceleration structure was built against the buffers just
        // destroyed and the per-draw offsets just rewritten. Rebuild it over the
        // fresh layout and re-point the passes that read the buffers directly.
        // The static vertex count (which bounds each BLAS's vertex range) moved
        // with the rebuild, so it is refreshed first.
        self.rt.static_vertex_count = new_vertex_count;
        if self.rt.accel.is_some() {
            self.rebuild_rt_accel()?;
        }
        Ok(())
    }

    // Rebuild the shared skinned-mesh vertex + index buffers, swapping in
    // fresh geometry for the slots named in `changes`. Driven by asset
    // hot-reload (`cn debug` only) when a `SkinnedMesh` re-import has a
    // different vertex / index count than its init-time slot. Walks every
    // `SkinnedDrawObject` in order: for each slot in `changes`, the new
    // vertices / indices are appended to a fresh CPU buffer; for unchanged
    // slots, the current geometry is read back from the live skinned
    // buffers and copied with index rebasing from the slot's old
    // `vertex_base` onto its new one. New DEVICE_LOCAL buffers are created
    // at the post-rebuild size, the contents uploaded through staging, and
    // the old buffers / memory dropped after the swap commits. Returns a
    // `SkinnedSlotLayout` per slot (in `skinned_index` order) so the
    // asset-hot-reload caller can refresh its `SkinnedMeshSourceEntry`s.
    //
    // The skinned IB stays `R16_UINT`; the skinned pipelines, shadow /
    // SSAO / SSR variants, and per-slot metadata (`texture_slot` /
    // `normal_map_slot` / `material` / `joint_count`) are untouched.
    // Skeleton-shape changes route through `update_skinned_skeleton`, not
    // this call. Mirrors `DxContext::rebuild_skinned_geometry`. Reached only
    // through the bin's `cn debug` runtime-mutation path (dead in the FFI lib,
    // live in the bin).
    pub(crate) fn rebuild_skinned_geometry(
        &mut self,
        changes: Vec<SkinnedDrawGeometryUpdate>,
    ) -> RenderResult<Vec<SkinnedSlotLayout>> {
        if self.skinned.vertex_buffer.is_null() || self.skinned.index_buffer.is_null() {
            return Err(RenderError::Other(
                "rebuild_skinned_geometry: no skinned vertex/index buffer (was \
                 upload_skinned called?)"
                    .into(),
            ));
        }

        self.wait_idle();

        let mut change_map: HashMap<usize, SkinnedDrawGeometryUpdate> =
            changes.into_iter().map(|c| (c.skinned_index, c)).collect();

        // Read back the live skinned buffers via HOST_VISIBLE staging.
        let old_v_bytes = self.skinned.vertex_buffer_bytes;
        let old_i_bytes = self.skinned.index_buffer_bytes;
        let old_vertices: Vec<SkinnedVertex> =
            readback_typed(self, self.skinned.vertex_buffer.buffer(), old_v_bytes)?;
        let old_indices: Vec<u32> =
            readback_typed(self, self.skinned.index_buffer.buffer(), old_i_bytes)?;

        let mut new_vertices: Vec<SkinnedVertex> = Vec::new();
        let mut new_indices: Vec<u32> = Vec::new();
        let mut layouts: Vec<SkinnedSlotLayout> =
            Vec::with_capacity(self.skinned.slots.draw_objects.len());
        // Captured per-slot new layout (applied to `skinned_draw_objects`
        // after the read-only walk to avoid aliasing `self`).
        let mut new_per_slot: Vec<(usize, u32, usize, usize, usize)> =
            Vec::with_capacity(self.skinned.slots.draw_objects.len());

        for (skinned_index, obj) in self.skinned.slots.draw_objects.iter().enumerate() {
            let new_v_base = new_vertices.len() as u32;
            let new_i_off = new_indices.len();

            if let Some(change) = change_map.remove(&skinned_index) {
                let new_v_count = change.vertices.len();
                let new_i_count = change.indices.len();
                new_vertices.extend_from_slice(&change.vertices);
                for &local in &change.indices {
                    new_indices.push(u32::from(local) + new_v_base);
                }
                layouts.push(SkinnedSlotLayout {
                    skinned_index,
                    vertex_base: new_v_base,
                    vertex_count: new_v_count,
                    index_count: new_i_count,
                });
                new_per_slot.push((
                    skinned_index,
                    new_v_base,
                    new_v_count,
                    new_i_off,
                    new_i_count,
                ));
            } else {
                // Unchanged slot: copy current geometry verbatim, rebasing
                // its absolute indices from old vertex_base onto the new
                // one.
                let v_start = obj.vertex_base as usize;
                let v_end = v_start + obj.vertex_count;
                if v_end > old_vertices.len() {
                    return Err(RenderError::Other(format!(
                        "rebuild_skinned_geometry: slot {} vertex region [{}, {}) \
                         out of bounds (buffer has {} vertices)",
                        skinned_index,
                        v_start,
                        v_end,
                        old_vertices.len()
                    )));
                }
                new_vertices.extend_from_slice(&old_vertices[v_start..v_end]);
                let i_end = obj.index_offset + obj.index_count;
                if i_end > old_indices.len() {
                    return Err(RenderError::Other(format!(
                        "rebuild_skinned_geometry: slot {} index region [{}, {}) \
                         out of bounds (buffer has {} indices)",
                        skinned_index,
                        obj.index_offset,
                        i_end,
                        old_indices.len()
                    )));
                }
                let old_base = obj.vertex_base;
                for &abs in &old_indices[obj.index_offset..i_end] {
                    let local = abs.checked_sub(old_base).ok_or_else(|| {
                        RenderError::Other(format!(
                            "rebuild_skinned_geometry: stale index {abs} below \
                             vertex_base {old_base} on slot {skinned_index}"
                        ))
                    })?;
                    new_indices.push(local + new_v_base);
                }
                layouts.push(SkinnedSlotLayout {
                    skinned_index,
                    vertex_base: new_v_base,
                    vertex_count: obj.vertex_count,
                    index_count: obj.index_count,
                });
                new_per_slot.push((
                    skinned_index,
                    new_v_base,
                    obj.vertex_count,
                    new_i_off,
                    obj.index_count,
                ));
            }
        }

        if !change_map.is_empty() {
            tracing::warn!(
                "rebuild_skinned_geometry: {} change(s) targeted skinned indices not \
                 in skinned_draw_objects (ignored)",
                change_map.len()
            );
        }

        if new_vertices.is_empty() || new_indices.is_empty() {
            return Err(RenderError::Other(
                "rebuild_skinned_geometry: post-rebuild buffers would be empty (no \
                 skinned draws to ship)"
                    .into(),
            ));
        }

        // Allocate new DEVICE_LOCAL skinned buffers + ship through staging. The
        // new buffers must carry the same usage flags `upload_skinned` added, or
        // the skinning paths lose their inputs after a size-changing skinned
        // reload (the RT-only IB flags ride along whenever the device is capable
        // so a live RT toggle keeps working across reloads).
        let new_v_bytes = std::mem::size_of_val(new_vertices.as_slice()) as u64;
        let new_i_bytes = std::mem::size_of_val(new_indices.as_slice()) as u64;
        let skinned_ib_rt = if self.hw.rt_capable {
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
        } else {
            vk::BufferUsageFlags::empty()
        };
        let new_vbuf = self.hw.alloc.create_buffer(
            new_v_bytes,
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        // Whole u32 words for the index buffer; see `upload_skinned`.
        let new_ibuf = self.hw.alloc.create_buffer(
            rt_geom::skinned_index_buffer_bytes(new_indices.len()) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | skinned_ib_rt,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let vert_bytes = bytemuck::cast_slice(&new_vertices);
        let idx_bytes = bytemuck::cast_slice(&new_indices);
        self.write_geometry_region(new_vbuf.buffer(), 0, vert_bytes)?;
        self.write_geometry_region(new_ibuf.buffer(), 0, idx_bytes)?;

        // Commit: the replaced buffers retire through the allocator.
        self.skinned.vertex_buffer = new_vbuf;
        self.skinned.vertex_buffer_bytes = new_v_bytes;
        self.skinned.index_buffer = new_ibuf;
        self.skinned.index_buffer_bytes = new_i_bytes;
        for (skinned_index, v_base, v_count, i_off, i_count) in new_per_slot {
            let obj = &mut self.skinned.slots.draw_objects[skinned_index];
            obj.vertex_base = v_base;
            obj.vertex_count = v_count;
            obj.index_offset = i_off;
            obj.index_count = i_count;
        }

        // The skin fold's descriptor sets still bind the replaced bind-pose VB
        // and a deformed ring sized for the old layout; re-point and re-size
        // both. `wait_idle` above gated every in-flight read.
        self.refresh_main_skin_geometry(new_vertices.len())?;
        Ok(layouts)
    }
}

// Read a DEVICE_LOCAL buffer's full contents back to CPU memory as a typed
// `Vec<T>`. Allocates a HOST_VISIBLE staging buffer, runs a one-shot
// `cmd_copy_buffer` from `src` into it (`wait_idle` already gated the source
// side; the one-shot's internal fence wait gates the destination), maps,
// and `copy_nonoverlapping`s into the Vec. `T`'s stride must match the
// buffer's stride exactly. Only reached through the (bin-only) geometry-rebuild
// path, so dead in the FFI lib.
fn readback_typed<T: Copy>(ctx: &VkContext, src: vk::Buffer, bytes: u64) -> RenderResult<Vec<T>> {
    if bytes == 0 {
        return Ok(Vec::new());
    }
    let stride = std::mem::size_of::<T>() as u64;
    if !bytes.is_multiple_of(stride) {
        return Err(RenderError::Other(format!(
            "readback_typed: buffer size {} not a multiple of T stride {}",
            bytes, stride
        )));
    }
    let count = (bytes / stride) as usize;
    let staging = ctx.hw.alloc.create_buffer(
        bytes,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    one_shot_submit(
        &ctx.hw.device,
        ctx.commands.command_pool,
        ctx.hw.graphics_queue,
        |cmd| {
            let copy = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(bytes);
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                ctx.hw.device.cmd_copy_buffer(
                    cmd,
                    src,
                    staging.buffer(),
                    std::slice::from_ref(&copy),
                )
            };
        },
    )?;

    let mut out: Vec<T> = Vec::with_capacity(count);
    // SAFETY: the staging buffer was created HOST_VISIBLE | HOST_COHERENT and sized to `size`,
    // which is at least the source length, so `mapped_ptr()` is a live mapping of that many bytes;
    // the source is a separate live allocation, so the ranges cannot overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(staging.mapped_ptr() as *const T, out.as_mut_ptr(), count);
        out.set_len(count);
    }
    Ok(out)
}
