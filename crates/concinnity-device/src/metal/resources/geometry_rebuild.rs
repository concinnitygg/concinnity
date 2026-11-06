//! Hot-reload rebuild of the shared static-mesh vertex + index buffers when
//! re-imported `.glb` source no longer fits each draw's init-time slot.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_engine::gfx::mesh_payload::{SkinnedVertex, Vertex};
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::backend::{GeometryRebuild, RenderBackend};
use concinnity_engine::render::{backend, slots::geometry_repack};
use objc2_metal::{MTLBuffer as _, MTLDevice as _, MTLResourceOptions};

use super::skinning::upload_skinned_index_buffer;
use crate::metal::context::{MtlContext, bytes_of_slice};
use crate::metal::error::allocation_failed;

impl GeometryRebuild for MtlContext {
    // Rebuild the shared skinned-mesh vertex + index buffers, swapping in
    // new geometry for the slots named in `changes`. Driven by asset
    // hot-reload (`cn run -f` only) when a SkinnedMesh's re-imported `.glb`
    // no longer fits in its init-time slot. Walks every
    // `SkinnedDrawObject` in order: for each slot in `changes`, the new
    // vertices / indices are appended to a fresh CPU buffer; for unchanged
    // slots, the current geometry is read back from the live
    // `skinned_vertex_buffer` / `skinned_index_buffer` (both
    // `StorageModeShared` so the pointers are CPU-readable) and copied
    // with index rebasing. New `MTLBuffer`s are created at the post-rebuild
    // size and swapped in after `wait_idle` so no in-flight command buffer
    // touches the old resource pair. The skinned pipelines and velocity /
    // SSAO / SSR variants are untouched -- only the per-slot
    // `vertex_base` / `vertex_count` / `index_offset` / `index_count` on
    // each `SkinnedDrawObject` (and the two GPU buffers themselves) move.
    // Skeleton-shape changes (joint-count mismatch) are still rejected one
    // level above this call (in `reload_assets`) -- they would need the
    // original `vert_lib_bytes` / `frag_lib_bytes` / `shadow_lib_bytes`
    // which `upload_skinned` consumes and drops.
    fn rebuild_skinned_geometry(
        &mut self,
        changes: Vec<backend::SkinnedDrawGeometryUpdate>,
    ) -> RenderResult<Vec<backend::SkinnedSlotLayout>> {
        let v_buf = self.skinned.vertex_buffer.as_ref().ok_or_else(|| {
            RenderError::Other(
                "rebuild_skinned_geometry: no skinned vertex buffer (was upload_skinned called?)"
                    .to_string(),
            )
        })?;
        let i_buf = self.skinned.index_buffer.as_ref().ok_or_else(|| {
            RenderError::Other(
                "rebuild_skinned_geometry: no skinned index buffer (was upload_skinned called?)"
                    .to_string(),
            )
        })?;

        // Stop the GPU + CPU pipelines so we can safely read the old buffers
        // and atomically swap. Costs a frame-time stall but only fires under
        // `cn run -f` and only when the source `.glb` size actually changed.
        self.wait_idle();

        let old_v_len = v_buf.length() / size_of::<SkinnedVertex>();
        // SAFETY: the buffer is `StorageModeShared`, so `contents()` is a live CPU mapping of its
        // bytes, and the length was derived from that buffer's own byte length divided by the
        // element size. The preceding `wait_idle` means the GPU is not writing it.
        let old_v_slice: &[SkinnedVertex] = unsafe {
            let ptr = v_buf.contents().as_ptr() as *const SkinnedVertex;
            std::slice::from_raw_parts(ptr, old_v_len)
        };
        let old_i_len = i_buf.length() / size_of::<u32>();
        // SAFETY: the buffer is `StorageModeShared`, so `contents()` is a live CPU mapping of its
        // bytes, and the length was derived from that buffer's own byte length divided by the
        // element size. The preceding `wait_idle` means the GPU is not writing it.
        let old_i_slice: &[u32] = unsafe {
            let ptr = i_buf.contents().as_ptr() as *const u32;
            std::slice::from_raw_parts(ptr, old_i_len)
        };

        let repacked = geometry_repack::repack_skinned_geometry(
            &self.state.skinned.draw_objects,
            old_v_slice,
            old_i_slice,
            changes,
        )?;
        if repacked.ignored_changes > 0 {
            tracing::warn!(
                "rebuild_skinned_geometry: {} change(s) targeted skinned indices not \
                 in skinned_draw_objects (ignored)",
                repacked.ignored_changes
            );
        }
        let new_vertices = &repacked.vertices;
        let new_indices = &repacked.indices;

        // Create new MTL buffers sized to the rebuilt layout.
        // SAFETY: the pointer and length describe the live `new_vertices` allocation, and Metal
        // copies those bytes into the new buffer before the call returns.
        let new_vertex_buffer = unsafe {
            let v_bytes = size_of_val(new_vertices.as_slice());
            let ptr = std::ptr::NonNull::new(new_vertices.as_ptr() as *mut _).ok_or_else(|| {
                RenderError::Other(
                    "rebuild_skinned_geometry: vertex slice pointer is null".to_string(),
                )
            })?;
            self.hw
                .device
                .newBufferWithBytes_length_options(
                    ptr,
                    v_bytes,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| allocation_failed("rebuild_skinned_geometry vertex buffer"))?
        };
        let new_index_buffer =
            upload_skinned_index_buffer(&self.hw.device, new_indices, "rebuild_skinned_geometry")?;

        // Apply the new per-slot layout.
        repacked.apply_to(&mut self.state.skinned.draw_objects);

        self.skinned.vertex_buffer = Some(new_vertex_buffer);
        self.skinned.index_buffer = Some(new_index_buffer);
        Ok(repacked.layouts())
    }

    // Swap in rebuilt shared static-mesh buffers for the draws named in
    // `changes`, when a reloaded mesh no longer fits its slot. The repack reads
    // the live `StorageModeShared` contents after `wait_idle`, so no in-flight
    // command buffer touches the old pair.
    fn rebuild_static_geometry(
        &mut self,
        changes: Vec<backend::DrawGeometryUpdate>,
    ) -> RenderResult<()> {
        // Stop the GPU + CPU pipelines so we can safely read the old
        // buffers and atomically swap. Costs a frame-time stall but only
        // fires under `cn run -f` and only when the source `.glb` size
        // actually changed.
        self.wait_idle();

        // Read views over the current shared buffers. `StorageModeShared`
        // means `contents()` is a CPU-addressable pointer aliasing the GPU
        // data; safe after `wait_idle`.
        let old_v_len = self.scene.vertex_buffer.length() / size_of::<Vertex>();
        // SAFETY: the buffer is `StorageModeShared`, so `contents()` is a live CPU mapping of its
        // bytes, and the length was derived from that buffer's own byte length divided by the
        // element size. The preceding `wait_idle` means the GPU is not writing it.
        let old_v_slice: &[Vertex] = unsafe {
            let ptr = self.scene.vertex_buffer.contents().as_ptr() as *const Vertex;
            std::slice::from_raw_parts(ptr, old_v_len)
        };
        let old_i_len = self.scene.index_buffer.length() / size_of::<u32>();
        // SAFETY: the buffer is `StorageModeShared`, so `contents()` is a live CPU mapping of its
        // bytes, and the length was derived from that buffer's own byte length divided by the
        // element size. The preceding `wait_idle` means the GPU is not writing it.
        let old_i_slice: &[u32] = unsafe {
            let ptr = self.scene.index_buffer.contents().as_ptr() as *const u32;
            std::slice::from_raw_parts(ptr, old_i_len)
        };

        let repacked = geometry_repack::repack_static_geometry(
            &self.state.draw.objects,
            old_v_slice,
            old_i_slice,
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

        // Place new buffers sized to the rebuilt layout. The outgoing pair is
        // replaced below; its ranges return to the pool on drop, withheld until
        // the frames the `wait_idle` above drained can no longer reference them.
        let new_vertex_buffer = self
            .hw
            .allocator
            .alloc_buffer_with_bytes(
                bytes_of_slice(new_vertices.as_slice()),
                MTLResourceOptions::StorageModeShared,
            )
            .map_err(|e| e.context("rebuild_static_geometry: vertex buffer"))?;
        let new_index_buffer = self
            .hw
            .allocator
            .alloc_buffer_with_bytes(
                bytes_of_slice(new_indices.as_slice()),
                MTLResourceOptions::StorageModeShared,
            )
            .map_err(|e| e.context("rebuild_static_geometry: index buffer"))?;

        for (layout, obj) in repacked
            .layouts
            .into_iter()
            .zip(&mut self.state.draw.objects)
        {
            layout.apply_to(obj);
        }
        self.state.placement.forget_layout();

        self.scene.vertex_buffer = new_vertex_buffer;
        self.scene.index_buffer = new_index_buffer;

        // The RT acceleration structure (if any) was built against the OLD
        // vertex/index buffers + draw-object offsets. After this swap its static
        // BLAS hold the stale geometry and its geometry table carries stale
        // offsets, so reflections would trace mismatched data -- and the RT shader
        // reads the (possibly smaller) new vertex buffer at old offsets, risking
        // an out-of-bounds fetch. Rebuild the BVH from the new geometry now. We
        // are past `wait_idle` on the editor-only hot-reload path, so a synchronous
        // full rebuild (the same path the init build and a seed from empty use) is
        // appropriate. Rebuild regardless of `dynamic_mode` -- even a build-once
        // (`Off`) BVH is invalid once its source buffers are replaced. A rebuild
        // failure leaves the prior BVH in place (`rebuild_rt_accel` only swaps on
        // success) and must NOT fail the geometry reload, which already succeeded.
        if self.rt.accel.is_some() {
            let albedo_count = self.scene.textures.len();
            if let Err(e) = self.rebuild_rt_accel(albedo_count) {
                tracing::warn!(
                    "rebuild_static_geometry: RT BVH rebuild failed, reflections may be stale: {e}"
                );
            }
        }
        Ok(())
    }
}
