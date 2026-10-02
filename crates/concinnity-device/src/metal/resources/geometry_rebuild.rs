//! Hot-reload rebuild of the shared static-mesh vertex + index buffers when
//! re-imported `.glb` source no longer fits each draw's init-time slot.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::gfx::mesh_payload::Vertex;
use concinnity_core::render::backend;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::geometry_repack;
use objc2_metal::{MTLBuffer as _, MTLResourceOptions};

use crate::metal::context::{MtlContext, bytes_of_slice};

impl MtlContext {
    // Swap in rebuilt shared static-mesh buffers for the draws named in
    // `changes`, when a reloaded mesh no longer fits its slot. The repack reads
    // the live `StorageModeShared` contents after `wait_idle`, so no in-flight
    // command buffer touches the old pair.
    pub(crate) fn rebuild_static_geometry(
        &mut self,
        changes: Vec<backend::DrawGeometryUpdate>,
    ) -> RenderResult<()> {
        // Stop the GPU + CPU pipelines so we can safely read the old
        // buffers and atomically swap. Costs a frame-time stall but only
        // fires under `cn debug` and only when the source `.glb` size
        // actually changed.
        self.wait_idle();

        // Read views over the current shared buffers. `StorageModeShared`
        // means `contents()` is a CPU-addressable pointer aliasing the GPU
        // data; safe after `wait_idle`.
        let old_v_len = self.scene.vertex_buffer.length() / std::mem::size_of::<Vertex>();
        // SAFETY: the buffer is `StorageModeShared`, so `contents()` is a live CPU mapping of its
        // bytes, and the length was derived from that buffer's own byte length divided by the
        // element size. The preceding `wait_idle` means the GPU is not writing it.
        let old_v_slice: &[Vertex] = unsafe {
            let ptr = self.scene.vertex_buffer.contents().as_ptr() as *const Vertex;
            std::slice::from_raw_parts(ptr, old_v_len)
        };
        let old_i_len = self.scene.index_buffer.length() / std::mem::size_of::<u32>();
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
