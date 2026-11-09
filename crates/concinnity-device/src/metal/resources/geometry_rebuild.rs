//! The shared static and skinned vertex + index buffers a size-changing asset
//! hot-reload reads back and replaces whole. Every buffer is
//! `StorageModeShared`, so the readback is a copy of `contents()`.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_engine::gfx::mesh_payload::{SkinnedVertex, Vertex};
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::backend::{GeometryBuffers, GeometryRebuild, RenderBackend};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice as _, MTLResourceOptions};

use super::skinning::upload_skinned_index_buffer;
use crate::metal::context::{MtlContext, bytes_of_slice};
use crate::metal::error::allocation_failed;

// A copy of `buffer`'s contents as `T`s, whole elements only.
fn read_shared<T: Copy>(buffer: &ProtocolObject<dyn MTLBuffer>) -> Vec<T> {
    let len = buffer.length() / size_of::<T>();
    // SAFETY: the buffer is `StorageModeShared`, so `contents()` is a live CPU mapping of its
    // bytes, `len` whole elements fit its byte length, and the caller drained the GPU, so
    // nothing writes it while it is copied.
    unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<T>(), len) }.to_vec()
}

impl GeometryBuffers for MtlContext {
    fn read_static_geometry(&mut self) -> RenderResult<(Vec<Vertex>, Vec<u32>)> {
        self.wait_idle();
        Ok((
            read_shared(&self.scene.vertex_buffer),
            read_shared(&self.scene.index_buffer),
        ))
    }

    // The outgoing pair's ranges return to the pool on drop, withheld until the
    // frames the read drained can no longer reference them.
    fn replace_static_geometry(
        &mut self,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> RenderResult<()> {
        let vertex_buffer = self
            .hw
            .allocator
            .alloc_buffer_with_bytes(
                bytes_of_slice(vertices),
                MTLResourceOptions::StorageModeShared,
            )
            .map_err(|e| e.context("rebuild_static_geometry: vertex buffer"))?;
        let index_buffer = self
            .hw
            .allocator
            .alloc_buffer_with_bytes(
                bytes_of_slice(indices),
                MTLResourceOptions::StorageModeShared,
            )
            .map_err(|e| e.context("rebuild_static_geometry: index buffer"))?;
        self.scene.vertex_buffer = vertex_buffer;
        self.scene.index_buffer = index_buffer;
        Ok(())
    }

    // The RT acceleration structure was built against the old buffers and draw
    // offsets, so its static BLAS hold stale geometry and the RT shader would
    // read the new vertex buffer at old offsets. Rebuild it synchronously,
    // whatever the dynamic mode: even a build-once BVH is invalid once its
    // source buffers are replaced. A failure keeps the prior BVH and must not
    // fail the reload, which already succeeded.
    fn static_geometry_moved(&mut self) {
        if self.rt.accel.is_some() {
            let albedo_count = self.scene.textures.len();
            if let Err(e) = self.rebuild_rt_accel(albedo_count) {
                tracing::warn!(
                    "rebuild_static_geometry: RT BVH rebuild failed, reflections may be stale: {e}"
                );
            }
        }
    }

    fn read_skinned_geometry(&mut self) -> RenderResult<(Vec<SkinnedVertex>, Vec<u32>)> {
        let (Some(vertex_buffer), Some(index_buffer)) =
            (&self.skinned.vertex_buffer, &self.skinned.index_buffer)
        else {
            return Err(RenderError::Other(
                "rebuild_skinned_geometry: no skinned buffers (was upload_skinned called?)".into(),
            ));
        };
        self.wait_idle();
        Ok((read_shared(vertex_buffer), read_shared(index_buffer)))
    }

    // The skinned pipelines and their velocity / SSAO / SSR variants are
    // untouched; only the buffers and the per-slot regions move.
    fn replace_skinned_geometry(
        &mut self,
        vertices: &[SkinnedVertex],
        indices: &[u32],
    ) -> RenderResult<()> {
        let ptr = std::ptr::NonNull::new(vertices.as_ptr().cast_mut()).ok_or_else(|| {
            RenderError::Other("rebuild_skinned_geometry: vertex slice pointer is null".into())
        })?;
        // SAFETY: the pointer and length describe the live `vertices` slice, and Metal copies
        // those bytes into the new buffer before the call returns.
        let vertex_buffer = unsafe {
            self.hw.device.newBufferWithBytes_length_options(
                ptr.cast(),
                size_of_val(vertices),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| allocation_failed("rebuild_skinned_geometry vertex buffer"))?;
        let index_buffer =
            upload_skinned_index_buffer(&self.hw.device, indices, "rebuild_skinned_geometry")?;
        self.skinned.vertex_buffer = Some(vertex_buffer);
        self.skinned.index_buffer = Some(index_buffer);
        Ok(())
    }
}

impl GeometryRebuild for MtlContext {}
