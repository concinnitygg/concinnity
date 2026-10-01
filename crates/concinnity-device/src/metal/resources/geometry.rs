//! The scene MtlContext lends to the `SceneHost` defaults, and the writer that
//! lands its streamed geometry in the shared vertex + index buffers.
//!
//! Both buffers are shared storage, so a write is a CPU copy into a region no
//! frame in flight reads. A released region is zeroed, so a stray draw of it
//! renders nothing rather than stale triangles.

use concinnity_core::render::backend::{GeometryEdit, SceneHost};
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::scene_state::{GeometryBuffer, GeometryWriter, SceneState};

use crate::metal::allocator::PooledBuffer;
use crate::metal::context::{
    MtlContext, debug_assert_main_thread, write_buffer_region, zero_buffer_region,
};

// The shared geometry buffers, written in place.
struct SharedBufferWriter<'a> {
    vertex: &'a PooledBuffer,
    index: &'a PooledBuffer,
}

impl SharedBufferWriter<'_> {
    fn buffer(&self, buffer: GeometryBuffer) -> &PooledBuffer {
        match buffer {
            GeometryBuffer::Vertex => self.vertex,
            GeometryBuffer::Index => self.index,
        }
    }
}

impl GeometryWriter for SharedBufferWriter<'_> {
    fn write(&mut self, buffer: GeometryBuffer, offset: usize, bytes: &[u8]) -> RenderResult<()> {
        write_buffer_region(self.buffer(buffer), offset, bytes)
    }

    fn release(&mut self, buffer: GeometryBuffer, offset: usize, len: usize) -> RenderResult<()> {
        zero_buffer_region(self.buffer(buffer), offset, len)
    }
}

impl SceneHost for MtlContext {
    fn scene(&self) -> Option<&SceneState> {
        Some(&self.state)
    }

    fn scene_mut(&mut self) -> Option<&mut SceneState> {
        debug_assert_main_thread("scene_mut");
        Some(&mut self.state)
    }

    fn edit_geometry(&mut self, edit: GeometryEdit<'_>) -> Option<RenderResult<()>> {
        debug_assert_main_thread("edit_geometry");
        let mut writer = SharedBufferWriter {
            vertex: &self.scene.vertex_buffer,
            index: &self.scene.index_buffer,
        };
        Some(edit(&mut self.state, &mut writer))
    }
}
