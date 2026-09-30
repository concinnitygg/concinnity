//! A backend that draws nothing: the smallest valid bodies for the required
//! methods and no optional family overridden, so every provided method runs its
//! default. Lets code that drives a `RenderBackend` be exercised without a GPU.

use super::*;

/// A `RenderBackend` that accepts every call and draws nothing.
pub struct NullBackend;

impl SceneControl for NullBackend {
    fn update_visibility(&mut self, _draw_idx: DrawIndex, _visible: bool) {}
    fn set_fade(&mut self, _fade: f32) {}
}

impl RenderBackend for NullBackend {
    fn window_closed(&mut self) -> bool {
        false
    }
    fn request_cursor_capture(&mut self) {}
    fn take_input(&mut self) -> InputSnapshot {
        InputSnapshot::default()
    }
    fn wait_idle(&self) {}
    fn draw_frame(&mut self, _params: FrameParams<'_>) -> RenderResult<()> {
        Ok(())
    }
    fn update_view(&mut self, _matrix: [[f32; 4]; 4]) {}
    fn update_models(&mut self, _updates: &[(DrawIndex, [[f32; 4]; 4])]) {}
    fn retire_draw_object(&mut self, _draw_idx: DrawIndex) {}
}

impl SkinnedDraws for NullBackend {
    fn upload_skinned(
        &mut self,
        _vertices: &[crate::gfx::mesh_payload::SkinnedVertex],
        _indices: &[u32],
        _draw_objects: alloc::vec::Vec<crate::gfx::render_types::SkinnedDrawObject>,
    ) -> RenderResult<()> {
        Ok(())
    }
    fn update_skinned_pose(&mut self, _skinned_index: SkinnedIndex, _matrices: &[[[f32; 4]; 4]]) {}
}

impl DrawStreaming for NullBackend {
    fn evict_texture_slot(&mut self, _slot: usize) -> RenderResult<()> {
        Ok(())
    }
    fn update_texture_slot(
        &mut self,
        _slot: usize,
        _image: &crate::bake::texture::TextureImage,
    ) -> RenderResult<()> {
        Ok(())
    }
    fn evict_mesh(&mut self, _draw_idx: DrawIndex, _retire_frame: u64) -> RenderResult<()> {
        Ok(())
    }
    fn upload_mesh(
        &mut self,
        _draw_idx: DrawIndex,
        _verts: &[crate::gfx::mesh_payload::Vertex],
        _idxs: &[u16],
        _frame: u64,
    ) -> RenderResult<()> {
        Ok(())
    }
    fn setup_chunk_streaming(
        &mut self,
        _chunk_vtx_bytes: usize,
        _chunk_idx_bytes: usize,
    ) -> RenderResult<()> {
        Ok(())
    }
    fn add_chunk_mesh(
        &mut self,
        _mesh: ChunkMesh<'_>,
        _dst: crate::render::draw_slot::SlotAlloc,
    ) -> RenderResult<()> {
        Ok(())
    }
    fn remove_chunk_mesh(&mut self, _draw_idx: DrawIndex, _retire_frame: u64) -> RenderResult<()> {
        Ok(())
    }
    fn set_chunk_model(&mut self, _draw_idx: DrawIndex, _model: [[f32; 4]; 4]) -> RenderResult<()> {
        Ok(())
    }
}

impl BackendProbe for NullBackend {
    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities::ALL
    }
}
impl LiveEdit for NullBackend {}
impl RenderTuning for NullBackend {}
impl SceneEffects for NullBackend {}
impl WindowControl for NullBackend {}
