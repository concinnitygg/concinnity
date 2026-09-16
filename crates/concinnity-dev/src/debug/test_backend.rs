//! A do-nothing `RenderBackend` for driving the debug dispatch and reload
//! passes without a GPU. It implements the required methods only, so every
//! optional hook (runtime decals and emitters, screenshot, cull readback,
//! pipeline rebuilds) reports the trait default's `Err`.

use concinnity_core::bake::texture::TextureImage;
use concinnity_core::gfx::mesh_payload;
use concinnity_core::gfx::render_types;
use concinnity_core::input::snapshot::InputSnapshot;
use concinnity_core::render::backend;
use concinnity_core::render::draw_slot;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::scene_flow;

pub(crate) struct StubBackend;

impl scene_flow::SceneControl for StubBackend {
    fn update_visibility(&mut self, _draw_idx: usize, _visible: bool) {}
    fn set_fade(&mut self, _fade: f32) {}
}

impl backend::RenderBackend for StubBackend {
    fn window_closed(&mut self) -> bool {
        false
    }
    fn capture_cursor(&mut self) {}
    fn take_input(&mut self) -> InputSnapshot {
        InputSnapshot::default()
    }
    fn wait_idle(&self) {}
    fn draw_frame(&mut self, _params: backend::FrameParams<'_>) -> RenderResult<()> {
        Ok(())
    }
    fn update_view(&mut self, _matrix: [[f32; 4]; 4]) {}
    fn update_models(&mut self, _updates: &[(u32, [[f32; 4]; 4])]) {}
    fn retire_draw_object(&mut self, _draw_idx: usize) {}
}

impl backend::SkinnedDraws for StubBackend {
    fn upload_skinned(
        &mut self,
        _vertices: &[mesh_payload::SkinnedVertex],
        _indices: &[u32],
        _draw_objects: Vec<render_types::SkinnedDrawObject>,
    ) -> RenderResult<()> {
        Ok(())
    }
    fn update_skinned_pose(&mut self, _skinned_index: usize, _matrices: &[[[f32; 4]; 4]]) {}
}

impl backend::DrawStreaming for StubBackend {
    fn evict_texture_slot(&mut self, _slot: usize) -> RenderResult<()> {
        Ok(())
    }
    fn update_texture_slot(&mut self, _slot: usize, _image: &TextureImage) -> RenderResult<()> {
        Ok(())
    }
    fn evict_mesh(&mut self, _draw_idx: usize, _retire_frame: u64) -> RenderResult<()> {
        Ok(())
    }
    fn upload_mesh(
        &mut self,
        _draw_idx: usize,
        _verts: &[mesh_payload::Vertex],
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
        _mesh: backend::ChunkMesh<'_>,
        _slot: draw_slot::SlotAlloc,
    ) -> RenderResult<()> {
        Ok(())
    }
    fn remove_chunk_mesh(&mut self, _draw_idx: usize, _retire_frame: u64) -> RenderResult<()> {
        Ok(())
    }
    fn set_chunk_model(&mut self, _draw_idx: usize, _model: [[f32; 4]; 4]) -> RenderResult<()> {
        Ok(())
    }
}

impl backend::WindowControl for StubBackend {}

impl backend::RenderTuning for StubBackend {}

impl backend::LiveEdit for StubBackend {}

impl backend::SceneEffects for StubBackend {}

impl backend::BackendProbe for StubBackend {
    fn capabilities(&self) -> backend::DeviceCapabilities {
        backend::DeviceCapabilities::ALL
    }
}
