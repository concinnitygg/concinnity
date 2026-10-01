//! A backend that draws nothing: the smallest valid bodies for the required
//! methods, no scene, and no optional family overridden, so every provided
//! method runs its scene-less default. Lets code that drives a `RenderBackend`
//! be exercised without a GPU.

use super::*;

/// A `RenderBackend` that accepts every call and draws nothing.
pub struct NullBackend;

impl SceneControl for NullBackend {
    fn update_visibility(&mut self, _draw_idx: DrawIndex, _visible: bool) {}
    fn set_fade(&mut self, _fade: f32) {}
}

impl SceneHost for NullBackend {
    fn scene(&self) -> Option<&crate::render::scene_state::SceneState> {
        None
    }
    fn scene_mut(&mut self) -> Option<&mut crate::render::scene_state::SceneState> {
        None
    }
    fn edit_geometry(&mut self, _edit: GeometryEdit<'_>) -> Option<RenderResult<()>> {
        None
    }
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
    fn setup_chunk_streaming(
        &mut self,
        _chunk_vtx_bytes: usize,
        _chunk_idx_bytes: usize,
    ) -> RenderResult<()> {
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
