//! A backend that draws nothing: the smallest valid bodies for the required
//! methods, no scene, and no optional family overridden, so every provided
//! method runs its scene-less default. Lets code that drives a `RenderBackend`
//! be exercised without a GPU.

use super::{
    BackendProbe, CullReadback, DecalPass, DeviceCapabilities, DrawStreaming, EmitterPass,
    FrameCapture, FrameParams, GeometryEdit, GeometryRebuild, LightTables, PlanetReshape,
    RenderBackend, RenderResult, SceneHost, SdfVolumeShaders, SkinnedDraws, TextureUploads,
    WindowControl, WorldReload, WorldShaders,
};
use crate::gfx::render_types::TextureSlot;
use crate::input::snapshot::InputSnapshot;
use crate::settings::apply::SettingsTarget;

/// A `RenderBackend` that accepts every call and draws nothing.
pub struct NullBackend;

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
    fn assert_owner_thread(&self) {}
    fn wait_idle(&self) {}
    fn draw_frame(&mut self, _params: FrameParams<'_>) -> RenderResult<()> {
        Ok(())
    }
}

impl WindowControl for NullBackend {
    fn window_closed(&mut self) -> bool {
        false
    }
    fn take_input(&mut self) -> InputSnapshot {
        InputSnapshot::default()
    }
    fn request_cursor_capture(&mut self) {}
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

impl TextureUploads for NullBackend {
    fn evict_texture_slot(&mut self, _slot: TextureSlot) -> RenderResult<()> {
        Ok(())
    }
    fn update_texture_slot(
        &mut self,
        _slot: TextureSlot,
        _image: &crate::bake::texture::TextureImage,
    ) -> RenderResult<()> {
        Ok(())
    }
}

impl DrawStreaming for NullBackend {
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
impl CullReadback for NullBackend {}
impl FrameCapture for NullBackend {}
impl GeometryRebuild for NullBackend {}
impl PlanetReshape for NullBackend {}
impl SettingsTarget for NullBackend {}
impl DecalPass for NullBackend {}
impl EmitterPass for NullBackend {}
impl LightTables for NullBackend {}
impl SdfVolumeShaders for NullBackend {}
impl WorldReload for NullBackend {}
impl WorldShaders for NullBackend {}
