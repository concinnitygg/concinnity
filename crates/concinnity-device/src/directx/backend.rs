//! RenderBackend impl for DxContext, one impl block per trait family; the
//! bodies come from `crate::forward`.

use concinnity_core::bake;
use concinnity_core::components;
use concinnity_core::gfx::mesh_payload;
use concinnity_core::gfx::mesh_payload::SkinnedVertex;
use concinnity_core::gfx::render_types;
use concinnity_core::gfx::render_types::SkinnedDrawObject;
use concinnity_core::input::keymap::KeyMap;
use concinnity_core::input::snapshot::InputSnapshot;
use concinnity_core::profile::RenderStats;
use concinnity_core::render::backend;
use concinnity_core::render::backend::{
    BackendProbe, DrawStreaming, FrameParams, LiveEdit, RenderBackend, RenderTuning, SceneEffects,
    SkinnedDraws, SkinnedIndex, WindowControl,
};
use concinnity_core::render::backend_init;
use concinnity_core::render::decal;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::particles;
use concinnity_core::render::reflection_probe;
use concinnity_core::render::volumetric_fog;
use concinnity_core::window::clipboard::Clipboard;
use concinnity_core::window::display_mode;

use super::context::{DxContext, debug_assert_main_thread};
use crate::forward::forward;

impl RenderBackend for DxContext {
    forward! { assert = debug_assert_main_thread;
        fn window_closed(&mut self) -> bool;
        fn request_cursor_capture(&mut self);
        fn take_input(&mut self) -> InputSnapshot;
        fn wait_idle(&self);
        fn draw_frame(&mut self, params: FrameParams<'_>) -> RenderResult<()>;
    }
}

impl SkinnedDraws for DxContext {
    forward! { assert = debug_assert_main_thread;
        fn upload_skinned_morphs(&mut self, morphs: Vec<Option<std::sync::Arc<mesh_payload::PayloadMorphs>>>) -> RenderResult<()>;
        // Every skinned stage is the engine's own here: the world's fragment is
        // shader model 5.1, which D3D12 cannot pair with the engine's 6.0
        // vertex (see `compile_skinned_shaders`).
        fn upload_skinned(&mut self, vertices: &[SkinnedVertex], indices: &[u32], draw_objects: Vec<SkinnedDrawObject>) -> RenderResult<()>;
    }
}

impl DrawStreaming for DxContext {
    forward! { assert = debug_assert_main_thread;
        fn evict_texture_slot(&mut self, slot: usize) -> RenderResult<()>;
        fn update_texture_slot(&mut self, slot: usize, image: &bake::texture::TextureImage) -> RenderResult<()>;
        fn evict_world_shader(&mut self, bucket: u32);
        fn setup_chunk_streaming(&mut self, chunk_vtx_bytes: usize, chunk_idx_bytes: usize) -> RenderResult<()>;
    }

    forward! { assert = debug_assert_main_thread;
        fn install_world_shader(&mut self, bucket: u32, programs: &concinnity_core::components::ShaderPrograms, prepared: Option<concinnity_core::render::backend::PreparedPipelines>) -> RenderResult<()>;
    }

    fn pipeline_builder(&self) -> Option<std::sync::Arc<dyn backend::PipelineBuilder>> {
        debug_assert_main_thread("pipeline_builder");
        Some(DxContext::pipeline_builder(self))
    }
}

impl WindowControl for DxContext {
    forward! { assert = debug_assert_main_thread;
        fn set_ui_cursor_hidden(&mut self, hidden: bool);
        fn cursor_outside_window(&self) -> bool;
        fn set_menu_mode(&mut self, on: bool);
        fn set_camera_capture(&mut self, capture: bool);
        fn set_vsync(&mut self, on: bool);
        fn set_window_mode(&mut self, mode: components::WindowMode);
        fn set_window_size(&mut self, width: u32, height: u32);
        fn display_modes(&self) -> Vec<display_mode::DisplayMode>;
        fn current_display_mode(&self) -> Option<display_mode::DisplayMode>;
        fn set_display_mode(&mut self, mode: display_mode::DisplayMode);
        fn set_keymap(&mut self, keymap: &KeyMap);
        fn logical_size(&self) -> (f32, f32);
    }

    fn clipboard(&mut self) -> Option<&mut dyn Clipboard> {
        debug_assert_main_thread("clipboard");
        Some(self.win_mut())
    }
}

impl RenderTuning for DxContext {
    forward! { assert = debug_assert_main_thread;
        fn set_reflection_probes(&mut self, probes: &[reflection_probe::ProbePlacement]);
        fn update_post_process(
            &mut self,
            tunables: render_types::PostProcessTunables,
        );
        fn set_ambient_intensity(&mut self, value: f32);
        fn update_directional_lights(&mut self, lights: &[components::DirectionalLight]);
        fn move_local_lights(&mut self, lights: &concinnity_core::render::lights::LightData, uniforms: &concinnity_core::gfx::render_types::LightUniforms) -> RenderResult<()>;
        fn apply_quality_settings(&mut self, settings: backend::QualitySettings) -> RenderResult<()>;
        fn set_shadow_cadence(&mut self, cadence: backend_init::ShadowCadence);
        fn update_quality_params(&mut self, settings: backend::QualitySettings);
        fn update_fog_settings(&mut self, settings: Option<volumetric_fog::FogSettings>);
    }
}

impl LiveEdit for DxContext {
    forward! { assert = debug_assert_main_thread;
        fn update_color_lut(&mut self, size: u32, data: &[u8]) -> RenderResult<()>;
        fn update_environment_map(&mut self, payload: &[u8]) -> RenderResult<()>;
        fn update_world_shader(&mut self, bucket: u32, programs: &concinnity_core::components::ShaderPrograms, prepared: Option<concinnity_core::render::backend::PreparedPipelines>) -> RenderResult<concinnity_core::render::backend::PipelineSwap>;
        fn replace_sdf_volume_pipelines(&mut self, volume: usize, programs: &concinnity_core::components::sdf_programs::SdfPrograms, prepared: Option<concinnity_core::render::backend::PreparedPipelines>) -> RenderResult<concinnity_core::render::backend::PipelineSwap>;
        fn update_skinned_mesh_geometry(&mut self, skinned_index: SkinnedIndex, vertex_base: u32, verts: &[mesh_payload::SkinnedVertex], idxs: &[u16]) -> RenderResult<()>;
        fn rebuild_skinned_geometry(&mut self, changes: Vec<backend::SkinnedDrawGeometryUpdate>) -> RenderResult<Vec<backend::SkinnedSlotLayout>>;
        fn rebuild_static_geometry(&mut self, changes: Vec<backend::DrawGeometryUpdate>) -> RenderResult<()>;
        fn set_material_params(&mut self, row: u32, params: [f32; concinnity_core::gfx::render_types::MATERIAL_PARAM_COUNT]);
    }

    // The swapchain config this live context was built with. A live `cn editor`
    // reload reuses this backend (via `reload_world`) only when the new world's
    // `swapchain_config` matches; otherwise the swap does a full rebuild.
    fn hot_swap_config(&self) -> Option<backend_init::SwapchainConfig> {
        Some(self.hw.swapchain_config)
    }

    // Rebuild the world's content on the retained device + window + swapchain.
    // Inherent method named `apply_world_reload` so this forwarder does not
    // shadow-and-recurse (mirrors the Metal backend).
    fn reload_world(&mut self, init: backend_init::BackendInit<'_>) -> RenderResult<()> {
        debug_assert_main_thread("reload_world");
        self.apply_world_reload(init)
    }

    fn shader_reload_flag(&self) -> Option<std::sync::Arc<std::sync::atomic::AtomicBool>> {
        self.shader_reload_pending()
    }
}

impl SceneEffects for DxContext {
    forward! { assert = debug_assert_main_thread;
        fn add_decal(&mut self, record: decal::DecalRecord) -> RenderResult<usize>;
        fn remove_decal(&mut self, decal_id: usize) -> RenderResult<()>;
        fn add_emitter(&mut self, record: particles::ParticleEmitterRecord) -> RenderResult<usize>;
        fn remove_emitter(&mut self, emitter_id: usize) -> RenderResult<()>;
    }
}

impl BackendProbe for DxContext {
    forward! { assert = debug_assert_main_thread;
        fn render_stats(&self) -> RenderStats;
        fn capabilities(&self) -> backend::DeviceCapabilities;
        fn gpu_profile(&self) -> backend::GpuProfile;
    }

    // Inherent method is named `capture_screenshot` to keep the forwarder
    // unambiguous (an inherent `screenshot` would shadow the trait method and
    // recurse); kept explicit out of the `forward!` macro for that rename.
    fn screenshot(&mut self, path: &str) -> RenderResult<String> {
        debug_assert_main_thread("screenshot");
        self.capture_screenshot(path)
    }

    // Inherent method is named `read_cull_status_buffer` for the same reason
    // `capture_screenshot` is: an inherent `read_cull_status` would shadow the
    // trait method and recurse.
    fn read_cull_status(&mut self) -> RenderResult<Vec<u32>> {
        debug_assert_main_thread("read_cull_status");
        self.read_cull_status_buffer()
    }
}
