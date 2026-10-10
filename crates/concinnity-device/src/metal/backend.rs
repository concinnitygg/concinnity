//! RenderBackend impl for MtlContext, one impl block per trait family; the
//! bodies come from `crate::forward`.

use concinnity_core::bake;
use concinnity_core::components;
use concinnity_core::gfx::mesh_payload;
use concinnity_core::gfx::mesh_payload::SkinnedVertex;
use concinnity_core::gfx::render_types::{
    MATERIAL_PARAM_COUNT, PostProcessTunables, SkinnedDrawObject,
};
use concinnity_core::input::keymap::KeyMap;
use concinnity_core::input::snapshot::InputSnapshot;
use concinnity_core::profile::RenderStats;
use concinnity_core::render::backend;
use concinnity_core::render::backend::{
    BackendProbe, DrawStreaming, FrameParams, LiveEdit, QualitySettings, RenderBackend,
    RenderTuning, SceneEffects, SkinnedDraws, SkinnedIndex, WindowControl,
};
use concinnity_core::render::backend_init;
use concinnity_core::render::decal;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::particles;
use concinnity_core::render::reflection_probe;
use concinnity_core::render::volumetric_fog;
use concinnity_core::window::clipboard::Clipboard;
use concinnity_core::window::display_mode;

use super::context::{MtlContext, debug_assert_main_thread};
use crate::forward::forward;

impl RenderBackend for MtlContext {
    forward! { assert = debug_assert_main_thread,
        via = self.window().appkit, via_mut = self.window_mut().appkit;
        fn take_input(&mut self) -> InputSnapshot;
    }

    fn request_cursor_capture(&mut self) {
        debug_assert_main_thread("request_cursor_capture");
        self.window_mut().appkit.capture_cursor();
    }

    forward! { assert = debug_assert_main_thread;
        fn wait_idle(&self);
    }

    fn draw_frame(&mut self, params: FrameParams<'_>) -> RenderResult<()> {
        // Not in the guarded `forward!` block: draw_frame needs the
        // MainThreadMarker as a *value* (it threads it into NSEvent pumping and
        // window ops), so it proves the invariant itself and returns Err off
        // the main thread rather than asserting: no point double-checking.
        //
        // A GPU-side failure surfaces asynchronously on a completed command
        // buffer, so a frame's error is reported here on a later call.
        if let Some(e) = self.take_device_error() {
            return Err(e);
        }
        MtlContext::draw_frame(self, params)
    }

    fn window_closed(&mut self) -> bool {
        // Metal's inherent method is &self; the trait takes &mut self for
        // parity with DX/VK.
        MtlContext::window_closed(self)
    }
}

impl SkinnedDraws for MtlContext {
    forward! { assert = debug_assert_main_thread;
        fn upload_skinned_morphs(&mut self, morphs: Vec<Option<std::sync::Arc<mesh_payload::PayloadMorphs>>>) -> RenderResult<()>;
        fn upload_skinned(&mut self, vertices: &[SkinnedVertex], indices: &[u32], draw_objects: Vec<SkinnedDrawObject>) -> RenderResult<()>;
    }
}

impl DrawStreaming for MtlContext {
    forward! { assert = debug_assert_main_thread;
        fn evict_texture_slot(&mut self, slot: usize) -> RenderResult<()>;
        fn evict_world_shader(&mut self, bucket: u32);
        fn update_texture_slot(&mut self, slot: usize, image: &bake::texture::TextureImage) -> RenderResult<()>;
        fn setup_chunk_streaming(&mut self, chunk_vtx_bytes: usize, chunk_idx_bytes: usize) -> RenderResult<()>;
    }

    forward! { assert = debug_assert_main_thread;
        fn install_world_shader(&mut self, bucket: u32, programs: &concinnity_core::components::ShaderPrograms, prepared: Option<backend::PreparedPipelines>) -> RenderResult<()>;
    }

    fn pipeline_builder(&self) -> Option<std::sync::Arc<dyn backend::PipelineBuilder>> {
        debug_assert_main_thread("pipeline_builder");
        Some(MtlContext::pipeline_builder(self))
    }
}

impl WindowControl for MtlContext {
    forward! { assert = debug_assert_main_thread,
        via = self.window().appkit, via_mut = self.window_mut().appkit;
        fn set_ui_cursor_hidden(&mut self, hidden: bool);
        fn cursor_outside_window(&self) -> bool;
        fn set_menu_mode(&mut self, on: bool);
        fn set_camera_capture(&mut self, capture: bool);
        fn set_window_mode(&mut self, mode: components::WindowMode);
        fn set_window_size(&mut self, width: u32, height: u32);
        fn display_modes(&self) -> Vec<display_mode::DisplayMode>;
        fn current_display_mode(&self) -> Option<display_mode::DisplayMode>;
        fn set_display_mode(&mut self, mode: display_mode::DisplayMode);
        fn set_keymap(&mut self, keymap: &KeyMap);
        fn logical_size(&self) -> (f32, f32);
        fn top_content_inset(&self) -> f32;
        fn clipboard(&mut self) -> Option<&mut dyn Clipboard>;
    }

    forward! { assert = debug_assert_main_thread;
        fn set_vsync(&mut self, on: bool);
    }
}

impl RenderTuning for MtlContext {
    forward! { assert = debug_assert_main_thread;
        fn set_reflection_probes(&mut self, probes: &[reflection_probe::ProbePlacement]);
        fn update_post_process(&mut self, tunables: PostProcessTunables);
        fn set_ambient_intensity(&mut self, value: f32);
        fn update_directional_lights(&mut self, lights: &[components::DirectionalLight]);
        fn move_local_lights(&mut self, lights: &concinnity_core::render::lights::LightData, uniforms: &concinnity_core::gfx::render_types::LightUniforms) -> RenderResult<()>;
        fn apply_quality_settings(&mut self, settings: QualitySettings) -> RenderResult<()>;
        fn set_shadow_cadence(&mut self, cadence: backend_init::ShadowCadence);
        fn update_quality_params(&mut self, settings: QualitySettings);
        fn update_fog_settings(&mut self, settings: Option<volumetric_fog::FogSettings>);
    }
}

impl LiveEdit for MtlContext {
    forward! { assert = debug_assert_main_thread;
        fn update_color_lut(&mut self, size: u32, data: &[u8]) -> RenderResult<()>;
        fn update_skinned_mesh_geometry(&mut self, skinned_index: SkinnedIndex, vertex_base: u32, verts: &[mesh_payload::SkinnedVertex], idxs: &[u16]) -> RenderResult<()>;
        fn rebuild_skinned_geometry(&mut self, changes: Vec<backend::SkinnedDrawGeometryUpdate>) -> RenderResult<Vec<backend::SkinnedSlotLayout>>;
        fn set_material_params(&mut self, row: u32, params: [f32; MATERIAL_PARAM_COUNT]);
        fn update_world_shader(&mut self, bucket: u32, programs: &concinnity_core::components::ShaderPrograms, prepared: Option<backend::PreparedPipelines>) -> RenderResult<backend::PipelineSwap>;
        fn replace_sdf_volume_pipelines(&mut self, volume: usize, programs: &concinnity_core::components::sdf_programs::SdfPrograms, prepared: Option<backend::PreparedPipelines>) -> RenderResult<backend::PipelineSwap>;
        fn update_environment_map(&mut self, payload: &[u8]) -> RenderResult<()>;
        fn rebuild_static_geometry(&mut self, changes: Vec<backend::DrawGeometryUpdate>) -> RenderResult<()>;
    }

    fn shader_reload_flag(&self) -> Option<std::sync::Arc<std::sync::atomic::AtomicBool>> {
        self.hot_reload
            .reload_pending
            .as_ref()
            .map(std::sync::Arc::clone)
    }

    // The swapchain config this live context can hot-swap a new world onto: the
    // ring depth plus the HDR-output request it was built with. A live `cn editor`
    // reload reuses this backend (via `reload_world`) only when the new world's
    // `swapchain_config` matches; otherwise the swap does a full rebuild.
    fn hot_swap_config(&self) -> Option<backend_init::SwapchainConfig> {
        Some(self.hw.swapchain_config)
    }

    // Inherent method is named `apply_world_reload` so this forwarder does not
    // shadow-and-recurse (mirrors `screenshot` / `capture_screenshot`).
    fn reload_world(&mut self, init: backend_init::BackendInit<'_>) -> RenderResult<()> {
        debug_assert_main_thread("reload_world");
        self.apply_world_reload(init)
    }
}

impl SceneEffects for MtlContext {
    forward! { assert = debug_assert_main_thread;
        fn add_decal(&mut self, record: decal::DecalRecord) -> RenderResult<usize>;
        fn remove_decal(&mut self, decal_id: usize) -> RenderResult<()>;
        fn add_emitter(&mut self, record: particles::ParticleEmitterRecord) -> RenderResult<usize>;
        fn remove_emitter(&mut self, emitter_id: usize) -> RenderResult<()>;
    }
}

impl BackendProbe for MtlContext {
    forward! { assert = debug_assert_main_thread;
        fn capabilities(&self) -> backend::DeviceCapabilities;
        fn gpu_profile(&self) -> backend::GpuProfile;
        fn render_stats(&self) -> RenderStats;
    }

    // Inherent method is named `capture_screenshot` to keep the forwarder
    // unambiguous (an inherent `screenshot` would shadow the trait method and
    // recurse). Mirrors the DX/VK backends.
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
