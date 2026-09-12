// src/directx/backend.rs
//
// RenderBackend impl for DxContext. Thin forwarders to the inherent
// methods scattered across directx/{context,resources}.rs.
//
// The trait splits into one supertrait per operation family, so this is one
// impl block per family, in the order they are declared on `RenderBackend`.
//
// Most forwarders are a mechanical 1:1 call into the inherent method of the
// same name, so each block states the signatures and the shared `forward!`
// macro writes the bodies. `assert = debug_assert_main_thread` guards the
// generated `&mut self` arms, so every mutation reached through the boxed
// trait object proves the main-thread invariant the `unsafe impl Send for
// DxContext` rests on. Forwarders that rename, drop args, or have a custom body
// stay hand-written beside the invocation. Mirrors src/metal/backend.rs.

use concinnity_core::bake;
use concinnity_core::components;
use concinnity_core::gfx::mesh_payload;
use concinnity_core::gfx::mesh_payload::{SkinnedVertex, Vertex};
use concinnity_core::gfx::profile::RenderStats;
use concinnity_core::gfx::render_types;
use concinnity_core::gfx::render_types::SkinnedDrawObject;
use concinnity_core::render::backend;
use concinnity_core::render::backend::{
    BackendProbe, ChunkMesh, DrawStreaming, FrameParams, LiveEdit, RenderBackend, RenderTuning,
    SceneEffects, SkinnedDraws, WindowControl,
};
use concinnity_core::render::backend_init;
use concinnity_core::render::decal;
use concinnity_core::render::display_mode;
use concinnity_core::render::draw_slot;
use concinnity_core::render::error;
use concinnity_core::render::input::RenderInput;
use concinnity_core::render::keymap::KeyMap;
use concinnity_core::render::particles;
use concinnity_core::render::reflection_probe;
use concinnity_core::render::volumetric_fog;

use super::context::{DxContext, debug_assert_main_thread};
use crate::forward::forward;

impl RenderBackend for DxContext {
    forward! { assert = debug_assert_main_thread;
        fn window_closed(&mut self) -> bool;
        fn capture_cursor(&mut self);
        fn take_input(&mut self) -> RenderInput;
        fn wait_idle(&self);
        fn draw_frame(&mut self, params: FrameParams<'_>) -> error::RenderResult<()>;
        fn update_view(&mut self, matrix: [[f32; 4]; 4]);
        fn update_models(&mut self, updates: &[(u32, [[f32; 4]; 4])]);
        fn retire_draw_object(&mut self, draw_idx: usize);
    }
}

impl SkinnedDraws for DxContext {
    forward! { assert = debug_assert_main_thread;
        fn update_skinned_pose(&mut self, skinned_index: usize, matrices: &[[[f32; 4]; 4]]);
        fn update_morph_weights(&mut self, skinned_index: usize, weights: &[f32]);
        fn reveal_skinned_instance(&mut self, instance_index: usize, model: [[f32; 4]; 4]);
        fn retire_skinned_draw_object(&mut self, skinned_index: usize);
        fn update_skinned_models(&mut self, updates: &[(u32, [[f32; 4]; 4])]);
    }

    fn upload_skinned(
        &mut self,
        vertices: &[SkinnedVertex],
        indices: &[u32],
        draw_objects: Vec<SkinnedDrawObject>,
    ) -> error::RenderResult<()> {
        debug_assert_main_thread("upload_skinned");
        // Every skinned stage is the engine's own here: the world's fragment is
        // shader model 5.1, which D3D12 cannot pair with the engine's 6.0
        // vertex (see `compile_skinned_shaders`).
        Ok(self.upload_skinned(vertices, indices, draw_objects)?)
    }

    // Trait method returns unit; the inherent returns Result (buffer
    // allocation can fail), so the forwarder logs instead of propagating.
    fn upload_skinned_morphs(
        &mut self,
        morphs: Vec<Option<std::sync::Arc<mesh_payload::PayloadMorphs>>>,
    ) {
        debug_assert_main_thread("upload_skinned_morphs");
        if let Err(e) = DxContext::upload_skinned_morphs(self, morphs) {
            tracing::error!("DirectX: morph target upload failed: {}", e);
        }
    }
}

impl DrawStreaming for DxContext {
    forward! { assert = debug_assert_main_thread;
        fn evict_texture_slot(&mut self, slot: usize) -> Result<(), String>;
        fn evict_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String>;
        fn seed_mesh_streaming(&mut self, vtx_offset: u64, vtx_bytes: u64, idx_offset: u64, idx_bytes: u64);
        fn remove_chunk_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String>;
        fn set_chunk_model(&mut self, draw_idx: usize, model: [[f32; 4]; 4]) -> Result<(), String>;
        fn clone_static_draw_object(&mut self, src_draw_idx: usize, model: [[f32; 4]; 4], dst: draw_slot::SlotAlloc) -> Result<(), String>;
        fn evict_world_shader(&mut self, bucket: u32);
    }

    // Typed-boundary forwarders: the inherent methods report `String` errors,
    // which `?` coerces to `RenderError::Other`. Sites that can classify a
    // failure construct the typed variant directly instead.
    fn update_texture_slot(
        &mut self,
        slot: usize,
        image: &bake::texture::TextureImage,
    ) -> error::RenderResult<()> {
        debug_assert_main_thread("update_texture_slot");
        Ok(DxContext::update_texture_slot(self, slot, image)?)
    }

    fn upload_mesh(
        &mut self,
        draw_idx: usize,
        verts: &[Vertex],
        idxs: &[u16],
        frame: u64,
    ) -> error::RenderResult<()> {
        debug_assert_main_thread("upload_mesh");
        Ok(DxContext::upload_mesh(self, draw_idx, verts, idxs, frame)?)
    }

    fn setup_chunk_streaming(
        &mut self,
        chunk_vtx_bytes: usize,
        chunk_idx_bytes: usize,
    ) -> error::RenderResult<()> {
        debug_assert_main_thread("setup_chunk_streaming");
        Ok(DxContext::setup_chunk_streaming(
            self,
            chunk_vtx_bytes,
            chunk_idx_bytes,
        )?)
    }

    fn add_chunk_mesh(
        &mut self,
        mesh: ChunkMesh<'_>,
        dst: draw_slot::SlotAlloc,
    ) -> error::RenderResult<()> {
        debug_assert_main_thread("add_chunk_mesh");
        DxContext::add_chunk_mesh(self, mesh, dst)
    }

    fn install_world_shader(
        &mut self,
        bucket: u32,
        shader: backend_init::WorldShader<'_>,
    ) -> error::RenderResult<()> {
        debug_assert_main_thread("install_world_shader");
        DxContext::install_world_shader(self, bucket, shader)
            .map_err(error::RenderError::ShaderCompile)
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
        fn apply_quality_settings(&mut self, settings: backend::QualitySettings);
        fn set_shadow_update(&mut self, update: components::ShadowUpdate);
        fn set_shadow_distance(&mut self, distance: u32);
        fn set_shadow_cascades(&mut self, count: u32);
        fn update_quality_params(&mut self, settings: backend::QualitySettings);
        fn update_fog_settings(&mut self, settings: Option<volumetric_fog::FogSettings>);
    }
}

impl LiveEdit for DxContext {
    forward! { assert = debug_assert_main_thread;
        fn update_color_lut(&mut self, size: u32, data: &[u8]) -> Result<(), String>;
        fn update_mesh_geometry(&mut self, draw_idx: usize, verts: &[mesh_payload::Vertex], idxs: &[u16], lod_alternates: &[(f32, Vec<u16>)]) -> Result<(), String>;
        fn update_world_shader_pipelines(&mut self, programs: &concinnity_core::components::ShaderPrograms) -> Result<(), String>;
        fn update_skinned_mesh_geometry(&mut self, skinned_index: usize, vertex_base: u32, verts: &[mesh_payload::SkinnedVertex], idxs: &[u16]) -> Result<(), String>;
        fn update_skinned_skeleton(&mut self, skinned_index: usize, new_joint_count: usize) -> Result<(), String>;
        fn rebuild_skinned_geometry(&mut self, changes: Vec<backend::SkinnedDrawGeometryUpdate>) -> Result<Vec<backend::SkinnedSlotLayout>, String>;
    }

    // The swapchain config this live context was built with. A live `cn editor`
    // reload reuses this backend (via `reload_world`) only when the new world's
    // `swapchain_config` matches; otherwise the swap does a full rebuild.
    fn hot_swap_config(&self) -> Option<backend_init::SwapchainConfig> {
        Some(self.swapchain_config)
    }

    // Rebuild the world's content on the retained device + window + swapchain.
    // Inherent method named `apply_world_reload` so this forwarder does not
    // shadow-and-recurse (mirrors the Metal backend).
    fn reload_world(&mut self, init: backend_init::BackendInit<'_>) -> error::RenderResult<()> {
        Ok(self.apply_world_reload(init)?)
    }

    fn update_environment_map(&mut self, payload: &[u8]) -> error::RenderResult<()> {
        debug_assert_main_thread("update_environment_map");
        Ok(DxContext::update_environment_map(self, payload)?)
    }

    fn rebuild_static_geometry(
        &mut self,
        changes: Vec<backend::DrawGeometryUpdate>,
    ) -> error::RenderResult<()> {
        debug_assert_main_thread("rebuild_static_geometry");
        Ok(DxContext::rebuild_static_geometry(self, changes)?)
    }

    fn draw_geometry_size(&self, draw_idx: usize) -> Option<(usize, usize)> {
        self.draw
            .objects
            .get(draw_idx)
            .map(|o| (o.vertex_count, o.index_count))
    }

    fn draw_lod_index_counts(&self, draw_idx: usize) -> Option<Vec<usize>> {
        self.draw
            .objects
            .get(draw_idx)
            .map(|o| o.lod_alternates.iter().map(|s| s.index_count).collect())
    }

    fn shader_reload_flag(&self) -> Option<std::sync::Arc<std::sync::atomic::AtomicBool>> {
        self.shader_reload_pending()
    }
}

impl SceneEffects for DxContext {
    forward! { assert = debug_assert_main_thread;
        fn add_decal(&mut self, record: decal::DecalRecord) -> Result<usize, String>;
        fn remove_decal(&mut self, decal_id: usize) -> Result<(), String>;
        fn add_emitter(&mut self, record: particles::ParticleEmitterRecord) -> Result<usize, String>;
        fn remove_emitter(&mut self, emitter_id: usize) -> Result<(), String>;
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
    fn screenshot(&mut self, path: &str) -> Result<String, String> {
        debug_assert_main_thread("screenshot");
        self.capture_screenshot(path)
    }

    // Inherent method is named `read_cull_status_buffer` for the same reason
    // `capture_screenshot` is: an inherent `read_cull_status` would shadow the
    // trait method and recurse.
    fn read_cull_status(&mut self) -> Result<Vec<u32>, String> {
        debug_assert_main_thread("read_cull_status");
        self.read_cull_status_buffer()
    }
}
