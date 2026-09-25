//! RenderBackend impl for MtlContext. Thin forwarders to the inherent
//! methods scattered across metal/{context,resources,streaming,draw}.rs.
//! Method resolution picks the inherent over the trait method when both
//! have the same name, so `self.draw_frame(...)` calls the inherent here.

use concinnity_core::bake;
use concinnity_core::components;
use concinnity_core::gfx::mesh_payload;
use concinnity_core::gfx::mesh_payload::{SkinnedVertex, Vertex};
use concinnity_core::gfx::render_types::{
    MaterialUniforms, PostProcessTunables, SkinnedDrawObject,
};
use concinnity_core::input::keymap::KeyMap;
use concinnity_core::input::snapshot::InputSnapshot;
use concinnity_core::profile::RenderStats;
use concinnity_core::render::backend;
use concinnity_core::render::backend::{
    BackendProbe, ChunkMesh, DrawStreaming, FrameParams, LiveEdit, QualitySettings, RenderBackend,
    RenderTuning, SceneEffects, SkinnedDraws, WindowControl,
};
use concinnity_core::render::backend_init;
use concinnity_core::render::decal;
use concinnity_core::render::draw_slot;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::particles;
use concinnity_core::render::reflection_probe;
use concinnity_core::render::volumetric_fog;
use concinnity_core::window::display_mode;

use super::context::{MtlContext, debug_assert_main_thread};
use crate::forward::forward;

// Most of the trait families are a mechanical 1:1 call into the inherent
// method of the same name, so the shared `forward!` macro writes those bodies
// from the signature list each impl block states. Inherent-over-trait method
// resolution makes the generated `self.<name>(...)` bind the inherent method,
// so there is no recursion. Methods that diverge (a receiver mismatch,
// dropped or renamed args, a custom body) are written out by hand beside the
// invocation.
//
// `assert = debug_assert_main_thread` guards the generated `&mut self` arms, so
// every mutation entry point reached through the boxed trait object proves the
// main-thread invariant the `unsafe impl Send for MtlContext` rests on: loud in
// debug, free in release. The `&self` arms stay unguarded: read-only access is
// the in-order parallel fan-out's whole point and is allowed off the main
// thread.
//
// `via` / `via_mut` forward the entry points whose implementation
// lives on the shared AppKit window layer rather than on `MtlContext` itself.
// Metal and Vulkan share that layer, so those arms carry no backend-specific
// behavior; only the main-thread assertion differs from a direct call.
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
        fn update_view(&mut self, matrix: [[f32; 4]; 4]);
        fn update_models(&mut self, updates: &[(u32, [[f32; 4]; 4])]);
        fn retire_draw_object(&mut self, draw_idx: usize);
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
        fn update_skinned_pose(&mut self, skinned_index: usize, matrices: &[[[f32; 4]; 4]]);
        fn update_morph_weights(&mut self, skinned_index: usize, weights: &[f32]);
        fn reveal_skinned_instance(&mut self, instance_index: usize, model: [[f32; 4]; 4]);
        fn retire_skinned_draw_object(&mut self, skinned_index: usize);
        fn update_skinned_models(&mut self, updates: &[(u32, [[f32; 4]; 4])]);
        fn upload_skinned_morphs(&mut self, morphs: Vec<Option<std::sync::Arc<mesh_payload::PayloadMorphs>>>) -> RenderResult<()>;
        fn upload_skinned(&mut self, vertices: &[SkinnedVertex], indices: &[u32], draw_objects: Vec<SkinnedDrawObject>) -> RenderResult<()>;
    }
}

impl DrawStreaming for MtlContext {
    forward! { assert = debug_assert_main_thread;
        fn evict_texture_slot(&mut self, slot: usize) -> RenderResult<()>;
        fn evict_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> RenderResult<()>;
        fn seed_mesh_streaming(&mut self, vtx_offset: u64, vtx_bytes: u64, idx_offset: u64, idx_bytes: u64);
        fn remove_chunk_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> RenderResult<()>;
        fn set_chunk_model(&mut self, draw_idx: usize, model: [[f32; 4]; 4]) -> RenderResult<()>;
        fn clone_static_draw_object(&mut self, src_draw_idx: usize, model: [[f32; 4]; 4], dst: draw_slot::SlotAlloc) -> RenderResult<()>;
        fn evict_world_shader(&mut self, bucket: u32);
        fn update_texture_slot(&mut self, slot: usize, image: &bake::texture::TextureImage) -> RenderResult<()>;
        fn upload_mesh(&mut self, draw_idx: usize, verts: &[Vertex], idxs: &[u16], frame: u64) -> RenderResult<()>;
        fn add_chunk_mesh(&mut self, mesh: ChunkMesh<'_>, dst: draw_slot::SlotAlloc) -> RenderResult<()>;
        fn setup_chunk_streaming(&mut self, chunk_vtx_bytes: usize, chunk_idx_bytes: usize) -> RenderResult<()>;
    }

    fn install_world_shader(
        &mut self,
        bucket: u32,
        shader: backend_init::WorldShader<'_>,
    ) -> RenderResult<()> {
        debug_assert_main_thread("install_world_shader");
        let programs = shader.programs.ok_or_else(|| {
            RenderError::ShaderCompile("shader bucket carries no programs".into())
        })?;
        MtlContext::install_world_shader(self, bucket, programs)
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
        fn apply_quality_settings(&mut self, settings: QualitySettings) -> RenderResult<()>;
        fn set_shadow_update(&mut self, update: components::ShadowUpdate);
        fn set_shadow_distance(&mut self, distance: u32);
        fn set_shadow_cascades(&mut self, count: u32);
        fn update_quality_params(&mut self, settings: QualitySettings);
        fn update_fog_settings(&mut self, settings: Option<volumetric_fog::FogSettings>);
    }
}

impl LiveEdit for MtlContext {
    forward! { assert = debug_assert_main_thread;
        fn update_color_lut(&mut self, size: u32, data: &[u8]) -> RenderResult<()>;
        fn update_mesh_geometry(&mut self, draw_idx: usize, verts: &[mesh_payload::Vertex], idxs: &[u16], lod_alternates: &[(f32, Vec<u16>)]) -> RenderResult<()>;
        fn update_skinned_mesh_geometry(&mut self, skinned_index: usize, vertex_base: u32, verts: &[mesh_payload::SkinnedVertex], idxs: &[u16]) -> RenderResult<()>;
        fn rebuild_skinned_geometry(&mut self, changes: Vec<backend::SkinnedDrawGeometryUpdate>) -> RenderResult<Vec<backend::SkinnedSlotLayout>>;
        fn update_skinned_skeleton(&mut self, skinned_index: usize, new_joint_count: usize) -> RenderResult<()>;
        fn set_draw_material(&mut self, draw_idx: usize, material: MaterialUniforms, texture_slot: usize, normal_map_slot: usize);
        fn set_draw_cull_distance(&mut self, draw_idx: usize, cull_distance: f32);
        fn update_world_shader(&mut self, bucket: u32, programs: &concinnity_core::components::ShaderPrograms) -> RenderResult<backend::WorldShaderSwap>;
        fn update_environment_map(&mut self, payload: &[u8]) -> RenderResult<()>;
        fn rebuild_static_geometry(&mut self, changes: Vec<backend::DrawGeometryUpdate>) -> RenderResult<()>;
    }

    fn shader_reload_flag(&self) -> Option<std::sync::Arc<std::sync::atomic::AtomicBool>> {
        self.hot_reload
            .reload_pending
            .as_ref()
            .map(std::sync::Arc::clone)
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
