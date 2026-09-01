// src/metal/backend.rs
//
// RenderBackend impl for MtlContext. Thin forwarders to the inherent
// methods scattered across metal/{context,resources,streaming,draw}.rs.
// Method resolution picks the inherent over the trait method when both
// have the same name, so `self.draw_frame(...)` calls the inherent here.

use crate::gfx::backend::{ChunkMesh, FrameParams, QualitySettings, RenderBackend};
use crate::gfx::error::{RenderError, RenderResult};
use crate::gfx::input::RenderInput;
use crate::gfx::mesh_payload::{SkinnedVertex, Vertex};
use crate::gfx::profile::RenderStats;
use crate::gfx::render_types::{MaterialUniforms, PostProcessTunables, SkinnedDrawObject};

use super::context::{MtlContext, debug_assert_main_thread};

// Generate `RenderBackend` methods that forward 1:1 to the inherent
// `MtlContext` method of the same name: each entry is the trait signature, and
// the generated body is `self.<name>(<args…>)`. Inherent-over-trait method
// resolution makes that call bind the inherent method (not this trait one), so
// there is no recursion. Methods that diverge from a straight forward (a
// receiver mismatch, dropped/renamed args, or a custom body) are written out
// by hand below the invocation.
//
// The `&mut self` arms prepend `debug_assert_main_thread` so every mutation
// entry point reached through the boxed `RenderBackend` proves the
// main-thread invariant the `unsafe impl Send for MtlContext` rests on:
// loud in debug, free in release. The `&self` arms stay unguarded: read-only
// access is the in-order parallel fan-out's whole point and is allowed off
// the main thread. `draw_frame` is hand-written below (it needs the
// `MainThreadMarker` as a value and self-checks), so it is not listed here.
macro_rules! forward {
    () => {};
    (fn $name:ident(&self $(, $arg:ident: $ty:ty)* $(,)?) -> $ret:ty; $($rest:tt)*) => {
        fn $name(&self $(, $arg: $ty)*) -> $ret { self.$name($($arg),*) }
        forward!($($rest)*);
    };
    (fn $name:ident(&self $(, $arg:ident: $ty:ty)* $(,)?); $($rest:tt)*) => {
        fn $name(&self $(, $arg: $ty)*) { self.$name($($arg),*) }
        forward!($($rest)*);
    };
    (fn $name:ident(&mut self $(, $arg:ident: $ty:ty)* $(,)?) -> $ret:ty; $($rest:tt)*) => {
        fn $name(&mut self $(, $arg: $ty)*) -> $ret {
            debug_assert_main_thread(stringify!($name));
            self.$name($($arg),*)
        }
        forward!($($rest)*);
    };
    (fn $name:ident(&mut self $(, $arg:ident: $ty:ty)* $(,)?); $($rest:tt)*) => {
        fn $name(&mut self $(, $arg: $ty)*) {
            debug_assert_main_thread(stringify!($name));
            self.$name($($arg),*)
        }
        forward!($($rest)*);
    };
}

// The same forwarding, for the entry points whose implementation lives on the
// shared AppKit window layer (`crate::appkit::AppKitWindow`) rather than on
// `MtlContext` itself. Metal and Vulkan share that layer, so these arms carry no
// backend-specific behaviour; only the main-thread assertion differs from a
// direct call.
macro_rules! forward_win {
    () => {};
    (fn $name:ident(&self $(, $arg:ident: $ty:ty)* $(,)?) -> $ret:ty; $($rest:tt)*) => {
        fn $name(&self $(, $arg: $ty)*) -> $ret { self.window.appkit.$name($($arg),*) }
        forward_win!($($rest)*);
    };
    (fn $name:ident(&mut self $(, $arg:ident: $ty:ty)* $(,)?) -> $ret:ty; $($rest:tt)*) => {
        fn $name(&mut self $(, $arg: $ty)*) -> $ret {
            debug_assert_main_thread(stringify!($name));
            self.window.appkit.$name($($arg),*)
        }
        forward_win!($($rest)*);
    };
    (fn $name:ident(&mut self $(, $arg:ident: $ty:ty)* $(,)?); $($rest:tt)*) => {
        fn $name(&mut self $(, $arg: $ty)*) {
            debug_assert_main_thread(stringify!($name));
            self.window.appkit.$name($($arg),*)
        }
        forward_win!($($rest)*);
    };
}

impl RenderBackend for MtlContext {
    forward_win! {
        fn capture_cursor(&mut self);
        fn set_ui_cursor_hidden(&mut self, hidden: bool);
        fn cursor_outside_window(&self) -> bool;
        fn set_menu_mode(&mut self, on: bool);
        fn set_camera_capture(&mut self, capture: bool);
        fn set_window_mode(&mut self, mode: crate::components::WindowMode);
        fn set_window_size(&mut self, width: u32, height: u32);
        fn display_modes(&self) -> Vec<crate::gfx::display_mode::DisplayMode>;
        fn current_display_mode(&self) -> Option<crate::gfx::display_mode::DisplayMode>;
        fn set_display_mode(&mut self, mode: crate::gfx::display_mode::DisplayMode);
        fn set_keymap(&mut self, keymap: &crate::gfx::keymap::KeyMap);
        fn take_input(&mut self) -> RenderInput;
        fn logical_size(&self) -> (f32, f32);
        fn top_content_inset(&self) -> f32;
    }

    forward! {
        fn set_reflection_probes(&mut self, probes: &[crate::gfx::reflection_probe::ProbePlacement]);
        fn set_vsync(&mut self, on: bool);
        fn update_post_process(&mut self, tunables: PostProcessTunables);
        fn set_ambient_intensity(&mut self, value: f32);
        fn update_directional_lights(&mut self, lights: &[crate::components::DirectionalLight]);
        fn apply_quality_settings(&mut self, settings: QualitySettings);
        fn set_shadow_update(&mut self, update: crate::components::ShadowUpdate);
        fn set_shadow_distance(&mut self, distance: u32);
        fn set_shadow_cascades(&mut self, count: u32);
        fn update_quality_params(&mut self, settings: QualitySettings);
        fn wait_idle(&self);
        fn update_view(&mut self, matrix: [[f32; 4]; 4]);
        fn update_models(&mut self, updates: &[(u32, [[f32; 4]; 4])]);
        fn retire_draw_object(&mut self, draw_idx: usize);
        fn update_skinned_pose(&mut self, skinned_index: usize, matrices: &[[[f32; 4]; 4]]);
        fn update_morph_weights(&mut self, skinned_index: usize, weights: &[f32]);
        fn reveal_skinned_instance(&mut self, instance_index: usize, model: [[f32; 4]; 4]);
        fn retire_skinned_draw_object(&mut self, skinned_index: usize);
        fn update_skinned_models(&mut self, updates: &[(u32, [[f32; 4]; 4])]);
        fn evict_texture_slot(&mut self, slot: usize) -> Result<(), String>;
        fn evict_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String>;
        fn seed_mesh_streaming(&mut self, vtx_offset: u64, vtx_bytes: u64, idx_offset: u64, idx_bytes: u64);
        fn remove_chunk_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String>;
        fn set_chunk_model(&mut self, draw_idx: usize, model: [[f32; 4]; 4]) -> Result<(), String>;
        fn capabilities(&self) -> crate::gfx::backend::DeviceCapabilities;
        fn gpu_profile(&self) -> crate::gfx::backend::GpuProfile;
        fn render_stats(&self) -> RenderStats;
        fn update_color_lut(&mut self, size: u32, data: &[u8]) -> Result<(), String>;
        fn update_fog_settings(&mut self, settings: Option<crate::gfx::volumetric_fog::FogSettings>);
        fn update_mesh_geometry(&mut self, draw_idx: usize, verts: &[crate::gfx::mesh_payload::Vertex], idxs: &[u16], lod_alternates: &[(f32, Vec<u16>)]) -> Result<(), String>;
        fn update_skinned_mesh_geometry(&mut self, skinned_index: usize, vertex_base: u32, verts: &[crate::gfx::mesh_payload::SkinnedVertex], idxs: &[u16]) -> Result<(), String>;
        fn rebuild_skinned_geometry(&mut self, changes: Vec<crate::gfx::backend::SkinnedDrawGeometryUpdate>) -> Result<Vec<crate::gfx::backend::SkinnedSlotLayout>, String>;
        fn update_skinned_skeleton(&mut self, skinned_index: usize, new_joint_count: usize) -> Result<(), String>;
        fn clone_static_draw_object(&mut self, src_draw_idx: usize, model: [[f32; 4]; 4], dst: crate::gfx::draw_slot::SlotAlloc) -> Result<(), String>;
        fn set_draw_material(&mut self, draw_idx: usize, material: MaterialUniforms, texture_slot: usize, normal_map_slot: usize);
        fn set_draw_cull_distance(&mut self, draw_idx: usize, cull_distance: f32);
        fn add_decal(&mut self, record: crate::gfx::decal::DecalRecord) -> Result<usize, String>;
        fn remove_decal(&mut self, decal_id: usize) -> Result<(), String>;
        fn add_emitter(&mut self, record: crate::gfx::particles::ParticleEmitterRecord) -> Result<usize, String>;
        fn remove_emitter(&mut self, emitter_id: usize) -> Result<(), String>;
        fn update_world_shader_pipelines(&mut self, vert_bytes: Option<&[u8]>, frag_bytes: Option<&[u8]>, shadow_bytes: Option<&[u8]>, vert_instanced_bytes: Option<&[u8]>) -> Result<(), String>;
        fn evict_world_shader(&mut self, bucket: u32);
    }

    // Methods that are NOT a 1:1 forward; written out by hand.

    // Typed-boundary forwarders: the inherent methods report `String` errors,
    // which `?` coerces to `RenderError::Other`. Sites that can classify a
    // failure construct the typed variant directly instead.
    fn upload_skinned(
        &mut self,
        vertices: &[SkinnedVertex],
        indices: &[u32],
        draw_objects: Vec<SkinnedDrawObject>,
        vert_bytes: &[u8],
        frag_bytes: &[u8],
        shadow_bytes: &[u8],
    ) -> RenderResult<()> {
        debug_assert_main_thread("upload_skinned");
        Ok(MtlContext::upload_skinned(
            self,
            vertices,
            indices,
            draw_objects,
            vert_bytes,
            frag_bytes,
            shadow_bytes,
        )?)
    }

    fn update_texture_slot(
        &mut self,
        slot: usize,
        image: &crate::bake::texture::TextureImage,
    ) -> RenderResult<()> {
        debug_assert_main_thread("update_texture_slot");
        Ok(MtlContext::update_texture_slot(self, slot, image)?)
    }

    fn upload_mesh(
        &mut self,
        draw_idx: usize,
        verts: &[Vertex],
        idxs: &[u16],
        frame: u64,
    ) -> RenderResult<()> {
        debug_assert_main_thread("upload_mesh");
        Ok(MtlContext::upload_mesh(self, draw_idx, verts, idxs, frame)?)
    }

    fn add_chunk_mesh(
        &mut self,
        mesh: ChunkMesh<'_>,
        dst: crate::gfx::draw_slot::SlotAlloc,
    ) -> RenderResult<()> {
        debug_assert_main_thread("add_chunk_mesh");
        Ok(MtlContext::add_chunk_mesh(self, mesh, dst)?)
    }

    fn update_environment_map(&mut self, payload: &[u8]) -> RenderResult<()> {
        debug_assert_main_thread("update_environment_map");
        Ok(MtlContext::update_environment_map(self, payload)?)
    }

    fn rebuild_static_geometry(
        &mut self,
        changes: Vec<crate::gfx::backend::DrawGeometryUpdate>,
    ) -> RenderResult<()> {
        debug_assert_main_thread("rebuild_static_geometry");
        Ok(MtlContext::rebuild_static_geometry(self, changes)?)
    }

    // The inherent takes the two stage slices it needs rather than the whole
    // payload struct: a bucket pipeline pairs `vertex_main` with
    // `fragment_main_bindless` and has no instanced or shadow variant.
    fn install_world_shader(
        &mut self,
        bucket: u32,
        shader: crate::gfx::backend_init::ShaderBytes<'_>,
    ) -> RenderResult<()> {
        debug_assert_main_thread("install_world_shader");
        MtlContext::install_world_shader(self, bucket, shader.vert, shader.frag)
            .map_err(RenderError::ShaderCompile)
    }

    // Trait method returns unit; the inherent returns Result (buffer
    // allocation can fail), so the forwarder logs instead of propagating.
    fn upload_skinned_morphs(
        &mut self,
        morphs: Vec<Option<std::sync::Arc<crate::gfx::mesh_payload::PayloadMorphs>>>,
    ) {
        debug_assert_main_thread("upload_skinned_morphs");
        if let Err(e) = MtlContext::upload_skinned_morphs(self, morphs) {
            tracing::error!("Metal: morph target upload failed: {}", e);
        }
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
        Ok(MtlContext::draw_frame(self, params)?)
    }

    fn window_closed(&mut self) -> bool {
        // Metal's inherent method is &self; the trait takes &mut self for
        // parity with DX/VK.
        MtlContext::window_closed(self)
    }

    // Inherent method is named `capture_screenshot` to keep the forwarder
    // unambiguous (an inherent `screenshot` would shadow the trait method and
    // recurse). Mirrors the DX/VK backends.
    fn screenshot(&mut self, path: &str) -> Result<String, String> {
        debug_assert_main_thread("screenshot");
        self.capture_screenshot(path)
    }

    fn setup_chunk_streaming(
        &mut self,
        chunk_vtx_bytes: usize,
        chunk_idx_bytes: usize,
        _texture_slot: usize,
        _normal_map_slot: usize,
    ) -> RenderResult<()> {
        debug_assert_main_thread("setup_chunk_streaming");
        // Metal binds chunk textures per draw, so the slot args are unused.
        Ok(self.setup_chunk_streaming(chunk_vtx_bytes, chunk_idx_bytes)?)
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
    fn hot_swap_config(&self) -> Option<crate::gfx::backend_init::SwapchainConfig> {
        Some(crate::gfx::backend_init::SwapchainConfig {
            frames_in_flight: self.frames_in_flight,
            hdr_display: self.hdr.display_requested,
            hdr_pq: self.hdr.pq_requested,
        })
    }

    // Inherent method is named `apply_world_reload` so this forwarder does not
    // shadow-and-recurse (mirrors `screenshot` / `capture_screenshot`).
    fn reload_world(
        &mut self,
        init: crate::gfx::backend_init::BackendInit<'_>,
    ) -> RenderResult<()> {
        debug_assert_main_thread("reload_world");
        Ok(self.apply_world_reload(init)?)
    }
}
