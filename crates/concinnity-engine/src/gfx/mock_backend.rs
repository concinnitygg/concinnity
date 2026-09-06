// src/gfx/mock_backend.rs
//
// Test-only recording RenderBackend plus the GraphicsSystem injection hooks.
// Compiled solely into the unit-test binary (the `mod` declaration in gfx is
// `#[cfg(test)]`). The mock records every call GraphicsSystem makes (with its
// key parameters) into a shared `MockState` the test holds an Arc to, returns
// plausible fabricated values where the trait needs one, and captures a
// snapshot of the `BackendInit` the system assembled so tests can assert on
// the built draw lists and resolved settings without any GPU.

use std::sync::{Arc, Mutex};

use crate::gfx::backend::{ChunkMesh, DeviceCapabilities, FrameParams, GpuProfile, RenderBackend};
use crate::gfx::backend_init::{BackendInit, ShadowParams, SwapchainConfig};
use crate::gfx::error::{RenderError, RenderResult};
use crate::gfx::input::RenderInput;
use crate::gfx::mesh_payload::{SkinnedVertex, Vertex};
use crate::gfx::render_types::{DrawObject, MaterialUniforms, SkinnedDrawObject};
use crate::gfx::scene_flow::SceneControl;

// Everything a test injects into GraphicsSystem before init: the settings
// store contents (so the on-disk file is never read or written), the GPU
// profile (so no device handle is created), and the backend factory run in
// place of the real `init_backend`.
pub(crate) struct TestHooks {
    pub settings: crate::config::Settings,
    pub gpu_profile: GpuProfile,
    pub(crate) backend_factory: BackendFactory,
}

pub(crate) type BackendFactory =
    Box<dyn FnMut(BackendInit<'_>) -> Option<Box<dyn RenderBackend>> + Send>;

// The parts of the assembled `BackendInit` worth asserting on, captured by the
// factory before the mock backend is handed back. `draw_objects` is moved out
// wholesale so tests can inspect offsets, texture slots, and bounds.
pub(crate) struct InitSnapshot {
    pub(crate) window_width: u32,
    pub(crate) window_height: u32,
    pub(crate) window_title: String,
    pub clear_color: [f32; 4],
    pub(crate) frames_in_flight: usize,
    pub vsync: bool,
    pub vertex_count: usize,
    pub index_count: usize,
    pub(crate) draw_objects: Vec<DrawObject>,
    pub(crate) instanced_cluster_count: usize,
    pub(crate) n_skinned: usize,
    pub texture_count: usize,
    pub(crate) text_atlas_count: usize,
    pub shadows: ShadowParams,
    pub(crate) anisotropy: u32,
    pub(crate) scene_required: bool,
    pub fog: bool,
    pub(crate) taa_enabled: bool,
    pub(crate) ssao_on: bool,
    pub(crate) rt_reflections_on: bool,
    pub(crate) rt_dynamic: concinnity_core::render::rt_geom::RtDynamicMode,
    pub(crate) rt_skinned_geometry: bool,
    // Per shader bucket, how many compiled programs the payload carries. A
    // bucket with none is a world that declared no Shader for it, which every
    // backend reads as "use the engine's own main-pass program".
    pub(crate) shader_program_counts: Vec<usize>,
}

// One recorded backend call with the parameters tests assert on.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Call {
    CaptureCursor,
    WaitIdle,
    // A live world reload was applied onto this (transplanted) backend instead
    // of building a fresh one -- the `cn editor` hot-swap path.
    ReloadWorld,
    TakeInput,
    DrawFrame {
        world_hidden: bool,
        text_calls: usize,
        cam_pos: [f32; 3],
        view_mode: crate::gfx::view_modes::ViewMode,
        show: crate::gfx::view_modes::ShowFlags,
    },
    UpdateView([[f32; 4]; 4]),
    UpdateModel(usize),
    RetireDrawObject(usize),
    UploadSkinned {
        vertices: usize,
        draws: usize,
    },
    UpdateSkinnedPose(usize),
    UpdateSkinnedModel(usize),
    RevealSkinnedInstance(usize),
    EvictTextureSlot(usize),
    UpdateTextureSlot {
        slot: usize,
        w: u32,
        h: u32,
    },
    EvictMesh(usize),
    UploadMesh {
        draw_idx: usize,
        vertices: usize,
        indices: usize,
    },
    SeedMeshStreaming,
    SetupChunkStreaming,
    AddChunkMesh,
    RemoveChunkMesh(usize),
    SetChunkModel(usize),
    SetUiCursorHidden(bool),
    SetMenuMode(bool),
    SetCameraCapture(bool),
    SetReflectionProbes(usize),
    SetVsync(bool),
    SetWindowMode(crate::components::WindowMode),
    SetWindowSize(u32, u32),
    SetDisplayMode(crate::gfx::display_mode::DisplayMode),
    SetAmbientIntensity(f32),
    // The directional lights pushed, as (direction, colour, intensity) per light.
    UpdateDirectionalLights(Vec<([f32; 3], [f32; 3], f32)>),
    UpdateFogSettings(Option<crate::gfx::volumetric_fog::FogSettings>),
    SetKeymap,
    SetShadowUpdate,
    SetShadowDistance(u32),
    SetShadowCascades(u32),
    UpdatePostProcess,
    ApplyQualitySettings,
    UpdateQualityParams,
    CloneStaticDrawObject {
        src: usize,
        new_idx: usize,
    },
    UpdateVisibility {
        draw_idx: usize,
        visible: bool,
    },
    // A draw slot's material rewritten in place. The uniforms are not compared
    // (they carry no PartialEq); the pool slots identify which material landed.
    SetDrawMaterial {
        draw_idx: usize,
        texture_slot: usize,
        normal_map_slot: usize,
    },
    SetDrawCullDistance(usize, f32),
    SetFade(f32),
}

// Shared mutable state behind the mock: the ordered call log, the captured
// init snapshot, per-slot latches for the latest model / visibility pushes,
// and the knobs tests flip to steer the next poll (window close, draw error,
// input snapshot).
pub(crate) struct MockState {
    pub calls: Vec<Call>,
    pub init: Option<InitSnapshot>,
    // Latest model matrix pushed per draw slot.
    pub models: std::collections::HashMap<usize, [[f32; 4]; 4]>,
    // Latest visibility pushed per draw slot.
    pub visibility: std::collections::HashMap<usize, bool>,
    // Returned by the next window_closed() poll.
    pub window_closed: bool,
    // When set, draw_frame returns this error instead of Ok.
    pub(crate) fail_draw: Option<RenderError>,
    // When set, reload_world returns this error instead of Ok (exercising the
    // hot-swap failure path where GraphicsSystem marks itself failed).
    pub(crate) fail_reload: Option<String>,
    // Snapshot the next take_input() returns, then reset to default
    // (matching a real backend's drain-on-poll semantics).
    pub(crate) next_input: RenderInput,
    // Reported logical viewport size.
    pub(crate) logical_size: (f32, f32),
    // Reported top content inset: the window chrome a real macOS window leaves
    // over the top of the frame. Set before a step to stand in for it.
    pub(crate) top_inset: f32,
    // Capabilities reported to init, which uses them to gray out the settings
    // rows the device cannot honour. Set before `init_graphics` to stand in for
    // a device missing a feature.
    pub caps: DeviceCapabilities,
}

impl Default for MockState {
    fn default() -> Self {
        Self {
            calls: Vec::new(),
            init: None,
            models: std::collections::HashMap::new(),
            visibility: std::collections::HashMap::new(),
            window_closed: false,
            fail_draw: None,
            fail_reload: None,
            next_input: RenderInput::default(),
            logical_size: (1280.0, 720.0),
            top_inset: 0.0,
            caps: DeviceCapabilities::ALL,
        }
    }
}

impl MockState {
    // Whether the call log contains `call`.
    pub(crate) fn saw(&self, call: &Call) -> bool {
        self.calls.contains(call)
    }

    // Count of DrawFrame entries in the call log.
    pub(crate) fn draw_frames(&self) -> usize {
        self.calls
            .iter()
            .filter(|c| matches!(c, Call::DrawFrame { .. }))
            .count()
    }

    // The last DrawFrame entry, if any frame was drawn.
    pub(crate) fn last_draw_frame(&self) -> Option<Call> {
        self.calls
            .iter()
            .rev()
            .find(|c| matches!(c, Call::DrawFrame { .. }))
            .cloned()
    }
}

pub(crate) struct MockBackend {
    state: Arc<Mutex<MockState>>,
    // The swapchain config this backend reports as hot-swappable, mirroring a
    // real backend that can reuse its window for a live world reload. `None`
    // makes it behave like DirectX / Vulkan (never hot-swap; always rebuild).
    hot_swap: Option<SwapchainConfig>,
}

// Capture the assembled `BackendInit` into the shared state (the parts tests
// assert on), consuming it. Shared by the factory's fresh build and the
// transplanted backend's `reload_world`, so both paths record identical
// snapshots and a test can tell which one ran by which state was written.
fn record_init(state: &Arc<Mutex<MockState>>, init: BackendInit<'_>) {
    let mut s = state.lock().unwrap();
    s.init = Some(InitSnapshot {
        shader_program_counts: init
            .shaders
            .iter()
            .map(|sh| sh.programs.map_or(0, |p| p.programs.len()))
            .collect(),
        window_width: init.window.width,
        window_height: init.window.height,
        window_title: init.window.title.clone(),
        clear_color: init.clear_color,
        frames_in_flight: init.frames_in_flight,
        vsync: init.vsync,
        vertex_count: init.scene.vertices.len(),
        index_count: init.scene.indices.len(),
        draw_objects: init.scene.draw_objects,
        instanced_cluster_count: init.scene.instanced_clusters.len(),
        n_skinned: init.scene.n_skinned,
        texture_count: init.media.textures.len(),
        text_atlas_count: init.media.text_atlases.len(),
        shadows: init.shadows,
        anisotropy: init.anisotropy,
        scene_required: init.requirements.scene,
        fog: init.fx.fog.is_some(),
        taa_enabled: init.post.taa_enabled,
        ssao_on: init.post.ssao.is_some(),
        rt_reflections_on: init.post.rt_reflections.is_some(),
        rt_dynamic: init.post.rt_dynamic,
        rt_skinned_geometry: init.post.rt_skinned_geometry,
    });
}

// A bare recording backend and the state it records into, for tests that drive
// a system against a backend directly rather than through GraphicsSystem's
// init. Never hot-swaps (no init ran, so it has no swapchain config to report).
pub(crate) fn recording_backend() -> (Arc<Mutex<MockState>>, MockBackend) {
    let state = Arc::new(Mutex::new(MockState::default()));
    let backend = MockBackend {
        state: Arc::clone(&state),
        hot_swap: None,
    };
    (state, backend)
}

// Hooks with default settings (nothing persisted) and an unclassified GPU:
// the preset resolves to Auto with no ceiling, so the world's authored values
// pass through unclamped. Returns the shared state for assertions.
pub(crate) fn recording_hooks() -> (Arc<Mutex<MockState>>, TestHooks) {
    recording_hooks_with(crate::config::Settings::default(), GpuProfile::UNKNOWN)
}

// Hooks with an explicit settings store + GPU profile, for tests exercising
// persisted overrides and the quality-preset ceiling.
pub(crate) fn recording_hooks_with(
    settings: crate::config::Settings,
    gpu_profile: GpuProfile,
) -> (Arc<Mutex<MockState>>, TestHooks) {
    let state = Arc::new(Mutex::new(MockState::default()));
    let factory_state = Arc::clone(&state);
    let backend_factory: BackendFactory = Box::new(move |init: BackendInit<'_>| {
        // A fresh build advertises the world's own config as hot-swappable, so a
        // backend built here can later be transplanted onto another world.
        let hot_swap = Some(init.swapchain_config());
        record_init(&factory_state, init);
        Some(Box::new(MockBackend {
            state: Arc::clone(&factory_state),
            hot_swap,
        }) as Box<dyn RenderBackend>)
    });
    (
        state,
        TestHooks {
            settings,
            gpu_profile,
            backend_factory,
        },
    )
}

impl MockBackend {
    fn record(&self, call: Call) {
        self.state.lock().unwrap().calls.push(call);
    }

    // A backend the test transplants into a rebuilt world (a `PendingBackend`),
    // recording into its own `state` so a reload is distinguishable from a fresh
    // factory build. `hot_swap` is the config it reports as hot-swappable: match
    // the new world's `swapchain_config` to exercise the reload path, differ to
    // exercise the swapchain-change full-rebuild path.
    pub(crate) fn transplant(
        state: Arc<Mutex<MockState>>,
        hot_swap: Option<SwapchainConfig>,
    ) -> MockBackend {
        MockBackend { state, hot_swap }
    }
}

impl SceneControl for MockBackend {
    fn update_visibility(&mut self, draw_idx: usize, visible: bool) {
        let mut s = self.state.lock().unwrap();
        s.visibility.insert(draw_idx, visible);
        s.calls.push(Call::UpdateVisibility { draw_idx, visible });
    }

    fn set_fade(&mut self, fade: f32) {
        self.record(Call::SetFade(fade));
    }
}

impl RenderBackend for MockBackend {
    fn window_closed(&mut self) -> bool {
        self.state.lock().unwrap().window_closed
    }

    fn capture_cursor(&mut self) {
        self.record(Call::CaptureCursor);
    }

    fn take_input(&mut self) -> RenderInput {
        let mut s = self.state.lock().unwrap();
        s.calls.push(Call::TakeInput);
        std::mem::take(&mut s.next_input)
    }

    fn wait_idle(&self) {
        self.record(Call::WaitIdle);
    }

    fn hot_swap_config(&self) -> Option<SwapchainConfig> {
        self.hot_swap
    }

    fn reload_world(&mut self, init: BackendInit<'_>) -> RenderResult<()> {
        let fail = self.state.lock().unwrap().fail_reload.clone();
        self.record(Call::ReloadWorld);
        if let Some(e) = fail {
            return Err(e.into());
        }
        // Record the reloaded world's content into this (transplanted) backend's
        // state, exactly as the factory would for a fresh build, so a test can
        // assert the reused backend now carries the new world.
        record_init(&self.state, init);
        Ok(())
    }

    fn draw_frame(&mut self, params: FrameParams<'_>) -> RenderResult<()> {
        let mut s = self.state.lock().unwrap();
        s.calls.push(Call::DrawFrame {
            world_hidden: params.world_hidden,
            text_calls: params.text_calls.len(),
            cam_pos: params.cam_pos,
            view_mode: params.view_mode,
            show: params.show,
        });
        match &s.fail_draw {
            Some(e) => Err(e.clone()),
            None => Ok(()),
        }
    }

    fn update_view(&mut self, matrix: [[f32; 4]; 4]) {
        self.record(Call::UpdateView(matrix));
    }

    fn update_models(&mut self, updates: &[(u32, [[f32; 4]; 4])]) {
        let mut s = self.state.lock().unwrap();
        for &(index, model) in updates {
            s.models.insert(index as usize, model);
            s.calls.push(Call::UpdateModel(index as usize));
        }
    }

    fn update_skinned_models(&mut self, updates: &[(u32, [[f32; 4]; 4])]) {
        let mut s = self.state.lock().unwrap();
        for &(index, _model) in updates {
            s.calls.push(Call::UpdateSkinnedModel(index as usize));
        }
    }

    fn retire_draw_object(&mut self, draw_idx: usize) {
        self.record(Call::RetireDrawObject(draw_idx));
    }

    fn upload_skinned(
        &mut self,
        vertices: &[SkinnedVertex],
        _indices: &[u32],
        draw_objects: Vec<SkinnedDrawObject>,
    ) -> RenderResult<()> {
        self.record(Call::UploadSkinned {
            vertices: vertices.len(),
            draws: draw_objects.len(),
        });
        Ok(())
    }

    fn update_skinned_pose(&mut self, skinned_index: usize, _matrices: &[[[f32; 4]; 4]]) {
        self.record(Call::UpdateSkinnedPose(skinned_index));
    }

    fn reveal_skinned_instance(&mut self, instance_index: usize, _model: [[f32; 4]; 4]) {
        self.record(Call::RevealSkinnedInstance(instance_index));
    }

    fn evict_texture_slot(&mut self, slot: usize) -> Result<(), String> {
        self.record(Call::EvictTextureSlot(slot));
        Ok(())
    }

    fn update_texture_slot(
        &mut self,
        slot: usize,
        image: &crate::bake::texture::TextureImage,
    ) -> RenderResult<()> {
        self.record(Call::UpdateTextureSlot {
            slot,
            w: image.width(),
            h: image.height(),
        });
        Ok(())
    }

    fn evict_mesh(&mut self, draw_idx: usize, _retire_frame: u64) -> Result<(), String> {
        self.record(Call::EvictMesh(draw_idx));
        Ok(())
    }

    fn upload_mesh(
        &mut self,
        draw_idx: usize,
        verts: &[Vertex],
        idxs: &[u16],
        _frame: u64,
    ) -> RenderResult<()> {
        self.record(Call::UploadMesh {
            draw_idx,
            vertices: verts.len(),
            indices: idxs.len(),
        });
        Ok(())
    }

    fn seed_mesh_streaming(
        &mut self,
        _vtx_offset: u64,
        _vtx_bytes: u64,
        _idx_offset: u64,
        _idx_bytes: u64,
    ) {
        self.record(Call::SeedMeshStreaming);
    }

    fn setup_chunk_streaming(
        &mut self,
        _chunk_vtx_bytes: usize,
        _chunk_idx_bytes: usize,
    ) -> RenderResult<()> {
        self.record(Call::SetupChunkStreaming);
        Ok(())
    }

    fn add_chunk_mesh(
        &mut self,
        _mesh: ChunkMesh<'_>,
        _dst: crate::gfx::draw_slot::SlotAlloc,
    ) -> RenderResult<()> {
        self.record(Call::AddChunkMesh);
        Ok(())
    }

    fn remove_chunk_mesh(&mut self, draw_idx: usize, _retire_frame: u64) -> Result<(), String> {
        self.record(Call::RemoveChunkMesh(draw_idx));
        Ok(())
    }

    fn set_chunk_model(&mut self, draw_idx: usize, _model: [[f32; 4]; 4]) -> Result<(), String> {
        self.record(Call::SetChunkModel(draw_idx));
        Ok(())
    }

    fn logical_size(&self) -> (f32, f32) {
        self.state.lock().unwrap().logical_size
    }

    fn top_content_inset(&self) -> f32 {
        self.state.lock().unwrap().top_inset
    }

    fn capabilities(&self) -> DeviceCapabilities {
        self.state.lock().unwrap().caps
    }

    fn set_ui_cursor_hidden(&mut self, hidden: bool) {
        self.record(Call::SetUiCursorHidden(hidden));
    }

    fn set_menu_mode(&mut self, on: bool) {
        self.record(Call::SetMenuMode(on));
    }

    fn set_camera_capture(&mut self, capture: bool) {
        self.record(Call::SetCameraCapture(capture));
    }

    fn set_reflection_probes(&mut self, probes: &[crate::gfx::reflection_probe::ProbePlacement]) {
        self.record(Call::SetReflectionProbes(probes.len()));
    }

    fn set_vsync(&mut self, on: bool) {
        self.record(Call::SetVsync(on));
    }

    fn set_window_mode(&mut self, mode: crate::components::WindowMode) {
        self.record(Call::SetWindowMode(mode));
    }

    fn set_window_size(&mut self, width: u32, height: u32) {
        self.record(Call::SetWindowSize(width, height));
    }

    fn set_display_mode(&mut self, mode: crate::gfx::display_mode::DisplayMode) {
        self.record(Call::SetDisplayMode(mode));
    }

    fn set_ambient_intensity(&mut self, value: f32) {
        self.record(Call::SetAmbientIntensity(value));
    }

    fn update_directional_lights(&mut self, lights: &[crate::components::DirectionalLight]) {
        self.record(Call::UpdateDirectionalLights(
            lights
                .iter()
                .map(|l| (l.direction, l.color, l.intensity))
                .collect(),
        ));
    }

    fn update_fog_settings(&mut self, settings: Option<crate::gfx::volumetric_fog::FogSettings>) {
        self.record(Call::UpdateFogSettings(settings));
    }

    fn set_keymap(&mut self, _keymap: &crate::gfx::keymap::KeyMap) {
        self.record(Call::SetKeymap);
    }

    fn set_shadow_update(&mut self, _update: crate::components::ShadowUpdate) {
        self.record(Call::SetShadowUpdate);
    }

    fn set_shadow_distance(&mut self, distance: u32) {
        self.record(Call::SetShadowDistance(distance));
    }

    fn set_shadow_cascades(&mut self, count: u32) {
        self.record(Call::SetShadowCascades(count));
    }

    fn update_post_process(&mut self, _tunables: crate::gfx::render_types::PostProcessTunables) {
        self.record(Call::UpdatePostProcess);
    }

    fn apply_quality_settings(&mut self, _settings: crate::gfx::backend::QualitySettings) {
        self.record(Call::ApplyQualitySettings);
    }

    fn update_quality_params(&mut self, _settings: crate::gfx::backend::QualitySettings) {
        self.record(Call::UpdateQualityParams);
    }

    fn clone_static_draw_object(
        &mut self,
        src_draw_idx: usize,
        _model: [[f32; 4]; 4],
        dst: crate::gfx::draw_slot::SlotAlloc,
    ) -> Result<(), String> {
        use crate::gfx::draw_slot::SlotAlloc;
        let new_idx = match dst {
            SlotAlloc::Reuse(i) | SlotAlloc::Append(i) => i,
        };
        self.record(Call::CloneStaticDrawObject {
            src: src_draw_idx,
            new_idx,
        });
        Ok(())
    }

    fn set_draw_material(
        &mut self,
        draw_idx: usize,
        _material: MaterialUniforms,
        texture_slot: usize,
        normal_map_slot: usize,
    ) {
        self.record(Call::SetDrawMaterial {
            draw_idx,
            texture_slot,
            normal_map_slot,
        });
    }

    fn set_draw_cull_distance(&mut self, draw_idx: usize, cull_distance: f32) {
        self.record(Call::SetDrawCullDistance(draw_idx, cull_distance));
    }
}
