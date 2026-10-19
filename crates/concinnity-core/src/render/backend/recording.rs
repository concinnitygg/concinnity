//! A `RenderBackend` that records every call made on it, for tests. It records
//! each call (with its key parameters) into a shared [`RecordingState`] the
//! test holds an `Arc` to, returns plausible fabricated values where the trait
//! needs one, and captures a snapshot of the `BackendInit` it was built from
//! so tests can assert on the built draw lists and resolved settings without
//! any GPU.

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use hashbrown::HashMap;
use spin::Mutex;

use crate::bake::texture;
use crate::components::{DirectionalLight, WindowMode};
use crate::gfx::mesh_payload::{SkinnedVertex, Vertex};
use crate::gfx::render_types::{
    DrawIndex, DrawObject, MaterialUniforms, SkinnedDrawObject, SkinnedIndex,
};
use crate::gfx::{render_types, view_modes};
use crate::input::keymap;
use crate::input::snapshot::InputSnapshot;
use crate::render::backend::{
    BackendProbe, ChunkMesh, DeviceCapabilities, DrawStreaming, FrameParams, LiveEdit,
    RenderBackend, RenderTuning, SceneEffects, SkinnedDraws, SkinnedFrame, WindowControl,
};
use crate::render::backend_init::{BackendInit, ShadowCadence, ShadowParams, SwapchainConfig};
use crate::render::error::{RenderError, RenderResult};
use crate::render::scene_state::SceneState;
use crate::render::{backend, draw_slot, reflection_probe, volumetric_fog};
use crate::settings::ResolvedGraphics;
use crate::settings::apply::{ApplyPlan, SettingsTarget, apply_plan};
use crate::window::display_mode;

/// The parts of an assembled `BackendInit` worth asserting on, captured before
/// the backend is handed back. `draw_objects` is moved out wholesale so tests
/// can inspect offsets, texture slots, and bounds.
pub struct InitSnapshot {
    /// The window's requested width.
    pub window_width: u32,
    /// The window's requested height.
    pub window_height: u32,
    /// The window's title.
    pub window_title: String,
    /// The frame's clear color.
    pub clear_color: [f32; 4],
    /// Frames in flight.
    pub frames_in_flight: usize,
    /// Whether presentation waits for vsync.
    pub vsync: bool,
    /// Whether frame capture was requested.
    pub capture: bool,
    /// Whether the backend draws into an embedder's surface.
    pub embedded_surface: bool,
    /// Shared vertex count.
    pub vertex_count: usize,
    /// Shared index count.
    pub index_count: usize,
    /// The built draw records.
    pub draw_objects: Vec<DrawObject>,
    /// Instanced clusters.
    pub instanced_cluster_count: usize,
    /// Skinned draws.
    pub n_skinned: usize,
    /// Texture pool entries.
    pub texture_count: usize,
    /// Text atlases.
    pub text_atlas_count: usize,
    /// The shadow parameters.
    pub shadows: ShadowParams,
    /// The scene sampler's anisotropy.
    pub anisotropy: u32,
    /// Whether the world needs the scene pass.
    pub scene_required: bool,
    /// Whether the fog pass is built.
    pub fog: bool,
    /// Whether grass is built.
    pub grass: bool,
    /// Whether TAA runs.
    pub taa_enabled: bool,
    /// Whether SSAO runs.
    pub ssao_enabled: bool,
    /// Whether ray-traced reflections run.
    pub rt_reflections_on: bool,
    /// The ray-tracing dynamic-geometry mode.
    pub rt_dynamic: crate::render::rt_geom::RtDynamicMode,
    /// Whether skinned geometry joins the acceleration structure.
    pub rt_skinned_geometry: bool,
    /// The DLSS preset.
    pub dlss_preset: crate::render::dlss::DlssPreset,
    /// Per shader bucket, how many compiled programs the payload carries. A
    /// bucket with none is a world that declared no Shader for it, which every
    /// backend reads as "use the engine's own main-pass program".
    pub shader_program_counts: Vec<usize>,
}

/// One recorded backend call with the parameters tests assert on.
#[derive(Debug, Clone, PartialEq)]
#[expect(
    missing_docs,
    reason = "each variant names the backend call it records"
)]
pub enum Call {
    RequestCursorCapture,
    WaitIdle,
    /// A live world reload applied onto this (transplanted) backend instead of
    /// building a fresh one.
    ReloadWorld,
    TakeInput,
    PumpEvents,
    DrawFrame {
        world_hidden: bool,
        text_calls: usize,
        cam_pos: [f32; 3],
        view_mode: view_modes::ViewMode,
        show: view_modes::ShowFlags,
        history_reset: bool,
        rebase: Option<crate::planet::Rebase>,
        capture: Option<String>,
    },
    FinishCaptures,
    UpdateView([[f32; 4]; 4]),
    UpdateModel(DrawIndex),
    RetireDrawObject(DrawIndex),
    UploadSkinned {
        vertices: usize,
        draws: usize,
    },
    UploadSkinnedMorphs,
    UpdateSkinnedPose(SkinnedIndex),
    UpdateSkinnedModel(SkinnedIndex),
    RevealSkinnedInstance(SkinnedIndex),
    EvictTextureSlot(usize),
    UpdateTextureSlot {
        slot: usize,
        w: u32,
        h: u32,
    },
    EvictMesh(DrawIndex),
    UploadMesh {
        draw_idx: DrawIndex,
        vertices: usize,
        indices: usize,
    },
    SeedMeshStreaming,
    SetupChunkStreaming,
    AddChunkMesh,
    RemoveChunkMesh(DrawIndex),
    SetChunkModel(DrawIndex),
    SetUiCursorHidden(bool),
    SetMenuMode(bool),
    SetCameraCapture(bool),
    SetReflectionProbes(Vec<reflection_probe::ProbePlacement>),
    SetVsync(bool),
    SetWindowMode(WindowMode),
    SetWindowSize(u32, u32),
    SetDisplayMode(display_mode::DisplayMode),
    SetAmbientIntensity(f32),
    /// The directional lights pushed, as (direction, color, intensity) per light.
    UpdateDirectionalLights(Vec<([f32; 3], [f32; 3], f32)>),
    UpdateFogSettings(Option<volumetric_fog::FogSettings>),
    SetKeymap,
    SetShadowCadence(ShadowCadence),
    UpdatePostProcess,
    ApplyQualitySettings,
    UpdateQualityParams,
    CloneStaticDrawObject {
        src: DrawIndex,
        new_idx: DrawIndex,
    },
    UpdateVisibility {
        draw_idx: DrawIndex,
        visible: bool,
    },
    /// A draw slot's material rewritten in place. The uniforms are not
    /// compared (they carry no `PartialEq`); the pool slots identify which
    /// material landed.
    SetDrawMaterial {
        draw_idx: DrawIndex,
        texture_slot: usize,
        normal_map_slot: usize,
    },
    SetDrawCullDistance(DrawIndex, f32),
    SetMaterialParams {
        row: u32,
        params: [f32; render_types::MATERIAL_PARAM_COUNT],
    },
    SetFade(f32),
    /// A shader bucket's pipeline installed, with the name its programs carry
    /// and the name the prepared pipeline it was handed was built from.
    InstallWorldShader {
        bucket: u32,
        name: String,
        prepared: Option<String>,
    },
    EvictWorldShader(u32),
    /// The presented frame saved to this path.
    Screenshot(String),
}

/// Shared mutable state behind the backend: the ordered call log, the captured
/// init snapshot, per-slot latches for the latest model / visibility pushes,
/// and the knobs tests flip to steer the next poll.
pub struct RecordingState {
    /// Every call, oldest first.
    pub calls: Vec<Call>,
    /// The last `BackendInit` the backend was built or reloaded from.
    pub init: Option<InitSnapshot>,
    /// Latest model matrix pushed per draw slot.
    pub models: HashMap<DrawIndex, [[f32; 4]; 4]>,
    /// Latest visibility pushed per draw slot.
    pub visibility: HashMap<DrawIndex, bool>,
    /// Returned by the next `window_closed()` poll.
    pub window_closed: bool,
    /// When set, `draw_frame` returns this error instead of Ok.
    pub fail_draw: Option<RenderError>,
    /// Run inside each `draw_frame`, standing in for the work of a real draw.
    pub on_draw: Option<Box<dyn Fn() + Send>>,
    /// Returned by `render_stats()`.
    pub render_stats: crate::profile::RenderStats,
    /// When set, `reload_world` returns this error instead of Ok.
    pub fail_reload: Option<RenderError>,
    /// When set, `upload_skinned_morphs` returns this error instead of Ok.
    pub fail_morph_upload: Option<RenderError>,
    /// When set, `update_texture_slot` returns this error instead of Ok.
    pub fail_texture_upload: Option<RenderError>,
    /// When set, `evict_texture_slot` returns this error instead of Ok.
    pub fail_texture_evict: Option<RenderError>,
    /// When set, a settings apply's rebuild returns this error instead of Ok.
    pub fail_quality: Option<RenderError>,
    /// Snapshot the next `take_input()` returns, then reset to default (a real
    /// backend's drain-on-poll semantics).
    pub next_input: InputSnapshot,
    /// Reported logical viewport size.
    pub logical_size: (f32, f32),
    /// Reported top content inset: the window chrome a real macOS window leaves
    /// over the top of the frame.
    pub top_inset: f32,
    /// Capabilities reported to init, which uses them to gray out the settings
    /// rows the device cannot honor.
    pub caps: DeviceCapabilities,
    /// Handed out by `pipeline_builder`; `None` stands in for a backend that
    /// builds every install itself.
    pub pipeline_builder: Option<Arc<dyn backend::PipelineBuilder>>,
}

/// A pipeline a test's builder prepares: the name of the programs it was built
/// from.
pub struct RecordingPipeline(pub String);

impl Default for RecordingState {
    fn default() -> Self {
        Self {
            calls: Vec::new(),
            init: None,
            models: HashMap::new(),
            visibility: HashMap::new(),
            window_closed: false,
            fail_draw: None,
            on_draw: None,
            render_stats: crate::profile::RenderStats::default(),
            fail_reload: None,
            fail_morph_upload: None,
            fail_texture_upload: None,
            fail_texture_evict: None,
            fail_quality: None,
            next_input: InputSnapshot::default(),
            logical_size: (1280.0, 720.0),
            top_inset: 0.0,
            caps: DeviceCapabilities::ALL,
            pipeline_builder: None,
        }
    }
}

impl RecordingState {
    /// Whether the call log contains `call`.
    pub fn saw(&self, call: &Call) -> bool {
        self.calls.contains(call)
    }

    /// Count of `DrawFrame` entries in the call log.
    pub fn draw_frames(&self) -> usize {
        self.calls
            .iter()
            .filter(|c| matches!(c, Call::DrawFrame { .. }))
            .count()
    }

    /// The last `DrawFrame` entry, if any frame was drawn.
    pub fn last_draw_frame(&self) -> Option<Call> {
        self.calls
            .iter()
            .rev()
            .find(|c| matches!(c, Call::DrawFrame { .. }))
            .cloned()
    }
}

/// The state a [`RecordingBackend`] records into, shared with the test.
pub type SharedRecording = Arc<Mutex<RecordingState>>;

/// A backend that records every call into a [`RecordingState`].
pub struct RecordingBackend {
    state: SharedRecording,
    // The swapchain config this backend reports as hot-swappable, mirroring a
    // real backend that can reuse its window for a live world reload. `None`
    // makes it behave like DirectX / Vulkan (never hot-swap; always rebuild).
    hot_swap: Option<SwapchainConfig>,
}

// Capture the assembled `BackendInit` into the shared state (the parts tests
// assert on), consuming it. Shared by a fresh build and a transplanted
// backend's `reload_world`, so both paths record identical snapshots and a
// test can tell which one ran by which state was written.
fn record_init(state: &SharedRecording, init: BackendInit<'_>) {
    state.lock().init = Some(InitSnapshot {
        shader_program_counts: init
            .shaders
            .iter()
            .map(|sh| sh.programs.map_or(0, |p| p.programs.len()))
            .collect(),
        window_width: init.window.width,
        window_height: init.window.height,
        window_title: init.window.title.clone(),
        clear_color: init.graphics.background.color,
        frames_in_flight: init.graphics.frames_in_flight(),
        vsync: init.graphics.frame_pacing.vsync,
        capture: init.capture,
        embedded_surface: init.embedded_surface.is_some(),
        vertex_count: init.scene.vertices.len(),
        index_count: init.scene.indices.len(),
        draw_objects: init.scene.draw_objects,
        instanced_cluster_count: init.scene.instanced_clusters.len(),
        n_skinned: init.scene.n_skinned,
        texture_count: init.media.textures.len(),
        text_atlas_count: init.media.text_atlases.len(),
        shadows: init.graphics.shadow_params(),
        anisotropy: init.graphics.texture_filtering.anisotropy,
        scene_required: init.requirements.scene,
        fog: init.fx.fog.is_some(),
        grass: init.fx.grass.is_some(),
        taa_enabled: init.graphics.taa(),
        ssao_enabled: init.graphics.ssao().is_some(),
        rt_reflections_on: init.graphics.rt_reflections().is_some(),
        rt_dynamic: init.launch.rt_dynamic,
        rt_skinned_geometry: init.launch.rt_skinned_geometry,
        dlss_preset: init.graphics.dlss_preset,
    });
}

/// A bare recording backend and the state it records into, for tests that
/// drive a system against a backend directly rather than through graphics
/// init. Never hot-swaps (no init ran, so it has no swapchain config to
/// report).
pub fn recording_backend() -> (SharedRecording, RecordingBackend) {
    let state = SharedRecording::default();
    let backend = RecordingBackend::transplant(Arc::clone(&state), None);
    (state, backend)
}

impl RecordingBackend {
    fn record(&self, call: Call) {
        self.state.lock().calls.push(call);
    }

    /// A backend recording into `state` that reports `hot_swap` as the
    /// swapchain config it can reuse. A test transplants one into a rebuilt
    /// world: match the new world's `swapchain_config` to exercise the reload
    /// path, differ to exercise the full rebuild.
    pub fn transplant(state: SharedRecording, hot_swap: Option<SwapchainConfig>) -> Self {
        Self { state, hot_swap }
    }

    /// A backend built from `init`, recording into `state`. It reports the
    /// world's own swapchain config as hot-swappable, so it can later be
    /// transplanted onto another world.
    pub fn from_init(state: SharedRecording, init: BackendInit<'_>) -> Self {
        let hot_swap = Some(init.swapchain_config());
        record_init(&state, init);
        Self::transplant(state, hot_swap)
    }
}

impl backend::SceneHost for RecordingBackend {
    fn scene(&self) -> Option<&SceneState> {
        None
    }
    fn scene_mut(&mut self) -> Option<&mut SceneState> {
        None
    }
    fn edit_geometry(&mut self, _: backend::GeometryEdit<'_>) -> Option<RenderResult<()>> {
        None
    }
}

impl RenderBackend for RecordingBackend {
    fn update_visibility(&mut self, draw_idx: DrawIndex, visible: bool) {
        let mut s = self.state.lock();
        s.visibility.insert(draw_idx, visible);
        s.calls.push(Call::UpdateVisibility { draw_idx, visible });
    }

    fn set_fade(&mut self, fade: f32) {
        self.record(Call::SetFade(fade));
    }

    fn live_edit(&mut self) -> Option<&mut dyn LiveEdit> {
        Some(self)
    }

    fn live_edit_ref(&self) -> Option<&dyn LiveEdit> {
        Some(self)
    }

    fn window_closed(&mut self) -> bool {
        self.state.lock().window_closed
    }

    fn request_cursor_capture(&mut self) {
        self.record(Call::RequestCursorCapture);
    }

    fn pump_events(&mut self) {
        self.record(Call::PumpEvents);
    }

    fn take_input(&mut self) -> InputSnapshot {
        let mut s = self.state.lock();
        s.calls.push(Call::TakeInput);
        core::mem::take(&mut s.next_input)
    }

    fn wait_idle(&self) {
        self.record(Call::WaitIdle);
    }

    fn finish_captures(&mut self) {
        self.record(Call::FinishCaptures);
    }

    fn draw_frame(&mut self, params: FrameParams<'_>) -> RenderResult<()> {
        let s = &mut *self.state.lock();
        s.calls.push(Call::DrawFrame {
            world_hidden: params.world_hidden,
            text_calls: params.text_calls.len(),
            cam_pos: params.cam_pos,
            view_mode: params.view_mode,
            show: params.show,
            history_reset: params.history_reset,
            rebase: params.rebase,
            capture: params.capture.map(str::to_string),
        });
        if let Some(on_draw) = &s.on_draw {
            on_draw();
        }
        match &s.fail_draw {
            Some(e) => Err(e.clone()),
            None => Ok(()),
        }
    }

    fn update_view(&mut self, matrix: [[f32; 4]; 4]) {
        self.record(Call::UpdateView(matrix));
    }

    fn update_models(&mut self, updates: &[(DrawIndex, [[f32; 4]; 4])]) {
        let mut s = self.state.lock();
        for &(index, model) in updates {
            s.models.insert(index, model);
            s.calls.push(Call::UpdateModel(index));
        }
    }

    fn retire_draw_object(&mut self, draw_idx: DrawIndex) {
        self.record(Call::RetireDrawObject(draw_idx));
    }
}

impl SkinnedDraws for RecordingBackend {
    fn update_skinned(&mut self, frame: SkinnedFrame<'_>) {
        let mut s = self.state.lock();
        for (index, _) in frame.poses.iter() {
            s.calls.push(Call::UpdateSkinnedPose(index));
        }
        for &(index, _model) in frame.models {
            s.calls.push(Call::UpdateSkinnedModel(index));
        }
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

    fn upload_skinned_morphs(
        &mut self,
        _morphs: Vec<Option<Arc<crate::gfx::mesh_payload::PayloadMorphs>>>,
    ) -> RenderResult<()> {
        let mut s = self.state.lock();
        s.calls.push(Call::UploadSkinnedMorphs);
        s.fail_morph_upload.clone().map_or(Ok(()), Err)
    }

    fn reveal_skinned_instance(&mut self, instance_index: SkinnedIndex, _model: [[f32; 4]; 4]) {
        self.record(Call::RevealSkinnedInstance(instance_index));
    }
}

impl DrawStreaming for RecordingBackend {
    fn install_world_shader(
        &mut self,
        bucket: u32,
        programs: &crate::components::ShaderPrograms,
        prepared: Option<backend::PreparedPipelines>,
    ) -> RenderResult<()> {
        self.record(Call::InstallWorldShader {
            bucket,
            name: programs.name.clone(),
            prepared: prepared
                .and_then(backend::PreparedPipelines::downcast::<RecordingPipeline>)
                .map(|p| p.0),
        });
        Ok(())
    }

    fn evict_world_shader(&mut self, bucket: u32) {
        self.record(Call::EvictWorldShader(bucket));
    }

    fn pipeline_builder(&self) -> Option<Arc<dyn backend::PipelineBuilder>> {
        self.state.lock().pipeline_builder.clone()
    }

    fn evict_texture_slot(&mut self, slot: usize) -> RenderResult<()> {
        let mut s = self.state.lock();
        s.calls.push(Call::EvictTextureSlot(slot));
        s.fail_texture_evict.clone().map_or(Ok(()), Err)
    }

    fn update_texture_slot(
        &mut self,
        slot: usize,
        image: &texture::TextureImage,
    ) -> RenderResult<()> {
        let mut s = self.state.lock();
        s.calls.push(Call::UpdateTextureSlot {
            slot,
            w: image.width(),
            h: image.height(),
        });
        s.fail_texture_upload.clone().map_or(Ok(()), Err)
    }

    fn evict_mesh(&mut self, draw_idx: DrawIndex, _retire_frame: u64) -> RenderResult<()> {
        self.record(Call::EvictMesh(draw_idx));
        Ok(())
    }

    fn upload_mesh(
        &mut self,
        draw_idx: DrawIndex,
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
        _dst: draw_slot::SlotAlloc,
    ) -> RenderResult<()> {
        self.record(Call::AddChunkMesh);
        Ok(())
    }

    fn remove_chunk_mesh(&mut self, draw_idx: DrawIndex, _retire_frame: u64) -> RenderResult<()> {
        self.record(Call::RemoveChunkMesh(draw_idx));
        Ok(())
    }

    fn set_chunk_model(&mut self, draw_idx: DrawIndex, _model: [[f32; 4]; 4]) -> RenderResult<()> {
        self.record(Call::SetChunkModel(draw_idx));
        Ok(())
    }

    fn clone_static_draw_object(
        &mut self,
        src_draw_idx: DrawIndex,
        _model: [[f32; 4]; 4],
        dst: draw_slot::SlotAlloc,
    ) -> RenderResult<()> {
        self.record(Call::CloneStaticDrawObject {
            src: src_draw_idx,
            new_idx: dst.slot(),
        });
        Ok(())
    }
}

impl WindowControl for RecordingBackend {
    fn logical_size(&self) -> (f32, f32) {
        self.state.lock().logical_size
    }

    fn top_content_inset(&self) -> f32 {
        self.state.lock().top_inset
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

    fn set_vsync(&mut self, on: bool) {
        self.record(Call::SetVsync(on));
    }

    fn set_window_mode(&mut self, mode: WindowMode) {
        self.record(Call::SetWindowMode(mode));
    }

    fn set_window_size(&mut self, width: u32, height: u32) {
        self.record(Call::SetWindowSize(width, height));
    }

    fn set_display_mode(&mut self, mode: display_mode::DisplayMode) {
        self.record(Call::SetDisplayMode(mode));
    }

    fn set_keymap(&mut self, _keymap: keymap::KeyMap) {
        self.record(Call::SetKeymap);
    }
}

impl RenderTuning for RecordingBackend {
    fn apply_settings(
        &mut self,
        graphics: &ResolvedGraphics,
        plan: &ApplyPlan,
    ) -> RenderResult<()> {
        apply_plan(self, graphics, plan)
    }

    fn set_reflection_probes(&mut self, probes: &[reflection_probe::ProbePlacement]) {
        self.record(Call::SetReflectionProbes(probes.to_vec()));
    }

    fn update_directional_lights(&mut self, lights: &[DirectionalLight]) {
        self.record(Call::UpdateDirectionalLights(
            lights
                .iter()
                .map(|l| (l.direction, l.color, l.intensity))
                .collect(),
        ));
    }

    fn update_fog_settings(&mut self, settings: Option<volumetric_fog::FogSettings>) {
        self.record(Call::UpdateFogSettings(settings));
    }
}

// The plan's pieces, each recorded as the call it stands for.
impl SettingsTarget for RecordingBackend {
    fn rebuild_features(&mut self, _graphics: &ResolvedGraphics) -> RenderResult<()> {
        let mut s = self.state.lock();
        s.calls.push(Call::ApplyQualitySettings);
        s.fail_quality.clone().map_or(Ok(()), Err)
    }

    fn push_feature_params(&mut self, _graphics: &ResolvedGraphics) {
        self.record(Call::UpdateQualityParams);
    }

    fn push_post_process(&mut self, _tunables: render_types::PostProcessTunables) {
        self.record(Call::UpdatePostProcess);
    }

    fn push_ambient(&mut self, value: f32) {
        self.record(Call::SetAmbientIntensity(value));
    }

    fn push_shadow_cadence(&mut self, cadence: ShadowCadence) {
        self.record(Call::SetShadowCadence(cadence));
    }

    fn push_vsync(&mut self, on: bool) {
        self.record(Call::SetVsync(on));
    }
}

impl LiveEdit for RecordingBackend {
    fn hot_swap_config(&self) -> Option<SwapchainConfig> {
        self.hot_swap
    }

    fn reload_world(&mut self, init: BackendInit<'_>) -> RenderResult<()> {
        let fail = self.state.lock().fail_reload.clone();
        self.record(Call::ReloadWorld);
        if let Some(e) = fail {
            return Err(e);
        }
        // The reloaded world's content lands in this (transplanted) backend's
        // state, exactly as a fresh build's would, so a test can assert the
        // reused backend now carries the new world.
        record_init(&self.state, init);
        Ok(())
    }

    fn set_draw_material(
        &mut self,
        draw_idx: DrawIndex,
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

    fn set_draw_cull_distance(&mut self, draw_idx: DrawIndex, cull_distance: f32) {
        self.record(Call::SetDrawCullDistance(draw_idx, cull_distance));
    }

    fn set_material_params(&mut self, row: u32, params: [f32; render_types::MATERIAL_PARAM_COUNT]) {
        self.record(Call::SetMaterialParams { row, params });
    }
}

// Not recorded: the defaults stand, so a test that reaches a decal or an
// emitter sees the same `Err` a backend without those passes reports.
impl SceneEffects for RecordingBackend {}

impl BackendProbe for RecordingBackend {
    fn capabilities(&self) -> DeviceCapabilities {
        self.state.lock().caps
    }

    fn render_stats(&self) -> crate::profile::RenderStats {
        self.state.lock().render_stats
    }

    fn screenshot(&mut self, path: &str) -> RenderResult<String> {
        self.record(Call::Screenshot(path.to_string()));
        Ok(path.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Calls land in the shared log in order, and the log's queries read it.
    #[test]
    fn calls_are_recorded_in_order() {
        let (state, mut backend) = recording_backend();
        backend.set_vsync(true);
        backend.update_visibility(DrawIndex(3), false);
        let s = state.lock();
        assert_eq!(
            s.calls,
            [
                Call::SetVsync(true),
                Call::UpdateVisibility {
                    draw_idx: DrawIndex(3),
                    visible: false
                }
            ]
        );
        assert!(s.saw(&Call::SetVsync(true)));
        assert_eq!(s.visibility.get(&DrawIndex(3)), Some(&false));
        assert_eq!(s.draw_frames(), 0);
        assert!(s.last_draw_frame().is_none());
    }

    // A failure knob turns the next matching call into its error.
    #[test]
    fn a_failure_knob_fails_its_call() {
        let (state, mut backend) = recording_backend();
        state.lock().fail_texture_evict = Some(RenderError::Other("evict".to_string()));
        assert!(backend.evict_texture_slot(1).is_err());
        assert!(state.lock().saw(&Call::EvictTextureSlot(1)));
    }
}
