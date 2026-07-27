// GraphicsSystem unit tests: drive run_init and run_step against the
// recording mock backend (gfx::mock_backend), a hand-assembled
// PipelineContext, and an in-memory blob. No GPU device is created and the
// on-disk settings store is never read or written: the injection seam
// (GraphicsSystem::test_hooks) supplies the settings, the GPU profile, and
// the backend factory.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::assets::{
    Camera3D, DespawnRequest, FrameInput, GraphicsConfig, HitRegion, Material, Prop, RenderHandle,
    ReparentRequest, Scene, SceneCommand, Shader, ShaderKind, SpawnRequest, Sprite,
    StreamingConfig, TextLabel, Transform, Window,
};
use crate::blob::BlobData;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{
    ComponentSlot, ComponentStorage, PayloadLocator, PipelineContext, Resources, StepResult,
    TextureHandle,
};
use crate::gfx::backend::{GpuProfile, GpuTier, GpuVendor};
use crate::gfx::backend_init::SwapchainConfig;
use crate::gfx::mock_backend::{
    Call, MockBackend, MockState, TestHooks, recording_hooks, recording_hooks_with,
};
use crate::gfx::profile::FrameProfile;
use crate::gfx::quality_preset::QualityPreset;

use super::GraphicsSystem;

const MESH: AssetId = AssetId(1);
const TEX: AssetId = AssetId(2);
const MAT: AssetId = AssetId(3);
const PROP: AssetId = AssetId(4);

// Owns the storage a PipelineContext borrows from, plus the in-memory blob
// serving every payload locator the builder handed out.
struct TestWorld {
    components: ComponentStorage,
    blob: BlobData,
    profile: FrameProfile,
    resources: Resources,
}

impl TestWorld {
    fn ctx(&mut self) -> PipelineContext<'_> {
        PipelineContext {
            components: &mut self.components,
            blob: &mut self.blob,
            profile: &mut self.profile,
            resources: &mut self.resources,
        }
    }
}

// Accumulates components and payload bytes, then seals into a TestWorld whose
// single in-memory blob section serves every locator it handed out.
struct WorldBuilder {
    components: ComponentStorage,
    section: Vec<u8>,
    texture_records: Vec<concinnity_core::ecs::ResourceRecord>,
    // Material data-resource records; each `push_textured_quad` bakes one Material
    // into `data_bytes` at the next handle, mirroring cook.
    material_records: Vec<concinnity_core::ecs::ResourceRecord>,
    // Mesh resource records; each `push_textured_quad` Mesh takes the next
    // handle in declaration order, matching the runtime's mesh-source table.
    mesh_records: Vec<concinnity_core::ecs::ResourceRecord>,
    // The singleton / pooled media kinds `push_resource` fills: font atlases,
    // the colour-grading LUT, the IBL environment, and skinned geometry. Each
    // stays empty unless a test authors one.
    font_records: Vec<concinnity_core::ecs::ResourceRecord>,
    color_lut_records: Vec<concinnity_core::ecs::ResourceRecord>,
    env_map_records: Vec<concinnity_core::ecs::ResourceRecord>,
    skinned_records: Vec<concinnity_core::ecs::ResourceRecord>,
}

impl WorldBuilder {
    fn new() -> Self {
        Self {
            components: ComponentStorage::default(),
            section: Vec::new(),
            texture_records: Vec::new(),
            material_records: Vec::new(),
            mesh_records: Vec::new(),
            font_records: Vec::new(),
            color_lut_records: Vec::new(),
            env_map_records: Vec::new(),
            skinned_records: Vec::new(),
        }
    }

    fn payload(&mut self, bytes: &[u8]) -> PayloadLocator {
        let offset = self.section.len() as u64;
        self.section.extend_from_slice(bytes);
        PayloadLocator {
            blob_index: 0,
            offset,
            len: bytes.len() as u64,
        }
    }

    fn push<C: ComponentSlot>(&mut self, c: C) {
        self.components.push_typed(c);
    }

    // One Shader whose payload container carries the given compiled stages.
    // The stage bytes are opaque to the mock, so any bytes serve.
    fn push_shader(&mut self, stages: &[(ShaderKind, &[u8])]) {
        let container = crate::assets::ShaderPayload {
            stages: stages.iter().map(|(k, b)| (*k, b.to_vec())).collect(),
        };
        let locator = self.payload(&container.encode().expect("encode shader payload"));
        self.push(Shader {
            locator: Some(locator),
            ..Default::default()
        });
    }

    // The vertex + fragment Shader every renderable world needs.
    fn push_shaders(&mut self) {
        self.push_shader(&[
            (ShaderKind::Vertex, b"vertex-shader-bytes"),
            (ShaderKind::Fragment, b"fragment-shader-bytes"),
        ]);
    }

    // One quad Mesh + a Texture-backed Material + a Prop placing it.
    fn push_textured_quad(&mut self, mesh: AssetId, _tex: AssetId, mat: AssetId, prop: AssetId) {
        let mesh_loc = self.payload(&quad_mesh_payload());
        // Meshes are resources now: the payload rides the resource stream at the
        // next mesh handle; the Prop below references it by that handle, as cook
        // resolves a `.mesh` name. `mesh`'s asset id is unused.
        let _ = mesh;
        let mesh_handle = crate::ecs::MeshHandle(self.mesh_records.len() as u32);
        self.mesh_records
            .push(concinnity_core::ecs::ResourceRecord {
                resource_kind: concinnity_core::ecs::ResourceKind::Mesh as u8,
                handle: mesh_handle.0,
                payload: Some(mesh_loc),
                data_bytes: Vec::new(),
            });
        let tex_loc = self.payload(&texture_payload(2, 2));
        // Textures are resources now: the payload rides the resource stream at
        // the next texture handle. The Material references `TextureHandle(0)` (the
        // first texture) regardless of any texture's asset id.
        let handle = self.texture_records.len() as u32;
        self.texture_records
            .push(concinnity_core::ecs::ResourceRecord {
                resource_kind: concinnity_core::ecs::ResourceKind::Texture as u8,
                handle,
                payload: Some(tex_loc),
                data_bytes: Vec::new(),
            });
        // Materials are a data resource: bake this one into `data_bytes` at the
        // next handle (serialized like cook does), and reference it by that
        // handle. `mat`'s asset id is unused now that the reference is a handle.
        let _ = mat;
        let mat_handle = self.material_records.len() as u32;
        let mat_bytes = postcard::to_allocvec(&Material {
            albedo: Some(TextureHandle(0)),
            ..Default::default()
        })
        .unwrap();
        self.material_records
            .push(concinnity_core::ecs::ResourceRecord {
                resource_kind: concinnity_core::ecs::ResourceKind::Material as u8,
                handle: mat_handle,
                payload: None,
                data_bytes: mat_bytes,
            });
        self.push(Prop {
            asset_id: prop,
            mesh: Some(mesh_handle),
            material: Some(crate::ecs::MaterialHandle(mat_handle)),
            position: [1.0, 2.0, 3.0],
            ..Default::default()
        });
    }

    fn build(mut self) -> TestWorld {
        let mut resources = Resources::new();
        // The renderer reads the shared texture pool from this table, exactly as
        // the runtime does after loading the blob's resource stream.
        resources.insert(crate::resource::TextureTable::from_records(
            &mut self.texture_records,
        ));
        resources.insert(crate::resource::MaterialTable::from_records(
            &mut self.material_records,
        ));
        resources.insert(crate::resource::MeshTable::from_records(
            &mut self.mesh_records,
        ));
        resources.insert(crate::resource::FontTable::from_records(
            &mut self.font_records,
        ));
        resources.insert(crate::resource::ColorLutTable::from_records(
            &mut self.color_lut_records,
        ));
        resources.insert(crate::resource::EnvironmentMapTable::from_records(
            &mut self.env_map_records,
        ));
        resources.insert(crate::resource::SkinnedMeshTable::from_records(
            &mut self.skinned_records,
        ));
        TestWorld {
            components: self.components,
            blob: BlobData::new(vec![Some(self.section)]),
            profile: FrameProfile::default(),
            resources,
        }
    }
}

// A unit quad (two triangles) in the compiled mesh payload format.
fn quad_mesh_payload() -> Vec<u8> {
    let v = |pos: [f32; 3], uv: [f32; 2]| {
        (
            pos,
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.75, 0.74, 0.72],
            uv,
        )
    };
    let vertices = [
        v([0.0, 0.0, 0.0], [0.0, 0.0]),
        v([1.0, 0.0, 0.0], [1.0, 0.0]),
        v([1.0, 0.0, 1.0], [1.0, 1.0]),
        v([0.0, 0.0, 1.0], [0.0, 1.0]),
    ];
    crate::gfx::mesh_payload::serialise(&vertices, &[0, 1, 2, 0, 2, 3])
}

// A w x h mid-gray RGBA image in the compiled texture payload format.
fn texture_payload(w: u32, h: u32) -> Vec<u8> {
    crate::build::texture::serialise(&crate::build::texture::TextureImage::rgba8(
        w,
        h,
        vec![0x7Fu8; (w * h * 4) as usize],
    ))
}

// A representative renderable world: window + config + shaders + camera and
// one textured quad placed by a prop.
fn scene_builder() -> WorldBuilder {
    titled_scene("mock world")
}

// The same representative world with a caller-chosen window title, so a test can
// tell two worlds apart in the captured `InitSnapshot` (the live-swap tests).
fn titled_scene(title: &str) -> WorldBuilder {
    let mut b = WorldBuilder::new();
    b.push(Window {
        title: title.to_string(),
        width: 640,
        height: 360,
        ..Default::default()
    });
    b.push(GraphicsConfig {
        clear_color: [0.1, 0.2, 0.3, 1.0],
        ..Default::default()
    });
    b.push_shaders();
    b.push(Camera3D::bake(Default::default()));
    b.push_textured_quad(MESH, TEX, MAT, PROP);
    b
}

// Run the same pre-init pass World::start performs (Prop decomposition),
// then GraphicsSystem init with the injected hooks. A successful init parks
// the built backend in the world's `ActiveRenderBackend` slot.
fn init_graphics(world: &mut TestWorld, hooks: TestHooks) -> GraphicsSystem {
    let mut gs = GraphicsSystem::new();
    gs.test_hooks = Some(hooks);
    let mut ctx = world.ctx();
    crate::ecs::decompose::run(&mut ctx);
    gs.run_init(&mut ctx);
    gs
}

// Whether the world's parked backend slot currently holds a backend.
fn backend_parked(world: &TestWorld) -> bool {
    world
        .resources
        .get::<crate::ecs::ActiveRenderBackend>()
        .is_some_and(|slot| slot.0.is_some())
}

// One frame the way the schedule runs it: OverlaySystem builds the draw list,
// SpawnSystem applies the entity churn, SettingsSystem applies the settings /
// scene command batches, StreamingSystem drives the streaming pools + publishes
// the camera-relative view, GraphicsSystem submits, then InputSystem publishes
// the FrameInput snapshot. A hard Stop from graphics aborts the tick before
// input, exactly like `World::step`. Fresh overlay / spawn / settings /
// streaming / input instances per step mirror persistent ones here: each reads
// its persistent state / cursors from parked resources, and these tests never
// leave a drained event in retention across a second step.
fn step(gs: &mut GraphicsSystem, world: &mut TestWorld) -> StepResult {
    use crate::ecs::System;
    let mut ctx = world.ctx();
    crate::gfx::overlay::OverlaySystem::new().step(&mut ctx);
    crate::spawn::SpawnSystem::new().step(&mut ctx);
    crate::gfx::settings_system::SettingsSystem::new().step(&mut ctx);
    crate::gfx::streaming_system::StreamingSystem::new().step(&mut ctx);
    let result = gs.run_step(&mut ctx);
    if result != StepResult::Stop {
        crate::gfx::input_system::InputSystem::new().step(&mut ctx);
    }
    result
}

fn lock(state: &Arc<Mutex<MockState>>) -> std::sync::MutexGuard<'_, MockState> {
    state.lock().unwrap()
}

#[test]
fn init_builds_draw_list_and_render_handles() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed, "init must succeed");
    assert!(backend_parked(&world), "mock backend parked in the slot");

    let s = lock(&state);
    let init = s.init.as_ref().expect("factory captured the BackendInit");
    assert_eq!(init.window_width, 640);
    assert_eq!(init.window_height, 360);
    assert_eq!(init.window_title, "mock world");
    assert_eq!(init.clear_color, [0.1, 0.2, 0.3, 1.0]);
    assert_eq!(init.frames_in_flight, 2);
    // One prop placing one mesh = one draw object over the quad geometry.
    assert_eq!(init.draw_objects.len(), 1);
    assert_eq!(init.vertex_count, 4);
    assert_eq!(init.index_count, 6);
    let draw = &init.draw_objects[0];
    assert_eq!(draw.index_count, 6);
    assert_eq!(draw.texture_slot, 0, "material albedo resolves to slot 0");
    assert!(draw.visible);
    // The prop's translation is baked into the draw's model matrix.
    assert_eq!(draw.model[3][0], 1.0);
    assert_eq!(draw.model[3][1], 2.0);
    assert_eq!(draw.model[3][2], 3.0);
    assert_eq!(init.texture_count, 1);
    assert_eq!(init.text_atlas_count, 0, "no fonts or sprite textures");
    assert_eq!(init.n_skinned, 0);
    assert_eq!(init.instanced_cluster_count, 0);
    assert!(init.scene_required, "a mesh world requires the scene chain");
    assert!(!init.fog, "no VolumetricFog declared");
    drop(s);

    // The prop entity received its GPU handle + init world matrix.
    let ctx = world.ctx();
    let handles: Vec<Vec<u32>> = ctx
        .query::<RenderHandle>()
        .map(|h| h.draws.clone())
        .collect();
    assert_eq!(handles, vec![vec![0]]);
    let globals: Vec<[[f32; 4]; 4]> = ctx
        .query::<crate::assets::GlobalTransform>()
        .map(|g| g.0)
        .collect();
    assert_eq!(globals.len(), 1);
    assert_eq!(globals[0][3][0], 1.0);
}

// Regression: init parks exactly one OverlayAssets carrying the captured HUD
// chip ids. A duplicated park block once re-took the already-moved fields
// (`std::mem::take`), so the second park overwrote the resource with empty
// defaults and the whole overlay -- HUD chips and menu alike -- silently drew
// nothing. The chips surviving to the parked resource guards that.
#[test]
fn init_parks_overlay_assets_with_the_hud_chips() {
    use crate::assets::{DebugHud, StatHud};
    use crate::gfx::overlay::OverlayAssets;

    let (_state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(StatHud {
        fps_label: Some(AssetId(10)),
        vram_label: Some(AssetId(11)),
        ..Default::default()
    });
    b.push(DebugHud {
        mouse_label: Some(AssetId(20)),
        passes_label: Some(AssetId(21)),
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed, "init must succeed");

    let overlay = world
        .resources
        .get::<OverlayAssets>()
        .expect("OverlayAssets parked at init");
    assert_eq!(
        overlay.stat_hud_chips,
        vec![AssetId(10), AssetId(11)],
        "StatHud chip ids survive to the parked OverlayAssets"
    );
    // DebugHud strip order is mouse, camera, sys, passes; only mouse + passes
    // were set here.
    assert_eq!(
        overlay.debug_hud_chips,
        vec![AssetId(20), AssetId(21)],
        "DebugHud chip ids survive to the parked OverlayAssets"
    );
}

// The built backend can be taken exactly once (the `cn editor` transplant).
#[test]
fn take_backend_yields_the_backend_once() {
    let (_state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let _gs = init_graphics(&mut world, hooks);
    assert!(
        crate::ecs::ActiveRenderBackend::take(&mut world.resources).is_some(),
        "the built backend is taken"
    );
    assert!(
        crate::ecs::ActiveRenderBackend::take(&mut world.resources).is_none(),
        "and only once"
    );
}

// A world carrying a transplanted backend whose swapchain config matches reuses
// it (calls reload_world) instead of building a fresh one, and the reused
// backend ends up carrying the new world's content -- proving the same instance
// was swapped over (the `cn editor` live SAVE hot path).
#[test]
fn pending_backend_reuses_instance_and_ends_with_new_world() {
    // World A builds its own backend (the factory records into state_a).
    let (state_a, hooks_a) = recording_hooks();
    let mut world_a = titled_scene("world A").build();
    let _gs_a = init_graphics(&mut world_a, hooks_a);
    let backend_a = crate::ecs::ActiveRenderBackend::take(&mut world_a.resources)
        .expect("world A built a backend");

    // Transplant it into world B (a different world, same swapchain config).
    let (state_b, hooks_b) = recording_hooks();
    let mut world_b = titled_scene("world B").build();
    world_b
        .resources
        .insert(crate::ecs::PendingBackend(backend_a));
    let gs_b = init_graphics(&mut world_b, hooks_b);

    assert!(!gs_b.failed);
    assert!(backend_parked(&world_b), "the reused backend is installed");
    let a = lock(&state_a);
    assert!(
        a.saw(&Call::ReloadWorld),
        "the transplanted backend was reloaded, not rebuilt"
    );
    assert_eq!(
        a.init.as_ref().unwrap().window_title,
        "world B",
        "the reused backend now carries the new world's content"
    );
    drop(a);
    assert!(
        state_b.lock().unwrap().init.is_none(),
        "world B never built a fresh backend"
    );
}

// A transplanted backend whose swapchain config differs (frames-in-flight, HDR)
// cannot reuse the window: it is idled + dropped and a fresh backend is built.
#[test]
fn pending_backend_swapchain_change_forces_full_rebuild() {
    let (state_t, _unused) = recording_hooks();
    let transplant = MockBackend::transplant(
        Arc::clone(&state_t),
        Some(SwapchainConfig {
            // The scene world uses the default 2 frames-in-flight; 3 mismatches.
            frames_in_flight: 3,
            hdr_display: false,
            hdr_pq: false,
        }),
    );
    let (state_b, hooks_b) = recording_hooks();
    let mut world = scene_builder().build();
    world
        .resources
        .insert(crate::ecs::PendingBackend(Box::new(transplant)));
    let gs = init_graphics(&mut world, hooks_b);

    assert!(!gs.failed);
    assert!(backend_parked(&world));
    let t = lock(&state_t);
    assert!(
        t.saw(&Call::WaitIdle),
        "the mismatched backend is idled before it is dropped"
    );
    assert!(
        !t.saw(&Call::ReloadWorld),
        "no reload is attempted across a swapchain change"
    );
    drop(t);
    assert!(
        state_b.lock().unwrap().init.is_some(),
        "a fresh backend is built for the changed swapchain"
    );
}

// The HDR axis of the swapchain identity is honoured too: a transplant whose
// hdr_display request differs from the new world's forces a full rebuild rather
// than an in-place reload (the pixel format would otherwise diverge).
#[test]
fn pending_backend_hdr_change_forces_full_rebuild() {
    let (state_t, _unused) = recording_hooks();
    let transplant = MockBackend::transplant(
        Arc::clone(&state_t),
        Some(SwapchainConfig {
            // The scene world requests no HDR (PostProcessConfig default); an
            // HDR-requesting transplant is a swapchain change.
            frames_in_flight: 2,
            hdr_display: true,
            hdr_pq: false,
        }),
    );
    let (state_b, hooks_b) = recording_hooks();
    let mut world = scene_builder().build();
    world
        .resources
        .insert(crate::ecs::PendingBackend(Box::new(transplant)));
    let gs = init_graphics(&mut world, hooks_b);

    assert!(!gs.failed);
    let t = lock(&state_t);
    assert!(
        t.saw(&Call::WaitIdle),
        "the HDR-mismatched backend is idled"
    );
    assert!(!t.saw(&Call::ReloadWorld), "no reload across an HDR change");
    drop(t);
    assert!(
        state_b.lock().unwrap().init.is_some(),
        "a fresh backend is built for the changed HDR output"
    );
}

// A transplanted backend that reports hot-swap-capable but fails the reload
// leaves GraphicsSystem failed (no silent fallback that loses the edit).
#[test]
fn reload_world_failure_marks_graphics_failed() {
    let (state_a, hooks_a) = recording_hooks();
    let mut world_a = scene_builder().build();
    let _gs_a = init_graphics(&mut world_a, hooks_a);
    let backend_a = crate::ecs::ActiveRenderBackend::take(&mut world_a.resources).unwrap();
    state_a.lock().unwrap().fail_reload = Some("boom".to_string());

    let (_state_b, hooks_b) = recording_hooks();
    let mut world_b = scene_builder().build();
    world_b
        .resources
        .insert(crate::ecs::PendingBackend(backend_a));
    let gs_b = init_graphics(&mut world_b, hooks_b);

    assert!(gs_b.failed, "a failed reload marks the system failed");
    assert!(!backend_parked(&world_b));
    assert!(state_a.lock().unwrap().saw(&Call::ReloadWorld));
}

#[test]
fn init_pushes_startup_backend_state() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let _gs = init_graphics(&mut world, hooks);

    let s = lock(&state);
    // A camera with no UI: plain first-person capture, menu mode off.
    assert!(s.saw(&Call::SetMenuMode(false)));
    assert!(s.saw(&Call::CaptureCursor));
    // The ambient scale + keymap are pushed once after construction.
    assert!(s.saw(&Call::SetAmbientIntensity(1.0)));
    assert!(s.saw(&Call::SetKeymap));
    // Reflection-probe placements are pushed once (auto-seed or empty).
    assert!(
        s.calls
            .iter()
            .any(|c| matches!(c, Call::SetReflectionProbes(_)))
    );
    // Windowed start: no window-mode override is applied.
    assert!(!s.calls.iter().any(|c| matches!(c, Call::SetWindowMode(_))));
}

// A menu / editor driver present at init (the editor's live-preview rebuild seeds
// a `MenuOverride` before re-running init) suppresses the first-person startup
// grab: the per-frame drive owns capture, and re-grabbing on every rebuild would
// re-hide and decouple the OS cursor, desyncing the free-cursor handoff.
#[test]
fn menu_driven_init_skips_the_first_person_cursor_grab() {
    let (state, hooks) = recording_hooks();
    // A plain first-person world (Camera3D, no HitRegion / KeyBinding) -- the same
    // world that grabs at startup above -- but with a driver already in control.
    let mut world = scene_builder().build();
    world.resources.insert(crate::ecs::MenuOverride(Some(true)));
    let _gs = init_graphics(&mut world, hooks);

    let s = lock(&state);
    assert!(
        !s.saw(&Call::CaptureCursor),
        "a menu/editor driver owns capture; init must not auto-grab"
    );
    // Menu mode still reflects the world's own UI (none here), unchanged.
    assert!(s.saw(&Call::SetMenuMode(false)));
}

#[test]
fn ui_only_world_trims_scene_features() {
    let (state, hooks) = recording_hooks();
    let mut b = WorldBuilder::new();
    b.push(Window::default());
    b.push(GraphicsConfig::default());
    b.push_shaders();
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed);
    let s = lock(&state);
    let init = s.init.as_ref().unwrap();
    assert!(init.draw_objects.is_empty());
    assert!(!init.scene_required, "no 3D content: scene chain trimmed");
    assert_eq!(init.shadows.map_size, 0, "shadows trimmed with the scene");
    assert!(!init.taa_enabled);
    assert!(!init.ssao_on, "screen-space effects trimmed with the scene");
    // No camera: the cursor is never captured.
    assert!(!s.saw(&Call::CaptureCursor));
}

#[test]
fn persisted_settings_override_authored_config() {
    let mut settings = crate::config::Settings::default();
    settings.graphics.quality_preset = Some(QualityPreset::Custom);
    settings.graphics.vsync = Some(true);
    settings.graphics.fps_cap = Some(30);
    settings.graphics.frames_in_flight = Some(3);
    settings.graphics.shadow_map_size = Some(1024);
    let (state, hooks) = recording_hooks_with(settings, GpuProfile::UNKNOWN);
    let mut world = scene_builder().build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed);
    assert_eq!(gs.fps_cap, 30, "persisted cap overrides the world's 0");
    assert_eq!(gs.quality_preset, QualityPreset::Custom);
    let s = lock(&state);
    let init = s.init.as_ref().unwrap();
    assert!(init.vsync, "persisted vsync overrides the authored false");
    assert_eq!(init.frames_in_flight, 3);
    assert_eq!(init.shadows.map_size, 1024, "explicit override wins");
}

#[test]
fn low_preset_ceiling_clamps_quality_knobs() {
    let mut settings = crate::config::Settings::default();
    settings.graphics.quality_preset = Some(QualityPreset::Low);
    let (state, hooks) = recording_hooks_with(settings, GpuProfile::UNKNOWN);
    let mut world = scene_builder().build();
    let gs = init_graphics(&mut world, hooks);

    // The authored GraphicsConfig defaults (2048 / 80 / 4 / 8) are clamped
    // under the Low ceiling; the authored baselines are kept for a later
    // preset up-shift.
    let s = lock(&state);
    let init = s.init.as_ref().unwrap();
    assert_eq!(init.shadows.map_size, 1024);
    assert_eq!(init.shadows.distance, 40);
    assert_eq!(init.shadows.cascades, 2);
    assert_eq!(init.anisotropy, 4);
    assert_eq!(gs.authored_shadow_map_size, 2048);
    assert_eq!(gs.authored_shadow_distance, 80);
    assert_eq!(gs.authored_shadow_cascades, 4);
    assert_eq!(gs.authored_anisotropy, 8);
}

#[test]
fn auto_preset_resolves_ceiling_from_gpu_tier() {
    // An integrated-tier GPU under the (first-launch) Auto preset takes the
    // Low ceiling; the injected hooks also skip the first-launch persist, so
    // the developer's settings file is never written.
    let profile = GpuProfile {
        vendor: GpuVendor::Intel,
        tier: GpuTier::Integrated,
        memory_budget_bytes: 0,
        unified_memory: false,
        discrete: false,
    };
    let (state, hooks) = recording_hooks_with(crate::config::Settings::default(), profile);
    let mut world = scene_builder().build();
    let gs = init_graphics(&mut world, hooks);

    assert_eq!(gs.quality_preset, QualityPreset::Auto);
    let s = lock(&state);
    assert_eq!(s.init.as_ref().unwrap().shadows.map_size, 1024);
}

#[test]
fn missing_shader_fails_init() {
    let (state, hooks) = recording_hooks();
    let mut b = WorldBuilder::new();
    b.push(Window::default());
    b.push(GraphicsConfig::default());
    b.push_textured_quad(MESH, TEX, MAT, PROP);
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(gs.failed, "no Shader: init must fail");
    assert!(!backend_parked(&world));
    assert!(lock(&state).init.is_none(), "backend never constructed");
}

#[test]
fn first_declared_scene_applies_start_visibility() {
    let scene_a = AssetId(20);
    let scene_b = AssetId(21);
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    // A second prop assigned to scene B; the first prop joins scene A.
    b.push_textured_quad(AssetId(5), AssetId(6), AssetId(7), AssetId(8));
    let mut world = b.build();
    {
        // Assign scenes on the two props before decomposition maps them to
        // SceneMember components.
        let mut ctx = world.ctx();
        for prop in ctx.query_mut::<Prop>() {
            prop.scene = Some(if prop.asset_id == PROP {
                scene_a
            } else {
                scene_b
            });
        }
        ctx.push(Scene {
            asset_id: scene_a,
            camera_shot: None,
        });
        ctx.push(Scene {
            asset_id: scene_b,
            camera_shot: None,
        });
    }
    let mut gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    {
        let s = lock(&state);
        assert_eq!(s.visibility.get(&0), Some(&true), "scene A prop visible");
        assert_eq!(s.visibility.get(&1), Some(&false), "scene B prop hidden");
    }

    // An imperative jump to scene B flips both.
    {
        let mut ctx = world.ctx();
        ctx.events_mut::<SceneCommand>().send(SceneCommand {
            scene: scene_b,
            transition: "Cut".to_string(),
        });
    }
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    let s = lock(&state);
    assert_eq!(s.visibility.get(&0), Some(&false));
    assert_eq!(s.visibility.get(&1), Some(&true));
}

#[test]
fn frame_steps_draw_and_publish_input() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    for _ in 0..2 {
        assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    }
    // take_input drains the snapshot per poll, so seed the third frame's
    // input just before the step that should publish it.
    lock(&state).next_input.mouse_x = 33.0;
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    assert_eq!(gs.frame_count, 3);

    let s = lock(&state);
    assert_eq!(s.draw_frames(), 3);
    assert_eq!(
        s.calls
            .iter()
            .filter(|c| matches!(c, Call::UpdateView(_)))
            .count(),
        3
    );
    assert_eq!(s.calls.iter().filter(|c| **c == Call::TakeInput).count(), 3);
    drop(s);

    // The polled input snapshot is republished as the FrameInput component +
    // resource for the camera / UI systems.
    let ctx = world.ctx();
    let inputs: Vec<&FrameInput> = ctx.query::<FrameInput>().collect();
    assert_eq!(inputs.len(), 1, "previous snapshot drained, one deposited");
    let res = ctx.resource::<FrameInput>().expect("resource published");
    assert_eq!(res.mouse_x, 33.0);
    assert_eq!(res.viewport, [1280.0, 720.0]);
    assert!(
        ctx.resource::<crate::ecs::MenuActive>().is_some(),
        "menu state published every frame"
    );
}

#[test]
fn camera_state_reaches_the_backend_each_frame() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    step(&mut gs, &mut world);
    // Move the camera the way Camera3DSystem would (write position + view).
    let moved_view = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-7.0, -8.0, -9.0, 1.0],
    ];
    {
        let mut ctx = world.ctx();
        let cam = ctx.query_mut::<Camera3D>().next().unwrap();
        cam.position = [7.0, 8.0, 9.0];
        cam.view_matrix = moved_view;
    }
    step(&mut gs, &mut world);

    let s = lock(&state);
    match s.last_draw_frame() {
        Some(Call::DrawFrame { cam_pos, .. }) => assert_eq!(cam_pos, [7.0, 8.0, 9.0]),
        other => panic!("expected a DrawFrame, got {other:?}"),
    }
    assert!(
        s.saw(&Call::UpdateView(moved_view)),
        "the moved view matrix reaches update_view"
    );
}

#[test]
fn transform_edit_pushes_new_model_matrix() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    {
        let mut ctx = world.ctx();
        let t = ctx.query_mut::<Transform>().next().unwrap();
        t.position = [10.0, 20.0, 30.0];
    }
    step(&mut gs, &mut world);

    let s = lock(&state);
    assert!(s.saw(&Call::UpdateModel(0)));
    let model = s.models.get(&0).expect("slot 0 model pushed");
    assert_eq!(model[3][0], 10.0);
    assert_eq!(model[3][1], 20.0);
    assert_eq!(model[3][2], 30.0);
}

#[test]
fn opaque_menu_backdrop_hides_world_and_freezes_gameplay_input() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    // UI presence (a HitRegion) flips the world into menu mode at init.
    b.push(HitRegion::default());
    // A visible, opaque, view-owned sprite spanning the whole 1280x720
    // reference canvas: a menu dim at full alpha.
    b.push(Sprite {
        asset_id: AssetId(40),
        x: 0.0,
        y: 0.0,
        width: 1280.0,
        height: 720.0,
        tint: [0.0, 0.0, 0.0, 1.0],
        visible: true,
        screen: Some(AssetId(41)),
        ..Default::default()
    });
    let mut world = b.build();
    // The active-screen state UiInputSystem publishes when a world-pausing
    // screen (id 41) is open; the overlay derives menu_active from it.
    world.ctx().insert_resource(crate::ecs::ScreenStack {
        layers: [(AssetId(41), 1)].into_iter().collect(),
        pauses_world: true,
        captures_input: true,
    });
    let mut gs = init_graphics(&mut world, hooks);
    assert!(lock(&state).saw(&Call::SetMenuMode(true)));

    lock(&state).next_input.forward = true;
    step(&mut gs, &mut world);

    {
        let s = lock(&state);
        match s.last_draw_frame() {
            Some(Call::DrawFrame { world_hidden, .. }) => {
                assert!(world_hidden, "opaque full-canvas backdrop skips the world");
            }
            other => panic!("expected a DrawFrame, got {other:?}"),
        }
        // Menu open: the camera capture is released each frame.
        assert!(s.saw(&Call::SetCameraCapture(false)));
    }
    {
        let ctx = world.ctx();
        // The App-level pacer clamps from this same resource next step.
        assert!(ctx.resource::<crate::ecs::MenuActive>().unwrap().0);
        let input = ctx.resource::<FrameInput>().unwrap();
        assert!(!input.forward, "gameplay input frozen behind the menu");
    }

    // Dimming the backdrop below full alpha keeps the menu active but the
    // world visible again.
    {
        let mut ctx = world.ctx();
        let sprite = ctx.query_mut::<Sprite>().next().unwrap();
        sprite.tint[3] = 0.5;
    }
    step(&mut gs, &mut world);
    let s = lock(&state);
    match s.last_draw_frame() {
        Some(Call::DrawFrame { world_hidden, .. }) => {
            assert!(!world_hidden, "translucent dim keeps the world render");
        }
        other => panic!("expected a DrawFrame, got {other:?}"),
    }
}

// A `MenuOverride(Some(true))` forces edit mode on a plain first-person world
// (no menu UI): the backend is put into menu mode, the cursor is released, the
// freeze resource is set, and gameplay input is frozen -- everything a menu open
// would do, without the world declaring one.
#[test]
fn menu_override_true_forces_cursor_free_and_freezes_input() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);
    // A plain camera world starts in first-person capture, menu mode off.
    assert!(lock(&state).saw(&Call::SetMenuMode(false)));

    world.resources.insert(crate::ecs::MenuOverride(Some(true)));
    lock(&state).next_input.forward = true;
    step(&mut gs, &mut world);

    let s = lock(&state);
    assert!(
        s.saw(&Call::SetMenuMode(true)),
        "override forces backend menu mode"
    );
    assert!(
        s.saw(&Call::SetCameraCapture(false)),
        "override releases the cursor"
    );
    drop(s);
    let ctx = world.ctx();
    assert!(
        ctx.resource::<crate::ecs::MenuActive>().unwrap().0,
        "freeze resource set"
    );
    assert!(
        !ctx.resource::<FrameInput>().unwrap().forward,
        "gameplay input frozen in edit mode"
    );
}

// A `MenuOverride(Some(false))` forces play mode: the backend stays in menu mode
// (so an editor session's clicks still route sanely), but the cursor is captured
// and gameplay input runs -- the world plays behind the editor.
#[test]
fn menu_override_false_captures_cursor_and_runs_input() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    world
        .resources
        .insert(crate::ecs::MenuOverride(Some(false)));
    lock(&state).next_input.forward = true;
    step(&mut gs, &mut world);

    let s = lock(&state);
    assert!(
        s.saw(&Call::SetCameraCapture(true)),
        "override captures the cursor"
    );
    drop(s);
    let ctx = world.ctx();
    assert!(
        !ctx.resource::<crate::ecs::MenuActive>().unwrap().0,
        "not frozen"
    );
    assert!(
        ctx.resource::<FrameInput>().unwrap().forward,
        "gameplay input runs in play mode"
    );
}

#[test]
fn window_close_stops_after_wait_idle() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    lock(&state).window_closed = true;
    assert_eq!(step(&mut gs, &mut world), StepResult::Stop);
    let s = lock(&state);
    assert!(s.saw(&Call::WaitIdle));
    assert_eq!(s.draw_frames(), 0, "no frame drawn after the close");
}

#[test]
fn draw_frame_error_stops_the_loop() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    lock(&state).fail_draw = Some("device lost".to_string());
    assert_eq!(step(&mut gs, &mut world), StepResult::Stop);
    assert!(lock(&state).saw(&Call::WaitIdle));
}

// Reaching the cap must Stop rather than report Done: `World::step` treats Done
// as "retire this system and carry on with the others", which would leave a real
// world running headlessly forever once the renderer removed itself.
#[test]
fn max_frames_stops_the_run() {
    let (_state, hooks) = recording_hooks();
    let mut b = WorldBuilder::new();
    b.push(Window::default());
    b.push(GraphicsConfig {
        max_frames: Some(2),
        ..Default::default()
    });
    b.push_shaders();
    b.push(Camera3D::bake(Default::default()));
    b.push_textured_quad(MESH, TEX, MAT, PROP);
    let mut world = b.build();
    let mut gs = init_graphics(&mut world, hooks);

    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    assert_eq!(step(&mut gs, &mut world), StepResult::Stop);
}

// A launch-imposed cap (the `cn export` shader-warm pass) overrides whatever the
// world asked for, including a world that set no cap at all and would otherwise
// run until its window closed.
#[test]
fn a_launch_frame_cap_overrides_the_world() {
    let (_state, hooks) = recording_hooks();
    let mut b = WorldBuilder::new();
    b.push(Window::default());
    b.push(GraphicsConfig::default());
    b.push_shaders();
    b.push(Camera3D::bake(Default::default()));
    b.push_textured_quad(MESH, TEX, MAT, PROP);
    let mut world = b.build();

    crate::app::dev_flags::set_max_frames(Some(1));
    let mut gs = init_graphics(&mut world, hooks);
    let result = step(&mut gs, &mut world);
    crate::app::dev_flags::set_max_frames(None);
    assert_eq!(result, StepResult::Stop);
}

#[test]
fn spawn_request_clones_template_draw_slot() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    {
        let mut ctx = world.ctx();
        ctx.events_mut::<SpawnRequest>().send(SpawnRequest {
            template: PROP,
            name: Some(AssetId(900)),
            transform: Transform {
                position: [5.0, 0.0, 0.0],
                rotation_deg: [0.0; 3],
                scale: [1.0; 3],
            },
            lifetime_secs: None,
        });
    }
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);

    assert!(lock(&state).saw(&Call::CloneStaticDrawObject { src: 0, new_idx: 1 }));
    let ctx = world.ctx();
    let mut handles: Vec<Vec<u32>> = ctx
        .query::<RenderHandle>()
        .map(|h| h.draws.clone())
        .collect();
    handles.sort();
    assert_eq!(handles, vec![vec![0], vec![1]], "spawned copy owns slot 1");
}

#[test]
fn visibility_request_switches_slots_and_hidden_tag() {
    use crate::assets::{Hidden, VisibilityRequest};

    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    {
        let mut ctx = world.ctx();
        ctx.events_mut::<VisibilityRequest>()
            .send(VisibilityRequest {
                name: PROP,
                visible: false,
            });
    }
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    assert!(lock(&state).saw(&Call::UpdateVisibility {
        draw_idx: 0,
        visible: false
    }));
    assert_eq!(
        world.ctx().query::<Hidden>().count(),
        1,
        "hide tags the entity"
    );

    {
        let mut ctx = world.ctx();
        ctx.events_mut::<VisibilityRequest>()
            .send(VisibilityRequest {
                name: PROP,
                visible: true,
            });
    }
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    assert!(lock(&state).saw(&Call::UpdateVisibility {
        draw_idx: 0,
        visible: true
    }));
    assert_eq!(
        world.ctx().query::<Hidden>().count(),
        0,
        "show clears the tag"
    );
}

#[test]
fn despawn_request_retires_draw_slots() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    {
        let mut ctx = world.ctx();
        ctx.events_mut::<DespawnRequest>()
            .send(DespawnRequest { name: PROP });
    }
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);

    assert!(lock(&state).saw(&Call::RetireDrawObject(0)));
    let ctx = world.ctx();
    assert_eq!(
        ctx.query::<RenderHandle>().count(),
        0,
        "despawned entity gone"
    );
}

// The streaming pools graphics init builds are parked in the `StreamingState`
// resource (StreamingSystem drives them each frame). Read its per-pool stats
// for the streaming assertions.
fn streaming_stats(world: &TestWorld) -> crate::gfx::streaming_system::StreamingStats {
    world
        .resources
        .get::<crate::gfx::streaming_system::StreamingState>()
        .expect("StreamingState parked at init")
        .streaming_stats()
}

#[test]
fn streaming_init_evicts_streamable_slots() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(StreamingConfig::default());
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    // The streamers were handed off to the parked StreamingState, so the
    // system's own scratch fields are cleared.
    assert!(gs.texture_streamer.is_none());
    assert!(gs.mesh_streamer.is_none());
    let stats = streaming_stats(&world);
    assert!(stats.texture.is_some(), "texture pool streaming");
    assert!(stats.mesh.is_some(), "mesh pool streaming");

    let s = lock(&state);
    // The albedo pool slot is evicted to a placeholder at init; the mesh
    // keeps its build-time region (cap covers the whole set) but is evicted
    // so the streamer brings it back nearest-first.
    assert!(s.saw(&Call::EvictTextureSlot(0)));
    assert!(s.saw(&Call::EvictMesh(0)));
}

#[test]
fn texture_streaming_uploads_evicted_slots() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(StreamingConfig::default());
    let mut world = b.build();
    let mut gs = init_graphics(&mut world, hooks);

    // The streamer decodes on a worker thread; step until the completed
    // upload lands (bounded so a regression fails rather than hangs).
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
        if lock(&state).calls.iter().any(|c| {
            matches!(
                c,
                Call::UpdateTextureSlot {
                    slot: 0,
                    w: 2,
                    h: 2
                }
            )
        }) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "texture never streamed back in: {:?}",
            lock(&state).calls
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    let (resident, pending, unloaded) = streaming_stats(&world).texture.unwrap();
    assert_eq!((resident, pending, unloaded), (1, 0, 0));
}

#[test]
fn mesh_streaming_reuploads_evicted_geometry() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(StreamingConfig::default());
    let mut world = b.build();
    let mut gs = init_graphics(&mut world, hooks);

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
        if lock(&state).calls.iter().any(|c| {
            matches!(
                c,
                Call::UploadMesh {
                    draw_idx: 0,
                    vertices: 4,
                    indices: 6,
                }
            )
        }) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "mesh never streamed back in: {:?}",
            lock(&state).calls
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    let (resident, pending, unloaded) = streaming_stats(&world).mesh.unwrap();
    assert_eq!((resident, pending, unloaded), (1, 0, 0));
}

// A VoxelWorld rebases the draw's view + camera onto the chunk render origin:
// StreamingSystem publishes the camera-relative pair and GraphicsSystem submits
// it. Exercises the view-rebase timing across the StreamingSystem/GraphicsSystem
// split (a non-voxel world leaves the absolute view untouched -- see
// `camera_state_reaches_the_backend_each_frame`). No chunk needs to be resident:
// the rebase is a function of the camera position alone.
#[test]
fn voxel_world_rebases_the_draw_view_onto_the_chunk_origin() {
    use crate::assets::VoxelWorld;
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    // Default chunk is 16x16 world units; a small view radius keeps the chunk
    // dispatch light (no chunk is awaited).
    b.push(VoxelWorld {
        seed: 1,
        view_radius: 1,
        ..Default::default()
    });
    let mut world = b.build();
    let mut gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);
    // The chunk pool was handed to StreamingSystem's parked state.
    assert!(
        streaming_stats(&world).chunk.is_some(),
        "chunk pool streaming"
    );

    // Place the camera two chunks east and three chunks north of the origin
    // (floor(40/16)=2, floor(-40/16)=-3), so the rebase is non-trivial.
    let cam_pos = [40.0, 5.0, -40.0];
    let view = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-40.0, -5.0, 40.0, 1.0],
    ];
    {
        let mut ctx = world.ctx();
        let cam = ctx.query_mut::<Camera3D>().next().unwrap();
        cam.position = cam_pos;
        cam.view_matrix = view;
    }
    step(&mut gs, &mut world);

    // Render origin = chunk (2, -3) -> world (32, -48).
    let origin = [32.0, 0.0, -48.0];
    let expected_cam = [8.0, 5.0, 8.0];
    let expected_view = crate::gfx::chunk_coord::camera_relative_view(view, cam_pos, origin);

    let s = lock(&state);
    match s.last_draw_frame() {
        Some(Call::DrawFrame { cam_pos: cp, .. }) => {
            assert_eq!(
                cp, expected_cam,
                "draw camera rebased onto the chunk origin"
            );
            assert_ne!(cp, cam_pos, "the rebase actually moved the camera");
        }
        other => panic!("expected a DrawFrame, got {other:?}"),
    }
    assert!(
        s.saw(&Call::UpdateView(expected_view)),
        "the camera-relative view reaches update_view, not the absolute one"
    );
    assert!(
        !s.saw(&Call::UpdateView(view)),
        "the absolute view is not what the backend received"
    );
}

// Drive a persistent SpawnSystem the way the schedule does -- against the
// world's parked backend -- without the rest of the frame, so a test can seed
// the world clock's inputs (Lifetime / Spawner / MenuActive) and observe just
// the churn.
fn spawn_step(spawn: &mut crate::spawn::SpawnSystem, world: &mut TestWorld) {
    use crate::ecs::System;
    let mut ctx = world.ctx();
    spawn.step(&mut ctx);
}

// The entity a name resolves to through the decomposition's name index.
fn entity_named(world: &mut TestWorld, name: AssetId) -> crate::ecs::Entity {
    *world
        .ctx()
        .resource::<crate::ecs::decompose::EntityByName>()
        .expect("name index published at load")
        .0
        .get(&name)
        .expect("name resolves to an entity")
}

// With no parked backend (graphics failed, or the editor transplanted it away)
// there are no draw slots to retire or clone, so the churn waits rather than
// dropping the events on the floor.
#[test]
fn spawn_system_without_a_backend_leaves_the_churn_pending() {
    let (_state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let _gs = init_graphics(&mut world, hooks);
    crate::ecs::ActiveRenderBackend::take(&mut world.resources).expect("backend was parked");

    {
        let mut ctx = world.ctx();
        ctx.events_mut::<DespawnRequest>()
            .send(DespawnRequest { name: PROP });
    }
    let mut spawn = crate::spawn::SpawnSystem::new();
    spawn_step(&mut spawn, &mut world);

    assert_eq!(
        world.ctx().query::<RenderHandle>().count(),
        1,
        "the despawn never ran without a backend to retire its slot"
    );
}

// A ReparentRequest naming both ends re-points the child's Parent edge and
// recomposes its world matrix under the new parent.
#[test]
fn reparent_request_repoints_the_child_under_the_named_parent() {
    const OTHER: AssetId = AssetId(8);
    let (_state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push_textured_quad(AssetId(5), AssetId(6), AssetId(7), OTHER);
    let mut world = b.build();
    let _gs = init_graphics(&mut world, hooks);
    let child = entity_named(&mut world, OTHER);
    let parent = entity_named(&mut world, PROP);

    {
        let mut ctx = world.ctx();
        ctx.events_mut::<ReparentRequest>().send(ReparentRequest {
            child: OTHER,
            parent: Some(PROP),
        });
    }
    let mut spawn = crate::spawn::SpawnSystem::new();
    spawn_step(&mut spawn, &mut world);

    let ctx = world.ctx();
    assert_eq!(
        ctx.get::<crate::assets::Parent>(child).map(|p| p.0),
        Some(parent),
        "the child hangs off the named parent"
    );
    // Both quads are placed at (1, 2, 3), so the child's world position is now
    // the parent's translation applied twice.
    let global = ctx
        .get::<crate::assets::GlobalTransform>(child)
        .expect("child keeps a world matrix");
    assert_eq!(
        [global.0[3][0], global.0[3][1], global.0[3][2]],
        [2.0, 4.0, 6.0],
        "the child's world matrix recomposed under the parent chain"
    );
}

// Regression guard: a ReparentRequest naming a parent that does NOT resolve is
// skipped entirely. Silently falling through to `None` would detach the child to
// a world root, so a typo'd parent name would teleport the child instead of
// doing nothing.
#[test]
fn reparent_request_with_an_unresolved_parent_is_skipped() {
    const OTHER: AssetId = AssetId(8);
    const GHOST: AssetId = AssetId(900);
    let (_state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push_textured_quad(AssetId(5), AssetId(6), AssetId(7), OTHER);
    let mut world = b.build();
    let _gs = init_graphics(&mut world, hooks);
    let child = entity_named(&mut world, OTHER);
    let parent = entity_named(&mut world, PROP);

    // Park the child under a real parent first, so a wrongful detach shows.
    {
        let mut ctx = world.ctx();
        crate::gfx::draw_list::reparent(&mut ctx, child, Some(parent));
        ctx.events_mut::<ReparentRequest>().send(ReparentRequest {
            child: OTHER,
            parent: Some(GHOST),
        });
    }
    let mut spawn = crate::spawn::SpawnSystem::new();
    spawn_step(&mut spawn, &mut world);

    assert_eq!(
        world.ctx().get::<crate::assets::Parent>(child).map(|p| p.0),
        Some(parent),
        "an unresolved parent name never detaches the child to a root"
    );
}

// A ReparentRequest with no parent named at all IS a detach: the child becomes a
// root and its world matrix falls back to its own transform.
#[test]
fn reparent_request_without_a_parent_detaches_the_child() {
    const OTHER: AssetId = AssetId(8);
    let (_state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push_textured_quad(AssetId(5), AssetId(6), AssetId(7), OTHER);
    let mut world = b.build();
    let _gs = init_graphics(&mut world, hooks);
    let child = entity_named(&mut world, OTHER);
    let parent = entity_named(&mut world, PROP);

    {
        let mut ctx = world.ctx();
        crate::gfx::draw_list::reparent(&mut ctx, child, Some(parent));
        ctx.events_mut::<ReparentRequest>().send(ReparentRequest {
            child: OTHER,
            parent: None,
        });
    }
    let mut spawn = crate::spawn::SpawnSystem::new();
    spawn_step(&mut spawn, &mut world);

    let ctx = world.ctx();
    assert!(
        ctx.get::<crate::assets::Parent>(child).is_none(),
        "an unnamed parent detaches the child"
    );
    let global = ctx.get::<crate::assets::GlobalTransform>(child).unwrap();
    assert_eq!(
        [global.0[3][0], global.0[3][1], global.0[3][2]],
        [1.0, 2.0, 3.0],
        "the detached child falls back to its own transform"
    );
}

// A Lifetime already at zero expires on the next tick (any dt >= 0), routing
// through the same cascade a DespawnRequest uses: the entity leaves the ECS and
// its draw slot is retired.
#[test]
fn expired_lifetime_despawns_the_entity_and_retires_its_slot() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let _gs = init_graphics(&mut world, hooks);
    let entity = entity_named(&mut world, PROP);
    world
        .ctx()
        .insert(entity, crate::assets::Lifetime { remaining: 0.0 });

    let mut spawn = crate::spawn::SpawnSystem::new();
    spawn_step(&mut spawn, &mut world);

    assert!(lock(&state).saw(&Call::RetireDrawObject(0)));
    assert_eq!(
        world.ctx().query::<RenderHandle>().count(),
        0,
        "the expired entity is gone"
    );
}

// A Spawner whose accumulator already covers its interval is due on the next
// tick (any dt >= 0): it clones its template's draw slot at the spawner's own
// transform, and the copy carries the spawner's Lifetime countdown.
#[test]
fn due_spawner_clones_its_template_at_its_own_transform() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let _gs = init_graphics(&mut world, hooks);
    {
        let mut ctx = world.ctx();
        let spawner = ctx.components.spawn();
        ctx.insert(
            spawner,
            crate::assets::Spawner {
                template: PROP,
                interval: 1.0,
                lifetime: 5.0,
                // Already at the interval, so the first tick is due at dt >= 0.
                elapsed: 1.0,
                count: 0,
            },
        );
        ctx.insert(
            spawner,
            Transform {
                position: [9.0, 0.0, 0.0],
                rotation_deg: [0.0; 3],
                scale: [1.0; 3],
            },
        );
    }

    let mut spawn = crate::spawn::SpawnSystem::new();
    spawn_step(&mut spawn, &mut world);

    assert!(lock(&state).saw(&Call::CloneStaticDrawObject { src: 0, new_idx: 1 }));
    let ctx = world.ctx();
    assert_eq!(
        ctx.query::<crate::assets::Lifetime>()
            .map(|l| l.remaining)
            .collect::<Vec<_>>(),
        vec![5.0],
        "the cadence copy carries the spawner's lifetime"
    );
    let spawner_count = ctx
        .query::<crate::assets::Spawner>()
        .map(|s| s.count)
        .next();
    assert_eq!(spawner_count, Some(1), "the spawner counted the copy");
}

// The world clock freezes behind an open menu: neither a due Lifetime nor a due
// Spawner advances while `MenuActive` is set, and both resume once it clears.
// Time-independent, so it holds whatever the real frame dt happens to be.
#[test]
fn menu_active_freezes_lifetimes_and_spawners() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let _gs = init_graphics(&mut world, hooks);
    let entity = entity_named(&mut world, PROP);
    {
        let mut ctx = world.ctx();
        ctx.insert(entity, crate::assets::Lifetime { remaining: 0.0 });
        let spawner = ctx.components.spawn();
        ctx.insert(
            spawner,
            crate::assets::Spawner {
                template: PROP,
                interval: 1.0,
                lifetime: 0.0,
                elapsed: 1.0,
                count: 0,
            },
        );
    }
    world.resources.insert(crate::ecs::MenuActive(true));

    let mut spawn = crate::spawn::SpawnSystem::new();
    spawn_step(&mut spawn, &mut world);

    {
        let s = lock(&state);
        assert!(
            !s.saw(&Call::RetireDrawObject(0)),
            "the expired Lifetime does not fire behind the menu"
        );
        assert!(
            !s.calls
                .iter()
                .any(|c| matches!(c, Call::CloneStaticDrawObject { .. })),
            "the due Spawner does not fire behind the menu"
        );
    }
    assert_eq!(
        world
            .ctx()
            .query::<crate::assets::Spawner>()
            .map(|s| s.count)
            .next(),
        Some(0),
        "the spawner's clock never advanced"
    );

    // Closing the menu resumes the same world clock.
    world.resources.insert(crate::ecs::MenuActive(false));
    spawn_step(&mut spawn, &mut world);
    assert!(
        lock(&state).saw(&Call::RetireDrawObject(0)),
        "the expiry fires once the menu closes"
    );
}

// The System trait's init/step delegate to run_init/run_step, which is the
// entry point `World::start` and the schedule actually use.
#[test]
fn system_trait_delegates_to_init_and_step() {
    use crate::ecs::System;
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = GraphicsSystem {
        test_hooks: Some(hooks),
        ..Default::default()
    };
    {
        let mut ctx = world.ctx();
        crate::ecs::decompose::run(&mut ctx);
        gs.init(&mut ctx);
    }
    assert!(
        lock(&state).init.is_some(),
        "System::init built the backend"
    );

    let result = {
        let mut ctx = world.ctx();
        gs.step(&mut ctx)
    };
    assert_eq!(result, StepResult::Continue);
    assert_eq!(lock(&state).draw_frames(), 1, "System::step drew a frame");
    // The Debug impl surfaces the two fields a failure report needs.
    let shown = format!("{gs:?}");
    assert!(shown.contains("frame_count: 1"), "got {shown}");
    assert!(shown.contains("failed: false"), "got {shown}");
}

// The quality-toggle and cycle key mappings round-trip: reading a key back after
// setting it yields what was set. The three call sites (read state, flip state,
// derive backend settings) share this mapping, so a drift here would silently
// desync the settings menu from the backend.
#[test]
fn quality_toggle_and_cycle_helpers_round_trip() {
    use crate::assets::{AaMode, ReflectionBlurResolution, SsgiResolution};
    use crate::gfx::settings::QUALITY_CYCLE_KEYS;

    let mut cfg = crate::assets::PostProcessConfig::default();
    for key in [
        "ssao",
        "ssr",
        "ray_traced_reflections",
        "ssgi",
        "auto_exposure",
    ] {
        super::set_quality_toggle(&mut cfg, key, true);
        assert_eq!(super::quality_toggle_on(&cfg, key), Some(true), "{key} on");
        super::set_quality_toggle(&mut cfg, key, false);
        assert_eq!(
            super::quality_toggle_on(&cfg, key),
            Some(false),
            "{key} off"
        );
    }
    assert_eq!(super::quality_toggle_on(&cfg, "not_a_toggle"), None);
    // An unknown key is ignored rather than panicking.
    super::set_quality_toggle(&mut cfg, "not_a_toggle", true);

    for key in QUALITY_CYCLE_KEYS {
        assert!(super::is_quality_cycle(key));
        let index = super::quality_cycle_index(&cfg, key).expect("a cycle key has an index");
        super::set_quality_cycle(&mut cfg, key, index);
        assert_eq!(super::quality_cycle_index(&cfg, key), Some(index), "{key}");
    }
    assert!(!super::is_quality_cycle("ssao"));
    assert_eq!(super::quality_cycle_index(&cfg, "ssao"), None);
    super::set_quality_cycle(&mut cfg, "not_a_cycle", 0);

    // A ceiling clamps each cycle knob DOWN, and never raises one.
    cfg.aa_mode = AaMode::Taa;
    cfg.ssgi_resolution = SsgiResolution::Full;
    cfg.ssgi_rays = 32;
    cfg.ssgi_steps = 48;
    cfg.reflection_blur_resolution = ReflectionBlurResolution::Full;
    let ceiling = crate::gfx::quality_preset::resolve_ceiling(
        QualityPreset::Low,
        &GpuProfile {
            vendor: GpuVendor::Intel,
            tier: GpuTier::Integrated,
            memory_budget_bytes: 0,
            unified_memory: false,
            discrete: false,
        },
    );
    for key in QUALITY_CYCLE_KEYS {
        super::clamp_quality_cycle(&mut cfg, key, &ceiling, false);
    }
    assert_eq!(cfg.aa_mode, ceiling.aa_mode);
    assert_eq!(cfg.ssgi_rays, ceiling.ssgi_rays);
    assert_eq!(cfg.ssgi_steps, ceiling.ssgi_steps);
    assert_eq!(cfg.ssgi_resolution, ceiling.ssgi_resolution);

    // An explicitly overridden knob is left alone by the same clamp.
    let mut kept = crate::assets::PostProcessConfig {
        ssgi_rays: 32,
        ..Default::default()
    };
    super::clamp_quality_cycle(&mut kept, "ssgi_rays", &ceiling, true);
    assert_eq!(
        kept.ssgi_rays, 32,
        "an explicit override survives the ceiling"
    );
    super::clamp_quality_cycle(&mut kept, "not_a_cycle", &ceiling, false);
}

// The scene world plus a caller-shaped PostProcessConfig. The persisted-override
// overlay and the upscaler are deliberately gated on the world declaring one
// (overriding a feature is meaningless without its tunables), so any test of
// those paths must author a config rather than lean on the defaults.
fn post_config_scene(cfg: crate::assets::PostProcessConfig) -> WorldBuilder {
    let mut b = scene_builder();
    b.push(cfg);
    b
}

// The resolved settings snapshot init hands to SettingsSystem: the live values
// every settings row displays and cycles, after the world's config, the
// persisted overrides, and the preset ceiling have all settled.
fn settings_state(world: &TestWorld) -> &crate::gfx::settings_system::SettingsState {
    world
        .resources
        .get::<crate::gfx::settings_system::SettingsState>()
        .expect("SettingsState parked at init")
}

// A GPU profile at a chosen tier, for the Auto-preset ceiling resolution.
fn profile_at(tier: GpuTier) -> GpuProfile {
    GpuProfile {
        vendor: GpuVendor::Amd,
        tier,
        memory_budget_bytes: 8 << 30,
        unified_memory: false,
        discrete: true,
    }
}

// Every persisted post-process choice overrides the world's authored value and
// reaches both the backend and the live settings snapshot. Under `Custom` there
// is no ceiling, so each override stands exactly as stored.
#[test]
fn persisted_post_process_overrides_win_over_authored_config() {
    use crate::assets::{AaMode, IndirectLighting, ReflectionBlurResolution, SsgiResolution};

    let mut settings = crate::config::Settings::default();
    settings.graphics.quality_preset = Some(QualityPreset::Custom);
    // Sliders (stored as user-facing values; init runs them through the same
    // clamp/transform the live drag uses).
    settings.graphics.exposure_ev = Some(2.0);
    settings.graphics.bloom_intensity = Some(0.25);
    settings.graphics.bloom_threshold = Some(1.5);
    settings.graphics.bloom_knee = Some(0.2);
    settings.graphics.vignette = Some(0.4);
    settings.graphics.lut_strength = Some(0.3);
    settings.graphics.ambient_intensity = Some(2.0);
    // Quality toggles, each authored off in the world below.
    settings.graphics.ssao = Some(true);
    settings.graphics.ssr = Some(true);
    settings.graphics.ray_traced_reflections = Some(true);
    settings.graphics.ssgi = Some(true);
    settings.graphics.auto_exposure = Some(true);
    // Cycle dropdowns.
    settings.graphics.aa_mode = Some(AaMode::Taa);
    settings.graphics.ssgi_resolution = Some(SsgiResolution::Full);
    settings.graphics.ssgi_rays = Some(16);
    settings.graphics.ssgi_steps = Some(24);
    settings.graphics.reflection_blur_resolution = Some(ReflectionBlurResolution::Full);
    // Per-feature sub-quality sliders.
    settings.graphics.ssao_radius = Some(0.9);
    settings.graphics.ssao_intensity = Some(2.0);
    settings.graphics.ssr_intensity = Some(0.4);
    settings.graphics.ssr_max_distance = Some(20.0);
    settings.graphics.ssgi_intensity = Some(1.5);
    settings.graphics.ssgi_max_distance = Some(4.0);
    settings.graphics.auto_exposure_min_ev = Some(-4.0);
    settings.graphics.auto_exposure_max_ev = Some(4.0);
    settings.graphics.auto_exposure_speed = Some(3.0);

    let (state, hooks) = recording_hooks_with(settings, GpuProfile::UNKNOWN);
    let mut world = post_config_scene(crate::assets::PostProcessConfig {
        // Deliberately the opposite of every override above, so a value that
        // survives to the backend can only have come from the store.
        aa_mode: AaMode::Off,
        exposure_ev: -3.0,
        ..Default::default()
    })
    .build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    {
        let s = lock(&state);
        let init = s.init.as_ref().unwrap();
        assert!(
            init.taa_enabled,
            "the persisted TAA mode reaches the backend"
        );
        assert!(
            init.ssao_on,
            "the persisted SSAO toggle reaches the backend"
        );
    }

    let live = settings_state(&world);
    // Toggles.
    assert!(live.post_config.ssao);
    assert!(live.post_config.ssr);
    assert!(live.post_config.ray_traced_reflections);
    assert_eq!(live.post_config.indirect_lighting, IndirectLighting::Ssgi);
    assert!(live.post_config.auto_exposure);
    // Cycles.
    assert_eq!(live.post_config.aa_mode, AaMode::Taa);
    assert_eq!(live.post_config.ssgi_resolution, SsgiResolution::Full);
    assert_eq!(live.post_config.ssgi_rays, 16);
    assert_eq!(live.post_config.ssgi_steps, 24);
    assert_eq!(
        live.post_config.reflection_blur_resolution,
        ReflectionBlurResolution::Full
    );
    // Sub-quality sliders.
    assert_eq!(live.post_config.ssao_radius, 0.9);
    assert_eq!(live.post_config.ssao_intensity, 2.0);
    assert_eq!(live.post_config.ssr_intensity, 0.4);
    assert_eq!(live.post_config.ssr_max_distance, 20.0);
    assert_eq!(live.post_config.ssgi_intensity, 1.5);
    assert_eq!(live.post_config.ssgi_max_distance, 4.0);
    assert_eq!(live.post_config.auto_exposure_min_ev, -4.0);
    assert_eq!(live.post_config.auto_exposure_max_ev, 4.0);
    assert_eq!(live.post_config.auto_exposure_speed, 3.0);
    // Post-process params: exposure is stored as EV and applied as 2^EV.
    assert_eq!(
        live.post_process.exposure, 4.0,
        "2^2 EV, not the authored -3"
    );
    assert_eq!(live.post_process.bloom_intensity, 0.25);
    assert_eq!(live.post_process.bloom_threshold, 1.5);
    assert_eq!(live.post_process.bloom_knee, 0.2);
    assert_eq!(live.post_process.vignette, 0.4);
    assert_eq!(live.post_process.lut_strength, 0.3);
    assert_eq!(live.ambient_intensity, 2.0);
    // The composite FXAA flag follows the final AA mode, not the authored one.
    assert_eq!(live.post_process.fxaa, 1.0, "TAA keeps the FXAA cleanup");
    // The world's pristine config is kept as the baseline a live preset change
    // re-clamps from, untouched by the overrides.
    assert_eq!(live.authored_post_config.aa_mode, AaMode::Off);
    assert!(!live.authored_post_config.ssao);
}

// A weak-tier ceiling forces the world's authored effects off -- but only where
// the user expressed no preference. The pristine authored config is kept so a
// later preset up-shift can restore exactly what the world asked for.
#[test]
fn low_preset_forces_authored_effects_off_but_keeps_the_baseline() {
    use crate::assets::{AaMode, IndirectLighting};

    let mut settings = crate::config::Settings::default();
    settings.graphics.quality_preset = Some(QualityPreset::Low);
    let (state, hooks) = recording_hooks_with(settings, profile_at(GpuTier::Integrated));
    let mut world = post_config_scene(crate::assets::PostProcessConfig {
        ssao: true,
        ssr: true,
        ray_traced_reflections: true,
        indirect_lighting: IndirectLighting::Ssgi,
        auto_exposure: true,
        aa_mode: AaMode::Taa,
        ..Default::default()
    })
    .build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    {
        let s = lock(&state);
        let init = s.init.as_ref().unwrap();
        assert!(
            !init.ssao_on,
            "the ceiling trims SSAO before the backend sizes it"
        );
        assert!(!init.taa_enabled, "TAA clamps down to a cheaper AA mode");
    }

    let live = settings_state(&world);
    assert!(!live.post_config.ssao);
    assert!(!live.post_config.ssr);
    assert!(!live.post_config.ray_traced_reflections);
    assert_eq!(live.post_config.indirect_lighting, IndirectLighting::Ibl);
    assert_ne!(live.post_config.aa_mode, AaMode::Taa);
    // Auto-exposure is cheap enough that even the weakest ceiling permits it, so
    // the world's authored choice stands.
    assert!(live.post_config.auto_exposure);
    // The authored baseline survives the clamp, so an up-shift restores it.
    assert!(live.authored_post_config.ssao);
    assert!(live.authored_post_config.ssr);
    assert_eq!(live.authored_post_config.aa_mode, AaMode::Taa);
}

// An explicit per-row override beats the preset ceiling: the user asked for this
// feature on a tier that would otherwise force it off, and that choice wins.
// Rows the user never touched still clamp.
#[test]
fn an_explicit_override_survives_the_preset_ceiling() {
    use crate::assets::AaMode;

    let mut settings = crate::config::Settings::default();
    settings.graphics.quality_preset = Some(QualityPreset::Low);
    settings.graphics.ssao = Some(true);
    settings.graphics.aa_mode = Some(AaMode::Taa);
    let (_state, hooks) = recording_hooks_with(settings, profile_at(GpuTier::Integrated));
    let mut world = post_config_scene(crate::assets::PostProcessConfig {
        ssao: true,
        ssr: true,
        aa_mode: AaMode::Taa,
        ..Default::default()
    })
    .build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    let live = settings_state(&world);
    assert!(live.post_config.ssao, "the explicit SSAO choice wins");
    assert_eq!(
        live.post_config.aa_mode,
        AaMode::Taa,
        "explicit AA mode wins"
    );
    assert!(
        !live.post_config.ssr,
        "a row the user never touched still clamps under the ceiling"
    );
}

// A ceiling only ever reduces: a world that authored nothing is not "upgraded"
// by a high tier, and the resolved preset is held for the master menu row.
#[test]
fn a_high_ceiling_never_enables_what_the_world_did_not_author() {
    let mut settings = crate::config::Settings::default();
    settings.graphics.quality_preset = Some(QualityPreset::Ultra);
    let (_state, hooks) = recording_hooks_with(settings, profile_at(GpuTier::HighDiscrete));
    let mut world = post_config_scene(Default::default()).build();
    let gs = init_graphics(&mut world, hooks);

    assert_eq!(gs.quality_preset, QualityPreset::Ultra);
    let live = settings_state(&world);
    assert!(!live.post_config.ssao, "an unauthored feature stays off");
    assert!(!live.post_config.ssr);
    assert!(!live.post_config.ray_traced_reflections);
}

// Under `Auto` the ceiling re-resolves from the detected tier each launch: a
// weaker GPU shadows a smaller map than a stronger one, and the strongest tier
// leaves the world's authored size alone.
#[test]
fn auto_preset_shadow_ceiling_tracks_the_detected_tier() {
    let sizes: Vec<u32> = [
        GpuTier::Integrated,
        GpuTier::EntryDiscrete,
        GpuTier::MidDiscrete,
        GpuTier::HighDiscrete,
    ]
    .into_iter()
    .map(|tier| {
        let (state, hooks) =
            recording_hooks_with(crate::config::Settings::default(), profile_at(tier));
        let mut world = scene_builder().build();
        let gs = init_graphics(&mut world, hooks);
        assert_eq!(gs.quality_preset, QualityPreset::Auto);
        lock(&state).init.as_ref().unwrap().shadows.map_size
    })
    .collect();

    assert!(
        sizes.windows(2).all(|w| w[0] <= w[1]),
        "a stronger tier never shadows at a smaller map: {sizes:?}"
    );
    assert_eq!(
        sizes.last().copied(),
        Some(2048),
        "the top tier leaves the world's authored 2048 unclamped"
    );
}

// Display-output, upscaling, and system/streaming preferences are restart-class:
// resolved once at init from the world's config overridden by the persisted
// choice, passed to the backend, and held for the settings rows. The window mode
// and display mode are applied to the backend after construction, since the
// window is always created windowed.
#[test]
fn persisted_display_and_system_overrides_reach_the_backend() {
    use crate::assets::{UpscaleQuality, UpscalerBackend, WindowMode};
    use crate::gfx::display_mode::DisplayMode;

    let mut settings = crate::config::Settings::default();
    settings.graphics.quality_preset = Some(QualityPreset::Custom);
    settings.graphics.window_mode = Some(WindowMode::Fullscreen);
    settings.graphics.resolution = Some([1920, 1080, 60]);
    settings.graphics.render_scale = Some(UpscaleQuality::Performance);
    settings.graphics.upscale_backend = Some(UpscalerBackend::Fsr3);
    settings.graphics.temporal_upscaling = Some(true);
    settings.graphics.hdr_display = Some(true);
    settings.graphics.hdr_pq = Some(true);
    settings.graphics.occlusion_two_pass = Some(true);
    settings.graphics.texture_cap = Some(192);
    settings.graphics.texture_budget = Some(8);
    settings.graphics.perf_stats = Some(false);
    settings.graphics.show_fps = Some(false);
    settings.graphics.show_vram = Some(false);

    let (state, hooks) = recording_hooks_with(settings, GpuProfile::UNKNOWN);
    let mut b = post_config_scene(Default::default());
    b.push(StreamingConfig::default());
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    {
        let s = lock(&state);
        assert!(
            s.saw(&Call::SetWindowMode(WindowMode::Fullscreen)),
            "a persisted non-windowed mode is applied after construction"
        );
        assert!(
            s.saw(&Call::SetDisplayMode(DisplayMode {
                width: 1920,
                height: 1080,
                refresh_hz: 60,
            })),
            "the chosen fullscreen mode is handed to the backend"
        );
    }

    let live = settings_state(&world);
    assert_eq!(live.render_scale, UpscaleQuality::Performance);
    assert_eq!(live.upscale_backend, UpscalerBackend::Fsr3);
    assert!(live.temporal_upscaling);
    assert!(live.hdr_display);
    assert!(live.hdr_pq);
    assert!(live.occlusion_two_pass);
    assert_eq!(live.texture_cap, 192);
    assert_eq!(live.texture_budget, 8);
    assert!(!live.perf_stats);
    assert!(!live.show_fps);
    assert!(!live.show_vram);
    assert_eq!(live.window_args.mode, WindowMode::Fullscreen);
    assert_eq!(
        live.resolution,
        Some(DisplayMode {
            width: 1920,
            height: 1080,
            refresh_hz: 60,
        })
    );
    // A backend that cannot enumerate modes falls back to the static preset list
    // rather than publishing an empty dropdown.
    assert!(
        !world
            .resources
            .get::<crate::ecs::DisplayModes>()
            .expect("mode list published for the dropdown")
            .0
            .is_empty()
    );
    // The resolved cap reaches the App-level pacer through its own resource.
    assert!(world.resources.get::<crate::ecs::FrameRateCap>().is_some());
}

// One settings row: the `setting:<key>:<verb>` HitRegion the menu builds plus
// the value TextLabel it points at. Init syncs the label to the live value while
// the regions are still present (UiInputSystem drains them afterwards).
fn push_settings_row(b: &mut WorldBuilder, key: &str, verb: &str, label: AssetId) {
    b.push(HitRegion {
        action: format!("setting:{key}:{verb}"),
        label: Some(label),
        ..Default::default()
    });
    b.push(TextLabel {
        asset_id: label,
        content: "<placeholder>".to_string(),
        color: LIT,
        ..Default::default()
    });
}

// The authored (non-grayed) row color the tests below start every label at, so a
// gray-out is visible as a change away from it.
const LIT: [f32; 3] = [0.9, 0.9, 0.9];

// The content of the TextLabel with `id`.
fn label_text(world: &mut TestWorld, id: AssetId) -> String {
    world
        .ctx()
        .query::<TextLabel>()
        .find(|l| l.asset_id == id)
        .expect("label present")
        .content
        .clone()
}

// The color of the TextLabel with `id`.
fn label_color(world: &mut TestWorld, id: AssetId) -> [f32; 3] {
    world
        .ctx()
        .query::<TextLabel>()
        .find(|l| l.asset_id == id)
        .expect("label present")
        .color
}

// Every settings row's value label is synced to its live value before the first
// render, so a persisted or authored choice shows instead of the build's
// placeholder. The master preset row carries the resolved tier under `Auto`,
// which the static option table cannot express.
#[test]
fn settings_rows_show_their_live_values_at_init() {
    use crate::assets::ShadowUpdate;

    let mut settings = crate::config::Settings::default();
    settings.graphics.vsync = Some(true);
    settings.graphics.fps_cap = Some(60);
    settings.graphics.shadow_map_size = Some(4096);
    settings.graphics.shadow_update = Some(ShadowUpdate::EveryFrame);
    settings.graphics.shadow_distance = Some(160);
    settings.graphics.shadow_cascades = Some(3);
    settings.graphics.anisotropy = Some(16);
    settings.graphics.frames_in_flight = Some(3);
    settings.graphics.occlusion_two_pass = Some(true);
    settings.graphics.texture_cap = Some(384);
    settings.graphics.hdr_display = Some(true);
    settings.graphics.ssao = Some(true);
    settings.graphics.ssgi_rays = Some(32);

    // A classified GPU, so the Auto preset resolves to a named tier below.
    let (_state, hooks) = recording_hooks_with(settings, profile_at(GpuTier::MidDiscrete));
    let mut b = post_config_scene(Default::default());
    // (key, verb, label id) for a spread across every value-label arm: booleans,
    // discrete numeric levels, enums, a quality toggle, and a cycle dropdown.
    let rows: Vec<(&str, &str, AssetId)> = vec![
        ("vsync", "next", AssetId(100)),
        ("fps_cap", "next", AssetId(101)),
        ("window_mode", "next", AssetId(102)),
        ("render_scale", "next", AssetId(103)),
        ("master_volume", "next", AssetId(104)),
        ("temporal_upscaling", "next", AssetId(105)),
        ("hdr_display", "next", AssetId(106)),
        ("hdr_pq", "next", AssetId(107)),
        ("perf_stats", "next", AssetId(108)),
        ("show_fps", "next", AssetId(109)),
        ("show_vram", "next", AssetId(110)),
        ("shadow_map_size", "next", AssetId(111)),
        ("shadow_update", "next", AssetId(112)),
        ("shadow_distance", "next", AssetId(113)),
        ("shadow_cascades", "next", AssetId(114)),
        ("anisotropy", "next", AssetId(115)),
        ("frames_in_flight", "next", AssetId(116)),
        ("occlusion_two_pass", "next", AssetId(117)),
        ("texture_quality", "next", AssetId(118)),
        ("ssao", "next", AssetId(119)),
        ("ssgi_rays", "open", AssetId(120)),
        ("graphics_quality", "next", AssetId(121)),
        // An unknown key has no options table, so its label is left alone.
        ("not_a_setting", "next", AssetId(122)),
    ];
    for &(key, verb, label) in &rows {
        push_settings_row(&mut b, key, verb, label);
    }
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    assert_eq!(
        label_text(&mut world, AssetId(100)),
        "On",
        "persisted vsync"
    );
    assert_eq!(label_text(&mut world, AssetId(101)), "60");
    assert_eq!(label_text(&mut world, AssetId(102)), "Windowed");
    assert_eq!(label_text(&mut world, AssetId(106)), "On", "hdr_display");
    assert_eq!(label_text(&mut world, AssetId(107)), "Off", "hdr_pq unset");
    assert_eq!(label_text(&mut world, AssetId(111)), "4096");
    assert_eq!(label_text(&mut world, AssetId(114)), "3");
    assert_eq!(label_text(&mut world, AssetId(115)), "16x");
    assert_eq!(label_text(&mut world, AssetId(116)), "3");
    assert_eq!(label_text(&mut world, AssetId(117)), "On", "occlusion");
    assert_eq!(label_text(&mut world, AssetId(119)), "On", "quality toggle");
    assert_eq!(label_text(&mut world, AssetId(120)), "32", "cycle dropdown");
    assert_eq!(
        label_text(&mut world, AssetId(121)),
        "Auto (High)",
        "the master row carries the tier the Auto preset resolved to, \
         which the static option table cannot express"
    );
    assert_eq!(
        label_text(&mut world, AssetId(122)),
        "<placeholder>",
        "an unknown setting key leaves its label untouched"
    );
    // Every non-dynamic row's label moved off the build's placeholder.
    for &(key, _, label) in &rows {
        if key == "not_a_setting" {
            continue;
        }
        assert_ne!(
            label_text(&mut world, label),
            "<placeholder>",
            "{key} still shows the build placeholder"
        );
    }
    // The cycle rows' value labels are captured for the runtime relabel, since
    // the HitRegions are drained right after init.
    let live = settings_state(&world);
    assert_eq!(
        live.cycle_value_labels.get("ssgi_rays"),
        Some(&AssetId(120))
    );
    assert_eq!(live.cycle_value_labels.get("vsync"), Some(&AssetId(100)));
}

// Each slider row's handle position and value label are synced to the live value
// at init: the handle sits at the value's fraction along the track, and the label
// shows the formatted value. The exposure slider stores 2^EV, so the sync must
// recover the EV to place the handle.
#[test]
fn slider_rows_sync_their_handle_and_label_to_the_live_value() {
    let mut settings = crate::config::Settings::default();
    settings.graphics.quality_preset = Some(QualityPreset::Custom);
    settings.graphics.exposure_ev = Some(0.0);
    settings.graphics.vignette = Some(0.5);
    let (_state, hooks) = recording_hooks_with(settings, GpuProfile::UNKNOWN);

    let mut b = post_config_scene(Default::default());
    // A drag region spanning x = 0..100 with a 10-wide handle: the handle travels
    // the 90 units between the track's start and its own right edge.
    for (key, handle, label) in [
        ("exposure", AssetId(200), AssetId(201)),
        ("vignette", AssetId(202), AssetId(203)),
        // A drag region for a key this system does not own is captured but has no
        // value to sync from.
        ("not_a_slider", AssetId(204), AssetId(205)),
    ] {
        b.push(HitRegion {
            action: format!("setting:{key}:drag"),
            x: 0.0,
            width: 100.0,
            drag_handle: Some(handle),
            label: Some(label),
            ..Default::default()
        });
        b.push(Sprite {
            asset_id: handle,
            width: 10.0,
            ..Default::default()
        });
        b.push(TextLabel {
            asset_id: label,
            content: "<placeholder>".to_string(),
            ..Default::default()
        });
    }
    // A drag region missing its handle / label is skipped rather than panicking.
    b.push(HitRegion {
        action: "setting:exposure:drag".to_string(),
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    let handle_x = |world: &mut TestWorld, id: AssetId| {
        world
            .ctx()
            .query::<Sprite>()
            .find(|s| s.asset_id == id)
            .expect("handle sprite present")
            .x
    };
    // Exposure spans -4..+4 EV, so 0 EV is mid-track: 0 + 0.5 * (100 - 10).
    assert_eq!(handle_x(&mut world, AssetId(200)), 45.0);
    assert_eq!(label_text(&mut world, AssetId(201)), "+0.0 EV");
    // Vignette spans 0..1, so 0.5 is also mid-track.
    assert_eq!(handle_x(&mut world, AssetId(202)), 45.0);
    assert_ne!(label_text(&mut world, AssetId(203)), "<placeholder>");
    // A key this system does not own leaves its row untouched.
    assert_eq!(handle_x(&mut world, AssetId(204)), 0.0);
    assert_eq!(label_text(&mut world, AssetId(205)), "<placeholder>");

    // The captured rows are handed to SettingsSystem for the live drag drain.
    let live = settings_state(&world);
    let keys: Vec<&str> = live.sliders.iter().map(|s| s.key.as_str()).collect();
    assert!(keys.contains(&"exposure"));
    assert!(keys.contains(&"vignette"));
    let exposure = live.sliders.iter().find(|s| s.key == "exposure").unwrap();
    assert_eq!(
        (exposure.track_x, exposure.track_w, exposure.handle_w),
        (0.0, 100.0, 10.0)
    );
}

// Every slider this system owns recovers a live value, so no row silently shows
// the build placeholder. Sliders backed by the persisted store (mouse
// sensitivity, FOV) are excluded: reading them would hit the on-disk settings
// file, which these tests never touch.
#[test]
fn every_owned_slider_key_recovers_a_live_value() {
    let (_state, hooks) = recording_hooks();
    let mut b = post_config_scene(Default::default());
    let keys = [
        "exposure",
        "bloom_intensity",
        "bloom_threshold",
        "bloom_knee",
        "vignette",
        "lut_strength",
        "ambient_intensity",
        "ssao_radius",
        "ssao_intensity",
        "ssr_intensity",
        "ssr_max_distance",
        "ssgi_intensity",
        "ssgi_max_distance",
        "auto_exposure_min_ev",
        "auto_exposure_max_ev",
        "auto_exposure_speed",
    ];
    for (i, key) in keys.iter().enumerate() {
        let handle = AssetId(300 + i as u32 * 2);
        let label = AssetId(301 + i as u32 * 2);
        b.push(HitRegion {
            action: format!("setting:{key}:drag"),
            x: 0.0,
            width: 100.0,
            drag_handle: Some(handle),
            label: Some(label),
            ..Default::default()
        });
        b.push(Sprite {
            asset_id: handle,
            width: 10.0,
            ..Default::default()
        });
        b.push(TextLabel {
            asset_id: label,
            content: "<placeholder>".to_string(),
            ..Default::default()
        });
    }
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    for (i, key) in keys.iter().enumerate() {
        let label = AssetId(301 + i as u32 * 2);
        assert_ne!(
            label_text(&mut world, label),
            "<placeholder>",
            "{key} never synced to a live value"
        );
    }
    assert_eq!(settings_state(&world).sliders.len(), keys.len());
}

// Each Controls-tab rebind row's value label is synced to the live bound key
// (the persisted rebind, or the engine default), and the rows are captured for
// the live rebind drain.
#[test]
fn rebind_rows_show_their_bound_keys_at_init() {
    use crate::assets::Key;
    use crate::gfx::keymap::{Bindable, KeyMap};

    let mut settings = crate::config::Settings::default();
    settings.controls.keymap = Some(KeyMap {
        forward: Key::Up,
        ..Default::default()
    });
    let (_state, hooks) = recording_hooks_with(settings, GpuProfile::UNKNOWN);

    let mut b = scene_builder();
    for (i, action) in Bindable::ALL.iter().enumerate() {
        let label = AssetId(400 + i as u32);
        b.push(HitRegion {
            action: format!("setting:{}:rebind", action.setting_key()),
            label: Some(label),
            ..Default::default()
        });
        b.push(TextLabel {
            asset_id: label,
            content: "<placeholder>".to_string(),
            ..Default::default()
        });
    }
    // A rebind region whose key names no bindable action is skipped.
    b.push(HitRegion {
        action: "setting:key_nonsense:rebind".to_string(),
        label: Some(AssetId(499)),
        ..Default::default()
    });
    b.push(TextLabel {
        asset_id: AssetId(499),
        content: "<placeholder>".to_string(),
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    assert_eq!(
        label_text(&mut world, AssetId(400)),
        Key::Up.display_name(),
        "the persisted forward rebind shows on its row"
    );
    for i in 1..Bindable::ALL.len() {
        assert_ne!(
            label_text(&mut world, AssetId(400 + i as u32)),
            "<placeholder>",
            "row {i} never synced to its default binding"
        );
    }
    assert_eq!(
        label_text(&mut world, AssetId(499)),
        "<placeholder>",
        "an unknown rebind key leaves its row alone"
    );
    let live = settings_state(&world);
    assert_eq!(live.rebind_rows.len(), Bindable::ALL.len());
    assert_eq!(
        live.keymap.forward,
        Key::Up,
        "the persisted map is the live one"
    );
}

// Every element a ScrollPanel lists is mapped to its panel's content band, so
// the draw path can scissor scroll rows to the panel and an off-band row does
// not bleed over the chrome. The bands are handed to OverlaySystem at init.
#[test]
fn scroll_panel_rows_clip_their_elements_to_the_panel_band() {
    use crate::assets::{ScrollPanel, ScrollRow};

    let (_state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(ScrollPanel {
        x: 10.0,
        y: 20.0,
        width: 300.0,
        height: 400.0,
        rows: vec![
            ScrollRow {
                elements: vec![AssetId(500), AssetId(501)],
                ..Default::default()
            },
            ScrollRow {
                elements: vec![AssetId(502)],
                ..Default::default()
            },
        ],
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    let clips = &world
        .resources
        .get::<crate::gfx::overlay::OverlayAssets>()
        .expect("OverlayAssets parked at init")
        .clip_rects;
    let band = [10.0, 20.0, 300.0, 400.0];
    for id in [500, 501, 502] {
        assert_eq!(
            clips.get(&AssetId(id)),
            Some(&band),
            "element {id} clips to its panel's band"
        );
    }
    assert!(
        clips.get(&AssetId(503)).is_none(),
        "an unlisted element is unclipped"
    );
}

// A settings row whose feature the backend cannot honour is grayed out and made
// inert, and the gray expands from its value label to the whole scroll row
// (background, name, value, steppers) so the row reads as unavailable as a unit.
// Driven here by a device that reports a fixed upscaler, which grays the
// upscaler-selector row.
#[test]
fn a_capability_gated_row_grays_out_its_whole_scroll_row() {
    use crate::assets::{ScrollPanel, ScrollRow};

    let (state, hooks) = recording_hooks();
    state.lock().unwrap().caps = crate::gfx::backend::DeviceCapabilities {
        selectable_upscaler: false,
        ..crate::gfx::backend::DeviceCapabilities::ALL
    };
    let mut b = post_config_scene(Default::default());
    // The gated row: name + value labels, both listed in one scroll row.
    push_settings_row(&mut b, "upscale_backend", "next", AssetId(600));
    b.push(TextLabel {
        asset_id: AssetId(601),
        content: "Upscaler".to_string(),
        color: LIT,
        ..Default::default()
    });
    // An ungated row alongside it, which must stay lit.
    push_settings_row(&mut b, "vsync", "next", AssetId(602));
    b.push(ScrollPanel {
        rows: vec![
            ScrollRow {
                elements: vec![AssetId(600), AssetId(601)],
                ..Default::default()
            },
            ScrollRow {
                elements: vec![AssetId(602)],
                ..Default::default()
            },
        ],
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    assert_eq!(
        label_color(&mut world, AssetId(600)),
        DISABLED,
        "the gated row's value label"
    );
    assert_eq!(
        label_color(&mut world, AssetId(601)),
        DISABLED,
        "the gray expands to the row's name label, not just its value"
    );
    assert_eq!(
        label_color(&mut world, AssetId(602)),
        LIT,
        "an available row stays lit"
    );
    // A disabled region is dropped by UiInputSystem, so it never hovers or fires.
    let disabled: Vec<bool> = world
        .ctx()
        .query::<HitRegion>()
        .filter(|r| r.action.starts_with("setting:upscale_backend"))
        .map(|r| r.disabled)
        .collect();
    assert_eq!(disabled, vec![true]);
}

// The muted gray a disabled settings row is recolored to.
const DISABLED: [f32; 3] = crate::gfx::settings_system::rows::DISABLED_ROW_COLOR;

// The master "Display performance stats" toggle grays its two sub-rows when it
// is off, and the Resolution row grays outside fullscreen (windowed sizes come
// from the window itself, so the row does not apply).
#[test]
fn master_toggles_gray_the_rows_they_govern() {
    use crate::assets::{ScrollPanel, ScrollRow};

    let mut settings = crate::config::Settings::default();
    settings.graphics.perf_stats = Some(false);
    let (_state, hooks) = recording_hooks_with(settings, GpuProfile::UNKNOWN);

    let mut b = scene_builder();
    push_settings_row(&mut b, "show_fps", "next", AssetId(700));
    push_settings_row(&mut b, "show_vram", "next", AssetId(701));
    push_settings_row(&mut b, "resolution", "open", AssetId(702));
    push_settings_row(&mut b, "vsync", "next", AssetId(703));
    b.push(ScrollPanel {
        rows: [700, 701, 702, 703]
            .into_iter()
            .map(|id| ScrollRow {
                elements: vec![AssetId(id)],
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    assert_eq!(
        label_color(&mut world, AssetId(700)),
        DISABLED,
        "show_fps grays under the master toggle"
    );
    assert_eq!(label_color(&mut world, AssetId(701)), DISABLED, "show_vram");
    assert_eq!(
        label_color(&mut world, AssetId(702)),
        DISABLED,
        "the Resolution row grays outside fullscreen"
    );
    assert_eq!(
        label_color(&mut world, AssetId(703)),
        LIT,
        "an unrelated row"
    );
    // The captured labels carry the authored colors the restore reads back.
    let live = settings_state(&world);
    assert!(live.perf_sub_row_labels.contains(&(AssetId(700), LIT)));
    assert!(live.resolution_row_labels.contains(&(AssetId(702), LIT)));
}

// A w x h font atlas in the compiled payload format, carrying one glyph so the
// cap-height derivation has metrics to read.
fn font_payload(w: u32, h: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    for v in [w, h, 1, 32] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes.extend(vec![0xFFu8; (w * h * 4) as usize]);
    // One glyph: 'H', whose height drives `derive_cap_px`.
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&(b'H' as u32).to_le_bytes());
    for v in [0u16, 0, 16, 22] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in [18.0f32, 1.0, 22.0] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

impl WorldBuilder {
    // Register a resource record of `kind` at the next handle for that kind,
    // with `bytes` as its compiled payload, and return the handle.
    fn push_resource(&mut self, kind: concinnity_core::ecs::ResourceKind, bytes: &[u8]) -> u32 {
        let locator = self.payload(bytes);
        let handle = self.kind_records(kind).len() as u32;
        self.kind_records(kind)
            .push(concinnity_core::ecs::ResourceRecord {
                resource_kind: kind as u8,
                handle,
                payload: Some(locator),
                data_bytes: Vec::new(),
            });
        handle
    }

    // The record list backing a resource kind. Only the kinds the media tests
    // below author are routed; the quad builder owns mesh / texture / material.
    fn kind_records(
        &mut self,
        kind: concinnity_core::ecs::ResourceKind,
    ) -> &mut Vec<concinnity_core::ecs::ResourceRecord> {
        use concinnity_core::ecs::ResourceKind;
        match kind {
            ResourceKind::Texture => &mut self.texture_records,
            ResourceKind::Font => &mut self.font_records,
            ResourceKind::ColorLut => &mut self.color_lut_records,
            ResourceKind::EnvironmentMap => &mut self.env_map_records,
            ResourceKind::SkinnedMesh => &mut self.skinned_records,
            other => panic!("no record list wired for {other:?}"),
        }
    }
}

// Font atlases and sprite textures share the text-atlas pool: each font's atlas
// takes its handle's leading slot, and each distinct Texture a Sprite references
// is decoded and appended after them (drawn by the same pipeline).
#[test]
fn fonts_and_sprite_textures_share_the_text_atlas_pool() {
    use concinnity_core::ecs::ResourceKind;

    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push_resource(ResourceKind::Font, &font_payload(32, 32));
    b.push_resource(ResourceKind::Font, &font_payload(64, 64));
    // A second texture (handle 1; the quad's albedo owns handle 0) for a Sprite
    // to reference. Two sprites share it, so it is decoded once.
    let sprite_tex = b.push_resource(ResourceKind::Texture, &texture_payload(4, 4));
    for id in [800u32, 801] {
        b.push(Sprite {
            asset_id: AssetId(id),
            texture: Some(TextureHandle(sprite_tex)),
            visible: true,
            ..Default::default()
        });
    }
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    let s = lock(&state);
    assert_eq!(
        s.init.as_ref().unwrap().text_atlas_count,
        3,
        "two font atlases plus the one deduped sprite texture"
    );
    drop(s);

    let overlay = world
        .resources
        .get::<crate::gfx::overlay::OverlayAssets>()
        .expect("OverlayAssets parked at init");
    // A font's handle IS its atlas slot, so the pool's leading slots are the
    // fonts in handle order.
    assert_eq!(overlay.fonts.len(), 2);
    assert_eq!(
        overlay.fonts[&crate::ecs::FontHandle(0)].atlas_slot,
        0,
        "font handle 0 owns atlas slot 0"
    );
    assert_eq!(overlay.fonts[&crate::ecs::FontHandle(1)].atlas_slot, 1);
    assert_eq!(overlay.fonts[&crate::ecs::FontHandle(1)].atlas_w, 64);
    assert_eq!(overlay.fonts[&crate::ecs::FontHandle(0)].size_px, 32.0);
    // The sprite texture lands after the fonts.
    assert_eq!(
        overlay.sprite_texture_slots.get(&TextureHandle(sprite_tex)),
        Some(&2),
        "the sprite texture is appended after the font atlases"
    );
}

// A Sprite pointing at a texture that is not in the pool demotes to its solid
// tint fill rather than failing the world build.
#[test]
fn a_sprite_with_an_unknown_texture_keeps_its_tint() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(Sprite {
        asset_id: AssetId(810),
        texture: Some(TextureHandle(99)),
        visible: true,
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed, "an unresolvable sprite texture is not fatal");
    assert_eq!(
        lock(&state).init.as_ref().unwrap().text_atlas_count,
        0,
        "nothing was appended to the text-atlas pool"
    );
    assert!(
        world
            .resources
            .get::<crate::gfx::overlay::OverlayAssets>()
            .unwrap()
            .sprite_texture_slots
            .is_empty()
    );
}

// A malformed Font payload fails the world build rather than rendering blank
// text: the atlas is a build product, so a bad one means the build is broken.
#[test]
fn a_malformed_font_payload_fails_init() {
    use concinnity_core::ecs::ResourceKind;

    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push_resource(ResourceKind::Font, b"not-a-font");
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(gs.failed);
    assert!(!backend_parked(&world));
    assert!(lock(&state).init.is_none(), "backend never constructed");
}

// The EnvironmentMap and ColorLut payloads are read from their resource tables
// before the shared blob is released and handed to the backend. Both are
// singletons: the runtime uses handle 0 and logs any extras.
#[test]
fn environment_map_and_color_lut_payloads_survive_to_the_backend() {
    use concinnity_core::ecs::ResourceKind;

    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push_resource(ResourceKind::EnvironmentMap, b"ibl-cube-bytes");
    // A second map: extras are ignored, not fatal.
    b.push_resource(ResourceKind::EnvironmentMap, b"second-ibl");
    b.push_resource(ResourceKind::ColorLut, b"lut-bytes");
    b.push_resource(ResourceKind::ColorLut, b"second-lut");
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed, "extra singletons are ignored, not fatal");
    assert!(lock(&state).init.is_some(), "the backend was built");
}

// An EnvironmentMap whose locator does not resolve fails the world build: the
// IBL environment is a build product, so a payload that cannot be read means the
// build is broken rather than that the world wants no environment.
#[test]
fn an_unreadable_environment_map_payload_fails_init() {
    let (_state, hooks) = recording_hooks();
    let mut b = scene_builder();
    // A locator past the end of the blob's only section.
    b.env_map_records
        .push(concinnity_core::ecs::ResourceRecord {
            resource_kind: concinnity_core::ecs::ResourceKind::EnvironmentMap as u8,
            handle: 0,
            payload: Some(PayloadLocator {
                blob_index: 0,
                offset: 1 << 30,
                len: 16,
            }),
            data_bytes: Vec::new(),
        });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(gs.failed);
    assert!(!backend_parked(&world));
}

// A world's VolumetricFog resolves into the backend's fog pass. The first
// declared instance wins (one homogeneous medium is all the pass models).
#[test]
fn volumetric_fog_resolves_into_the_backend_init() {
    use crate::assets::VolumetricFog;

    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(VolumetricFog {
        enabled: true,
        density: 0.05,
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed);
    assert!(
        lock(&state).init.as_ref().unwrap().fog,
        "the fog pass is on"
    );
    // The bookkeeping the world.jsonl reload pass dedupes against is seeded from
    // whatever was passed into the backend constructor.
    assert!(gs.last_fog_settings.is_some());
}

// A `VolumetricFog` with `enabled = false` yields no fog pass at all, the same
// as declaring none: the renderer skips the pass rather than running it at zero
// density.
#[test]
fn disabled_volumetric_fog_skips_the_fog_pass() {
    use crate::assets::VolumetricFog;

    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(VolumetricFog {
        enabled: false,
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed);
    assert!(!lock(&state).init.as_ref().unwrap().fog);
    assert!(gs.last_fog_settings.is_none());
}

// A world's declared ReflectionProbes are handed to the backend as placements,
// replacing the auto-seed a probe-less world falls back to.
#[test]
fn declared_reflection_probes_replace_the_auto_seed() {
    use crate::assets::ReflectionProbe;

    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    for x in [0.0, 10.0] {
        b.push(ReflectionProbe {
            position: [x, 1.0, 0.0],
            half_extents: [4.0; 3],
        });
    }
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed);
    assert!(
        lock(&state).saw(&Call::SetReflectionProbes(2)),
        "both declared placements reach the backend"
    );
    assert_eq!(
        world.ctx().query::<ReflectionProbe>().count(),
        0,
        "the probes are drained into placements"
    );
}

// An InstancedProp bakes its instances into one GPU cluster rather than a draw
// object each, and the component is drained (there is no per-frame update path).
#[test]
fn instanced_prop_bakes_its_instances_into_one_cluster() {
    use crate::assets::{InstanceTransform, InstancedProp};

    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(InstancedProp {
        asset_id: AssetId(820),
        mesh: Some(crate::ecs::MeshHandle(0)),
        material: Some(crate::ecs::MaterialHandle(0)),
        instances: (0..3)
            .map(|i| InstanceTransform {
                position: [i as f32 * 2.0, 0.0, 0.0],
                rotation_deg: [0.0; 3],
                scale: [1.0; 3],
            })
            .collect(),
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed);
    let s = lock(&state);
    let init = s.init.as_ref().unwrap();
    assert_eq!(init.instanced_cluster_count, 1, "one mesh = one cluster");
    assert_eq!(
        init.draw_objects.len(),
        1,
        "the instances ride the cluster, not one draw object each"
    );
    drop(s);
    assert_eq!(world.ctx().query::<InstancedProp>().count(), 0, "drained");
}

// The one-shot world FX (decals, emitters, water, glass, SDF volumes) are
// resolved at init and drained: each is baked into a backend record at
// construction and has no per-frame update path, so leaving the components
// behind would only invite a second, stale build.
#[test]
fn one_shot_world_fx_are_resolved_and_drained_at_init() {
    use crate::assets::{Decal, GlassPanel, ParticleEmitter, SdfVolume, WaterSurface};

    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(Decal {
        asset_id: AssetId(830),
        texture: Some(TextureHandle(0)),
        size: [1.0; 3],
        visible: true,
        ..Default::default()
    });
    b.push(ParticleEmitter {
        asset_id: AssetId(831),
        texture: Some(TextureHandle(0)),
        max_particles: 16,
        visible: true,
        ..Default::default()
    });
    b.push(WaterSurface::default());
    b.push(GlassPanel::default());
    let sdf_frag = b.payload(b"sdf-fragment-bytes");
    b.push(SdfVolume {
        asset_id: AssetId(832),
        extent: [2.0; 3],
        locator: Some(sdf_frag),
        visible: true,
        ..Default::default()
    });
    // A volume whose fragment shader never compiled is skipped with a warning
    // rather than failing the whole world build.
    b.push(SdfVolume {
        asset_id: AssetId(833),
        extent: [2.0; 3],
        locator: None,
        visible: true,
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed, "a payload-less SdfVolume is not fatal");
    assert!(lock(&state).init.is_some());
    let ctx = world.ctx();
    assert_eq!(ctx.query::<Decal>().count(), 0);
    assert_eq!(ctx.query::<ParticleEmitter>().count(), 0);
    assert_eq!(ctx.query::<WaterSurface>().count(), 0);
    assert_eq!(ctx.query::<GlassPanel>().count(), 0);
    assert_eq!(ctx.query::<SdfVolume>().count(), 0);
}

// A backend factory that cannot build (no device, an unsupported surface) leaves
// the system failed rather than stepping on with no backend.
#[test]
fn a_backend_that_fails_to_build_marks_graphics_failed() {
    let (state, mut hooks) = recording_hooks();
    hooks.backend_factory = Box::new(|_init| None);
    let mut world = scene_builder().build();
    let gs = init_graphics(&mut world, hooks);

    assert!(gs.failed);
    assert!(!backend_parked(&world));
    assert!(lock(&state).init.is_none());
    // A failed system is Done on its next step rather than drawing.
    let mut gs = gs;
    assert_eq!(step(&mut gs, &mut world), StepResult::Done);
}

// A two-triangle skinned strip bound to a two-joint chain, in the compiled
// skinned-payload format. `n` vertices span x = 0..n-1, each fully weighted to
// joint 0, so the bind-pose AABB is predictable.
fn skinned_payload(n: u16, lods: &[(f32, Vec<u16>)]) -> Vec<u8> {
    use crate::gfx::mesh_payload::{PayloadJoint, SkinnedVertex, serialise_skinned_with_lods};
    let vertices: Vec<SkinnedVertex> = (0..n)
        .map(|i| SkinnedVertex {
            pos: [i as f32, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            color: [1.0; 3],
            uv: [0.0, 0.0],
            joints: [0, 0, 0, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
        })
        .collect();
    let joint = |name: &str, parent: i32| PayloadJoint {
        name: name.to_string(),
        parent,
        translation: [0.0; 3],
        rotation_deg: [0.0; 3],
        scale: [1.0; 3],
    };
    serialise_skinned_with_lods(
        &vertices,
        &[0, 1, 2],
        &[joint("root", -1), joint("child", 0)],
        &crate::gfx::mesh_payload::PayloadMorphs::default(),
        lods,
    )
}

// Register a SkinnedMesh resource: the placement / material / spawn reserve ride
// the baked `data_bytes`, the geometry + skeleton ride the compiled payload, and
// the table index IS the mesh's `SkinnedMeshHandle`.
fn push_skinned_mesh(b: &mut WorldBuilder, name: AssetId, sm: crate::assets::SkinnedMesh, n: u16) {
    let locator = b.payload(&skinned_payload(n, &[]));
    let handle = b.skinned_records.len() as u32;
    let data = postcard::to_allocvec(&(name.0, sm)).unwrap();
    b.skinned_records
        .push(concinnity_core::ecs::ResourceRecord {
            resource_kind: concinnity_core::ecs::ResourceKind::SkinnedMesh as u8,
            handle,
            payload: Some(locator),
            data_bytes: data,
        });
}

// A SkinnedMesh's geometry is decoded, merged into the shared skinned buffers,
// and uploaded; each mesh publishes a SkeletonPose for AnimationSystem to drive,
// registers its name so a runtime spawn can resolve the template, and a mesh
// with a capsule additionally gets a CharacterRig for PhysicsSystem.
#[test]
fn skinned_mesh_world_uploads_geometry_and_publishes_poses() {
    use crate::assets::{CharacterCapsule, SkinnedMesh};

    const RIGGED: AssetId = AssetId(840);
    const PLAIN: AssetId = AssetId(841);

    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    push_skinned_mesh(
        &mut b,
        RIGGED,
        SkinnedMesh {
            asset_id: RIGGED,
            material: Some(crate::ecs::MaterialHandle(0)),
            position: [5.0, 0.0, 0.0],
            capsule: Some(CharacterCapsule {
                half_height: 0.9,
                radius: 0.3,
            }),
            ..Default::default()
        },
        4,
    );
    push_skinned_mesh(
        &mut b,
        PLAIN,
        SkinnedMesh {
            asset_id: PLAIN,
            // No material: the texture reference resolves to the pool slot.
            texture: Some(TextureHandle(0)),
            ..Default::default()
        },
        3,
    );
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    {
        let s = lock(&state);
        assert!(
            s.saw(&Call::UploadSkinned {
                // The two meshes' vertices merge into one shared buffer.
                vertices: 7,
                draws: 2,
            }),
            "both meshes' geometry rides one upload: {:?}",
            s.calls
        );
        // Skinned draws are counted at construction so the GPU-cull buffers are
        // sized for the merged total, even though the geometry uploads after.
        assert_eq!(s.init.as_ref().unwrap().n_skinned, 2);
    }

    let ctx = world.ctx();
    // One SkeletonPose per mesh, each keyed to its handle and template draw.
    let mut poses: Vec<(u32, usize)> = ctx
        .query::<crate::assets::SkeletonPose>()
        .map(|p| (p.mesh_id.0, p.skinned_index))
        .collect();
    poses.sort();
    assert_eq!(poses, vec![(0, 0), (1, 1)]);
    // Only the capsule-carrying mesh gets a rig.
    let rigs: Vec<u32> = ctx
        .query::<crate::assets::CharacterRig>()
        .map(|r| r.target.0)
        .collect();
    assert_eq!(rigs, vec![0], "only the capsule mesh gets a character rig");
    // The name index the debug animation commands address a mesh by.
    let names = ctx
        .resource::<crate::gfx::skinned_mesh_map::SkinnedMeshNameIndex>()
        .expect("name index published before AnimationSystem inits");
    assert_eq!(names.0.get(&RIGGED).map(|h| h.0), Some(0));
    assert_eq!(names.0.get(&PLAIN).map(|h| h.0), Some(1));
    // Each template is registered under its mesh name, so a runtime SpawnRequest
    // resolves it the same way a static placement resolves.
    let by_name = ctx
        .resource::<crate::ecs::decompose::EntityByName>()
        .unwrap();
    assert!(by_name.0.contains_key(&RIGGED));
    assert!(by_name.0.contains_key(&PLAIN));
}

// `max_instances` pre-reserves hidden bind-pose copies of a skinned mesh, each
// with its OWN vertex region in the shared buffer: the GPU skin fold writes the
// deformed buffer keyed by global vertex index, so two live instances sharing a
// region would clobber each other's pose. A runtime spawn reveals one of these
// without growing any GPU buffer.
#[test]
fn skinned_instance_reserves_get_their_own_vertex_regions() {
    use crate::assets::SkinnedMesh;

    const HERO: AssetId = AssetId(842);
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    push_skinned_mesh(
        &mut b,
        HERO,
        SkinnedMesh {
            asset_id: HERO,
            max_instances: 3,
            ..Default::default()
        },
        4,
    );
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    let s = lock(&state);
    assert!(
        s.saw(&Call::UploadSkinned {
            // The template plus three copies, each carrying its own 4 vertices.
            vertices: 16,
            draws: 4,
        }),
        "each reserved copy owns a distinct vertex region: {:?}",
        s.calls
    );
    assert!(
        s.saw(&Call::SeedSkinnedInstancePool(3)),
        "the backend's instance pool is seeded with the reserved copies"
    );
    assert_eq!(s.init.as_ref().unwrap().n_skinned, 4);
}

// A SkinnedMesh referencing a material that is not in the table fails the world
// build: a silent fallback would render the character with the wrong surface.
#[test]
fn a_skinned_mesh_with_an_unknown_material_fails_init() {
    use crate::assets::SkinnedMesh;

    let (_state, hooks) = recording_hooks();
    let mut b = scene_builder();
    push_skinned_mesh(
        &mut b,
        AssetId(843),
        SkinnedMesh {
            asset_id: AssetId(843),
            material: Some(crate::ecs::MaterialHandle(99)),
            ..Default::default()
        },
        3,
    );
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(gs.failed);
    assert!(!backend_parked(&world));
}

// A SkinnedMesh whose geometry payload is malformed, and one with no compiled
// payload at all, each fail the world build rather than dropping the character.
#[test]
fn a_skinned_mesh_without_usable_geometry_fails_init() {
    use crate::assets::SkinnedMesh;
    use concinnity_core::ecs::{ResourceKind, ResourceRecord};

    let record = |b: &mut WorldBuilder, payload: Option<PayloadLocator>| {
        let data = postcard::to_allocvec(&(
            844u32,
            SkinnedMesh {
                asset_id: AssetId(844),
                ..Default::default()
            },
        ))
        .unwrap();
        b.skinned_records.push(ResourceRecord {
            resource_kind: ResourceKind::SkinnedMesh as u8,
            handle: 0,
            payload,
            data_bytes: data,
        });
    };

    // Malformed payload bytes (no "SKMV" magic).
    let mut b = scene_builder();
    let loc = b.payload(b"not-a-skinned-mesh");
    record(&mut b, Some(loc));
    let (_state, hooks) = recording_hooks();
    let mut world = b.build();
    assert!(init_graphics(&mut world, hooks).failed, "malformed payload");

    // No payload at all (the build never compiled the geometry).
    let mut b = scene_builder();
    record(&mut b, None);
    let (_state, hooks) = recording_hooks();
    let mut world = b.build();
    assert!(init_graphics(&mut world, hooks).failed, "no payload");

    // Baked data that does not decode as a SkinnedMesh record.
    let mut b = scene_builder();
    b.skinned_records.push(ResourceRecord {
        resource_kind: ResourceKind::SkinnedMesh as u8,
        handle: 0,
        payload: None,
        data_bytes: vec![0xFF; 3],
    });
    let (_state, hooks) = recording_hooks();
    let mut world = b.build();
    assert!(init_graphics(&mut world, hooks).failed, "undecodable data");
}

// The per-frame skinned push: AnimationSystem's poses reach the GPU each frame,
// and a runtime-spawned instance's Transform follows it. Both are skipped behind
// an open menu -- animation is frozen there, so the last upload still stands.
#[test]
fn skinned_poses_reach_the_backend_each_frame_and_freeze_behind_a_menu() {
    use crate::assets::SkinnedMesh;

    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    push_skinned_mesh(
        &mut b,
        AssetId(845),
        SkinnedMesh {
            asset_id: AssetId(845),
            ..Default::default()
        },
        3,
    );
    let mut world = b.build();
    let mut gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    step(&mut gs, &mut world);
    assert!(
        lock(&state).saw(&Call::UpdateSkinnedPose(0)),
        "the template's pose is pushed each frame"
    );

    // Freeze the world the way an open menu does.
    world.resources.insert(crate::ecs::MenuOverride(Some(true)));
    lock(&state).calls.clear();
    step(&mut gs, &mut world);
    assert!(
        !lock(&state).saw(&Call::UpdateSkinnedPose(0)),
        "pose uploads are skipped while a menu is open"
    );
}

impl WorldBuilder {
    // Bake a Material into the resource stream at the next handle and return it,
    // the way cook does (Materials are a data resource; all their data lives in
    // the baked bytes).
    fn push_material(&mut self, mat: Material) -> crate::ecs::MaterialHandle {
        let handle = self.material_records.len() as u32;
        self.material_records
            .push(concinnity_core::ecs::ResourceRecord {
                resource_kind: concinnity_core::ecs::ResourceKind::Material as u8,
                handle,
                payload: None,
                data_bytes: postcard::to_allocvec(&mat).unwrap(),
            });
        crate::ecs::MaterialHandle(handle)
    }
}

// Every texture -- albedo, normal map, terrain secondaries, emissive, packed ORM
// -- lives once in the shared pool at its own slot, so each reference resolves
// through the same handle-indexed lookup rather than a per-role pool. An unset
// reference falls back to its sentinel: slot 0 for the albedo-region maps (which
// the shader gates on) and the flat-normal fallback for the normal maps.
#[test]
fn every_material_texture_reference_resolves_to_its_shared_pool_slot() {
    use concinnity_core::ecs::ResourceKind;

    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    // Four more textures beside the quad's albedo (handle 0), so each reference
    // below points at a distinct slot.
    let slots: Vec<u32> = (0..5)
        .map(|_| b.push_resource(ResourceKind::Texture, &texture_payload(2, 2)))
        .collect();
    let mat = b.push_material(Material {
        albedo: Some(TextureHandle(slots[0])),
        normal_map: Some(TextureHandle(slots[1])),
        albedo_secondary: Some(TextureHandle(slots[2])),
        emissive_map: Some(TextureHandle(slots[3])),
        orm_map: Some(TextureHandle(slots[4])),
        terrain_blend: 0.5,
        ..Default::default()
    });
    b.push(Prop {
        asset_id: AssetId(850),
        mesh: Some(crate::ecs::MeshHandle(0)),
        material: Some(mat),
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    let s = lock(&state);
    let init = s.init.as_ref().unwrap();
    assert_eq!(init.texture_count, 6, "one pool holds every texture");
    let draw = init
        .draw_objects
        .iter()
        .find(|d| d.texture_slot == slots[0] as usize)
        .expect("the prop's draw resolved its albedo to the pool slot");
    assert_eq!(
        draw.normal_map_slot, slots[1] as usize,
        "a normal map resolves to the same shared pool, not a separate one"
    );
    assert_eq!(draw.material.albedo_secondary_index, slots[2]);
    assert_eq!(draw.material.emissive_map_index, slots[3]);
    assert_eq!(draw.material.orm_map_index, slots[4]);
    // An unset secondary normal selects the flat-normal fallback, one past the
    // last real texture, so a slope layer without its own map perturbs nothing.
    assert_eq!(draw.material.normal_secondary_index, 6);
}

// A material whose texture handle is past the end of the pool is a resolution
// error: cook validates every reference exists, so this only fires on a corrupt
// build, and failing loudly beats sampling an unrelated texture. Each reference
// role guards independently.
#[test]
fn a_material_texture_handle_past_the_pool_fails_init() {
    let out_of_range = Some(TextureHandle(99));
    let cases: Vec<(&str, Material)> = vec![
        (
            "albedo",
            Material {
                albedo: out_of_range,
                ..Default::default()
            },
        ),
        (
            "normal_map",
            Material {
                normal_map: out_of_range,
                ..Default::default()
            },
        ),
        (
            "albedo_secondary",
            Material {
                albedo_secondary: out_of_range,
                ..Default::default()
            },
        ),
        (
            "emissive_map",
            Material {
                emissive_map: out_of_range,
                ..Default::default()
            },
        ),
        (
            "orm_map",
            Material {
                orm_map: out_of_range,
                ..Default::default()
            },
        ),
    ];
    for (role, mat) in cases {
        let (_state, hooks) = recording_hooks();
        let mut b = scene_builder();
        b.push_material(mat);
        let mut world = b.build();
        assert!(
            init_graphics(&mut world, hooks).failed,
            "an out-of-range {role} handle must fail the build"
        );
    }
}

// A Material whose baked bytes do not decode fails the world build: the record
// is a build product, so a corrupt one means the build is broken.
#[test]
fn an_undecodable_material_record_fails_init() {
    let (_state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.material_records
        .push(concinnity_core::ecs::ResourceRecord {
            resource_kind: concinnity_core::ecs::ResourceKind::Material as u8,
            handle: 1,
            payload: None,
            data_bytes: vec![0xFF; 2],
        });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(gs.failed);
    assert!(!backend_parked(&world));
}

// The instanced vertex stage is optional -- it is only needed once a world
// declares an InstancedProp -- so its bytes are read and passed through when
// the shader's container carries the stage.
#[test]
fn an_instanced_vertex_shader_payload_reaches_the_backend() {
    use crate::assets::{InstanceTransform, InstancedProp};

    let (state, hooks) = recording_hooks();
    // scene_builder()'s shader carries no instanced stage, so build the same
    // world with a three-stage shader instead.
    let mut b = WorldBuilder::new();
    b.push(Window {
        title: "mock world".to_string(),
        width: 640,
        height: 360,
        ..Default::default()
    });
    b.push(GraphicsConfig {
        clear_color: [0.1, 0.2, 0.3, 1.0],
        ..Default::default()
    });
    b.push_shader(&[
        (ShaderKind::Vertex, b"vertex-shader-bytes"),
        (ShaderKind::Fragment, b"fragment-shader-bytes"),
        (
            ShaderKind::VertexInstanced,
            b"instanced-vertex-shader-bytes",
        ),
    ]);
    b.push(Camera3D::bake(Default::default()));
    b.push_textured_quad(MESH, TEX, MAT, PROP);
    b.push(InstancedProp {
        asset_id: AssetId(851),
        mesh: Some(crate::ecs::MeshHandle(0)),
        material: Some(crate::ecs::MaterialHandle(0)),
        instances: vec![InstanceTransform {
            position: [0.0; 3],
            rotation_deg: [0.0; 3],
            scale: [1.0; 3],
        }],
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed);
    assert_eq!(
        lock(&state).init.as_ref().unwrap().instanced_cluster_count,
        1
    );
}

// A Shader that never compiled (no payload locator) fails the world build
// rather than starting with a pipeline that cannot draw.
#[test]
fn a_shader_without_a_payload_fails_init() {
    let (_state, hooks) = recording_hooks();
    let mut b = WorldBuilder::new();
    b.push(Window::default());
    b.push(GraphicsConfig::default());
    b.push(Shader {
        locator: None,
        ..Default::default()
    });
    let mut world = b.build();
    assert!(
        init_graphics(&mut world, hooks).failed,
        "a Shader with no compiled payload must fail the build"
    );
}

// A skinned mesh's LOD alternates share its vertex region: the runtime skinned
// index buffer is u16, so every alternate's mesh-relative indices are rebased
// onto the same vertex base as LOD0. Each reserved instance copy gets its own
// rebased set against its own region.
#[test]
fn skinned_lod_alternates_rebase_onto_their_slot_vertex_region() {
    use crate::assets::SkinnedMesh;
    use concinnity_core::ecs::{ResourceKind, ResourceRecord};

    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    // LOD0 is 3 indices; one alternate adds 3 more, all over the same 4 vertices.
    let locator = b.payload(&skinned_payload(4, &[(10.0, vec![0, 1, 2])]));
    let data = postcard::to_allocvec(&(
        852u32,
        SkinnedMesh {
            asset_id: AssetId(852),
            max_instances: 1,
            ..Default::default()
        },
    ))
    .unwrap();
    b.skinned_records.push(ResourceRecord {
        resource_kind: ResourceKind::SkinnedMesh as u8,
        handle: 0,
        payload: Some(locator),
        data_bytes: data,
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    let s = lock(&state);
    assert!(
        s.saw(&Call::UploadSkinned {
            // Template + one reserved copy, each with its own 4 vertices; the LOD
            // alternate reuses its slot's region rather than duplicating it.
            vertices: 8,
            draws: 2,
        }),
        "an LOD alternate shares its slot's vertex region: {:?}",
        s.calls
    );
    assert!(s.saw(&Call::SeedSkinnedInstancePool(1)));
}

// A world declaring no Window / GraphicsConfig still builds against the system's
// defaults: the components are optional, and their absence is not an error.
#[test]
fn a_world_without_a_window_or_config_builds_on_the_defaults() {
    let (state, hooks) = recording_hooks();
    let mut b = WorldBuilder::new();
    b.push_shaders();
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed);
    let s = lock(&state);
    let init = s.init.as_ref().unwrap();
    assert_eq!(init.frames_in_flight, 2, "the system's own default");
    assert_eq!(init.window_width, Window::default().width);
}

// The step exits Done when the backend has been taken out of the world (the
// editor's live transplant), rather than drawing against nothing.
#[test]
fn a_step_without_a_parked_backend_finishes_the_system() {
    let (_state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);
    crate::ecs::ActiveRenderBackend::take(&mut world.resources).expect("backend was parked");

    assert_eq!(step(&mut gs, &mut world), StepResult::Done);
}

// A ReparentRequest whose child name does not resolve finds nothing to move (a
// reparent naming an entity a despawn already removed is a no-op, not a panic).
#[test]
fn reparent_request_with_an_unresolved_child_is_skipped() {
    let (_state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let _gs = init_graphics(&mut world, hooks);

    {
        let mut ctx = world.ctx();
        ctx.events_mut::<ReparentRequest>().send(ReparentRequest {
            child: AssetId(901),
            parent: Some(PROP),
        });
    }
    let mut spawn = crate::spawn::SpawnSystem::new();
    spawn_step(&mut spawn, &mut world);

    let child = entity_named(&mut world, PROP);
    assert!(
        world
            .ctx()
            .get::<crate::assets::Children>(child)
            .is_none_or(|c| c.0.is_empty()),
        "nothing was parented under the named parent"
    );
}

// A SpawnRequest or Spawner naming a template that does not resolve is skipped:
// there is nothing to copy, so no draw slot is cloned.
#[test]
fn a_spawn_naming_an_unknown_template_is_skipped() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let _gs = init_graphics(&mut world, hooks);
    {
        let mut ctx = world.ctx();
        ctx.events_mut::<SpawnRequest>().send(SpawnRequest {
            template: AssetId(902),
            name: Some(AssetId(903)),
            transform: Transform::default(),
            lifetime_secs: None,
        });
        let spawner = ctx.components.spawn();
        ctx.insert(
            spawner,
            crate::assets::Spawner {
                template: AssetId(904),
                interval: 1.0,
                lifetime: 0.0,
                elapsed: 1.0,
                count: 0,
            },
        );
    }
    let mut spawn = crate::spawn::SpawnSystem::new();
    spawn_step(&mut spawn, &mut world);

    assert!(
        !lock(&state)
            .calls
            .iter()
            .any(|c| matches!(c, Call::CloneStaticDrawObject { .. })),
        "an unresolvable template clones nothing"
    );
    assert_eq!(
        world.ctx().query::<RenderHandle>().count(),
        1,
        "only the original"
    );
}

// A spawn naming a SKINNED template dispatches to the instance-pool path rather
// than cloning a static draw slot: a skinned copy claims one of the hidden
// bind-pose instances reserved at init. With every reserved slot taken (here:
// none reserved) the spawn yields nothing rather than growing a GPU buffer.
#[test]
fn a_spawn_naming_a_skinned_template_takes_the_instance_pool_path() {
    use crate::assets::SkinnedMesh;

    const HERO: AssetId = AssetId(853);
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    push_skinned_mesh(
        &mut b,
        HERO,
        SkinnedMesh {
            asset_id: HERO,
            // No instances reserved, so the pool has nothing to hand out.
            max_instances: 0,
            ..Default::default()
        },
        3,
    );
    let mut world = b.build();
    let _gs = init_graphics(&mut world, hooks);
    let poses_before = world.ctx().query::<crate::assets::SkeletonPose>().count();

    {
        let mut ctx = world.ctx();
        ctx.events_mut::<SpawnRequest>().send(SpawnRequest {
            template: HERO,
            name: Some(AssetId(905)),
            transform: Transform::default(),
            lifetime_secs: Some(2.0),
        });
        // A cadence spawn of the same skinned template takes the same path.
        let spawner = ctx.components.spawn();
        ctx.insert(
            spawner,
            crate::assets::Spawner {
                template: HERO,
                interval: 1.0,
                lifetime: 2.0,
                elapsed: 1.0,
                count: 0,
            },
        );
    }
    let mut spawn = crate::spawn::SpawnSystem::new();
    spawn_step(&mut spawn, &mut world);

    assert!(
        !lock(&state)
            .calls
            .iter()
            .any(|c| matches!(c, Call::CloneStaticDrawObject { .. })),
        "a skinned template never clones a static draw slot"
    );
    assert_eq!(
        world.ctx().query::<crate::assets::SkeletonPose>().count(),
        poses_before,
        "an exhausted instance pool spawns nothing rather than growing a buffer"
    );
}

// The hot-reload seams the `cn debug` binary drives through. The library never
// calls them, so they are exercised here: the source catalogues are captured
// only under `cn debug` (this world is a plain build, so there are none), and the
// apply parts hand out a disjoint mutable screen of the backend + bookkeeping.
#[test]
fn hot_reload_seams_hand_out_the_backend_and_the_captured_sources() {
    use crate::assets::VolumetricFog;

    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(VolumetricFog {
        enabled: true,
        ..Default::default()
    });
    let mut world = b.build();
    let mut gs = init_graphics(&mut world, hooks);

    assert!(
        gs.take_hot_reload_sources().is_none(),
        "a plain build captures no source catalogue"
    );

    let mut backend = crate::ecs::ActiveRenderBackend::take(&mut world.resources).unwrap();
    let parts = gs.hot_reload_apply_parts(backend.as_mut());
    assert!(
        parts.last_fog_settings.is_some(),
        "the reload pass dedupes against the fog init pushed to the backend"
    );
    assert!(
        parts.world_reload.is_none(),
        "the texture-name map is captured only under cn debug"
    );
    parts.backend.wait_idle();
    assert!(
        lock(&state).saw(&Call::WaitIdle),
        "the parts reach the backend"
    );
}

// Shrinkable seed VRAM: by default the whole streamed mesh set is baked into the
// shared buffers, so streaming reuses space but never shrinks GPU memory. When
// the residency cap is smaller than the streamed set, the resident geometry is
// compacted and a smaller seed headroom -- sized to the cap-many largest meshes
// -- is reserved for the streamed ones, so the GPU buffers are born small.
#[test]
fn a_residency_cap_below_the_streamed_set_reserves_a_smaller_seed() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    // Three streamable quads against a cap of one, so the planner has a set to
    // shrink (the cap plus its margin must still fall short of the set).
    for i in 0..2 {
        b.push_textured_quad(
            AssetId(10 + i),
            AssetId(20 + i),
            AssetId(30 + i),
            AssetId(860 + i),
        );
    }
    b.push(StreamingConfig {
        mesh_cap: 1,
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    let s = lock(&state);
    assert!(
        s.saw(&Call::SeedMeshStreaming),
        "the backend's sub-allocator is seeded with the smaller headroom block"
    );
    let init = s.init.as_ref().unwrap();
    assert_eq!(init.draw_objects.len(), 3);
    // The compaction reserves headroom for the two most-resident meshes rather
    // than baking all three quads (4 vertices each) in.
    assert!(
        init.vertex_count < 12,
        "the seed buffer is born smaller than the whole streamed set: {}",
        init.vertex_count
    );
}

// A Story's stage images must be resident even though no Sprite references them
// yet: the story system swaps them onto the stage sprites at runtime, so they are
// gathered into the text-atlas pool alongside the sprite textures.
#[test]
fn story_stage_images_are_resident_before_any_sprite_references_them() {
    use crate::assets::{Story, StoryImage, StoryNode, StoryPage, StoryStage};
    use concinnity_core::ecs::ResourceKind;

    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    // Four textures beside the quad's albedo, one per stage position.
    let slots: Vec<u32> = (0..4)
        .map(|_| b.push_resource(ResourceKind::Texture, &texture_payload(2, 2)))
        .collect();
    let image = |slot: u32| {
        Some(StoryImage {
            texture: TextureHandle(slot),
            ..Default::default()
        })
    };
    b.push(Story {
        asset_id: AssetId(870),
        nodes: vec![StoryNode {
            pages: vec![StoryPage {
                stage: StoryStage {
                    bg: image(slots[0]),
                    left: image(slots[1]),
                    ..Default::default()
                },
                ..Default::default()
            }],
            // A choice menu carries its own stage dressing too.
            choice_stage: StoryStage {
                center: image(slots[2]),
                right: image(slots[3]),
                ..Default::default()
            },
            ..Default::default()
        }],
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    assert_eq!(
        lock(&state).init.as_ref().unwrap().text_atlas_count,
        4,
        "every stage image is decoded into the pool ahead of the swap"
    );
    let overlay = world
        .resources
        .get::<crate::gfx::overlay::OverlayAssets>()
        .unwrap();
    for slot in &slots {
        assert!(
            overlay
                .sprite_texture_slots
                .contains_key(&TextureHandle(*slot)),
            "stage texture {slot} never became resident"
        );
    }
}

// Viewport picking is opt-in: with a PickIndex resource present at init (the
// editor injects one), init captures a candidate per prop and the step
// publishes the index with world-space bounds; without it, nothing is captured
// and nothing is published.
#[test]
fn pick_index_publishes_world_bounds_only_when_opted_in() {
    // Opted in: the quad prop at [1,2,3] indexes with its translated bounds.
    let (_state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    world.resources.insert(crate::ecs::PickIndex::default());
    let mut gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);
    assert_eq!(gs.pick_candidates.len(), 1, "one candidate per prop");
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);

    let index = world
        .resources
        .get::<crate::ecs::PickIndex>()
        .expect("step publishes the index");
    assert_eq!(index.entries.len(), 1);
    let e = &index.entries[0];
    assert_eq!(e.asset_id, PROP);
    // Unit quad local bounds x/z in [0,1], y = 0, translated by [1,2,3].
    assert_eq!(e.bb_min, [1.0, 2.0, 3.0]);
    assert_eq!(e.bb_max, [2.0, 2.0, 4.0]);

    // Not opted in: no candidates, and the resource never appears.
    let (_state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);
    assert!(gs.pick_candidates.is_empty());
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    assert!(world.resources.get::<crate::ecs::PickIndex>().is_none());
}

// The editor's per-frame EditorHidden set collapses a hidden prop's draw slots
// to a degenerate model matrix (nothing rasterizes) and drops it from the
// published PickIndex; clearing the set restores both the next step.
#[test]
fn editor_hidden_collapses_draws_and_skips_the_pick_index() {
    const COLLAPSED: [[f32; 4]; 4] = [[0.0; 4], [0.0; 4], [0.0; 4], [0.0, 0.0, 0.0, 1.0]];
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    world.resources.insert(crate::ecs::PickIndex::default());
    let mut gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    world
        .resources
        .insert(crate::ecs::EditorHidden([PROP].into_iter().collect()));
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    {
        let index = world.resources.get::<crate::ecs::PickIndex>().unwrap();
        assert!(index.entries.is_empty(), "a hidden prop is not pickable");
        let s = lock(&state);
        let model = s.models.get(&0).expect("slot 0 model pushed");
        assert_eq!(*model, COLLAPSED, "the hidden prop's slot is degenerate");
    }

    world.resources.insert(crate::ecs::EditorHidden::default());
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    let index = world.resources.get::<crate::ecs::PickIndex>().unwrap();
    assert_eq!(index.entries.len(), 1, "clearing the set restores the prop");
    let s = lock(&state);
    let model = s.models.get(&0).unwrap();
    assert_ne!(*model, COLLAPSED, "the real transform is pushed again");
}
