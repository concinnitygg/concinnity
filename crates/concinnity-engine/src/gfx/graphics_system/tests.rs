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
    Scene, SceneCommand, SceneReel, ShaderKind, ShaderStage, SpawnRequest, Sprite, StreamingConfig,
    Transform, Window,
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
}

impl WorldBuilder {
    fn new() -> Self {
        Self {
            components: ComponentStorage::default(),
            section: Vec::new(),
            texture_records: Vec::new(),
            material_records: Vec::new(),
            mesh_records: Vec::new(),
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

    // The vertex + fragment ShaderStage pair every renderable world needs.
    // The payload bytes are opaque to the mock, so any bytes serve.
    fn push_shaders(&mut self) {
        let vert = self.payload(b"vertex-shader-bytes");
        self.push(ShaderStage {
            kind: ShaderKind::Vertex,
            locator: Some(vert),
            ..Default::default()
        });
        let frag = self.payload(b"fragment-shader-bytes");
        self.push(ShaderStage {
            kind: ShaderKind::Fragment,
            locator: Some(frag),
            ..Default::default()
        });
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

    fn build(self) -> TestWorld {
        let mut resources = Resources::new();
        // The renderer reads the shared texture pool from this table, exactly as
        // the runtime does after loading the blob's resource stream.
        resources.insert(crate::resource::TextureTable::from_records(
            &self.texture_records,
        ));
        resources.insert(crate::resource::MaterialTable::from_records(
            &self.material_records,
        ));
        resources.insert(crate::resource::MeshTable::from_records(&self.mesh_records));
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
    let mut bytes = Vec::with_capacity(8 + (w * h * 4) as usize);
    bytes.extend_from_slice(&w.to_le_bytes());
    bytes.extend_from_slice(&h.to_le_bytes());
    bytes.extend(vec![0x7Fu8; (w * h * 4) as usize]);
    bytes
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
// then GraphicsSystem init with the injected hooks.
fn init_graphics(world: &mut TestWorld, hooks: TestHooks) -> GraphicsSystem {
    let mut gs = GraphicsSystem::new();
    gs.test_hooks = Some(hooks);
    let mut ctx = world.ctx();
    crate::ecs::decompose::run(&mut ctx);
    gs.run_init(&mut ctx);
    gs
}

fn step(gs: &mut GraphicsSystem, world: &mut TestWorld) -> StepResult {
    let mut ctx = world.ctx();
    gs.run_step(&mut ctx)
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
    assert!(gs.backend.is_some(), "mock backend installed");

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

// The built backend can be taken exactly once (the `cn editor` transplant).
#[test]
fn take_backend_yields_the_backend_once() {
    let (_state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);
    assert!(gs.take_backend().is_some(), "the built backend is taken");
    assert!(gs.take_backend().is_none(), "and only once");
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
    let mut gs_a = init_graphics(&mut world_a, hooks_a);
    let backend_a = gs_a.take_backend().expect("world A built a backend");

    // Transplant it into world B (a different world, same swapchain config).
    let (state_b, hooks_b) = recording_hooks();
    let mut world_b = titled_scene("world B").build();
    world_b
        .resources
        .insert(crate::ecs::PendingBackend(backend_a));
    let gs_b = init_graphics(&mut world_b, hooks_b);

    assert!(!gs_b.failed);
    assert!(gs_b.backend.is_some(), "the reused backend is installed");
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
    assert!(gs.backend.is_some());
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
    let mut gs_a = init_graphics(&mut world_a, hooks_a);
    let backend_a = gs_a.take_backend().unwrap();
    state_a.lock().unwrap().fail_reload = Some("boom".to_string());

    let (_state_b, hooks_b) = recording_hooks();
    let mut world_b = scene_builder().build();
    world_b
        .resources
        .insert(crate::ecs::PendingBackend(backend_a));
    let gs_b = init_graphics(&mut world_b, hooks_b);

    assert!(gs_b.failed, "a failed reload marks the system failed");
    assert!(gs_b.backend.is_none());
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
    assert!(!s.saw(&Call::SetWindowMode));
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
fn missing_shader_stage_fails_init() {
    let (state, hooks) = recording_hooks();
    let mut b = WorldBuilder::new();
    b.push(Window::default());
    b.push(GraphicsConfig::default());
    b.push_textured_quad(MESH, TEX, MAT, PROP);
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(gs.failed, "no vertex ShaderStage: init must fail");
    assert!(gs.backend.is_none());
    assert!(lock(&state).init.is_none(), "backend never constructed");
}

#[test]
fn scene_reel_applies_start_scene_visibility() {
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
            duration_secs: None,
            transition: "Cut".to_string(),
            camera_shot: None,
        });
        ctx.push(Scene {
            asset_id: scene_b,
            duration_secs: None,
            transition: "Cut".to_string(),
            camera_shot: None,
        });
        ctx.push(SceneReel {
            asset_id: AssetId(22),
            scenes: vec![scene_a, scene_b],
            looping: false,
            start_index: 0,
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
        view: Some(AssetId(41)),
        ..Default::default()
    });
    let mut world = b.build();
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
    assert!(gs.menu_active_prev, "pacer sees the menu next frame");
    {
        let ctx = world.ctx();
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

#[test]
fn max_frames_finishes_the_system() {
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
    assert_eq!(step(&mut gs, &mut world), StepResult::Done);
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
            name: AssetId(900),
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

#[test]
fn streaming_init_evicts_streamable_slots() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(StreamingConfig::default());
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    let s = lock(&state);
    // The albedo pool slot is evicted to a placeholder at init; the mesh
    // keeps its build-time region (cap covers the whole set) but is evicted
    // so the streamer brings it back nearest-first.
    assert!(s.saw(&Call::EvictTextureSlot(0)));
    assert!(s.saw(&Call::EvictMesh(0)));
    assert!(gs.texture_streamer.is_some());
    assert!(gs.mesh_streamer.is_some());
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
    let (resident, pending, unloaded) = gs.texture_streamer.as_ref().unwrap().stats();
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
    let (resident, pending, unloaded) = gs.mesh_streamer.as_ref().unwrap().stats();
    assert_eq!((resident, pending, unloaded), (1, 0, 0));
}
