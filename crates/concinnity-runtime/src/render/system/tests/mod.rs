// GraphicsSystem unit tests: drive its init and run_step against core's
// recording backend (built through `hooks`) and a World over an in-memory blob. No GPU device is created and the on-disk settings store is
// never read or written: a test's settings are the `StoredSettings` resource
// it inserts, and the injection seam supplies the GPU profile and backend.

use concinnity_core::bake::texture;
use concinnity_core::components::{
    BackgroundOverrides, Camera3D, CharacterRig, FrameInput, GlobalTransform, GraphicsConfig,
    HitRegion, Material, Prop, RenderHandle, Scene, SceneCommand, Shader, ShaderPrograms,
    SkeletonPose, SkinnedMesh, Sprite, StreamingConfig, TextLabel, UiAction, Window,
    assets::compiled_programs,
};
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{
    ComponentSlot, FontHandle, FrameRateCap, MaterialHandle, MenuActive, MenuOverride, MeshHandle,
    PayloadLocator, Ref, ScreenStack, StepResult, TextureHandle, World,
};
use concinnity_core::gfx::render_types::{DrawIndex, SkinnedIndex};
use concinnity_core::gfx::{chunk_coord, mesh_payload};
use concinnity_core::launch::{LaunchArgs, LaunchRequest};
use concinnity_core::math::SaturatingFrom;
use concinnity_core::render::backend::init::SwapchainConfig;
use concinnity_core::render::backend::{GpuProfile, GpuTier, GpuVendor};
use concinnity_core::render::{backend, backend::error, overlay::text};
use concinnity_core::resource::{
    ColorLutTable, EnvironmentMapTable, FontTable, MaterialTable, MeshTable, SkinnedMeshTable,
    TextureTable,
};
use concinnity_core::settings::{ResolvedGraphics, SettingKey};
use concinnity_host::store::blob::BlobData;
use std::sync::{Arc, Mutex};

use super::GraphicsSystem;
use concinnity_core::render::backend::recording::{
    Call, RecordingBackend, RecordingState, SharedRecording,
};
use concinnity_core::settings::quality::QualityPreset;
use hooks::{TestHooks, recording_hooks, recording_hooks_with};

mod assets;
mod frame;
pub(super) mod hooks;
mod init;
mod settings;
mod settings_rows;
mod skinned;
mod streaming;

const MESH: AssetId = AssetId(1);

const TEX: AssetId = AssetId(2);

const MAT: AssetId = AssetId(3);

const PROP: AssetId = AssetId(4);

// A component add held until `build` has made the world it goes into.
type PendingAdd = Box<dyn FnOnce(&mut World)>;

// Accumulates components and payload bytes, then seals into a World whose
// single in-memory blob section serves every locator it handed out. The
// components are added once the section is sealed into the world's payload
// store.
struct WorldBuilder {
    pending: Vec<PendingAdd>,
    section: Vec<u8>,
    texture_records: Vec<concinnity_core::ecs::ResourceRecord>,
    // Material data-resource records; each `push_textured_quad` bakes one Material
    // into `data_bytes` at the next handle, mirroring cook.
    material_records: Vec<concinnity_core::ecs::ResourceRecord>,
    // Mesh resource records; each `push_textured_quad` Mesh takes the next
    // handle in declaration order, matching the runtime's mesh-source table.
    mesh_records: Vec<concinnity_core::ecs::ResourceRecord>,
    // The singleton / pooled media kinds `push_resource` fills: font atlases,
    // the color-grading LUT, the IBL environment, and skinned geometry. Each
    // stays empty unless a test authors one.
    font_records: Vec<concinnity_core::ecs::ResourceRecord>,
    color_lut_records: Vec<concinnity_core::ecs::ResourceRecord>,
    env_map_records: Vec<concinnity_core::ecs::ResourceRecord>,
    skinned_records: Vec<concinnity_core::ecs::ResourceRecord>,
}

impl WorldBuilder {
    fn new() -> Self {
        Self {
            pending: Vec::new(),
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
        self.pending.push(Box::new(move |world| {
            world.push(c);
        }));
    }

    fn push_identified<C: ComponentSlot>(&mut self, id: AssetId, c: C) {
        self.pending.push(Box::new(move |world| {
            world.push_identified(id, c);
        }));
    }

    // One Shader whose payload carries the given compiled programs, one per
    // entry. The artifact bytes are opaque to the mock, so any bytes serve.
    fn push_shader(&mut self, entries: &[&str]) {
        let container = ShaderPrograms {
            name: "shader".into(),
            vertex: None,
            fragment: concinnity_core::components::ShaderSource {
                path: "shaders/wall.hlsl".into(),
                text: "float4 shade(VertexOut v, GpuObjectData od) { return 1.0; }".into(),
            },
            programs: entries
                .iter()
                .map(|e| compiled_programs::CompiledProgram {
                    entry: e.to_string(),
                    source_digest: 1,
                    artifact: b"program-bytes".to_vec(),
                })
                .collect(),
        };
        let locator = self.payload(&container.encode().expect("encode shader payload"));
        self.push(Shader {
            locator: Some(locator),
            ..Default::default()
        });
    }

    // A Shader with the two programs a world default needs.
    fn push_shaders(&mut self) {
        self.push_shader(&["vertex_main_bindless", "fragment_main_bindless"]);
    }

    // One quad Mesh + a Texture-backed Material + a Prop placing it.
    fn push_textured_quad(&mut self, mesh: AssetId, _tex: AssetId, mat: AssetId, prop: AssetId) {
        let mesh_loc = self.payload(&quad_mesh_payload());
        // Meshes are resources now: the payload rides the resource stream at the
        // next mesh handle; the Prop below references it by that handle, as cook
        // resolves a `.mesh` name. `mesh`'s asset id is unused.
        let _ = mesh;
        let mesh_handle = MeshHandle::new(u32::saturating_from(self.mesh_records.len()));
        self.mesh_records
            .push(concinnity_core::ecs::ResourceRecord {
                resource_kind: concinnity_core::ecs::ResourceKind::Mesh,
                handle: mesh_handle.0,
                payload: Some(mesh_loc),
                data_bytes: Vec::new(),
            });
        let tex_loc = self.payload(&texture_payload(2, 2));
        // Textures are resources now: the payload rides the resource stream at
        // the next texture handle. The Material references `TextureHandle::new(0)` (the
        // first texture) regardless of any texture's asset id.
        let handle = u32::saturating_from(self.texture_records.len());
        self.texture_records
            .push(concinnity_core::ecs::ResourceRecord {
                resource_kind: concinnity_core::ecs::ResourceKind::Texture,
                handle,
                payload: Some(tex_loc),
                data_bytes: Vec::new(),
            });
        // Materials are a data resource: bake this one into `data_bytes` at the
        // next handle (serialized like cook does), and reference it by that
        // handle. `mat`'s asset id is unused now that the reference is a handle.
        let _ = mat;
        let mat_handle = u32::saturating_from(self.material_records.len());
        let mat_bytes = postcard::to_allocvec(&Material {
            albedo: Some(TextureHandle::new(0)),
            ..Default::default()
        })
        .unwrap();
        self.material_records
            .push(concinnity_core::ecs::ResourceRecord {
                resource_kind: concinnity_core::ecs::ResourceKind::Material,
                handle: mat_handle,
                payload: None,
                data_bytes: mat_bytes,
            });
        self.push_identified(
            prop,
            Prop {
                mesh: Some(mesh_handle),
                material: Some(MaterialHandle::new(mat_handle)),
                position: [1.0, 2.0, 3.0],
                ..Default::default()
            },
        );
    }

    fn build(mut self) -> World {
        let mut world = World::from_payloads(Box::new(BlobData::new(vec![Some(self.section)])));
        // The renderer reads the shared texture pool from this table, exactly as
        // the runtime does after loading the blob's resource stream.
        world.insert_resource(TextureTable::from_records(&mut self.texture_records));
        world.insert_resource(MaterialTable::from_records(&mut self.material_records));
        world.insert_resource(MeshTable::from_records(&mut self.mesh_records));
        world.insert_resource(FontTable::from_records(&mut self.font_records));
        world.insert_resource(ColorLutTable::from_records(&mut self.color_lut_records));
        world.insert_resource(EnvironmentMapTable::from_records(&mut self.env_map_records));
        world.insert_resource(SkinnedMeshTable::from_records(&mut self.skinned_records));
        for add in self.pending {
            add(&mut world);
        }
        world
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
    mesh_payload::serialize(&vertices, &[0, 1, 2, 0, 2, 3])
}

// A w x h mid-gray RGBA image in the compiled texture payload format.
fn texture_payload(w: u32, h: u32) -> Vec<u8> {
    texture::serialize(&texture::TextureImage::rgba8(
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
    configured_scene(title, GraphicsConfig::default())
}

// The representative world with `graphics` as its settings, over the scene's
// background color.
fn configured_scene(title: &str, graphics: GraphicsConfig) -> WorldBuilder {
    let mut b = WorldBuilder::new();
    b.push(Window {
        title: title.to_string(),
        width: 640,
        height: 360,
        ..Default::default()
    });
    b.push(GraphicsConfig {
        background: BackgroundOverrides {
            color: Some([0.1, 0.2, 0.3, 1.0]),
        },
        ..graphics
    });
    b.push_shaders();
    b.push(Camera3D::bake(Default::default()));
    b.push_textured_quad(MESH, TEX, MAT, PROP);
    b
}

// Run the same pre-init pass World::start performs (Prop decomposition),
// then GraphicsSystem init with the injected hooks. A successful init parks
// the built backend in the world's `ActiveRenderBackend` slot.
fn init_graphics(world: &mut World, hooks: TestHooks) -> GraphicsSystem {
    let mut gs = GraphicsSystem::new(None);
    gs.test_hooks = Some(hooks);
    let mut ctx = world.context();
    concinnity_core::ecs::decompose::run(&mut ctx);
    concinnity_core::ecs::System::init(&mut gs, &mut ctx);
    gs
}

// Whether the world's parked backend slot currently holds a backend.
fn backend_parked(world: &World) -> bool {
    world
        .resource::<concinnity_core::ecs::ActiveRenderBackend>()
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
fn step(gs: &mut GraphicsSystem, world: &mut World) -> StepResult {
    use concinnity_core::ecs::System;
    let mut ctx = world.context();
    concinnity_core::render::overlay::OverlaySystem::new().step(&mut ctx);
    concinnity_core::spawn::SpawnSystem::new().step(&mut ctx);
    concinnity_core::settings::system::SettingsSystem::new().step(&mut ctx);
    concinnity_core::render::streaming::pools::StreamingSystem::default().step(&mut ctx);
    let result = gs.run_step(&mut ctx);
    if result != StepResult::Stop {
        // InputSystem's init seeds the one FrameInput row; the fresh
        // per-step instance here needs the same seed without the init's
        // settings-file read.
        if ctx.query::<FrameInput>().next().is_none() {
            ctx.push(FrameInput::default());
        }
        concinnity_core::input::system::InputSystem::new().step(&mut ctx);
    }
    result
}

fn lock(state: &SharedRecording) -> impl core::ops::DerefMut<Target = RecordingState> + '_ {
    state.lock()
}

// Register a SkinnedMesh resource over `skinned_payload(n, &[])`.
fn push_skinned_mesh(b: &mut WorldBuilder, name: AssetId, sm: SkinnedMesh, n: u16) {
    push_skinned_payload(b, name, sm, &skinned_payload(n, &[]));
}

// One settings row: the setting-action HitRegion the menu builds plus
// the value TextLabel it points at. Init syncs the label to the live value while
// the regions are still present (UiInputSystem drains them afterwards).
fn push_settings_row(b: &mut WorldBuilder, key: &str, verb: &str, label: AssetId) {
    b.push(HitRegion {
        action: act(serde_json::json!({"setting": {"key": key, "verb": verb}})),
        label: Some(Ref::new(label)),
        ..Default::default()
    });
    b.push_identified(
        label,
        TextLabel {
            content: "<placeholder>".to_string(),
            color: LIT,
            ..Default::default()
        },
    );
}

// The content of the TextLabel with `id`.
fn label_text(world: &mut World, id: AssetId) -> String {
    world
        .context()
        .get_by_id::<TextLabel>(id)
        .expect("label present")
        .content
        .clone()
}

// A test action from its text form, with integer targets.
fn act(action: serde_json::Value) -> Option<UiAction> {
    Some(serde_json::from_value(action).unwrap())
}

// Quality effects at their defaults, adjusted by `f`.
fn effects(f: impl FnOnce(&mut ResolvedGraphics)) -> ResolvedGraphics {
    let mut effects = ResolvedGraphics::default();
    f(&mut effects);
    effects
}

// The scene world setting caller-shaped quality effects.
fn effects_scene(effects: ResolvedGraphics) -> WorldBuilder {
    configured_scene("mock world", effects_config(effects))
}

// The resolved settings snapshot init hands to SettingsSystem: the live values
// every settings row displays and cycles, after the world's config, the
// persisted overrides, and the preset ceiling have all settled.
fn settings_state(world: &World) -> &concinnity_core::settings::system::SettingsState {
    world
        .resource::<concinnity_core::settings::system::SettingsSlot>()
        .expect("SettingsSlot parked at init")
        .0
        .as_ref()
        .expect("SettingsState in its slot between steps")
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

// A two-triangle skinned strip bound to a two-joint chain, in the compiled
// skinned-payload format. `n` vertices span x = 0..n-1, each fully weighted to
// joint 0, so the bind-pose AABB is predictable.
fn skinned_payload(n: u16, lods: &[(f32, Vec<u16>)]) -> Vec<u8> {
    skinned_payload_with_morphs(n, lods, &mesh_payload::PayloadMorphs::default())
}

// Register a SkinnedMesh resource: the placement / material / spawn reserve ride
// the baked `data_bytes`, the geometry + skeleton ride the compiled payload, and
// the table index IS the mesh's `SkinnedMeshHandle`.
fn push_skinned_payload(b: &mut WorldBuilder, name: AssetId, sm: SkinnedMesh, payload: &[u8]) {
    let locator = b.payload(payload);
    let handle = u32::saturating_from(b.skinned_records.len());
    let data = postcard::to_allocvec(&(name.0, sm)).unwrap();
    b.skinned_records
        .push(concinnity_core::ecs::ResourceRecord {
            resource_kind: concinnity_core::ecs::ResourceKind::SkinnedMesh,
            handle,
            payload: Some(locator),
            data_bytes: data,
        });
}

// The authored (non-grayed) row color the tests below start every label at, so a
// gray-out is visible as a change away from it.
const LIT: [f32; 3] = [0.9, 0.9, 0.9];

// A `GraphicsConfig` setting every field of the quality-governed sections to
// `effects`'.
fn effects_config(effects: ResolvedGraphics) -> GraphicsConfig {
    GraphicsConfig {
        ambient_occlusion: effects.ambient_occlusion.into(),
        reflections: effects.reflections.into(),
        indirect_lighting: effects.indirect_lighting.into(),
        anti_aliasing: effects.anti_aliasing.into(),
        exposure: effects.exposure.into(),
        ..Default::default()
    }
}

// `skinned_payload` carrying a morph-target block.
fn skinned_payload_with_morphs(
    n: u16,
    lods: &[(f32, Vec<u16>)],
    morphs: &mesh_payload::PayloadMorphs,
) -> Vec<u8> {
    use concinnity_core::gfx::mesh_payload::{
        PayloadJoint, SkinnedVertex, serialize_skinned_with_lods,
    };
    let vertices: Vec<SkinnedVertex> = (0..n)
        .map(|i| SkinnedVertex {
            pos: [f32::from(i), 0.0, 0.0],
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
    serialize_skinned_with_lods(
        &vertices,
        &[0, 1, 2],
        &[joint("root", -1), joint("child", 0)],
        morphs,
        lods,
    )
}
