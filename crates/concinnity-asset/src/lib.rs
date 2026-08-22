//! concinnity-asset
//!
//! The authored-data schema for the engine's assets: the plain structs, enums,
//! and serde `Default`s a world.jsonl declares, plus the identity (`AssetId`) and
//! typed reference (`AssetRef<T>`) primitives they are built from.
//!
//! This crate holds DATA ONLY. All behavior lives above it: the ECS `Component`
//! impls in concinnity-core, the name -> id interner the resolver seam points at
//! in concinnity-cpu, validation and companion expansion in concinnity-cook. The
//! crate is `#![no_std]` (using only `core` + `alloc`) with serde as its single
//! dependency, so it can never pull in engine logic and is consumable from doc
//! tooling and external authoring tools alike.

#![no_std]

extern crate alloc;
#[cfg(test)]
extern crate std;

mod handle;
mod id;
mod locator;
mod reference;
mod resolver;
#[cfg(test)]
mod test_support;

pub use handle::{
    AudioClipHandle, ColorLutHandle, CubemapTextureHandle, EnvironmentMapHandle, FontHandle,
    MaterialHandle, MeshHandle, ShaderHandle, SkinnedMeshHandle, TextureHandle,
    de_audio_clip_handle_vec, de_opt_audio_clip_handle, de_opt_font_handle, de_opt_material_handle,
    de_opt_mesh_handle, de_opt_shader_handle, de_opt_skinned_mesh_handle, de_opt_texture_handle,
    de_texture_handle,
};
pub use id::{AssetId, de_opt_asset_ref};
pub use locator::PayloadLocator;
pub use reference::{AssetRef, de_opt_asset_ref_typed};
pub use resolver::{
    HandleResolveFn, ResolveFn, set_audio_clip_handle_resolver, set_font_handle_resolver,
    set_material_handle_resolver, set_mesh_handle_resolver, set_name_resolver,
    set_shader_handle_resolver, set_skinned_mesh_handle_resolver, set_texture_handle_resolver,
};

// Asset data schema: one module per asset type, mirroring the impl-side layout
// under concinnity-core/src/assets. Each holds the plain struct(s), enum(s),
// `Default`, and serde derives; the matching ECS behavior lives in core.
mod application;
mod audio_bus;
mod audio_clip;
mod audio_cue;
mod audio_emitter;
mod behavior;
mod block_type;
mod camera3d;
mod camera_shot;
mod color_lut;
mod cubemap_texture;
mod debug_hud;
mod decal;
mod directional_light;
mod engine_defaults;
mod environment_map;
mod file;
mod font;
mod fps_counter;
mod glass_panel;
mod graphics_config;
mod hit_region;
mod instanced_prop;
mod key_binding;
mod layout_container;
mod light_rig;
mod loading_overlay;
mod main_menu;
mod material;
mod material_palette;
mod mesh;
mod model;
mod option_select;
mod panel;
mod particle_emitter;
mod physics_config;
mod physics_joint;
mod point_light;
mod post_process_config;
mod prefab;
mod procedural_mesh;
mod prop;
mod prop_body;
mod rect_area_light;
mod reflection_probe;
mod rigid_body;
mod room;
mod scene;
mod scene_import;
mod screen;
mod scroll_panel;
mod sdf_volume;
mod shader;
mod skinned_mesh;
mod slider;
mod spawner;
mod spot_light;
mod sprite;
mod stat_hud;
mod story;
mod story_import;
mod streaming_config;
mod text_input;
mod text_label;
mod texture;
mod trigger_volume;
mod variables;
mod volumetric_fog;
mod voxel_chunk;
mod voxel_world;
mod water_surface;
mod window;

pub use application::{AppLimits, ApplicationArgs};
pub use audio_bus::AudioBus;
pub use audio_clip::AudioClip;
pub use audio_cue::{AudioCue, CueKind};
pub use audio_emitter::{AudioEmitter, Rolloff};
pub use behavior::{
    Behavior, BehaviorExpr, BehaviorLiteral, BehaviorLocal, BehaviorNode, BehaviorQuery,
    BehaviorSource,
};
pub use block_type::BlockType;
pub use camera_shot::CameraShot;
pub use camera3d::{Camera3DArgs, CameraController, FollowController, FollowDrive};
pub use color_lut::ColorLut;
pub use cubemap_texture::CubemapTexture;
pub use debug_hud::DebugHud;
pub use decal::Decal;
pub use directional_light::DirectionalLight;
pub use engine_defaults::EngineDefaults;
pub use environment_map::EnvironmentMap;
pub use file::{FileArgs, FileKind};
pub use font::Font;
pub use fps_counter::FpsCounter;
pub use glass_panel::GlassPanel;
pub use graphics_config::{GraphicsConfig, ShadowUpdate};
pub use hit_region::HitRegion;
pub use instanced_prop::{InstanceTransform, InstancedProp};
pub use key_binding::KeyBinding;
pub use layout_container::{Justify, LabelBox, LabelPlacement, LayoutContainer, LayoutRow};
pub use light_rig::LightRig;
pub use loading_overlay::LoadingOverlay;
pub use main_menu::{MainMenu, MainMenuItem, SettingsProfile};
pub use material::Material;
pub use material_palette::{MaterialPalette, PaletteEntry};
pub use mesh::{Mesh, VertexData};
pub use model::{Model, SubMeshRef};
pub use option_select::OptionSelect;
pub use panel::Panel;
pub use particle_emitter::ParticleEmitter;
pub use physics_config::PhysicsConfig;
pub use physics_joint::{PhysicsJoint, PhysicsJointKind};
pub use point_light::PointLight;
pub use post_process_config::{
    AaMode, DEFAULT_SSGI_RAYS, DEFAULT_SSGI_STEPS, IndirectLighting, PostProcessConfig,
    ReflectionBlurResolution, SsgiResolution, UpscaleQuality, UpscalerBackend,
};
pub use prefab::{Prefab, PrefabEntry, PrefabKind};
pub use procedural_mesh::ProceduralMesh;
pub use prop::{Prop, PropCollider};
pub use prop_body::PropBody;
pub use rect_area_light::RectAreaLight;
pub use reflection_probe::ReflectionProbe;
pub use rigid_body::RigidBody;
pub use room::RoomArgs;
pub use scene::Scene;
pub use scene_import::SceneImport;
pub use screen::{Screen, ScreenInput};
pub use scroll_panel::{ScrollGroup, ScrollPanel, ScrollRow};
pub use sdf_volume::{SDF_PARAMS_LEN, SdfVolume};
pub use shader::{Shader, ShaderKind, ShaderPayload, StageSource};
pub use skinned_mesh::{
    CharacterCapsule, MorphDelta, SkeletonJoint, SkinnedMesh, SkinnedVertexData,
};
pub use slider::Slider;
pub use spawner::SpawnerArgs;
pub use spot_light::SpotLight;
pub use sprite::{Sprite, SpriteFit};
pub use stat_hud::StatHud;
pub use story::{
    Story, StoryChoice, StoryCompareOp, StoryCondition, StoryGate, StoryImage, StoryNode, StoryOp,
    StoryPage, StoryPlayback, StoryReload, StoryScaffold, StorySpeaker, StoryStage,
};
pub use story_import::StoryImport;
pub use streaming_config::StreamingConfig;
pub use text_input::TextInput;
pub use text_label::{TextAlign, TextLabel};
pub use texture::Texture;
pub use trigger_volume::{TriggerFilter, TriggerVolume};
pub use variables::{VariableDecl, Variables};
pub use volumetric_fog::VolumetricFog;
pub use voxel_chunk::VoxelChunk;
pub use voxel_world::VoxelWorld;
pub use water_surface::{MAX_WATER_WAVES, WaterSurface, WaterWave};
pub use window::{Window, WindowMode};
