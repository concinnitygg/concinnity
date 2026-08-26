//! The authoring-only half of the asset vocabulary: what a world declares and
//! the cook consumes.
//!
//! Membership is the component registry's `build_only` and `resource` groups
//! (types the cook expands into components, and payloads it compiles into the
//! blob's resource stream) plus the args schemas of the five types whose
//! authored form differs from the component it bakes into. Those five are named
//! here for the asset they declare, so the authoring form of `X` is `cook::X`
//! and the runtime form is [`components::X`](crate::components); no `Args` name
//! reaches the surface.

pub use crate::app_config::AppConfigArgs as AppConfig;
pub use crate::audio_clip::AudioClip;
pub use crate::camera_shot::CameraShot;
pub use crate::camera3d::Camera3DArgs as Camera3D;
pub use crate::character_model::CharacterModel;
pub use crate::character_schema::{
    CharacterSchema, KeyPolarity, PanelSection, ProportionGroup, SchemaJoint, SchemaKey,
    SchemaRegion, ShapePreset, SynthParams, SynthesizedTarget,
};
pub use crate::color_lut::ColorLut;
pub use crate::cubemap_texture::CubemapTexture;
pub use crate::engine_defaults::EngineDefaults;
pub use crate::environment_map::EnvironmentMap;
pub use crate::file::FileArgs as File;
pub use crate::font::Font;
pub use crate::light_rig::LightRig;
pub use crate::main_menu::{MainMenu, MainMenuItem, SettingsProfile};
pub use crate::material::Material;
pub use crate::material_palette::{MaterialPalette, PaletteEntry};
pub use crate::mesh::{Mesh, VertexData};
pub use crate::option_select::OptionSelect;
pub use crate::panel::Panel;
pub use crate::prefab::{Prefab, PrefabEntry, PrefabKind};
pub use crate::room::RoomArgs as Room;
pub use crate::scene_import::SceneImport;
pub use crate::skinned_mesh::{
    CharacterCapsule, MorphDelta, SkeletonJoint, SkinnedMesh, SkinnedVertexData,
};
pub use crate::slider::Slider;
pub use crate::spawner::SpawnerArgs as Spawner;
pub use crate::story_import::StoryImport;
pub use crate::texture::Texture;
