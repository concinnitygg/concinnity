//! The half of the authoring vocabulary core owns: the resources the cook
//! compiles into the blob's resource stream, and the args schemas of the five
//! types whose authored form differs from the component it bakes into.
//!
//! Membership is the component registry's `resource` group plus those five.
//! Each of the five is named here for the asset it declares, so the authoring
//! form of `X` is `cook::X` and the runtime form [`components::X`](super); no
//! `Args` name reaches the surface.
//!
//! The build-only assets, which the cook expands away before a blob is written,
//! are the other half of this namespace and live in
//! `concinnity_world::registry::build_only`.

pub use super::app_config::AppConfigArgs as AppConfig;
pub use super::audio_clip::AudioClip;
pub use super::camera3d::Camera3DArgs as Camera3D;
pub use super::color_lut::ColorLut;
pub use super::cubemap_texture::CubemapTexture;
pub use super::environment_map::EnvironmentMap;
pub use super::file::FileArgs as File;
pub use super::font::Font;
pub use super::material::Material;
pub use super::mesh::{Mesh, VertexData};
pub use super::room::RoomArgs as Room;
pub use super::skinned_mesh::{
    CharacterCapsule, MorphDelta, SkeletonJoint, SkinnedMesh, SkinnedVertexData,
};
pub use super::spawner::SpawnerArgs as Spawner;
pub use super::texture::Texture;
