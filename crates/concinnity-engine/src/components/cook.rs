//! The half of the authoring vocabulary core owns: the resources the cook
//! compiles into the blob's resource stream, and the args schemas of the seven
//! types whose authored form differs from the component it bakes into.
//!
//! Membership is the component registry's `resource` group plus those seven.
//! Each of the seven is named here for the asset it declares, so the authoring
//! form of `X` is `cook::X` and the runtime form [`components::X`](super); no
//! `Args` name reaches the surface.
//!
//! The build-only assets, which the cook expands away before a blob is written,
//! are the other half of this namespace and live in
//! `concinnity_cook::authoring::registry::build_only`.

pub use crate::app::config::AppConfigArgs as AppConfig;
pub use crate::camera::camera3d::Camera3DArgs as Camera3D;
pub use crate::camera::track::CameraTrackArgs as CameraTrack;
pub use crate::geometry::room::RoomArgs as Room;
pub use crate::planet::PlanetGroundArgs as PlanetGround;
pub use crate::resource::audio_clip::AudioClip;
pub use crate::resource::color_lut::ColorLut;
pub use crate::resource::cubemap_texture::CubemapTexture;
pub use crate::resource::environment_map::EnvironmentMap;
pub use crate::resource::file::FileArgs as File;
pub use crate::resource::font::Font;
pub use crate::resource::material::Material;
pub use crate::resource::mesh::{Mesh, VertexData};
pub use crate::resource::skinned_mesh::{
    CharacterCapsule, MorphDelta, SkeletonJoint, SkinnedMesh, SkinnedVertexData,
};
pub use crate::resource::texture::Texture;
pub use crate::spawn::spawner::SpawnerArgs as Spawner;
