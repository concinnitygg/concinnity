//! The half of the authoring vocabulary core owns: the resources the cook
//! compiles into the blob's resource stream, and the args schemas of the six
//! types whose authored form differs from the component it bakes into.
//!
//! Membership is the component registry's `resource` group plus those six.
//! Each of the six is named here for the asset it declares, so the authoring
//! form of `X` is `cook::X` and the runtime form [`components::X`](super); no
//! `Args` name reaches the surface.
//!
//! The build-only assets, which the cook expands away before a blob is written,
//! are the other half of this namespace and live in
//! `concinnity_cook::authoring::registry::build_only`.

pub use crate::components::animation::skinned_mesh::{
    CharacterCapsule, MorphDelta, SkeletonJoint, SkinnedMesh, SkinnedVertexData,
};
pub use crate::components::app::app_config::AppConfigArgs as AppConfig;
pub use crate::components::assets::file::FileArgs as File;
pub use crate::components::assets::font::Font;
pub use crate::components::assets::material::Material;
pub use crate::components::assets::mesh::{Mesh, VertexData};
pub use crate::components::assets::room::RoomArgs as Room;
pub use crate::components::assets::texture::Texture;
pub use crate::components::audio::clip::AudioClip;
pub use crate::components::camera::camera3d::Camera3DArgs as Camera3D;
pub use crate::components::camera::track::CameraTrackArgs as CameraTrack;
pub use crate::components::gameplay::spawner::SpawnerArgs as Spawner;
pub use crate::components::graphics::color_lut::ColorLut;
pub use crate::components::light::cubemap_texture::CubemapTexture;
pub use crate::components::light::environment_map::EnvironmentMap;
