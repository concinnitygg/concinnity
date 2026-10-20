//! Asset components a scene references: meshes, models, materials, textures,
//! fonts, shaders and their compiled programs, SDF volumes, rooms, files,
//! scenes, decals, particles, sprites, and glass panels.

pub mod compiled_programs;
pub(super) mod decal;
pub(super) mod file;
pub(super) mod font;
pub(super) mod glass_panel;
pub(super) mod material;
pub(super) mod mesh;
pub(super) mod model;
pub(super) mod particle_emitter;
pub mod procedural_mesh;
pub(super) mod room;
pub(super) mod scene;
pub mod sdf_programs;
pub mod sdf_volume;
pub mod shader;
pub(super) mod sprite;
pub(super) mod texture;
