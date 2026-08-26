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

include!(concat!(env!("OUT_DIR"), "/source_hash.rs"));
include!(concat!(env!("OUT_DIR"), "/asset_docs.rs"));

pub mod doc_model;

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
    set_audio_clip_handle_resolver, set_font_handle_resolver, set_material_handle_resolver,
    set_mesh_handle_resolver, set_name_resolver, set_shader_handle_resolver,
    set_skinned_mesh_handle_resolver, set_texture_handle_resolver,
};

// Asset data schema: one module per asset type, mirroring the impl-side layout
// under concinnity-core/src/components. Each holds the plain struct(s), enum(s),
// `Default`, and serde derives; the matching ECS behavior lives in core.
mod app_config;
mod audio_bus;
mod audio_clip;
mod audio_cue;
mod audio_emitter;
mod behavior;
mod block_type;
mod camera3d;
mod camera_shot;
mod character_model;
mod character_schema;
mod character_shape;
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

pub mod components;
pub mod cook;

// The flat vocabulary surface, assembled from the two namespaces above so the
// partition is the only place membership is decided. Engine-internal code names
// these types directly; `concinnity::components` and `concinnity::cook` are the
// modules, globbed.
pub use components::*;
pub use cook::*;

// Bounds and capacities the engine reads off the schema. Not vocabulary: they
// declare nothing, so they stay out of both namespaces.
pub use post_process_config::{DEFAULT_SSGI_RAYS, DEFAULT_SSGI_STEPS};
pub use sdf_volume::SDF_PARAMS_LEN;
pub use water_surface::MAX_WATER_WAVES;
