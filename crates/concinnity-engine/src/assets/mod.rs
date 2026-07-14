// src/assets/mod.rs
//
// Client-side asset surface: a thin re-export of concinnity-core's asset types
// under the historical `crate::assets::*` paths. All assets are pure data now:
// every system became an internal client system living in its own domain module
// (`gfx::graphics_system`, `gfx::camera_controller`, `gfx::animation`,
// `audio::system`, `ui`, `hud`) or subsystem crate (`concinnity_physics`),
// constructed by `World::start` from the components a world declares.
pub use concinnity_core::assets::*;

// Submodule paths referenced explicitly elsewhere in the client, e.g.
// `crate::assets::shader_stage::ShaderKind`,
// `crate::assets::sdf_volume::sdf_volume_blob_indices`. Re-export the modules
// themselves so those paths keep resolving against core. (Audio-clip residency
// moved to `crate::resource::AudioClipTable::blob_indices` when AudioClip became
// a resource.)
pub use concinnity_core::assets::{procedural_mesh, sdf_volume, shader_stage};
