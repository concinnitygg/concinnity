//! Client-side asset surface: a thin re-export of concinnity-core's asset types
//! under the historical `crate::assets::*` paths. All assets are pure data now:
//! every system became an internal client system living in its own domain module
//! (`gfx::graphics_system`, `gfx::camera_controller`, `gfx::animation`, `ui`,
//! `hud`, `physics`) or subsystem crate (`concinnity_audio`),
//! constructed by `World::start` from the components a world declares.
pub use concinnity_core::assets::*;
