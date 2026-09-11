//! Client-side component surface: a thin re-export of concinnity-core's
//! component types under the historical `crate::components::*` paths. All
//! components are pure data now: every system became an internal client system
//! living in its own domain module (`gfx::graphics_system`,
//! `gfx::camera_controller`, `gfx::animation`, `ui`, `hud`, `physics`) or
//! audio module (`crate::audio`), constructed by `World::start` from the
//! components a world declares.
pub use concinnity_core::components::*;
