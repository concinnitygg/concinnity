//! On-screen HUD overlays. Internal systems (not declarable assets): each is
//! built by the host's system table when the world declares its matching
//! request component (`FpsCounter` / `StatHud` / `DebugHud` / `LoadingOverlay`),
//! and drives the referenced UI elements each frame.

pub mod debug_hud;
pub mod fps_counter;
pub mod loading_overlay;
mod rate_window;
pub mod stat_hud;
