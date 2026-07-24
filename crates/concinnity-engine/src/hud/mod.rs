// src/hud/mod.rs
//
// On-screen HUD overlays. Internal systems (not declarable assets): each is
// constructed by `World::start` when the world declares its matching request
// component (`FpsCounter` / `StatHud` / `DebugHud` / `LoadingOverlay`), and
// drives the referenced UI elements each frame.

pub(crate) mod debug_hud;
pub(crate) mod fps_counter;
pub(crate) mod loading_overlay;
pub(crate) mod stat_hud;
