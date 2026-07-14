// src/debug/state.rs
//
// The shared world-snapshot data model the debug server exposes. `wire::server`
// rebuilds it once per frame from the live `World`; `dispatch::handle_request`
// reads it to answer client queries. Kept as plain data (no sockets, no engine
// driving) so the dispatcher stays unit-testable against a hand-built snapshot.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use concinnity_core::shutdown::ShutdownToken;

use crate::gfx::streaming_system::StreamingStats;

// The world snapshot rebuilt by `tick`. The asset/system lists are not cheap
// to rebuild, so they refresh on an interval while `frame` advances every tick.
#[derive(Default)]
pub(crate) struct DebugState {
    pub(super) frame: u64,
    pub(super) system_count: usize,
    pub(super) component_count: usize,
    pub(super) systems: Vec<String>,
    pub(super) assets: Vec<AssetEntry>,
    // `AssetId` -> declared name, indexed by id. Captured once after build.
    pub(super) names: Vec<String>,
    // Asset-streaming `(resident, pending, unloaded)` counts, refreshed every
    // tick (cheap) so the `streaming` command reflects live progress.
    pub(super) streaming: StreamingStats,
    // Per-system CPU step times (micros) from the last completed frame,
    // refreshed every tick for the `profile` command.
    pub(super) profile_systems: Vec<(String, u32)>,
    // Render-backend stats from the most recent frame, for `profile`.
    pub(super) profile_render: crate::gfx::profile::RenderStats,
    // App shutdown token, set once via `DebugHook::attach_shutdown`. The
    // `shutdown` command cancels it to exit the engine cleanly.
    pub(super) shutdown_token: Option<ShutdownToken>,
    // Shared shader-reload flag captured from the active graphics backend.
    // `Some` once `tick` has seen a `GraphicsSystem` whose backend opted into
    // hot-reload (Metal under `cn debug`); `None` otherwise. The
    // `reload-shaders` command flips this to `true` and the backend polls it
    // at the top of `draw_frame`.
    pub(super) shader_reload: Option<Arc<AtomicBool>>,
    // Shared asset-reload flag captured from `GraphicsSystem`. `Some` once
    // `tick` has seen a `cn debug` world with at least one file-backed
    // `Texture`; `None` otherwise. The `reload-assets` command flips it; the
    // engine consumes it at the top of `GraphicsSystem::step`.
    pub(super) asset_reload: Option<Arc<AtomicBool>>,
    // Active-camera pose, refreshed every tick for the `camera-get` query.
    // `None` until the first tick that finds a `Camera3D` (a world with no
    // camera never sets it).
    pub(super) camera: Option<CameraSnapshot>,
}

// A read-only snapshot of the active `Camera3D`, served by `camera-get`. The
// matching `camera-set` mutation lives in `super::commands` /
// `super::runtime_spawn`.
#[derive(Clone, serde::Serialize)]
pub(crate) struct CameraSnapshot {
    pub(super) position: [f32; 3],
    pub(super) yaw: f32,
    pub(super) pitch: f32,
    pub(super) fov_y_degrees: f32,
    pub(super) near: f32,
    pub(super) far: f32,
}

// A structural census entry. Per-asset names are not retained at runtime
// (`BlobAssetDef::to_def` drops them), so this carries only kind + type
// discriminant; the `names` table handles the id -> name remap separately.
#[derive(Clone, serde::Serialize)]
pub(crate) struct AssetEntry {
    pub(super) kind: String,
    pub(super) discriminant: u8,
}
