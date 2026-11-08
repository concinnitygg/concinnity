//! The access ids of the resource and event types this crate defines, which a
//! system's declared [`Access`] names them by.

use crate::ecs::access_id::ensure_event_queues;
use crate::ecs::{Access, EventStore};

/// Create every event queue `access` may touch among this crate's
/// [`ACCESS_IDS`]: the table's `prepare_events` for a host with no ids of its
/// own.
pub fn prepare_events(store: &mut EventStore, access: Access) {
    ensure_event_queues(&[ACCESS_IDS], store, access);
}

crate::define_access_ids! {
    /// The resource and event types this crate defines.
    pub static ACCESS_IDS;
    /// The first access id past [`ACCESS_IDS`], where another registry starts.
    pub const ACCESS_ID_COUNT;
    base: 0,
    resources: [
        crate::components::FrameInput,
        crate::protocol::MenuActive,
        crate::protocol::SimTiming,
        crate::protocol::FrameTime,
        crate::protocol::MenuOverride,
        crate::protocol::DesiredCursor,
        crate::protocol::HudLayers,
        crate::protocol::ScreenStack,
        crate::protocol::FlyCam,
        crate::protocol::CursorState,
        crate::protocol::WindowMinimized,
        crate::protocol::HudPrefs,
        crate::protocol::HudHidden,
        crate::protocol::OpenDropdown,
        crate::protocol::ScheduleMode,
        crate::ecs::EntityById,
        crate::environment::planet::FrameRebases,
        crate::environment::planet::PlanetFrame,
        crate::simulation::camera::track::CameraTrackStatus,
        crate::protocol::DisabledSettingRows,
        crate::protocol::DisplayModes,
        crate::protocol::InputSampledAt,
        crate::protocol::ScreenshotRequest,
        crate::protocol::FrameCaptureRequest,
        crate::render::scene::flow::ActiveSceneFlow,
        crate::render::streaming::scene_residency::SceneResidencyStatus,
        crate::budget::MemoryBudget,
        crate::budget::ThreadBudget,
        crate::budget::ResidentMemory,
        concinnity_core::platform::CompiledPlatforms,
        concinnity_core::platform::GraphicsApiStatus,
        crate::benchmark::BenchmarkRun,
        crate::render::streaming::pools::StreamingRuntime,
        crate::settings::StoredSettings,
        crate::settings::SettingsRevision,
        crate::render::overlay::OverlayFrame,
        crate::render::overlay::OverlayAssets,
        crate::render::overlay::OverlayRecycle,
        crate::input::mailbox::InputMailbox,
    ],
    events: [
        crate::components::ControlsCommand,
        crate::components::InteractEvent,
        crate::components::RootMotionEvent,
        crate::components::ScreenCommand,
        crate::components::ScreenShown,
        crate::components::SettingCommand,
        crate::components::SceneCommand,
        crate::components::StoryCommand,
        crate::components::StoryReload,
        crate::components::PlayCue,
        crate::components::AudioCommand,
    ],
}
