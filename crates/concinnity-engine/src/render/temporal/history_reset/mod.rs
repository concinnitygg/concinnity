//! When the temporal history every accumulating pass reprojects from stops
//! describing the frame about to render: a camera cut, a scene switch, or a
//! change to the render settings. Whatever moves the camera discontinuously (a
//! [`CameraTrack`](crate::components::CameraTrack) cut, a behavior teleport, a
//! debug camera write) says so through
//! [`PendingHistoryReset::raise`](crate::protocol::PendingHistoryReset::raise).
//! The tracker folds those causes in with what it sees itself, a change of
//! camera or scene and a view that strays far from the camera's predicted
//! motion, so every temporal consumer (TAA, the upscalers, SSGI, the occlusion
//! pyramid) drops its history on the same frame. A resize is not among them: each
//! backend rebuilds its temporal targets, and with them their history, on the
//! frame the surface changes.

mod latch;
mod motion;
mod tracker;

pub use latch::{ConsumedReset, UpscalerResetLatch};
pub use tracker::{FrameClock, HistoryResetTracker, HistoryView};
