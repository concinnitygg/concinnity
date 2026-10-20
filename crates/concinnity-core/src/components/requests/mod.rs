//! One-shot request components the runtime consumes: spawn, despawn, reparent,
//! visibility, wake, scene, and setting requests.

pub(super) mod despawn_request;
pub(super) mod reparent_request;
pub(super) mod scene_command;
pub(super) mod setting_command;
pub(super) mod spawn_request;
pub(super) mod visibility_request;
pub(super) mod wake_request;
