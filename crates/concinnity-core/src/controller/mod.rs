//! The input-driven camera controllers. Each is an internal system, built by
//! the host's system table from a `Camera3D`'s controller settings.

/// First-person / fly-through camera controller. `pub` so the editor crate can
/// zero the controller's velocity behind an externally driven pose.
pub mod camera;
mod look_controls;
/// Third-person character controller, built instead of the first-person one
/// when the controlling camera's controller has a `follow` block.
pub mod third_person;
