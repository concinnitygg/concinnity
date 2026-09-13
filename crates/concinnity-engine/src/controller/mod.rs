/// First-person / fly-through camera controller. Internal system, constructed by
/// `World::start` from a `Camera3D`'s controller settings. `pub` so the editor
/// crate can zero the controller's velocity behind an externally driven pose.
pub mod camera;
pub(crate) mod look_controls;
// Third-person character controller. Internal system, constructed instead of
// Camera3DSystem when the controlling camera's controller has a `follow` block.
pub(crate) mod third_person;
