//! The input-driven camera controllers. Each is an internal system, built by
//! the host's system table from a `Camera3D`'s controller settings.

/// First-person / fly-through camera controller. `pub` so the editor crate can
/// zero the controller's velocity behind an externally driven pose.
pub mod camera;
pub(crate) mod held;
pub(crate) mod interact_event;
pub(crate) mod interactable;
mod look_controls;
pub(crate) mod pickup;
/// Third-person character controller, built instead of the first-person one
/// when the controlling camera's controller has a `follow` block.
pub mod third_person;

use crate::components::{Camera3D, CameraController, CameraTrack};
use crate::ecs::World;

// The controller of the first controlled `Camera3D`: first-person without a
// `follow` block, third-person with one. A `controller: null` camera or one a
// CameraTrack owns gets none.
fn controlled_camera(world: &World) -> Option<CameraController> {
    if world.query::<CameraTrack>().next().is_some() {
        return None;
    }
    world.query::<Camera3D>().find_map(|c| c.controller.clone())
}
