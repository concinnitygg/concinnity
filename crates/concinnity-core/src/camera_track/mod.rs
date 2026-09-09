//! The scripted camera path: the keys a world's
//! [`CameraTrack`](crate::components::CameraTrack) baked into, the system that
//! plays them, and where it has reached.
//!
//! The track's two lists are played against one clock and never read each
//! other, so travelling and turning compose without either bending the other.
//! That clock is the fixed simulation step, which is what makes a run
//! repeatable: a slower machine samples the same path at the same track times,
//! just at fewer of them.

mod status;
mod system;
mod timeline;

pub use status::CameraTrackStatus;
pub use system::CameraTrackSystem;
