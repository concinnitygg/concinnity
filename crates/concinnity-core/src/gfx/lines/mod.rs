//! The world-space line segment any system can submit for a frame: a trajectory
//! arc, a tether or beam, a patrol path, the editor's origin axes. Lines are
//! scene geometry, not overlay, so the depth-tested pass occludes them behind
//! whatever is in front of them.
//!
//! [`build_vertices`] expands each segment into the camera-facing ribbon the
//! line pass rasterises.

mod expand;

pub use expand::{LineCamera, build_vertices, build_vertices_into};

/// One world-space line to draw this frame. Colour is per endpoint, so a line
/// that fades out with distance is a single request with a transparent far end.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Line {
    /// World-space start point.
    pub start: [f32; 3],
    /// World-space end point.
    pub end: [f32; 3],
    /// Linear-space RGBA at `start` / `end`, interpolated along the run.
    pub start_color: [f32; 4],
    /// Linear-space RGBA at `end`.
    pub end_color: [f32; 4],
    /// On-screen thickness in pixels, held constant at any distance.
    pub width_px: f32,
}
