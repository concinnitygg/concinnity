//! What the camera sees, measured once a frame: [`ViewFacts`].
//!
//! Part of the facts come from the camera and the world as they stand: the
//! camera's height over the ground and the sea, how far it moved and how far
//! the things in view moved against it, the light on the ground, and how
//! large each subject (the planet, each water surface, the prop a reflection
//! probe rides) is, how far away, and whether it is in view. The rest is
//! measured from the frame's depth: a coarse grid of it is sampled on the GPU
//! and read back a few frames later without waiting, and each sample is placed
//! on what it shows (sky, a water surface, the planet, the prop, or anything
//! else). That gives each subject's share of the screen and the distance to
//! its nearest visible part, and the share of sky.
//!
//! A measurement stands for the frames after it until a newer one arrives; a
//! view that does not continue the last drops it.

mod facts;
mod measure;
mod movers;
mod subjects;

pub use facts::{Seen, Sighting, Subject, SubjectKind, VIEW_SUBJECTS, ViewFacts};
pub(crate) use measure::{MAX_MEASURED_AGE, MeasureView, ViewMeasure};
pub(crate) use movers::{MoverView, ViewMovers};
pub(crate) use subjects::ViewSubjects;

#[cfg(test)]
mod tests;
