//! Frames saved as images at authored moments of a run: the [`Screenshot`]
//! asset, the launch's `--screenshot-at` shots, and the system that requests
//! each one as the simulation clock reaches it.

mod asset;
mod schedule;
mod system;

pub use asset::{Screenshot, is_contained_file, normal_file};
pub(crate) use system::{ScreenshotSystem, takes_screenshots};
