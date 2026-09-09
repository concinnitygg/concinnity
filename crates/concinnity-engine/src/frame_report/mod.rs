//! Whole-frame measurement: what a world's
//! [`FrameReport`](crate::components::FrameReport) gates in.
//!
//! One row is recorded per frame, holding its wall time beside the render
//! stats and system timings for it. When the run ends the rows reduce to a
//! distribution: median and tail percentiles, the CPU work and GPU time behind
//! them, and the passes and systems that owned each.
//!
//! The rows carry the segment the frame was drawn in, where the run names one,
//! so the same reduction runs over each stretch of the run as well as over the
//! whole of it.

mod reduce;
mod report;
mod sample;
mod stats;
mod system;

pub use reduce::{PassShare, ReduceOptions, Report, SegmentReport, SystemCost};
pub use report::to_text;
pub use sample::{FrameRun, FrameSample, MAX_SYSTEM_TIMINGS};
pub use stats::Distribution;
pub use system::FrameReportSystem;
