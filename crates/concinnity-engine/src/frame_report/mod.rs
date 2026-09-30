//! Whole-frame measurement: what a world's
//! [`FrameReport`](concinnity_core::components::FrameReport) gates in.
//!
//! One row is recorded per frame, holding its wall time beside the render
//! stats and system timings for it. When the run ends the rows reduce to a
//! distribution: median and tail percentiles, the CPU work and GPU time behind
//! them, and the passes and systems that owned each. The slowest frames are
//! also named one by one, since a hitch shows in a distribution only as its
//! maximum.
//!
//! The rows carry the segment the frame was drawn in, where the run names one,
//! so the same reduction runs over each stretch of the run as well as over the
//! whole of it.

mod reduce;
mod report;
mod sample;
mod stats;
mod system;

pub use reduce::{
    FrameCost, PassShare, ReduceOptions, Report, SegmentReport, SlowFrame, SystemCost,
};
pub use report::to_text;
pub use sample::{FrameRun, FrameSample, MAX_SYSTEM_TIMINGS};
pub use stats::Distribution;
pub use system::FrameReportSystem;
