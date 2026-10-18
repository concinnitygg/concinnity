//! Whole-frame measurement: what a benchmarked run, armed from a world's
//! [`BenchmarkConfig`](crate::components::BenchmarkConfig), gates in.
//!
//! One row is recorded per frame, holding its wall time beside the render
//! stats and system timings for it. When the run ends the rows reduce to a
//! distribution: median and tail percentiles, the CPU work and GPU time behind
//! them, and the passes and systems that owned each. The slowest frames are
//! also named one by one, since a hitch shows in a distribution only as its
//! maximum.
//!
//! The rows carry the phase the frame was drawn in, so the same reduction runs
//! over each stretch of the run as well as over the whole of it.
//!
//! A report that saves frames as images asks the render side for each one as
//! the run reaches its moment. Frames run while the window is minimized or
//! hidden draw nothing and are left out of the measurement.

mod capture;
mod hidden;
mod reduce;
mod report;
mod sample;
mod sample_log;
mod shown;
mod stats;
mod system;

pub use system::FrameReportSystem;
