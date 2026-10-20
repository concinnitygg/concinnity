//! Whole-frame measurement: what a benchmarked run, armed from a world's
//! [`BenchmarkConfig`](crate::components::BenchmarkConfig), gates in.
//!
//! Each frame's wall time, render stats and system timings are tallied as it
//! is drawn, into histograms and running means sized when the run starts, so a
//! run of any length measures in the memory it began with. When the run ends
//! the tallies reduce to a distribution: median and tail percentiles, the CPU
//! work and GPU time behind them, and the passes and systems that owned each.
//! The slowest frames are also kept whole and named one by one, since a hitch
//! shows in a distribution only as its maximum.
//!
//! Each frame is tallied under the phase it was drawn in as well as under the
//! whole run, so the same reduction covers each stretch of the run.
//!
//! A report that saves frames as images asks the render side for each one as
//! the run reaches its moment. Frames run while the window is minimized or
//! hidden draw nothing and are left out of the measurement.

mod capture;
mod hidden;
mod histogram;
mod reduce;
mod report;
mod sample;
mod shown;
mod stats;
mod system;
mod tally;

pub use system::FrameReportSystem;
