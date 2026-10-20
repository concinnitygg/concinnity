//! A benchmarked run: the world's [`BenchmarkConfig`](crate::components::BenchmarkConfig),
//! or the default one, armed over the world before it starts.
//!
//! Arming takes the camera away from the player and hands it a track that
//! holds still through the warm-up and then plays the config's legs, and
//! publishes a [`BenchmarkRun`] that the measurement reads its phases from.

mod arm;
pub(crate) mod config;
mod run;

pub use arm::arm;
pub use run::BenchmarkRun;
