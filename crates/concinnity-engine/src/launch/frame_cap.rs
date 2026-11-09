// The frame-rate cap in force for a run.

use super::LaunchRequest;
use crate::benchmark::BenchmarkRun;
use crate::ecs::World;
use crate::protocol::FrameRateCap;

/// The cap in force in `world` (0 is uncapped): the launch's own cap, else a
/// benchmarked run's, else the one the world publishes.
pub fn frame_cap(world: &World) -> u32 {
    let launch = world
        .resource::<LaunchRequest>()
        .and_then(|launch| launch.args.limit_fps);
    let benchmark = world
        .resource::<BenchmarkRun>()
        .map(|run| run.config.fps_cap);
    let published = world.resource::<FrameRateCap>().map(|c| c.0);
    launch.or(benchmark).or(published).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The launch's cap holds over a benchmark's, which holds over the
    // published one; a world with none is uncapped.
    #[test]
    fn the_launch_outranks_a_benchmark_which_outranks_the_world() {
        use crate::components::BenchmarkConfig;
        use crate::launch::LaunchArgs;

        let mut world = World::new();
        assert_eq!(frame_cap(&world), 0);
        world.insert_resource(FrameRateCap(60));
        assert_eq!(frame_cap(&world), 60);
        world.insert_resource(BenchmarkRun::new(BenchmarkConfig::default()));
        assert_eq!(frame_cap(&world), 0);
        world.insert_resource(LaunchRequest {
            args: LaunchArgs {
                limit_fps: Some(1000),
                ..LaunchArgs::NONE
            },
            ..Default::default()
        });
        assert_eq!(frame_cap(&world), 1000);
    }
}
