// The host half of frame submission: core's `submit::submit` timed by
// the monotonic clock, with a device loss reported to the crash handler.

use concinnity_core::render::backend::RenderBackend;
use concinnity_core::render::frame::snapshot::RenderSnapshot;
use concinnity_core::render::frame::submit::{self, FramePolicy, SubmitOutcome};

pub(crate) fn submit(
    policy: &mut FramePolicy,
    snap: &mut RenderSnapshot,
    backend: &mut dyn RenderBackend,
) -> SubmitOutcome {
    let outcome = submit::submit(
        policy,
        snap,
        backend,
        concinnity_host::clock::monotonic_micros,
    );
    if let Some(detail) = &outcome.device_lost {
        crate::crash::report_device_lost(detail);
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::profile::RenderStats;
    use concinnity_core::render::backend::recording::recording_backend;
    use concinnity_host::clock::micros_since;
    use std::time::Instant;

    // Submit one frame to a backend whose draw takes `draw` and reports
    // `gpu_wait_us`, returning the frame's render stats and the wall time the
    // whole call took.
    fn submit_timed(draw: std::time::Duration, gpu_wait_us: u32) -> (RenderStats, u32) {
        let (state, mut backend) = recording_backend();
        {
            let mut s = state.lock();
            s.on_draw = Some(Box::new(move || std::thread::sleep(draw)));
            s.render_stats.gpu_wait_us = gpu_wait_us;
        }
        let started = Instant::now();
        let outcome = submit(
            &mut FramePolicy::default(),
            &mut RenderSnapshot::default(),
            &mut backend,
        );
        let wall = micros_since(started);
        (outcome.render_stats.expect("a drawn frame"), wall)
    }

    // The host clock times the whole submission.
    #[test]
    fn the_render_cpu_time_covers_the_whole_submission() {
        let (stats, wall) = submit_timed(std::time::Duration::from_millis(2), 0);
        assert!(stats.render_cpu_us >= 2_000, "{} us", stats.render_cpu_us);
        assert!(
            stats.render_cpu_us <= wall,
            "{} > {wall}",
            stats.render_cpu_us
        );
    }
}
