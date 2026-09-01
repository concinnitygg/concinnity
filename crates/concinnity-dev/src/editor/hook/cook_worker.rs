// src/editor/hook/cook_worker.rs
//
// EditorHook: the blob-cook worker behind the console's build command. It
// compiles the serialized entries on a worker thread with an operation card fed
// by the pipeline's compile progress, on the engine's bounded job pool, under
// the one-cook-at-a-time guard.

use super::*;

impl EditorHook {
    // Run one blob cook of `content` off-thread. The caller must already hold
    // `console_build_running` (claimed via `swap`); it is released when the
    // worker finishes. `done` runs on the worker with the outcome and the
    // elapsed seconds.
    pub(super) fn spawn_cook_worker(
        &self,
        label: &str,
        content: String,
        done: impl FnOnce(std::io::Result<()>, f32) + Send + 'static,
    ) {
        let running = self.console_build_running.clone();
        let op = self.notifier.begin_op(label);
        std::thread::spawn(move || {
            let start = std::time::Instant::now();
            // Fractional progress from the parallel payload compile; the
            // other stages leave the bar indeterminate. The bounded pool
            // keeps the compile from fanning onto rayon's global pool
            // against the render thread.
            let report = |p: concinnity_cook::BuildProgress| {
                if p.stage == "compile" {
                    op.set(p.done, p.total);
                } else {
                    op.set(0, 0);
                }
            };
            let outcome = crate::jobs::pool().install(|| {
                crate::authoring::build_world_str_to_disk_with_progress(&content, Some(&report))
            });
            op.finish();
            done(outcome, start.elapsed().as_secs_f32());
            running.store(false, std::sync::atomic::Ordering::SeqCst);
        });
    }
}
