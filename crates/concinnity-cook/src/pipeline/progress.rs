//! Progress reporting for a build: the stages a build runs, the events each
//! one reports, and the counter the pipeline reports them through.

use std::sync::atomic::{AtomicU32, Ordering};

/// A stage of the build pipeline, in the order a build runs them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuildStage {
    /// Source files (glTF, FBX) imported into the inline data their assets
    /// compile from.
    Import,
    /// Asset payloads compiled, or reused from the build cache.
    Compile,
    /// The blob files and `world-lock.json` written.
    Write,
    /// Preview thumbnails rendered for the asset browser.
    Thumbnails,
}

/// A progress report from the build pipeline. A stage runs from its
/// [`Started`](Self::Started) report until the next stage starts or the build
/// returns. A stage with nothing to do may not report at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildProgress<'a> {
    /// A stage began.
    Started {
        /// The stage.
        stage: BuildStage,
        /// Its items of work; 0 when it cannot count them.
        total: u32,
    },
    /// Work on one item began. Items of a parallel stage overlap.
    ItemStarted {
        /// The running stage.
        stage: BuildStage,
        /// The asset id or source file being worked on.
        item: &'a str,
    },
    /// Work on one item ended.
    ItemFinished {
        /// The running stage.
        stage: BuildStage,
        /// The asset id or source file that finished.
        item: &'a str,
        /// The stage's items finished so far, this one included.
        done: u32,
        /// The stage's items of work.
        total: u32,
        /// Whether the build cache already held this item's output.
        reused: bool,
    },
}

/// The callback a build reports its progress through. It is called from the
/// parallel compile's worker threads, so it must be cheap and non-blocking.
pub type ProgressFn<'a> = &'a (dyn Fn(BuildProgress<'_>) + Sync);

// Where a build reports: the caller's callback, or nowhere.
#[derive(Clone, Copy)]
pub(crate) struct Progress<'a>(Option<ProgressFn<'a>>);

impl<'a> Progress<'a> {
    pub(crate) fn new(sink: Option<ProgressFn<'a>>) -> Self {
        Self(sink)
    }

    pub(crate) fn none() -> Self {
        Self(None)
    }

    // Report `stage` started with `total` items and count its items from here.
    pub(crate) fn stage(self, stage: BuildStage, total: u32) -> StageProgress<'a> {
        if let Some(sink) = self.0 {
            sink(BuildProgress::Started { stage, total });
        }
        StageProgress {
            sink: self.0,
            stage,
            total,
            done: AtomicU32::new(0),
        }
    }
}

// One running stage's item counter. Shared across the parallel compile's
// workers by reference.
pub(crate) struct StageProgress<'a> {
    sink: Option<ProgressFn<'a>>,
    stage: BuildStage,
    total: u32,
    done: AtomicU32,
}

impl StageProgress<'_> {
    // A stage that reports nowhere, for callers that drive a stage's work
    // directly.
    pub(crate) fn silent(stage: BuildStage) -> Self {
        Progress::none().stage(stage, 0)
    }

    pub(crate) fn begin(&self, item: &str) {
        if let Some(sink) = self.sink {
            sink(BuildProgress::ItemStarted {
                stage: self.stage,
                item,
            });
        }
    }

    pub(crate) fn end(&self, item: &str, reused: bool) {
        let done = self.done.fetch_add(1, Ordering::Relaxed) + 1;
        if let Some(sink) = self.sink {
            sink(BuildProgress::ItemFinished {
                stage: self.stage,
                item,
                done,
                total: self.total,
                reused,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn recorded(run: impl FnOnce(Progress<'_>)) -> Vec<String> {
        let log = Mutex::new(Vec::new());
        let sink = |p: BuildProgress<'_>| log.lock().unwrap().push(format!("{p:?}"));
        run(Progress::new(Some(&sink)));
        log.into_inner().unwrap()
    }

    #[test]
    fn a_stage_reports_its_start_then_counts_each_finished_item() {
        let log = recorded(|progress| {
            let stage = progress.stage(BuildStage::Compile, 2);
            stage.begin("a");
            stage.end("a", true);
            stage.begin("b");
            stage.end("b", false);
        });
        assert_eq!(
            log,
            [
                "Started { stage: Compile, total: 2 }",
                "ItemStarted { stage: Compile, item: \"a\" }",
                "ItemFinished { stage: Compile, item: \"a\", done: 1, total: 2, reused: true }",
                "ItemStarted { stage: Compile, item: \"b\" }",
                "ItemFinished { stage: Compile, item: \"b\", done: 2, total: 2, reused: false }",
            ]
        );
    }
}
