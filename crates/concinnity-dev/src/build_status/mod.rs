//! The status a world build shows while it runs: a line per step with its
//! progress, redrawn in place on a terminal and streamed line by line to a
//! pipe, with the warnings the build logs listed above the steps.

mod board;
mod capture;
mod render;
mod terminal;

use concinnity_core::math::SaturatingFrom;
use std::path::Path;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use concinnity_cook::build_only::LoadedWorld;
use concinnity_cook::{BuildProgress, BuildReport, BuildStage};
use concinnity_core::platform::PlatformSet;
use concinnity_host::thread::{ThreadRole, set_current_thread_role};

use board::{Board, NoteLevel, Outcome, RowState, Step};
use terminal::Surface;

// How often a live display redraws.
const FRAME: Duration = Duration::from_millis(80);

/// How much of what a build logs its status shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Verbosity {
    /// Warnings and errors.
    #[default]
    Normal,
    /// Every informational message as well, such as each imported mesh.
    Verbose,
}

// The board and the surface it draws to, shared with the redraw thread and
// the tracing capture.
struct Shared {
    board: Mutex<Board>,
    surface: Mutex<Surface>,
}

impl Shared {
    fn board(&self) -> MutexGuard<'_, Board> {
        self.board
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn note(&self, level: NoteLevel, text: String) {
        self.board().note(level, text);
    }

    // Write whatever changed since the last frame.
    fn draw(&self, tick: usize, last: bool) {
        let mut surface = self
            .surface
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let frame = {
            let mut board = self.board();
            surface.frame(&mut board, Instant::now(), tick, terminal::width(), last)
        };
        if !frame.is_empty() {
            eprint!("{frame}");
        }
    }
}

/// A build's status display, live from [`start`](Self::start) until the build
/// ends. Dropping it before [`finish`](Self::finish) ends it as failed.
pub(crate) struct BuildStatus {
    shared: Arc<Shared>,
    redraw: Option<(mpsc::Sender<()>, JoinHandle<()>)>,
    ended: bool,
}

impl BuildStatus {
    /// Show the build of `world` for `targets`, starting with its load.
    pub(crate) fn start(world: &str, targets: PlatformSet, verbosity: Verbosity) -> Self {
        let now = Instant::now();
        let mut board = Board::new(now);
        board.begin(Step::Load, 0, now);
        let surface = Surface::for_stderr();
        let targets: Vec<String> = targets.iter().map(|p| format!("{p:?}")).collect();
        let header = render::header(&shown_path(Path::new(world)), &targets.join(" + "));
        eprintln!("{}", header.paint(surface.colored()));

        let live = matches!(surface, Surface::Live { .. });
        let shared = Arc::new(Shared {
            board: Mutex::new(board),
            surface: Mutex::new(surface),
        });
        capture::install();
        let shown = match verbosity {
            Verbosity::Normal => tracing::Level::WARN,
            Verbosity::Verbose => tracing::Level::INFO,
        };
        capture::attach(Arc::clone(&shared), shown);

        // A stream has nothing to animate, so it redraws less often: only
        // closed steps and new notes reach it.
        let period = if live { FRAME } else { FRAME * 4 };
        let (stop, stopped) = mpsc::channel::<()>();
        let thread_shared = Arc::clone(&shared);
        let redraw = std::thread::Builder::new()
            .name("build-status".into())
            .spawn(move || {
                set_current_thread_role(ThreadRole::Background);
                let mut tick = 0;
                while stopped.recv_timeout(period) == Err(RecvTimeoutError::Timeout) {
                    tick += 1;
                    thread_shared.draw(tick, false);
                }
            })
            .ok()
            .map(|handle| (stop, handle));
        Self {
            shared,
            redraw,
            ended: false,
        }
    }

    /// Close the load step with what it produced.
    pub(crate) fn loaded(&self, loaded: &LoadedWorld) {
        let summary = load_summary(
            loaded.authored.len(),
            loaded.assets.len(),
            loaded.injected.len(),
        );
        self.shared.board().close(Some(summary), Instant::now());
    }

    /// Record one progress report from the cook.
    pub(crate) fn progress(&self, progress: BuildProgress<'_>) {
        self.shared.board().apply(progress, Instant::now());
    }

    /// End the display with the build's report.
    pub(crate) fn finish(mut self, report: &BuildReport) {
        let data_dir = report
            .blobs
            .first()
            .and_then(|(path, _)| path.parent())
            .map(shown_path)
            .unwrap_or_default();
        let outcome = Outcome::Built {
            written: report.blobs.iter().map(|(_, size)| size).sum(),
            data_dir,
        };
        let thumbnails = thumbnail_summary(&self.shared.board(), report);
        self.end(outcome, thumbnails);
    }

    /// End the display as failed, leaving the error to be printed below it.
    pub(crate) fn fail(mut self) {
        self.end(Outcome::Failed, None);
    }

    /// End the display as failed by the world's validation `errors`, listing
    /// each above the steps.
    pub(crate) fn fail_validation(mut self, errors: &[String]) {
        for error in errors {
            self.shared.note(NoteLevel::Error, error.clone());
        }
        let detail = render::plural(
            u32::saturating_from(errors.len()),
            "validation error",
            "validation errors",
        );
        self.end(Outcome::Failed, Some(detail));
    }

    #[expect(
        clippy::let_underscore_must_use,
        reason = "the redraw thread may already have stopped on its own"
    )]
    fn end(&mut self, outcome: Outcome, summary: Option<String>) {
        if std::mem::replace(&mut self.ended, true) {
            return;
        }
        if let Some((stop, handle)) = self.redraw.take() {
            let _ = stop.send(());
            if handle.join().is_err() {
                tracing::warn!("build status: the redraw thread panicked");
            }
        }
        capture::release();
        self.shared.board().end(outcome, summary, Instant::now());
        self.shared.draw(0, true);
    }
}

impl Drop for BuildStatus {
    fn drop(&mut self) {
        self.end(Outcome::Failed, None);
    }
}

// The load step's summary: how far the authored entries expanded, and the
// defaults the build added.
fn load_summary(authored: usize, assets: usize, injected: usize) -> String {
    let counted = render::counted(u32::saturating_from(assets), "asset");
    let mut summary = if authored == assets {
        counted
    } else {
        let entries = render::plural(u32::saturating_from(authored), "entry", "entries");
        format!("{entries} → {counted}")
    };
    if injected > 0 {
        let added = render::counted(u32::saturating_from(injected), "default");
        summary.push_str(" · ");
        summary.push_str(&added);
        summary.push_str(" added");
    }
    summary
}

// The thumbnails step reports from the build's own counts, which tell an
// asset with no preview apart from one rendered. `None` when it never ran.
fn thumbnail_summary(board: &Board, report: &BuildReport) -> Option<String> {
    let running = board.rows.last().filter(|r| r.state == RowState::Running)?;
    (running.step == Step::Cook(BuildStage::Thumbnails)).then(|| {
        board::thumbnail_summary(
            u32::saturating_from(report.thumbnails.baked),
            u32::saturating_from(report.thumbnails.reused),
        )
    })
}

// `path` relative to the working directory when it lies under it.
fn shown_path(path: &Path) -> String {
    let relative = std::env::current_dir()
        .ok()
        .and_then(|cwd| path.strip_prefix(cwd).ok().map(Path::to_path_buf));
    relative.as_deref().unwrap_or(path).display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_load_summary_shows_expansion_and_added_defaults() {
        assert_eq!(load_summary(3, 3, 0), "3 assets");
        assert_eq!(load_summary(1, 40, 0), "1 entry → 40 assets");
        assert_eq!(
            load_summary(2, 1_204, 2),
            "2 entries → 1,204 assets · 2 defaults added"
        );
    }

    #[test]
    fn a_path_under_the_working_directory_is_shown_relative() {
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(
            shown_path(&cwd.join("worlds").join("a.jsonl")),
            Path::new("worlds").join("a.jsonl").display().to_string()
        );
        assert_eq!(shown_path(Path::new("x/y.jsonl")), "x/y.jsonl");
    }
}
