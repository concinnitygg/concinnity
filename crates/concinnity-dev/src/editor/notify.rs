//! Editor notifications: the inbox of toast messages the rest of the editor
//! (and its worker threads) pushes into, the long operations they report
//! progress on, and the feed that turns both into the engine's `Toast` events.
//! The Console panel keeps the full history; a toast is the additional
//! at-a-glance surface for a result the user should not need the console open
//! to see.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use concinnity_engine::components::{Toast, ToastProgress, ToastTone};

// Visible card cap; older live toasts collapse into the "+N more" line.
pub(crate) const MAX_VISIBLE: u32 = 4;

// Seconds a finished operation's card takes to fade away.
const OP_FADE_S: f32 = 0.3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Level {
    Info,
    Success,
    Error,
}

impl Level {
    // How long a toast of this level lives; `None` holds until clicked.
    fn ttl(self) -> Option<Duration> {
        match self {
            Self::Info | Self::Success => Some(Duration::from_secs(4)),
            Self::Error => None,
        }
    }

    fn tone(self) -> ToastTone {
        match self {
            Self::Info => ToastTone::Info,
            Self::Success => ToastTone::Success,
            Self::Error => ToastTone::Error,
        }
    }
}

// What clicking a toast does, beyond dismissing it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Action {
    OpenConsole,
    GoToBehaviorFault,
    // Select the asset of this name and open its editing surface.
    SelectAsset(String),
}

// A message pushed and not yet fed to the toasts.
#[derive(Debug)]
struct Note {
    level: Level,
    message: String,
    action: Option<Action>,
}

// A long operation's shared progress state: the worker that runs it updates
// the counters, the feed reads them each frame. `total == 0` marks progress
// as indeterminate (the operation cannot count its work yet).
struct OpShared {
    id: u64,
    label: String,
    done: AtomicU32,
    total: AtomicU32,
    finished: AtomicBool,
}

impl OpShared {
    fn progress(&self) -> ToastProgress {
        let total = self.total.load(Ordering::Relaxed);
        if total == 0 {
            return ToastProgress::Ongoing;
        }
        let done = self.done.load(Ordering::Relaxed);
        ToastProgress::Fraction((done as f32 / total as f32).min(1.0))
    }
}

// The worker's handle to its operation card. Explicitly finished on
// completion; dropping the last handle (a worker unwinding included) also
// retires the card, so no path can leave one up forever.
#[derive(Clone)]
pub(crate) struct OpHandle(Arc<OpShared>);

impl OpHandle {
    pub(crate) fn set(&self, done: u32, total: u32) {
        self.0.done.store(done, Ordering::Relaxed);
        self.0.total.store(total, Ordering::Relaxed);
    }

    pub(crate) fn finish(&self) {
        self.0.finished.store(true, Ordering::Relaxed);
    }
}

#[derive(Default)]
struct Queue {
    // Push order: oldest first.
    notes: Vec<Note>,
    ops: Vec<Arc<OpShared>>,
}

// The ids operations are told apart by, across every notifier.
static NEXT_OP: AtomicU64 = AtomicU64::new(0);

// The shared handle to the inbox: cloned into worker threads (a cook worker
// reports its finish through it) and drained by the hook each frame. A
// poisoned lock drops the push rather than panicking the frame loop.
#[derive(Clone, Default)]
pub(crate) struct Notifier(Arc<Mutex<Queue>>);

impl Notifier {
    pub(crate) fn push(&self, level: Level, message: &str) {
        self.push_with(level, message, None);
    }

    pub(crate) fn push_with(&self, level: Level, message: &str, action: Option<Action>) {
        if let Ok(mut q) = self.0.lock() {
            q.notes.push(Note {
                level,
                message: message.to_string(),
                action,
            });
        }
    }

    pub(crate) fn info(&self, message: &str) {
        self.push(Level::Info, message);
    }

    pub(crate) fn success(&self, message: &str) {
        self.push(Level::Success, message);
    }

    pub(crate) fn error_with(&self, message: &str, action: Action) {
        self.push_with(Level::Error, message, Some(action));
    }

    // Start an operation card; the returned handle is the worker's reporter.
    pub(crate) fn begin_op(&self, label: &str) -> OpHandle {
        let shared = Arc::new(OpShared {
            id: NEXT_OP.fetch_add(1, Ordering::Relaxed),
            label: label.to_string(),
            done: AtomicU32::new(0),
            total: AtomicU32::new(0),
            finished: AtomicBool::new(false),
        });
        // A poisoned queue still hands out a working (orphan) handle.
        if let Ok(mut q) = self.0.lock() {
            q.ops.push(Arc::clone(&shared));
        }
        OpHandle(shared)
    }
}

// A toast the feed shows, kept to replay into a fresh world and to find its
// action when it is clicked away.
#[derive(Debug)]
struct Live {
    key: String,
    level: Level,
    message: String,
    action: Option<Action>,
    born: Instant,
}

impl Live {
    fn expired(&self, now: Instant) -> bool {
        self.level
            .ttl()
            .is_some_and(|ttl| now.saturating_duration_since(self.born) >= ttl)
    }

    // The toast showing this message for the time it has left.
    fn toast(&self, now: Instant) -> Toast {
        let seconds = self.level.ttl().map_or(f32::INFINITY, |ttl| {
            ttl.saturating_sub(now.saturating_duration_since(self.born))
                .as_secs_f32()
        });
        Toast {
            key: self.key.clone(),
            seconds,
            ..Toast::new(&self.message, self.level.tone())
        }
    }
}

// An operation the feed shows, and the progress it last showed. It holds no
// handle, so a worker dropping its own still retires the operation.
struct ShownOp {
    id: u64,
    label: String,
    progress: ToastProgress,
}

impl ShownOp {
    fn toast(&self) -> Toast {
        Toast {
            key: op_key(self.id),
            seconds: f32::INFINITY,
            progress: Some(self.progress),
            ..Toast::new(&self.label, ToastTone::Info)
        }
    }
}

fn op_key(id: u64) -> String {
    format!("op.{id}")
}

fn note_key(message: &str) -> String {
    format!("note.{message}")
}

// The hook's side of the toasts: what it has shown, fed into the engine's
// toast stack as `Toast` events.
#[derive(Default)]
pub(crate) struct ToastFeed {
    live: Vec<Live>,
    ops: Vec<ShownOp>,
}

impl ToastFeed {
    // Whether any toast or operation is shown.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.live.is_empty() && self.ops.is_empty()
    }

    // The messages shown, oldest first.
    #[cfg(test)]
    pub(crate) fn messages(&self) -> impl Iterator<Item = &str> {
        self.live.iter().map(|l| l.message.as_str())
    }

    // Take in what was pushed since the last frame and the operations'
    // progress, writing a toast for each change into `out`.
    pub(crate) fn feed(&mut self, notifier: &Notifier, now: Instant, out: &mut Vec<Toast>) {
        self.live.retain(|l| !l.expired(now));
        let Ok(mut q) = notifier.0.lock() else {
            return;
        };
        for note in q.notes.drain(..) {
            self.show(note, now, out);
        }
        // An op retires when finished, or when every worker-side handle is
        // gone (the queue's own Arc is the last one standing).
        q.ops
            .retain(|op| !op.finished.load(Ordering::Relaxed) && Arc::strong_count(op) > 1);
        self.ops.retain(|shown| {
            let running = q.ops.iter().any(|op| op.id == shown.id);
            if !running {
                out.push(Toast {
                    seconds: OP_FADE_S,
                    progress: None,
                    ..shown.toast()
                });
            }
            running
        });
        for op in &q.ops {
            let progress = op.progress();
            match self.ops.iter_mut().find(|s| s.id == op.id) {
                Some(shown) if shown.progress == progress => {}
                Some(shown) => {
                    shown.progress = progress;
                    out.push(shown.toast());
                }
                None => {
                    let shown = ShownOp {
                        id: op.id,
                        label: op.label.clone(),
                        progress,
                    };
                    out.push(shown.toast());
                    self.ops.push(shown);
                }
            }
        }
    }

    // A repeat of a live message refreshes its card instead of stacking a
    // twin.
    fn show(&mut self, note: Note, now: Instant, out: &mut Vec<Toast>) {
        let key = note_key(&note.message);
        let live = match self.live.iter().position(|l| l.key == key) {
            Some(i) => {
                let mut l = self.live.remove(i);
                l.level = note.level;
                l.action = note.action;
                l.born = now;
                l
            }
            None => Live {
                key,
                level: note.level,
                message: note.message,
                action: note.action,
                born: now,
            },
        };
        out.push(live.toast(now));
        self.live.push(live);
    }

    // Every shown toast with time left, for a world that has none yet.
    pub(crate) fn replay(&mut self, now: Instant, out: &mut Vec<Toast>) {
        self.live.retain(|l| !l.expired(now));
        out.extend(self.live.iter().map(|l| l.toast(now)));
        out.extend(self.ops.iter().map(ShownOp::toast));
    }

    // The card `key` was clicked away: forget it, and hand back its action.
    pub(crate) fn dismiss(&mut self, key: &str) -> Option<Action> {
        let i = self.live.iter().position(|l| l.key == key)?;
        self.live.remove(i).action
    }
}

#[cfg(test)]
mod tests;
