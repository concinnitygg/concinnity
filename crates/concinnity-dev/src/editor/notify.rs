// src/editor/notify.rs
//
// Editor notifications: the queue of transient toast messages the rest of the
// editor (and its worker threads) pushes into, plus the pure lifetime and
// stack policy the overlay draws from. The card geometry lives in
// `toast_overlay.rs`; the per-frame drive and click routing in
// `hook/notify_drive.rs`. The Console panel keeps the full history; a toast is
// the additional at-a-glance surface for a result the user should not need the
// console open to see.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// Visible card cap; older live toasts collapse into the "+N more" row.
pub(crate) const MAX_VISIBLE: usize = 4;

// Auto-dismissing toasts fade out over this tail of their lifetime.
const FADE: Duration = Duration::from_millis(300);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Level {
    Info,
    Success,
    Error,
}

impl Level {
    // How long a toast of this level lives; `None` is sticky until clicked.
    fn ttl(self) -> Option<Duration> {
        match self {
            Level::Info | Level::Success => Some(Duration::from_secs(4)),
            Level::Error => None,
        }
    }
}

// What clicking a toast does, beyond dismissing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Action {
    OpenConsole,
    GoToBehaviorFault,
}

#[derive(Debug)]
struct Toast {
    level: Level,
    message: String,
    action: Option<Action>,
    born: Instant,
}

impl Toast {
    // 1.0 for most of the toast's life, easing to 0.0 over the fade tail.
    // Sticky levels never fade.
    fn alpha(&self, now: Instant) -> f32 {
        let Some(ttl) = self.level.ttl() else {
            return 1.0;
        };
        let age = now.saturating_duration_since(self.born);
        if age >= ttl {
            return 0.0;
        }
        let left = ttl - age;
        (left.as_secs_f32() / FADE.as_secs_f32()).min(1.0)
    }

    fn expired(&self, now: Instant) -> bool {
        self.level
            .ttl()
            .is_some_and(|ttl| now.saturating_duration_since(self.born) >= ttl)
    }
}

// A long operation's shared progress state: the worker that runs it updates
// the counters, the drive reads them each frame. `total == 0` marks progress
// as indeterminate (the operation cannot count its work yet).
struct OpShared {
    label: String,
    done: AtomicU32,
    total: AtomicU32,
    finished: AtomicBool,
    born: Instant,
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

// One operation card, computed for this frame. `fraction` is real measured
// progress; `None` draws the indeterminate sweep (`phase` is the operation's
// age in seconds, which the overlay animates from).
#[derive(Debug)]
pub(crate) struct OpCard {
    pub label: String,
    pub fraction: Option<f32>,
    pub phase: f32,
}

// One visible card, newest first (slot 0 sits nearest the anchor corner). A
// card's action stays in the queue; `click_card` hands it back on dismissal.
#[derive(Debug)]
pub(crate) struct Card {
    pub level: Level,
    pub message: String,
    pub alpha: f32,
}

// The stack the overlay draws this frame.
#[derive(Debug, Default)]
pub(crate) struct Stack {
    // Running operations, nearest the anchor corner (below the toasts).
    pub ops: Vec<OpCard>,
    pub cards: Vec<Card>,
    // Live toasts beyond the visible cap (the "+N more" row; 0 hides it).
    pub overflow: usize,
}

#[derive(Default)]
struct Queue {
    // Push order: oldest first.
    toasts: Vec<Toast>,
    ops: Vec<Arc<OpShared>>,
}

impl Queue {
    fn push_at(&mut self, level: Level, message: String, action: Option<Action>, now: Instant) {
        // A repeat of a live toast refreshes it instead of stacking a twin.
        if let Some(t) = self
            .toasts
            .iter_mut()
            .find(|t| t.level == level && t.message == message)
        {
            t.born = now;
            t.action = action;
            return;
        }
        self.toasts.push(Toast {
            level,
            message,
            action,
            born: now,
        });
    }

    fn prune(&mut self, now: Instant) {
        self.toasts.retain(|t| !t.expired(now));
        // An op retires when finished, or when every worker-side handle is
        // gone (the queue's own Arc is the last one standing).
        self.ops
            .retain(|op| !op.finished.load(Ordering::Relaxed) && Arc::strong_count(op) > 1);
    }

    fn begin_op(&mut self, label: String, now: Instant) -> OpHandle {
        let shared = Arc::new(OpShared {
            label,
            done: AtomicU32::new(0),
            total: AtomicU32::new(0),
            finished: AtomicBool::new(false),
            born: now,
        });
        self.ops.push(shared.clone());
        OpHandle(shared)
    }

    fn stack_at(&mut self, now: Instant) -> Stack {
        self.prune(now);
        let ops = self
            .ops
            .iter()
            .map(|op| {
                let total = op.total.load(Ordering::Relaxed);
                OpCard {
                    label: op.label.clone(),
                    fraction: (total > 0)
                        .then(|| (op.done.load(Ordering::Relaxed) as f32 / total as f32).min(1.0)),
                    phase: now.saturating_duration_since(op.born).as_secs_f32(),
                }
            })
            .collect();
        let visible = self.toasts.len().min(MAX_VISIBLE);
        let cards = self
            .toasts
            .iter()
            .rev()
            .take(visible)
            .map(|t| Card {
                level: t.level,
                message: t.message.clone(),
                alpha: t.alpha(now),
            })
            .collect();
        Stack {
            ops,
            cards,
            overflow: self.toasts.len() - visible,
        }
    }

    // Remove the toast shown in visible slot `slot` (newest-first order) and
    // hand back its action.
    fn click_at(&mut self, slot: usize, now: Instant) -> Option<Action> {
        self.prune(now);
        let visible = self.toasts.len().min(MAX_VISIBLE);
        if slot >= visible {
            return None;
        }
        let index = self.toasts.len() - 1 - slot;
        self.toasts.remove(index).action
    }
}

// The shared handle to the queue: cloned into worker threads (a cook worker
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
            q.push_at(level, message.to_string(), action, Instant::now());
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
        match self.0.lock() {
            Ok(mut q) => q.begin_op(label.to_string(), Instant::now()),
            // A poisoned queue still hands out a working (orphan) handle.
            Err(_) => OpHandle(Arc::new(OpShared {
                label: label.to_string(),
                done: AtomicU32::new(0),
                total: AtomicU32::new(0),
                finished: AtomicBool::new(false),
                born: Instant::now(),
            })),
        }
    }

    // Whether any toast or operation is retained (expired-but-unpruned
    // included, so the drive still runs the frame that prunes them). The idle
    // fast path.
    pub(crate) fn is_empty(&self) -> bool {
        self.0
            .lock()
            .map(|q| q.toasts.is_empty() && q.ops.is_empty())
            .unwrap_or(true)
    }

    // The stack to draw this frame; prunes expired toasts on the way.
    pub(crate) fn stack(&self) -> Stack {
        self.0
            .lock()
            .map(|mut q| q.stack_at(Instant::now()))
            .unwrap_or_default()
    }

    // Dismiss the card in visible slot `slot`, returning its action.
    pub(crate) fn click_card(&self, slot: usize) -> Option<Action> {
        self.0
            .lock()
            .ok()
            .and_then(|mut q| q.click_at(slot, Instant::now()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queue_with(entries: &[(Level, &str)], now: Instant) -> Queue {
        let mut q = Queue::default();
        for (level, msg) in entries {
            q.push_at(*level, msg.to_string(), None, now);
        }
        q
    }

    #[test]
    fn info_expires_and_errors_stick() {
        let now = Instant::now();
        let mut q = queue_with(&[(Level::Info, "done"), (Level::Error, "broke")], now);
        let later = now + Duration::from_secs(10);
        let stack = q.stack_at(later);
        assert_eq!(stack.cards.len(), 1);
        assert_eq!(stack.cards[0].level, Level::Error);
        assert_eq!(stack.cards[0].alpha, 1.0, "sticky toasts never fade");
    }

    #[test]
    fn alpha_fades_over_the_tail_of_the_lifetime() {
        let now = Instant::now();
        let t = Toast {
            level: Level::Info,
            message: String::new(),
            action: None,
            born: now,
        };
        assert_eq!(t.alpha(now), 1.0);
        assert_eq!(t.alpha(now + Duration::from_secs(3)), 1.0, "pre-fade");
        let mid_fade = t.alpha(now + Duration::from_millis(3850));
        assert!(
            (0.0..1.0).contains(&mid_fade),
            "inside the fade tail: {mid_fade}"
        );
        assert_eq!(t.alpha(now + Duration::from_secs(4)), 0.0, "expired");
    }

    #[test]
    fn stack_caps_visible_cards_and_counts_overflow_newest_first() {
        let now = Instant::now();
        let mut q = Queue::default();
        for i in 0..6 {
            q.push_at(Level::Error, format!("e{i}"), None, now);
        }
        let stack = q.stack_at(now);
        assert_eq!(stack.cards.len(), MAX_VISIBLE);
        assert_eq!(stack.overflow, 2);
        assert_eq!(stack.cards[0].message, "e5", "slot 0 is the newest");
        assert_eq!(stack.cards[MAX_VISIBLE - 1].message, "e2");
    }

    #[test]
    fn a_repeat_refreshes_instead_of_stacking() {
        let now = Instant::now();
        let mut q = queue_with(&[(Level::Info, "saved")], now);
        let later = now + Duration::from_secs(3);
        q.push_at(Level::Info, "saved".to_string(), None, later);
        assert_eq!(q.toasts.len(), 1, "coalesced");
        // The refreshed toast lives a full lifetime from the repeat.
        let stack = q.stack_at(later + Duration::from_secs(3));
        assert_eq!(stack.cards.len(), 1);
        // The same text at a different level is a different toast.
        q.push_at(Level::Error, "saved".to_string(), None, later);
        assert_eq!(q.toasts.len(), 2);
    }

    #[test]
    fn click_dismisses_the_slot_and_returns_its_action() {
        let now = Instant::now();
        let mut q = Queue::default();
        q.push_at(Level::Error, "older".to_string(), None, now);
        q.push_at(
            Level::Error,
            "newest".to_string(),
            Some(Action::OpenConsole),
            now,
        );
        assert_eq!(q.click_at(0, now), Some(Action::OpenConsole));
        assert_eq!(q.toasts.len(), 1);
        assert_eq!(q.toasts[0].message, "older");
        assert_eq!(q.click_at(5, now), None, "a slot past the stack is a miss");
        assert_eq!(q.toasts.len(), 1);
    }

    #[test]
    fn ops_report_progress_and_retire_on_finish_or_drop() {
        let now = Instant::now();
        let mut q = Queue::default();
        let op = q.begin_op("Cooking".to_string(), now);
        let stack = q.stack_at(now);
        assert_eq!(stack.ops.len(), 1);
        assert_eq!(stack.ops[0].fraction, None, "no counts yet: indeterminate");
        op.set(3, 12);
        let stack = q.stack_at(now);
        assert_eq!(stack.ops[0].fraction, Some(0.25));
        op.finish();
        assert!(q.stack_at(now).ops.is_empty(), "finished ops retire");
        // Dropping the last worker handle retires the card too, so a worker
        // that unwinds can never leave one up forever.
        let op2 = q.begin_op("Saving".to_string(), now);
        assert_eq!(q.stack_at(now).ops.len(), 1);
        drop(op2);
        assert!(q.stack_at(now).ops.is_empty());
    }

    #[test]
    fn notifier_is_shared_across_clones() {
        let n = Notifier::default();
        let clone = n.clone();
        assert!(n.is_empty());
        clone.success("from a worker");
        assert!(!n.is_empty());
        let stack = n.stack();
        assert_eq!(stack.cards.len(), 1);
        assert_eq!(stack.cards[0].message, "from a worker");
        assert_eq!(
            n.click_card(0),
            None,
            "an actionless card dismisses plainly"
        );
        assert!(n.is_empty());
    }
}
