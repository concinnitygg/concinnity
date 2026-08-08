// src/crash/ring.rs
//
// Bounded in-memory capture of recent tracing events for crash reports. A
// fixed ring of recycled line buffers behind a mutex: the write path formats
// into an existing buffer with byte-capped output, so steady state allocates
// nothing and the lock is held only while one line is formatted. Purely
// passive: it sees only events other code already emits.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

pub(crate) const RING_CAPACITY: usize = 256;
pub(crate) const MAX_LINE_BYTES: usize = 256;

pub(crate) struct LogRing {
    lines: Mutex<VecDeque<String>>,
    started: Instant,
}

impl LogRing {
    pub(crate) fn new() -> Self {
        Self {
            lines: Mutex::new(VecDeque::with_capacity(RING_CAPACITY)),
            started: Instant::now(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, VecDeque<String>> {
        // A panic while the lock was held leaves at worst a garbled line;
        // recent logs are still worth reporting.
        self.lines.lock().unwrap_or_else(|p| p.into_inner())
    }

    // Append one line, evicting the oldest at capacity and recycling its
    // buffer. `fill` receives a cleared buffer wrapped in a byte cap.
    fn push_with(&self, fill: impl FnOnce(&mut BoundedLine<'_>)) {
        let mut lines = self.lock();
        let mut buf = if lines.len() == RING_CAPACITY {
            lines.pop_front().unwrap_or_default()
        } else {
            String::with_capacity(MAX_LINE_BYTES)
        };
        buf.clear();
        fill(&mut BoundedLine(&mut buf));
        lines.push_back(buf);
    }

    pub(crate) fn push_event(&self, level: &tracing::Level, target: &str, event: &tracing::Event) {
        let elapsed = self.started.elapsed().as_secs_f64();
        self.push_with(|line| {
            let _ = write!(line, "+{elapsed:.3}s {level} {target}: ");
            event.record(&mut LineVisitor {
                line,
                seen_any: false,
            });
        });
    }

    // Oldest-first copy of the ring. Allocates; called only when a report is
    // being written.
    pub(crate) fn snapshot(&self) -> Vec<String> {
        self.lock().iter().cloned().collect()
    }

    // Snapshot that gives up instead of blocking, for use where the crashing
    // thread may already hold the lock.
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    pub(crate) fn try_snapshot(&self) -> Vec<String> {
        match self.lines.try_lock() {
            Ok(lines) => lines.iter().cloned().collect(),
            Err(_) => Vec::new(),
        }
    }
}

pub(crate) fn global() -> &'static LogRing {
    static RING: OnceLock<LogRing> = OnceLock::new();
    RING.get_or_init(LogRing::new)
}

// Byte-capped sink for one ring line: appends stop (on a char boundary) once
// the line reaches `MAX_LINE_BYTES`, so no event can grow a buffer past its
// preallocated capacity.
struct BoundedLine<'a>(&'a mut String);

impl std::fmt::Write for BoundedLine<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let remaining = MAX_LINE_BYTES.saturating_sub(self.0.len());
        if remaining == 0 {
            return Ok(());
        }
        if s.len() <= remaining {
            self.0.push_str(s);
        } else {
            let mut end = remaining;
            while !s.is_char_boundary(end) {
                end -= 1;
            }
            self.0.push_str(&s[..end]);
        }
        Ok(())
    }
}

// Formats an event's fields into the line: the `message` field verbatim,
// every other field as ` key=value`.
struct LineVisitor<'a, 'b> {
    line: &'a mut BoundedLine<'b>,
    seen_any: bool,
}

impl tracing::field::Visit for LineVisitor<'_, '_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.line, "{value:?}");
        } else {
            let sep = if self.seen_any { " " } else { "" };
            let _ = write!(self.line, "{sep}{}={value:?}", field.name());
        }
        self.seen_any = true;
    }
}

/// A `tracing` layer that keeps the most recent INFO-and-above log lines in a
/// bounded in-memory ring for inclusion in crash reports. Passive and cheap:
/// lines are formatted into recycled fixed-size buffers under a short lock,
/// and nothing is emitted per frame by the layer itself.
pub struct RingLayer;

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for RingLayer {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let meta = event.metadata();
        if *meta.level() > tracing::Level::INFO {
            return;
        }
        global().push_event(meta.level(), meta.target(), event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_line(ring: &LogRing, text: &str) {
        ring.push_with(|line| {
            let _ = write!(line, "{text}");
        });
    }

    #[test]
    fn ring_keeps_the_newest_lines_in_order() {
        let ring = LogRing::new();
        for i in 0..RING_CAPACITY + 40 {
            push_line(&ring, &format!("line {i}"));
        }
        let snap = ring.snapshot();
        assert_eq!(snap.len(), RING_CAPACITY);
        assert_eq!(snap.first().unwrap(), "line 40");
        assert_eq!(
            snap.last().unwrap(),
            &format!("line {}", RING_CAPACITY + 39)
        );
    }

    #[test]
    fn lines_cap_at_the_byte_limit_on_char_boundaries() {
        let ring = LogRing::new();
        push_line(&ring, &"\u{e9}".repeat(MAX_LINE_BYTES));
        let snap = ring.snapshot();
        assert_eq!(snap.len(), 1);
        assert!(snap[0].len() <= MAX_LINE_BYTES);
        assert!(snap[0].chars().all(|c| c == '\u{e9}'));
    }

    #[test]
    fn concurrent_writers_never_lose_the_ring() {
        let ring = std::sync::Arc::new(LogRing::new());
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let ring = ring.clone();
                std::thread::spawn(move || {
                    for i in 0..200 {
                        push_line(&ring, &format!("t{t} line {i}"));
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let snap = ring.snapshot();
        assert_eq!(snap.len(), RING_CAPACITY);
        // Every retained line is complete, none were torn by contention.
        assert!(
            snap.iter()
                .all(|l| l.starts_with('t') && l.contains(" line "))
        );
    }

    #[test]
    fn layer_mirrors_info_events_into_the_global_ring() {
        use tracing_subscriber::layer::SubscriberExt;
        let subscriber = tracing_subscriber::registry().with(RingLayer);
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(answer = 42, "ring capture probe");
            tracing::debug!("below the ring threshold");
        });
        let snap = global().snapshot();
        let probe = snap.iter().find(|l| l.contains("ring capture probe"));
        let probe = probe.expect("INFO event captured");
        assert!(probe.contains("answer=42"));
        assert!(probe.contains("INFO"));
        assert!(!snap.iter().any(|l| l.contains("below the ring threshold")));
    }
}
