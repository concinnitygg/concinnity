//! Routes what the build logs into the status display, so a warning the cook
//! raises lands above the rows instead of tearing through them, and the
//! per-asset chatter below the warning level stays out of the way.

use std::sync::{Arc, Mutex, OnceLock};

use tracing::Level;
use tracing_subscriber::layer::{Context, Layer};

use super::Shared;
use super::board::NoteLevel;

// The display notes go to while a build runs, and the most verbose level it
// shows.
static ACTIVE: Mutex<Option<(Arc<Shared>, Level)>> = Mutex::new(None);

// Install the process's tracing subscriber: this layer plus the crash ring,
// which keeps the recent log for a crash report. A no-op once installed, or
// when another subscriber already is.
pub(super) fn install() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        let _ = tracing_subscriber::registry()
            .with(NoteLayer)
            .with(concinnity_engine::crash::RingLayer)
            .try_init();
    });
}

// Send notes to `shared` until `release`, showing events up to `level`.
pub(super) fn attach(shared: Arc<Shared>, level: Level) {
    *lock() = Some((shared, level));
}

pub(super) fn release() {
    *lock() = None;
}

fn lock() -> std::sync::MutexGuard<'static, Option<(Arc<Shared>, Level)>> {
    ACTIVE.lock().unwrap_or_else(|e| e.into_inner())
}

// The note an event at `level` becomes, or `None` when it is more verbose
// than `shown`.
fn note_level(level: Level, shown: Level) -> Option<NoteLevel> {
    if level > shown {
        return None;
    }
    Some(match level {
        Level::ERROR => NoteLevel::Error,
        Level::WARN => NoteLevel::Warning,
        _ => NoteLevel::Info,
    })
}

struct NoteLayer;

// Nothing past info ever becomes a note, so those callsites are never
// enabled; the crash ring keeps no more than that either.
impl<S: tracing::Subscriber> Layer<S> for NoteLayer {
    fn register_callsite(
        &self,
        meta: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        if *meta.level() <= Level::INFO {
            tracing::subscriber::Interest::always()
        } else {
            tracing::subscriber::Interest::never()
        }
    }

    fn enabled(&self, meta: &tracing::Metadata<'_>, _ctx: Context<'_, S>) -> bool {
        *meta.level() <= Level::INFO
    }

    fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
        Some(tracing::level_filters::LevelFilter::INFO)
    }

    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let level = *event.metadata().level();
        let active = lock().clone();
        let shown = active.as_ref().map_or(Level::WARN, |(_, shown)| *shown);
        let Some(note_level) = note_level(level, shown) else {
            return;
        };
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        let target = visitor
            .log_target
            .as_deref()
            .unwrap_or(event.metadata().target());
        let text = match origin(target) {
            Some(origin) => format!("{origin}: {}{}", visitor.message, visitor.fields),
            None => format!("{}{}", visitor.message, visitor.fields),
        };
        match active {
            Some((shared, _)) => shared.note(note_level, text),
            // Between builds: written straight out, as the log would have.
            None => {
                let note = super::board::Note {
                    level: note_level,
                    text,
                };
                for line in super::render::note(&note) {
                    eprintln!("{}", line.plain_text());
                }
            }
        }
    }
}

// The crate a message from outside the engine came from, which names it in
// the note; `None` for the engine's own.
fn origin(target: &str) -> Option<&str> {
    let krate = target.split("::").next().unwrap_or(target);
    (!krate.starts_with("concinnity")).then_some(krate)
}

// An event's message and fields. A message bridged from the `log` crate
// carries its origin as `log.*` fields: its target is kept to name it, the
// rest is dropped.
#[derive(Default)]
struct MessageVisitor {
    message: String,
    fields: String,
    log_target: Option<String>,
}

impl tracing::field::Visit for MessageVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        use std::fmt::Write as _;
        match field.name() {
            "message" => {
                let _ = write!(self.message, "{value:?}");
            }
            name if name.starts_with("log.") => {}
            name => {
                let _ = write!(self.fields, " {name}={value:?}");
            }
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        match field.name() {
            "message" => self.message.push_str(value),
            "log.target" => self.log_target = Some(value.to_string()),
            _ => self.record_debug(field, &value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_above_the_shown_level_are_dropped() {
        assert_eq!(
            note_level(Level::WARN, Level::WARN),
            Some(NoteLevel::Warning)
        );
        assert_eq!(
            note_level(Level::ERROR, Level::WARN),
            Some(NoteLevel::Error)
        );
        assert_eq!(note_level(Level::INFO, Level::WARN), None);
        assert_eq!(note_level(Level::INFO, Level::INFO), Some(NoteLevel::Info));
        assert_eq!(note_level(Level::DEBUG, Level::INFO), None);
    }

    #[test]
    fn a_message_from_outside_the_engine_names_its_crate() {
        assert_eq!(origin("fbxcel::tree::any"), Some("fbxcel"));
        assert_eq!(origin("naga"), Some("naga"));
        assert_eq!(origin("concinnity_cook::compile::program"), None);
    }
}
