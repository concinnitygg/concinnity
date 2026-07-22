// src/editor/console.rs
//
// The Console panel's core: the bounded log ring, the shared sink other editor
// code (and worker threads) push lines into, a tracing layer mirroring this
// crate's events into that sink, the slash-command parser, and the /del name
// autocomplete matcher. Everything here is pure or lock-guarded state; the
// panel layout lives in `console_panel.rs` and the actions in
// `hook/console_edit.rs`.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

// Retained log lines; older lines fall off the front.
pub(crate) const LOG_CAP: usize = 300;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Severity {
    Info,
    Warn,
    Error,
    // An echoed input line (drawn dimmer, prefixed "> ").
    Command,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ConsoleLine {
    pub severity: Severity,
    pub text: String,
}

// The bounded log ring. Multi-line pushes split into one entry per line so the
// window math stays line-based.
#[derive(Debug)]
pub(crate) struct ConsoleLog {
    lines: VecDeque<ConsoleLine>,
    cap: usize,
}

impl ConsoleLog {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            lines: VecDeque::new(),
            cap: cap.max(1),
        }
    }

    pub(crate) fn push(&mut self, severity: Severity, text: &str) {
        for line in text.lines() {
            if self.lines.len() == self.cap {
                self.lines.pop_front();
            }
            self.lines.push_back(ConsoleLine {
                severity,
                text: line.to_string(),
            });
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.lines.len()
    }

    // Clone the `count` lines starting at `first` (short at the tail).
    pub(crate) fn window(&self, first: usize, count: usize) -> Vec<ConsoleLine> {
        self.lines.iter().skip(first).take(count).cloned().collect()
    }
}

// The shared handle to the log: cloned into worker threads and the tracing
// layer, and read by the panel each frame. A poisoned lock drops the line
// rather than panicking the frame loop.
#[derive(Clone)]
pub(crate) struct ConsoleSink(Arc<Mutex<ConsoleLog>>);

impl Default for ConsoleSink {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(ConsoleLog::new(LOG_CAP))))
    }
}

impl ConsoleSink {
    pub(crate) fn push(&self, severity: Severity, text: &str) {
        if let Ok(mut log) = self.0.lock() {
            log.push(severity, text);
        }
    }

    // Never blocks: the tracing layer may fire from any thread mid-frame, and
    // losing a mirrored line beats stalling whoever holds the lock.
    fn push_nonblocking(&self, severity: Severity, text: &str) {
        if let Ok(mut log) = self.0.try_lock() {
            log.push(severity, text);
        }
    }

    pub(crate) fn info(&self, text: &str) {
        self.push(Severity::Info, text);
    }
    pub(crate) fn warn(&self, text: &str) {
        self.push(Severity::Warn, text);
    }
    pub(crate) fn error(&self, text: &str) {
        self.push(Severity::Error, text);
    }

    pub(crate) fn len(&self) -> usize {
        self.0.lock().map(|log| log.len()).unwrap_or(0)
    }

    pub(crate) fn window(&self, first: usize, count: usize) -> Vec<ConsoleLine> {
        self.0
            .lock()
            .map(|log| log.window(first, count))
            .unwrap_or_default()
    }
}

// A tracing layer mirroring this crate's events into the sink, so editor
// logging (save failures, live-preview rebuild errors, rm notices) shows in
// the console without each site pushing explicitly. Filtered to the editor
// crate and INFO+ at install (`install_tracing`); bounded by the ring and
// non-blocking by `push_nonblocking`.
struct ConsoleTracingLayer {
    sink: ConsoleSink,
}

struct MessageVisitor(String);

impl tracing::field::Visit for MessageVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            use std::fmt::Write;
            let _ = write!(self.0, "{value:?}");
        }
    }
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for ConsoleTracingLayer {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut visitor = MessageVisitor(String::new());
        event.record(&mut visitor);
        if visitor.0.is_empty() {
            return;
        }
        let severity = match *event.metadata().level() {
            tracing::Level::ERROR => Severity::Error,
            tracing::Level::WARN => Severity::Warn,
            _ => Severity::Info,
        };
        self.sink.push_nonblocking(severity, &visitor.0);
    }
}

// Install the editor's global tracing subscriber: the usual stderr formatter
// (same RUST_LOG handling as the engine's `init_logging`) plus the console
// mirror for this crate's INFO+ events. Replaces the `init_logging` call in
// the editor entry point; a no-op if a subscriber is already installed.
pub(crate) fn install_tracing(sink: ConsoleSink) {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::{EnvFilter, Layer};

    let default = if cfg!(debug_assertions) {
        "info"
    } else {
        "warn"
    };
    let fmt = tracing_subscriber::fmt::layer()
        .with_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default)));
    let mirror =
        ConsoleTracingLayer { sink }.with_filter(tracing_subscriber::filter::filter_fn(|meta| {
            meta.target().starts_with("concinnity_editor") && *meta.level() <= tracing::Level::INFO
        }));
    let _ = tracing_subscriber::registry()
        .with(fmt)
        .with(mirror)
        .try_init();
}

// A parsed console input line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Command {
    // A line without a leading '/': just echoed to the log.
    Echo(String),
    Add {
        target: String,
        name: Option<String>,
    },
    Del {
        name: String,
    },
    // Compile the current world's blobs (named "cook" in user-facing text).
    Cook,
    Help,
}

// One registered slash command: its /help row plus its argument parser.
// Adding a command is one entry here (plus its `Command` variant and dispatch
// arm in `hook/console_edit.rs`).
pub(crate) struct CommandSpec {
    pub name: &'static str,
    pub usage: &'static str,
    pub blurb: &'static str,
    parse: fn(&str) -> Result<Command, String>,
}

pub(crate) const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        name: "add",
        usage: "/add <target> [name]",
        blurb: "add an asset: a file path, a type name, or inline JSON",
        parse: parse_add,
    },
    CommandSpec {
        name: "del",
        usage: "/del <name>",
        blurb: "remove an authored asset by name",
        parse: parse_del,
    },
    CommandSpec {
        name: "cook",
        usage: "/cook",
        blurb: "compile the current world's blobs",
        parse: parse_cook,
    },
    CommandSpec {
        name: "help",
        usage: "/help",
        blurb: "list commands",
        parse: parse_help,
    },
];

fn parse_add(rest: &str) -> Result<Command, String> {
    if rest.is_empty() {
        return Err("usage: /add <target> [name]".to_string());
    }
    // Inline JSON contains spaces; the whole remainder is the target.
    if rest.starts_with('{') {
        return Ok(Command::Add {
            target: rest.to_string(),
            name: None,
        });
    }
    let mut words = rest.split_whitespace();
    let target = words.next().unwrap_or_default().to_string();
    let name = words.next().map(String::from);
    if words.next().is_some() {
        return Err("too many arguments; usage: /add <target> [name]".to_string());
    }
    Ok(Command::Add { target, name })
}

fn parse_del(rest: &str) -> Result<Command, String> {
    let mut words = rest.split_whitespace();
    match (words.next(), words.next()) {
        (Some(name), None) => Ok(Command::Del {
            name: name.to_string(),
        }),
        _ => Err("usage: /del <name>".to_string()),
    }
}

fn parse_cook(rest: &str) -> Result<Command, String> {
    if rest.is_empty() {
        Ok(Command::Cook)
    } else {
        Err("usage: /cook".to_string())
    }
}

fn parse_help(rest: &str) -> Result<Command, String> {
    if rest.is_empty() {
        Ok(Command::Help)
    } else {
        Err("usage: /help".to_string())
    }
}

// Parse one submitted input line. A line without a leading '/' echoes; an
// unknown or malformed command errors with its usage.
pub(crate) fn parse_command(line: &str) -> Result<Command, String> {
    let line = line.trim();
    let Some(rest) = line.strip_prefix('/') else {
        return Ok(Command::Echo(line.to_string()));
    };
    let (name, args) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
    let Some(spec) = COMMANDS.iter().find(|c| c.name == name) else {
        return Err(format!("unknown command /{name}; try /help"));
    };
    (spec.parse)(args.trim())
}

// The /help body, one row per registered command.
pub(crate) fn help_lines() -> Vec<String> {
    let width = COMMANDS.iter().map(|c| c.usage.len()).max().unwrap_or(0);
    let mut lines: Vec<String> = COMMANDS
        .iter()
        .map(|c| format!("{:width$}  {}", c.usage, c.blurb))
        .collect();
    lines.push("a line without / is echoed to the log".to_string());
    lines
}

// The best completion of `prefix` among `names`: the shortest strictly longer
// prefix match, ties broken lexicographically so the pick is deterministic.
pub(crate) fn completion<'a>(
    prefix: &str,
    names: impl IntoIterator<Item = &'a str>,
) -> Option<&'a str> {
    if prefix.is_empty() {
        return None;
    }
    names
        .into_iter()
        .filter(|n| n.starts_with(prefix) && n.len() > prefix.len())
        .min_by_key(|n| (n.len(), *n))
}

// The inline ghost suffix for a partially typed `/del <name>`, or `None` when
// the input is any other shape (ghost only ever completes the name argument).
pub(crate) fn del_ghost<'a>(
    input: &str,
    names: impl IntoIterator<Item = &'a str>,
) -> Option<String> {
    let rest = input.strip_prefix("/del")?;
    let prefix = rest.strip_prefix(' ')?.trim_start();
    if prefix.is_empty() || prefix.contains(char::is_whitespace) {
        return None;
    }
    completion(prefix, names).map(|full| full[prefix.len()..].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_ring_drops_the_oldest_past_the_cap() {
        let mut log = ConsoleLog::new(3);
        for i in 0..5 {
            log.push(Severity::Info, &format!("line {i}"));
        }
        assert_eq!(log.len(), 3);
        let lines = log.window(0, 10);
        let texts: Vec<&str> = lines.iter().map(|l| l.text.as_str()).collect();
        assert_eq!(texts, ["line 2", "line 3", "line 4"]);
    }

    #[test]
    fn multi_line_push_splits_and_empty_push_adds_nothing() {
        let mut log = ConsoleLog::new(10);
        log.push(Severity::Error, "first\nsecond");
        log.push(Severity::Info, "");
        assert_eq!(log.len(), 2);
        let lines = log.window(0, 10);
        assert_eq!(lines[0].text, "first");
        assert_eq!(lines[1].text, "second");
        assert_eq!(lines[1].severity, Severity::Error);
    }

    #[test]
    fn window_clips_to_the_tail() {
        let mut log = ConsoleLog::new(10);
        for i in 0..4 {
            log.push(Severity::Info, &format!("{i}"));
        }
        let lines = log.window(2, 5);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "2");
        assert!(log.window(9, 5).is_empty());
    }

    #[test]
    fn sink_is_shared_across_clones() {
        let sink = ConsoleSink::default();
        let clone = sink.clone();
        clone.info("from a worker");
        assert_eq!(sink.len(), 1);
        assert_eq!(sink.window(0, 1)[0].text, "from a worker");
    }

    #[test]
    fn bare_lines_echo() {
        assert_eq!(
            parse_command("hello world"),
            Ok(Command::Echo("hello world".to_string()))
        );
    }

    #[test]
    fn add_parses_target_optional_name_and_inline_json() {
        assert_eq!(
            parse_command("/add Logger"),
            Ok(Command::Add {
                target: "Logger".to_string(),
                name: None,
            })
        );
        assert_eq!(
            parse_command("/add models/scene.glb hall"),
            Ok(Command::Add {
                target: "models/scene.glb".to_string(),
                name: Some("hall".to_string()),
            })
        );
        let json = r#"{"type": "Logger", "name": "log"}"#;
        assert_eq!(
            parse_command(&format!("/add {json}")),
            Ok(Command::Add {
                target: json.to_string(),
                name: None,
            })
        );
        assert!(parse_command("/add").is_err());
        assert!(parse_command("/add a b c").is_err());
    }

    #[test]
    fn del_takes_exactly_one_name() {
        assert_eq!(
            parse_command("/del cube"),
            Ok(Command::Del {
                name: "cube".to_string(),
            })
        );
        assert!(parse_command("/del").is_err());
        assert!(parse_command("/del a b").is_err());
    }

    #[test]
    fn cook_and_help_take_no_arguments() {
        assert_eq!(parse_command("/cook"), Ok(Command::Cook));
        assert_eq!(parse_command("/help"), Ok(Command::Help));
        assert!(parse_command("/cook now").is_err());
        assert!(parse_command("/help me").is_err());
    }

    #[test]
    fn unknown_commands_point_at_help() {
        let err = parse_command("/frobnicate").unwrap_err();
        assert!(err.contains("/frobnicate"), "got: {err}");
        assert!(err.contains("/help"), "got: {err}");
    }

    #[test]
    fn help_covers_every_registered_command() {
        let help = help_lines().join("\n");
        for spec in COMMANDS {
            assert!(help.contains(spec.usage), "missing {}", spec.usage);
        }
    }

    #[test]
    fn completion_prefers_the_shortest_then_lexicographic_match() {
        let names = ["cube_red", "cube", "cube_blue", "wall"];
        assert_eq!(completion("cu", names), Some("cube"));
        assert_eq!(
            completion("cube_", names),
            Some("cube_red"),
            "shorter beats lexicographically-earlier"
        );
        assert_eq!(completion("wal", names), Some("wall"));
        assert_eq!(completion("x", names), None);
        assert_eq!(completion("", names), None, "empty prefix never completes");
        assert_eq!(
            completion("cube", ["cube"]),
            None,
            "an exact match has nothing left to complete"
        );
    }

    #[test]
    fn del_ghost_completes_only_the_del_name_argument() {
        let names = ["cube_red", "wall"];
        assert_eq!(del_ghost("/del cu", names), Some("be_red".to_string()));
        assert_eq!(del_ghost("/del cube_red", names), None);
        assert_eq!(del_ghost("/del ", names), None);
        assert_eq!(del_ghost("/del a b", names), None);
        assert_eq!(del_ghost("/add cu", names), None);
        assert_eq!(del_ghost("cu", names), None);
    }
}
