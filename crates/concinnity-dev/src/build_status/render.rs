//! Turns a board into styled lines of text. Nothing here touches the
//! terminal: the caller picks the width and whether to paint.

use std::time::{Duration, Instant};

use super::board::{Board, Note, NoteLevel, Outcome, Row, RowState};

const INDENT: &str = "  ";
const LABEL_WIDTH: usize = 12;
const SUMMARY_WIDTH: usize = 46;
const BAR_WIDTH: usize = 24;
const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
// An item in flight for longer than this has its age shown beside it.
const SLOW_ITEM: Duration = Duration::from_secs(1);

// The role a run of text plays, which decides its color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Tone {
    Plain,
    Label,
    Detail,
    Heading,
    Done,
    Fail,
    Warning,
    Progress,
}

impl Tone {
    fn sgr(self) -> &'static str {
        match self {
            Tone::Plain => "",
            Tone::Label => "1",
            Tone::Detail => "2",
            Tone::Heading => "1;32",
            Tone::Done => "32",
            Tone::Fail => "1;31",
            Tone::Warning => "1;33",
            Tone::Progress => "36",
        }
    }
}

// One line of output as styled runs of text.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct Line(Vec<(Tone, String)>);

impl Line {
    pub(super) fn push(mut self, tone: Tone, text: impl Into<String>) -> Self {
        self.0.push((tone, text.into()));
        self
    }

    pub(super) fn plain_text(&self) -> String {
        self.0.iter().map(|(_, text)| text.as_str()).collect()
    }

    // The line cut to `width` columns, its last column an ellipsis when cut.
    pub(super) fn fit(self, width: usize) -> Self {
        let len: usize = self.0.iter().map(|(_, t)| t.chars().count()).sum();
        if len <= width {
            return self;
        }
        let mut room = width.saturating_sub(1);
        let mut out = Line::default();
        for (tone, text) in self.0 {
            if room == 0 {
                break;
            }
            let kept: String = text.chars().take(room).collect();
            room -= kept.chars().count();
            out.0.push((tone, kept));
        }
        if width > 0 {
            out.0.push((Tone::Detail, "…".to_string()));
        }
        out
    }

    // The line as terminal text: escape-coded when `color`, plain otherwise.
    pub(super) fn paint(&self, color: bool) -> String {
        if !color {
            return self.plain_text();
        }
        let mut out = String::new();
        for (tone, text) in &self.0 {
            match tone.sgr() {
                "" => out.push_str(text),
                sgr => out.push_str(&format!("\x1b[{sgr}m{text}\x1b[0m")),
            }
        }
        out
    }
}

pub(super) fn header(world: &str, platform: &str) -> Line {
    Line::default()
        .push(Tone::Heading, "Building ")
        .push(Tone::Plain, world)
        .push(Tone::Detail, format!(" for {platform}"))
}

// A step's line. A running step shows its progress, and its spinner turns
// with `tick`; a closed one shows its summary and how long it took.
pub(super) fn row(row: &Row, now: Instant, tick: usize) -> Line {
    let label = format!("{:<LABEL_WIDTH$}", row.step.label());
    let line = Line::default().push(Tone::Plain, INDENT);
    match &row.state {
        RowState::Done { summary } => line
            .push(Tone::Done, "✓ ")
            .push(Tone::Label, label)
            .push(Tone::Plain, format!("{summary:<SUMMARY_WIDTH$}"))
            .push(Tone::Detail, format!(" {:>6}", duration(row.elapsed))),
        RowState::Failed { detail } => line
            .push(Tone::Fail, "✗ ")
            .push(Tone::Label, label)
            .push(Tone::Plain, format!("{detail:<SUMMARY_WIDTH$}"))
            .push(Tone::Detail, format!(" {:>6}", duration(row.elapsed))),
        RowState::Running => {
            let spinner = SPINNER[tick % SPINNER.len()];
            let mut line = line
                .push(Tone::Progress, format!("{spinner} "))
                .push(Tone::Label, label);
            if row.total > 0 {
                let (filled, rest) = bar(row.done, row.total, BAR_WIDTH);
                line = line
                    .push(Tone::Progress, filled)
                    .push(Tone::Detail, rest)
                    .push(
                        Tone::Plain,
                        format!("  {}/{}", group(row.done), group(row.total)),
                    );
            }
            let elapsed = now.saturating_duration_since(row.started);
            line = line.push(Tone::Detail, format!("  {}", duration(elapsed)));
            if let Some((item, age)) = row.current_item(now) {
                let item = shown_item(item);
                let item = if age >= SLOW_ITEM {
                    format!("  {item} ({})", duration(age))
                } else {
                    format!("  {item}")
                };
                line = line.push(Tone::Detail, item);
            }
            line
        }
    }
}

// The closing line: how long the build took, what it wrote, and how many
// problems it logged on the way.
pub(super) fn footer(board: &Board) -> Option<Line> {
    let (outcome, elapsed) = board.outcome.as_ref()?;
    let line = match outcome {
        Outcome::Built { written, data_dir } => Line::default()
            .push(Tone::Heading, "Finished")
            .push(Tone::Plain, format!(" in {}", duration(*elapsed)))
            .push(
                Tone::Detail,
                format!(" · {} written to {data_dir}", bytes(*written)),
            ),
        Outcome::Failed => Line::default()
            .push(Tone::Fail, "Failed")
            .push(Tone::Plain, format!(" after {}", duration(*elapsed))),
    };
    let problems = [(board.errors, "error"), (board.warnings, "warning")]
        .into_iter()
        .filter(|(n, _)| *n > 0)
        .map(|(n, noun)| counted(n as u32, noun))
        .collect::<Vec<_>>();
    if problems.is_empty() {
        return Some(line);
    }
    Some(
        line.push(Tone::Plain, " · ")
            .push(Tone::Warning, problems.join(", ")),
    )
}

// A note's lines: a level tag on the first, the rest of a multi-line message
// indented beneath it.
pub(super) fn note(note: &Note) -> Vec<Line> {
    let (tone, tag) = match note.level {
        NoteLevel::Error => (Tone::Fail, "error"),
        NoteLevel::Warning => (Tone::Warning, "warning"),
        NoteLevel::Info => (Tone::Detail, "info"),
    };
    let text_tone = match note.level {
        NoteLevel::Info => Tone::Detail,
        _ => Tone::Plain,
    };
    let mut lines = note.text.lines();
    let first = Line::default()
        .push(tone, tag)
        .push(text_tone, format!(": {}", lines.next().unwrap_or("")));
    std::iter::once(first)
        .chain(lines.map(|l| Line::default().push(text_tone, format!("    {l}"))))
        .collect()
}

// A source file is named by its file name; an asset by its id.
fn shown_item(item: &str) -> &str {
    item.rsplit(['/', '\\']).next().unwrap_or(item)
}

// A progress bar `width` cells wide, as its filled and remaining runs. The
// filled run ends in a half cell when the fraction lands mid-cell.
pub(super) fn bar(done: u32, total: u32, width: usize) -> (String, String) {
    let halves = match total {
        0 => 0,
        total => (u64::from(done.min(total)) * width as u64 * 2 / u64::from(total)) as usize,
    };
    let full = halves / 2;
    let half = halves % 2 == 1;
    let mut filled = "━".repeat(full);
    if half {
        filled.push('╸');
    }
    let rest = "─".repeat(width - full - usize::from(half));
    (filled, rest)
}

// `n` with its thousands grouped: 1234567 reads 1,234,567.
pub(super) fn group(n: u32) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

// `n` of `noun`, pluralized by a trailing s.
pub(super) fn counted(n: u32, noun: &str) -> String {
    plural(n, noun, &format!("{noun}s"))
}

// `n` of a noun whose plural is spelled out.
pub(super) fn plural(n: u32, one: &str, many: &str) -> String {
    match n {
        1 => format!("1 {one}"),
        n => format!("{} {many}", group(n)),
    }
}

pub(super) fn bytes(n: u64) -> String {
    const UNITS: [&str; 4] = ["KiB", "MiB", "GiB", "TiB"];
    if n < 1024 {
        return format!("{n} B");
    }
    let mut size = n as f64 / 1024.0;
    let mut unit = 0;
    while size >= 1024.0 && unit + 1 < UNITS.len() {
        size /= 1024.0;
        unit += 1;
    }
    format!("{size:.1} {}", UNITS[unit])
}

pub(super) fn duration(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 1.0 {
        format!("{}ms", d.as_millis())
    } else if secs < 60.0 {
        format!("{secs:.1}s")
    } else {
        format!("{}m {:02}s", d.as_secs() / 60, d.as_secs() % 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_status::board::Step;
    use concinnity_cook::{BuildProgress, BuildStage};

    #[test]
    fn numbers_group_by_thousands_and_count_their_noun() {
        assert_eq!(group(0), "0");
        assert_eq!(group(999), "999");
        assert_eq!(group(1_000), "1,000");
        assert_eq!(group(1_234_567), "1,234,567");
        assert_eq!(counted(1, "file"), "1 file");
        assert_eq!(counted(2_144, "asset"), "2,144 assets");
        assert_eq!(counted(0, "blob"), "0 blobs");
    }

    #[test]
    fn sizes_and_durations_read_at_a_glance() {
        assert_eq!(bytes(512), "512 B");
        assert_eq!(bytes(2048), "2.0 KiB");
        assert_eq!(bytes(458_784_260), "437.5 MiB");
        assert_eq!(duration(Duration::from_millis(42)), "42ms");
        assert_eq!(duration(Duration::from_millis(4_250)), "4.2s");
        assert_eq!(duration(Duration::from_secs(125)), "2m 05s");
    }

    #[test]
    fn a_bar_fills_by_half_cells_and_keeps_its_width() {
        assert_eq!(bar(0, 10, 4), (String::new(), "────".into()));
        assert_eq!(bar(5, 10, 4), ("━━".into(), "──".into()));
        assert_eq!(bar(3, 8, 4), ("━╸".into(), "──".into()));
        assert_eq!(bar(10, 10, 4), ("━━━━".into(), String::new()));
        assert_eq!(bar(12, 10, 4), ("━━━━".into(), String::new()));
    }

    #[test]
    fn a_fitted_line_ends_in_an_ellipsis_only_when_cut() {
        let line = Line::default()
            .push(Tone::Label, "abc")
            .push(Tone::Detail, "defgh");
        assert_eq!(line.clone().fit(8).plain_text(), "abcdefgh");
        assert_eq!(line.clone().fit(6).plain_text(), "abcde…");
        assert_eq!(line.fit(2).plain_text(), "a…");
    }

    #[test]
    fn painting_wraps_each_styled_run_and_leaves_plain_runs_bare() {
        let line = Line::default().push(Tone::Plain, "a").push(Tone::Done, "b");
        assert_eq!(line.paint(false), "ab");
        assert_eq!(line.paint(true), "a\x1b[32mb\x1b[0m");
    }

    #[test]
    fn a_running_row_shows_its_bar_count_and_slowest_item() {
        let t0 = Instant::now();
        let mut board = Board::new(t0);
        let import = BuildStage::Import;
        board.begin(Step::Cook(import), 4, t0);
        board.apply(
            BuildProgress::ItemStarted {
                stage: import,
                item: "assets/city/City.fbx",
            },
            t0,
        );
        let now = t0 + Duration::from_secs(3);
        let text = row(&board.rows[0], now, 0).plain_text();
        assert_eq!(
            text,
            format!(
                "  ⠋ Import      {}  0/4  3.0s  City.fbx (3.0s)",
                "─".repeat(BAR_WIDTH)
            )
        );
    }

    #[test]
    fn a_closed_row_aligns_its_summary_and_time() {
        let t0 = Instant::now();
        let mut board = Board::new(t0);
        board.begin(Step::Load, 0, t0);
        board.close(Some("12 assets".into()), t0 + Duration::from_millis(30));
        let text = row(&board.rows[0], t0, 0).plain_text();
        assert_eq!(
            text,
            format!("  ✓ Load        {:<SUMMARY_WIDTH$}   30ms", "12 assets")
        );
    }

    #[test]
    fn the_footer_names_the_output_and_any_problems() {
        let t0 = Instant::now();
        let mut board = Board::new(t0);
        assert_eq!(footer(&board), None);
        board.note(NoteLevel::Warning, "w".into());
        board.end(
            Outcome::Built {
                written: 3 * 1024 * 1024,
                data_dir: ".concinnity/data".into(),
            },
            None,
            t0 + Duration::from_millis(1_500),
        );
        assert_eq!(
            footer(&board).unwrap().plain_text(),
            "Finished in 1.5s · 3.0 MiB written to .concinnity/data · 1 warning"
        );
    }

    #[test]
    fn a_multi_line_note_indents_its_continuation() {
        let lines = note(&Note {
            level: NoteLevel::Warning,
            text: "shader 'sky': compiling 'main':\nline 3: unused".into(),
        });
        let text: Vec<String> = lines.iter().map(Line::plain_text).collect();
        assert_eq!(
            text,
            [
                "warning: shader 'sky': compiling 'main':",
                "    line 3: unused"
            ]
        );
    }
}
