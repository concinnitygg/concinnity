//! Where a build's status lands: redrawn in place on a terminal, or appended
//! line by line to a pipe or log, where a redraw would only be noise.

use std::fmt::Write as _;
use std::io::IsTerminal;
use std::time::Instant;

use super::board::Board;
use super::render;

// Begin and end a synchronized update, so a terminal that supports it paints
// each frame whole. Terminals without it ignore both.
const SYNC_BEGIN: &str = "\x1b[?2026h";
const SYNC_END: &str = "\x1b[?2026l";
// Assumed when the terminal will not say how wide it is.
const FALLBACK_WIDTH: usize = 100;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Surface {
    // A terminal: the rows are redrawn in place, `drawn` lines tall.
    Live { drawn: usize },
    // A pipe or log: each row is written once it closes, `printed` so far.
    Stream { printed: usize },
}

impl Surface {
    // Live on a terminal that takes escape codes, streamed anywhere else.
    pub(super) fn for_stderr() -> Self {
        let stderr = std::io::stderr();
        // On Windows the console only reads escape codes once asked to.
        let escapes = anstyle_query::windows::enable_ansi_colors().unwrap_or(true);
        if stderr.is_terminal() && escapes {
            Surface::Live { drawn: 0 }
        } else {
            Surface::Stream { printed: 0 }
        }
    }

    pub(super) fn colored(&self) -> bool {
        matches!(self, Surface::Live { .. })
    }

    // The text that brings the output up to date with `board`: the notes
    // that arrived since the last frame, then the rows. `last` is the closing
    // frame, which adds the footer.
    pub(super) fn frame(
        &mut self,
        board: &mut Board,
        now: Instant,
        tick: usize,
        width: usize,
        last: bool,
    ) -> String {
        let color = self.colored();
        let mut out = String::new();
        let notes = board.take_notes();
        let footer = if last { render::footer(board) } else { None };
        match self {
            Surface::Live { drawn } => {
                out.push_str(SYNC_BEGIN);
                if *drawn > 0 {
                    let _ = write!(out, "\x1b[{drawn}A");
                }
                out.push_str("\x1b[J");
                for line in notes.iter().flat_map(render::note) {
                    let _ = writeln!(out, "{}", line.paint(color));
                }
                // One column spare, so no line lands on the terminal's last
                // column and wraps.
                let fit = width.saturating_sub(1);
                for row in &board.rows {
                    let line = render::row(row, now, tick).fit(fit);
                    let _ = writeln!(out, "{}", line.paint(color));
                }
                *drawn = board.rows.len();
                if let Some(footer) = footer {
                    let _ = writeln!(out, "{}", footer.fit(fit).paint(color));
                }
                out.push_str(SYNC_END);
            }
            Surface::Stream { printed } => {
                for line in notes.iter().flat_map(render::note) {
                    let _ = writeln!(out, "{}", line.paint(color));
                }
                let closed = board.rows[*printed..]
                    .iter()
                    .take_while(|r| !r.is_running());
                for row in closed {
                    let _ = writeln!(out, "{}", render::row(row, now, tick).paint(color));
                    *printed += 1;
                }
                if let Some(footer) = footer {
                    let _ = writeln!(out, "{}", footer.paint(color));
                }
            }
        }
        out
    }
}

pub(super) fn width() -> usize {
    terminal_size::terminal_size_of(std::io::stderr())
        .map(|(w, _)| usize::from(w.0))
        .unwrap_or(FALLBACK_WIDTH)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_status::board::{NoteLevel, Outcome, Step};
    use concinnity_cook::BuildStage;
    use std::time::Duration;

    fn built() -> Outcome {
        Outcome::Built {
            written: 2048,
            data_dir: "data".into(),
        }
    }

    #[test]
    fn a_stream_writes_each_row_once_it_closes() {
        let t0 = Instant::now();
        let mut board = Board::new(t0);
        let mut surface = Surface::Stream { printed: 0 };
        board.begin(Step::Load, 0, t0);
        assert_eq!(surface.frame(&mut board, t0, 0, 80, false), "");

        board.begin(Step::Cook(BuildStage::Compile), 2, t0);
        board.note(NoteLevel::Warning, "careful".into());
        let frame = surface.frame(&mut board, t0, 0, 80, false);
        let lines: Vec<&str> = frame.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "warning: careful");
        assert!(lines[1].starts_with("  ✓ Load"), "{}", lines[1]);
        assert_eq!(surface.frame(&mut board, t0, 0, 80, false), "");

        board.end(built(), None, t0 + Duration::from_secs(2));
        let frame = surface.frame(&mut board, t0, 0, 80, true);
        let lines: Vec<&str> = frame.lines().collect();
        assert!(lines[0].starts_with("  ✓ Compile"), "{}", lines[0]);
        assert!(lines[1].starts_with("Finished in 2.0s"), "{}", lines[1]);
        assert!(!frame.contains('\x1b'), "a stream carries no escape codes");
    }

    #[test]
    fn a_live_frame_redraws_over_the_rows_it_drew_last() {
        let t0 = Instant::now();
        let mut board = Board::new(t0);
        let mut surface = Surface::Live { drawn: 0 };
        board.begin(Step::Load, 0, t0);
        let first = surface.frame(&mut board, t0, 0, 80, false);
        assert!(!first.contains("\x1b[1A"), "nothing to move over yet");
        assert_eq!(surface, Surface::Live { drawn: 1 });

        board.begin(Step::Cook(BuildStage::Compile), 2, t0);
        let second = surface.frame(&mut board, t0, 1, 80, false);
        assert!(second.starts_with(&format!("{SYNC_BEGIN}\x1b[1A\x1b[J")));
        assert!(second.ends_with(SYNC_END));
        assert_eq!(surface, Surface::Live { drawn: 2 });
    }

    #[test]
    fn a_live_frame_keeps_every_row_inside_the_width() {
        let t0 = Instant::now();
        let mut board = Board::new(t0);
        let mut surface = Surface::Live { drawn: 0 };
        board.begin(Step::Cook(BuildStage::Compile), 10, t0);
        let frame = surface.frame(&mut board, t0, 0, 30, false);
        let body = frame
            .trim_start_matches(SYNC_BEGIN)
            .trim_start_matches("\x1b[J")
            .trim_end_matches(SYNC_END);
        for line in body.lines() {
            let visible = strip_escapes(line);
            assert!(visible.chars().count() <= 29, "{visible:?}");
        }
    }

    fn strip_escapes(text: &str) -> String {
        let mut out = String::new();
        let mut chars = text.chars();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                for c in chars.by_ref() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }
}
