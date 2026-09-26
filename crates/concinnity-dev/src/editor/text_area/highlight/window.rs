//! A line's spans as color runs over what its row label shows: the cells
//! `[left, left + width)` of the line, one label character per cell (a tab
//! draws as the spaces it expands to).

use concinnity_core::components::ColorRun;

use super::Span;
use crate::editor::text_area::view;

// Append the runs that color `spans` of `line` within the window to `out`,
// clipped to it and counted from its first cell.
pub(crate) fn window_runs(
    line: &str,
    spans: &[Span],
    left: usize,
    width: usize,
    out: &mut Vec<ColorRun>,
) {
    if spans.is_empty() {
        return;
    }
    // The cell each character column starts in, and the line's end.
    let mut cells = Vec::with_capacity(line.len() + 1);
    let mut cell = 0;
    for c in line.chars() {
        cells.push(cell);
        cell = view::advance(cell, c);
    }
    cells.push(cell);
    let cell_of = |col: usize| cells[col.min(cells.len() - 1)];
    let right = left + width;
    for span in spans {
        let a = cell_of(span.start).max(left);
        let b = cell_of(span.start + span.len).min(right);
        if b > a {
            out.push(ColorRun {
                start: (a - left) as u32,
                length: (b - a) as u32,
                color: span.token.color(),
            });
        }
    }
}
