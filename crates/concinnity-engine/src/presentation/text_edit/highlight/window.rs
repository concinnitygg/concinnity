//! A line's spans as color runs over what its row label shows: the cells
//! `[left, left + width)` of the line, one label character per cell (a tab
//! draws as the spaces it expands to).

use alloc::vec::Vec;

use crate::components::ColorRun;
use concinnity_core::math::SaturatingFrom;

use super::super::view;
use super::Span;

/// Append the runs that color `spans` of `line` within the window to `out`,
/// clipped to it and counted from its first cell. The spans are in order and
/// never overlap, so one walk along the line finds every span's cells.
pub(crate) fn window_runs(
    line: &str,
    spans: &[Span],
    left: usize,
    width: usize,
    out: &mut Vec<ColorRun>,
) {
    let right = left + width;
    let mut chars = line.chars();
    let (mut col, mut cell) = (0usize, 0usize);
    // The cell character column `target` starts in, walking forward from the
    // last one asked for; past the line's end it is the end.
    let mut cell_at = |target: usize| {
        while col < target {
            let Some(c) = chars.next() else {
                break;
            };
            cell = view::advance(cell, c);
            col += 1;
        }
        cell
    };
    for span in spans {
        let a = cell_at(span.start).max(left);
        let b = cell_at(span.start + span.len).min(right);
        if b > a {
            out.push(ColorRun {
                start: u32::saturating_from(a - left),
                length: u32::saturating_from(b - a),
                color: span.token.color(),
            });
        }
    }
}
