//! The text as it appears in a grid of monospace cells: tabs expand to the next
//! tab stop, so a character column and the cell it draws in can differ. The
//! scroll window over that grid lives here too.

pub(crate) const TAB_WIDTH: usize = 4;

// Columns kept clear between the caret and the window's left / right edge when
// a horizontal scroll brings it back into view.
const H_MARGIN: usize = 4;

// The cell a character column starts in.
pub(crate) fn visual_col(line: &str, col: usize) -> usize {
    line.chars().take(col).fold(0, advance)
}

// The cell after character `c` drawn starting at cell `v`.
pub(crate) fn advance(v: usize, c: char) -> usize {
    if c == '\t' {
        (v / TAB_WIDTH + 1) * TAB_WIDTH
    } else {
        v + 1
    }
}

// How many cells a whole line takes.
pub(crate) fn line_width(line: &str) -> usize {
    line.chars().fold(0, advance)
}

// The character column nearest the (fractional) cell position `v`: a point on
// the right half of a character lands after it.
pub(crate) fn col_at_visual(line: &str, v: f32) -> usize {
    let mut cell = 0;
    for (col, c) in line.chars().enumerate() {
        let next = advance(cell, c);
        if v < (cell + next) as f32 * 0.5 {
            return col;
        }
        cell = next;
    }
    line.chars().count()
}

// The cells `[left, left + width)` of a line as drawable text: tabs become
// spaces, and a character the code face cannot draw becomes '?' so the cells
// after it stay aligned.
pub(crate) fn display_slice(line: &str, left: usize, width: usize) -> String {
    let mut out = String::new();
    let mut cell = 0;
    for c in line.chars() {
        let next = advance(cell, c);
        let glyph = if c == '\t' || c.is_ascii_graphic() || c == ' ' {
            if c == '\t' { ' ' } else { c }
        } else {
            '?'
        };
        for v in cell..next {
            if v >= left + width {
                return out;
            }
            if v >= left {
                out.push(glyph);
            }
        }
        cell = next;
    }
    out
}

// The window over the grid: its first line and first cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Scroll {
    pub(crate) top: usize,
    pub(crate) left: usize,
}

// How much of the grid the window shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ViewSize {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl Default for ViewSize {
    fn default() -> Self {
        Self { rows: 1, cols: 1 }
    }
}

impl Scroll {
    // Scroll the least distance that shows cell `vcol` of `line`.
    pub(crate) fn reveal(&mut self, line: usize, vcol: usize, view: ViewSize) {
        let rows = view.rows.max(1);
        if line < self.top {
            self.top = line;
        } else if line >= self.top + rows {
            self.top = line + 1 - rows;
        }
        let cols = view.cols.max(1);
        let margin = H_MARGIN.min(cols / 4);
        if vcol < self.left + margin {
            self.left = vcol.saturating_sub(margin);
        } else if vcol + margin >= self.left + cols {
            self.left = vcol + margin + 1 - cols;
        }
    }

    // Keep the window over the text: never past the last line or the widest
    // line (plus a cell for the caret after it).
    pub(crate) fn clamp(&mut self, line_count: usize, widest: usize, view: ViewSize) {
        self.top = self.top.min(line_count.saturating_sub(view.rows.max(1)));
        self.left = self.left.min((widest + 1).saturating_sub(view.cols.max(1)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tabs_expand_to_the_next_stop() {
        assert_eq!(visual_col("\tx", 1), 4);
        assert_eq!(visual_col("ab\tx", 3), 4);
        assert_eq!(visual_col("abcd\tx", 5), 8);
        assert_eq!(line_width("a\tb"), 5);
        assert_eq!(visual_col("héllo", 2), 2, "one cell per character");
    }

    #[test]
    fn col_at_visual_rounds_to_the_nearer_boundary() {
        assert_eq!(col_at_visual("abc", 0.4), 0);
        assert_eq!(col_at_visual("abc", 0.6), 1);
        assert_eq!(
            col_at_visual("abc", 9.0),
            3,
            "past the end lands at the end"
        );
        // A tab spans cells 0..4: its left half maps before it, its right half after.
        assert_eq!(col_at_visual("\tx", 1.5), 0);
        assert_eq!(col_at_visual("\tx", 2.5), 1);
        assert_eq!(col_at_visual("", 3.0), 0);
    }

    #[test]
    fn display_slice_windows_cells_and_expands_tabs() {
        assert_eq!(display_slice("abcdef", 2, 3), "cde");
        assert_eq!(display_slice("\tx", 0, 8), "    x");
        assert_eq!(display_slice("\tx", 2, 8), "  x", "a tab cut by the window");
        assert_eq!(
            display_slice("aé!", 0, 8),
            "a?!",
            "undrawable characters keep their cell"
        );
        assert_eq!(display_slice("abc", 5, 4), "");
    }

    #[test]
    fn reveal_scrolls_the_least_distance() {
        let view = ViewSize { rows: 10, cols: 40 };
        let mut s = Scroll::default();
        s.reveal(25, 0, view);
        assert_eq!(s.top, 16, "the line lands on the last row");
        s.reveal(3, 0, view);
        assert_eq!(s.top, 3, "the line lands on the first row");
        s.reveal(5, 0, view);
        assert_eq!(s.top, 3, "a visible line does not scroll");
    }

    #[test]
    fn reveal_keeps_a_margin_horizontally() {
        let view = ViewSize { rows: 10, cols: 40 };
        let mut s = Scroll::default();
        s.reveal(0, 50, view);
        assert_eq!(s.left, 50 + 4 + 1 - 40);
        s.reveal(0, 12, view);
        assert_eq!(s.left, 8);
        s.reveal(0, 2, view);
        assert_eq!(s.left, 0);
    }

    #[test]
    fn clamp_keeps_the_window_over_the_text() {
        let view = ViewSize { rows: 10, cols: 40 };
        let mut s = Scroll {
            top: 100,
            left: 100,
        };
        s.clamp(25, 60, view);
        assert_eq!(s, Scroll { top: 15, left: 21 });
        s.clamp(3, 10, view);
        assert_eq!(s, Scroll::default(), "a short text never scrolls");
    }
}
