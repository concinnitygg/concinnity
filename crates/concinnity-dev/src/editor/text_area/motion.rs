//! Caret motion over a buffer that needs no view: character and word steps,
//! line starts, and word extents. Pure functions of the text.

use super::buffer::{Buffer, Pos};

// What a word step treats as one run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    Space,
    Word,
    Punct,
}

fn class(c: char) -> Class {
    if c.is_whitespace() {
        Class::Space
    } else if c.is_alphanumeric() || c == '_' {
        Class::Word
    } else {
        Class::Punct
    }
}

// One character left, onto the previous line's end from column 0.
pub(crate) fn left(b: &Buffer, p: Pos) -> Pos {
    if p.col > 0 {
        Pos::new(p.line, p.col - 1)
    } else if p.line > 0 {
        Pos::new(p.line - 1, b.line_len(p.line - 1))
    } else {
        p
    }
}

// One character right, onto the next line's start from a line's end.
pub(crate) fn right(b: &Buffer, p: Pos) -> Pos {
    if p.col < b.line_len(p.line) {
        Pos::new(p.line, p.col + 1)
    } else if p.line + 1 < b.line_count() {
        Pos::new(p.line + 1, 0)
    } else {
        p
    }
}

// The start of the word before `p`: skip whitespace, then one run of a class.
// At a line's start it steps onto the previous line's end.
pub(crate) fn word_left(b: &Buffer, p: Pos) -> Pos {
    if p.col == 0 {
        return left(b, p);
    }
    let chars: Vec<char> = b.line(p.line).chars().collect();
    let mut col = p.col;
    while col > 0 && class(chars[col - 1]) == Class::Space {
        col -= 1;
    }
    if col > 0 {
        let run = class(chars[col - 1]);
        while col > 0 && class(chars[col - 1]) == run {
            col -= 1;
        }
    }
    Pos::new(p.line, col)
}

// The end of the word after `p`: skip whitespace, then one run of a class.
// At a line's end it steps onto the next line's start.
pub(crate) fn word_right(b: &Buffer, p: Pos) -> Pos {
    let chars: Vec<char> = b.line(p.line).chars().collect();
    if p.col >= chars.len() {
        return right(b, p);
    }
    let mut col = p.col;
    while col < chars.len() && class(chars[col]) == Class::Space {
        col += 1;
    }
    if col < chars.len() {
        let run = class(chars[col]);
        while col < chars.len() && class(chars[col]) == run {
            col += 1;
        }
    }
    Pos::new(p.line, col)
}

// The column of a line's first non-whitespace character (its end if blank).
pub(crate) fn indent_end(line: &str) -> usize {
    line.chars().take_while(|c| c.is_whitespace()).count()
}

// Home: the first non-whitespace character, or column 0 when already there
// (or inside the indentation), so a second press reaches the true start.
pub(crate) fn smart_home(b: &Buffer, p: Pos) -> Pos {
    let indent = indent_end(b.line(p.line));
    let col = if p.col == indent { 0 } else { indent };
    Pos::new(p.line, col)
}

pub(crate) fn line_end(b: &Buffer, p: Pos) -> Pos {
    Pos::new(p.line, b.line_len(p.line))
}

// The run of one character class around `p` (the word a double-click takes).
// A position between two runs takes the one after it, or the one before at a
// line's end.
pub(crate) fn word_at(b: &Buffer, p: Pos) -> (Pos, Pos) {
    let chars: Vec<char> = b.line(p.line).chars().collect();
    if chars.is_empty() {
        return (p, p);
    }
    let at = p.col.min(chars.len() - 1);
    let run = class(chars[at]);
    let mut start = at;
    while start > 0 && class(chars[start - 1]) == run {
        start -= 1;
    }
    let mut end = at + 1;
    while end < chars.len() && class(chars[end]) == run {
        end += 1;
    }
    (Pos::new(p.line, start), Pos::new(p.line, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(text: &str) -> Buffer {
        Buffer::from_text(text)
    }

    #[test]
    fn left_and_right_cross_line_breaks() {
        let t = b("ab\ncd");
        assert_eq!(left(&t, Pos::new(1, 0)), Pos::new(0, 2));
        assert_eq!(right(&t, Pos::new(0, 2)), Pos::new(1, 0));
        assert_eq!(
            left(&t, Pos::new(0, 0)),
            Pos::new(0, 0),
            "stops at the start"
        );
        assert_eq!(
            right(&t, Pos::new(1, 2)),
            Pos::new(1, 2),
            "stops at the end"
        );
    }

    #[test]
    fn word_left_skips_space_then_one_run() {
        let t = b("let foo_bar = x.y;");
        assert_eq!(
            word_left(&t, Pos::new(0, 18)),
            Pos::new(0, 17),
            "the ';' run"
        );
        assert_eq!(
            word_left(&t, Pos::new(0, 12)),
            Pos::new(0, 4),
            "space then word"
        );
        assert_eq!(word_left(&t, Pos::new(0, 4)), Pos::new(0, 0));
    }

    #[test]
    fn word_right_skips_space_then_one_run() {
        let t = b("let foo_bar = x.y;");
        assert_eq!(word_right(&t, Pos::new(0, 0)), Pos::new(0, 3));
        assert_eq!(word_right(&t, Pos::new(0, 3)), Pos::new(0, 11));
        assert_eq!(
            word_right(&t, Pos::new(0, 11)),
            Pos::new(0, 13),
            "the '=' run"
        );
    }

    #[test]
    fn word_steps_cross_line_breaks() {
        let t = b("ab\n  cd");
        assert_eq!(word_right(&t, Pos::new(0, 2)), Pos::new(1, 0));
        assert_eq!(word_left(&t, Pos::new(1, 0)), Pos::new(0, 2));
    }

    #[test]
    fn smart_home_toggles_between_indent_and_column_zero() {
        let t = b("    code");
        assert_eq!(smart_home(&t, Pos::new(0, 6)), Pos::new(0, 4));
        assert_eq!(smart_home(&t, Pos::new(0, 4)), Pos::new(0, 0));
        assert_eq!(smart_home(&t, Pos::new(0, 0)), Pos::new(0, 4));
        assert_eq!(line_end(&t, Pos::new(0, 1)), Pos::new(0, 8));
    }

    #[test]
    fn word_at_takes_the_run_under_a_position() {
        let t = b("foo(bar_1, x)");
        assert_eq!(
            word_at(&t, Pos::new(0, 5)),
            (Pos::new(0, 4), Pos::new(0, 9))
        );
        assert_eq!(
            word_at(&t, Pos::new(0, 3)),
            (Pos::new(0, 3), Pos::new(0, 4))
        );
        assert_eq!(
            word_at(&t, Pos::new(0, 13)),
            (Pos::new(0, 12), Pos::new(0, 13))
        );
        let blank = b("");
        assert_eq!(
            word_at(&blank, Pos::new(0, 0)),
            (Pos::new(0, 0), Pos::new(0, 0))
        );
    }
}
