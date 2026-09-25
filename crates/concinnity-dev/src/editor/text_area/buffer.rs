//! The text as a list of lines, addressed by (line, character column). Columns
//! count `char`s, never bytes, so a multi-byte character is one step for the
//! caret and no edit can split one.

// A place in the text, between two characters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub(crate) struct Pos {
    pub(crate) line: usize,
    pub(crate) col: usize,
}

impl Pos {
    pub(crate) const fn new(line: usize, col: usize) -> Self {
        Self { line, col }
    }
}

// The line break a loaded text used, restored when it is written back out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum LineEnding {
    #[default]
    Lf,
    CrLf,
}

// Never empty: an empty text is one empty line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Buffer {
    lines: Vec<String>,
    ending: LineEnding,
}

impl Default for Buffer {
    fn default() -> Self {
        Self::from_text("")
    }
}

// `text` with every CRLF and lone CR turned into LF.
pub(crate) fn normalize_newlines(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

// The byte offset of character column `col` in `s`, or its length past the end.
fn byte_of(s: &str, col: usize) -> usize {
    s.char_indices().nth(col).map_or(s.len(), |(b, _)| b)
}

// Where the caret lands after inserting `text` at `at`.
pub(crate) fn end_of_insert(at: Pos, text: &str) -> Pos {
    match text.rfind('\n') {
        None => Pos::new(at.line, at.col + text.chars().count()),
        Some(last) => Pos::new(
            at.line + text.matches('\n').count(),
            text[last + 1..].chars().count(),
        ),
    }
}

impl Buffer {
    // Load `text`, keeping its line ending for `text()` to write back. A text
    // that uses CRLF anywhere is treated as CRLF throughout.
    pub(crate) fn from_text(text: &str) -> Self {
        let ending = if text.contains("\r\n") {
            LineEnding::CrLf
        } else {
            LineEnding::Lf
        };
        let lines = normalize_newlines(text)
            .split('\n')
            .map(String::from)
            .collect();
        Self { lines, ending }
    }

    // The whole text in its original line ending.
    pub(crate) fn text(&self) -> String {
        let sep = match self.ending {
            LineEnding::Lf => "\n",
            LineEnding::CrLf => "\r\n",
        };
        self.lines.join(sep)
    }

    pub(crate) fn line_count(&self) -> usize {
        self.lines.len()
    }

    pub(crate) fn line(&self, i: usize) -> &str {
        self.lines.get(i).map_or("", String::as_str)
    }

    pub(crate) fn line_len(&self, i: usize) -> usize {
        self.line(i).chars().count()
    }

    pub(crate) fn end(&self) -> Pos {
        let last = self.lines.len() - 1;
        Pos::new(last, self.line_len(last))
    }

    // `p` pulled onto the text: the last line at most, the line's end at most.
    pub(crate) fn clamp(&self, p: Pos) -> Pos {
        let line = p.line.min(self.lines.len() - 1);
        Pos::new(line, p.col.min(self.line_len(line)))
    }

    // Insert LF-separated `text` at `at`, returning the position after it.
    pub(crate) fn insert(&mut self, at: Pos, text: &str) -> Pos {
        let at = self.clamp(at);
        let line = &mut self.lines[at.line];
        let tail = line.split_off(byte_of(line, at.col));
        let mut parts = text.split('\n');
        line.push_str(parts.next().unwrap_or(""));
        let mut row = at.line;
        for part in parts {
            row += 1;
            self.lines.insert(row, part.to_string());
        }
        self.lines[row].push_str(&tail);
        end_of_insert(at, text)
    }

    // The LF-separated text between two positions, in either order.
    pub(crate) fn slice(&self, a: Pos, b: Pos) -> String {
        let (start, end) = ordered(self.clamp(a), self.clamp(b));
        if start.line == end.line {
            let line = self.line(start.line);
            return line[byte_of(line, start.col)..byte_of(line, end.col)].to_string();
        }
        let first = self.line(start.line);
        let mut out = first[byte_of(first, start.col)..].to_string();
        for i in start.line + 1..end.line {
            out.push('\n');
            out.push_str(self.line(i));
        }
        let last = self.line(end.line);
        out.push('\n');
        out.push_str(&last[..byte_of(last, end.col)]);
        out
    }

    // Remove the text between two positions (in either order), returning it.
    pub(crate) fn remove(&mut self, a: Pos, b: Pos) -> String {
        let (start, end) = ordered(self.clamp(a), self.clamp(b));
        let removed = self.slice(start, end);
        let last = self.line(end.line);
        let tail = last[byte_of(last, end.col)..].to_string();
        self.lines.drain(start.line + 1..=end.line);
        let first = &mut self.lines[start.line];
        first.truncate(byte_of(first, start.col));
        first.push_str(&tail);
        removed
    }
}

// Two positions, earlier first.
pub(crate) fn ordered(a: Pos, b: Pos) -> (Pos, Pos) {
    if a <= b { (a, b) } else { (b, a) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(b: &Buffer) -> Vec<&str> {
        (0..b.line_count()).map(|i| b.line(i)).collect()
    }

    #[test]
    fn text_round_trips_exactly() {
        for text in ["", "a", "a\nb", "a\nb\n", "\n\n", "héllo\nwörld\n"] {
            assert_eq!(Buffer::from_text(text).text(), text, "{text:?}");
        }
    }

    #[test]
    fn crlf_loads_as_lines_and_writes_back_as_crlf() {
        let b = Buffer::from_text("a\r\nb\r\n");
        assert_eq!(lines(&b), ["a", "b", ""]);
        assert_eq!(b.text(), "a\r\nb\r\n");
    }

    #[test]
    fn normalize_newlines_handles_crlf_and_lone_cr() {
        assert_eq!(normalize_newlines("a\r\nb\rc\n"), "a\nb\nc\n");
    }

    #[test]
    fn empty_text_is_one_empty_line() {
        let b = Buffer::from_text("");
        assert_eq!(b.line_count(), 1);
        assert_eq!(b.end(), Pos::new(0, 0));
    }

    #[test]
    fn clamp_pulls_a_position_onto_the_text() {
        let b = Buffer::from_text("abc\nde");
        assert_eq!(b.clamp(Pos::new(0, 9)), Pos::new(0, 3));
        assert_eq!(b.clamp(Pos::new(7, 1)), Pos::new(1, 1));
    }

    #[test]
    fn insert_within_a_line() {
        let mut b = Buffer::from_text("ac");
        assert_eq!(b.insert(Pos::new(0, 1), "b"), Pos::new(0, 2));
        assert_eq!(b.text(), "abc");
    }

    #[test]
    fn insert_multiple_lines_splits_around_the_caret() {
        let mut b = Buffer::from_text("head tail");
        let end = b.insert(Pos::new(0, 4), "1\n22\n333");
        assert_eq!(lines(&b), ["head1", "22", "333 tail"]);
        assert_eq!(end, Pos::new(2, 3));
    }

    #[test]
    fn insert_is_char_indexed() {
        let mut b = Buffer::from_text("héllo");
        b.insert(Pos::new(0, 2), "-");
        assert_eq!(b.text(), "hé-llo");
    }

    #[test]
    fn remove_across_lines_joins_the_ends() {
        let mut b = Buffer::from_text("one\ntwo\nthree");
        let removed = b.remove(Pos::new(2, 2), Pos::new(0, 1));
        assert_eq!(removed, "ne\ntwo\nth");
        assert_eq!(b.text(), "oree");
    }

    #[test]
    fn remove_a_line_break() {
        let mut b = Buffer::from_text("a\nb");
        assert_eq!(b.remove(Pos::new(0, 1), Pos::new(1, 0)), "\n");
        assert_eq!(b.text(), "ab");
    }

    #[test]
    fn slice_and_insert_are_inverse() {
        let mut b = Buffer::from_text("alpha\nbeta\ngamma");
        let (s, e) = (Pos::new(0, 2), Pos::new(2, 3));
        let cut = b.remove(s, e);
        b.insert(s, &cut);
        assert_eq!(b.text(), "alpha\nbeta\ngamma");
    }

    #[test]
    fn end_of_insert_counts_lines_and_chars() {
        assert_eq!(end_of_insert(Pos::new(3, 2), "xy"), Pos::new(3, 4));
        assert_eq!(end_of_insert(Pos::new(3, 2), "x\nyé"), Pos::new(4, 2));
        assert_eq!(end_of_insert(Pos::new(3, 2), "\n"), Pos::new(4, 0));
    }
}
