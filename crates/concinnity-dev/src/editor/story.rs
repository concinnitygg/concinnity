// src/editor/story.rs
//
// The data half of the Story panel: a plain line-based text model over the
// Markdown source file a `StoryImport` entry references. The panel edits one
// line at a time through a real `TextInput` (the engine's existing single-line
// primitive), so multiline editing needs no new runtime asset: these helpers
// own the line structure (split / join / navigation bounds) and the panel and
// hook stay thin. Pure and world-free.

// The story text as editable lines. Always at least one (possibly empty) line,
// so the panel has a current line to edit even for a new / empty file.
pub(crate) fn lines_of(content: &str) -> Vec<String> {
    let mut lines: Vec<String> = content.lines().map(String::from).collect();
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

// The lines re-joined for writing: newline-separated with a trailing newline,
// the canonical text-file shape (`lines()` on the result round-trips).
pub(crate) fn join_lines(lines: &[String]) -> String {
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

// Split line `i` at `caret` (a character index, clamped): the text after the
// caret moves to a new line below. Returns the new current line index (the
// tail line, caret at its start).
pub(crate) fn split_line(lines: &mut Vec<String>, i: usize, caret: usize) -> usize {
    let i = i.min(lines.len().saturating_sub(1));
    let line = std::mem::take(&mut lines[i]);
    let byte = line
        .char_indices()
        .nth(caret)
        .map(|(b, _)| b)
        .unwrap_or(line.len());
    let (head, tail) = line.split_at(byte);
    lines[i] = head.to_string();
    lines.insert(i + 1, tail.to_string());
    i + 1
}

// Join line `i` onto the previous line (Backspace at column 0). Returns the
// new current line index and the caret sitting at the join point, or `None`
// when there is no previous line to join.
pub(crate) fn join_with_previous(lines: &mut Vec<String>, i: usize) -> Option<(usize, usize)> {
    if i == 0 || i >= lines.len() {
        return None;
    }
    let tail = lines.remove(i);
    let caret = lines[i - 1].chars().count();
    lines[i - 1].push_str(&tail);
    Some((i - 1, caret))
}

// The starter story a "+ Create story" writes: minimal but exercising the
// format's main constructs (frontmatter with a declared character, a node
// heading, an attributed line, plain narration). Pinned parseable by test.
pub(crate) const STARTER_STORY: &str = "---
title: Untitled Story
characters:
  narrator: Narrator
---

# start

**narrator:** Once upon a time...

The story begins here.
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_round_trip_through_join() {
        let content = "a\nb\n\nc\n";
        let lines = lines_of(content);
        assert_eq!(lines, ["a", "b", "", "c"]);
        assert_eq!(join_lines(&lines), content);
    }

    #[test]
    fn empty_content_still_yields_an_editable_line() {
        assert_eq!(lines_of(""), [""]);
        assert_eq!(join_lines(&lines_of("")), "\n");
    }

    #[test]
    fn split_line_moves_the_tail_below_the_caret() {
        let mut lines = vec!["hello world".to_string()];
        let cur = split_line(&mut lines, 0, 5);
        assert_eq!(cur, 1);
        assert_eq!(lines, ["hello", " world"]);
        // Caret past the end splits into an empty new line.
        let cur = split_line(&mut lines, 1, 99);
        assert_eq!(cur, 2);
        assert_eq!(lines, ["hello", " world", ""]);
    }

    #[test]
    fn split_line_is_utf8_safe() {
        let mut lines = vec!["héllo".to_string()];
        split_line(&mut lines, 0, 2);
        assert_eq!(lines, ["hé", "llo"]);
    }

    #[test]
    fn join_with_previous_lands_the_caret_at_the_seam() {
        let mut lines = vec!["hé".to_string(), "llo".to_string()];
        let (cur, caret) = join_with_previous(&mut lines, 1).unwrap();
        assert_eq!((cur, caret), (0, 2), "caret in characters, not bytes");
        assert_eq!(lines, ["héllo"]);
        assert_eq!(join_with_previous(&mut lines, 0), None, "no previous line");
    }

    // The starter story must parse with the real story pipeline; a format
    // change breaks this test instead of shipping a broken template.
    #[test]
    fn starter_story_parses() {
        concinnity_cook::build_only::validate_story_source(STARTER_STORY)
            .expect("the starter story template parses");
    }
}
