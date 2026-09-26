//! The lines a label draws: its content broken to `wrap_width` and capped at
//! `max_lines`, and, for a label with color runs, which authored character
//! each drawn one came from.

use super::{LoadedFont, advance_px};
use crate::components::TextLabel;
use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;
use core::ops::Range;

const ELLIPSIS: &str = "...";

/// A label's drawn text.
pub(super) struct LaidOut<'a> {
    pub(super) text: Cow<'a, str>,
    /// The authored character index each drawn character came from, for a label
    /// with color runs whose drawn text differs from its content. Empty when
    /// the two line up one to one.
    pub(super) sources: Vec<u32>,
}

impl LaidOut<'_> {
    /// The authored character the `i`-th drawn character came from.
    pub(super) fn source(&self, i: usize) -> usize {
        self.sources.get(i).map_or(i, |&s| s as usize)
    }
}

// The content a label actually draws. Wrapping measures in the label's own
// pixel space with `label.scale`, which gives the same breaks as measuring in
// window pixels: a screen-owned label scales its advances and its wrap width by
// the same overlay factor. A centered label has no container (it is fitted to
// the viewport), so it is left alone. Borrows the authored content whenever no
// line breaks or truncates, so a fitting label allocates nothing.
pub(super) fn laid_out<'a>(label: &'a TextLabel, font: &LoadedFont) -> LaidOut<'a> {
    let content = label.content.as_str();
    let borrowed = LaidOut {
        text: Cow::Borrowed(content),
        sources: Vec::new(),
    };
    if label.centered || (label.wrap_width <= 0.0 && label.max_lines == 0) {
        return borrowed;
    }
    let mut lines: Vec<Range<usize>> = Vec::new();
    let mut authored_lines = 0usize;
    let mut base = 0usize;
    for authored in content.split('\n') {
        authored_lines += 1;
        if label.wrap_width > 0.0 {
            wrap_line(
                authored,
                base,
                font,
                label.scale,
                label.wrap_width,
                &mut lines,
            );
        } else {
            lines.push(base..base + authored.len());
        }
        base += authored.len() + 1;
    }
    let max = label.max_lines as usize;
    let truncated = max > 0 && lines.len() > max;
    if truncated {
        lines.truncate(max);
    }
    if !truncated && lines.len() == authored_lines {
        return borrowed;
    }
    // The last drawn line keeps only the prefix that fits beside an ellipsis.
    if truncated && let Some(last) = lines.last_mut() {
        let cut = ellipsis_cut(&content[last.clone()], font, label.scale, label.wrap_width);
        last.end = last.start + cut;
    }
    let mut text = String::with_capacity(content.len() + ELLIPSIS.len());
    for (i, line) in lines.iter().enumerate() {
        if i > 0 {
            text.push('\n');
        }
        text.push_str(&content[line.clone()]);
    }
    if truncated {
        text.push_str(ELLIPSIS);
    }
    let sources = if label.color_runs.is_empty() {
        Vec::new()
    } else {
        source_map(content, &lines, truncated)
    };
    LaidOut {
        text: Cow::Owned(text),
        sources,
    }
}

// The authored character index of every drawn character: each line's
// characters in order, the break ahead of a line standing in for the character
// that ended the line before, and an ellipsis taking the character it hides.
fn source_map(content: &str, lines: &[Range<usize>], ellipsis: bool) -> Vec<u32> {
    let mut out = Vec::with_capacity(content.len() + ELLIPSIS.len());
    // Characters before `byte`, advanced along the increasing line starts.
    let (mut byte, mut chars) = (0usize, 0u32);
    for (i, line) in lines.iter().enumerate() {
        if i > 0 {
            out.push(chars);
        }
        chars += content[byte..line.start].chars().count() as u32;
        for _ in content[line.clone()].chars() {
            out.push(chars);
            chars += 1;
        }
        byte = line.end;
    }
    if ellipsis {
        out.extend([chars; ELLIPSIS.len()]);
    }
    out
}

// Greedily pack `line`'s words into `out` as byte ranges of the content `line`
// starts at `base` in, breaking at spaces. A word too wide to fit a line of its
// own is split mid-word, since leaving it whole would put it back outside the
// container wrapping exists to respect. Widths accumulate one glyph advance at
// a time in authored order, matching a from-scratch measure of the same text.
fn wrap_line(
    line: &str,
    base: usize,
    font: &LoadedFont,
    scale: f32,
    width: f32,
    out: &mut Vec<Range<usize>>,
) {
    let advance = |ch: char| advance_px(ch, font, scale);
    let mut push = |start: usize, end: usize| out.push(base + start..base + end);
    // The line under construction, `line[start..end]`, and its measured width.
    let (mut start, mut end) = (0usize, 0usize);
    let mut current_width = 0.0_f32;
    // Byte offset of the next word (words are separated by single spaces).
    let mut pos = 0usize;
    for word in line.split(' ') {
        let word_end = pos + word.len();
        // The candidate: the word appended to the current line (joined by the
        // space between them), or the word alone when the line is empty.
        let (cand_start, cand_width) = if end > start {
            let mut w = current_width;
            for ch in line[end..word_end].chars() {
                w += advance(ch);
            }
            (start, w)
        } else {
            let mut w = 0.0_f32;
            for ch in word.chars() {
                w += advance(ch);
            }
            (pos, w)
        };
        if cand_width <= width {
            (start, end, current_width) = (cand_start, word_end, cand_width);
            pos = word_end + 1;
            continue;
        }
        if end > start {
            push(start, end);
        }
        // The word now starts a line of its own; split it if even that overflows.
        (start, end) = (pos, word_end);
        loop {
            let (mut w, mut chars) = (0.0_f32, 0usize);
            for ch in line[start..end].chars() {
                w += advance(ch);
                chars += 1;
            }
            if w <= width || chars <= 1 {
                current_width = w;
                break;
            }
            // The longest head (at least one char) that fits the width.
            let mut acc = 0.0_f32;
            let mut head_end = start;
            for (i, ch) in line[start..end].char_indices() {
                let next = acc + advance(ch);
                if head_end > start && next > width {
                    break;
                }
                acc = next;
                head_end = start + i + ch.len_utf8();
            }
            push(start, head_end);
            start = head_end;
        }
        pos = word_end + 1;
    }
    push(start, end);
}

// Byte length of the longest prefix of `line` that fits `width` with a
// trailing ellipsis, found in one forward scan. Prefix widths accumulate one
// glyph advance at a time in authored order, with the ellipsis advances added
// after, matching a from-scratch measure of the same candidate. A zero width
// (capping lines without wrapping them) keeps the whole line.
fn ellipsis_cut(line: &str, font: &LoadedFont, scale: f32, width: f32) -> usize {
    if width <= 0.0 {
        return line.len();
    }
    let ellipsis_w: f32 = ELLIPSIS.chars().map(|ch| advance_px(ch, font, scale)).sum();
    let mut end = 0;
    let mut prefix_w = 0.0_f32;
    for (i, ch) in line.char_indices() {
        let w = prefix_w + advance_px(ch, font, scale);
        if w + ellipsis_w > width {
            break;
        }
        prefix_w = w;
        end = i + ch.len_utf8();
    }
    end
}
