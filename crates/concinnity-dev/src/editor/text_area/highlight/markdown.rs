//! Markdown: headings, emphasis, links, inline code, and fenced code blocks
//! (a fence runs across lines until it closes).

use super::{Highlighter, LineState, Scan, Span, Token, push};

pub(crate) static MARKDOWN: Markdown = Markdown;

#[derive(Debug)]
pub(crate) struct Markdown;

const IN_FENCE: LineState = LineState(1);

// Whether `line` opens or closes a code fence: up to three spaces, then three
// backticks or tildes.
fn is_fence(line: &str) -> bool {
    let indent = line.len() - line.trim_start_matches(' ').len();
    let rest = &line[indent..];
    indent <= 3 && (rest.starts_with("```") || rest.starts_with("~~~"))
}

// A `#` to `######` heading: the marks, then a space or the end of the line.
fn is_heading(line: &str) -> bool {
    let rest = line.trim_start_matches(' ');
    if line.len() - rest.len() > 3 {
        return false;
    }
    let marks = rest.len() - rest.trim_start_matches('#').len();
    (1..=6).contains(&marks) && rest[marks..].chars().next().is_none_or(|c| c == ' ')
}

impl Highlighter for Markdown {
    fn line(&self, line: &str, state: LineState, out: &mut Vec<Span>) -> LineState {
        let chars = line.chars().count();
        if state == IN_FENCE {
            push(out, 0, chars, Token::Code);
            return if is_fence(line) {
                LineState::default()
            } else {
                IN_FENCE
            };
        }
        if is_fence(line) {
            push(out, 0, chars, Token::Code);
            return IN_FENCE;
        }
        if is_heading(line) {
            push(out, 0, chars, Token::Heading);
            return LineState::default();
        }
        inline(line, out);
        LineState::default()
    }
}

// The spans inside one line of text: code, links, and emphasis.
fn inline(line: &str, out: &mut Vec<Span>) {
    let mut s = Scan::new(line);
    while !s.done() {
        let start = s.at;
        let c = s.peek(0).unwrap_or(' ');
        let end = match c {
            '`' => code_span(&s),
            '[' => link(&s),
            '*' | '_' => emphasis(&s, c),
            _ => None,
        };
        match end {
            Some(end) => {
                let token = match c {
                    '`' => Token::Code,
                    '[' => Token::Link,
                    _ => Token::Emphasis,
                };
                push(out, start, end, token);
                s.at = end;
            }
            None => s.at += 1,
        }
    }
}

// `` `code` ``: the end just past the closing backtick run of the same length.
fn code_span(s: &Scan) -> Option<usize> {
    let ticks = (0..).take_while(|&i| s.peek(i) == Some('`')).count();
    let mut i = s.at + ticks;
    while i < s.len() {
        let run = (i..s.len()).take_while(|&j| s.chars[j] == '`').count();
        if run == ticks {
            return Some(i + run);
        }
        i += run.max(1);
    }
    None
}

// `[text](target)`: the end just past the `)`.
fn link(s: &Scan) -> Option<usize> {
    let close = (s.at + 1..s.len()).find(|&i| s.chars[i] == ']')?;
    if s.chars.get(close + 1) != Some(&'(') {
        return None;
    }
    let paren = (close + 2..s.len()).find(|&i| s.chars[i] == ')')?;
    Some(paren + 1)
}

// `*em*`, `**strong**`, `_em_`, `__strong__`: the end just past the closing
// marks. An opening mark is followed by text and a closing one follows text;
// underscores inside a word (`snake_case`) mark nothing.
fn emphasis(s: &Scan, mark: char) -> Option<usize> {
    let marks = (0..)
        .take_while(|&i| s.peek(i) == Some(mark))
        .count()
        .min(2);
    let inner = s.at + marks;
    let intraword = |i: usize| s.chars.get(i).is_some_and(|c| c.is_alphanumeric());
    if s.chars.get(inner).is_none_or(|c| c.is_whitespace())
        || (mark == '_' && s.at > 0 && intraword(s.at - 1))
    {
        return None;
    }
    let mut i = inner + 1;
    while i + marks <= s.len() {
        let closes = (0..marks).all(|k| s.chars[i + k] == mark)
            && !s.chars[i - 1].is_whitespace()
            && !(mark == '_' && intraword(i + marks));
        if closes {
            return Some(i + marks);
        }
        i += 1;
    }
    None
}
