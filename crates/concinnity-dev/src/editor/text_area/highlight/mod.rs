//! Syntax highlighting for the text area: a highlighter per language turns one
//! line into colored spans, carrying what a construct left open (a block
//! comment, a code fence) to the next line. Each line's start state is cached
//! (`states`), so an edit rescans from its own line down and no further than
//! the lines drawn. `window` fits a line's spans to the cells a row shows.

mod hlsl;
mod markdown;
pub(crate) mod states;
pub(crate) mod window;

pub(crate) use hlsl::HLSL;
pub(crate) use markdown::MARKDOWN;

use crate::editor::theme;

// What a span of source is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Token {
    Keyword,
    Type,
    Number,
    String,
    Comment,
    Preprocessor,
    // A name the engine provides.
    Engine,
    Heading,
    Emphasis,
    Link,
    Code,
}

impl Token {
    pub(crate) fn color(self) -> [f32; 3] {
        match self {
            Token::Keyword => theme::CODE_KEYWORD,
            Token::Type => theme::CODE_TYPE,
            Token::Number => theme::CODE_NUMBER,
            Token::String => theme::CODE_STRING,
            Token::Comment => theme::CODE_COMMENT,
            Token::Preprocessor => theme::CODE_PREPROCESSOR,
            Token::Engine => theme::CODE_ENGINE,
            Token::Heading => theme::CODE_HEADING,
            Token::Emphasis => theme::CODE_EMPHASIS,
            Token::Link => theme::CODE_LINK,
            Token::Code => theme::CODE_LITERAL,
        }
    }
}

// Characters `[start, start + len)` of a line, by character column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Span {
    pub(crate) start: usize,
    pub(crate) len: usize,
    pub(crate) token: Token,
}

// What a line leaves open for the next one; each language gives its own
// values meaning, and the default is nothing open.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct LineState(pub(crate) u8);

pub(crate) trait Highlighter: Sync + std::fmt::Debug {
    // Append `line`'s spans to `out`, in order and never overlapping, for a
    // line starting in `state`; returns the state the next line starts in.
    fn line(&self, line: &str, state: LineState, out: &mut Vec<Span>) -> LineState;
}

// A line as characters, with a cursor, for the per-language scanners.
struct Scan {
    chars: Vec<char>,
    at: usize,
}

impl Scan {
    fn new(line: &str) -> Self {
        Self {
            chars: line.chars().collect(),
            at: 0,
        }
    }

    fn peek(&self, ahead: usize) -> Option<char> {
        self.chars.get(self.at + ahead).copied()
    }

    fn done(&self) -> bool {
        self.at >= self.chars.len()
    }

    fn starts_with(&self, s: &str) -> bool {
        s.chars().enumerate().all(|(i, c)| self.peek(i) == Some(c))
    }

    // The column `pat` next starts at, from the cursor.
    fn find(&self, pat: &str) -> Option<usize> {
        let pat: Vec<char> = pat.chars().collect();
        (self.at..self.chars.len()).find(|&i| self.chars[i..].starts_with(&pat))
    }

    fn len(&self) -> usize {
        self.chars.len()
    }
}

fn push(out: &mut Vec<Span>, start: usize, end: usize, token: Token) {
    if end > start {
        out.push(Span {
            start,
            len: end - start,
            token,
        });
    }
}

#[cfg(test)]
mod tests;
