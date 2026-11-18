//! Syntax highlighting for the text area: a highlighter per language turns one
//! line into colored spans, carrying what a construct left open (a block
//! comment, a code fence) to the next line. Each line's start state is cached
//! (`states`), so an edit rescans from its own line down and no further than
//! the lines drawn. `window` fits a line's spans to the cells a row shows.

use alloc::vec::Vec;

mod hlsl;
mod markdown;
pub(crate) mod states;
pub(crate) mod window;

pub use hlsl::{HLSL, SDF_HLSL};
pub use markdown::MARKDOWN;

/// What a span of source is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token {
    /// A language keyword.
    Keyword,
    /// A type name.
    Type,
    /// A number literal.
    Number,
    /// A string literal.
    String,
    /// A comment.
    Comment,
    /// A preprocessor directive.
    Preprocessor,
    /// A name the engine provides.
    Engine,
    /// A Markdown heading.
    Heading,
    /// Emphasized Markdown text.
    Emphasis,
    /// A Markdown link.
    Link,
    /// Inline Markdown code.
    Code,
}

impl Token {
    /// The linear RGB color the token draws in.
    pub fn color(self) -> [f32; 3] {
        match self {
            Self::Keyword => [0.78, 0.56, 0.96],
            Self::Type => [0.42, 0.78, 0.86],
            Self::Number => [0.72, 0.86, 0.56],
            Self::String => [0.92, 0.66, 0.46],
            Self::Comment => [0.46, 0.54, 0.48],
            Self::Preprocessor => [0.88, 0.52, 0.72],
            Self::Engine => [0.96, 0.80, 0.40],
            Self::Heading => [0.56, 0.74, 1.00],
            Self::Emphasis => [0.90, 0.72, 0.94],
            Self::Link => [0.44, 0.78, 0.96],
            Self::Code => [0.74, 0.84, 0.62],
        }
    }
}

/// Characters `[start, start + len)` of a line, by character column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// The first character column.
    pub start: usize,
    /// How many characters.
    pub len: usize,
    /// What the characters are.
    pub token: Token,
}

/// What a line leaves open for the next one; each language gives its own
/// values meaning, and the default is nothing open.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LineState(pub u8);

/// Turns one line of a language into colored spans.
pub trait Highlighter: Sync + core::fmt::Debug {
    /// Append `line`'s spans to `out`, in order and never overlapping, for a
    /// line starting in `state`; returns the state the next line starts in.
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
