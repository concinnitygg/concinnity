//! HLSL: keywords, types, numbers, strings, comments (a block comment may run
//! across lines), preprocessor directives, and the names the engine provides a
//! Shader, from the engine's own vocabulary table.

use concinnity_core::render::shader_programs::vocabulary::{
    Block, ENTRIES, Kind, RECORD_STRUCT, VARYING_STRUCT,
};

use super::{Highlighter, LineState, Scan, Span, Token, push};

pub(crate) static HLSL: Hlsl = Hlsl;

#[derive(Debug)]
pub(crate) struct Hlsl;

const IN_BLOCK_COMMENT: LineState = LineState(1);

const KEYWORDS: &[&str] = &[
    "break",
    "case",
    "cbuffer",
    "centroid",
    "const",
    "continue",
    "default",
    "discard",
    "do",
    "else",
    "false",
    "for",
    "groupshared",
    "if",
    "in",
    "inline",
    "inout",
    "linear",
    "nointerpolation",
    "noperspective",
    "out",
    "packoffset",
    "precise",
    "register",
    "return",
    "sample",
    "static",
    "struct",
    "switch",
    "tbuffer",
    "true",
    "typedef",
    "uniform",
    "void",
    "while",
];

// Scalar types, which also take vector (`float3`) and matrix (`float4x4`)
// suffixes.
const SCALARS: &[&str] = &[
    "bool",
    "int",
    "uint",
    "dword",
    "half",
    "float",
    "double",
    "min16float",
    "min10float",
    "min16int",
    "min12int",
    "min16uint",
];

const OBJECT_TYPES: &[&str] = &[
    "Buffer",
    "ByteAddressBuffer",
    "ConstantBuffer",
    "RWBuffer",
    "RWByteAddressBuffer",
    "RWStructuredBuffer",
    "RWTexture1D",
    "RWTexture2D",
    "RWTexture3D",
    "SamplerComparisonState",
    "SamplerState",
    "StructuredBuffer",
    "Texture1D",
    "Texture2D",
    "Texture2DArray",
    "Texture3D",
    "TextureCube",
    "TextureCubeArray",
    "matrix",
    "vector",
];

fn is_type(word: &str) -> bool {
    if OBJECT_TYPES.contains(&word) {
        return true;
    }
    SCALARS.iter().any(|s| {
        word.strip_prefix(s)
            .is_some_and(|rest| match rest.as_bytes() {
                [] => true,
                [n] => (b'1'..=b'4').contains(n),
                [r, b'x', c] => (b'1'..=b'4').contains(r) && (b'1'..=b'4').contains(c),
                _ => false,
            })
    })
}

// A name the engine provides on its own: a helper, a block, or one of the
// structs the hooks take.
fn is_engine_name(word: &str) -> bool {
    word == RECORD_STRUCT
        || word == VARYING_STRUCT
        || Block::ALL.iter().any(|b| b.name() == word)
        || ENTRIES
            .iter()
            .any(|e| e.kind == Kind::Helper && e.name == word)
}

// Whether `field`, read through `owner.`, is one the engine provides. A block
// field names its block; the hooks' structs can be held under any name.
fn is_engine_field(owner: Option<&str>, field: &str) -> bool {
    ENTRIES.iter().any(|e| {
        e.name == field
            && match e.kind {
                Kind::Helper => false,
                Kind::BlockField(b) => owner == Some(b.name()),
                Kind::RecordField | Kind::Varying => true,
            }
    })
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

impl Highlighter for Hlsl {
    fn line(&self, line: &str, state: LineState, out: &mut Vec<Span>) -> LineState {
        let mut s = Scan::new(line);
        if state == IN_BLOCK_COMMENT && !close_block_comment(&mut s, out) {
            return IN_BLOCK_COMMENT;
        }
        // The identifier just before a `.`, for a field read through it.
        let mut owner: Option<String> = None;
        let mut after_dot = false;
        while !s.done() {
            let c = s.peek(0).unwrap_or(' ');
            let start = s.at;
            if s.starts_with("//") {
                push(out, start, s.len(), Token::Comment);
                break;
            }
            if s.starts_with("/*") {
                s.at += 2;
                if !close_block_comment_from(&mut s, start, out) {
                    return IN_BLOCK_COMMENT;
                }
                after_dot = false;
                continue;
            }
            if c == '"' {
                scan_string(&mut s);
                push(out, start, s.at, Token::String);
            } else if c == '#' && s.chars[..start].iter().all(|c| c.is_whitespace()) {
                s.at += 1;
                while s.peek(0).is_some_and(|c| c == ' ') {
                    s.at += 1;
                }
                while s.peek(0).is_some_and(is_ident) {
                    s.at += 1;
                }
                push(out, start, s.at, Token::Preprocessor);
            } else if c.is_ascii_digit()
                || (c == '.' && s.peek(1).is_some_and(|d| d.is_ascii_digit()))
            {
                scan_number(&mut s, start);
                push(out, start, s.at, Token::Number);
            } else if is_ident_start(c) {
                while s.peek(0).is_some_and(is_ident) {
                    s.at += 1;
                }
                let word: String = s.chars[start..s.at].iter().collect();
                let token = if after_dot {
                    is_engine_field(owner.as_deref(), &word).then_some(Token::Engine)
                } else if KEYWORDS.contains(&word.as_str()) {
                    Some(Token::Keyword)
                } else if is_type(&word) {
                    Some(Token::Type)
                } else if is_engine_name(&word) {
                    Some(Token::Engine)
                } else {
                    None
                };
                if let Some(token) = token {
                    push(out, start, s.at, token);
                }
                owner = Some(word);
                after_dot = false;
                continue;
            } else {
                s.at += 1;
            }
            after_dot = c == '.';
            if !after_dot && !c.is_whitespace() {
                owner = None;
            }
        }
        LineState::default()
    }
}

// Color up to and including the `*/` a line opens inside; `false` when the
// comment runs past the line.
fn close_block_comment(s: &mut Scan, out: &mut Vec<Span>) -> bool {
    close_block_comment_from(s, 0, out)
}

fn close_block_comment_from(s: &mut Scan, start: usize, out: &mut Vec<Span>) -> bool {
    match s.find("*/") {
        Some(end) => {
            s.at = end + 2;
            push(out, start, s.at, Token::Comment);
            true
        }
        None => {
            s.at = s.len();
            push(out, start, s.at, Token::Comment);
            false
        }
    }
}

// Past a `"..."` literal, or to the end of the line when it is not closed.
fn scan_string(s: &mut Scan) {
    s.at += 1;
    while let Some(c) = s.peek(0) {
        s.at += 1;
        match c {
            '\\' => s.at += 1,
            '"' => break,
            _ => {}
        }
    }
    s.at = s.at.min(s.len());
}

// Past a numeric literal: digits, a fraction, an exponent, a hex prefix, and a
// type suffix (`1.0f`, `0x1Fu`, `2e-3h`).
fn scan_number(s: &mut Scan, start: usize) {
    let hex =
        s.chars[start..].starts_with(&['0', 'x']) || s.chars[start..].starts_with(&['0', 'X']);
    while let Some(c) = s.peek(0) {
        let exponent_sign =
            (c == '+' || c == '-') && !hex && matches!(s.chars[s.at - 1], 'e' | 'E');
        if c.is_ascii_alphanumeric() || c == '.' || exponent_sign {
            s.at += 1;
        } else {
            break;
        }
    }
}
