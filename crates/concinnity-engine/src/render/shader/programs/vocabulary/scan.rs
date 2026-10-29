//! Reading HLSL source for the vocabulary drift tests: identifiers, whether a
//! function is defined, and which fields a struct declares.

use alloc::collections::BTreeSet;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

pub(super) fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

// Every identifier in `text`.
pub(super) fn identifiers(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !is_ident(c))
        .filter(|w| w.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_'))
}

// `text` without its `//` comments.
pub(super) fn strip_comments(text: &str) -> String {
    text.lines()
        .map(|l| l.split_once("//").map_or(l, |(code, _)| code))
        .collect::<Vec<_>>()
        .join("\n")
}

// Whether `name` is defined in `source`: a macro taking arguments, or a
// function whose parameter list is followed by a body rather than a `;`.
pub(super) fn defines_helper(source: &str, name: &str) -> bool {
    if source.contains(&alloc::format!("#define {name}(")) {
        return true;
    }
    let call = alloc::format!("{name}(");
    source.match_indices(&call).any(|(at, _)| {
        let line_start = source[..at].rfind('\n').map_or(0, |i| i + 1);
        let before = source[line_start..at].trim();
        if before.is_empty() || !before.chars().all(is_ident) {
            return false;
        }
        let Some(close) = source[at..].find(')') else {
            return false;
        };
        source[at + close + 1..].trim_start().starts_with('{')
    })
}

// The field names `struct name { ... }` declares in `source`.
pub(super) fn struct_fields(source: &str, name: &str) -> BTreeSet<String> {
    let head = alloc::format!("struct {name}");
    let at = source
        .match_indices(&head)
        .map(|(i, _)| i + head.len())
        .find(|&i| !source[i..].starts_with(is_ident))
        .unwrap_or_else(|| panic!("no `struct {name}`"));
    let body = &source[at..];
    let open = body.find('{').unwrap() + 1;
    let close = body.find("};").unwrap();
    body[open..close]
        .split(';')
        .filter_map(|decl| {
            // Drop attributes, a semantic, and an array extent; the name is
            // the last word left.
            let decl = decl.rsplit("]]").next().unwrap();
            let decl = decl.split(':').next().unwrap();
            let decl = decl.split('[').next().unwrap();
            identifiers(decl).last().map(ToString::to_string)
        })
        .collect()
}
