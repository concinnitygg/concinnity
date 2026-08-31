//! Reading the workspace's own Rust sources, for the guard tests that scan it.
//!
//! Several checks in this workspace are source scans rather than runtime
//! assertions, because what they forbid cannot be observed at runtime without
//! already having lost: a test that opens a window hangs, and a test that takes
//! a lock twice deadlocks. Neither reports a failure. Reading the code before
//! it runs is what turns those into an assertion.
//!
//! The brace matching here is the part worth sharing. A test body is found by
//! counting braces from its `fn` line, and a `{` inside a string literal is not
//! a block: a fixture like `"{ not json"` -- exactly what a malformed-input
//! test writes -- otherwise runs the body to the end of the file and sweeps in
//! whatever the neighbouring tests do.

use std::path::{Path, PathBuf};

/// A function body found in a source file.
pub struct FnBody {
    /// The function's name.
    pub name: String,
    /// 1-indexed line the `fn` appears on.
    pub line: usize,
    /// The body's text, with whole-line comments removed.
    pub text: String,
}

/// Every `.rs` file under `dirs`, skipping build output and untracked trees.
///
/// `skip_file` is a file name the caller wants left out: a scan whose own
/// source states the shape it forbids would otherwise report itself.
pub fn rust_sources(dirs: &[PathBuf], skip_file: &str) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for dir in dirs {
        collect(dir, skip_file, &mut files);
    }
    files
}

fn collect(dir: &Path, skip_file: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let path = entry.expect("directory entry reads").path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if path.is_dir() {
            if !matches!(name, "target" | "private" | "vendor" | "tmp" | ".git") {
                collect(&path, skip_file, out);
            }
        } else if path.extension().is_some_and(|e| e == "rs") && name != skip_file {
            out.push(path);
        }
    }
}

// Brace depth change across one line, ignoring braces inside string, raw
// string, and character literals, and inside a line comment.
//
// `in_raw` carries an open raw string across lines: `r#"..."#` may span them.
fn scan_line(line: &str, in_raw: &mut Option<usize>) -> isize {
    let bytes: Vec<char> = line.chars().collect();
    let mut depth = 0isize;
    let mut i = 0;

    // Finish a raw string opened on an earlier line.
    if let Some(hashes) = *in_raw {
        let close: String = core::iter::once('"')
            .chain(core::iter::repeat_n('#', hashes))
            .collect();
        match line.find(&close) {
            Some(at) => {
                *in_raw = None;
                i = line[..at + close.len()].chars().count();
            }
            None => return 0,
        }
    }

    while i < bytes.len() {
        let c = bytes[i];
        // A line comment ends the line.
        if c == '/' && bytes.get(i + 1) == Some(&'/') {
            break;
        }
        // Raw string: r, some hashes, then a quote.
        if c == 'r' {
            let mut j = i + 1;
            let mut hashes = 0;
            while bytes.get(j) == Some(&'#') {
                hashes += 1;
                j += 1;
            }
            if bytes.get(j) == Some(&'"') {
                let close: String = core::iter::once('"')
                    .chain(core::iter::repeat_n('#', hashes))
                    .collect();
                let rest: String = bytes[j + 1..].iter().collect();
                match rest.find(&close) {
                    Some(at) => {
                        i = j + 1 + rest[..at + close.len()].chars().count();
                        continue;
                    }
                    None => {
                        *in_raw = Some(hashes);
                        return depth;
                    }
                }
            }
        }
        // Ordinary string, with backslash escapes.
        if c == '"' {
            i += 1;
            while i < bytes.len() {
                if bytes[i] == '\\' {
                    i += 2;
                    continue;
                }
                if bytes[i] == '"' {
                    break;
                }
                i += 1;
            }
            i += 1;
            continue;
        }
        // Character literal, including `'{'` and the escaped forms. A lifetime
        // (`'a`) has no closing quote, so only treat it as a literal when one
        // is where a literal would put it.
        if c == '\'' {
            let closes_at = if bytes.get(i + 1) == Some(&'\\') {
                i + 3
            } else {
                i + 2
            };
            if bytes.get(closes_at) == Some(&'\'') {
                i = closes_at + 1;
                continue;
            }
        }
        if c == '{' {
            depth += 1;
        } else if c == '}' {
            depth -= 1;
        }
        i += 1;
    }
    depth
}

// The body starting at `start`, brace-matched to its close.
fn body_at(lines: &[&str], start: usize) -> (usize, String) {
    let mut depth = 0isize;
    let mut opened = false;
    let mut in_raw = None;
    let mut end = start;
    for (j, line) in lines.iter().enumerate().skip(start) {
        depth += scan_line(line, &mut in_raw);
        if depth > 0 {
            opened = true;
        }
        end = j;
        if opened && depth <= 0 {
            break;
        }
    }
    let text = lines[start..=end]
        .iter()
        .filter(|l| !l.trim_start().starts_with("//"))
        .copied()
        .collect::<Vec<_>>()
        .join("\n");
    (end, text)
}

fn name_after_fn(line: &str) -> String {
    line.split("fn ")
        .nth(1)
        .unwrap_or_default()
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

/// Every `#[test]` function in `text`.
pub fn test_bodies(text: &str) -> Vec<FnBody> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if line.trim() != "#[test]" {
            continue;
        }
        let Some(start) = (i + 1..lines.len().min(i + 8)).find(|&j| lines[j].contains("fn "))
        else {
            continue;
        };
        let (_, body) = body_at(&lines, start);
        out.push(FnBody {
            name: name_after_fn(lines[start]),
            line: start + 1,
            text: body,
        });
    }
    out
}

/// Every function in `text`, `#[test]` or not: the helpers a test may call.
pub fn fn_bodies(text: &str) -> Vec<FnBody> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if !line.contains("fn ") {
            continue;
        }
        let name = name_after_fn(line);
        if name.is_empty() {
            continue;
        }
        let (_, body) = body_at(&lines, i);
        out.push(FnBody {
            name,
            line: i + 1,
            text: body,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_brace_in_a_string_does_not_open_a_block() {
        let source = "#[test]\nfn t() {\n    write(\"{not json\");\n}\nfn after() { second(); }\n";
        let bodies = test_bodies(source);

        assert_eq!(bodies.len(), 1);
        assert!(bodies[0].text.contains("not json"));
        assert!(
            !bodies[0].text.contains("second()"),
            "the body stopped at its own close: {:?}",
            bodies[0].text
        );
    }

    #[test]
    fn a_brace_in_a_raw_string_does_not_open_a_block() {
        let source =
            "#[test]\nfn t() {\n    write(r#\"{\"a\":1}\"#);\n}\nfn after() { second(); }\n";
        let bodies = test_bodies(source);

        assert_eq!(bodies.len(), 1);
        assert!(!bodies[0].text.contains("second()"), "{:?}", bodies[0].text);
    }

    #[test]
    fn a_brace_character_literal_does_not_open_a_block() {
        let source = "#[test]\nfn t() {\n    let c = '{';\n}\nfn after() { second(); }\n";
        let bodies = test_bodies(source);

        assert_eq!(bodies.len(), 1);
        assert!(!bodies[0].text.contains("second()"), "{:?}", bodies[0].text);
    }

    #[test]
    fn a_brace_in_a_line_comment_does_not_open_a_block() {
        let source = "#[test]\nfn t() {\n    do_it(); // trailing {\n}\nfn after() { second(); }\n";
        let bodies = test_bodies(source);

        assert_eq!(bodies.len(), 1);
        assert!(!bodies[0].text.contains("second()"), "{:?}", bodies[0].text);
    }

    #[test]
    fn a_multi_line_raw_string_is_carried_across_lines() {
        let source = "#[test]\nfn t() {\n    let s = r#\"\n{ unbalanced\n\"#;\n}\nfn after() { second(); }\n";
        let bodies = test_bodies(source);

        assert_eq!(bodies.len(), 1);
        assert!(!bodies[0].text.contains("second()"), "{:?}", bodies[0].text);
    }

    #[test]
    fn nested_blocks_close_at_the_right_brace() {
        let source =
            "#[test]\nfn t() {\n    if x {\n        y();\n    }\n}\nfn after() { second(); }\n";
        let bodies = test_bodies(source);

        assert!(bodies[0].text.contains("y()"));
        assert!(!bodies[0].text.contains("second()"), "{:?}", bodies[0].text);
    }

    #[test]
    fn a_lifetime_is_not_a_character_literal() {
        let source = "#[test]\nfn t() {\n    let v: Vec<&'static str> = vec![];\n}\nfn after() { second(); }\n";
        let bodies = test_bodies(source);

        assert_eq!(bodies.len(), 1);
        assert!(!bodies[0].text.contains("second()"), "{:?}", bodies[0].text);
    }

    #[test]
    fn helpers_are_found_alongside_tests() {
        let source = "fn helper() {\n    work();\n}\n\n#[test]\nfn t() {\n    helper();\n}\n";
        let names: Vec<String> = fn_bodies(source).into_iter().map(|b| b.name).collect();

        assert!(names.contains(&"helper".to_string()));
        assert!(names.contains(&"t".to_string()));
        assert_eq!(test_bodies(source).len(), 1, "only the #[test] is a test");
    }

    #[test]
    fn comment_lines_are_stripped_from_a_body() {
        let source = "#[test]\nfn t() {\n    // set_state_dir(x);\n    real();\n}\n";
        let body = &test_bodies(source)[0].text;

        assert!(!body.contains("set_state_dir"));
        assert!(body.contains("real()"));
    }
}
