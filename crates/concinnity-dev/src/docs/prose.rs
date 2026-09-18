// Rustdoc as reference prose: what a page keeps of a type's doc, and how its
// examples read.

use concinnity_cook::authoring::world::{entry_line, parse_entry};

// Collapse a multi-line doc to a single line, for somewhere with room for one.
pub(super) fn collapse_doc(doc: &str) -> String {
    doc.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) fn first_paragraph(doc: &str) -> String {
    let para = doc.split("\n\n").next().unwrap_or("");
    para.split('\n')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

// Drop ```rust blocks. They are for the Rust caller reading `concinnity::components`;
// this reference is for the world.jsonl author, who is never shown a Rust name.
pub(super) fn strip_rust_blocks(doc: &str) -> String {
    let mut out = String::new();
    let mut in_rust = false;
    for line in doc.lines() {
        let trimmed = line.trim();
        if in_rust {
            if trimmed == "```" {
                in_rust = false;
            }
            continue;
        }
        if trimmed.starts_with("```rust") || trimmed == "```no_run" || trimmed == "```ignore" {
            in_rust = true;
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

// Rewrite each ```json block that holds a world entry as the line the world
// file takes, so an example copies into world.jsonl as it stands. The source
// may spread the entry over several lines for rustdoc; the page shows the
// writer's compact form. A block that is not an entry is left as written.
pub(super) fn entry_examples_as_lines(doc: &str) -> String {
    let mut out = String::new();
    let mut block: Option<Vec<&str>> = None;
    for line in doc.lines() {
        let trimmed = line.trim();
        match &mut block {
            None if trimmed == "```json" => block = Some(Vec::new()),
            None => {
                out.push_str(line);
                out.push('\n');
            }
            Some(body) if trimmed == "```" => {
                let body = std::mem::take(body).join("\n");
                let entry = parse_entry(&body).ok();
                let text = entry.and_then(|e| entry_line(&e).ok()).unwrap_or(body);
                out.push_str(&format!("```json\n{text}\n```\n"));
                block = None;
            }
            Some(body) => body.push(line),
        }
    }
    if let Some(body) = block {
        out.push_str("```json\n");
        out.push_str(&body.join("\n"));
        out.push('\n');
    }
    out
}

// Remove markdown table lines (starting with '|') from a doc string.
// Collapses the resulting double-blank lines left behind.
pub(super) fn strip_table_lines(doc: &str) -> String {
    let mut out = String::new();
    let mut prev_blank = false;
    for line in doc.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('|') {
            continue;
        }
        let is_blank = trimmed.is_empty();
        if is_blank && prev_blank {
            continue;
        }
        out.push_str(line);
        out.push('\n');
        prev_blank = is_blank;
    }
    out.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_entry_example_renders_as_its_world_line() {
        let doc = "Declares it.\n\n```json\n[\"Window\", {\n  \"title\": \"Game\",\n  \"width\": 1280\n}]\n```\n\nAfter.";
        assert_eq!(
            entry_examples_as_lines(doc),
            "Declares it.\n\n```json\n[\"Window\",{\"title\":\"Game\",\"width\":1280}]\n```\n\nAfter.\n"
        );
    }

    #[test]
    fn a_json_block_that_is_not_an_entry_is_left_as_written() {
        let doc = "```json\n{\n  \"a\": 1\n}\n```\n";
        assert_eq!(entry_examples_as_lines(doc), doc);
    }

    #[test]
    fn rust_blocks_and_table_lines_are_dropped() {
        let doc = "Keep.\n\n```rust\nlet x = 1;\n```\n\n| a | b |\n|---|---|\n\nAfter.";
        assert_eq!(
            strip_table_lines(&strip_rust_blocks(doc)),
            "Keep.\n\nAfter."
        );
    }

    #[test]
    fn the_first_paragraph_joins_its_lines() {
        assert_eq!(first_paragraph("One\ntwo.\n\nThree."), "One two.");
        assert_eq!(collapse_doc("One.\n\n  Two.  \n"), "One. Two.");
    }
}
