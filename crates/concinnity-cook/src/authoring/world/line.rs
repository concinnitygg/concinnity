// The world file's line format: one `["Type", {args}]` entry per line. The
// rest of the cook works on the `{"type", "args"}` object this converts each
// line into, so the file's shape is decided here and nowhere else.

use serde_json::{Map, Value};

/// A world file line that does not hold an entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineError {
    /// The 1-based line number.
    pub line: usize,
    /// What is wrong with the line.
    pub message: String,
}

impl std::fmt::Display for LineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

/// Every line of a world file that does not hold an entry, in file order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError(pub Vec<LineError>);

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, line) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str("\n")?;
            }
            write!(f, "{line}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ParseError {}

const ENTRY_SHAPE: &str = "an entry is a [\"Type\", {args}] array";

/// Parse world file text into its entries, each as a `{"type", "args"}`
/// object.
///
/// A line is a JSON array of `["Type"]` or `["Type", {args}]`; blank lines and
/// lines starting with `//` are skipped. Every malformed line is reported, not
/// just the first. Entry order is preserved.
pub fn parse_world_jsonl(content: &str) -> Result<Vec<Value>, ParseError> {
    let mut entries = Vec::new();
    let mut errors = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        match parse_entry(trimmed) {
            Ok(entry) => entries.push(entry),
            Err(message) => errors.push(LineError {
                line: i + 1,
                message,
            }),
        }
    }
    if errors.is_empty() {
        Ok(entries)
    } else {
        Err(ParseError(errors))
    }
}

/// Parse one entry written as a world line, `["Type", {args}]` or `["Type"]`,
/// into a `{"type", "args"}` object.
pub fn parse_entry(text: &str) -> Result<Value, String> {
    let value: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    entry_from_line(value)
}

/// Convert one parsed world line into a `{"type", "args"}` object. A line
/// with no args column reads as empty args.
pub fn entry_from_line(value: Value) -> Result<Value, String> {
    let columns = match value {
        Value::Array(columns) => columns,
        Value::Object(_) => return Err(format!("{ENTRY_SHAPE}, not an object")),
        other => return Err(format!("{ENTRY_SHAPE}, not {}", kind(&other))),
    };
    if !(1..=2).contains(&columns.len()) {
        return Err(format!(
            "{ENTRY_SHAPE} of one or two columns, not {}",
            columns.len()
        ));
    }
    let mut columns = columns.into_iter();
    let Some(Value::String(ty)) = columns.next() else {
        return Err("the first column is the type name, a string".to_string());
    };
    let args = match columns.next() {
        None => Value::Object(Map::new()),
        Some(args @ Value::Object(_)) => args,
        Some(other) => {
            return Err(format!(
                "'{ty}': the second column is the args, an object, not {}",
                kind(&other)
            ));
        }
    };
    let mut entry = Map::new();
    entry.insert("type".to_string(), Value::String(ty));
    entry.insert("args".to_string(), args);
    Ok(Value::Object(entry))
}

/// The world line for one `{"type", "args"}` entry: compact
/// `["Type", {args}]`, both columns always written, no newline. Fails on an
/// entry a line cannot hold (no type, args that are not an object, or a key
/// beside the two), rather than dropping what does not fit.
pub fn entry_line(entry: &Value) -> std::io::Result<String> {
    let bad = |msg: String| std::io::Error::new(std::io::ErrorKind::InvalidData, msg);
    let obj = entry
        .as_object()
        .ok_or_else(|| bad(format!("an entry is an object, not {}", kind(entry))))?;
    if let Some(key) = obj.keys().find(|k| !matches!(k.as_str(), "type" | "args")) {
        return Err(bad(format!(
            "a world line cannot hold the entry key `{key}`"
        )));
    }
    let ty = obj
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("an entry needs a string `type`".to_string()))?;
    let empty = Value::Object(Map::new());
    let args = match obj.get("args") {
        None | Some(Value::Null) => &empty,
        Some(args @ Value::Object(_)) => args,
        Some(other) => {
            return Err(bad(format!(
                "'{ty}': args are an object, not {}",
                kind(other)
            )));
        }
    };
    let line = Value::Array(vec![Value::String(ty.to_string()), args.clone()]);
    serde_json::to_string(&line).map_err(|e| bad(e.to_string()))
}

/// Serialize entries as world file text, one line per entry.
pub fn write_world_jsonl(entries: &[Value]) -> std::io::Result<String> {
    let mut out = String::new();
    for entry in entries {
        out.push_str(&entry_line(entry)?);
        out.push('\n');
    }
    Ok(out)
}

fn kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a bool",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn errors(content: &str) -> Vec<LineError> {
        parse_world_jsonl(content).unwrap_err().0
    }

    #[test]
    fn empty_text_holds_no_entries() {
        assert!(parse_world_jsonl("").unwrap().is_empty());
    }

    #[test]
    fn blank_and_comment_lines_are_skipped() {
        assert!(
            parse_world_jsonl("\n  \n// a comment\n")
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn a_two_column_line_becomes_a_type_and_args_entry() {
        let entries = parse_world_jsonl(r#"["Prop",{"$id":"p","mesh":"m"}]"#).unwrap();
        assert_eq!(
            entries,
            [json!({"type": "Prop", "args": {"$id": "p", "mesh": "m"}})]
        );
    }

    #[test]
    fn a_one_column_line_reads_as_empty_args() {
        let entries = parse_world_jsonl(r#"["Scene"]"#).unwrap();
        assert_eq!(entries, [json!({"type": "Scene", "args": {}})]);
    }

    #[test]
    fn entries_keep_file_order() {
        let entries = parse_world_jsonl("[\"Logger\",{\"$id\":\"a\"}]\n\n[\"Window\"]\n").unwrap();
        assert_eq!(entries[0]["args"]["$id"], "a");
        assert_eq!(entries[1]["type"], "Window");
    }

    #[test]
    fn an_object_line_names_the_tuple_form() {
        let errs = errors(r#"{"type":"Window","args":{}}"#);
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].line, 1);
        assert!(errs[0].message.contains("[\"Type\", {args}]"), "{errs:?}");
        assert!(errs[0].message.contains("not an object"), "{errs:?}");
    }

    #[test]
    fn malformed_arrays_say_which_column_is_wrong() {
        let errs = errors("[]\n[\"A\",{},{}]\n[7,{}]\n[\"Prop\",[]]\n[\"Prop\",null]\n\"Prop\"\n");
        let lines: Vec<usize> = errs.iter().map(|e| e.line).collect();
        assert_eq!(lines, [1, 2, 3, 4, 5, 6]);
        assert!(errs[0].message.contains("one or two columns, not 0"));
        assert!(errs[1].message.contains("not 3"));
        assert!(errs[2].message.contains("type name, a string"));
        assert!(errs[3].message.contains("'Prop'") && errs[3].message.contains("an array"));
        assert!(errs[4].message.contains("not null"));
        assert!(errs[5].message.contains("not a string"));
    }

    #[test]
    fn invalid_json_is_reported_with_its_line() {
        let errs = errors("[\"Scene\"]\n[not json\n");
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].line, 2);
        assert!(errs[0].to_string().starts_with("line 2: "));
    }

    #[test]
    fn every_malformed_line_is_reported() {
        let err = parse_world_jsonl("{}\n[\"Scene\"]\n{}\n").unwrap_err();
        assert_eq!(err.0.len(), 2);
        assert_eq!(err.to_string().lines().count(), 2);
    }

    #[test]
    fn the_writer_emits_both_columns_compactly() {
        let out = write_world_jsonl(&[
            json!({"type": "Scene"}),
            json!({"type": "Window", "args": null}),
            json!({"type": "Prop", "args": {"$id": "p", "position": [0, 1, 2]}}),
        ])
        .unwrap();
        assert_eq!(
            out,
            "[\"Scene\",{}]\n[\"Window\",{}]\n[\"Prop\",{\"$id\":\"p\",\"position\":[0,1,2]}]\n"
        );
    }

    #[test]
    fn written_text_parses_back_to_the_same_entries() {
        let entries = vec![
            json!({"type": "GraphicsConfig", "args": {"clear_color": [0.05, 0.06, 0.09, 1.0]}}),
            json!({"type": "Prop", "args": {"$id": "box", "mesh": "m", "nested": {"a": [1]}}}),
            json!({"type": "Include", "args": {"path": "lighting.jsonl"}}),
        ];
        let text = write_world_jsonl(&entries).unwrap();
        assert_eq!(parse_world_jsonl(&text).unwrap(), entries);
        assert_eq!(
            write_world_jsonl(&parse_world_jsonl(&text).unwrap()).unwrap(),
            text
        );
    }

    #[test]
    fn the_writer_refuses_what_a_line_cannot_hold() {
        for entry in [
            json!({"args": {}}),
            json!({"type": "Prop", "args": [1]}),
            json!({"type": "Prop", "name": "x"}),
            json!(["Prop", {}]),
        ] {
            let err = entry_line(&entry).unwrap_err();
            assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{entry}");
        }
    }

    #[test]
    fn parse_entry_reads_a_single_line() {
        assert_eq!(
            parse_entry(r#"["Window",{"title":"t"}]"#).unwrap(),
            json!({"type": "Window", "args": {"title": "t"}})
        );
        assert!(parse_entry(r#"{"type":"Window"}"#).is_err());
    }
}
