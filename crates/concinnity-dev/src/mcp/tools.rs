//! The MCP tool surface, built from the debug server's own verb table.
//!
//! Nothing here restates a verb: the name, the description, the read-only hint,
//! and the input schema all come from `crate::debug::catalog`, so a verb added
//! to the table becomes a tool without a second list to keep in step.

use serde_json::{Map, Value, json};

use crate::debug::catalog;
use crate::debug::verb::Verb;

/// Every cataloged verb as a `tools/list` entry.
pub(super) fn list() -> Vec<Value> {
    catalog::all().map(descriptor).collect()
}

fn descriptor(verb: &Verb) -> Value {
    let mut tool = json!({
        "name": verb.name,
        "description": verb.description,
        "inputSchema": verb.schema(),
    });
    if verb.access.is_read_only() {
        tool["annotations"] = json!({ "readOnlyHint": true });
    }
    tool
}

/// The arguments one call carries, rejecting any shape a verb cannot take.
pub(super) fn arguments(value: Value) -> Result<Map<String, Value>, String> {
    match value {
        Value::Null => Ok(Map::new()),
        Value::Object(map) => Ok(map),
        _ => Err("arguments must be an object".to_string()),
    }
}

/// A tool result carrying one text block.
pub(super) fn text_result(text: &str, is_error: bool) -> Value {
    json!({
        "content": [{ "type": "text", "text": text }],
        "isError": is_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_cataloged_verb_becomes_one_tool() {
        let tools = list();
        assert_eq!(tools.len(), catalog::all().count());
        for (tool, verb) in tools.iter().zip(catalog::all()) {
            assert_eq!(tool["name"], verb.name);
            assert_eq!(tool["description"], verb.description);
            assert_eq!(tool["inputSchema"], verb.schema());
        }
    }

    #[test]
    fn only_read_only_verbs_carry_the_hint() {
        for (tool, verb) in list().iter().zip(catalog::all()) {
            let hinted = tool["annotations"]["readOnlyHint"] == json!(true);
            assert_eq!(hinted, verb.access.is_read_only(), "{}", verb.name);
        }
    }

    #[test]
    fn absent_arguments_are_an_empty_object() {
        assert_eq!(arguments(Value::Null), Ok(Map::new()));
    }

    #[test]
    fn non_object_arguments_are_rejected() {
        assert!(arguments(json!([1, 2])).is_err());
        assert!(arguments(json!("state")).is_err());
    }

    #[test]
    fn a_text_result_carries_one_block_and_its_error_flag() {
        let ok = text_result("hello", false);
        assert_eq!(ok["content"][0]["type"], "text");
        assert_eq!(ok["content"][0]["text"], "hello");
        assert_eq!(ok["isError"], json!(false));
        assert_eq!(text_result("nope", true)["isError"], json!(true));
    }
}
