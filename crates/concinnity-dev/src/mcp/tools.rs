//! The MCP tool surface, built from the debug protocol's own verb catalog.
//!
//! Nothing here restates a verb: the name, the description, the read-only hint,
//! and the input schema all come from `crate::debug::catalog`, so a verb added
//! to the dispatcher becomes a tool without a second table to keep in step.

use serde_json::{Map, Value, json};

use crate::debug::catalog::{self, Command};

/// Every catalogued verb as a `tools/list` entry.
pub(super) fn list() -> Vec<Value> {
    catalog::all().iter().map(descriptor).collect()
}

fn descriptor(command: &Command) -> Value {
    let mut tool = json!({
        "name": command.name,
        "description": command.description,
        "inputSchema": command.schema(),
    });
    if command.access.is_read_only() {
        tool["annotations"] = json!({ "readOnlyHint": true });
    }
    tool
}

/// The arguments one call carries, rejecting any shape a verb body cannot take.
pub(super) fn arguments(value: &Value) -> Result<Map<String, Value>, String> {
    match value {
        Value::Null => Ok(Map::new()),
        Value::Object(map) => Ok(map.clone()),
        _ => Err("arguments must be an object".to_string()),
    }
}

/// The debug-protocol request one tool call carries.
pub(super) fn payload(name: &str, arguments: &Map<String, Value>) -> String {
    let mut body = arguments.clone();
    // Written last so an argument named "cmd" cannot redirect the call.
    body.insert("cmd".to_string(), Value::String(name.to_string()));
    Value::Object(body).to_string()
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

    fn payload_value(name: &str, args: &Value) -> Value {
        let args = arguments(args).expect("valid arguments");
        serde_json::from_str(&payload(name, &args)).unwrap()
    }

    #[test]
    fn every_catalogued_verb_becomes_one_tool() {
        let tools = list();
        assert_eq!(tools.len(), catalog::all().len());
        for (tool, command) in tools.iter().zip(catalog::all()) {
            assert_eq!(tool["name"], command.name);
            assert_eq!(tool["description"], command.description);
            assert_eq!(tool["inputSchema"], command.schema());
        }
    }

    #[test]
    fn only_read_only_verbs_carry_the_hint() {
        for (tool, command) in list().iter().zip(catalog::all()) {
            let hinted = tool["annotations"]["readOnlyHint"] == json!(true);
            assert_eq!(hinted, command.access.is_read_only(), "{}", command.name);
        }
    }

    #[test]
    fn a_payload_names_the_verb_and_carries_its_arguments() {
        let body = payload_value("decal-remove", &json!({ "id": 3 }));
        assert_eq!(body["cmd"], "decal-remove");
        assert_eq!(body["id"], json!(3));
    }

    #[test]
    fn absent_arguments_still_name_the_verb() {
        assert_eq!(
            payload_value("ping", &Value::Null),
            json!({ "cmd": "ping" })
        );
    }

    #[test]
    fn an_argument_named_cmd_cannot_redirect_the_call() {
        let body = payload_value("ping", &json!({ "cmd": "shutdown" }));
        assert_eq!(body["cmd"], "ping");
    }

    #[test]
    fn non_object_arguments_are_rejected() {
        assert!(arguments(&json!([1, 2])).is_err());
        assert!(arguments(&json!("state")).is_err());
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
