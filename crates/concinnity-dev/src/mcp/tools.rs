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

/// The debug-protocol request one tool call forwards.
pub(super) fn payload(name: &str, arguments: &Value) -> Result<String, String> {
    let mut body = match arguments {
        Value::Null => Map::new(),
        Value::Object(map) => map.clone(),
        _ => return Err("arguments must be an object".to_string()),
    };
    // Written last so an argument named "cmd" cannot redirect the call.
    body.insert("cmd".to_string(), Value::String(name.to_string()));
    Ok(Value::Object(body).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload_value(name: &str, arguments: &Value) -> Value {
        serde_json::from_str(&payload(name, arguments).expect("valid arguments")).unwrap()
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
        assert!(payload("ping", &json!([1, 2])).is_err());
        assert!(payload("ping", &json!("state")).is_err());
    }
}
