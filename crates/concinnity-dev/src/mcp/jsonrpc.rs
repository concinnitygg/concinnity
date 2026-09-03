//! The JSON-RPC 2.0 message layer: reading one incoming line and building the
//! response for it.
//!
//! Transport-agnostic on purpose. Nothing here knows about MCP methods, the
//! debug protocol, or stdio; `super::server` supplies the meaning and
//! `super::stdio` the framing.

use serde_json::{Value, json};

pub(super) const PARSE_ERROR: i64 = -32700;
pub(super) const INVALID_REQUEST: i64 = -32600;
pub(super) const METHOD_NOT_FOUND: i64 = -32601;
pub(super) const INVALID_PARAMS: i64 = -32602;

/// What one incoming line turned out to be.
pub(super) enum Incoming {
    /// A call that must be answered, whatever the outcome.
    Call {
        id: Value,
        method: String,
        params: Value,
    },
    /// Nothing to answer: a notification, or a malformed message carrying no
    /// id to answer with. Either way the sender expects no reply.
    Silent,
    /// Unreadable, but carrying an id to answer with.
    Invalid {
        id: Value,
        code: i64,
        message: String,
    },
}

/// Read one newline-delimited message.
pub(super) fn parse(line: &str) -> Incoming {
    let Ok(value) = serde_json::from_str::<Value>(line) else {
        return invalid(Value::Null, PARSE_ERROR, "invalid JSON");
    };
    let Some(object) = value.as_object() else {
        return invalid(Value::Null, INVALID_REQUEST, "request must be an object");
    };

    let id = match object.get("id") {
        None | Some(Value::Null) => None,
        Some(id) if id.is_string() || id.is_number() => Some(id.clone()),
        Some(_) => {
            return invalid(
                Value::Null,
                INVALID_REQUEST,
                "id must be a string or number",
            );
        }
    };

    let versioned = object.get("jsonrpc").and_then(Value::as_str) == Some("2.0");
    let method = object.get("method").and_then(Value::as_str);
    let params = object.get("params").cloned().unwrap_or(Value::Null);

    match (versioned, method, id) {
        (true, Some(method), Some(id)) => Incoming::Call {
            id,
            method: method.to_string(),
            params,
        },
        (true, Some(_), None) => Incoming::Silent,
        (_, _, Some(id)) => invalid(id, INVALID_REQUEST, "not a JSON-RPC 2.0 request"),
        (_, _, None) => Incoming::Silent,
    }
}

fn invalid(id: Value, code: i64, message: &str) -> Incoming {
    Incoming::Invalid {
        id,
        code,
        message: message.to_string(),
    }
}

/// A successful response carrying `result`.
pub(super) fn result(id: &Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

/// An error response. A request whose id could not be read answers with null.
pub(super) fn error(id: &Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call_parts(line: &str) -> (Value, String, Value) {
        match parse(line) {
            Incoming::Call { id, method, params } => (id, method, params),
            _ => panic!("expected a call: {line}"),
        }
    }

    fn invalid_parts(line: &str) -> (Value, i64) {
        match parse(line) {
            Incoming::Invalid { id, code, .. } => (id, code),
            _ => panic!("expected an invalid message: {line}"),
        }
    }

    #[test]
    fn a_call_carries_its_id_method_and_params() {
        let (id, method, params) = call_parts(r#"{"jsonrpc":"2.0","id":7,"method":"ping"}"#);
        assert_eq!(id, json!(7));
        assert_eq!(method, "ping");
        assert_eq!(params, Value::Null);
    }

    #[test]
    fn a_string_id_is_kept_as_a_string() {
        let (id, _, _) = call_parts(r#"{"jsonrpc":"2.0","id":"a1","method":"ping"}"#);
        assert_eq!(id, json!("a1"));
    }

    #[test]
    fn params_are_passed_through_untouched() {
        let (_, _, params) = call_parts(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"ping"}}"#,
        );
        assert_eq!(params, json!({ "name": "ping" }));
    }

    #[test]
    fn a_message_without_an_id_is_a_notification() {
        assert!(matches!(
            parse(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#),
            Incoming::Silent
        ));
    }

    #[test]
    fn an_explicit_null_id_is_a_notification() {
        assert!(matches!(
            parse(r#"{"jsonrpc":"2.0","id":null,"method":"ping"}"#),
            Incoming::Silent
        ));
    }

    #[test]
    fn unparseable_input_is_a_parse_error_with_a_null_id() {
        let (id, code) = invalid_parts("{not json");
        assert_eq!(id, Value::Null);
        assert_eq!(code, PARSE_ERROR);
    }

    #[test]
    fn a_non_object_message_is_an_invalid_request() {
        assert_eq!(invalid_parts("[1,2,3]").1, INVALID_REQUEST);
        assert_eq!(invalid_parts("\"ping\"").1, INVALID_REQUEST);
    }

    #[test]
    fn a_wrong_version_answers_with_the_same_id() {
        let (id, code) = invalid_parts(r#"{"jsonrpc":"1.0","id":4,"method":"ping"}"#);
        assert_eq!(id, json!(4));
        assert_eq!(code, INVALID_REQUEST);
    }

    #[test]
    fn a_missing_method_answers_with_the_same_id() {
        let (id, code) = invalid_parts(r#"{"jsonrpc":"2.0","id":4}"#);
        assert_eq!(id, json!(4));
        assert_eq!(code, INVALID_REQUEST);
    }

    #[test]
    fn a_structured_id_is_rejected_rather_than_echoed() {
        let (id, code) = invalid_parts(r#"{"jsonrpc":"2.0","id":{"a":1},"method":"ping"}"#);
        assert_eq!(id, Value::Null);
        assert_eq!(code, INVALID_REQUEST);
    }

    #[test]
    fn a_malformed_message_with_no_id_is_dropped() {
        assert!(matches!(parse(r#"{"jsonrpc":"2.0"}"#), Incoming::Silent));
        assert!(matches!(parse(r#"{"method":"ping"}"#), Incoming::Silent));
    }

    #[test]
    fn responses_carry_the_protocol_version_and_the_id() {
        let ok = result(&json!(3), json!({ "a": 1 }));
        assert_eq!(ok["jsonrpc"], "2.0");
        assert_eq!(ok["id"], json!(3));
        assert_eq!(ok["result"], json!({ "a": 1 }));

        let bad = error(&json!("x"), METHOD_NOT_FOUND, "nope");
        assert_eq!(bad["id"], json!("x"));
        assert_eq!(bad["error"]["code"], json!(METHOD_NOT_FOUND));
        assert_eq!(bad["error"]["message"], "nope");
        assert!(bad.get("result").is_none());
    }
}
