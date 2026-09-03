//! The MCP server core: the methods this server answers, and what each answers
//! with.
//!
//! The executor is injected rather than opened here, so the whole protocol is
//! exercised in tests without a socket. `super::bridge` supplies the real one.

use serde_json::{Value, json};

use super::jsonrpc::{self, INVALID_PARAMS, Incoming, METHOD_NOT_FOUND};
use super::tools;
use crate::debug::catalog;

/// The protocol revision this server speaks, used when the client offers none.
const PROTOCOL_VERSION: &str = "2025-06-18";

/// Revisions this server also answers, kept because a client that offers one of
/// them is told its own version back rather than being asked to downgrade.
const SUPPORTED: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

const SERVER_NAME: &str = "concinnity";

/// A method failure: the JSON-RPC code and the message to report.
type Failure = (i64, String);

/// The protocol state machine. `execute` sends one debug-protocol request and
/// returns the server's raw reply, or a transport failure to report as one.
pub(super) struct Server<E> {
    execute: E,
}

impl<E: Fn(&str) -> Result<String, String>> Server<E> {
    pub(super) fn new(execute: E) -> Self {
        Self { execute }
    }

    /// Answer one incoming line, or `None` when the sender expects no reply.
    pub(super) fn handle(&self, line: &str) -> Option<Value> {
        match jsonrpc::parse(line) {
            Incoming::Call { id, method, params } => Some(match self.answer(&method, &params) {
                Ok(result) => jsonrpc::result(&id, result),
                Err((code, message)) => jsonrpc::error(&id, code, &message),
            }),
            Incoming::Invalid { id, code, message } => Some(jsonrpc::error(&id, code, &message)),
            Incoming::Silent => None,
        }
    }

    fn answer(&self, method: &str, params: &Value) -> Result<Value, Failure> {
        match method {
            "initialize" => Ok(initialize(params)),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": tools::list() })),
            "tools/call" => self.call(params),
            other => Err((METHOD_NOT_FOUND, format!("unknown method: {other}"))),
        }
    }

    fn call(&self, params: &Value) -> Result<Value, Failure> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| (INVALID_PARAMS, "tools/call needs a tool name".to_string()))?;
        if catalog::find(name).is_none() {
            return Err((INVALID_PARAMS, format!("unknown tool: {name}")));
        }
        let arguments = params.get("arguments").cloned().unwrap_or(Value::Null);
        let payload = tools::payload(name, &arguments).map_err(|e| (INVALID_PARAMS, e))?;

        Ok(match (self.execute)(&payload) {
            Ok(reply) => {
                let failed = !reply_ok(&reply);
                tool_result(&reply, failed)
            }
            Err(message) => tool_result(&message, true),
        })
    }
}

fn initialize(params: &Value) -> Value {
    let offered = params.get("protocolVersion").and_then(Value::as_str);
    let version = offered
        .filter(|v| SUPPORTED.contains(v))
        .unwrap_or(PROTOCOL_VERSION);
    json!({
        "protocolVersion": version,
        "capabilities": { "tools": {} },
        "serverInfo": { "name": SERVER_NAME, "version": env!("CARGO_PKG_VERSION") },
    })
}

// A reply the engine accepted carries `"ok": true`; anything else, an
// unparseable reply included, is reported to the client as a failed call.
fn reply_ok(reply: &str) -> bool {
    serde_json::from_str::<Value>(reply)
        .ok()
        .and_then(|v| v.get("ok").and_then(Value::as_bool))
        .unwrap_or(false)
}

fn tool_result(text: &str, is_error: bool) -> Value {
    json!({
        "content": [{ "type": "text", "text": text }],
        "isError": is_error,
    })
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    // A server whose executor records every payload it is handed and answers
    // with `reply`, so a test can assert what reached the wire without one.
    struct Fake {
        sent: RefCell<Vec<String>>,
        reply: Result<String, String>,
    }

    impl Fake {
        fn answering(reply: &str) -> Self {
            Self {
                sent: RefCell::new(Vec::new()),
                reply: Ok(reply.to_string()),
            }
        }

        fn failing(error: &str) -> Self {
            Self {
                sent: RefCell::new(Vec::new()),
                reply: Err(error.to_string()),
            }
        }

        fn server(&self) -> Server<impl Fn(&str) -> Result<String, String> + '_> {
            Server::new(move |payload: &str| {
                self.sent.borrow_mut().push(payload.to_string());
                self.reply.clone()
            })
        }
    }

    fn ok_server() -> Server<impl Fn(&str) -> Result<String, String>> {
        Server::new(|_: &str| Ok(r#"{"ok":true}"#.to_string()))
    }

    fn answer(server: &Server<impl Fn(&str) -> Result<String, String>>, line: &str) -> Value {
        server.handle(line).expect("a call is always answered")
    }

    #[test]
    fn initialize_reports_tools_and_this_server() {
        let reply = answer(
            &ok_server(),
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}"#,
        );
        let result = &reply["result"];
        assert_eq!(result["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(result["capabilities"]["tools"], json!({}));
        assert_eq!(result["serverInfo"]["name"], SERVER_NAME);
        assert_eq!(result["serverInfo"]["version"], env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn initialize_echoes_an_older_version_the_server_still_speaks() {
        let reply = answer(
            &ok_server(),
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}"#,
        );
        assert_eq!(reply["result"]["protocolVersion"], "2024-11-05");
    }

    #[test]
    fn initialize_falls_back_when_the_offered_version_is_unknown() {
        for params in [r#"{"protocolVersion":"1999-01-01"}"#, "{}"] {
            let line =
                format!(r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{params}}}"#);
            let reply = answer(&ok_server(), &line);
            assert_eq!(reply["result"]["protocolVersion"], PROTOCOL_VERSION);
        }
    }

    #[test]
    fn the_initialized_notification_is_not_answered() {
        assert!(
            ok_server()
                .handle(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#)
                .is_none()
        );
    }

    #[test]
    fn ping_answers_an_empty_result() {
        let reply = answer(&ok_server(), r#"{"jsonrpc":"2.0","id":2,"method":"ping"}"#);
        assert_eq!(reply["id"], json!(2));
        assert_eq!(reply["result"], json!({}));
    }

    #[test]
    fn tools_list_matches_the_catalog_one_to_one() {
        let reply = answer(
            &ok_server(),
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/list"}"#,
        );
        let tools = reply["result"]["tools"]
            .as_array()
            .expect("tools is an array");
        assert_eq!(tools.len(), catalog::all().len());
        for (tool, command) in tools.iter().zip(catalog::all()) {
            assert_eq!(tool["name"], command.name);
            assert_eq!(tool["description"], command.description);
            assert_eq!(tool["inputSchema"], command.schema());
        }
    }

    #[test]
    fn a_tool_call_forwards_the_verb_and_its_arguments() {
        let fake = Fake::answering(r#"{"ok":true,"id":4}"#);
        let reply = answer(
            &fake.server(),
            r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"decal-remove","arguments":{"id":4}}}"#,
        );

        let sent: Value = serde_json::from_str(&fake.sent.borrow()[0]).unwrap();
        assert_eq!(sent, json!({ "cmd": "decal-remove", "id": 4 }));
        assert_eq!(reply["result"]["isError"], json!(false));
        assert_eq!(reply["result"]["content"][0]["type"], "text");
        assert_eq!(
            reply["result"]["content"][0]["text"],
            r#"{"ok":true,"id":4}"#
        );
    }

    #[test]
    fn a_tool_call_without_arguments_sends_the_bare_verb() {
        let fake = Fake::answering(r#"{"ok":true,"pong":true}"#);
        answer(
            &fake.server(),
            r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"ping"}}"#,
        );
        assert_eq!(fake.sent.borrow()[0], r#"{"cmd":"ping"}"#);
    }

    #[test]
    fn a_rejected_command_is_an_error_result_not_a_protocol_error() {
        let fake = Fake::answering(r#"{"ok":false,"error":"no camera"}"#);
        let reply = answer(
            &fake.server(),
            r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"camera-get"}}"#,
        );
        assert!(reply.get("error").is_none());
        assert_eq!(reply["result"]["isError"], json!(true));
        assert!(
            reply["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("no camera")
        );
    }

    #[test]
    fn a_transport_failure_is_an_error_result_carrying_its_message() {
        let fake = Fake::failing("cannot connect to ws://127.0.0.1:8777");
        let reply = answer(
            &fake.server(),
            r#"{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"state"}}"#,
        );
        assert_eq!(reply["result"]["isError"], json!(true));
        assert!(
            reply["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("cannot connect")
        );
    }

    #[test]
    fn an_unparseable_reply_is_reported_as_a_failed_call() {
        let fake = Fake::answering("not json");
        let reply = answer(
            &fake.server(),
            r#"{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"state"}}"#,
        );
        assert_eq!(reply["result"]["isError"], json!(true));
    }

    #[test]
    fn an_unknown_tool_never_reaches_the_executor() {
        let fake = Fake::answering(r#"{"ok":true}"#);
        let reply = answer(
            &fake.server(),
            r#"{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"launch"}}"#,
        );
        assert_eq!(reply["error"]["code"], json!(INVALID_PARAMS));
        assert!(fake.sent.borrow().is_empty());
    }

    #[test]
    fn malformed_call_params_are_still_answered() {
        for params in ["{}", r#"{"name":"ping","arguments":[1]}"#, r#"{"name":7}"#] {
            let line =
                format!(r#"{{"jsonrpc":"2.0","id":11,"method":"tools/call","params":{params}}}"#);
            let reply = answer(&ok_server(), &line);
            assert_eq!(reply["id"], json!(11), "{params}");
            assert_eq!(reply["error"]["code"], json!(INVALID_PARAMS), "{params}");
        }
    }

    #[test]
    fn an_unknown_method_answers_with_the_same_id() {
        let reply = answer(
            &ok_server(),
            r#"{"jsonrpc":"2.0","id":"abc","method":"resources/list"}"#,
        );
        assert_eq!(reply["id"], json!("abc"));
        assert_eq!(reply["error"]["code"], json!(METHOD_NOT_FOUND));
    }

    #[test]
    fn unreadable_lines_answer_only_when_there_is_an_id() {
        assert!(ok_server().handle("{not json at all").is_some());
        assert!(ok_server().handle(r#"{"jsonrpc":"2.0"}"#).is_none());
    }
}
