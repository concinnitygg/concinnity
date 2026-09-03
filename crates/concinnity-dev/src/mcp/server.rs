//! The MCP server core: the methods this server answers, and what each answers
//! with.
//!
//! `initialize`, `tools/list` and `ping` are answered from the verb catalog
//! alone, so both ends serve them locally. Only `tools/call` needs the running
//! world, and that is the one thing the injected executor carries out: the app
//! runs it against its own snapshot, the stdio bridge forwards it. The
//! injection is also what makes the whole protocol testable without a socket.

use serde_json::{Map, Value, json};

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

/// How one `tools/call` is carried out, once the server has checked that the
/// verb exists and its arguments are an object.
pub(super) trait Executor {
    /// The `tools/call` result for a catalogued verb, failures included: a
    /// rejected command is an error result, never a protocol error.
    fn call(&self, name: &str, arguments: &Map<String, Value>) -> Value;
}

/// The protocol state machine.
pub(super) struct Server<E> {
    execute: E,
}

impl<E: Executor> Server<E> {
    pub(super) fn new(execute: E) -> Self {
        Self { execute }
    }

    /// Answer one message, or `None` when the sender expects no reply.
    pub(super) fn handle(&self, message: &str) -> Option<Value> {
        match jsonrpc::parse(message) {
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
        let arguments = tools::arguments(&arguments).map_err(|e| (INVALID_PARAMS, e))?;
        Ok(self.execute.call(name, &arguments))
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

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    // An executor that records every call it is handed and answers with a
    // fixed result, so a test can assert what reached it without a transport.
    struct Fake {
        seen: RefCell<Vec<Value>>,
        result: Value,
    }

    impl Fake {
        fn answering(result: Value) -> Self {
            Self {
                seen: RefCell::new(Vec::new()),
                result,
            }
        }
    }

    impl Executor for &Fake {
        fn call(&self, name: &str, arguments: &Map<String, Value>) -> Value {
            self.seen.borrow_mut().push(json!({
                "name": name,
                "arguments": Value::Object(arguments.clone()),
            }));
            self.result.clone()
        }
    }

    // A server whose calls all succeed, for the paths a call never reaches.
    struct Always;

    impl Executor for Always {
        fn call(&self, _name: &str, _arguments: &Map<String, Value>) -> Value {
            tools::text_result(r#"{"ok":true}"#, false)
        }
    }

    fn ok_server() -> Server<Always> {
        Server::new(Always)
    }

    fn answer<E: Executor>(server: &Server<E>, message: &str) -> Value {
        server.handle(message).expect("a call is always answered")
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
        let fake = Fake::answering(tools::text_result(r#"{"ok":true,"id":4}"#, false));
        let reply = answer(
            &Server::new(&fake),
            r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"decal-remove","arguments":{"id":4}}}"#,
        );

        assert_eq!(
            fake.seen.borrow()[0],
            json!({ "name": "decal-remove", "arguments": { "id": 4 } })
        );
        assert_eq!(reply["result"]["isError"], json!(false));
        assert_eq!(
            reply["result"]["content"][0]["text"],
            r#"{"ok":true,"id":4}"#
        );
    }

    #[test]
    fn a_tool_call_without_arguments_carries_an_empty_object() {
        let fake = Fake::answering(tools::text_result(r#"{"ok":true}"#, false));
        answer(
            &Server::new(&fake),
            r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"ping"}}"#,
        );
        assert_eq!(
            fake.seen.borrow()[0],
            json!({ "name": "ping", "arguments": {} })
        );
    }

    #[test]
    fn an_executor_failure_is_an_error_result_not_a_protocol_error() {
        let fake = Fake::answering(tools::text_result("cannot connect", true));
        let reply = answer(
            &Server::new(&fake),
            r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"state"}}"#,
        );
        assert!(reply.get("error").is_none());
        assert_eq!(reply["result"]["isError"], json!(true));
        assert_eq!(reply["result"]["content"][0]["text"], "cannot connect");
    }

    #[test]
    fn an_unknown_tool_never_reaches_the_executor() {
        let fake = Fake::answering(tools::text_result(r#"{"ok":true}"#, false));
        let reply = answer(
            &Server::new(&fake),
            r#"{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"launch"}}"#,
        );
        assert_eq!(reply["error"]["code"], json!(INVALID_PARAMS));
        assert!(fake.seen.borrow().is_empty());
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
    fn unreadable_messages_answer_only_when_there_is_an_id() {
        assert!(ok_server().handle("{not json at all").is_some());
        assert!(ok_server().handle(r#"{"jsonrpc":"2.0"}"#).is_none());
    }
}
