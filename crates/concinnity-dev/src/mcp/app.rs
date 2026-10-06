//! The MCP server a running app serves on its debug port.
//!
//! Each call runs against the live world snapshot through the debug server's
//! verb table, and its reply becomes the one text block of the tool result.

use serde_json::{Map, Value, json};
use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};

use super::server::{Executor, Server};
use super::{http, tools};
use crate::debug::catalog;
use crate::debug::state::DebugState;

/// The MCP server behind one app's debug port. Shared across connections: each
/// answers one request and closes.
pub(crate) struct AppServer(Server<Dispatcher>);

impl AppServer {
    pub(crate) fn new(shared: Arc<Mutex<DebugState>>) -> Self {
        Self(Server::new(Dispatcher { shared }))
    }

    /// Answer one HTTP request read from `input`.
    pub(crate) fn serve<R: BufRead, W: Write>(
        &self,
        input: &mut R,
        output: &mut W,
    ) -> std::io::Result<()> {
        http::serve(&self.0, input, output)
    }
}

/// Runs each call against the world snapshot the debug hook maintains.
struct Dispatcher {
    shared: Arc<Mutex<DebugState>>,
}

impl Executor for Dispatcher {
    fn call(&self, name: &str, arguments: Map<String, Value>) -> Value {
        match catalog::run(name, arguments, &self.shared) {
            Ok(fields) => tools::text_result(&accepted(fields).to_string(), false),
            Err(error) => {
                let reply = json!({ "ok": false, "error": error });
                tools::text_result(&reply.to_string(), true)
            }
        }
    }
}

// A verb's reply fields, after the flag that says the engine accepted the call.
fn accepted(fields: Value) -> Value {
    let mut reply = Map::new();
    reply.insert("ok".to_string(), Value::Bool(true));
    if let Value::Object(fields) = fields {
        reply.extend(fields);
    }
    Value::Object(reply)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use serde_json::json;

    use super::*;

    fn snapshot() -> Arc<Mutex<DebugState>> {
        Arc::new(Mutex::new(DebugState::default()))
    }

    fn call(name: &str, arguments: Value) -> Value {
        let dispatcher = Dispatcher { shared: snapshot() };
        let arguments = tools::arguments(arguments).expect("valid arguments");
        dispatcher.call(name, arguments)
    }

    fn reply(result: &Value) -> Value {
        serde_json::from_str(result["content"][0]["text"].as_str().expect("text")).unwrap()
    }

    #[test]
    fn a_read_only_verb_answers_from_the_snapshot() {
        let result = call("state", Value::Null);
        assert_eq!(result["isError"], json!(false));
        let reply = reply(&result);
        assert_eq!(reply["ok"], json!(true));
        assert_eq!(reply["frame"], json!(0));
        // The flag leads, ahead of the verb's own fields.
        assert_eq!(
            reply.as_object().unwrap().keys().next().map(String::as_str),
            Some("ok")
        );
    }

    #[test]
    fn a_verb_the_world_cannot_serve_is_a_failed_call() {
        // A default snapshot holds no camera, so the verb refuses.
        let result = call("camera-get", Value::Null);
        assert_eq!(result["isError"], json!(true));
        let reply = reply(&result);
        assert_eq!(reply["ok"], json!(false));
        assert!(reply["error"].as_str().unwrap().contains("no Camera3D"));
    }

    #[test]
    fn arguments_reach_the_verb() {
        // The rejected `op` is only visible if the arguments, not just the
        // verb, reached the table.
        let result = call(
            "quality-set",
            json!({ "setting": "ssao", "op": "sideways" }),
        );
        assert_eq!(result["isError"], json!(true));
        assert!(
            result["content"][0]["text"]
                .as_str()
                .expect("text")
                .contains("sideways")
        );
    }

    #[test]
    fn a_call_over_the_transport_answers_from_the_same_snapshot() {
        let body = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"ping"}}"#;
        let request = format!(
            "POST /mcp HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let mut output = Vec::new();
        AppServer::new(snapshot())
            .serve(&mut Cursor::new(request), &mut output)
            .expect("serve");

        let response = String::from_utf8(output).expect("utf-8");
        let (head, body) = response.split_once("\r\n\r\n").expect("a header block");
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        let parsed: Value = serde_json::from_str(body).expect("a JSON-RPC response");
        assert_eq!(parsed["result"]["isError"], json!(false));
        assert!(
            parsed["result"]["content"][0]["text"]
                .as_str()
                .expect("text")
                .contains(r#""pong":true"#)
        );
    }
}
