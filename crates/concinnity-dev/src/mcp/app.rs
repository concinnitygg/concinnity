//! The MCP server a running app serves on its debug port.
//!
//! Each call runs against the live world snapshot through the same socket-free
//! dispatcher the verb catalog describes, so `{"cmd": ...}` is an internal seam
//! between the tool surface and the dispatcher rather than a wire format.

use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};

use serde_json::{Map, Value};

use super::server::{Executor, Server};
use super::{http, tools};
use crate::debug::dispatch::handle_request;
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
    fn call(&self, name: &str, arguments: &Map<String, Value>) -> Value {
        let reply = handle_request(&tools::payload(name, arguments), &self.shared);
        tools::text_result(&reply, !accepted(&reply))
    }
}

// A reply the engine accepted carries `"ok": true`; anything else, an
// unparseable reply included, is reported to the client as a failed call.
fn accepted(reply: &str) -> bool {
    serde_json::from_str::<Value>(reply)
        .ok()
        .and_then(|v| v.get("ok").and_then(Value::as_bool))
        .unwrap_or(false)
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
        let arguments = tools::arguments(&arguments).expect("valid arguments");
        dispatcher.call(name, &arguments)
    }

    #[test]
    fn a_read_only_verb_answers_from_the_snapshot() {
        let result = call("state", Value::Null);
        assert_eq!(result["isError"], json!(false));
        let reply: Value =
            serde_json::from_str(result["content"][0]["text"].as_str().expect("text")).unwrap();
        assert_eq!(reply["ok"], json!(true));
        assert_eq!(reply["frame"], json!(0));
    }

    #[test]
    fn a_verb_the_world_cannot_serve_is_a_failed_call() {
        // A default snapshot holds no camera, so the dispatcher rejects it.
        let result = call("camera-get", Value::Null);
        assert_eq!(result["isError"], json!(true));
    }

    #[test]
    fn arguments_reach_the_dispatcher_as_the_request_body() {
        // The rejected `op` is only visible if the body, not just the verb,
        // reached the dispatcher.
        let result = call(
            "quality-set",
            json!({ "setting": "shadows", "op": "sideways" }),
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

    #[test]
    fn an_unparseable_reply_is_a_failed_call() {
        assert!(!accepted("not json"));
        assert!(!accepted(r#"{"ok":false}"#));
        assert!(accepted(r#"{"ok":true}"#));
    }
}
