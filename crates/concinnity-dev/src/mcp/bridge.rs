//! The stdio bridge's executor: one `tools/call` forwarded to a running app.
//!
//! Everything else the bridge answers itself from the verb catalog, so a client
//! still connects, lists the tools, and pings when no app is running. Only a
//! call needs a live world, and an app that is not there is reported as a failed
//! call naming the commands that start one.

use serde_json::{Map, Value, json};

use super::server::Executor;
use super::{remote, tools};

/// Forwards every call to the app serving MCP on `port`.
pub(super) struct Forward {
    port: u16,
}

impl Forward {
    pub(super) fn new(port: u16) -> Self {
        Self { port }
    }
}

impl Executor for Forward {
    fn call(&self, name: &str, arguments: &Map<String, Value>) -> Value {
        let message = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": { "name": name, "arguments": Value::Object(arguments.clone()) },
        });
        match remote::post(self.port, &message) {
            Ok(response) => unwrap_result(&response),
            Err(message) => tools::text_result(&explain(self.port, &message), true),
        }
    }
}

// The app answers a call with a tool result, which is returned as it stands. A
// protocol error instead means the app rejected the call outright, so report
// its message as a failed call rather than an empty success.
fn unwrap_result(response: &Value) -> Value {
    if let Some(result) = response.get("result") {
        return result.clone();
    }
    let message = response
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("the app answered with neither a result nor an error");
    tools::text_result(message, true)
}

// Nothing listening means no app is running, which is the one failure a client
// can act on, so name the commands that start one.
fn explain(port: u16, message: &str) -> String {
    if message.contains("cannot connect") {
        format!(
            "{message}\nStart an app with `cn debug`, or `cn editor --debug-port {port}`, then retry."
        )
    } else {
        message.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unreachable_app_names_the_commands_that_start_one() {
        let text = explain(8777, "cannot connect to http://127.0.0.1:8777/mcp: refused");
        assert!(text.contains("cn debug"));
        assert!(text.contains("cn editor --debug-port 8777"));
    }

    #[test]
    fn other_failures_are_reported_as_they_are() {
        let message = "timed out waiting for http://127.0.0.1:8777/mcp (>5s)";
        assert_eq!(explain(8777, message), message);
    }

    #[test]
    fn the_apps_result_is_returned_as_it_stands() {
        let result =
            json!({ "content": [{ "type": "text", "text": "{\"ok\":true}" }], "isError": false });
        let response = json!({ "jsonrpc": "2.0", "id": 1, "result": result });
        assert_eq!(unwrap_result(&response), result);
    }

    #[test]
    fn a_protocol_error_becomes_a_failed_call_carrying_its_message() {
        let response = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "error": { "code": -32602, "message": "unknown tool: launch" },
        });
        let result = unwrap_result(&response);
        assert_eq!(result["isError"], json!(true));
        assert_eq!(result["content"][0]["text"], "unknown tool: launch");
    }

    #[test]
    fn a_reply_that_is_neither_is_still_a_failed_call() {
        let result = unwrap_result(&json!({ "jsonrpc": "2.0", "id": 1 }));
        assert_eq!(result["isError"], json!(true));
    }
}
