//! The executor: one debug-server round trip per tool call.
//!
//! Reuses the same localhost WebSocket client the `cn debug` subcommands drive,
//! so a tool call and a hand-typed `cn debug send` reach the engine by the one
//! path, with the same timeouts.

use crate::debug::client;

/// Send one debug-protocol request to the server on `port`, returning its raw
/// reply text.
pub(super) fn execute(port: u16, payload: &str) -> Result<String, String> {
    client::request_text(port, payload).map_err(|msg| explain(port, &msg))
}

// Nothing listening means no app is running, which is the one failure a client
// can act on, so name the commands that start one.
fn explain(port: u16, msg: &str) -> String {
    if msg.contains("cannot connect") {
        format!(
            "{msg}\nStart an app with `cn debug`, or `cn editor --debug-port {port}`, then retry."
        )
    } else {
        msg.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unreachable_server_names_the_commands_that_start_one() {
        let text = explain(8777, "cannot connect to ws://127.0.0.1:8777: refused");
        assert!(text.contains("cn debug"));
        assert!(text.contains("cn editor --debug-port 8777"));
    }

    #[test]
    fn other_failures_are_reported_as_they_are() {
        let msg = "timed out waiting for reply (>5s)";
        assert_eq!(explain(8777, msg), msg);
    }
}
