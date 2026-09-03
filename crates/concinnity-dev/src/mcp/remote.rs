//! The client half of the debug transport: one JSON-RPC message posted to a
//! running app's MCP endpoint.
//!
//! One request per connection, and every connect and read bounded, so a gone or
//! wedged app surfaces a clear error instead of hanging the caller.

use std::io::ErrorKind;
use std::time::Duration;

use serde_json::Value;

// Bound the connect so a missing app fails fast instead of hanging.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
// Bound the reply so a wedged app surfaces a timeout, not a hang.
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(5);

/// The endpoint an app serves MCP on.
pub(super) fn endpoint(port: u16) -> String {
    format!("http://127.0.0.1:{port}/mcp")
}

/// Post one JSON-RPC message and return the response it was answered with.
pub(super) fn post(port: u16, message: &Value) -> Result<Value, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_recv_response(Some(RESPONSE_TIMEOUT))
        .timeout_recv_body(Some(RESPONSE_TIMEOUT))
        .build()
        .into();

    let url = endpoint(port);
    let mut response = agent
        .post(&url)
        .header("Content-Type", "application/json")
        .send(message.to_string())
        .map_err(|e| describe(port, &e))?;
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("cannot read the reply from {url}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("malformed reply from {url}: {e}"))
}

// Name the two failures a caller can act on -- nothing listening, and an app
// that stopped answering -- and pass everything else through.
fn describe(port: u16, error: &ureq::Error) -> String {
    let url = endpoint(port);
    match error {
        ureq::Error::Io(io)
            if matches!(
                io.kind(),
                ErrorKind::ConnectionRefused | ErrorKind::ConnectionReset
            ) =>
        {
            format!("cannot connect to {url}: {io}")
        }
        ureq::Error::Timeout(_) => format!(
            "timed out waiting for {url} (>{}s)",
            RESPONSE_TIMEOUT.as_secs()
        ),
        other => format!("request to {url} failed: {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_endpoint_is_loopback_only() {
        assert_eq!(endpoint(8777), "http://127.0.0.1:8777/mcp");
    }

    #[test]
    fn nothing_listening_is_reported_as_a_failure_to_connect() {
        let refused = ureq::Error::Io(ErrorKind::ConnectionRefused.into());
        let text = describe(8777, &refused);
        assert!(text.contains("cannot connect"), "{text}");
        assert!(text.contains("http://127.0.0.1:8777/mcp"), "{text}");
    }

    #[test]
    fn other_io_failures_keep_their_own_wording() {
        let broken = ureq::Error::Io(ErrorKind::BrokenPipe.into());
        assert!(describe(8777, &broken).contains("failed"));
    }
}
