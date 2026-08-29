// src/debug/protocol.rs
//
// Pure, socket-free helpers shared by the `cn debug` client commands: request
// payload validation, reply inspection, and the `watch` target enum. Kept out
// of the transport module (`super::wire::client`) so they stay unit-testable
// without a live server.

use serde_json::Value;

// Validate a raw `send` payload: it must be a JSON object carrying a "cmd"
// field. Returns the re-serialized object on success.
pub(super) fn validate_payload(json: &str) -> Result<String, String> {
    let value: Value = serde_json::from_str(json).map_err(|e| format!("invalid JSON: {e}"))?;
    let obj = value
        .as_object()
        .ok_or_else(|| "JSON must be an object with a \"cmd\" field".to_string())?;
    if !obj.contains_key("cmd") {
        return Err("JSON object must include a \"cmd\" field".to_string());
    }
    Ok(value.to_string())
}

// True when a reply carries `"ok": true`.
pub(super) fn reply_ok(reply: &Value) -> bool {
    reply.get("ok").and_then(Value::as_bool).unwrap_or(false)
}

/// A read-only snapshot the `watch` command can poll. Each maps to the matching
/// server command; `label` names it in the poll banner. Argv parsing lives in
/// the CLI binary, which maps its own value-enum onto this type.
#[derive(Clone, Copy, Debug)]
pub enum WatchTarget {
    /// The camera's live pose.
    Camera,
    /// The world's system and component state.
    State,
    /// Streaming residency counts and byte budgets.
    Streaming,
    /// Per-frame CPU and GPU timings.
    Profile,
}

impl WatchTarget {
    pub(super) fn cmd(self) -> &'static str {
        match self {
            WatchTarget::Camera => "camera-get",
            WatchTarget::State => "state",
            WatchTarget::Streaming => "streaming",
            WatchTarget::Profile => "profile",
        }
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            WatchTarget::Camera => "camera",
            WatchTarget::State => "state",
            WatchTarget::Streaming => "streaming",
            WatchTarget::Profile => "profile",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_payload_accepts_object_with_cmd() {
        let out = validate_payload(r#"{"cmd":"state"}"#).expect("valid payload");
        let value: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(value.get("cmd").and_then(Value::as_str), Some("state"));
    }

    #[test]
    fn validate_payload_preserves_extra_fields() {
        let out = validate_payload(r#"{"cmd":"decal-remove","id":7}"#).expect("valid payload");
        let value: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(value.get("id").and_then(Value::as_u64), Some(7));
    }

    #[test]
    fn validate_payload_rejects_invalid_json() {
        assert!(validate_payload("{not json").is_err());
    }

    #[test]
    fn validate_payload_rejects_non_object() {
        assert!(validate_payload(r#"["cmd","state"]"#).is_err());
        assert!(validate_payload(r#""state""#).is_err());
    }

    #[test]
    fn validate_payload_rejects_missing_cmd() {
        assert!(validate_payload(r#"{"id":1}"#).is_err());
    }

    #[test]
    fn reply_ok_reads_the_ok_field() {
        assert!(reply_ok(&serde_json::json!({ "ok": true })));
        assert!(!reply_ok(&serde_json::json!({ "ok": false })));
        assert!(!reply_ok(&serde_json::json!({ "pong": true })));
    }

    #[test]
    fn watch_target_maps_to_server_commands() {
        assert_eq!(WatchTarget::Camera.cmd(), "camera-get");
        assert_eq!(WatchTarget::State.cmd(), "state");
        assert_eq!(WatchTarget::Streaming.cmd(), "streaming");
        assert_eq!(WatchTarget::Profile.cmd(), "profile");
        assert_eq!(WatchTarget::Camera.label(), "camera");
        assert_eq!(WatchTarget::Profile.label(), "profile");
    }
}
