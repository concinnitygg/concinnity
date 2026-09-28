//! Runtime spawn / crossfade command handlers (`decal-add`,
//! `emitter-add`, `anim-crossfade`, …) plus their request-body structs and the
//! shared `error_reply` helper. Each parses its JSON body, pushes a command onto
//! the debug server's `RuntimeQueue`, and blocks on a one-shot reply channel the
//! per-frame debug drive fulfils. The query commands + dispatch live in
//! `super::dispatch::handle_request`.

use concinnity_core::components::InputKey;
use concinnity_core::components::SettingOp;
use concinnity_core::components::StoryCommand;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::input::keymap::Bindable;
use concinnity_core::settings::SettingKey;

use super::anim_command::AnimCommand;
use super::runtime_spawn::{
    BackendCommand, CameraMoveArgs, CameraSetArgs, DecalSpawnArgs, EmitterSpawnArgs,
    RuntimeCommand, RuntimeQueue, WorldCommand,
};

// Maximum wait for the per-frame debug drive to apply a runtime command and
// reply. The drive runs once per frame, so a healthy 60 Hz engine replies
// inside ~16 ms; 1 s gives plenty of headroom even on a slow boot frame (4K
// HDR bake) without leaving an MCP client hanging forever if the engine has
// stalled.
const SPAWN_REPLY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1);

// Run one runtime command end to end: open a one-shot reply channel, build the
// command around its sender and push it onto `queue`, then block up to
// `timeout` for the debug drive to answer. The engine's `Result<T, String>`
// reply becomes JSON via `on_ok` on success, or the shared `error_reply` on an
// engine error or a timeout (tagged with `label`, the command name).
fn run_with_reply<T, C: Into<RuntimeCommand>>(
    queue: &RuntimeQueue,
    label: &str,
    timeout: std::time::Duration,
    command: impl FnOnce(std::sync::mpsc::SyncSender<Result<T, String>>) -> C,
    on_ok: impl FnOnce(T) -> String,
) -> String {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    queue.enqueue(command(tx));
    match rx.recv_timeout(timeout) {
        Ok(Ok(value)) => on_ok(value),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply(&format!("{label}: timed out waiting for engine")),
    }
}

// Parse a command's JSON body, tagging a decode failure with `label` (the
// command name) as a ready-to-return `error_reply` string. The transport's
// `"cmd"` key is stripped first, so every other key must be a declared field.
fn parse_request<T: serde::de::DeserializeOwned>(label: &str, text: &str) -> Result<T, String> {
    let fail = |e: serde_json::Error| error_reply(&format!("{label}: {e}"));
    let mut body: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(text).map_err(fail)?;
    body.remove("cmd");
    serde_json::from_value(serde_json::Value::Object(body)).map_err(fail)
}

pub(super) fn handle_decal_add(queue: &RuntimeQueue, text: &str) -> String {
    let args: DecalSpawnArgs = match parse_request("decal-add", text) {
        Ok(r) => r,
        Err(reply) => return reply,
    };
    run_with_reply(
        queue,
        "decal-add",
        SPAWN_REPLY_TIMEOUT,
        |reply| BackendCommand::DecalAdd { args, reply },
        |id| serde_json::json!({ "ok": true, "id": id }).to_string(),
    )
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct IdRequest {
    id: usize,
}

pub(super) fn handle_decal_remove(queue: &RuntimeQueue, text: &str) -> String {
    let req: IdRequest = match parse_request("decal-remove", text) {
        Ok(r) => r,
        Err(reply) => return reply,
    };
    run_with_reply(
        queue,
        "decal-remove",
        SPAWN_REPLY_TIMEOUT,
        |reply| BackendCommand::DecalRemove { id: req.id, reply },
        |()| serde_json::json!({ "ok": true, "removed": true }).to_string(),
    )
}

pub(super) fn handle_emitter_add(queue: &RuntimeQueue, text: &str) -> String {
    let args: EmitterSpawnArgs = match parse_request("emitter-add", text) {
        Ok(r) => r,
        Err(reply) => return reply,
    };
    run_with_reply(
        queue,
        "emitter-add",
        SPAWN_REPLY_TIMEOUT,
        |reply| BackendCommand::EmitterAdd { args, reply },
        |id| serde_json::json!({ "ok": true, "id": id }).to_string(),
    )
}

pub(super) fn handle_emitter_remove(queue: &RuntimeQueue, text: &str) -> String {
    let req: IdRequest = match parse_request("emitter-remove", text) {
        Ok(r) => r,
        Err(reply) => return reply,
    };
    run_with_reply(
        queue,
        "emitter-remove",
        SPAWN_REPLY_TIMEOUT,
        |reply| BackendCommand::EmitterRemove { id: req.id, reply },
        |()| serde_json::json!({ "ok": true, "removed": true }).to_string(),
    )
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AnimCrossfadeRequest {
    #[serde(default)]
    target: String,
    #[serde(default)]
    weights: Vec<f32>,
    #[serde(default)]
    duration_secs: f32,
}

pub(super) fn handle_anim_crossfade(queue: &RuntimeQueue, text: &str, names: &[String]) -> String {
    let req: AnimCrossfadeRequest = match parse_request("anim-crossfade", text) {
        Ok(r) => r,
        Err(reply) => return reply,
    };
    let target = match resolve_target("anim-crossfade", &req.target, names) {
        Ok(t) => t,
        Err(e) => return error_reply(&e),
    };
    run_with_reply(
        queue,
        "anim-crossfade",
        SPAWN_REPLY_TIMEOUT,
        |reply| AnimCommand::Crossfade {
            target,
            weights: req.weights,
            duration_secs: req.duration_secs,
            reply,
        },
        |()| serde_json::json!({ "ok": true, "queued": true }).to_string(),
    )
}

// Resolve a `target` asset name against the interner names table (indexed by
// `AssetId`; a small linear scan is fine for a debug command that fires at
// most a few times per second).
fn resolve_target(cmd: &str, target: &str, names: &[String]) -> Result<AssetId, String> {
    if target.is_empty() {
        return Err(format!("{cmd}: missing 'target'"));
    }
    names
        .iter()
        .position(|n| n == target)
        .map(|idx| AssetId(idx as u32))
        .ok_or_else(|| format!("{cmd}: unknown asset name '{target}'"))
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AnimParamRequest {
    #[serde(default)]
    target: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    value: f32,
}

pub(super) fn handle_anim_param(queue: &RuntimeQueue, text: &str, names: &[String]) -> String {
    let req: AnimParamRequest = match parse_request("anim-param", text) {
        Ok(r) => r,
        Err(reply) => return reply,
    };
    if req.name.is_empty() {
        return error_reply("anim-param: missing 'name' (a graph parameter)");
    }
    let target = match resolve_target("anim-param", &req.target, names) {
        Ok(t) => t,
        Err(e) => return error_reply(&e),
    };
    run_with_reply(
        queue,
        "anim-param",
        SPAWN_REPLY_TIMEOUT,
        |reply| AnimCommand::SetParam {
            target,
            name: req.name,
            value: req.value,
            reply,
        },
        |()| serde_json::json!({ "ok": true, "queued": true }).to_string(),
    )
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AnimStateRequest {
    #[serde(default)]
    target: String,
}

pub(super) fn handle_anim_state(queue: &RuntimeQueue, text: &str, names: &[String]) -> String {
    let req: AnimStateRequest = match parse_request("anim-state", text) {
        Ok(r) => r,
        Err(reply) => return reply,
    };
    let target = match resolve_target("anim-state", &req.target, names) {
        Ok(t) => t,
        Err(e) => return error_reply(&e),
    };
    run_with_reply(
        queue,
        "anim-state",
        SPAWN_REPLY_TIMEOUT,
        |reply| AnimCommand::QueryState { target, reply },
        |report| {
            let params: serde_json::Map<String, serde_json::Value> = report
                .params
                .into_iter()
                .map(|(name, value)| (name, serde_json::json!(value)))
                .collect();
            serde_json::json!({
                "ok": true,
                "state": report.state,
                "clock_secs": report.clock_secs,
                "fading_from": report.fading_from,
                "fade_progress": report.fade_progress,
                "blend_weights": report.blend_weights,
                "params": params,
            })
            .to_string()
        },
    )
}

// Longer than the spawn timeout: the capture idles the GPU, copies the
// swapchain image back, and PNG-encodes + writes it on the render thread, which
// can take noticeably longer than a simple state mutation.
const SCREENSHOT_REPLY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ScreenshotRequest {
    #[serde(default)]
    path: String,
}

pub(super) fn handle_screenshot(queue: &RuntimeQueue, text: &str) -> String {
    let req: ScreenshotRequest = match parse_request("screenshot", text) {
        Ok(r) => r,
        Err(reply) => return reply,
    };
    let path = req.path.trim().to_string();
    if path.is_empty() {
        return error_reply("screenshot: missing 'path'");
    }
    if !std::path::Path::new(&path)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("png"))
    {
        return error_reply("screenshot: 'path' must end in .png");
    }
    run_with_reply(
        queue,
        "screenshot",
        SCREENSHOT_REPLY_TIMEOUT,
        |reply| BackendCommand::Screenshot { path, reply },
        |path| serde_json::json!({ "ok": true, "path": path }).to_string(),
    )
}

// Read back the GPU cull's per-object status buffer and report the outcome
// histogram. Takes no parameters: the reply is the whole live cull list tallied
// by `concinnity_core::gfx::cull_status`.
//
// The readback idles the device, so it gets the screenshot timeout rather than
// the one-frame spawn timeout.
pub(super) fn handle_cull_status(queue: &RuntimeQueue) -> String {
    run_with_reply(
        queue,
        "cull-status",
        SCREENSHOT_REPLY_TIMEOUT,
        |reply| BackendCommand::CullStatus { reply },
        |raw| cull_status_reply(&raw),
    )
}

// Shape one raw cull-status readback into the command's JSON reply. Split out
// from the queueing so the reply shape is testable without an engine.
fn cull_status_reply(raw: &[u32]) -> String {
    let c = concinnity_core::gfx::cull_status::tally(raw);
    serde_json::json!({
        "ok": true,
        "objects": c.total(),
        "drawn": c.drawn,
        "frustum_culled": c.frustum_culled,
        "hiz_candidate": c.hiz_candidate,
        "hiz_culled": c.hiz_culled,
        "redrawn": c.redrawn,
        "unknown": c.unknown,
        // The two derived numbers a Hi-Z A/B actually compares: everything the
        // cull let through, and everything the Hi-Z test rejected across both
        // phases.
        "visible": c.visible(),
        "hiz_rejected": c.hiz_rejected(),
    })
    .to_string()
}

// Teleport the active camera. `position` / `yaw` / `pitch` are required in
// practice (the probe always sends them); missing fields fall back to zero,
// matching the decal / emitter request shape. `yaw` / `pitch` are radians;
// `fov_y_degrees` is omitted to leave the camera's FOV untouched.
pub(super) fn handle_camera_set(queue: &RuntimeQueue, text: &str) -> String {
    let args: CameraSetArgs = match parse_request("camera-set", text) {
        Ok(r) => r,
        Err(reply) => return reply,
    };
    run_with_reply(
        queue,
        "camera-set",
        SPAWN_REPLY_TIMEOUT,
        |reply| WorldCommand::CameraSet { args, reply },
        |()| serde_json::json!({ "ok": true, "set": true }).to_string(),
    )
}

// Flip a quality feature toggle. `setting` is one of the toggle keys (ssao / ssr
// / ray_traced_reflections / ssgi / auto_exposure); `op` cycles it (next | prev,
// both flip the toggle). Defaults match the decal / camera request shape.
#[derive(serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
struct QualitySetRequest {
    setting: String,
    op: String,
}

impl Default for QualitySetRequest {
    fn default() -> Self {
        Self {
            setting: String::new(),
            op: "next".to_string(),
        }
    }
}

// Toggle a Quality-group setting live by injecting the same `SettingCommand`
// the settings menu emits, so the engine runs its real `apply_quality_settings`
// rebuild. `cn debug` only; lets a headless harness exercise the live toggle
// path and screenshot the result. The reply fires once the command is queued
// (the GraphicsSystem applies it on its next step, before the next present).
pub(super) fn handle_quality_set(queue: &RuntimeQueue, text: &str) -> String {
    let req: QualitySetRequest = match parse_request("quality-set", text) {
        Ok(r) => r,
        Err(reply) => return reply,
    };
    if req.setting.is_empty() {
        return error_reply("quality-set: missing 'setting'");
    }
    let op = match req.op.as_str() {
        "next" | "" => SettingOp::Next,
        "prev" => SettingOp::Prev,
        other => {
            return error_reply(&format!(
                "quality-set: unknown op '{other}' (use next | prev)"
            ));
        }
    };
    let Some(setting) = SettingKey::parse(&req.setting).filter(|key| key.is_quality_toggle())
    else {
        return error_reply(&format!(
            "quality-set: '{}' is not a quality toggle (use {})",
            req.setting,
            super::catalog::QUALITY_TOGGLE_NAMES.join(" | ")
        ));
    };
    run_with_reply(
        queue,
        "quality-set",
        SPAWN_REPLY_TIMEOUT,
        |reply| WorldCommand::QualitySet { setting, op, reply },
        |()| serde_json::json!({ "ok": true, "queued": true }).to_string(),
    )
}

// Rebind a movement action to a key. `setting` is the engine key
// (`key_forward` / `key_backward` / `key_left` / `key_right` / `key_sprint` /
// `key_jump` / `key_interact`); `key` is a canonical `InputKey` variant name
// (`W`, `Space`, `Shift`, `Num1`, `Up`, ...).
#[derive(serde::Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct RebindRequest {
    setting: String,
    key: String,
}

// Rebind a movement key live by injecting the same `Rebind` `SettingCommand` the
// settings menu emits on a capture, so the engine runs its real swap + persist +
// `set_keymap` path. `cn debug` only; lets a headless harness exercise the live
// rebind and screenshot the row label flipping. The reply fires once queued.
pub(super) fn handle_rebind(queue: &RuntimeQueue, text: &str) -> String {
    let req: RebindRequest = match parse_request("rebind", text) {
        Ok(r) => r,
        Err(reply) => return reply,
    };
    if req.setting.is_empty() {
        return error_reply("rebind: missing 'setting' (e.g. key_forward)");
    }
    let Some(SettingKey::KeyRebind(action)) = SettingKey::parse(&req.setting) else {
        return error_reply(&format!(
            "rebind: '{}' is not a key rebind (use {})",
            req.setting,
            Bindable::ALL.map(Bindable::setting_key).join(" | ")
        ));
    };
    // The canonical `InputKey` serializes to its variant name, so a JSON string
    // deserializes straight to it (W, Space, Shift, Num1, Up, ...).
    let key: InputKey = match serde_json::from_value(serde_json::Value::String(req.key.clone())) {
        Ok(k) => k,
        Err(_) => {
            return error_reply(&format!(
                "rebind: unknown key '{}' (use a InputKey variant like W / Space / Shift)",
                req.key
            ));
        }
    };
    run_with_reply(
        queue,
        "rebind",
        SPAWN_REPLY_TIMEOUT,
        |reply| WorldCommand::Rebind { action, key, reply },
        |()| serde_json::json!({ "ok": true, "queued": true }).to_string(),
    )
}

// Despawn an authored placement by its declared name.
#[derive(serde::Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct DespawnCmdRequest {
    target: String,
}

// Remove an authored placement (and its descendants) live by name: enqueue a
// Despawn command the per-frame drive forwards to `dispatch_despawn`, which
// sends a `DespawnRequest` the GraphicsSystem applies on its next step (hide the
// draw slots + despawn the entity, cascading to children). `cn debug` only; lets
// a headless harness remove an entity and screenshot it gone. The reply fires
// once the command is queued.
pub(super) fn handle_despawn(queue: &RuntimeQueue, text: &str) -> String {
    let req: DespawnCmdRequest = match parse_request("despawn", text) {
        Ok(r) => r,
        Err(reply) => return reply,
    };
    if req.target.trim().is_empty() {
        return error_reply("despawn: missing 'target'");
    }
    run_with_reply(
        queue,
        "despawn",
        SPAWN_REPLY_TIMEOUT,
        |reply| WorldCommand::Despawn {
            name: req.target,
            reply,
        },
        |()| serde_json::json!({ "ok": true, "queued": true }).to_string(),
    )
}

// Drive the story system: `action` is `start`, `advance`, `choose` / `slot`
// (with an `option` index), or one of the quick-row controls (`auto`,
// `skip`, `log`, `save`, `load`).
#[derive(serde::Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct StoryCmdRequest {
    action: String,
    option: usize,
}

// Send the story system the same `StoryCommand` a stage click or key press
// fires, so a headless harness can start, advance, and choose through a
// story and screenshot each page. `cn debug` only. The reply fires once the
// command is queued.
pub(super) fn handle_story(queue: &RuntimeQueue, text: &str) -> String {
    let req: StoryCmdRequest = match parse_request("story", text) {
        Ok(r) => r,
        Err(reply) => return reply,
    };
    let Some(command) = StoryCommand::from_verb(&req.action, Some(req.option)) else {
        return error_reply(&format!("story: unknown action '{}'", req.action));
    };
    run_with_reply(
        queue,
        "story",
        SPAWN_REPLY_TIMEOUT,
        |reply| WorldCommand::Story { command, reply },
        |()| serde_json::json!({ "ok": true, "queued": true }).to_string(),
    )
}

// Re-parent an authored placement. `child` is moved under `parent`; a null or
// omitted `parent` detaches the child to a root.
#[derive(serde::Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct ReparentCmdRequest {
    target: String,
    parent: Option<String>,
}

// Re-parent an authored placement live by name: enqueue a Reparent command the
// per-frame drive forwards to `dispatch_reparent`, which sends a
// `ReparentRequest` the GraphicsSystem applies on its next step (re-point the
// Parent edge + recompose world matrices). `cn debug` only; lets a headless
// harness move an entity under a new parent and screenshot the result. The reply
// fires once queued.
pub(super) fn handle_reparent(queue: &RuntimeQueue, text: &str) -> String {
    let req: ReparentCmdRequest = match parse_request("reparent", text) {
        Ok(r) => r,
        Err(reply) => return reply,
    };
    if req.target.trim().is_empty() {
        return error_reply("reparent: missing 'target'");
    }
    run_with_reply(
        queue,
        "reparent",
        SPAWN_REPLY_TIMEOUT,
        |reply| WorldCommand::Reparent {
            child: req.target,
            // An empty / whitespace parent name detaches the target to a root.
            parent: req.parent.filter(|p| !p.trim().is_empty()),
            reply,
        },
        |()| serde_json::json!({ "ok": true, "queued": true }).to_string(),
    )
}

// Spawn a runtime copy of an authored placement. `template` names the existing
// placement to copy; `name` is the new instance's identity. `position` /
// `rotation_deg` / `scale` place it (scale defaults to unit). A non-null
// `lifetime` (seconds) makes the instance auto-despawn after that long, which
// is what exercises draw-slot recycling.
#[derive(serde::Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct SpawnCmdRequest {
    template: String,
    name: String,
    position: [f32; 3],
    rotation_deg: [f32; 3],
    scale: [f32; 3],
    lifetime: Option<f32>,
}

// Instantiate a copy of an authored placement live by name: enqueue a Spawn
// command the per-frame drive forwards to `dispatch_spawn`, which sends a
// `SpawnRequest` the GraphicsSystem applies on its next step (cloning the
// template's draw slots into recycled slots and building the new entity).
// `cn debug` only; lets a headless harness spawn an instance and screenshot it,
// then watch its Lifetime expire. The reply fires once the command is queued.
pub(super) fn handle_spawn(queue: &RuntimeQueue, text: &str) -> String {
    let req: SpawnCmdRequest = match parse_request("spawn", text) {
        Ok(r) => r,
        Err(reply) => return reply,
    };
    if req.template.trim().is_empty() {
        return error_reply("spawn: missing 'template'");
    }
    if req.name.trim().is_empty() {
        return error_reply("spawn: missing 'name'");
    }
    run_with_reply(
        queue,
        "spawn",
        SPAWN_REPLY_TIMEOUT,
        |reply| WorldCommand::Spawn {
            template: req.template,
            name: req.name,
            position: req.position,
            rotation_deg: req.rotation_deg,
            scale: req.scale,
            lifetime: req.lifetime,
            reply,
        },
        |()| serde_json::json!({ "ok": true, "queued": true }).to_string(),
    )
}

// Move the active camera by a per-frame delta over a span of frames. All delta
// fields default to 0 and `frames` to 0 (an indefinite hold cleared by
// `camera-stop`); a profiling harness can then sustain motion mid-screenshot to
// surface temporal effects. `yaw` / `pitch` are radians.
pub(super) fn handle_camera_move(queue: &RuntimeQueue, text: &str) -> String {
    let args: CameraMoveArgs = match parse_request("camera-move", text) {
        Ok(r) => r,
        Err(reply) => return reply,
    };
    let frames = args.frames;
    let holding = frames == 0;
    run_with_reply(
        queue,
        "camera-move",
        SPAWN_REPLY_TIMEOUT,
        |reply| WorldCommand::CameraMove { args, reply },
        |()| serde_json::json!({ "ok": true, "frames": frames, "holding": holding }).to_string(),
    )
}

pub(super) fn handle_camera_stop(queue: &RuntimeQueue) -> String {
    run_with_reply(
        queue,
        "camera-stop",
        SPAWN_REPLY_TIMEOUT,
        |reply| WorldCommand::CameraStop { reply },
        |()| serde_json::json!({ "ok": true, "stopped": true }).to_string(),
    )
}

// Parse a payload with the same request struct the live handler uses, without
// enqueueing anything, so `super::catalog`'s schemas can be checked against the
// real serde shapes.
#[cfg(test)]
pub(super) fn parse_probe(cmd: &str, text: &str) -> Result<(), String> {
    fn probe<T: serde::de::DeserializeOwned>(cmd: &str, text: &str) -> Result<(), String> {
        parse_request::<T>(cmd, text).map(|_| ())
    }
    match cmd {
        "decal-add" => probe::<DecalSpawnArgs>(cmd, text),
        "decal-remove" | "emitter-remove" => probe::<IdRequest>(cmd, text),
        "emitter-add" => probe::<EmitterSpawnArgs>(cmd, text),
        "anim-crossfade" => probe::<AnimCrossfadeRequest>(cmd, text),
        "anim-param" => probe::<AnimParamRequest>(cmd, text),
        "anim-state" => probe::<AnimStateRequest>(cmd, text),
        "screenshot" => probe::<ScreenshotRequest>(cmd, text),
        "camera-set" => probe::<CameraSetArgs>(cmd, text),
        "camera-move" => probe::<CameraMoveArgs>(cmd, text),
        "quality-set" => probe::<QualitySetRequest>(cmd, text),
        "rebind" => probe::<RebindRequest>(cmd, text),
        "despawn" => probe::<DespawnCmdRequest>(cmd, text),
        "reparent" => probe::<ReparentCmdRequest>(cmd, text),
        "spawn" => probe::<SpawnCmdRequest>(cmd, text),
        "story" => probe::<StoryCmdRequest>(cmd, text),
        other => Err(format!("{other}: no request struct")),
    }
}

pub(super) fn error_reply(msg: &str) -> String {
    serde_json::json!({ "ok": false, "error": msg }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debug::anim_command::dispatch_anim_command;
    use crate::debug::anim_command::tests::{flat_world, graph_world};
    use crate::test_support;
    use concinnity_core::ecs::World;
    use concinnity_host::thread::asset_id;

    #[test]
    fn camera_set_request_parses_full_payload() {
        let req: CameraSetArgs = parse_request("camera-set",
            r#"{"cmd":"camera-set","position":[1.0,2.0,3.0],"yaw":0.5,"pitch":-0.25,"fov_y_degrees":60.0}"#,
        )
        .expect("valid payload parses");
        assert_eq!(req.position, [1.0, 2.0, 3.0]);
        assert_eq!(req.yaw, 0.5);
        assert_eq!(req.pitch, -0.25);
        assert_eq!(req.fov_y_degrees, Some(60.0));
    }

    #[test]
    fn camera_set_request_fov_optional() {
        let req: CameraSetArgs = parse_request(
            "camera-set",
            r#"{"cmd":"camera-set","position":[0.0,1.0,0.0],"yaw":0.0,"pitch":0.0}"#,
        )
        .expect("payload without fov parses");
        assert_eq!(req.position, [0.0, 1.0, 0.0]);
        assert!(req.fov_y_degrees.is_none());
    }

    #[test]
    fn camera_set_request_defaults_for_missing_fields() {
        let req: CameraSetArgs =
            parse_request("camera-set", r#"{"cmd":"camera-set"}"#).expect("bare command parses");
        assert_eq!(req.position, [0.0, 0.0, 0.0]);
        assert_eq!(req.yaw, 0.0);
        assert_eq!(req.pitch, 0.0);
        assert!(req.fov_y_degrees.is_none());
    }

    #[test]
    fn camera_set_request_rejects_malformed() {
        // position must be three numbers; a string is a hard parse error.
        assert!(
            parse_request::<CameraSetArgs>(
                "camera-set",
                r#"{"cmd":"camera-set","position":"nope"}"#
            )
            .is_err()
        );
    }

    #[test]
    fn camera_move_request_parses_full_payload() {
        let req: CameraMoveArgs = parse_request("camera-move",
            r#"{"cmd":"camera-move","forward":2.0,"right":-1.0,"up":0.5,"yaw":0.1,"pitch":-0.2,"frames":30}"#,
        )
        .expect("valid payload parses");
        assert_eq!(req.forward, 2.0);
        assert_eq!(req.right, -1.0);
        assert_eq!(req.up, 0.5);
        assert_eq!(req.yaw, 0.1);
        assert_eq!(req.pitch, -0.2);
        assert_eq!(req.frames, 30);
    }

    #[test]
    fn camera_move_request_defaults_to_zero_hold() {
        // A bare command leaves every delta at 0 and frames at 0 (indefinite
        // hold), matching the spawn-request default convention.
        let req: CameraMoveArgs =
            parse_request("camera-move", r#"{"cmd":"camera-move"}"#).expect("bare command parses");
        assert_eq!(req.forward, 0.0);
        assert_eq!(req.frames, 0);
    }

    #[test]
    fn camera_move_request_rejects_malformed() {
        // frames must be an unsigned integer; a string is a hard parse error.
        assert!(
            parse_request::<CameraMoveArgs>(
                "camera-move",
                r#"{"cmd":"camera-move","frames":"lots"}"#
            )
            .is_err()
        );
    }

    #[test]
    fn story_request_parses_action_and_option() {
        let req: StoryCmdRequest =
            parse_request("story", r#"{"cmd":"story","action":"choose","option":1}"#)
                .expect("valid parses");
        assert_eq!(req.action, "choose");
        assert_eq!(req.option, 1);
        let req: StoryCmdRequest =
            parse_request("story", r#"{"cmd":"story","action":"advance"}"#).expect("valid parses");
        assert_eq!(req.action, "advance");
        assert_eq!(req.option, 0);
    }

    #[test]
    fn despawn_request_parses_target() {
        let req: DespawnCmdRequest =
            parse_request("despawn", r#"{"cmd":"despawn","target":"crate_a"}"#)
                .expect("valid parses");
        assert_eq!(req.target, "crate_a");
    }

    #[test]
    fn despawn_request_defaults_to_an_empty_target() {
        let req: DespawnCmdRequest =
            parse_request("despawn", r#"{"cmd":"despawn"}"#).expect("bare command parses");
        assert!(req.target.is_empty());
    }

    #[test]
    fn reparent_request_parses_target_and_parent() {
        let req: ReparentCmdRequest = parse_request(
            "reparent",
            r#"{"cmd":"reparent","target":"box_a","parent":"frame"}"#,
        )
        .expect("valid parses");
        assert_eq!(req.target, "box_a");
        assert_eq!(req.parent.as_deref(), Some("frame"));
    }

    #[test]
    fn reparent_request_parent_optional() {
        let req: ReparentCmdRequest =
            parse_request("reparent", r#"{"cmd":"reparent","target":"box_a"}"#)
                .expect("bare parent parses");
        assert_eq!(req.target, "box_a");
        assert!(req.parent.is_none());
    }

    // Assert a handler reply is the error shape and names the problem.
    fn assert_err_reply(reply: &str, needle: &str) {
        assert!(
            reply.contains(r#""ok":false"#),
            "expected an error reply, got: {reply}"
        );
        assert!(
            reply.contains(needle),
            "expected '{needle}' in error reply: {reply}"
        );
    }

    #[test]
    fn error_reply_is_json_with_ok_false() {
        let reply = error_reply("boom");
        assert_err_reply(&reply, "boom");
    }

    // Malformed / validation-failing command text returns an error reply
    // before anything is enqueued.

    #[test]
    fn decal_add_rejects_malformed_json() {
        assert_err_reply(
            &handle_decal_add(&RuntimeQueue::default(), "not json"),
            "decal-add",
        );
    }

    #[test]
    fn decal_remove_rejects_missing_id() {
        assert_err_reply(
            &handle_decal_remove(&RuntimeQueue::default(), r#"{"cmd":"decal-remove"}"#),
            "decal-remove",
        );
    }

    #[test]
    fn emitter_add_rejects_malformed_json() {
        assert_err_reply(
            &handle_emitter_add(&RuntimeQueue::default(), "not json"),
            "emitter-add",
        );
    }

    #[test]
    fn emitter_remove_rejects_missing_id() {
        assert_err_reply(
            &handle_emitter_remove(&RuntimeQueue::default(), "{}"),
            "emitter-remove",
        );
    }

    #[test]
    fn anim_crossfade_rejects_malformed_json() {
        assert_err_reply(
            &handle_anim_crossfade(&RuntimeQueue::default(), "not json", &[]),
            "anim-crossfade",
        );
    }

    #[test]
    fn anim_crossfade_requires_a_target() {
        assert_err_reply(
            &handle_anim_crossfade(&RuntimeQueue::default(), "{}", &[]),
            "missing 'target'",
        );
    }

    #[test]
    fn anim_crossfade_rejects_an_unknown_target_name() {
        let names = vec!["hero".to_string()];
        let reply = handle_anim_crossfade(
            &RuntimeQueue::default(),
            r#"{"target":"villain","weights":[1.0]}"#,
            &names,
        );
        assert_err_reply(&reply, "unknown asset name 'villain'");
    }

    #[test]
    fn anim_param_rejects_malformed_json() {
        assert_err_reply(
            &handle_anim_param(&RuntimeQueue::default(), "not json", &[]),
            "anim-param",
        );
    }

    #[test]
    fn anim_param_requires_a_parameter_name() {
        assert_err_reply(
            &handle_anim_param(&RuntimeQueue::default(), r#"{"target":"hero"}"#, &[]),
            "missing 'name'",
        );
    }

    #[test]
    fn anim_param_requires_a_target() {
        assert_err_reply(
            &handle_anim_param(&RuntimeQueue::default(), r#"{"name":"speed"}"#, &[]),
            "missing 'target'",
        );
    }

    #[test]
    fn anim_param_rejects_an_unknown_target_name() {
        let names = vec!["hero".to_string()];
        let reply = handle_anim_param(
            &RuntimeQueue::default(),
            r#"{"target":"villain","name":"speed","value":1.0}"#,
            &names,
        );
        assert_err_reply(&reply, "unknown asset name 'villain'");
    }

    #[test]
    fn anim_state_rejects_malformed_json() {
        assert_err_reply(
            &handle_anim_state(&RuntimeQueue::default(), "not json", &[]),
            "anim-state",
        );
    }

    #[test]
    fn anim_state_requires_a_target() {
        assert_err_reply(
            &handle_anim_state(&RuntimeQueue::default(), "{}", &[]),
            "missing 'target'",
        );
    }

    #[test]
    fn anim_state_rejects_an_unknown_target_name() {
        let names = vec!["hero".to_string()];
        assert_err_reply(
            &handle_anim_state(&RuntimeQueue::default(), r#"{"target":"villain"}"#, &names),
            "unknown asset name 'villain'",
        );
    }

    #[test]
    fn resolve_target_maps_a_name_to_its_table_index() {
        let names = vec!["a".to_string(), "b".to_string()];
        let id = resolve_target("cmd", "b", &names).expect("known name resolves");
        assert_eq!(id, AssetId(1));
    }

    // The reply shape a Hi-Z A/B reads: every outcome counted, plus the two
    // derived numbers (`visible`, `hiz_rejected`) that survive the phase-2
    // rewrite of a candidate.
    #[test]
    fn cull_status_reply_reports_every_outcome() {
        use concinnity_core::gfx::cull_status::CullStatus;
        let raw = [
            CullStatus::DRAWN,
            CullStatus::DRAWN,
            CullStatus::CULLED,
            CullStatus::HIZ_CANDIDATE,
            CullStatus::REDRAW,
            CullStatus::HIZ_CULLED,
        ];
        let r: serde_json::Value =
            serde_json::from_str(&cull_status_reply(&raw)).expect("valid JSON");
        assert_eq!(r["ok"], true);
        assert_eq!(r["objects"], 6);
        assert_eq!(r["drawn"], 2);
        assert_eq!(r["frustum_culled"], 1);
        assert_eq!(r["hiz_candidate"], 1);
        assert_eq!(r["redrawn"], 1);
        assert_eq!(r["hiz_culled"], 1);
        assert_eq!(r["unknown"], 0);
        assert_eq!(r["visible"], 3);
        assert_eq!(r["hiz_rejected"], 2);
    }

    // A world whose cull never ran reads back nothing; the reply must be a
    // clean all-zero histogram rather than an error, so a probe can tell "no
    // objects" from "the command failed".
    #[test]
    fn cull_status_reply_of_an_empty_readback_is_all_zero() {
        let r: serde_json::Value =
            serde_json::from_str(&cull_status_reply(&[])).expect("valid JSON");
        assert_eq!(r["ok"], true);
        assert_eq!(r["objects"], 0);
        assert_eq!(r["visible"], 0);
        assert_eq!(r["hiz_rejected"], 0);
    }

    #[test]
    fn screenshot_rejects_malformed_json() {
        assert_err_reply(
            &handle_screenshot(&RuntimeQueue::default(), "not json"),
            "screenshot",
        );
    }

    #[test]
    fn screenshot_requires_a_path() {
        assert_err_reply(
            &handle_screenshot(&RuntimeQueue::default(), "{}"),
            "missing 'path'",
        );
        assert_err_reply(
            &handle_screenshot(&RuntimeQueue::default(), r#"{"path":"   "}"#),
            "missing 'path'",
        );
    }

    #[test]
    fn screenshot_requires_a_png_path() {
        assert_err_reply(
            &handle_screenshot(&RuntimeQueue::default(), r#"{"path":"/tmp/out.rs"}"#),
            ".png",
        );
        assert_err_reply(
            &handle_screenshot(&RuntimeQueue::default(), r#"{"path":"/tmp/out"}"#),
            ".png",
        );
    }

    #[test]
    fn camera_set_handler_rejects_malformed_json() {
        assert_err_reply(
            &handle_camera_set(&RuntimeQueue::default(), r#"{"position":"nope"}"#),
            "camera-set",
        );
    }

    #[test]
    fn camera_move_handler_rejects_malformed_json() {
        assert_err_reply(
            &handle_camera_move(&RuntimeQueue::default(), r#"{"frames":"lots"}"#),
            "camera-move",
        );
    }

    #[test]
    fn quality_set_rejects_malformed_json() {
        assert_err_reply(
            &handle_quality_set(&RuntimeQueue::default(), "not json"),
            "quality-set",
        );
    }

    #[test]
    fn quality_set_requires_a_setting() {
        assert_err_reply(
            &handle_quality_set(&RuntimeQueue::default(), "{}"),
            "missing 'setting'",
        );
    }

    #[test]
    fn quality_set_rejects_an_unknown_op() {
        assert_err_reply(
            &handle_quality_set(
                &RuntimeQueue::default(),
                r#"{"setting":"ssao","op":"sideways"}"#,
            ),
            "unknown op 'sideways'",
        );
    }

    // Only the five feature toggles are reachable: an AA mode or a display
    // setting is refused before anything is queued, so it is never persisted.
    #[test]
    fn quality_set_rejects_a_key_outside_the_toggles() {
        let queue = RuntimeQueue::default();
        for key in ["taa", "aa_mode", "vsync"] {
            let body = format!(r#"{{"setting":"{key}"}}"#);
            assert_err_reply(
                &handle_quality_set(&queue, &body),
                "is not a quality toggle",
            );
        }
        assert!(queue.drain().is_empty());
    }

    #[test]
    fn rebind_rejects_malformed_json() {
        assert_err_reply(
            &handle_rebind(&RuntimeQueue::default(), "not json"),
            "rebind",
        );
    }

    #[test]
    fn rebind_requires_a_setting() {
        assert_err_reply(
            &handle_rebind(&RuntimeQueue::default(), r#"{"key":"W"}"#),
            "missing 'setting'",
        );
    }

    #[test]
    fn rebind_rejects_an_unknown_key_name() {
        assert_err_reply(
            &handle_rebind(
                &RuntimeQueue::default(),
                r#"{"setting":"key_forward","key":"NotAKey"}"#,
            ),
            "unknown key 'NotAKey'",
        );
    }

    // A gamepad rebind or an unknown action is refused: the verb binds keys only.
    #[test]
    fn rebind_rejects_an_unknown_setting() {
        let queue = RuntimeQueue::default();
        for key in ["pad_jump", "key_nope"] {
            let body = format!(r#"{{"setting":"{key}","key":"W"}}"#);
            assert_err_reply(&handle_rebind(&queue, &body), "is not a key rebind");
        }
        assert!(queue.drain().is_empty());
    }

    #[test]
    fn despawn_rejects_malformed_json() {
        assert_err_reply(
            &handle_despawn(&RuntimeQueue::default(), "not json"),
            "despawn",
        );
    }

    #[test]
    fn despawn_requires_a_target() {
        assert_err_reply(
            &handle_despawn(&RuntimeQueue::default(), "{}"),
            "missing 'target'",
        );
        assert_err_reply(
            &handle_despawn(&RuntimeQueue::default(), r#"{"target":"  "}"#),
            "missing 'target'",
        );
    }

    #[test]
    fn story_rejects_malformed_json() {
        assert_err_reply(&handle_story(&RuntimeQueue::default(), "not json"), "story");
    }

    #[test]
    fn story_rejects_an_unknown_action() {
        assert_err_reply(
            &handle_story(&RuntimeQueue::default(), r#"{"action":"dance"}"#),
            "unknown action 'dance'",
        );
    }

    #[test]
    fn reparent_rejects_malformed_json() {
        assert_err_reply(
            &handle_reparent(&RuntimeQueue::default(), "not json"),
            "reparent",
        );
    }

    #[test]
    fn reparent_requires_a_target() {
        assert_err_reply(
            &handle_reparent(&RuntimeQueue::default(), "{}"),
            "missing 'target'",
        );
        assert_err_reply(
            &handle_reparent(&RuntimeQueue::default(), r#"{"target":" "}"#),
            "missing 'target'",
        );
    }

    #[test]
    fn spawn_rejects_malformed_json() {
        assert_err_reply(&handle_spawn(&RuntimeQueue::default(), "not json"), "spawn");
    }

    #[test]
    fn spawn_requires_template_and_name() {
        assert_err_reply(
            &handle_spawn(&RuntimeQueue::default(), "{}"),
            "missing 'template'",
        );
        assert_err_reply(
            &handle_spawn(&RuntimeQueue::default(), r#"{"template":"crate_a"}"#),
            "missing 'name'",
        );
    }

    // Success-path handler tests. Each handler blocks on a reply channel the
    // per-frame drive normally fulfils; here a worker thread runs the handler
    // against a queue of its own while the test drains it and answers in its
    // place.

    // A heavily loaded test host can stall either thread past the handler's
    // one-second engine timeout; when the handler reports that timeout the
    // whole exchange is retried, so the tests assert the reply semantics
    // rather than the scheduler. Reply sends never unwrap for the same
    // reason: a timed-out handler has already dropped its receiver.
    fn drive_handler(
        handler: impl Fn(&RuntimeQueue) -> String + Send + Sync + 'static,
        mut drive: impl FnMut(&RuntimeQueue),
    ) -> String {
        let handler = std::sync::Arc::new(handler);
        for _ in 0..5 {
            let queue = RuntimeQueue::default();
            let h = std::sync::Arc::clone(&handler);
            let handler_queue = queue.clone();
            let worker = std::thread::spawn(move || h(&handler_queue));
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while !worker.is_finished() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "handler never returned"
                );
                drive(&queue);
                std::thread::yield_now();
            }
            let response = worker.join().expect("handler thread panicked");
            if !response.contains("timed out waiting for engine") {
                return response;
            }
        }
        panic!("handler kept timing out under load");
    }

    // Drive a runtime handler, answering each queued command with `reply`. A
    // command `reply` hands back is not the one under test and stays unanswered.
    fn drive_runtime_handler(
        handler: impl Fn(&RuntimeQueue) -> String + Send + Sync + 'static,
        mut reply: impl FnMut(RuntimeCommand) -> Option<RuntimeCommand>,
    ) -> String {
        drive_handler(handler, |queue| {
            for cmd in queue.drain() {
                assert!(reply(cmd).is_none(), "unexpected runtime command");
            }
        })
    }

    #[test]
    fn decal_add_round_trips_args_and_reports_the_new_id() {
        let reply = drive_runtime_handler(
            |q| {
                handle_decal_add(
                    q,
                    r#"{"texture":"grid","position":[1.0,2.0,3.0],"size":[2.0,2.0,2.0],"tint":[1.0,0.0,0.0,1.0]}"#,
                )
            },
            |cmd| match cmd {
                RuntimeCommand::Backend(BackendCommand::DecalAdd { args, reply }) => {
                    assert_eq!(args.texture.as_deref(), Some("grid"));
                    assert_eq!(args.position, [1.0, 2.0, 3.0]);
                    assert_eq!(args.size, [2.0, 2.0, 2.0]);
                    assert_eq!(args.tint, [1.0, 0.0, 0.0, 1.0]);
                    let _ = reply.send(Ok(7));
                    None
                }
                other => Some(other),
            },
        );
        assert!(reply.contains(r#""ok":true"#), "got: {reply}");
        assert!(reply.contains(r#""id":7"#), "got: {reply}");
    }

    #[test]
    fn decal_add_surfaces_an_engine_error() {
        let reply = drive_runtime_handler(
            |q| handle_decal_add(q, "{}"),
            |cmd| match cmd {
                RuntimeCommand::Backend(BackendCommand::DecalAdd { reply, .. }) => {
                    let _ = reply.send(Err("no free decal slot".to_string()));
                    None
                }
                other => Some(other),
            },
        );
        assert_err_reply(&reply, "no free decal slot");
    }

    // Answer any runtime-spawn command with an engine-side failure, so the
    // handler under test takes its `Ok(Err(e))` reply branch.
    fn reply_engine_error(cmd: RuntimeCommand) -> Option<RuntimeCommand> {
        match cmd {
            RuntimeCommand::Backend(BackendCommand::DecalAdd { reply, .. }) => {
                drop(reply.send(Err("boom".into())))
            }
            RuntimeCommand::Backend(BackendCommand::DecalRemove { reply, .. }) => {
                drop(reply.send(Err("boom".into())))
            }
            RuntimeCommand::Backend(BackendCommand::EmitterAdd { reply, .. }) => {
                drop(reply.send(Err("boom".into())))
            }
            RuntimeCommand::Backend(BackendCommand::EmitterRemove { reply, .. }) => {
                drop(reply.send(Err("boom".into())))
            }
            RuntimeCommand::Backend(BackendCommand::Screenshot { reply, .. }) => {
                drop(reply.send(Err("boom".into())))
            }
            RuntimeCommand::Backend(BackendCommand::CullStatus { reply }) => {
                drop(reply.send(Err("boom".into())))
            }
            RuntimeCommand::World(WorldCommand::CameraSet { reply, .. }) => {
                drop(reply.send(Err("boom".into())))
            }
            RuntimeCommand::World(WorldCommand::CameraMove { reply, .. }) => {
                drop(reply.send(Err("boom".into())))
            }
            RuntimeCommand::World(WorldCommand::CameraStop { reply }) => {
                drop(reply.send(Err("boom".into())))
            }
            RuntimeCommand::World(WorldCommand::QualitySet { reply, .. }) => {
                drop(reply.send(Err("boom".into())))
            }
            RuntimeCommand::World(WorldCommand::Rebind { reply, .. }) => {
                drop(reply.send(Err("boom".into())))
            }
            RuntimeCommand::World(WorldCommand::Despawn { reply, .. }) => {
                drop(reply.send(Err("boom".into())))
            }
            RuntimeCommand::World(WorldCommand::Reparent { reply, .. }) => {
                drop(reply.send(Err("boom".into())))
            }
            RuntimeCommand::World(WorldCommand::Spawn { reply, .. }) => {
                drop(reply.send(Err("boom".into())))
            }
            RuntimeCommand::World(WorldCommand::Story { reply, .. }) => {
                drop(reply.send(Err("boom".into())))
            }
            RuntimeCommand::Anim(cmd) => return Some(RuntimeCommand::Anim(cmd)),
        }
        None
    }

    // Every runtime-spawn handler surfaces an engine-side rejection as an error
    // reply. Drives each handler with a valid payload and answers `Err`.
    // A named handler invocation: its label plus a boxed thunk that runs it.
    type NamedHandler = (
        &'static str,
        Box<dyn Fn(&RuntimeQueue) -> String + Send + Sync>,
    );

    #[test]
    fn runtime_handlers_surface_engine_errors() {
        let handlers: Vec<NamedHandler> = vec![
            (
                "decal-remove",
                Box::new(|q| handle_decal_remove(q, r#"{"id":0}"#)),
            ),
            ("emitter-add", Box::new(|q| handle_emitter_add(q, "{}"))),
            (
                "emitter-remove",
                Box::new(|q| handle_emitter_remove(q, r#"{"id":0}"#)),
            ),
            (
                "screenshot",
                Box::new(|q| handle_screenshot(q, r#"{"path":"x.png"}"#)),
            ),
            ("cull-status", Box::new(handle_cull_status)),
            ("camera-set", Box::new(|q| handle_camera_set(q, "{}"))),
            ("camera-move", Box::new(|q| handle_camera_move(q, "{}"))),
            ("camera-stop", Box::new(handle_camera_stop)),
            (
                "quality-set",
                Box::new(|q| handle_quality_set(q, r#"{"setting":"ssao"}"#)),
            ),
            (
                "rebind",
                Box::new(|q| handle_rebind(q, r#"{"setting":"key_forward","key":"W"}"#)),
            ),
            (
                "despawn",
                Box::new(|q| handle_despawn(q, r#"{"target":"x"}"#)),
            ),
            (
                "reparent",
                Box::new(|q| handle_reparent(q, r#"{"target":"x"}"#)),
            ),
            (
                "spawn",
                Box::new(|q| handle_spawn(q, r#"{"template":"t","name":"n"}"#)),
            ),
            (
                "story",
                Box::new(|q| handle_story(q, r#"{"action":"start"}"#)),
            ),
        ];
        for (name, h) in handlers {
            let reply = drive_runtime_handler(h, reply_engine_error);
            assert!(
                reply.contains(r#""ok":false"#) && reply.contains("boom"),
                "{name} should surface the engine error: {reply}"
            );
        }
    }

    // When nothing drains the queue, each handler's `recv_timeout` elapses and
    // it reports a timeout. Running them concurrently bounds the test to about
    // one timeout interval.
    #[test]
    fn runtime_handlers_report_a_timeout_when_the_engine_never_replies() {
        let workers: Vec<(&str, std::thread::JoinHandle<String>)> = vec![
            (
                "decal-add",
                std::thread::spawn(|| handle_decal_add(&RuntimeQueue::default(), "{}")),
            ),
            (
                "decal-remove",
                std::thread::spawn(|| handle_decal_remove(&RuntimeQueue::default(), r#"{"id":0}"#)),
            ),
            (
                "emitter-add",
                std::thread::spawn(|| handle_emitter_add(&RuntimeQueue::default(), "{}")),
            ),
            (
                "emitter-remove",
                std::thread::spawn(|| {
                    handle_emitter_remove(&RuntimeQueue::default(), r#"{"id":0}"#)
                }),
            ),
            (
                "camera-set",
                std::thread::spawn(|| handle_camera_set(&RuntimeQueue::default(), "{}")),
            ),
            (
                "camera-move",
                std::thread::spawn(|| handle_camera_move(&RuntimeQueue::default(), "{}")),
            ),
            (
                "camera-stop",
                std::thread::spawn(|| handle_camera_stop(&RuntimeQueue::default())),
            ),
            (
                "quality-set",
                std::thread::spawn(|| {
                    handle_quality_set(&RuntimeQueue::default(), r#"{"setting":"ssao"}"#)
                }),
            ),
            (
                "rebind",
                std::thread::spawn(|| {
                    handle_rebind(
                        &RuntimeQueue::default(),
                        r#"{"setting":"key_forward","key":"W"}"#,
                    )
                }),
            ),
            (
                "despawn",
                std::thread::spawn(|| {
                    handle_despawn(&RuntimeQueue::default(), r#"{"target":"x"}"#)
                }),
            ),
            (
                "reparent",
                std::thread::spawn(|| {
                    handle_reparent(&RuntimeQueue::default(), r#"{"target":"x"}"#)
                }),
            ),
            (
                "spawn",
                std::thread::spawn(|| {
                    handle_spawn(&RuntimeQueue::default(), r#"{"template":"t","name":"n"}"#)
                }),
            ),
            (
                "story",
                std::thread::spawn(|| {
                    handle_story(&RuntimeQueue::default(), r#"{"action":"start"}"#)
                }),
            ),
        ];
        for (name, w) in workers {
            let reply = w.join().expect("handler thread panicked");
            assert!(
                reply.contains("timed out waiting for engine"),
                "{name}: {reply}"
            );
        }
    }

    #[test]
    fn decal_remove_reports_removed() {
        let reply = drive_runtime_handler(
            |q| handle_decal_remove(q, r#"{"id":3}"#),
            |cmd| match cmd {
                RuntimeCommand::Backend(BackendCommand::DecalRemove { id, reply }) => {
                    assert_eq!(id, 3);
                    let _ = reply.send(Ok(()));
                    None
                }
                other => Some(other),
            },
        );
        assert!(reply.contains(r#""removed":true"#), "got: {reply}");
    }

    #[test]
    fn emitter_add_defaults_and_reports_the_new_id() {
        let reply = drive_runtime_handler(
            |q| handle_emitter_add(q, "{}"),
            |cmd| match cmd {
                RuntimeCommand::Backend(BackendCommand::EmitterAdd { args, reply }) => {
                    // A bare command carries the emitter defaults through.
                    assert_eq!(args.direction, [0.0, 1.0, 0.0]);
                    assert_eq!(args.max_particles, 256);
                    let _ = reply.send(Ok(2));
                    None
                }
                other => Some(other),
            },
        );
        assert!(reply.contains(r#""id":2"#), "got: {reply}");
    }

    #[test]
    fn emitter_remove_reports_removed() {
        let reply = drive_runtime_handler(
            |q| handle_emitter_remove(q, r#"{"id":5}"#),
            |cmd| match cmd {
                RuntimeCommand::Backend(BackendCommand::EmitterRemove { id, reply }) => {
                    assert_eq!(id, 5);
                    let _ = reply.send(Ok(()));
                    None
                }
                other => Some(other),
            },
        );
        assert!(reply.contains(r#""removed":true"#), "got: {reply}");
    }

    #[test]
    fn screenshot_echoes_the_saved_path() {
        let reply = drive_runtime_handler(
            |q| handle_screenshot(q, r#"{"path":"shot.png"}"#),
            |cmd| match cmd {
                RuntimeCommand::Backend(BackendCommand::Screenshot { path, reply }) => {
                    assert_eq!(path, "shot.png");
                    let _ = reply.send(Ok("shot.png".to_string()));
                    None
                }
                other => Some(other),
            },
        );
        assert!(reply.contains(r#""path":"shot.png""#), "got: {reply}");
    }

    // The path is validated and written as the same trimmed value.
    #[test]
    fn screenshot_forwards_the_trimmed_path() {
        let reply = drive_runtime_handler(
            |q| handle_screenshot(q, r#"{"path":" shot.png "}"#),
            |cmd| match cmd {
                RuntimeCommand::Backend(BackendCommand::Screenshot { path, reply }) => {
                    assert_eq!(path, "shot.png");
                    let _ = reply.send(Ok(path));
                    None
                }
                other => Some(other),
            },
        );
        assert!(reply.contains(r#""path":"shot.png""#), "got: {reply}");
    }

    #[test]
    fn camera_set_round_trips_the_pose() {
        let reply = drive_runtime_handler(
            |q| {
                handle_camera_set(
                    q,
                    r#"{"position":[1.0,2.0,3.0],"yaw":0.5,"pitch":-0.25,"fov_y_degrees":60.0}"#,
                )
            },
            |cmd| match cmd {
                RuntimeCommand::World(WorldCommand::CameraSet { args, reply }) => {
                    assert_eq!(args.position, [1.0, 2.0, 3.0]);
                    assert_eq!(args.yaw, 0.5);
                    assert_eq!(args.pitch, -0.25);
                    assert_eq!(args.fov_y_degrees, Some(60.0));
                    let _ = reply.send(Ok(()));
                    None
                }
                other => Some(other),
            },
        );
        assert!(reply.contains(r#""set":true"#), "got: {reply}");
    }

    #[test]
    fn camera_move_reports_finite_frames() {
        let reply = drive_runtime_handler(
            |q| handle_camera_move(q, r#"{"forward":1.5,"frames":3}"#),
            |cmd| match cmd {
                RuntimeCommand::World(WorldCommand::CameraMove { args, reply }) => {
                    assert_eq!(args.forward, 1.5);
                    assert_eq!(args.frames, 3);
                    let _ = reply.send(Ok(()));
                    None
                }
                other => Some(other),
            },
        );
        assert!(reply.contains(r#""frames":3"#), "got: {reply}");
        assert!(reply.contains(r#""holding":false"#), "got: {reply}");
    }

    #[test]
    fn camera_move_defaults_to_an_indefinite_hold() {
        let reply = drive_runtime_handler(
            |q| handle_camera_move(q, "{}"),
            |cmd| match cmd {
                RuntimeCommand::World(WorldCommand::CameraMove { reply, .. }) => {
                    let _ = reply.send(Ok(()));
                    None
                }
                other => Some(other),
            },
        );
        assert!(reply.contains(r#""holding":true"#), "got: {reply}");
    }

    #[test]
    fn camera_stop_reports_stopped() {
        let reply = drive_runtime_handler(handle_camera_stop, |cmd| match cmd {
            RuntimeCommand::World(WorldCommand::CameraStop { reply }) => {
                let _ = reply.send(Ok(()));
                None
            }
            other => Some(other),
        });
        assert!(reply.contains(r#""stopped":true"#), "got: {reply}");
    }

    #[test]
    fn quality_set_maps_ops_and_reports_queued() {
        let reply = drive_runtime_handler(
            |q| handle_quality_set(q, r#"{"setting":"ssao","op":"prev"}"#),
            |cmd| match cmd {
                RuntimeCommand::World(WorldCommand::QualitySet { setting, op, reply }) => {
                    assert_eq!(setting, SettingKey::Ssao);
                    assert_eq!(op, SettingOp::Prev);
                    let _ = reply.send(Ok(()));
                    None
                }
                other => Some(other),
            },
        );
        assert!(reply.contains(r#""queued":true"#), "got: {reply}");

        // An omitted op defaults to Next.
        let reply = drive_runtime_handler(
            |q| handle_quality_set(q, r#"{"setting":"ssao"}"#),
            |cmd| match cmd {
                RuntimeCommand::World(WorldCommand::QualitySet { op, reply, .. }) => {
                    assert_eq!(op, SettingOp::Next);
                    let _ = reply.send(Ok(()));
                    None
                }
                other => Some(other),
            },
        );
        assert!(reply.contains(r#""queued":true"#), "got: {reply}");
    }

    #[test]
    fn rebind_resolves_the_key_variant() {
        let reply = drive_runtime_handler(
            |q| handle_rebind(q, r#"{"setting":"key_forward","key":"Space"}"#),
            |cmd| match cmd {
                RuntimeCommand::World(WorldCommand::Rebind { action, key, reply }) => {
                    assert_eq!(action, Bindable::Forward);
                    assert_eq!(key, InputKey::Space);
                    let _ = reply.send(Ok(()));
                    None
                }
                other => Some(other),
            },
        );
        assert!(reply.contains(r#""queued":true"#), "got: {reply}");
    }

    #[test]
    fn despawn_round_trips_the_target() {
        let reply = drive_runtime_handler(
            |q| handle_despawn(q, r#"{"target":"crate_a"}"#),
            |cmd| match cmd {
                RuntimeCommand::World(WorldCommand::Despawn { name, reply }) => {
                    assert_eq!(name, "crate_a");
                    let _ = reply.send(Ok(()));
                    None
                }
                other => Some(other),
            },
        );
        assert!(reply.contains(r#""queued":true"#), "got: {reply}");
    }

    #[test]
    fn story_maps_every_action_to_its_command() {
        use concinnity_core::components::StoryCommand;
        let cases = [
            ("start", StoryCommand::Start),
            ("continue", StoryCommand::Continue),
            ("advance", StoryCommand::Advance),
            ("auto", StoryCommand::ToggleAuto),
            ("skip", StoryCommand::ToggleSkip),
            ("log", StoryCommand::ToggleLog),
            ("save", StoryCommand::OpenSave),
            ("load", StoryCommand::OpenLoad),
            ("pause", StoryCommand::TogglePause),
            ("settings", StoryCommand::OpenSettings),
            ("settings_back", StoryCommand::CloseSettings),
        ];
        for (action, expected) in cases {
            let text = format!(r#"{{"action":"{action}"}}"#);
            let reply = drive_runtime_handler(
                move |q| handle_story(q, &text),
                |cmd| match cmd {
                    RuntimeCommand::World(WorldCommand::Story { command, reply }) => {
                        assert_eq!(command, expected, "action '{action}'");
                        let _ = reply.send(Ok(()));
                        None
                    }
                    other => Some(other),
                },
            );
            assert!(reply.contains(r#""queued":true"#), "got: {reply}");
        }
    }

    #[test]
    fn story_choose_and_slot_carry_the_option_index() {
        use concinnity_core::components::StoryCommand;
        for (action, expected) in [
            ("choose", StoryCommand::Choose(2)),
            ("slot", StoryCommand::Slot(2)),
        ] {
            let text = format!(r#"{{"action":"{action}","option":2}}"#);
            let reply = drive_runtime_handler(
                move |q| handle_story(q, &text),
                |cmd| match cmd {
                    RuntimeCommand::World(WorldCommand::Story { command, reply }) => {
                        assert_eq!(command, expected);
                        let _ = reply.send(Ok(()));
                        None
                    }
                    other => Some(other),
                },
            );
            assert!(reply.contains(r#""queued":true"#), "got: {reply}");
        }
    }

    #[test]
    fn reparent_filters_a_whitespace_parent_to_detach() {
        let reply = drive_runtime_handler(
            |q| handle_reparent(q, r#"{"target":"box_a","parent":"  "}"#),
            |cmd| match cmd {
                RuntimeCommand::World(WorldCommand::Reparent {
                    child,
                    parent,
                    reply,
                }) => {
                    assert_eq!(child, "box_a");
                    assert!(parent.is_none(), "whitespace parent must detach");
                    let _ = reply.send(Ok(()));
                    None
                }
                other => Some(other),
            },
        );
        assert!(reply.contains(r#""queued":true"#), "got: {reply}");
    }

    #[test]
    fn reparent_keeps_a_real_parent() {
        let reply = drive_runtime_handler(
            |q| handle_reparent(q, r#"{"target":"box_a","parent":"frame"}"#),
            |cmd| match cmd {
                RuntimeCommand::World(WorldCommand::Reparent { parent, reply, .. }) => {
                    assert_eq!(parent.as_deref(), Some("frame"));
                    let _ = reply.send(Ok(()));
                    None
                }
                other => Some(other),
            },
        );
        assert!(reply.contains(r#""queued":true"#), "got: {reply}");
    }

    #[test]
    fn spawn_round_trips_transform_and_lifetime() {
        let reply = drive_runtime_handler(
            |q| {
                handle_spawn(
                    q,
                    r#"{"template":"crate_a","name":"crate_b","position":[1.0,0.0,-1.0],"rotation_deg":[0.0,90.0,0.0],"scale":[2.0,2.0,2.0],"lifetime":2.5}"#,
                )
            },
            |cmd| match cmd {
                RuntimeCommand::World(WorldCommand::Spawn {
                    template,
                    name,
                    position,
                    rotation_deg,
                    scale,
                    lifetime,
                    reply,
                }) => {
                    assert_eq!(template, "crate_a");
                    assert_eq!(name, "crate_b");
                    assert_eq!(position, [1.0, 0.0, -1.0]);
                    assert_eq!(rotation_deg, [0.0, 90.0, 0.0]);
                    assert_eq!(scale, [2.0, 2.0, 2.0]);
                    assert_eq!(lifetime, Some(2.5));
                    let _ = reply.send(Ok(()));
                    None
                }
                other => Some(other),
            },
        );
        assert!(reply.contains(r#""queued":true"#), "got: {reply}");
    }

    // Animation handler success paths: drive a real system built from a small
    // world while the handler blocks, routing each queued command through
    // `dispatch_anim_command` exactly as the per-frame debug drive would.
    fn drive_anim_handler(
        world: &mut World,
        handler: impl Fn(&RuntimeQueue) -> String + Send + Sync + 'static,
    ) -> String {
        drive_handler(handler, |queue| {
            for cmd in queue.drain() {
                let RuntimeCommand::Anim(cmd) = cmd else {
                    panic!("unexpected runtime command");
                };
                dispatch_anim_command(cmd, concinnity_engine::ecs::animation_system_mut(world));
            }
        })
    }

    #[test]
    fn anim_param_queues_a_graph_parameter_write() {
        let _guard = test_support::lock();
        let mut world = graph_world();
        let names = asset_id::name_table();
        let reply = drive_anim_handler(&mut world, move |q| {
            handle_anim_param(q, r#"{"target":"hero","name":"speed","value":1.0}"#, &names)
        });
        assert!(reply.contains(r#""queued":true"#), "got: {reply}");
    }

    #[test]
    fn anim_param_surfaces_an_unknown_parameter() {
        let _guard = test_support::lock();
        let mut world = graph_world();
        let names = asset_id::name_table();
        let reply = drive_anim_handler(&mut world, move |q| {
            handle_anim_param(
                q,
                r#"{"target":"hero","name":"altitude","value":1.0}"#,
                &names,
            )
        });
        assert_err_reply(&reply, "no parameter 'altitude'");
    }

    #[test]
    fn anim_state_reports_the_live_graph_state() {
        let _guard = test_support::lock();
        let mut world = graph_world();
        let names = asset_id::name_table();
        let reply = drive_anim_handler(&mut world, move |q| {
            handle_anim_state(q, r#"{"target":"hero"}"#, &names)
        });
        assert!(reply.contains(r#""ok":true"#), "got: {reply}");
        assert!(reply.contains(r#""state":"idle""#), "got: {reply}");
        assert!(reply.contains(r#""params":{"speed":0.0}"#), "got: {reply}");
    }

    #[test]
    fn anim_crossfade_queues_on_a_flat_target() {
        let _guard = test_support::lock();
        let mut world = flat_world();
        let names = asset_id::name_table();
        let reply = drive_anim_handler(&mut world, move |q| {
            handle_anim_crossfade(
                q,
                r#"{"target":"hero","weights":[0.0,1.0],"duration_secs":0.5}"#,
                &names,
            )
        });
        assert!(reply.contains(r#""queued":true"#), "got: {reply}");
    }

    #[test]
    fn anim_crossfade_is_rejected_on_a_graph_target() {
        let _guard = test_support::lock();
        let mut world = graph_world();
        let names = asset_id::name_table();
        let reply = drive_anim_handler(&mut world, move |q| {
            handle_anim_crossfade(q, r#"{"target":"hero","weights":[1.0,0.0]}"#, &names)
        });
        assert_err_reply(&reply, "graph-driven");
    }
}
