// src/debug/commands.rs
//
// Runtime spawn / crossfade WebSocket command handlers (`decal-add`,
// `emitter-add`, `anim-crossfade`, …) plus their request-body structs and the
// shared `error_reply` helper. Each parses its JSON body, enqueues onto the
// matching process-wide queue (`super::runtime_spawn` /
// `crate::app::anim_runtime`), and blocks on a one-shot reply channel the
// per-frame debug drive fulfils. The query commands + dispatch live in
// `super::server::handle_request`.

// Maximum wait for `GraphicsSystem::step` to drain a runtime-spawn command
// and reply. The drain runs once per frame, so a healthy 60 Hz engine
// replies inside ~16 ms; 1 s gives plenty of headroom even on a slow boot
// frame (4K HDR bake) without leaving a WS client hanging forever if the
// engine has stalled.
const SPAWN_REPLY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1);

#[derive(serde::Deserialize)]
#[serde(default)]
struct DecalAddRequest {
    #[serde(skip)]
    _cmd: String,
    texture: Option<String>,
    position: [f32; 3],
    rotation_deg: [f32; 3],
    size: [f32; 3],
    tint: [f32; 4],
}

impl Default for DecalAddRequest {
    fn default() -> Self {
        Self {
            _cmd: String::new(),
            texture: None,
            position: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            size: [1.0, 1.0, 1.0],
            tint: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

pub(super) fn handle_decal_add(text: &str) -> String {
    let req: DecalAddRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("decal-add: {e}")),
    };
    let args = super::runtime_spawn::DecalSpawnArgs {
        texture: req.texture,
        position: req.position,
        rotation_deg: req.rotation_deg,
        size: req.size,
        tint: req.tint,
    };
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::DecalAdd {
        args,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(id)) => serde_json::json!({ "ok": true, "id": id }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("decal-add: timed out waiting for engine"),
    }
}

#[derive(serde::Deserialize)]
struct IdRequest {
    id: usize,
}

pub(super) fn handle_decal_remove(text: &str) -> String {
    let req: IdRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("decal-remove: {e}")),
    };
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::DecalRemove {
        id: req.id,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "removed": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("decal-remove: timed out waiting for engine"),
    }
}

#[derive(serde::Deserialize)]
#[serde(default)]
struct EmitterAddRequest {
    #[serde(skip)]
    _cmd: String,
    texture: Option<String>,
    position: [f32; 3],
    direction: [f32; 3],
    spread_deg: f32,
    speed_min: f32,
    speed_max: f32,
    lifetime_min: f32,
    lifetime_max: f32,
    gravity: [f32; 3],
    spawn_rate: f32,
    max_particles: u32,
    size_start: f32,
    size_end: f32,
    color_start: [f32; 4],
    color_end: [f32; 4],
}

impl Default for EmitterAddRequest {
    fn default() -> Self {
        let d = super::runtime_spawn::EmitterSpawnArgs::default();
        Self {
            _cmd: String::new(),
            texture: d.texture,
            position: d.position,
            direction: d.direction,
            spread_deg: d.spread_deg,
            speed_min: d.speed_min,
            speed_max: d.speed_max,
            lifetime_min: d.lifetime_min,
            lifetime_max: d.lifetime_max,
            gravity: d.gravity,
            spawn_rate: d.spawn_rate,
            max_particles: d.max_particles,
            size_start: d.size_start,
            size_end: d.size_end,
            color_start: d.color_start,
            color_end: d.color_end,
        }
    }
}

pub(super) fn handle_emitter_add(text: &str) -> String {
    let req: EmitterAddRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("emitter-add: {e}")),
    };
    let args = super::runtime_spawn::EmitterSpawnArgs {
        texture: req.texture,
        position: req.position,
        direction: req.direction,
        spread_deg: req.spread_deg,
        speed_min: req.speed_min,
        speed_max: req.speed_max,
        lifetime_min: req.lifetime_min,
        lifetime_max: req.lifetime_max,
        gravity: req.gravity,
        spawn_rate: req.spawn_rate,
        max_particles: req.max_particles,
        size_start: req.size_start,
        size_end: req.size_end,
        color_start: req.color_start,
        color_end: req.color_end,
    };
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::EmitterAdd {
        args,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(id)) => serde_json::json!({ "ok": true, "id": id }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("emitter-add: timed out waiting for engine"),
    }
}

pub(super) fn handle_emitter_remove(text: &str) -> String {
    let req: IdRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("emitter-remove: {e}")),
    };
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::EmitterRemove {
        id: req.id,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "removed": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("emitter-remove: timed out waiting for engine"),
    }
}

#[derive(serde::Deserialize)]
struct AnimCrossfadeRequest {
    #[serde(default)]
    target: String,
    #[serde(default)]
    weights: Vec<f32>,
    #[serde(default)]
    duration_secs: f32,
}

pub(super) fn handle_anim_crossfade(text: &str, names: &[String]) -> String {
    let req: AnimCrossfadeRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("anim-crossfade: {e}")),
    };
    if req.target.is_empty() {
        return error_reply("anim-crossfade: missing 'target'");
    }
    // The names table is `Vec<String>` indexed by `AssetId`; a small linear
    // scan is fine for a debug command that fires at most a few times per
    // second.
    let Some(asset_idx) = names.iter().position(|n| n == &req.target) else {
        return error_reply(&format!(
            "anim-crossfade: unknown asset name '{}'",
            req.target
        ));
    };
    let target = crate::ecs::asset_id::AssetId(asset_idx as u32);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    crate::app::anim_runtime::enqueue(crate::app::anim_runtime::AnimCommand::Crossfade {
        req: crate::app::anim_runtime::CrossfadeRequest {
            target,
            weights: req.weights,
            duration_secs: req.duration_secs,
        },
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "queued": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("anim-crossfade: timed out waiting for engine"),
    }
}

// Resolve a `target` asset name against the interner names table (indexed by
// `AssetId`; a small linear scan is fine for a debug command that fires at
// most a few times per second).
fn resolve_target(
    cmd: &str,
    target: &str,
    names: &[String],
) -> Result<crate::ecs::asset_id::AssetId, String> {
    if target.is_empty() {
        return Err(format!("{cmd}: missing 'target'"));
    }
    names
        .iter()
        .position(|n| n == target)
        .map(|idx| crate::ecs::asset_id::AssetId(idx as u32))
        .ok_or_else(|| format!("{cmd}: unknown asset name '{target}'"))
}

#[derive(serde::Deserialize)]
struct AnimParamRequest {
    #[serde(default)]
    target: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    value: f32,
}

pub(super) fn handle_anim_param(text: &str, names: &[String]) -> String {
    let req: AnimParamRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("anim-param: {e}")),
    };
    if req.name.is_empty() {
        return error_reply("anim-param: missing 'name' (a graph parameter)");
    }
    let target = match resolve_target("anim-param", &req.target, names) {
        Ok(t) => t,
        Err(e) => return error_reply(&e),
    };
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    crate::app::anim_runtime::enqueue(crate::app::anim_runtime::AnimCommand::SetParam {
        req: crate::app::anim_runtime::SetParamRequest {
            target,
            name: req.name,
            value: req.value,
        },
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "queued": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("anim-param: timed out waiting for engine"),
    }
}

#[derive(serde::Deserialize)]
struct AnimStateRequest {
    #[serde(default)]
    target: String,
}

pub(super) fn handle_anim_state(text: &str, names: &[String]) -> String {
    let req: AnimStateRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("anim-state: {e}")),
    };
    let target = match resolve_target("anim-state", &req.target, names) {
        Ok(t) => t,
        Err(e) => return error_reply(&e),
    };
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    crate::app::anim_runtime::enqueue(crate::app::anim_runtime::AnimCommand::QueryState {
        target,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(report)) => {
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
        }
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("anim-state: timed out waiting for engine"),
    }
}

// Longer than the spawn timeout: the capture idles the GPU, copies the
// swapchain image back, and PNG-encodes + writes it on the render thread, which
// can take noticeably longer than a simple state mutation.
const SCREENSHOT_REPLY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

#[derive(serde::Deserialize)]
struct ScreenshotRequest {
    #[serde(default)]
    path: String,
}

pub(super) fn handle_screenshot(text: &str) -> String {
    let req: ScreenshotRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("screenshot: {e}")),
    };
    if req.path.trim().is_empty() {
        return error_reply("screenshot: missing 'path'");
    }
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::Screenshot {
        path: req.path,
        reply: tx,
    });
    match rx.recv_timeout(SCREENSHOT_REPLY_TIMEOUT) {
        Ok(Ok(path)) => serde_json::json!({ "ok": true, "path": path }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("screenshot: timed out waiting for engine"),
    }
}

// Teleport the active camera. `position` / `yaw` / `pitch` are required in
// practice (the probe always sends them); missing fields fall back to the
// defaults below, matching the decal / emitter request shape. `yaw` / `pitch`
// are radians; `fov_y_degrees` is omitted to leave the camera's FOV untouched.
#[derive(serde::Deserialize)]
#[serde(default)]
struct CameraSetRequest {
    #[serde(skip)]
    _cmd: String,
    position: [f32; 3],
    yaw: f32,
    pitch: f32,
    fov_y_degrees: Option<f32>,
}

impl Default for CameraSetRequest {
    fn default() -> Self {
        Self {
            _cmd: String::new(),
            position: [0.0, 0.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            fov_y_degrees: None,
        }
    }
}

pub(super) fn handle_camera_set(text: &str) -> String {
    let req: CameraSetRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("camera-set: {e}")),
    };
    let args = super::runtime_spawn::CameraSetArgs {
        position: req.position,
        yaw: req.yaw,
        pitch: req.pitch,
        fov_y_degrees: req.fov_y_degrees,
    };
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::CameraSet {
        args,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "set": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("camera-set: timed out waiting for engine"),
    }
}

// Toggle a Quality-group graphics setting. `setting` is the engine key
// (taa / ssao / ssr / ssgi / auto_exposure); `op` cycles it (next | prev,
// both flip a binary toggle). Defaults match the decal / camera request shape.
#[derive(serde::Deserialize)]
#[serde(default)]
struct QualitySetRequest {
    #[serde(skip)]
    _cmd: String,
    setting: String,
    op: String,
}

impl Default for QualitySetRequest {
    fn default() -> Self {
        Self {
            _cmd: String::new(),
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
pub(super) fn handle_quality_set(text: &str) -> String {
    let req: QualitySetRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("quality-set: {e}")),
    };
    if req.setting.is_empty() {
        return error_reply("quality-set: missing 'setting'");
    }
    let op = match req.op.as_str() {
        "next" | "" => crate::assets::SettingOp::Next,
        "prev" => crate::assets::SettingOp::Prev,
        other => {
            return error_reply(&format!(
                "quality-set: unknown op '{other}' (use next | prev)"
            ));
        }
    };
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::QualitySet {
        setting: req.setting,
        op,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "queued": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("quality-set: timed out waiting for engine"),
    }
}

// Rebind a movement action to a key. `setting` is the engine key
// (`key_forward` / `key_backward` / `key_left` / `key_right` / `key_sprint` /
// `key_jump` / `key_interact`); `key` is a canonical `Key` variant name
// (`W`, `Space`, `Shift`, `Num1`, `Up`, ...).
#[derive(serde::Deserialize, Default)]
#[serde(default)]
struct RebindRequest {
    #[serde(skip)]
    _cmd: String,
    setting: String,
    key: String,
}

// Rebind a movement key live by injecting the same `Rebind` `SettingCommand` the
// settings menu emits on a capture, so the engine runs its real swap + persist +
// `set_keymap` path. `cn debug` only; lets a headless harness exercise the live
// rebind and screenshot the row label flipping. The reply fires once queued.
pub(super) fn handle_rebind(text: &str) -> String {
    let req: RebindRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("rebind: {e}")),
    };
    if req.setting.is_empty() {
        return error_reply("rebind: missing 'setting' (e.g. key_forward)");
    }
    // The canonical `Key` serializes to its variant name, so a JSON string
    // deserializes straight to it (W, Space, Shift, Num1, Up, ...).
    let key: crate::assets::Key =
        match serde_json::from_value(serde_json::Value::String(req.key.clone())) {
            Ok(k) => k,
            Err(_) => {
                return error_reply(&format!(
                    "rebind: unknown key '{}' (use a Key variant like W / Space / Shift)",
                    req.key
                ));
            }
        };
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::Rebind {
        setting: req.setting,
        key,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "queued": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("rebind: timed out waiting for engine"),
    }
}

// Despawn an authored placement by its declared name.
#[derive(serde::Deserialize, Default)]
#[serde(default)]
struct DespawnCmdRequest {
    #[serde(skip)]
    _cmd: String,
    name: String,
}

// Remove an authored placement (and its descendants) live by name: enqueue a
// Despawn command the per-frame drive forwards to `dispatch_despawn`, which
// sends a `DespawnRequest` the GraphicsSystem applies on its next step (hide the
// draw slots + despawn the entity, cascading to children). `cn debug` only; lets
// a headless harness remove an entity and screenshot it gone. The reply fires
// once the command is queued.
pub(super) fn handle_despawn(text: &str) -> String {
    let req: DespawnCmdRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("despawn: {e}")),
    };
    if req.name.trim().is_empty() {
        return error_reply("despawn: missing 'name'");
    }
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::Despawn {
        name: req.name,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "queued": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("despawn: timed out waiting for engine"),
    }
}

// Drive the story system: `action` is `start`, `advance`, `choose` / `slot`
// (with an `option` index), or one of the quick-row controls (`auto`,
// `skip`, `log`, `save`, `load`).
#[derive(serde::Deserialize, Default)]
#[serde(default)]
struct StoryCmdRequest {
    #[serde(skip)]
    _cmd: String,
    action: String,
    option: usize,
}

// Send the story system the same `StoryCommand` a stage click or key press
// fires, so a headless harness can start, advance, and choose through a
// story and screenshot each page. `cn debug` only. The reply fires once the
// command is queued.
pub(super) fn handle_story(text: &str) -> String {
    let req: StoryCmdRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("story: {e}")),
    };
    let command = match req.action.as_str() {
        "start" => crate::assets::StoryCommand::Start,
        "continue" => crate::assets::StoryCommand::Continue,
        "advance" => crate::assets::StoryCommand::Advance,
        "choose" => crate::assets::StoryCommand::Choose(req.option),
        "auto" => crate::assets::StoryCommand::ToggleAuto,
        "skip" => crate::assets::StoryCommand::ToggleSkip,
        "log" => crate::assets::StoryCommand::ToggleLog,
        "save" => crate::assets::StoryCommand::OpenSave,
        "load" => crate::assets::StoryCommand::OpenLoad,
        "slot" => crate::assets::StoryCommand::Slot(req.option),
        "pause" => crate::assets::StoryCommand::TogglePause,
        "settings" => crate::assets::StoryCommand::OpenSettings,
        "settings_back" => crate::assets::StoryCommand::CloseSettings,
        other => return error_reply(&format!("story: unknown action '{other}'")),
    };
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::Story {
        command,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "queued": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("story: timed out waiting for engine"),
    }
}

// Re-parent an authored placement. `child` is moved under `parent`; a null or
// omitted `parent` detaches the child to a root.
#[derive(serde::Deserialize, Default)]
#[serde(default)]
struct ReparentCmdRequest {
    #[serde(skip)]
    _cmd: String,
    child: String,
    parent: Option<String>,
}

// Re-parent an authored placement live by name: enqueue a Reparent command the
// per-frame drive forwards to `dispatch_reparent`, which sends a
// `ReparentRequest` the GraphicsSystem applies on its next step (re-point the
// Parent edge + recompose world matrices). `cn debug` only; lets a headless
// harness move an entity under a new parent and screenshot the result. The reply
// fires once queued.
pub(super) fn handle_reparent(text: &str) -> String {
    let req: ReparentCmdRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("reparent: {e}")),
    };
    if req.child.trim().is_empty() {
        return error_reply("reparent: missing 'child'");
    }
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::Reparent {
        child: req.child,
        // An empty / whitespace parent name detaches the child to a root.
        parent: req.parent.filter(|p| !p.trim().is_empty()),
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "queued": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("reparent: timed out waiting for engine"),
    }
}

// Spawn a runtime copy of an authored placement. `template` names the existing
// placement to copy; `name` is the new instance's identity. `position` /
// `rotation_deg` / `scale` place it (scale defaults to unit). A non-null
// `lifetime` (seconds) makes the instance auto-despawn after that long, which
// is what exercises draw-slot recycling.
#[derive(serde::Deserialize, Default)]
#[serde(default)]
struct SpawnCmdRequest {
    #[serde(skip)]
    _cmd: String,
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
pub(super) fn handle_spawn(text: &str) -> String {
    let req: SpawnCmdRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("spawn: {e}")),
    };
    if req.template.trim().is_empty() {
        return error_reply("spawn: missing 'template'");
    }
    if req.name.trim().is_empty() {
        return error_reply("spawn: missing 'name'");
    }
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::Spawn {
        template: req.template,
        name: req.name,
        position: req.position,
        rotation_deg: req.rotation_deg,
        scale: req.scale,
        lifetime: req.lifetime,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "queued": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("spawn: timed out waiting for engine"),
    }
}

// Move the active camera by a per-frame delta over a span of frames. All delta
// fields default to 0 and `frames` to 0 (an indefinite hold cleared by
// `camera-stop`); a profiling harness can then sustain motion mid-screenshot to
// surface temporal effects. `yaw` / `pitch` are radians.
#[derive(serde::Deserialize)]
#[serde(default)]
struct CameraMoveRequest {
    #[serde(skip)]
    _cmd: String,
    forward: f32,
    right: f32,
    up: f32,
    yaw: f32,
    pitch: f32,
    frames: u32,
}

impl Default for CameraMoveRequest {
    fn default() -> Self {
        Self {
            _cmd: String::new(),
            forward: 0.0,
            right: 0.0,
            up: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            frames: 0,
        }
    }
}

pub(super) fn handle_camera_move(text: &str) -> String {
    let req: CameraMoveRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("camera-move: {e}")),
    };
    let args = super::runtime_spawn::CameraMoveArgs {
        forward: req.forward,
        right: req.right,
        up: req.up,
        yaw: req.yaw,
        pitch: req.pitch,
        frames: req.frames,
    };
    let frames = args.frames;
    let holding = frames == 0;
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::CameraMove {
        args,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => {
            serde_json::json!({ "ok": true, "frames": frames, "holding": holding }).to_string()
        }
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("camera-move: timed out waiting for engine"),
    }
}

pub(super) fn handle_camera_stop() -> String {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::CameraStop { reply: tx });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "stopped": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("camera-stop: timed out waiting for engine"),
    }
}

pub(super) fn error_reply(msg: &str) -> String {
    serde_json::json!({ "ok": false, "error": msg }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_set_request_parses_full_payload() {
        let req: CameraSetRequest = serde_json::from_str(
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
        let req: CameraSetRequest = serde_json::from_str(
            r#"{"cmd":"camera-set","position":[0.0,1.0,0.0],"yaw":0.0,"pitch":0.0}"#,
        )
        .expect("payload without fov parses");
        assert_eq!(req.position, [0.0, 1.0, 0.0]);
        assert!(req.fov_y_degrees.is_none());
    }

    #[test]
    fn camera_set_request_defaults_for_missing_fields() {
        let req: CameraSetRequest =
            serde_json::from_str(r#"{"cmd":"camera-set"}"#).expect("bare command parses");
        assert_eq!(req.position, [0.0, 0.0, 0.0]);
        assert_eq!(req.yaw, 0.0);
        assert_eq!(req.pitch, 0.0);
        assert!(req.fov_y_degrees.is_none());
    }

    #[test]
    fn camera_set_request_rejects_malformed() {
        // position must be three numbers; a string is a hard parse error.
        assert!(
            serde_json::from_str::<CameraSetRequest>(r#"{"cmd":"camera-set","position":"nope"}"#)
                .is_err()
        );
    }

    #[test]
    fn camera_move_request_parses_full_payload() {
        let req: CameraMoveRequest = serde_json::from_str(
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
        let req: CameraMoveRequest =
            serde_json::from_str(r#"{"cmd":"camera-move"}"#).expect("bare command parses");
        assert_eq!(req.forward, 0.0);
        assert_eq!(req.frames, 0);
    }

    #[test]
    fn camera_move_request_rejects_malformed() {
        // frames must be an unsigned integer; a string is a hard parse error.
        assert!(
            serde_json::from_str::<CameraMoveRequest>(r#"{"cmd":"camera-move","frames":"lots"}"#)
                .is_err()
        );
    }

    #[test]
    fn story_request_parses_action_and_option() {
        let req: StoryCmdRequest =
            serde_json::from_str(r#"{"cmd":"story","action":"choose","option":1}"#)
                .expect("valid parses");
        assert_eq!(req.action, "choose");
        assert_eq!(req.option, 1);
        let req: StoryCmdRequest =
            serde_json::from_str(r#"{"cmd":"story","action":"advance"}"#).expect("valid parses");
        assert_eq!(req.action, "advance");
        assert_eq!(req.option, 0);
    }

    #[test]
    fn despawn_request_parses_name() {
        let req: DespawnCmdRequest =
            serde_json::from_str(r#"{"cmd":"despawn","name":"crate_a"}"#).expect("valid parses");
        assert_eq!(req.name, "crate_a");
    }

    #[test]
    fn despawn_request_defaults_to_empty_name() {
        let req: DespawnCmdRequest =
            serde_json::from_str(r#"{"cmd":"despawn"}"#).expect("bare command parses");
        assert!(req.name.is_empty());
    }

    #[test]
    fn reparent_request_parses_child_and_parent() {
        let req: ReparentCmdRequest =
            serde_json::from_str(r#"{"cmd":"reparent","child":"box_a","parent":"frame"}"#)
                .expect("valid parses");
        assert_eq!(req.child, "box_a");
        assert_eq!(req.parent.as_deref(), Some("frame"));
    }

    #[test]
    fn reparent_request_parent_optional() {
        let req: ReparentCmdRequest = serde_json::from_str(r#"{"cmd":"reparent","child":"box_a"}"#)
            .expect("bare parent parses");
        assert_eq!(req.child, "box_a");
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
    // before anything is enqueued, so these run without touching the
    // process-global queues.

    #[test]
    fn decal_add_rejects_malformed_json() {
        assert_err_reply(&handle_decal_add("not json"), "decal-add");
    }

    #[test]
    fn decal_remove_rejects_missing_id() {
        assert_err_reply(
            &handle_decal_remove(r#"{"cmd":"decal-remove"}"#),
            "decal-remove",
        );
    }

    #[test]
    fn emitter_add_rejects_malformed_json() {
        assert_err_reply(&handle_emitter_add("not json"), "emitter-add");
    }

    #[test]
    fn emitter_remove_rejects_missing_id() {
        assert_err_reply(&handle_emitter_remove("{}"), "emitter-remove");
    }

    #[test]
    fn anim_crossfade_rejects_malformed_json() {
        assert_err_reply(&handle_anim_crossfade("not json", &[]), "anim-crossfade");
    }

    #[test]
    fn anim_crossfade_requires_a_target() {
        assert_err_reply(&handle_anim_crossfade("{}", &[]), "missing 'target'");
    }

    #[test]
    fn anim_crossfade_rejects_an_unknown_target_name() {
        let names = vec!["hero".to_string()];
        let reply = handle_anim_crossfade(r#"{"target":"villain","weights":[1.0]}"#, &names);
        assert_err_reply(&reply, "unknown asset name 'villain'");
    }

    #[test]
    fn anim_param_rejects_malformed_json() {
        assert_err_reply(&handle_anim_param("not json", &[]), "anim-param");
    }

    #[test]
    fn anim_param_requires_a_parameter_name() {
        assert_err_reply(
            &handle_anim_param(r#"{"target":"hero"}"#, &[]),
            "missing 'name'",
        );
    }

    #[test]
    fn anim_param_requires_a_target() {
        assert_err_reply(
            &handle_anim_param(r#"{"name":"speed"}"#, &[]),
            "missing 'target'",
        );
    }

    #[test]
    fn anim_param_rejects_an_unknown_target_name() {
        let names = vec!["hero".to_string()];
        let reply = handle_anim_param(r#"{"target":"villain","name":"speed","value":1.0}"#, &names);
        assert_err_reply(&reply, "unknown asset name 'villain'");
    }

    #[test]
    fn anim_state_rejects_malformed_json() {
        assert_err_reply(&handle_anim_state("not json", &[]), "anim-state");
    }

    #[test]
    fn anim_state_requires_a_target() {
        assert_err_reply(&handle_anim_state("{}", &[]), "missing 'target'");
    }

    #[test]
    fn anim_state_rejects_an_unknown_target_name() {
        let names = vec!["hero".to_string()];
        assert_err_reply(
            &handle_anim_state(r#"{"target":"villain"}"#, &names),
            "unknown asset name 'villain'",
        );
    }

    #[test]
    fn resolve_target_maps_a_name_to_its_table_index() {
        let names = vec!["a".to_string(), "b".to_string()];
        let id = resolve_target("cmd", "b", &names).expect("known name resolves");
        assert_eq!(id, crate::ecs::asset_id::AssetId(1));
    }

    #[test]
    fn screenshot_rejects_malformed_json() {
        assert_err_reply(&handle_screenshot("not json"), "screenshot");
    }

    #[test]
    fn screenshot_requires_a_path() {
        assert_err_reply(&handle_screenshot("{}"), "missing 'path'");
        assert_err_reply(&handle_screenshot(r#"{"path":"   "}"#), "missing 'path'");
    }

    #[test]
    fn camera_set_handler_rejects_malformed_json() {
        assert_err_reply(&handle_camera_set(r#"{"position":"nope"}"#), "camera-set");
    }

    #[test]
    fn camera_move_handler_rejects_malformed_json() {
        assert_err_reply(&handle_camera_move(r#"{"frames":"lots"}"#), "camera-move");
    }

    #[test]
    fn quality_set_rejects_malformed_json() {
        assert_err_reply(&handle_quality_set("not json"), "quality-set");
    }

    #[test]
    fn quality_set_requires_a_setting() {
        assert_err_reply(&handle_quality_set("{}"), "missing 'setting'");
    }

    #[test]
    fn quality_set_rejects_an_unknown_op() {
        assert_err_reply(
            &handle_quality_set(r#"{"setting":"taa","op":"sideways"}"#),
            "unknown op 'sideways'",
        );
    }

    #[test]
    fn rebind_rejects_malformed_json() {
        assert_err_reply(&handle_rebind("not json"), "rebind");
    }

    #[test]
    fn rebind_requires_a_setting() {
        assert_err_reply(&handle_rebind(r#"{"key":"W"}"#), "missing 'setting'");
    }

    #[test]
    fn rebind_rejects_an_unknown_key_name() {
        assert_err_reply(
            &handle_rebind(r#"{"setting":"key_forward","key":"NotAKey"}"#),
            "unknown key 'NotAKey'",
        );
    }

    #[test]
    fn despawn_rejects_malformed_json() {
        assert_err_reply(&handle_despawn("not json"), "despawn");
    }

    #[test]
    fn despawn_requires_a_name() {
        assert_err_reply(&handle_despawn("{}"), "missing 'name'");
        assert_err_reply(&handle_despawn(r#"{"name":"  "}"#), "missing 'name'");
    }

    #[test]
    fn story_rejects_malformed_json() {
        assert_err_reply(&handle_story("not json"), "story");
    }

    #[test]
    fn story_rejects_an_unknown_action() {
        assert_err_reply(
            &handle_story(r#"{"action":"dance"}"#),
            "unknown action 'dance'",
        );
    }

    #[test]
    fn reparent_rejects_malformed_json() {
        assert_err_reply(&handle_reparent("not json"), "reparent");
    }

    #[test]
    fn reparent_requires_a_child() {
        assert_err_reply(&handle_reparent("{}"), "missing 'child'");
        assert_err_reply(&handle_reparent(r#"{"child":" "}"#), "missing 'child'");
    }

    #[test]
    fn spawn_rejects_malformed_json() {
        assert_err_reply(&handle_spawn("not json"), "spawn");
    }

    #[test]
    fn spawn_requires_template_and_name() {
        assert_err_reply(&handle_spawn("{}"), "missing 'template'");
        assert_err_reply(&handle_spawn(r#"{"template":"crate_a"}"#), "missing 'name'");
    }

    // Success-path handler tests. Each handler blocks on a reply channel the
    // per-frame drive normally fulfils; here a worker thread runs the handler
    // while the test drains the process-global queue and answers in its place.
    // Serialised behind the shared test lock because the queue is
    // process-global; any unrelated command drained alongside is re-enqueued
    // untouched.

    use crate::app::pending::test_support;
    use crate::debug::runtime_spawn::{self, RuntimeCommand};

    // A heavily loaded test host can stall either thread past the handler's
    // one-second engine timeout; when the handler reports that timeout the
    // whole exchange is retried, so the tests assert the reply semantics
    // rather than the scheduler. Reply sends never unwrap for the same
    // reason: a timed-out handler has already dropped its receiver.
    fn drive_runtime_handler(
        handler: impl Fn() -> String + Send + Sync + 'static,
        mut reply: impl FnMut(RuntimeCommand) -> Option<RuntimeCommand>,
    ) -> String {
        let handler = std::sync::Arc::new(handler);
        for _ in 0..5 {
            let h = std::sync::Arc::clone(&handler);
            let worker = std::thread::spawn(move || h());
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            let mut replied = false;
            while !worker.is_finished() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "handler never returned"
                );
                for cmd in runtime_spawn::drain() {
                    if replied {
                        runtime_spawn::enqueue(cmd);
                    } else if let Some(foreign) = reply(cmd) {
                        runtime_spawn::enqueue(foreign);
                    } else {
                        replied = true;
                    }
                }
                std::thread::yield_now();
            }
            let response = worker.join().expect("handler thread panicked");
            if !response.contains("timed out waiting for engine") {
                return response;
            }
            // The timed-out attempt may have left its command queued; answer
            // it into the void (the receiver is gone) and keep anything
            // foreign, then try again.
            for cmd in runtime_spawn::drain() {
                if let Some(foreign) = reply(cmd) {
                    runtime_spawn::enqueue(foreign);
                }
            }
        }
        panic!("handler kept timing out under load");
    }

    #[test]
    fn decal_add_round_trips_args_and_reports_the_new_id() {
        let _guard = test_support::lock();
        let reply = drive_runtime_handler(
            || {
                handle_decal_add(
                    r#"{"texture":"grid","position":[1.0,2.0,3.0],"size":[2.0,2.0,2.0],"tint":[1.0,0.0,0.0,1.0]}"#,
                )
            },
            |cmd| match cmd {
                RuntimeCommand::DecalAdd { args, reply } => {
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
        let _guard = test_support::lock();
        let reply = drive_runtime_handler(
            || handle_decal_add("{}"),
            |cmd| match cmd {
                RuntimeCommand::DecalAdd { reply, .. } => {
                    let _ = reply.send(Err("no free decal slot".to_string()));
                    None
                }
                other => Some(other),
            },
        );
        assert_err_reply(&reply, "no free decal slot");
    }

    #[test]
    fn decal_remove_reports_removed() {
        let _guard = test_support::lock();
        let reply = drive_runtime_handler(
            || handle_decal_remove(r#"{"id":3}"#),
            |cmd| match cmd {
                RuntimeCommand::DecalRemove { id, reply } => {
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
        let _guard = test_support::lock();
        let reply = drive_runtime_handler(
            || handle_emitter_add("{}"),
            |cmd| match cmd {
                RuntimeCommand::EmitterAdd { args, reply } => {
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
        let _guard = test_support::lock();
        let reply = drive_runtime_handler(
            || handle_emitter_remove(r#"{"id":5}"#),
            |cmd| match cmd {
                RuntimeCommand::EmitterRemove { id, reply } => {
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
        let _guard = test_support::lock();
        let reply = drive_runtime_handler(
            || handle_screenshot(r#"{"path":"shot.png"}"#),
            |cmd| match cmd {
                RuntimeCommand::Screenshot { path, reply } => {
                    assert_eq!(path, "shot.png");
                    let _ = reply.send(Ok("shot.png".to_string()));
                    None
                }
                other => Some(other),
            },
        );
        assert!(reply.contains(r#""path":"shot.png""#), "got: {reply}");
    }

    #[test]
    fn camera_set_round_trips_the_pose() {
        let _guard = test_support::lock();
        let reply = drive_runtime_handler(
            || {
                handle_camera_set(
                    r#"{"position":[1.0,2.0,3.0],"yaw":0.5,"pitch":-0.25,"fov_y_degrees":60.0}"#,
                )
            },
            |cmd| match cmd {
                RuntimeCommand::CameraSet { args, reply } => {
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
        let _guard = test_support::lock();
        let reply = drive_runtime_handler(
            || handle_camera_move(r#"{"forward":1.5,"frames":3}"#),
            |cmd| match cmd {
                RuntimeCommand::CameraMove { args, reply } => {
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
        let _guard = test_support::lock();
        let reply = drive_runtime_handler(
            || handle_camera_move("{}"),
            |cmd| match cmd {
                RuntimeCommand::CameraMove { reply, .. } => {
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
        let _guard = test_support::lock();
        let reply = drive_runtime_handler(handle_camera_stop, |cmd| match cmd {
            RuntimeCommand::CameraStop { reply } => {
                let _ = reply.send(Ok(()));
                None
            }
            other => Some(other),
        });
        assert!(reply.contains(r#""stopped":true"#), "got: {reply}");
    }

    #[test]
    fn quality_set_maps_ops_and_reports_queued() {
        let _guard = test_support::lock();
        let reply = drive_runtime_handler(
            || handle_quality_set(r#"{"setting":"taa","op":"prev"}"#),
            |cmd| match cmd {
                RuntimeCommand::QualitySet { setting, op, reply } => {
                    assert_eq!(setting, "taa");
                    assert_eq!(op, crate::assets::SettingOp::Prev);
                    let _ = reply.send(Ok(()));
                    None
                }
                other => Some(other),
            },
        );
        assert!(reply.contains(r#""queued":true"#), "got: {reply}");

        // An omitted op defaults to Next.
        let reply = drive_runtime_handler(
            || handle_quality_set(r#"{"setting":"ssao"}"#),
            |cmd| match cmd {
                RuntimeCommand::QualitySet { op, reply, .. } => {
                    assert_eq!(op, crate::assets::SettingOp::Next);
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
        let _guard = test_support::lock();
        let reply = drive_runtime_handler(
            || handle_rebind(r#"{"setting":"key_forward","key":"Space"}"#),
            |cmd| match cmd {
                RuntimeCommand::Rebind {
                    setting,
                    key,
                    reply,
                } => {
                    assert_eq!(setting, "key_forward");
                    assert_eq!(key, crate::assets::Key::Space);
                    let _ = reply.send(Ok(()));
                    None
                }
                other => Some(other),
            },
        );
        assert!(reply.contains(r#""queued":true"#), "got: {reply}");
    }

    #[test]
    fn despawn_round_trips_the_name() {
        let _guard = test_support::lock();
        let reply = drive_runtime_handler(
            || handle_despawn(r#"{"name":"crate_a"}"#),
            |cmd| match cmd {
                RuntimeCommand::Despawn { name, reply } => {
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
        use crate::assets::StoryCommand;
        let _guard = test_support::lock();
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
                move || handle_story(&text),
                |cmd| match cmd {
                    RuntimeCommand::Story { command, reply } => {
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
        use crate::assets::StoryCommand;
        let _guard = test_support::lock();
        for (action, expected) in [
            ("choose", StoryCommand::Choose(2)),
            ("slot", StoryCommand::Slot(2)),
        ] {
            let text = format!(r#"{{"action":"{action}","option":2}}"#);
            let reply = drive_runtime_handler(
                move || handle_story(&text),
                |cmd| match cmd {
                    RuntimeCommand::Story { command, reply } => {
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
        let _guard = test_support::lock();
        let reply = drive_runtime_handler(
            || handle_reparent(r#"{"child":"box_a","parent":"  "}"#),
            |cmd| match cmd {
                RuntimeCommand::Reparent {
                    child,
                    parent,
                    reply,
                } => {
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
        let _guard = test_support::lock();
        let reply = drive_runtime_handler(
            || handle_reparent(r#"{"child":"box_a","parent":"frame"}"#),
            |cmd| match cmd {
                RuntimeCommand::Reparent { parent, reply, .. } => {
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
        let _guard = test_support::lock();
        let reply = drive_runtime_handler(
            || {
                handle_spawn(
                    r#"{"template":"crate_a","name":"crate_b","position":[1.0,0.0,-1.0],"rotation_deg":[0.0,90.0,0.0],"scale":[2.0,2.0,2.0],"lifetime":2.5}"#,
                )
            },
            |cmd| match cmd {
                RuntimeCommand::Spawn {
                    template,
                    name,
                    position,
                    rotation_deg,
                    scale,
                    lifetime,
                    reply,
                } => {
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

    // Animation handler success paths: these enqueue onto the runtime crate's
    // animation queue, which only `AnimationSystem::apply_runtime_commands`
    // can drain. Drive a real system built from a small world while the
    // handler blocks, exactly as the per-frame debug drive would.

    fn anim_clip(name: &str, duration: f32) -> crate::assets::Animation {
        let mut a: crate::assets::Animation = serde_json::from_value(serde_json::json!({
            "target": "hero",
            "duration": duration,
            "looping": true,
        }))
        .unwrap();
        a.asset_id = crate::ecs::asset_id::intern(name);
        a
    }

    fn hero_graph() -> crate::assets::AnimGraph {
        let mut g: crate::assets::AnimGraph = serde_json::from_value(serde_json::json!({
            "target": "hero",
            "parameters": [{"name": "speed", "default": 0.0}],
            "initial": "idle",
            "states": [
                {"name": "idle", "clip": "idle_clip"},
                {"name": "run", "clip": "run_clip"}
            ],
            "transitions": [
                {"from": "idle", "to": "run",
                 "conditions": [{"parameter": "speed", "op": "gt", "value": 0.5}]},
                {"from": "run", "to": "idle",
                 "conditions": [{"parameter": "speed", "op": "le", "value": 0.5}]}
            ]
        }))
        .unwrap();
        g.asset_id = crate::ecs::asset_id::intern("hero_graph");
        g
    }

    fn graph_world() -> crate::ecs::World {
        let mut world = crate::ecs::World::new_empty();
        world.add_component(anim_clip("idle_clip", 1.0));
        world.add_component(anim_clip("run_clip", 0.8));
        world.add_component(hero_graph());
        world.start().unwrap();
        world
    }

    fn flat_world() -> crate::ecs::World {
        let mut world = crate::ecs::World::new_empty();
        world.add_component(anim_clip("wave_clip", 1.0));
        world.add_component(anim_clip("bow_clip", 0.5));
        world.start().unwrap();
        world
    }

    fn with_anim<R>(
        world: &mut crate::ecs::World,
        f: impl FnOnce(&mut crate::gfx::animation::AnimationSystem) -> R,
    ) -> R {
        for system in world.systems_mut() {
            if let crate::ecs::SystemAsset::AnimationSystem(anim) = system {
                return f(anim);
            }
        }
        panic!("AnimationSystem not constructed");
    }

    // Same retry rationale as `drive_runtime_handler`: a stalled test host
    // can trip the handler's engine timeout, which is a scheduler artifact,
    // not the semantics under test.
    fn drive_anim_handler(
        world: &mut crate::ecs::World,
        handler: impl Fn() -> String + Send + Sync + 'static,
    ) -> String {
        let handler = std::sync::Arc::new(handler);
        for _ in 0..5 {
            let h = std::sync::Arc::clone(&handler);
            let worker = std::thread::spawn(move || h());
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while !worker.is_finished() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "anim handler never returned"
                );
                with_anim(world, |anim| anim.apply_runtime_commands());
                std::thread::yield_now();
            }
            // One final drain so a command left behind by a handler timeout
            // can never leak into another test.
            with_anim(world, |anim| anim.apply_runtime_commands());
            let response = worker.join().expect("handler thread panicked");
            if !response.contains("timed out waiting for engine") {
                return response;
            }
        }
        panic!("anim handler kept timing out under load");
    }

    #[test]
    fn anim_param_queues_a_graph_parameter_write() {
        let _guard = test_support::lock();
        let mut world = graph_world();
        let names = crate::ecs::asset_id::name_table();
        let reply = drive_anim_handler(&mut world, move || {
            handle_anim_param(r#"{"target":"hero","name":"speed","value":1.0}"#, &names)
        });
        assert!(reply.contains(r#""queued":true"#), "got: {reply}");
    }

    #[test]
    fn anim_param_surfaces_an_unknown_parameter() {
        let _guard = test_support::lock();
        let mut world = graph_world();
        let names = crate::ecs::asset_id::name_table();
        let reply = drive_anim_handler(&mut world, move || {
            handle_anim_param(r#"{"target":"hero","name":"altitude","value":1.0}"#, &names)
        });
        assert_err_reply(&reply, "no parameter 'altitude'");
    }

    #[test]
    fn anim_state_reports_the_live_graph_state() {
        let _guard = test_support::lock();
        let mut world = graph_world();
        let names = crate::ecs::asset_id::name_table();
        let reply = drive_anim_handler(&mut world, move || {
            handle_anim_state(r#"{"target":"hero"}"#, &names)
        });
        assert!(reply.contains(r#""ok":true"#), "got: {reply}");
        assert!(reply.contains(r#""state":"idle""#), "got: {reply}");
        assert!(reply.contains(r#""params":{"speed":0.0}"#), "got: {reply}");
    }

    #[test]
    fn anim_crossfade_queues_on_a_flat_target() {
        let _guard = test_support::lock();
        let mut world = flat_world();
        let names = crate::ecs::asset_id::name_table();
        let reply = drive_anim_handler(&mut world, move || {
            handle_anim_crossfade(
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
        let names = crate::ecs::asset_id::name_table();
        let reply = drive_anim_handler(&mut world, move || {
            handle_anim_crossfade(r#"{"target":"hero","weights":[1.0,0.0]}"#, &names)
        });
        assert_err_reply(&reply, "graph-driven");
    }
}
