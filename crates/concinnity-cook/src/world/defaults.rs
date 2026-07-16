// src/world/defaults.rs
// Engine-default injection: complete a rendering world with the standard
// assets it does not declare itself. A world that renders (has a
// GraphicsConfig after the first companion round) receives a DebugHud with
// its chip TextLabels and font, a StatHud with its chips when the world
// declares a MainMenu (the menu's performance-stats toggles drive them), and,
// when an EnvironmentMap is present, the sky mesh that displays it.
//
// Override and opt-out rules:
//   - Declaring an asset of the same type pre-empts the whole default (an
//     authored StatHud suppresses the synthesized one; its unset label fields
//     are still filled in with default chips).
//   - Declaring an asset with a default's exact name and type replaces that
//     one piece (an authored `fps_chip` TextLabel restyles the fps chip).
//   - A default's name landing on an unrelated asset type is a hard error.
//   - An `EngineDefaults` asset turns individual defaults off entirely.

use super::expand::{ExpandReport, asset_name, type_norm};
use crate::assets::EngineDefaults;

pub(crate) const HUD_FONT_NAME: &str = "hud_font";
const HUD_FONT_SIZE_PX: u32 = 20;

// The sky mesh must stay inside the camera far plane.
const SKY_SIZE_MAX: f32 = 400.0;
const SKY_FAR_FRACTION: f32 = 0.9;
const CAMERA_FAR_DEFAULT: f32 = 200.0;

// Consume every EngineDefaults entry and apply the enabled defaults to the
// world. Runs after the content expansion passes and the first companion round
// (so the GraphicsConfig render marker is present when implied) and before
// menu expansion (so the MainMenu lines that gate the StatHud are still
// visible).
pub(crate) fn inject_engine_defaults(
    assets: &mut Vec<serde_json::Value>,
    report: &mut ExpandReport,
) -> Result<(), String> {
    let toggles = drain_toggles(assets)?;
    let renders = assets.iter().any(|v| type_norm(v) == "graphicsconfig");

    if toggles.sky {
        inject_sky(assets, report)?;
    }
    if !renders {
        return Ok(());
    }
    // The StatHud exists to serve the menu's performance-stats toggles, so it
    // is synthesized only for worlds with a MainMenu; an authored StatHud
    // still gets its unset labels filled in anywhere.
    let has_menu = assets.iter().any(|v| type_norm(v) == "mainmenu");
    let has_stat_hud = assets.iter().any(|v| type_norm(v) == "stathud");
    if toggles.hud && (has_menu || has_stat_hud) {
        inject_hud(
            assets,
            report,
            "hud",
            "StatHud",
            "stat_hud",
            &[
                ("fps_label", "fps_chip"),
                ("vram_label", "vram_chip"),
                ("ram_label", "ram_chip"),
                ("ev_label", "ev_chip"),
                ("edr_label", "edr_chip"),
            ],
        )?;
    }
    if toggles.debug_hud {
        inject_hud(
            assets,
            report,
            "debug_hud",
            "DebugHud",
            "debug_hud",
            &[
                ("passes_label", "passes_chip"),
                ("mouse_label", "mouse_chip"),
                ("camera_label", "camera_chip"),
                ("sys_label", "sys_chip"),
            ],
        )?;
    }
    // A story world with no menu of its own gets an Escape-toggled pause menu.
    // Injected after the HUD defaults so it does not turn on the StatHud: the
    // trimmed pause menu carries no performance-stats toggles to drive it.
    if toggles.story_pause_menu {
        inject_story_pause_menu(assets, report)?;
    }
    Ok(())
}

// Give a story world an Escape pause menu (Resume / Save / Load / Settings /
// Quit) when it plays a Story but declares no MainMenu. The menu targets the
// story's own title screen for its Quit item and uses the trimmed settings
// profile: a 2D story renders no scene, so only window / output and volume are
// worth configuring. Runs after story expansion, so the compiled Story asset
// (named for the sanitized import) and its `<name>_title` screen are present.
fn inject_story_pause_menu(
    assets: &mut Vec<serde_json::Value>,
    report: &mut ExpandReport,
) -> Result<(), String> {
    // An authored MainMenu takes over the pause role; only inject when the
    // world declares none of its own.
    if assets.iter().any(|v| type_norm(v) == "mainmenu") {
        return Ok(());
    }
    let Some(prefix) = assets
        .iter()
        .find(|v| type_norm(v) == "story")
        .map(asset_name)
        .filter(|n| !n.is_empty())
    else {
        return Ok(());
    };

    let name = format!("{}_pause", prefix);
    // No MainMenu exists (checked above), so this can only flag a same-name
    // collision with an unrelated asset, which is a hard error rather than a
    // silent skip.
    if name_claimed(assets, &name, "MainMenu")? {
        return Ok(());
    }

    // Main Menu returns to the story's own title screen; it is offered only
    // when the story has one (a story built without a title screen has nowhere
    // to return to, so its pause menu skips the item). Quit always exits to
    // desktop.
    let title_screen = format!("{}_title", prefix);
    let has_title = assets
        .iter()
        .any(|v| type_norm(v) == "screen" && asset_name(v) == title_screen);

    let mut items = vec![
        serde_json::json!({ "label": "Resume", "action": "story:pause" }),
        serde_json::json!({ "label": "Save", "action": "story:save" }),
        serde_json::json!({ "label": "Load", "action": "story:load" }),
        serde_json::json!({ "label": "Settings", "action": "story:settings" }),
    ];
    if has_title {
        items.push(serde_json::json!({
            "label": "Main Menu",
            "action": format!("screen:show:{}", title_screen),
        }));
    }
    items.push(serde_json::json!({ "label": "Quit", "action": "quit" }));

    // The story system drives the pause and settings navigation (Resume /
    // Settings / the settings Back), so closing returns to the stage instead of
    // dismissing to an empty screen. A translucent backdrop rather than the
    // opaque MainMenu default, so the menu reads as an overlay. `toggle_key` is
    // empty: Escape is a separate story-driven binding (below), not the
    // MainMenu's own screen toggle. `settings_back_action` both routes Back
    // through the story and forces the settings screen to be generated.
    let args = serde_json::json!({
        "toggle_key": "",
        "dim": [0.0, 0.0, 0.0, 0.6],
        "settings_profile": "minimal",
        "settings_back_action": "story:settings_back",
        "items": items,
    });
    inject(assets, report, "story_pause_menu", &name, "MainMenu", args);

    // Escape toggles the pause through the story system.
    inject(
        assets,
        report,
        "story_pause_menu",
        &format!("{}_key", name),
        "KeyBinding",
        serde_json::json!({ "key": "Escape", "action": "story:pause" }),
    );

    // Point the story at its pause menu and settings entry screens so the story
    // system can show them. Those screens are generated later by the menu
    // expansion; the names are stable and intern to the same ids.
    patch_story_scaffold(assets, prefix.as_str(), &name);
    Ok(())
}

// Add the injected pause menu's screen (`<menu>`) and settings entry screen
// (`<menu>_settings_video`) to the Story asset's scaffold, so the story system
// resolves them like every other stage reference.
fn patch_story_scaffold(assets: &mut [serde_json::Value], story_name: &str, menu_name: &str) {
    let pause_screen = menu_name.to_string();
    let settings_screen = format!("{}_settings_video", menu_name);
    for v in assets.iter_mut() {
        if type_norm(v) != "story" || asset_name(v) != story_name {
            continue;
        }
        // Best-effort: a build-generated Story always carries an object args +
        // scaffold, but a hand-authored one might be malformed. Skip the patch
        // rather than panic; the malformed Story surfaces its own error at
        // validation / compile.
        let Some(args) = v.get_mut("args").and_then(|a| a.as_object_mut()) else {
            return;
        };
        let Some(scaffold) = args
            .entry("scaffold")
            .or_insert_with(|| serde_json::json!({}))
            .as_object_mut()
        else {
            return;
        };
        scaffold.insert("pause".to_string(), serde_json::json!(pause_screen));
        scaffold.insert("settings".to_string(), serde_json::json!(settings_screen));
        return;
    }
}

// Remove EngineDefaults entries from the world (they are build directives, not
// runtime assets) and merge them into one set of toggles. More than one entry
// is ambiguous and rejected.
fn drain_toggles(assets: &mut Vec<serde_json::Value>) -> Result<EngineDefaults, String> {
    let mut found: Option<(String, EngineDefaults)> = None;
    let mut result = Vec::with_capacity(assets.len());
    for value in assets.drain(..) {
        if type_norm(&value) != "enginedefaults" {
            result.push(value);
            continue;
        }
        let name = asset_name(&value);
        if let Some((first, _)) = &found {
            return Err(format!(
                "EngineDefaults '{}': the world already declares EngineDefaults '{}'; \
                 declare at most one",
                name, first
            ));
        }
        let args = value
            .get("args")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        let toggles: EngineDefaults = serde_json::from_value(args)
            .map_err(|e| format!("EngineDefaults '{}': invalid args: {}", name, e))?;
        found = Some((name, toggles));
    }
    *assets = result;
    Ok(found.map(|(_, t)| t).unwrap_or_default())
}

// Whether the world already provides `name` as an asset of `asset_type` (the
// user's version replaces the default). A name held by a different type is a
// hard error: the default cannot be injected and silently skipping it would
// hide the conflict.
fn name_claimed(
    assets: &[serde_json::Value],
    name: &str,
    asset_type: &str,
) -> Result<bool, String> {
    for v in assets {
        if asset_name(v) != name {
            continue;
        }
        if type_norm(v) == asset_type.to_lowercase().replace('_', "") {
            return Ok(true);
        }
        return Err(format!(
            "engine default '{}' ({}) collides with your {} asset of the same name; \
             rename that asset or disable the default with an EngineDefaults entry",
            name,
            asset_type,
            v.get("type").and_then(|t| t.as_str()).unwrap_or("?"),
        ));
    }
    Ok(false)
}

// Push one injected asset and record it in the report.
fn inject(
    assets: &mut Vec<serde_json::Value>,
    report: &mut ExpandReport,
    injected_by: &'static str,
    name: &str,
    asset_type: &str,
    args: serde_json::Value,
) {
    assets.push(serde_json::json!({
        "name": name,
        "type": asset_type,
        "args": args.clone(),
    }));
    report.record(name, asset_type, args, injected_by);
}

// The chip TextLabel every injected HUD readout writes into.
fn chip_args() -> serde_json::Value {
    serde_json::json!({
        "font": HUD_FONT_NAME,
        "scale": 0.7,
        "color": [1, 1, 1],
        "background": [0.0, 0.18, 0.32, 0.85],
        "padding": 5,
    })
}

// Synthesize a HUD asset when the world declares none, fill its unset label
// fields with the default chip names, and inject any chips (and their font)
// those fields now reference but the world does not provide.
fn inject_hud(
    assets: &mut Vec<serde_json::Value>,
    report: &mut ExpandReport,
    injected_by: &'static str,
    hud_type: &str,
    default_name: &str,
    fields: &[(&str, &str)],
) -> Result<(), String> {
    let hud_type_norm = hud_type.to_lowercase().replace('_', "");
    let hud_index = match assets.iter().position(|v| type_norm(v) == hud_type_norm) {
        Some(i) => i,
        None => {
            if name_claimed(assets, default_name, hud_type)? {
                // Unreachable in practice: a same-name same-type asset would
                // have matched the type scan above.
                return Ok(());
            }
            inject(
                assets,
                report,
                injected_by,
                default_name,
                hud_type,
                serde_json::json!({}),
            );
            assets.len() - 1
        }
    };

    // Fill unset label fields on the HUD (authored or synthesized) and collect
    // the chip names those fields now reference.
    let mut needed_chips: Vec<&str> = Vec::new();
    {
        let hud = &mut assets[hud_index];
        if hud.get("args").is_none() {
            hud["args"] = serde_json::json!({});
        }
        let args = hud
            .get_mut("args")
            .and_then(|a| a.as_object_mut())
            .ok_or_else(|| format!("{} '{}': args must be an object", hud_type, default_name))?;
        for (field, chip) in fields {
            let unset = args
                .get(*field)
                .and_then(|v| v.as_str())
                .map(|s| s.is_empty())
                .unwrap_or(true);
            if unset {
                args.insert(field.to_string(), serde_json::json!(chip));
                needed_chips.push(chip);
            }
        }
    }

    // Keep the report's copy of a synthesized HUD in sync with the filled-in
    // label fields, so the lock shows the args the blob actually carries.
    if let Some(entry) = report
        .injected
        .iter_mut()
        .find(|i| i.name == default_name && i.asset_type == hud_type)
    {
        entry.args = assets[hud_index]
            .get("args")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
    }

    let mut chip_injected = false;
    for chip in needed_chips {
        if name_claimed(assets, chip, "TextLabel")? {
            continue;
        }
        inject(assets, report, injected_by, chip, "TextLabel", chip_args());
        chip_injected = true;
    }

    if chip_injected && !name_claimed(assets, HUD_FONT_NAME, "Font")? {
        inject(
            assets,
            report,
            injected_by,
            HUD_FONT_NAME,
            "Font",
            serde_json::json!({ "size_px": HUD_FONT_SIZE_PX }),
        );
    }
    Ok(())
}

// The sky mesh that displays an EnvironmentMap: an inside-out skybox cube plus
// the Prop that draws it. Injected only when the world has an EnvironmentMap
// and no skybox-generator mesh of its own. The half-extent tracks the first
// camera's far plane (sky depth is pinned to the far plane, so the mesh only
// needs to enclose the camera while staying inside it).
fn inject_sky(
    assets: &mut Vec<serde_json::Value>,
    report: &mut ExpandReport,
) -> Result<(), String> {
    if !assets.iter().any(|v| type_norm(v) == "environmentmap") {
        return Ok(());
    }
    let has_sky_mesh = assets.iter().any(|v| {
        type_norm(v) == "proceduralmesh"
            && v.get("args")
                .and_then(|a| a.get("generator"))
                .and_then(|g| g.as_str())
                == Some("skybox")
    });
    if has_sky_mesh {
        return Ok(());
    }

    let far = assets
        .iter()
        .find(|v| type_norm(v) == "camera3d")
        .and_then(|v| v.get("args"))
        .and_then(|a| a.get("far"))
        .and_then(|f| f.as_f64())
        .map(|f| f as f32)
        .unwrap_or(CAMERA_FAR_DEFAULT);
    let size = (far * SKY_FAR_FRACTION).min(SKY_SIZE_MAX);

    let mesh_claimed = name_claimed(assets, "sky_mesh", "ProceduralMesh")?;
    if !mesh_claimed {
        inject(
            assets,
            report,
            "sky",
            "sky_mesh",
            "ProceduralMesh",
            serde_json::json!({ "generator": "skybox", "size": size }),
        );
    }
    if !name_claimed(assets, "mat_sky", "Material")? {
        inject(
            assets,
            report,
            "sky",
            "mat_sky",
            "Material",
            serde_json::json!({ "roughness": 1.0, "metallic": 0.0, "tint": [1.0, 1.0, 1.0] }),
        );
    }
    if !name_claimed(assets, "sky", "Prop")? {
        inject(
            assets,
            report,
            "sky",
            "sky",
            "Prop",
            serde_json::json!({
                "mesh": "sky_mesh",
                "material": "mat_sky",
                "position": [0.0, 0.0, 0.0],
            }),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world(lines: &[serde_json::Value]) -> Vec<serde_json::Value> {
        lines.to_vec()
    }

    fn gfx() -> serde_json::Value {
        serde_json::json!({"name":"gfx","type":"GraphicsConfig","args":{}})
    }

    fn names_of_type(assets: &[serde_json::Value], t: &str) -> Vec<String> {
        assets
            .iter()
            .filter(|v| type_norm(v) == t)
            .map(asset_name)
            .collect()
    }

    #[test]
    fn non_rendering_world_gets_no_defaults() {
        let mut assets = world(&[serde_json::json!({"name":"w","type":"Window","args":{}})]);
        let mut report = ExpandReport::default();
        inject_engine_defaults(&mut assets, &mut report).unwrap();
        assert_eq!(assets.len(), 1);
        assert!(report.injected.is_empty());
    }

    #[test]
    fn rendering_world_gets_debug_hud_but_no_menu_or_stat_hud() {
        let mut assets = world(&[gfx()]);
        let mut report = ExpandReport::default();
        inject_engine_defaults(&mut assets, &mut report).unwrap();

        // No MainMenu is ever injected, and without one there is no StatHud.
        assert!(names_of_type(&assets, "mainmenu").is_empty());
        assert!(names_of_type(&assets, "stathud").is_empty());
        assert_eq!(names_of_type(&assets, "debughud"), vec!["debug_hud"]);
        let chips = names_of_type(&assets, "textlabel");
        for chip in ["passes_chip", "mouse_chip", "camera_chip", "sys_chip"] {
            assert!(chips.contains(&chip.to_string()), "missing {chip}");
        }
        assert!(!chips.contains(&"fps_chip".to_string()));
        assert_eq!(names_of_type(&assets, "font"), vec![HUD_FONT_NAME]);
        assert_eq!(report.injected.len(), 1 + 4 + 1);
    }

    #[test]
    fn main_menu_world_also_gets_the_stat_hud_set() {
        let mut assets = world(&[
            gfx(),
            serde_json::json!({"name":"pause","type":"MainMenu","args":{}}),
        ]);
        let mut report = ExpandReport::default();
        inject_engine_defaults(&mut assets, &mut report).unwrap();

        assert_eq!(names_of_type(&assets, "mainmenu"), vec!["pause"]);
        assert_eq!(names_of_type(&assets, "stathud"), vec!["stat_hud"]);
        assert_eq!(names_of_type(&assets, "debughud"), vec!["debug_hud"]);
        let chips = names_of_type(&assets, "textlabel");
        for chip in [
            "fps_chip",
            "vram_chip",
            "ram_chip",
            "ev_chip",
            "edr_chip",
            "passes_chip",
            "mouse_chip",
            "camera_chip",
            "sys_chip",
        ] {
            assert!(chips.contains(&chip.to_string()), "missing {chip}");
        }
        assert_eq!(names_of_type(&assets, "font"), vec![HUD_FONT_NAME]);
        assert_eq!(report.injected.len(), 2 + 9 + 1);
    }

    #[test]
    fn authored_stat_hud_gets_unset_labels_filled() {
        let mut assets = world(&[
            gfx(),
            serde_json::json!({"name":"hud","type":"StatHud","args":{"fps_label":"my_fps"}}),
            serde_json::json!({"name":"my_fps","type":"TextLabel","args":{"font":"hud_font"}}),
        ]);
        let mut report = ExpandReport::default();
        inject_engine_defaults(&mut assets, &mut report).unwrap();

        let hud = assets.iter().find(|v| type_norm(v) == "stathud").unwrap();
        assert_eq!(hud["args"]["fps_label"], serde_json::json!("my_fps"));
        assert_eq!(hud["args"]["vram_label"], serde_json::json!("vram_chip"));
        let chips = names_of_type(&assets, "textlabel");
        assert!(chips.contains(&"vram_chip".to_string()));
        assert!(!chips.contains(&"fps_chip".to_string()));
    }

    #[test]
    fn authored_chip_replaces_the_injected_one() {
        let mut assets = world(&[
            gfx(),
            serde_json::json!({"name":"menu","type":"MainMenu","args":{}}),
            serde_json::json!({"name":"fps_chip","type":"TextLabel","args":{"scale":1.4}}),
        ]);
        let mut report = ExpandReport::default();
        inject_engine_defaults(&mut assets, &mut report).unwrap();
        let fps_chips: Vec<_> = assets
            .iter()
            .filter(|v| asset_name(v) == "fps_chip")
            .collect();
        assert_eq!(fps_chips.len(), 1);
        assert_eq!(fps_chips[0]["args"]["scale"], serde_json::json!(1.4));
    }

    #[test]
    fn default_name_on_unrelated_type_is_an_error() {
        let mut assets = world(&[
            gfx(),
            serde_json::json!({"name":"debug_hud","type":"Window","args":{}}),
        ]);
        let mut report = ExpandReport::default();
        let err = inject_engine_defaults(&mut assets, &mut report).unwrap_err();
        assert!(err.contains("debug_hud"));
        assert!(err.contains("EngineDefaults"));
    }

    #[test]
    fn engine_defaults_flags_opt_out() {
        let mut assets = world(&[
            gfx(),
            serde_json::json!({"name":"menu","type":"MainMenu","args":{}}),
            serde_json::json!({"name":"d","type":"EngineDefaults","args":{
                "hud": false, "debug_hud": false
            }}),
        ]);
        let mut report = ExpandReport::default();
        inject_engine_defaults(&mut assets, &mut report).unwrap();
        assert!(names_of_type(&assets, "stathud").is_empty());
        assert!(names_of_type(&assets, "debughud").is_empty());
        // The directive itself never reaches the compiled world.
        assert!(names_of_type(&assets, "enginedefaults").is_empty());
    }

    #[test]
    fn duplicate_engine_defaults_is_an_error() {
        let mut assets = world(&[
            serde_json::json!({"name":"a","type":"EngineDefaults","args":{}}),
            serde_json::json!({"name":"b","type":"EngineDefaults","args":{}}),
        ]);
        let mut report = ExpandReport::default();
        let err = inject_engine_defaults(&mut assets, &mut report).unwrap_err();
        assert!(err.contains("at most one"));
    }

    #[test]
    fn environment_map_gets_a_sky_mesh() {
        let mut assets = world(&[
            gfx(),
            serde_json::json!({"name":"env","type":"EnvironmentMap","args":{"generator":"sky"}}),
        ]);
        let mut report = ExpandReport::default();
        inject_engine_defaults(&mut assets, &mut report).unwrap();
        assert_eq!(names_of_type(&assets, "proceduralmesh"), vec!["sky_mesh"]);
        assert_eq!(names_of_type(&assets, "material"), vec!["mat_sky"]);
        assert_eq!(names_of_type(&assets, "prop"), vec!["sky"]);
    }

    #[test]
    fn sky_size_tracks_camera_far_and_caps_at_showcase_value() {
        let mut assets = world(&[
            gfx(),
            serde_json::json!({"name":"cam","type":"Camera3D","args":{"far":100.0}}),
            serde_json::json!({"name":"env","type":"EnvironmentMap","args":{"generator":"sky"}}),
        ]);
        let mut report = ExpandReport::default();
        inject_engine_defaults(&mut assets, &mut report).unwrap();
        let mesh = assets
            .iter()
            .find(|v| type_norm(v) == "proceduralmesh")
            .unwrap();
        assert_eq!(mesh["args"]["size"], serde_json::json!(90.0));

        let mut assets = world(&[
            gfx(),
            serde_json::json!({"name":"cam","type":"Camera3D","args":{"far":900.0}}),
            serde_json::json!({"name":"env","type":"EnvironmentMap","args":{"generator":"sky"}}),
        ]);
        let mut report = ExpandReport::default();
        inject_engine_defaults(&mut assets, &mut report).unwrap();
        let mesh = assets
            .iter()
            .find(|v| type_norm(v) == "proceduralmesh")
            .unwrap();
        assert_eq!(mesh["args"]["size"], serde_json::json!(400.0));
    }

    #[test]
    fn authored_skybox_mesh_suppresses_sky_injection() {
        let mut assets = world(&[
            gfx(),
            serde_json::json!({"name":"env","type":"EnvironmentMap","args":{"generator":"sky"}}),
            serde_json::json!({"name":"dome","type":"ProceduralMesh","args":{"generator":"skybox","size":250.0}}),
        ]);
        let mut report = ExpandReport::default();
        inject_engine_defaults(&mut assets, &mut report).unwrap();
        assert_eq!(names_of_type(&assets, "proceduralmesh"), vec!["dome"]);
        assert!(names_of_type(&assets, "prop").is_empty());
    }

    #[test]
    fn no_environment_map_means_no_sky() {
        let mut assets = world(&[gfx()]);
        let mut report = ExpandReport::default();
        inject_engine_defaults(&mut assets, &mut report).unwrap();
        assert!(names_of_type(&assets, "proceduralmesh").is_empty());
    }

    #[test]
    fn story_world_injects_a_pause_menu() {
        let mut assets = world(&[
            gfx(),
            serde_json::json!({"name":"tale","type":"Story","args":{}}),
            serde_json::json!({"name":"tale_title","type":"Screen","args":{}}),
        ]);
        let mut report = ExpandReport::default();
        inject_engine_defaults(&mut assets, &mut report).unwrap();

        let menu = assets
            .iter()
            .find(|v| type_norm(v) == "mainmenu")
            .expect("pause menu injected");
        assert_eq!(asset_name(menu), "tale_pause");
        assert_eq!(menu["args"]["settings_profile"], "minimal");
        // The story system drives the pause: no MainMenu screen toggle, and Back
        // routes through the story so it returns to whichever menu opened it.
        assert_eq!(menu["args"]["toggle_key"], "");
        assert_eq!(menu["args"]["settings_back_action"], "story:settings_back");
        // Translucent backdrop, not the opaque MainMenu default.
        let alpha = menu["args"]["dim"][3].as_f64().unwrap();
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "pause dim should be translucent"
        );

        let items = menu["args"]["items"].as_array().unwrap();
        let actions: Vec<&str> = items
            .iter()
            .map(|i| i["action"].as_str().unwrap())
            .collect();
        for action in [
            "story:pause",
            "story:save",
            "story:load",
            "story:settings",
            // Main Menu (returns to the story's own title screen) and Quit are
            // now separate items.
            "screen:show:tale_title",
            "quit",
        ] {
            assert!(actions.contains(&action), "missing pause action {action}");
        }
        // Main Menu sits above Quit, and Quit is last.
        let labels: Vec<&str> = items.iter().map(|i| i["label"].as_str().unwrap()).collect();
        assert_eq!(labels.last(), Some(&"Quit"));
        let main_menu = labels.iter().position(|l| *l == "Main Menu").unwrap();
        let quit = labels.iter().position(|l| *l == "Quit").unwrap();
        assert!(main_menu < quit, "Main Menu should sit above Quit");

        // Escape is a story-driven binding, not the MainMenu's own toggle.
        let key = assets
            .iter()
            .find(|v| type_norm(v) == "keybinding" && asset_name(v) == "tale_pause_key")
            .expect("Escape binding injected");
        assert_eq!(key["args"]["key"], "Escape");
        assert_eq!(key["args"]["action"], "story:pause");

        // The Story scaffold points at the pause + settings screens.
        let story = assets.iter().find(|v| type_norm(v) == "story").unwrap();
        assert_eq!(story["args"]["scaffold"]["pause"], "tale_pause");
        assert_eq!(
            story["args"]["scaffold"]["settings"],
            "tale_pause_settings_video"
        );

        // The trimmed menu drives no StatHud, so none is injected for a story.
        assert!(names_of_type(&assets, "stathud").is_empty());
    }

    #[test]
    fn story_without_a_title_screen_quits_to_desktop() {
        let mut assets = world(&[
            gfx(),
            serde_json::json!({"name":"tale","type":"Story","args":{}}),
        ]);
        let mut report = ExpandReport::default();
        inject_engine_defaults(&mut assets, &mut report).unwrap();
        let menu = assets
            .iter()
            .find(|v| type_norm(v) == "mainmenu")
            .expect("pause menu injected");
        let items = menu["args"]["items"].as_array().unwrap();
        let quit = items.last().unwrap();
        assert_eq!(quit["action"], "quit");
        // With no title screen there is no Main Menu item to return to.
        let labels: Vec<&str> = items.iter().map(|i| i["label"].as_str().unwrap()).collect();
        assert!(
            !labels.contains(&"Main Menu"),
            "no title screen -> no Main Menu item"
        );
    }

    #[test]
    fn authored_menu_suppresses_the_story_pause_menu() {
        let mut assets = world(&[
            gfx(),
            serde_json::json!({"name":"tale","type":"Story","args":{}}),
            serde_json::json!({"name":"tale_title","type":"Screen","args":{}}),
            serde_json::json!({"name":"my_menu","type":"MainMenu","args":{}}),
        ]);
        let mut report = ExpandReport::default();
        inject_engine_defaults(&mut assets, &mut report).unwrap();
        assert_eq!(names_of_type(&assets, "mainmenu"), vec!["my_menu"]);
    }

    #[test]
    fn engine_defaults_opts_out_of_the_story_pause_menu() {
        let mut assets = world(&[
            gfx(),
            serde_json::json!({"name":"tale","type":"Story","args":{}}),
            serde_json::json!({"name":"tale_title","type":"Screen","args":{}}),
            serde_json::json!({"name":"d","type":"EngineDefaults","args":{
                "story_pause_menu": false
            }}),
        ]);
        let mut report = ExpandReport::default();
        inject_engine_defaults(&mut assets, &mut report).unwrap();
        assert!(names_of_type(&assets, "mainmenu").is_empty());
    }

    #[test]
    fn malformed_story_args_do_not_panic_the_pause_injection() {
        // A hand-authored Story with a non-object args is malformed, but the
        // pause injection must skip the scaffold patch gracefully rather than
        // panic; the malformed Story surfaces its own error later.
        let mut assets = world(&[
            gfx(),
            serde_json::json!({"name":"tale","type":"Story","args":[]}),
            serde_json::json!({"name":"tale_title","type":"Screen","args":{}}),
        ]);
        let mut report = ExpandReport::default();
        inject_engine_defaults(&mut assets, &mut report).unwrap();
        // The menu still injects; only the scaffold patch was skipped.
        assert!(assets.iter().any(|v| type_norm(v) == "mainmenu"));
    }

    #[test]
    fn non_story_world_gets_no_pause_menu() {
        let mut assets = world(&[gfx()]);
        let mut report = ExpandReport::default();
        inject_engine_defaults(&mut assets, &mut report).unwrap();
        assert!(names_of_type(&assets, "mainmenu").is_empty());
    }

    #[test]
    fn synthesized_hud_report_args_include_filled_labels() {
        let mut assets = world(&[
            gfx(),
            serde_json::json!({"name":"menu","type":"MainMenu","args":{}}),
        ]);
        let mut report = ExpandReport::default();
        inject_engine_defaults(&mut assets, &mut report).unwrap();
        let entry = report
            .injected
            .iter()
            .find(|i| i.asset_type == "StatHud")
            .unwrap();
        assert_eq!(entry.args["fps_label"], serde_json::json!("fps_chip"));
    }
}
