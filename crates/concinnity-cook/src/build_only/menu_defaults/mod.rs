// The engine defaults that need the authoring vocabulary, and so cannot wait
// for the runtime pass that injects the rest (`concinnity_core::defaults`): a
// StatHud for a world with a MainMenu, whose performance-stats toggles drive
// its chips, and an Escape-toggled pause MainMenu for a story world that
// declares none. Both are stated in terms of MainMenu, which the build expands
// away, so the runtime cannot see the condition either one keys off.
//
// The chips and font of the StatHud injected here are filled in at world
// start, alongside every other injected default.
//
// An `EngineDefaults` entry turns either one off. It is read rather than
// consumed: it is a stored component now, and the runtime pass drains it.

use super::expand::{ExpandReport, asset_name, type_norm};
use concinnity_core::components::EngineDefaults;

// Complete a world with the two defaults stated in build-only terms. Runs
// before menu expansion, so an injected MainMenu expands like an authored one,
// and after story expansion, so the compiled Story and its title screen are
// present.
pub(crate) fn inject_menu_defaults(
    assets: &mut Vec<serde_json::Value>,
    report: &mut ExpandReport,
) -> Result<(), String> {
    let toggles = declared_toggles(assets)?;
    if toggles.hud {
        inject_stat_hud(assets, report)?;
    }
    // Injected after the StatHud so it does not turn one on: the trimmed pause
    // menu carries no performance-stats toggles to drive it.
    if toggles.story_pause_menu {
        inject_story_pause_menu(assets, report)?;
    }
    Ok(())
}

// The toggles the world declares, or all-on when it declares none. More than
// one entry is ambiguous and rejected here as well as by the singleton check,
// so this pass never has to pick.
fn declared_toggles(assets: &[serde_json::Value]) -> Result<EngineDefaults, String> {
    let mut declared = assets.iter().filter(|v| type_norm(v) == "enginedefaults");
    let Some(value) = declared.next() else {
        return Ok(EngineDefaults::default());
    };
    if let Some(second) = declared.next() {
        return Err(format!(
            "EngineDefaults '{}': the world already declares EngineDefaults '{}'; \
             declare at most one",
            asset_name(second),
            asset_name(value)
        ));
    }
    let args = value
        .get("args")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    serde_json::from_value(args).map_err(|e| {
        format!(
            "EngineDefaults '{}': invalid args: {}",
            asset_name(value),
            e
        )
    })
}

// A world with a MainMenu gets the stats strip its video settings toggle. The
// chips themselves are minted at world start, so what is injected here is the
// bare component that says the world wants one.
fn inject_stat_hud(
    assets: &mut Vec<serde_json::Value>,
    report: &mut ExpandReport,
) -> Result<(), String> {
    let has_menu = assets.iter().any(|v| type_norm(v) == "mainmenu");
    let has_hud = assets.iter().any(|v| type_norm(v) == "stathud");
    if !has_menu || has_hud {
        return Ok(());
    }
    if name_claimed(assets, report, "hud", "stat_hud", "StatHud")? {
        // Unreachable in practice: a same-name same-type asset would have
        // matched the type scan above.
        return Ok(());
    }
    inject(
        assets,
        report,
        "hud",
        "stat_hud",
        "StatHud",
        serde_json::json!({}),
    );
    Ok(())
}

// Give a story world an Escape pause menu (Resume / Save / Load / Settings /
// Quit) when it plays a Story but declares no MainMenu. The menu targets the
// story's own title screen for its Quit item and uses the trimmed settings
// profile: a 2D story renders no scene, so only window / output and volume are
// worth configuring.
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
    if name_claimed(assets, report, "story_pause_menu", &name, "MainMenu")? {
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

// Whether the world already provides `name`. A claimed name is recorded as a
// shadow, so a listing can show the default the world overrides rather than
// leaving it unaccounted for. A name held by a different type is a hard error:
// the default cannot be injected and silently skipping it would hide the
// conflict.
fn name_claimed(
    assets: &mut [serde_json::Value],
    report: &mut ExpandReport,
    injected_by: &'static str,
    name: &str,
    asset_type: &str,
) -> Result<bool, String> {
    let Some(claim) = assets.iter().find(|v| asset_name(v) == name) else {
        return Ok(false);
    };
    if type_norm(claim) != asset_type.to_lowercase().replace('_', "") {
        return Err(format!(
            "engine default '{}' ({}) collides with your {} asset of the same name; \
             rename that asset or disable the default with an EngineDefaults entry",
            name,
            asset_type,
            claim.get("type").and_then(|t| t.as_str()).unwrap_or("?"),
        ));
    }
    report.record_shadowed(name, asset_type, injected_by, serde_json::json!({}));
    Ok(true)
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

#[cfg(test)]
mod tests;
