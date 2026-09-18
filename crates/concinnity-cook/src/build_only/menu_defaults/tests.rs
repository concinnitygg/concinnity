use super::*;

fn gfx() -> serde_json::Value {
    serde_json::json!({"type":"GraphicsConfig","args":{"$id":"gfx"}})
}

fn names_of_type(assets: &[serde_json::Value], t: RegisteredType) -> Vec<String> {
    assets
        .iter()
        .filter(|v| registered_type(v) == Some(t))
        .map(asset_name)
        .collect()
}

fn inject(assets: &mut Vec<serde_json::Value>) -> Result<ExpandReport, String> {
    let mut report = ExpandReport::default();
    inject_menu_defaults(assets, &mut report)?;
    Ok(report)
}

#[test]
fn a_menu_world_gets_the_stats_strip() {
    let mut assets = vec![
        gfx(),
        serde_json::json!({"type":"MainMenu","args":{"$id":"pause"}}),
    ];
    let report = inject(&mut assets).unwrap();
    assert_eq!(
        names_of_type(&assets, RegisteredType::StatHud),
        vec!["stat_hud"]
    );
    // The chips and their font are minted at world start, not here.
    assert!(names_of_type(&assets, RegisteredType::TextLabel).is_empty());
    assert!(names_of_type(&assets, RegisteredType::Font).is_empty());
    assert_eq!(report.injected.len(), 1);
    assert_eq!(report.injected[0].injected_by, "hud");
}

#[test]
fn a_world_without_a_menu_gets_no_stats_strip() {
    let mut assets = vec![gfx()];
    inject(&mut assets).unwrap();
    assert!(names_of_type(&assets, RegisteredType::StatHud).is_empty());
}

#[test]
fn an_authored_stat_hud_is_left_alone() {
    let mut assets = vec![
        gfx(),
        serde_json::json!({"type":"MainMenu","args":{"$id":"pause"}}),
        serde_json::json!({"type":"StatHud","args":{"$id":"hud","fps_label":"my_fps"}}),
    ];
    inject(&mut assets).unwrap();
    assert_eq!(names_of_type(&assets, RegisteredType::StatHud), vec!["hud"]);
}

#[test]
fn the_hud_toggle_opts_out() {
    let mut assets = vec![
        gfx(),
        serde_json::json!({"type":"MainMenu","args":{"$id":"pause"}}),
        serde_json::json!({"type":"EngineDefaults","args":{"$id":"d","hud": false}}),
    ];
    inject(&mut assets).unwrap();
    assert!(names_of_type(&assets, RegisteredType::StatHud).is_empty());
}

// The directive is a stored component now: the runtime pass drains it, so the
// build must leave it in the world.
#[test]
fn the_directive_survives_the_build() {
    let mut assets = vec![
        gfx(),
        serde_json::json!({"type":"EngineDefaults","args":{"$id":"d","sky": false}}),
    ];
    inject(&mut assets).unwrap();
    assert_eq!(
        names_of_type(&assets, RegisteredType::EngineDefaults),
        vec!["d"]
    );
}

#[test]
fn a_second_directive_is_an_error() {
    let mut assets = vec![
        serde_json::json!({"type":"EngineDefaults","args":{"$id":"a"}}),
        serde_json::json!({"type":"EngineDefaults","args":{"$id":"b"}}),
    ];
    let err = inject(&mut assets).unwrap_err();
    assert!(err.contains("at most one"), "{err}");
}

#[test]
fn malformed_directive_args_are_an_error() {
    let mut assets = vec![serde_json::json!({
        "type":"EngineDefaults","args":{"$id":"d","hud":"yes"}
    })];
    let err = inject(&mut assets).unwrap_err();
    assert!(err.contains("EngineDefaults 'd'"), "{err}");
    assert!(err.contains("invalid args"), "{err}");
}

#[test]
fn a_directive_without_args_keeps_every_default_on() {
    let mut assets = vec![
        gfx(),
        serde_json::json!({"type":"MainMenu","args":{"$id":"pause"}}),
        serde_json::json!({"type":"EngineDefaults","args":{"$id":"d"}}),
    ];
    inject(&mut assets).unwrap();
    assert_eq!(
        names_of_type(&assets, RegisteredType::StatHud),
        vec!["stat_hud"]
    );
}

#[test]
fn a_story_world_gets_a_pause_menu() {
    let mut assets = vec![
        gfx(),
        serde_json::json!({"type":"Story","args":{"$id":"tale"}}),
        serde_json::json!({"type":"Screen","args":{"$id":"tale_title"}}),
    ];
    inject(&mut assets).unwrap();

    let menu = assets
        .iter()
        .find(|v| registered_type(v) == Some(RegisteredType::MainMenu))
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
        .find(|v| {
            registered_type(v) == Some(RegisteredType::KeyBinding)
                && asset_name(v) == "tale_pause_key"
        })
        .expect("Escape binding injected");
    assert_eq!(key["args"]["key"], "Escape");
    assert_eq!(key["args"]["action"], "story:pause");

    // The Story scaffold points at the pause + settings screens.
    let story = assets
        .iter()
        .find(|v| registered_type(v) == Some(RegisteredType::Story))
        .unwrap();
    assert_eq!(story["args"]["scaffold"]["pause"], "tale_pause");
    assert_eq!(
        story["args"]["scaffold"]["settings"],
        "tale_pause_settings_video"
    );

    // The trimmed menu drives no StatHud, so none is injected for a story.
    assert!(names_of_type(&assets, RegisteredType::StatHud).is_empty());
}

#[test]
fn a_story_without_a_title_screen_quits_to_desktop() {
    let mut assets = vec![
        gfx(),
        serde_json::json!({"type":"Story","args":{"$id":"tale"}}),
    ];
    inject(&mut assets).unwrap();
    let menu = assets
        .iter()
        .find(|v| registered_type(v) == Some(RegisteredType::MainMenu))
        .expect("pause menu injected");
    let items = menu["args"]["items"].as_array().unwrap();
    assert_eq!(items.last().unwrap()["action"], "quit");
    let labels: Vec<&str> = items.iter().map(|i| i["label"].as_str().unwrap()).collect();
    assert!(
        !labels.contains(&"Main Menu"),
        "no title screen -> no Main Menu item"
    );
}

#[test]
fn an_authored_menu_suppresses_the_story_pause_menu() {
    let mut assets = vec![
        gfx(),
        serde_json::json!({"type":"Story","args":{"$id":"tale"}}),
        serde_json::json!({"type":"Screen","args":{"$id":"tale_title"}}),
        serde_json::json!({"type":"MainMenu","args":{"$id":"my_menu"}}),
    ];
    inject(&mut assets).unwrap();
    assert_eq!(
        names_of_type(&assets, RegisteredType::MainMenu),
        vec!["my_menu"]
    );
}

#[test]
fn the_story_pause_toggle_opts_out() {
    let mut assets = vec![
        gfx(),
        serde_json::json!({"type":"Story","args":{"$id":"tale"}}),
        serde_json::json!({"type":"Screen","args":{"$id":"tale_title"}}),
        serde_json::json!({"type":"EngineDefaults","args":{
            "$id":"d",
            "story_pause_menu": false
        }}),
    ];
    inject(&mut assets).unwrap();
    assert!(names_of_type(&assets, RegisteredType::MainMenu).is_empty());
}

#[test]
fn malformed_story_args_do_not_panic_the_pause_injection() {
    // A hand-authored Story with a non-object args is malformed: it cannot
    // carry the `$id` a pause menu is named after, so the injection skips it
    // rather than panic, and the malformed Story surfaces its own error later.
    let mut assets = vec![
        gfx(),
        serde_json::json!({"type":"Story","args":[]}),
        serde_json::json!({"type":"Screen","args":{"$id":"tale_title"}}),
    ];
    inject(&mut assets).unwrap();
    assert!(
        !assets
            .iter()
            .any(|v| registered_type(v) == Some(RegisteredType::MainMenu))
    );
}

#[test]
fn a_non_story_world_gets_no_pause_menu() {
    let mut assets = vec![gfx()];
    inject(&mut assets).unwrap();
    assert!(names_of_type(&assets, RegisteredType::MainMenu).is_empty());
}

// A default's name held by an unrelated type cannot be injected, and skipping
// it silently would hide the clash.
#[test]
fn a_default_name_held_by_another_type_is_an_error() {
    let mut assets = vec![
        gfx(),
        serde_json::json!({"type":"Story","args":{"$id":"tale"}}),
        serde_json::json!({"type":"Window","args":{"$id":"tale_pause"}}),
    ];
    let err = inject(&mut assets).unwrap_err();
    assert!(err.contains("tale_pause"), "{err}");
    assert!(err.contains("Window"), "{err}");

    let mut assets = vec![
        gfx(),
        serde_json::json!({"type":"MainMenu","args":{"$id":"pause"}}),
        serde_json::json!({"type":"Window","args":{"$id":"stat_hud"}}),
    ];
    let err = inject(&mut assets).unwrap_err();
    assert!(err.contains("stat_hud"), "{err}");
}

// A Story whose `scaffold` is not an object is malformed; the patch is skipped
// rather than replacing what the author wrote.
#[test]
fn a_non_object_scaffold_is_left_alone() {
    let mut assets = vec![serde_json::json!({
        "type":"Story","args":{"$id":"tale","scaffold": 7}
    })];
    patch_story_scaffold(&mut assets, "tale", "tale_pause");
    assert_eq!(assets[0]["args"]["scaffold"], 7);
}

// Running the pass twice (a re-cook of an already expanded list) must not stack
// a second strip.
#[test]
fn injecting_twice_yields_one_stat_hud() {
    let mut assets = vec![
        gfx(),
        serde_json::json!({"type":"MainMenu","args":{"$id":"pause"}}),
    ];
    inject(&mut assets).unwrap();
    inject(&mut assets).unwrap();
    assert_eq!(
        names_of_type(&assets, RegisteredType::StatHud),
        vec!["stat_hud"]
    );
}
