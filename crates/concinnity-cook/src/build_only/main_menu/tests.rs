use super::*;

// Dump a probe world that opens one menu screen directly, for a real-GPU
// screenshot probe (the headless probe cannot click through the menu). Picks
// the screen from `CN_PROBE_TAB` (`menu` for the button list, else a settings
// tab: video | audio | controls, default controls) and the settings breadth
// from `CN_PROBE_PROFILE` (full | minimal, default full). It expands a
// MainMenu, then flips the chosen Screen to `initial` so it shows at launch.
// `#[ignore]`d: run explicitly, e.g.
//   CN_PROBE_PROFILE=minimal CN_PROBE_TAB=video cargo test -p concinnity-cook \
//       dump_settings_tab_probe_world -- --ignored
// then `concinnity debug -f world.jsonl` + `concinnity debug screenshot`.
#[test]
#[ignore]
fn dump_settings_tab_probe_world() {
    let tab = std::env::var("CN_PROBE_TAB").unwrap_or_else(|_| "controls".to_string());
    let profile = std::env::var("CN_PROBE_PROFILE").unwrap_or_else(|_| "full".to_string());
    let out = std::env::var("CN_PROBE_OUT").unwrap_or_else(|_| "world.jsonl".to_string());
    // The Minimal profile is the pause-menu shape, so give it the pause item
    // list; Full keeps the bare default items.
    let menu_args = if profile == "minimal" {
        serde_json::json!({
            "title": "Paused",
            "settings_profile": "minimal",
            "items": [
                {"label": "Resume", "action": "return"},
                {"label": "Save", "action": "story:save"},
                {"label": "Load", "action": "story:load"},
                {"label": "Settings", "action": "settings"},
                {"label": "Quit to Title", "action": "quit"},
            ],
        })
    } else {
        serde_json::json!({"title": "Probe", "settings_profile": profile})
    };
    let mut assets = vec![
        serde_json::json!({"name":"win","type":"Window","args":{"width":1280,"height":720}}),
        serde_json::json!({"name":"gfx","type":"GraphicsConfig","args":{}}),
        serde_json::json!({"name":"main_menu","type":"MainMenu","args": menu_args}),
    ];
    expand_main_menus(&mut assets).unwrap();
    // Show the chosen screen at launch (the menu screen for `menu`, else the
    // named settings tab); every other Screen is off.
    let target = if tab == "menu" {
        "main_menu".to_string()
    } else {
        format!("main_menu_settings_{tab}")
    };
    for v in &mut assets {
        if type_norm(v) == "screen" {
            v["args"]["initial"] = serde_json::json!(asset_name(v) == target);
        }
    }
    let mut body = String::new();
    for v in &assets {
        body.push_str(&serde_json::to_string(v).unwrap());
        body.push('\n');
    }
    std::fs::write(&out, body).unwrap();
}

fn names(assets: &[serde_json::Value]) -> Vec<String> {
    assets.iter().map(asset_name).collect()
}

fn by_name<'a>(assets: &'a [serde_json::Value], name: &str) -> &'a serde_json::Value {
    assets
        .iter()
        .find(|v| asset_name(v) == name)
        .unwrap_or_else(|| panic!("no asset named {name}"))
}

#[test]
fn passes_through_without_menus() {
    let mut assets = vec![serde_json::json!({"name":"x","type":"Window","args":{}})];
    expand_main_menus(&mut assets).unwrap();
    assert_eq!(assets.len(), 1);
    assert_eq!(assets[0]["type"], "Window");
}

#[test]
fn bare_menu_expands_to_default_layout() {
    let mut assets = vec![serde_json::json!({"name":"main_menu","type":"MainMenu"})];
    expand_main_menus(&mut assets).unwrap();

    // No MainMenu survives.
    assert!(!assets.iter().any(|v| type_norm(v) == "mainmenu"));

    // The main screen and a toggle binding exist. The screen starts closed by
    // default: the scene shows first and the toggle key opens the menu.
    assert_eq!(by_name(&assets, "main_menu")["type"], "Screen");
    assert_eq!(by_name(&assets, "main_menu")["args"]["initial"], false);
    assert_eq!(by_name(&assets, "main_menu_toggle")["type"], "KeyBinding");
    assert_eq!(
        by_name(&assets, "main_menu_toggle")["args"]["action"],
        "screen:toggle:main_menu"
    );

    // Three items -> three label/button pairs.
    let ns = names(&assets);
    for i in 0..3 {
        assert!(ns.contains(&format!("main_menu_label_{i}")));
        assert!(ns.contains(&format!("main_menu_btn_{i}")));
    }

    // Return resolves to screen:hide, Quit passes through, Settings opens the
    // generated sub-screen.
    assert_eq!(
        by_name(&assets, "main_menu_btn_0")["args"]["action"],
        "screen:hide"
    );
    assert_eq!(
        by_name(&assets, "main_menu_btn_1")["args"]["action"],
        "screen:show:main_menu_settings_video"
    );
    assert_eq!(
        by_name(&assets, "main_menu_btn_2")["args"]["action"],
        "quit"
    );

    // The default-tab (video) settings screen and its Back button exist.
    assert_eq!(
        by_name(&assets, "main_menu_settings_video")["type"],
        "Screen"
    );
    assert_eq!(
        by_name(&assets, "main_menu_settings_video")["args"]["initial"],
        false
    );
    // Back returns to the menu screen (not screen:hide, since tabs navigate
    // explicitly rather than as a restore-prev modal).
    assert_eq!(
        by_name(&assets, "main_menu_settings_video_btn_back")["args"]["action"],
        "screen:show:main_menu"
    );
    // The video tab carries its own (accent) tab header and a vsync row.
    assert_eq!(
        by_name(&assets, "main_menu_settings_video_tab_video")["args"]["content"],
        "Video"
    );
    let opt = by_name(&assets, "main_menu_settings_video_opt_vsync");
    assert_eq!(opt["type"], "OptionSelect");
    assert_eq!(opt["args"]["setting"], "vsync");

    // A follow-cursor sprite and a backdrop exist for the main screen.
    assert_eq!(
        by_name(&assets, "main_menu_cursor")["args"]["follow_cursor"],
        true
    );
    assert_eq!(by_name(&assets, "main_menu_dim")["type"], "Sprite");
}

#[test]
fn missing_name_is_an_error() {
    let mut assets = vec![serde_json::json!({"type":"MainMenu","args":{}})];
    let err = expand_main_menus(&mut assets).unwrap_err();
    assert_eq!(err, "MainMenu: missing `name`");
}

#[test]
fn invalid_args_name_the_menu() {
    let mut assets = vec![serde_json::json!({
        "name":"m","type":"MainMenu","args":{"button_width":"wide"}
    })];
    let err = expand_main_menus(&mut assets).unwrap_err();
    assert!(err.contains("MainMenu 'm'"), "{err}");
    assert!(err.contains("invalid args"), "{err}");
}

// A non-centered menu is a column anchored at the menu's own x/y instead of
// the reference canvas center and top margin.
#[test]
fn a_non_centered_menu_anchors_its_column_at_x_and_y() {
    let mut assets = vec![serde_json::json!({
        "name":"m","type":"MainMenu",
        "args":{"centered":false,"x":200.0,"y":80.0,"button_width":300.0}
    })];
    expand_main_menus(&mut assets).unwrap();
    // The first button is centered on x, starting at y.
    let btn = by_name(&assets, "m_btn_0");
    assert_eq!(btn["args"]["x"], 200.0 - 300.0 / 2.0);
    assert_eq!(btn["args"]["y"], 80.0);
    assert_eq!(by_name(&assets, "m_label_0")["args"]["x"], 200.0);
}

// The settings tab of a non-centered menu keeps the narrower column form,
// sized from the menu's button width rather than spanning the canvas.
#[test]
fn a_non_centered_settings_tab_keeps_the_column_row_width() {
    let mut assets = vec![serde_json::json!({
        "name":"m","type":"MainMenu",
        "args":{"centered":false,"x":300.0,"y":100.0,"button_width":300.0}
    })];
    expand_main_menus(&mut assets).unwrap();
    // 300 * 1.85 = 555, centered on x -> left edge at 300 - 555/2.
    let card = by_name(&assets, "m_settings_video_bg_0");
    assert_eq!(card["args"]["width"], 555.0);
    assert_eq!(card["args"]["x"], 300.0 - 555.0 / 2.0);
    // Rows stack down from the menu's own y, below the tab bar.
    let tab_y = by_name(&assets, "m_settings_video_tabbtn_audio")["args"]["y"]
        .as_f64()
        .unwrap();
    assert_eq!(tab_y, 100.0);
    assert!(card["args"]["y"].as_f64().unwrap() > tab_y);
}

#[test]
fn video_tab_emits_a_row_per_setting() {
    let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
    expand_main_menus(&mut assets).unwrap();
    for (setting, label) in [
        ("vsync", "Vsync"),
        ("fps_cap", "Frame Rate"),
        ("window_mode", "Window Mode"),
        ("resolution", "Resolution"),
        ("render_scale", "Render Scale"),
        ("upscale_backend", "Upscaler"),
    ] {
        let opt = by_name(&assets, &format!("m_settings_video_opt_{setting}"));
        assert_eq!(opt["type"], "OptionSelect");
        assert_eq!(opt["args"]["setting"], setting);
        assert_eq!(opt["args"]["label"], label);
    }
}

#[test]
fn video_tab_leads_with_the_master_quality_row() {
    let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
    expand_main_menus(&mut assets).unwrap();
    // The master preset row is an ungrouped OptionSelect bound to the
    // graphics_quality setting (the runtime knows its options + how to apply).
    let opt = by_name(&assets, "m_settings_video_opt_graphics_quality");
    assert_eq!(opt["type"], "OptionSelect");
    assert_eq!(opt["args"]["setting"], "graphics_quality");
    assert_eq!(opt["args"]["label"], "Graphics Quality");
    // It leads the tab: it is emitted before the first core row (vsync).
    let master_pos = assets
        .iter()
        .position(|v| asset_name(v) == "m_settings_video_opt_graphics_quality")
        .expect("master row");
    let vsync_pos = assets
        .iter()
        .position(|v| asset_name(v) == "m_settings_video_opt_vsync")
        .expect("vsync row");
    assert!(
        master_pos < vsync_pos,
        "master quality row should lead the tab"
    );
}

#[test]
fn video_tab_emits_an_exposure_slider() {
    let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
    expand_main_menus(&mut assets).unwrap();
    let sld = by_name(&assets, "m_settings_video_sld_exposure");
    assert_eq!(sld["type"], "Slider");
    assert_eq!(sld["args"]["setting"], "exposure");
    assert_eq!(sld["args"]["label"], "Exposure");
    // The slider carries the menu font through to its own expansion.
    assert_eq!(sld["args"]["font"], "m_font");
    // Sliders are a Video-only row today: the other tabs emit none.
    assert!(
        !assets
            .iter()
            .any(|v| asset_name(v) == "m_settings_audio_sld_exposure")
    );
}

#[test]
fn settings_emits_a_screen_per_tab() {
    let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
    expand_main_menus(&mut assets).unwrap();
    for suffix in ["video", "audio", "controls"] {
        let screen = by_name(&assets, &format!("m_settings_{suffix}"));
        assert_eq!(screen["type"], "Screen", "tab screen {suffix} missing");
        assert_eq!(screen["args"]["initial"], false);
        // Every tab returns to the menu screen via Back.
        assert_eq!(
            by_name(&assets, &format!("m_settings_{suffix}_btn_back"))["args"]["action"],
            "screen:show:m"
        );
    }
}

#[test]
fn audio_and_controls_tabs_carry_their_rows() {
    let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
    expand_main_menus(&mut assets).unwrap();
    // Audio: a master-volume row.
    let vol = by_name(&assets, "m_settings_audio_opt_master_volume");
    assert_eq!(vol["type"], "OptionSelect");
    assert_eq!(vol["args"]["setting"], "master_volume");
    // Controls: a mouse-sensitivity slider plus rebind rows and the
    // read-only Pause reference.
    let sens = by_name(&assets, "m_settings_controls_sld_mouse_sensitivity");
    assert_eq!(sens["type"], "Slider");
    assert_eq!(sens["args"]["setting"], "mouse_sensitivity");
    // The read-only Pause reference is display-only (no HitRegion).
    assert_eq!(
        by_name(&assets, "m_settings_controls_keyname_0")["args"]["content"],
        "Pause"
    );
    assert_eq!(
        by_name(&assets, "m_settings_controls_keyval_0")["args"]["content"],
        "Esc"
    );
    assert!(
        !assets
            .iter()
            .any(|v| asset_name(v) == "m_settings_controls_keyname_0_btn")
    );
}

// Each rebindable action emits a name + value label and a HitRegion firing
// its `setting:<key>:rebind` capture action; Pause stays display-only.
#[test]
fn controls_tab_emits_rebind_rows() {
    let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
    expand_main_menus(&mut assets).unwrap();
    for (i, (label, setting)) in [
        ("Move Forward", "key_forward"),
        ("Move Back", "key_backward"),
        ("Move Left", "key_left"),
        ("Move Right", "key_right"),
        ("Sprint", "key_sprint"),
        ("Jump", "key_jump"),
        ("Interact", "key_interact"),
        // The gamepad rows continue the index sequence in their group.
        ("Sprint", "pad_sprint"),
        ("Jump", "pad_jump"),
        ("Interact", "pad_interact"),
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(
            by_name(&assets, &format!("m_settings_controls_rebind_name_{i}"))["args"]["content"],
            *label
        );
        // A placeholder value, synced to the live key map at runtime.
        assert_eq!(
            by_name(&assets, &format!("m_settings_controls_rebind_val_{i}"))["args"]["content"],
            "--"
        );
        let btn = by_name(&assets, &format!("m_settings_controls_rebind_btn_{i}"));
        assert_eq!(btn["type"], "HitRegion");
        assert_eq!(btn["args"]["action"], format!("setting:{setting}:rebind"));
        // The region's label points at the value so the client refreshes it.
        assert_eq!(
            btn["args"]["label"],
            format!("m_settings_controls_rebind_val_{i}")
        );
    }
    // Pause is read-only: no rebind HitRegion beyond the rows above.
    assert!(
        !assets
            .iter()
            .any(|v| asset_name(v).starts_with("m_settings_controls_rebind_btn_10"))
    );
    // The gamepad rows sit under a collapsible group header alongside the
    // stick sliders.
    assert_eq!(
        by_name(&assets, "m_settings_controls_grphdr_0")["args"]["content"],
        "+ Gamepad"
    );
    // The stick sliders emit Slider specs (expanded by the slider pass).
    let slider = by_name(&assets, "m_settings_controls_sld_gamepad_look_sensitivity");
    assert_eq!(slider["type"], "Slider");
    assert_eq!(slider["args"]["setting"], "gamepad_look_sensitivity");
    let slider = by_name(&assets, "m_settings_controls_sld_gamepad_deadzone");
    assert_eq!(slider["args"]["setting"], "gamepad_deadzone");
}

#[test]
fn tab_bar_switches_between_tabs() {
    let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
    expand_main_menus(&mut assets).unwrap();
    // The active tab gets an accent label + underline marker and NO button;
    // the other tabs are buttons that switch to their screen.
    assert_eq!(
        by_name(&assets, "m_settings_video_tabmark")["type"],
        "Sprite"
    );
    assert!(
        !assets
            .iter()
            .any(|v| asset_name(v) == "m_settings_video_tabbtn_video")
    );
    assert_eq!(
        by_name(&assets, "m_settings_video_tabbtn_audio")["args"]["action"],
        "screen:show:m_settings_audio"
    );
    assert_eq!(
        by_name(&assets, "m_settings_video_tabbtn_controls")["args"]["action"],
        "screen:show:m_settings_controls"
    );
    // From the controls tab you can hop back to video.
    assert_eq!(
        by_name(&assets, "m_settings_controls_tabbtn_video")["args"]["action"],
        "screen:show:m_settings_video"
    );
}

#[test]
fn labels_are_not_centered_so_layout_wins() {
    let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
    expand_main_menus(&mut assets).unwrap();
    assert_eq!(by_name(&assets, "m_label_0")["args"]["centered"], false);
}

#[test]
fn custom_items_pass_actions_through_verbatim() {
    let mut assets = vec![serde_json::json!({
        "name": "title",
        "type": "MainMenu",
        "args": { "items": [
            {"label":"New Game","action":"scene:level_1"},
            {"label":"Quit","action":"quit"}
        ]}
    })];
    expand_main_menus(&mut assets).unwrap();
    assert_eq!(
        by_name(&assets, "title_btn_0")["args"]["action"],
        "scene:level_1"
    );
    assert_eq!(by_name(&assets, "title_btn_1")["args"]["action"], "quit");
    // No settings item -> no settings sub-screen.
    assert!(!assets.iter().any(|v| asset_name(v) == "title_settings"));
}

#[test]
fn toggle_key_empty_emits_no_binding() {
    let mut assets = vec![serde_json::json!({
        "name": "m", "type": "MainMenu", "args": { "toggle_key": "" }
    })];
    expand_main_menus(&mut assets).unwrap();
    assert!(!assets.iter().any(|v| type_norm(v) == "keybinding"));
}

#[test]
fn cursor_disabled_emits_no_cursor_sprite() {
    let mut assets = vec![serde_json::json!({
        "name": "m", "type": "MainMenu", "args": { "cursor": false }
    })];
    expand_main_menus(&mut assets).unwrap();
    assert!(!assets.iter().any(|v| asset_name(v) == "m_cursor"));
}

#[test]
fn dim_alpha_zero_emits_no_backdrop() {
    let mut assets = vec![serde_json::json!({
        "name": "m", "type": "MainMenu", "args": { "dim": [0.0, 0.0, 0.0, 0.0] }
    })];
    expand_main_menus(&mut assets).unwrap();
    assert!(!assets.iter().any(|v| asset_name(v) == "m_dim"));
}

// A MainMenu that omits `dim` (as Bistro declares its menu) inherits the
// opaque default, so the emitted backdrop fully covers the scene -- the cue
// the renderer keys off to skip the world while the menu is open.
#[test]
fn default_menu_emits_opaque_backdrop() {
    let mut assets = vec![serde_json::json!({
        "name": "main_menu", "type": "MainMenu",
        "args": { "title": "Bistro_v5_2", "initial": false }
    })];
    expand_main_menus(&mut assets).unwrap();
    let tint = &by_name(&assets, "main_menu_dim")["args"]["tint"];
    assert_eq!(
        tint[3].as_f64().unwrap(),
        1.0,
        "default menu backdrop must be opaque (tint={tint})"
    );
}

#[test]
fn generated_name_collision_is_an_error() {
    let mut assets = vec![
        serde_json::json!({"name":"m","type":"MainMenu","args":{"toggle_key":""}}),
        serde_json::json!({"name":"m_btn_0","type":"Sprite","args":{}}),
    ];
    let err = expand_main_menus(&mut assets).unwrap_err();
    assert!(err.contains("m_btn_0"));
    assert!(err.contains("collides"));
}

#[test]
fn title_emits_a_heading_label() {
    let mut assets = vec![serde_json::json!({
        "name": "m", "type": "MainMenu", "args": { "title": "Paused" }
    })];
    expand_main_menus(&mut assets).unwrap();
    assert_eq!(by_name(&assets, "m_title")["args"]["content"], "Paused");
}

// The menu emits its own built-in font and references it explicitly. It
// cannot rely on the auto-injected default font, which is only injected when
// the world declares no Font at all (a HUD font would suppress it), leaving
// the labels with no font and no rendered text.
#[test]
fn emits_own_font_and_labels_reference_it() {
    let mut assets =
        vec![serde_json::json!({"name":"m","type":"MainMenu","args":{"toggle_key":""}})];
    expand_main_menus(&mut assets).unwrap();
    let font = by_name(&assets, "m_font");
    assert_eq!(font["type"], "Font");
    // No `path` means the menu font compiles from the bundled default font.
    assert!(font["args"].get("path").is_none());
    assert_eq!(by_name(&assets, "m_label_0")["args"]["font"], "m_font");
    // The generated settings sub-screen shares the same font.
    assert_eq!(
        by_name(&assets, "m_settings_video_label_back")["args"]["font"],
        "m_font"
    );
    // The emitted OptionSelect carries the menu font through to its own
    // expansion.
    assert_eq!(
        by_name(&assets, "m_settings_video_opt_vsync")["args"]["font"],
        "m_font"
    );
}

#[test]
fn emitted_font_size_follows_font_px() {
    // With no override the emitted font uses the MainMenu `font_px` default.
    let mut assets =
        vec![serde_json::json!({"name":"m","type":"MainMenu","args":{"toggle_key":""}})];
    expand_main_menus(&mut assets).unwrap();
    assert_eq!(by_name(&assets, "m_font")["args"]["size_px"], 48);

    // An explicit `font_px` is the size the build leans on for the font it
    // emits when the menu declares none.
    let mut assets = vec![serde_json::json!({
        "name": "m",
        "type": "MainMenu",
        "args": { "toggle_key": "", "font_px": 32 }
    })];
    expand_main_menus(&mut assets).unwrap();
    assert_eq!(by_name(&assets, "m_font")["args"]["size_px"], 32);
}

#[test]
fn custom_font_is_used_and_none_emitted() {
    let mut assets = vec![
        serde_json::json!({"name":"f","type":"Font","args":{"path":"my.ttf","size_px":32}}),
        serde_json::json!({"name":"m","type":"MainMenu","args":{"font":"f","toggle_key":""}}),
    ];
    expand_main_menus(&mut assets).unwrap();
    assert!(!assets.iter().any(|v| asset_name(v) == "m_font"));
    assert_eq!(by_name(&assets, "m_label_0")["args"]["font"], "f");
}

// The chrome of a settings tab (heading, tab bar, scroll band, scrollbar,
// Back) stays within the reference canvas. Body rows may overflow the band
// (that is what scrolling is for) and are clipped, so they are excluded;
// only the band itself and the fixed chrome must fit.
#[test]
fn settings_chrome_and_band_fit_on_screen() {
    let [ref_w, ref_h] = UI_REFERENCE_SIZE;
    let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
    expand_main_menus(&mut assets).unwrap();
    // The Controls tab is the tallest chrome (it carries the most rows under
    // the band); its band + Back must clear the canvas.
    for suffix in ["video", "audio", "controls"] {
        let screen = format!("m_settings_{suffix}");
        // The ScrollPanel's band fits.
        let panel = by_name(&assets, &format!("{screen}_scroll"));
        let by = panel["args"]["y"].as_f64().unwrap();
        let bh = panel["args"]["height"].as_f64().unwrap();
        assert!(
            by >= 0.0 && by + bh <= ref_h as f64,
            "{screen} band off-screen"
        );
        // Back sits below the band but on screen.
        let back = by_name(&assets, &format!("{screen}_btn_back"));
        let back_y = back["args"]["y"].as_f64().unwrap();
        let back_h = back["args"]["height"].as_f64().unwrap();
        assert!(back_y >= by + bh, "{screen} Back overlaps the band");
        assert!(back_y + back_h <= ref_h as f64, "{screen} Back off-screen");
        // The scrollbar gutter stays inside the canvas width.
        let track = by_name(&assets, &format!("{screen}_scrolltrack"));
        let tx = track["args"]["x"].as_f64().unwrap();
        let tw = track["args"]["width"].as_f64().unwrap();
        assert!(
            tx + tw <= ref_w as f64,
            "{screen} scrollbar off the right edge"
        );
    }
}

// The settings body lives in a ScrollPanel: a band rect, a thumb + track,
// and one row per body element pointing at that element's expanded children.
#[test]
fn settings_tab_emits_a_scroll_panel() {
    let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
    expand_main_menus(&mut assets).unwrap();
    let panel = by_name(&assets, "m_settings_video_scroll");
    assert_eq!(panel["type"], "ScrollPanel");
    assert_eq!(panel["args"]["thumb"], "m_settings_video_scrollthumb");
    assert_eq!(panel["args"]["track"], "m_settings_video_scrolltrack");
    // The thumb + track sprites exist.
    assert_eq!(
        by_name(&assets, "m_settings_video_scrollthumb")["type"],
        "Sprite"
    );
    // The vsync row references the OptionSelect's expanded value label.
    let rows = panel["args"]["rows"].as_array().unwrap();
    let vsync_row = rows
        .iter()
        .find(|r| {
            r["elements"]
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e == "m_settings_video_opt_vsync_value")
        })
        .expect("a row listing the vsync value label");
    assert_eq!(vsync_row["group"], -1);
}

// The Video "Advanced" group (gid 1): a header row that toggles group 1, the
// render-scale row + exposure slider tagged into group 1, and a ScrollGroup
// that starts collapsed.
#[test]
fn video_advanced_group_collapses_render_scale_and_exposure() {
    let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
    expand_main_menus(&mut assets).unwrap();
    // Header label + toggle region.
    assert_eq!(
        by_name(&assets, "m_settings_video_grphdr_1")["args"]["content"],
        "+ Advanced"
    );
    assert_eq!(
        by_name(&assets, "m_settings_video_grpbtn_1")["args"]["action"],
        "group:toggle:1"
    );
    // The panel declares the Quality + Advanced groups, both collapsed.
    let panel = by_name(&assets, "m_settings_video_scroll");
    let groups = panel["args"]["groups"].as_array().unwrap();
    assert_eq!(groups.len(), 2);
    let advanced = groups
        .iter()
        .find(|g| g["header"] == "m_settings_video_grphdr_1")
        .expect("Advanced group present");
    assert_eq!(advanced["collapsed"], true);
    // render_scale + exposure rows are tagged into group 1. The first element
    // of each row is its background card, so search every element.
    let rows = panel["args"]["rows"].as_array().unwrap();
    let in_advanced = |needle: &str| {
        rows.iter()
            .find(|r| {
                r["elements"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|e| e.as_str().unwrap().contains(needle))
            })
            .map(|r| r["group"].as_i64().unwrap())
    };
    assert_eq!(in_advanced("opt_render_scale"), Some(1));
    assert_eq!(in_advanced("sld_exposure"), Some(1));
    // The live post-process sliders also live inside the Advanced group.
    for key in [
        "sld_bloom_intensity",
        "sld_bloom_threshold",
        "sld_bloom_knee",
        "sld_vignette",
        "sld_lut_strength",
        "sld_ambient_intensity",
        "sld_fov",
    ] {
        assert_eq!(
            in_advanced(key),
            Some(1),
            "{key} should be in the Advanced group"
        );
    }
    // The display-output / upscaling preference + system / streaming restart
    // rows also live in Advanced.
    for key in [
        "opt_upscale_backend",
        "opt_temporal_upscaling",
        "opt_hdr_display",
        "opt_hdr_pq",
        "opt_frames_in_flight",
        "opt_occlusion_two_pass",
        "opt_texture_quality",
    ] {
        assert_eq!(
            in_advanced(key),
            Some(1),
            "{key} should be in the Advanced group"
        );
    }
}

// The Video "Quality" group (gid 0): a header row that toggles group 0 and
// the render-feature toggles + SSGI sub-quality dropdowns tagged into group
// 0, the panel declaring it collapsed.
#[test]
fn video_quality_group_holds_render_feature_toggles() {
    let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
    expand_main_menus(&mut assets).unwrap();
    assert_eq!(
        by_name(&assets, "m_settings_video_grphdr_0")["args"]["content"],
        "+ Quality"
    );
    assert_eq!(
        by_name(&assets, "m_settings_video_grpbtn_0")["args"]["action"],
        "group:toggle:0"
    );
    let panel = by_name(&assets, "m_settings_video_scroll");
    let groups = panel["args"]["groups"].as_array().unwrap();
    let quality = groups
        .iter()
        .find(|g| g["header"] == "m_settings_video_grphdr_0")
        .expect("Quality group present");
    assert_eq!(quality["collapsed"], true);
    assert_eq!(quality["title"], "Quality");
    // Every toggle row is tagged into group 0.
    let rows = panel["args"]["rows"].as_array().unwrap();
    let group_of = |needle: &str| {
        rows.iter()
            .find(|r| {
                r["elements"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|e| e.as_str().unwrap().contains(needle))
            })
            .map(|r| r["group"].as_i64().unwrap())
    };
    for key in [
        "opt_aa_mode",
        "opt_ssao",
        "opt_ssr",
        "opt_ray_traced_reflections",
        "opt_reflection_blur_resolution",
        "opt_ssgi",
        "opt_ssgi_resolution",
        "opt_ssgi_rays",
        "opt_ssgi_steps",
        "opt_shadow_map_size",
        "opt_shadow_update",
        "opt_shadow_distance",
        "opt_shadow_cascades",
        "opt_auto_exposure",
        "opt_anisotropy",
        // The per-feature sub-quality sliders share the Quality group.
        "sld_ssao_radius",
        "sld_ssao_intensity",
        "sld_ssr_intensity",
        "sld_ssr_max_distance",
        "sld_ssgi_intensity",
        "sld_ssgi_max_distance",
        "sld_auto_exposure_min_ev",
        "sld_auto_exposure_max_ev",
        "sld_auto_exposure_speed",
    ] {
        assert_eq!(
            group_of(key),
            Some(0),
            "{key} should be in the Quality group"
        );
    }
}

// Regression: the gid in a group header's `group:toggle:<gid>` action is
// used at runtime as an INDEX into `ScrollPanel.groups`, and a row's `group`
// tag is the same index. So each group's position in the groups vec must
// equal the gid baked into its header/row references. A mismatch toggled the
// wrong group (clicking "Quality" flipped "Advanced" and vice versa).
#[test]
fn group_toggle_gid_indexes_its_own_group() {
    let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
    expand_main_menus(&mut assets).unwrap();
    let panel = by_name(&assets, "m_settings_video_scroll");
    let groups = panel["args"]["groups"].as_array().unwrap();
    for (gid, group) in groups.iter().enumerate() {
        // The group at index `gid` owns the `grphdr_<gid>` / `grpbtn_<gid>`
        // header, whose toggle action carries that same gid.
        let header_name = format!("m_settings_video_grphdr_{gid}");
        assert_eq!(group["header"], header_name, "group {gid} out of gid order");
        assert_eq!(
            by_name(&assets, &format!("m_settings_video_grpbtn_{gid}"))["args"]["action"],
            format!("group:toggle:{gid}")
        );
        // The header label's title matches the group at this index.
        let content = by_name(&assets, &header_name)["args"]["content"]
            .as_str()
            .unwrap()
            .to_string();
        let title = group["title"].as_str().unwrap();
        assert!(
            content.ends_with(title),
            "header {content:?} does not match group {title:?} at index {gid}"
        );
    }
}

// Every settings body row gets a semi-transparent card background, drawn
// before (behind) the row's content and listed first in the row's elements
// so it reflows / clips / hides with the row.
#[test]
fn settings_rows_have_card_backgrounds() {
    let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
    expand_main_menus(&mut assets).unwrap();
    let panel = by_name(&assets, "m_settings_video_scroll");
    let rows = panel["args"]["rows"].as_array().unwrap();
    for row in rows {
        let elems = row["elements"].as_array().unwrap();
        let bg = elems[0].as_str().unwrap();
        assert!(
            bg.contains("_bg_"),
            "row missing a leading bg element: {elems:?}"
        );
        let sprite = by_name(&assets, bg);
        assert_eq!(sprite["type"], "Sprite");
        let alpha = sprite["args"]["tint"][3].as_f64().unwrap();
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "bg card should be semi-transparent"
        );
    }
    // The first row's card is declared before that row's content (the vsync
    // OptionSelect), so the shared sprite/text pass draws it behind the row.
    let names: Vec<String> = assets.iter().map(asset_name).collect();
    let bg_idx = names
        .iter()
        .position(|n| n == "m_settings_video_bg_0")
        .expect("first row card");
    let content_idx = names
        .iter()
        .position(|n| n == "m_settings_video_opt_vsync")
        .expect("vsync row");
    assert!(
        bg_idx < content_idx,
        "card must precede row content for z-order"
    );
}

// Hover is color-only by default: each generated HitRegion's hover_scale
// equals the scale of the label it restyles (the MainMenu hover_scale
// multiplier defaults to 1.0), so a hovered item changes color without
// growing or shifting out of its card.
#[test]
fn default_menu_hover_is_color_only() {
    let mut assets =
        vec![serde_json::json!({"name":"m","type":"MainMenu","args":{"title":"Paused"}})];
    expand_main_menus(&mut assets).unwrap();

    let label_scale = |name: &str| by_name(&assets, name)["args"]["scale"].as_f64().unwrap();

    // Raw HitRegions (menu items, tabs, group headers, Back): the region's
    // hover_scale matches its label's scale, so hover does not resize it.
    let mut checked = 0;
    for v in &assets {
        if type_norm(v) != "hitregion" {
            continue;
        }
        let args = &v["args"];
        let (Some(label), Some(hs)) = (args["label"].as_str(), args["hover_scale"].as_f64()) else {
            continue;
        };
        if label.is_empty() {
            continue;
        }
        let ls = label_scale(label);
        assert!(
            (hs - ls).abs() < 1e-6,
            "{}: hover_scale {hs} != label scale {ls}",
            asset_name(v)
        );
        checked += 1;
    }
    assert!(checked > 0, "no labeled hover regions were checked");

    // OptionSelect rows carry an absolute hover_scale equal to their text
    // scale, so the value label also keeps its size on hover.
    for v in &assets {
        if type_norm(v) != "optionselect" {
            continue;
        }
        let ts = v["args"]["text_scale"].as_f64().unwrap();
        let hs = v["args"]["hover_scale"].as_f64().unwrap();
        assert!(
            (hs - ts).abs() < 1e-6,
            "optionselect hover_scale {hs} != text_scale {ts}"
        );
    }
}

// A non-default hover_scale still grows the hovered text, as a multiplier on
// each item's own scale, so the emphasis feature is preserved.
#[test]
fn hover_scale_multiplies_label_scale() {
    let mut assets = vec![serde_json::json!({
        "name":"m","type":"MainMenu","args":{"hover_scale":2.0,"toggle_key":""}
    })];
    expand_main_menus(&mut assets).unwrap();
    let label_scale = by_name(&assets, "m_label_0")["args"]["scale"]
        .as_f64()
        .unwrap();
    let region_hs = by_name(&assets, "m_btn_0")["args"]["hover_scale"]
        .as_f64()
        .unwrap();
    assert!(
        (region_hs - label_scale * 2.0).abs() < 1e-6,
        "expected {} got {region_hs}",
        label_scale * 2.0
    );
}

// The row content (the OptionSelect) is inset within its card by the same
// padding on the left and right, so the name does not touch the card edge.
#[test]
fn settings_row_content_is_inset_evenly_within_card() {
    let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
    expand_main_menus(&mut assets).unwrap();
    let card = by_name(&assets, "m_settings_video_bg_0");
    let opt = by_name(&assets, "m_settings_video_opt_vsync");
    let card_x = card["args"]["x"].as_f64().unwrap();
    let card_w = card["args"]["width"].as_f64().unwrap();
    let opt_x = opt["args"]["x"].as_f64().unwrap();
    let opt_w = opt["args"]["width"].as_f64().unwrap();
    let left_pad = opt_x - card_x;
    let right_pad = (card_x + card_w) - (opt_x + opt_w);
    assert!(left_pad > 0.0, "content should be inset from the left edge");
    assert!(
        (left_pad - right_pad).abs() < 1e-3,
        "left pad {left_pad} != right pad {right_pad}"
    );
}

// The Minimal profile emits only a Video and an Audio tab; there is no
// Controls tab screen and no tab button pointing at one.
#[test]
fn minimal_profile_emits_only_video_and_audio_tabs() {
    let mut assets = vec![serde_json::json!({
        "name": "m", "type": "MainMenu", "args": { "settings_profile": "minimal" }
    })];
    expand_main_menus(&mut assets).unwrap();

    assert_eq!(by_name(&assets, "m_settings_video")["type"], "Screen");
    assert_eq!(by_name(&assets, "m_settings_audio")["type"], "Screen");
    assert!(
        !assets
            .iter()
            .any(|v| asset_name(v) == "m_settings_controls"),
        "Minimal must not emit a Controls tab screen"
    );
    // The Video tab bar switches to Audio but offers no Controls button.
    assert_eq!(
        by_name(&assets, "m_settings_video_tabbtn_audio")["args"]["action"],
        "screen:show:m_settings_audio"
    );
    assert!(
        !assets
            .iter()
            .any(|v| asset_name(v) == "m_settings_video_tabbtn_controls"),
        "Minimal Video tab bar must not offer Controls"
    );
}

// The Minimal Video tab is trimmed to the window / output basics: window
// mode, resolution, vsync, frame rate. No graphics-quality preset, no
// performance-stats toggles, and no Quality / Advanced groups.
#[test]
fn minimal_video_tab_is_trimmed_to_output_basics() {
    let mut assets = vec![serde_json::json!({
        "name": "m", "type": "MainMenu", "args": { "settings_profile": "minimal" }
    })];
    expand_main_menus(&mut assets).unwrap();

    for setting in ["window_mode", "resolution", "vsync", "fps_cap"] {
        let opt = by_name(&assets, &format!("m_settings_video_opt_{setting}"));
        assert_eq!(opt["type"], "OptionSelect");
        assert_eq!(opt["args"]["setting"], setting);
    }
    for dropped in [
        "m_settings_video_opt_graphics_quality",
        "m_settings_video_opt_perf_stats",
        "m_settings_video_opt_show_fps",
        "m_settings_video_opt_show_vram",
        "m_settings_video_opt_ssgi",
        "m_settings_video_grphdr_0",
        "m_settings_video_grphdr_1",
    ] {
        assert!(
            !assets.iter().any(|v| asset_name(v) == dropped),
            "Minimal Video must not emit {dropped}"
        );
    }
    // The scroll panel declares no collapsible groups.
    let panel = by_name(&assets, "m_settings_video_scroll");
    assert!(panel["args"]["groups"].as_array().unwrap().is_empty());
}

// The Minimal Audio tab keeps the master-volume row (it shares the Full
// Audio body).
#[test]
fn minimal_audio_tab_keeps_master_volume() {
    let mut assets = vec![serde_json::json!({
        "name": "m", "type": "MainMenu", "args": { "settings_profile": "minimal" }
    })];
    expand_main_menus(&mut assets).unwrap();
    let vol = by_name(&assets, "m_settings_audio_opt_master_volume");
    assert_eq!(vol["type"], "OptionSelect");
    assert_eq!(vol["args"]["setting"], "master_volume");
}

// A `settings_back_action` generates the settings screen even with no
// "settings" item and routes the Back button through it (rather than the
// default screen:show:<menu>).
#[test]
fn settings_back_action_generates_screen_and_overrides_back() {
    let mut assets = vec![serde_json::json!({
        "name": "m",
        "type": "MainMenu",
        "args": {
            "settings_profile": "minimal",
            "settings_back_action": "story:settings_back",
            "items": [{"label": "Settings", "action": "story:settings"}]
        }
    })];
    expand_main_menus(&mut assets).unwrap();
    // The screen is generated despite no item using the "settings"
    // convenience.
    assert_eq!(by_name(&assets, "m_settings_video")["type"], "Screen");
    // Both tabs' Back buttons fire the override.
    for suffix in ["video", "audio"] {
        assert_eq!(
            by_name(&assets, &format!("m_settings_{suffix}_btn_back"))["args"]["action"],
            "story:settings_back"
        );
    }
}

// Menu item labels, the heading, and Back center with real font metrics
// (align center) rather than the glyph-width estimate.
#[test]
fn menu_labels_center_with_real_metrics() {
    let mut assets =
        vec![serde_json::json!({"name":"m","type":"MainMenu","args":{"title":"Paused"}})];
    expand_main_menus(&mut assets).unwrap();
    assert_eq!(by_name(&assets, "m_title")["args"]["align"], "center");
    assert_eq!(by_name(&assets, "m_label_0")["args"]["align"], "center");
    assert_eq!(
        by_name(&assets, "m_settings_video_label_back")["args"]["align"],
        "center"
    );
}

// The default (Full) profile is unchanged: it still emits the Controls tab
// and the graphics-quality preset the trimmed profile drops.
#[test]
fn full_profile_still_emits_controls_and_quality() {
    let mut assets = vec![serde_json::json!({"name": "m", "type": "MainMenu"})];
    expand_main_menus(&mut assets).unwrap();
    assert_eq!(by_name(&assets, "m_settings_controls")["type"], "Screen");
    assert_eq!(
        by_name(&assets, "m_settings_video_opt_graphics_quality")["type"],
        "OptionSelect"
    );
}
