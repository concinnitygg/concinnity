// Which rows a settings tab shows, and how one row is emitted. The tables name
// the setting keys and labels; the runtime
// (`concinnity_engine::gfx::settings`) knows each key's options and how to
// apply it.

use crate::authoring::registry::build_only::{MainMenu, SettingsProfile};

// Settings tabs, left to right: (screen-name suffix, tab label). Each tab is its
// own Screen; the active tab bakes its own highlight, so switching tabs needs no
// runtime state, only a screen:show.
const SETTINGS_TABS: [(&str, &str); 3] = [
    ("video", "Video"),
    ("audio", "Audio"),
    ("controls", "Controls"),
];

// The tabs a settings profile shows, left to right. Full spans all three;
// Minimal drops Controls (a world with no scene to move a camera through has
// no gameplay keys to rebind), leaving Video + Audio.
pub(super) fn settings_tabs(profile: SettingsProfile) -> &'static [(&'static str, &'static str)] {
    match profile {
        SettingsProfile::Full => &SETTINGS_TABS,
        SettingsProfile::Minimal => &SETTINGS_TABS[..2],
    }
}

// Setting rows per tab, top to bottom: (setting key, display label). The runtime
// (`concinnity_engine::gfx::settings`) knows each key's options and how to apply
// it; this only chooses which rows appear.
const VIDEO_ROWS: [(&str, &str); 7] = [
    ("vsync", "Vsync"),
    ("fps_cap", "Frame Rate"),
    ("window_mode", "Window Mode"),
    ("resolution", "Resolution"),
    // Stats-HUD display: the master toggle leads, then the per-readout toggles.
    // The master grays the two sub-rows out (rather than hiding them) when off.
    ("perf_stats", "Display Performance Stats"),
    ("show_fps", "Show Framerate"),
    ("show_vram", "Show VRAM Usage"),
];
// Video rows under the Minimal profile: window and output basics only, for a
// world that renders no 3D scene (nothing to configure quality for). No
// graphics-quality preset, no performance-stats toggles, no Quality / Advanced
// groups.
const VIDEO_MINIMAL_ROWS: [(&str, &str); 4] = [
    ("window_mode", "Window Mode"),
    ("resolution", "Resolution"),
    ("vsync", "Vsync"),
    ("fps_cap", "Frame Rate"),
];
// Rows tucked under the Video "Advanced" collapsible group (collapsed by
// default), so the top of the Video tab stays uncrowded. More live
// post-process sliders join these later. Cycle rows then slider rows.
const VIDEO_ADVANCED_ROWS: [(&str, &str); 8] = [
    ("render_scale", "Render Scale"),
    // Upscaler backend (Auto/FSR3/DLSS/XeSS). Restart-required, independent of the
    // quality preset; DirectX / Vulkan only (Metal uses MetalFX, so the row is
    // inert there). Sits next to render scale since it only matters with temporal
    // upscaling on.
    ("upscale_backend", "Upscaler"),
    // Display-output / upscaling preferences (Off/On + render-scale cycle).
    // Restart-required and independent of the quality preset.
    ("temporal_upscaling", "Temporal Upscaling"),
    ("hdr_display", "HDR Display"),
    ("hdr_pq", "HDR10 (PQ)"),
    // System / streaming restart preferences. Buffering depth, two-pass occlusion
    // culling, and texture-streaming quality (pool size + upload budget together).
    ("frames_in_flight", "Frame Buffering"),
    ("occlusion_two_pass", "Occlusion Culling"),
    ("texture_quality", "Texture Quality"),
];
// Live post-process sliders in the Advanced group. Each key's value range,
// display format, and apply path live in the client (`concinnity_engine::gfx::settings` +
// `graphics_system`); a row here only chooses which sliders appear. All but
// `ambient_intensity` are pure `PostProcessParams` fields applied via
// `update_post_process`; `ambient_intensity` rides a dedicated backend setter
// (Metal live; see the client `graphics_system`).
const VIDEO_ADVANCED_SLIDERS: [(&str, &str); 8] = [
    ("exposure", "Exposure"),
    ("bloom_intensity", "Bloom"),
    ("bloom_threshold", "Bloom Threshold"),
    ("bloom_knee", "Bloom Knee"),
    ("vignette", "Vignette"),
    ("lut_strength", "Color Grade"),
    ("ambient_intensity", "Ambient"),
    // Camera vertical field of view (degrees). Live, independent of the preset.
    ("fov", "Field of View"),
];
// Quality toggles in the Video "Quality" collapsible group (collapsed by
// default): the heavier render features. Each is an Off/On cycle row. The
// client (`concinnity_engine::gfx::settings` + `graphics_system`) knows each key's options and
// applies it live by rebuilding the affected render resources; on backends
// without a live path the choice persists and applies at the next launch.
const VIDEO_QUALITY_ROWS: [(&str, &str); 15] = [
    ("aa_mode", "Anti-Aliasing"),
    ("ssao", "Ambient Occlusion"),
    ("ssr", "Screen-Space Reflections"),
    ("ray_traced_reflections", "Ray-Traced Reflections"),
    // Reflection blur resolution dropdown, grouped under the reflection toggles
    // it governs (SSR + ray-traced).
    ("reflection_blur_resolution", "Reflection Blur"),
    ("ssgi", "Global Illumination"),
    // SSGI gather sub-quality (multi-option dropdowns), grouped under the GI
    // toggle. The runtime knows each key's options and applies them live.
    ("ssgi_resolution", "GI Resolution"),
    ("ssgi_rays", "GI Rays"),
    ("ssgi_steps", "GI Steps"),
    // Shadow quality: cascade map resolution (restart-required) + re-render
    // cadence (live) + distance (live) + cascade count (live). Preset-governed
    // like the toggles above.
    ("shadow_map_size", "Shadow Resolution"),
    ("shadow_update", "Shadow Update"),
    ("shadow_distance", "Shadow Distance"),
    ("shadow_cascades", "Shadow Cascades"),
    ("auto_exposure", "Auto Exposure"),
    // Anisotropic texture filtering (restart-required). Preset-governed like the
    // toggles above.
    ("anisotropy", "Anisotropic Filtering"),
];
// Per-feature sub-quality sliders in the Video "Quality" group, tuning the
// features the toggles / dropdowns above enable. Applied live on Metal by
// mutating the backend's stored *Settings (no pass rebuild); look-tuning knobs,
// independent of the master quality preset.
const VIDEO_QUALITY_SLIDERS: [(&str, &str); 9] = [
    ("ssao_radius", "AO Radius"),
    ("ssao_intensity", "AO Intensity"),
    ("ssr_intensity", "Reflection Intensity"),
    ("ssr_max_distance", "Reflection Distance"),
    ("ssgi_intensity", "GI Intensity"),
    ("ssgi_max_distance", "GI Distance"),
    ("auto_exposure_min_ev", "Auto Exposure Min"),
    ("auto_exposure_max_ev", "Auto Exposure Max"),
    ("auto_exposure_speed", "Auto Exposure Speed"),
];
const AUDIO_ROWS: [(&str, &str); 4] = [
    ("master_volume", "Master Volume"),
    ("music_volume", "Music Volume"),
    ("sfx_volume", "SFX Volume"),
    ("voice_volume", "Voice Volume"),
];
// Controls-tab sliders, top to bottom: (setting key, display label). Mouse
// sensitivity is a continuous slider (the client maps the 1..100 track to a
// radians-per-pixel value) applied live by the camera controller.
const CONTROLS_SLIDERS: [(&str, &str); 1] = [("mouse_sensitivity", "Sensitivity")];
// Rebindable gameplay actions shown under the Controls tab: (display label,
// setting key). Each emits a clickable row that captures a new key; the client
// (`concinnity_engine::gfx::keymap` + `graphics_system`) owns the live key map and applies a
// rebind without a restart. The setting keys match `Bindable::setting_key`.
const CONTROLS_REBINDS: [(&str, &str); 7] = [
    ("Move Forward", "key_forward"),
    ("Move Back", "key_backward"),
    ("Move Left", "key_left"),
    ("Move Right", "key_right"),
    ("Sprint", "key_sprint"),
    ("Jump", "key_jump"),
    ("Interact", "key_interact"),
];
// Read-only key reference shown under the Controls tab: (action, key). Pause
// (Escape) carries cursor-release / menu semantics that are fixed per-backend,
// so it is shown for reference rather than made rebindable.
const CONTROLS_KEYS: [(&str, &str); 1] = [("Pause", "Esc")];
// Gamepad sliders in the Controls "Gamepad" group: look-stick sensitivity
// (1..100 mapped to a radians-per-second rate) and the radial stick deadzone
// (shown as a percentage of deflection). Both applied live via ControlsCommand.
const CONTROLS_PAD_SLIDERS: [(&str, &str); 2] = [
    ("gamepad_look_sensitivity", "Stick Sensitivity"),
    ("gamepad_deadzone", "Stick Deadzone"),
];
// Rebindable gamepad actions in the same group: (display label, setting key).
// Each emits a clickable row that captures a button press; the setting keys
// match `GamepadAction::setting_key`. Movement and look ride the sticks (with
// the d-pad as a digital fallback) and pause rides Start, so only the
// button-driven actions are rebindable.
const CONTROLS_PAD_REBINDS: [(&str, &str); 3] = [
    ("Sprint", "pad_sprint"),
    ("Jump", "pad_jump"),
    ("Interact", "pad_interact"),
];

// One row of a settings tab's scrollable body.
#[derive(Clone, Copy)]
pub(super) enum BodyRow {
    // An OptionSelect cycle row: (setting key, label, group index or -1).
    Option(&'static str, &'static str, i32),
    // A Slider row: (setting key, label, group index or -1).
    Slider(&'static str, &'static str, i32),
    // A read-only key-reference row: (action label, key text, index, group).
    Key(&'static str, &'static str, usize, i32),
    // A key-rebind row: (action label, setting key, index, group). Like a Key
    // row but with a HitRegion that captures a new binding on click.
    Rebind(&'static str, &'static str, usize, i32),
    // A collapsible-group header: (group index, title). Always shown.
    GroupHeader(usize, &'static str),
}

// A collapsible group declared by a tab.
pub(super) struct GroupSpec {
    pub(super) gid: usize,
    pub(super) title: &'static str,
    pub(super) collapsed: bool,
}

// The body rows + collapsible groups for one settings tab, top to bottom.
pub(super) fn settings_body_rows(
    active: &str,
    profile: SettingsProfile,
) -> (Vec<BodyRow>, Vec<GroupSpec>) {
    // The Minimal profile shows a trimmed Video tab (window / output basics
    // only) and shares the Full Audio tab; it never emits a Controls tab.
    if profile == SettingsProfile::Minimal && active != "audio" {
        let rows = VIDEO_MINIMAL_ROWS
            .iter()
            .map(|&(s, l)| BodyRow::Option(s, l, -1))
            .collect();
        return (rows, Vec::new());
    }
    match active {
        "audio" => (
            AUDIO_ROWS
                .iter()
                .map(|&(s, l)| BodyRow::Option(s, l, -1))
                .collect(),
            Vec::new(),
        ),
        "controls" => {
            let mut rows: Vec<BodyRow> = CONTROLS_SLIDERS
                .iter()
                .map(|&(s, l)| BodyRow::Slider(s, l, -1))
                .collect();
            // Rebindable gameplay keys, each a clickable capture row.
            for (i, &(label, setting)) in CONTROLS_REBINDS.iter().enumerate() {
                rows.push(BodyRow::Rebind(label, setting, i, -1));
            }
            // Read-only reference (Pause / Escape) below the rebindable rows.
            for (i, &(action, key)) in CONTROLS_KEYS.iter().enumerate() {
                rows.push(BodyRow::Key(action, key, i, -1));
            }
            // The gamepad rows sit in their own collapsible group. The rebind
            // indices continue the keyboard rows' sequence so every rebind
            // row's element names stay unique within the screen.
            rows.push(BodyRow::GroupHeader(0, "Gamepad"));
            for &(s, l) in &CONTROLS_PAD_SLIDERS {
                rows.push(BodyRow::Slider(s, l, 0));
            }
            for (i, &(label, setting)) in CONTROLS_PAD_REBINDS.iter().enumerate() {
                rows.push(BodyRow::Rebind(
                    label,
                    setting,
                    CONTROLS_REBINDS.len() + i,
                    0,
                ));
            }
            (
                rows,
                vec![GroupSpec {
                    gid: 0,
                    title: "Gamepad",
                    collapsed: true,
                }],
            )
        }
        // Video: the three core rows, then a "Quality" group holding the
        // render-feature toggles, then an "Advanced" group holding the
        // render-scale row + the live sliders. Both groups collapsed by default
        // so the top of the tab stays uncrowded.
        //
        // A group's `gid` is used at runtime as an index into the panel's groups
        // list, so each group's gid MUST equal its position in the `GroupSpec`
        // vec below (and a row's group tag references that same gid). Quality is
        // declared first, so it is gid 0; Advanced second, so gid 1.
        _ => {
            // The master "Graphics Quality" preset leads the tab (ungrouped, so it
            // is always visible); the runtime cycles Auto/Low/Medium/High/Ultra/
            // Custom and re-derives the toggles + render scale under its ceiling.
            let mut rows: Vec<BodyRow> =
                vec![BodyRow::Option("graphics_quality", "Graphics Quality", -1)];
            rows.extend(VIDEO_ROWS.iter().map(|&(s, l)| BodyRow::Option(s, l, -1)));
            rows.push(BodyRow::GroupHeader(0, "Quality"));
            for &(s, l) in &VIDEO_QUALITY_ROWS {
                rows.push(BodyRow::Option(s, l, 0));
            }
            // The per-feature sub-quality sliders follow the toggles in the same
            // Quality group.
            for &(s, l) in &VIDEO_QUALITY_SLIDERS {
                rows.push(BodyRow::Slider(s, l, 0));
            }
            rows.push(BodyRow::GroupHeader(1, "Advanced"));
            for &(s, l) in &VIDEO_ADVANCED_ROWS {
                rows.push(BodyRow::Option(s, l, 1));
            }
            for &(s, l) in &VIDEO_ADVANCED_SLIDERS {
                rows.push(BodyRow::Slider(s, l, 1));
            }
            (
                rows,
                vec![
                    GroupSpec {
                        gid: 0,
                        title: "Quality",
                        collapsed: true,
                    },
                    GroupSpec {
                        gid: 1,
                        title: "Advanced",
                        collapsed: true,
                    },
                ],
            )
        }
    }
}

// A settings-body row: its element name, the setting it drives, display label,
// font, position/size in overlay space, text scale, and the menu style it
// inherits colors and row height from. Shared by the OptionSelect and Slider
// row builders, which take the same inputs.
pub(super) struct SettingsRow<'a> {
    pub(super) name: &'a str,
    pub(super) setting: &'a str,
    pub(super) label: &'a str,
    pub(super) font: &'a str,
    pub(super) x: f32,
    pub(super) y: f32,
    pub(super) width: f32,
    pub(super) scale: f32,
    pub(super) style: &'a MainMenu,
}

// Build an OptionSelect cycle-row asset for the settings body.
pub(super) fn option_select_row(row: &SettingsRow) -> serde_json::Value {
    let &SettingsRow {
        name,
        setting,
        label,
        font,
        x,
        y,
        width,
        scale,
        style,
    } = row;
    serde_json::json!({
        "name": name,
        "type": "OptionSelect",
        "args": {
            "setting": setting,
            "label": label,
            "x": x,
            "y": y,
            "width": width,
            "height": style.button_height,
            "font": font,
            "text_color": style.text_color,
            "value_color": style.text_color,
            "text_scale": scale,
            "hover_color": style.hover_color,
            // `style.hover_scale` is a multiplier on the row's text scale, so the
            // value label keeps its size on hover (only the color changes) unless
            // the menu opts into a grow. The OptionSelect forwards this absolute
            // scale to its value-label hover region.
            "hover_scale": scale * style.hover_scale,
        }
    })
}

// Build a Slider row asset for the settings body.
pub(super) fn slider_row(row: &SettingsRow) -> serde_json::Value {
    let &SettingsRow {
        name,
        setting,
        label,
        font,
        x,
        y,
        width,
        scale,
        style,
    } = row;
    serde_json::json!({
        "name": name,
        "type": "Slider",
        "args": {
            "setting": setting,
            "label": label,
            "x": x,
            "y": y,
            "width": width,
            "height": style.button_height,
            "font": font,
            "text_color": style.text_color,
            "value_color": style.text_color,
            "text_scale": scale,
            "handle_color": [
                style.hover_color[0], style.hover_color[1], style.hover_color[2], 1.0
            ],
        }
    })
}
