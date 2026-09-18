// Which rows a settings tab shows, and how one row is emitted. The tables pair
// each `SettingKey` with its label; core's `SettingKey` owns the key vocabulary
// and the runtime (`concinnity_engine::settings`) knows each key's options and
// how to apply it.

use concinnity_core::components::GamepadAction;
use concinnity_core::input::keymap::Bindable;
use concinnity_core::settings::SettingKey;

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

// Setting rows per tab, top to bottom: (setting, display label). The runtime
// (`concinnity_engine::settings`) knows each setting's options and how to apply
// it; this only chooses which rows appear.
const VIDEO_ROWS: [(SettingKey, &str); 7] = [
    (SettingKey::Vsync, "Vsync"),
    (SettingKey::FpsCap, "Frame Rate"),
    (SettingKey::WindowMode, "Window Mode"),
    (SettingKey::Resolution, "Resolution"),
    // Stats-HUD display: the master toggle leads, then the per-readout toggles.
    // The master grays the two sub-rows out (rather than hiding them) when off.
    (SettingKey::PerfStats, "Display Performance Stats"),
    (SettingKey::ShowFps, "Show Framerate"),
    (SettingKey::ShowVram, "Show VRAM Usage"),
];
// Video rows under the Minimal profile: window and output basics only, for a
// world that renders no 3D scene (nothing to configure quality for). No
// graphics-quality preset, no performance-stats toggles, no Quality / Advanced
// groups.
const VIDEO_MINIMAL_ROWS: [(SettingKey, &str); 4] = [
    (SettingKey::WindowMode, "Window Mode"),
    (SettingKey::Resolution, "Resolution"),
    (SettingKey::Vsync, "Vsync"),
    (SettingKey::FpsCap, "Frame Rate"),
];
// Rows tucked under the Video "Advanced" collapsible group (collapsed by
// default), so the top of the Video tab stays uncrowded. More live
// post-process sliders join these later. Cycle rows then slider rows.
const VIDEO_ADVANCED_ROWS: [(SettingKey, &str); 8] = [
    (SettingKey::RenderScale, "Render Scale"),
    // Upscaler backend (Auto/FSR3/DLSS/XeSS). Restart-required, independent of the
    // quality preset; DirectX / Vulkan only (Metal uses MetalFX, so the row is
    // inert there). Sits next to render scale since it only matters with temporal
    // upscaling on.
    (SettingKey::UpscaleBackend, "Upscaler"),
    // Display-output / upscaling preferences (Off/On + render-scale cycle).
    // Restart-required and independent of the quality preset.
    (SettingKey::TemporalUpscaling, "Temporal Upscaling"),
    (SettingKey::HdrDisplay, "HDR Display"),
    (SettingKey::HdrPq, "HDR10 (PQ)"),
    // System / streaming restart preferences. Buffering depth, two-pass occlusion
    // culling, and texture-streaming quality (pool size + upload budget together).
    (SettingKey::FramesInFlight, "Frame Buffering"),
    (SettingKey::OcclusionTwoPass, "Occlusion Culling"),
    (SettingKey::TextureQuality, "Texture Quality"),
];
// Live post-process sliders in the Advanced group. Each slider's value range,
// display format, and apply path live in the client (`concinnity_engine::settings` +
// graphics system); a row here only chooses which sliders appear. All but
// `AmbientIntensity` are pure `PostProcessParams` fields applied via
// `update_post_process`; `AmbientIntensity` rides a dedicated backend setter
// (Metal live; see the client graphics system).
const VIDEO_ADVANCED_SLIDERS: [(SettingKey, &str); 8] = [
    (SettingKey::Exposure, "Exposure"),
    (SettingKey::BloomIntensity, "Bloom"),
    (SettingKey::BloomThreshold, "Bloom Threshold"),
    (SettingKey::BloomKnee, "Bloom Knee"),
    (SettingKey::Vignette, "Vignette"),
    (SettingKey::LutStrength, "Color Grade"),
    (SettingKey::AmbientIntensity, "Ambient"),
    // Camera vertical field of view (degrees). Live, independent of the preset.
    (SettingKey::Fov, "Field of View"),
];
// Quality toggles in the Video "Quality" collapsible group (collapsed by
// default): the heavier render features. Each is an Off/On cycle row. The
// client (`concinnity_engine::settings` + its system) knows each setting's options and
// applies it live by rebuilding the affected render resources; on backends
// without a live path the choice persists and applies at the next launch.
const VIDEO_QUALITY_ROWS: [(SettingKey, &str); 15] = [
    (SettingKey::AaMode, "Anti-Aliasing"),
    (SettingKey::Ssao, "Ambient Occlusion"),
    (SettingKey::Ssr, "Screen-Space Reflections"),
    (SettingKey::RayTracedReflections, "Ray-Traced Reflections"),
    // Reflection blur resolution dropdown, grouped under the reflection toggles
    // it governs (SSR + ray-traced).
    (SettingKey::ReflectionBlurResolution, "Reflection Blur"),
    (SettingKey::Ssgi, "Global Illumination"),
    // SSGI gather sub-quality (multi-option dropdowns), grouped under the GI
    // toggle. The runtime knows each key's options and applies them live.
    (SettingKey::SsgiResolution, "GI Resolution"),
    (SettingKey::SsgiRays, "GI Rays"),
    (SettingKey::SsgiSteps, "GI Steps"),
    // Shadow quality: cascade map resolution (restart-required) + re-render
    // cadence (live) + distance (live) + cascade count (live). Preset-governed
    // like the toggles above.
    (SettingKey::ShadowMapSize, "Shadow Resolution"),
    (SettingKey::ShadowUpdate, "Shadow Update"),
    (SettingKey::ShadowDistance, "Shadow Distance"),
    (SettingKey::ShadowCascades, "Shadow Cascades"),
    (SettingKey::AutoExposure, "Auto Exposure"),
    // Anisotropic texture filtering (restart-required). Preset-governed like the
    // toggles above.
    (SettingKey::Anisotropy, "Anisotropic Filtering"),
];
// Per-feature sub-quality sliders in the Video "Quality" group, tuning the
// features the toggles / dropdowns above enable. Applied live on Metal by
// mutating the backend's stored *Settings (no pass rebuild); look-tuning knobs,
// independent of the master quality preset.
const VIDEO_QUALITY_SLIDERS: [(SettingKey, &str); 9] = [
    (SettingKey::SsaoRadius, "AO Radius"),
    (SettingKey::SsaoIntensity, "AO Intensity"),
    (SettingKey::SsrIntensity, "Reflection Intensity"),
    (SettingKey::SsrMaxDistance, "Reflection Distance"),
    (SettingKey::SsgiIntensity, "GI Intensity"),
    (SettingKey::SsgiMaxDistance, "GI Distance"),
    (SettingKey::AutoExposureMinEv, "Auto Exposure Min"),
    (SettingKey::AutoExposureMaxEv, "Auto Exposure Max"),
    (SettingKey::AutoExposureSpeed, "Auto Exposure Speed"),
];
const AUDIO_ROWS: [(SettingKey, &str); 4] = [
    (SettingKey::MasterVolume, "Master Volume"),
    (SettingKey::MusicVolume, "Music Volume"),
    (SettingKey::SfxVolume, "SFX Volume"),
    (SettingKey::VoiceVolume, "Voice Volume"),
];
// Controls-tab sliders, top to bottom: (setting, display label). Mouse
// sensitivity is a continuous slider (the client maps the 1..100 track to a
// radians-per-pixel value) applied live by the camera controller.
const CONTROLS_SLIDERS: [(SettingKey, &str); 1] = [(SettingKey::MouseSensitivity, "Sensitivity")];
// Rebindable gameplay actions shown under the Controls tab: (display label,
// setting). Each emits a clickable row that captures a new key; the client
// (`concinnity_core::input::keymap` + the graphics system) owns the live key map and applies a
// rebind without a restart.
const CONTROLS_REBINDS: [(&str, SettingKey); 7] = [
    ("Move Forward", SettingKey::KeyRebind(Bindable::Forward)),
    ("Move Back", SettingKey::KeyRebind(Bindable::Backward)),
    ("Move Left", SettingKey::KeyRebind(Bindable::Left)),
    ("Move Right", SettingKey::KeyRebind(Bindable::Right)),
    ("Sprint", SettingKey::KeyRebind(Bindable::Sprint)),
    ("Jump", SettingKey::KeyRebind(Bindable::Jump)),
    ("Interact", SettingKey::KeyRebind(Bindable::Interact)),
];
// Read-only key reference shown under the Controls tab: (action, key). Pause
// (Escape) carries cursor-release / menu semantics that are fixed per-backend,
// so it is shown for reference rather than made rebindable.
const CONTROLS_KEYS: [(&str, &str); 1] = [("Pause", "Esc")];
// Gamepad sliders in the Controls "Gamepad" group: look-stick sensitivity
// (1..100 mapped to a radians-per-second rate) and the radial stick deadzone
// (shown as a percentage of deflection). Both applied live via ControlsCommand.
const CONTROLS_PAD_SLIDERS: [(SettingKey, &str); 2] = [
    (SettingKey::GamepadLookSensitivity, "Stick Sensitivity"),
    (SettingKey::GamepadDeadzone, "Stick Deadzone"),
];
// Rebindable gamepad actions in the same group: (display label, setting).
// Each emits a clickable row that captures a button press. Movement and look
// ride the sticks (with the d-pad as a digital fallback) and pause rides Start,
// so only the button-driven actions are rebindable.
const CONTROLS_PAD_REBINDS: [(&str, SettingKey); 3] = [
    ("Sprint", SettingKey::PadRebind(GamepadAction::Sprint)),
    ("Jump", SettingKey::PadRebind(GamepadAction::Jump)),
    ("Interact", SettingKey::PadRebind(GamepadAction::Interact)),
];

// One row of a settings tab's scrollable body.
#[derive(Clone, Copy)]
pub(super) enum BodyRow {
    // An OptionSelect cycle row: (setting, label, group index or -1).
    Option(SettingKey, &'static str, i32),
    // A Slider row: (setting, label, group index or -1).
    Slider(SettingKey, &'static str, i32),
    // A read-only key-reference row: (action label, key text, index, group).
    Key(&'static str, &'static str, usize, i32),
    // A key-rebind row: (action label, setting, index, group). Like a Key
    // row but with a HitRegion that captures a new binding on click.
    Rebind(&'static str, SettingKey, usize, i32),
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
            let mut rows: Vec<BodyRow> = vec![BodyRow::Option(
                SettingKey::GraphicsQuality,
                "Graphics Quality",
                -1,
            )];
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
    pub(super) setting: SettingKey,
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
        "type": "OptionSelect",
        "args": {
            "$id": name,
            "setting": setting.as_str(),
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
        "type": "Slider",
        "args": {
            "$id": name,
            "setting": setting.as_str(),
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
