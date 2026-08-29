//! One-line main-menu schema.

use std::vec;

/// A ready-made menu declared in a single line.
///
/// `MainMenu` is a build-time shorthand. It expands into the assets a menu is
/// built from: a [Screen](#screen) layer, a dim backdrop [Sprite](#sprite), a
/// [TextLabel](#textlabel) and [HitRegion](#hitregion) for each item, an
/// optional [KeyBinding](#keybinding) that toggles the menu, and an optional
/// in-engine mouse cursor [Sprite](#sprite). So `world.jsonl` stays small.
///
/// The bare form gives a centered Return / Settings / Quit menu that starts
/// closed, with Escape opening it, so the scene itself shows first. Set
/// `"initial": true` to show the menu as soon as the world loads:
///
/// Declaring a `MainMenu` also injects the [StatHud](#stathud) (and its chip
/// labels) at build time when the world declares none, so the menu's
/// performance-stats toggles have chips to drive.
///
/// **Items.** Each item has a `label` (the text) and an `action` fired on
/// click. `action` takes the same vocabulary as [HitRegion](#hitregion)
/// (`"scene:<name>"`, `"quit"`, `"screen:show:<name>"`, `"screen:hide"`,
/// `"screen:toggle:<name>"`) plus two conveniences resolved against this menu:
/// - `"return"`: hide this menu (the same as `"screen:hide"`).
/// - `"settings"`: open a generated settings sub-menu that has a Back button.
///
/// **Generated names** are prefixed with the menu's `name` (`<name>_btn_0`,
/// `<name>_label_0`, `<name>_cursor`, ...), so they never clash with
/// hand-authored assets and you never reference them by hand.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MainMenu {
    /// Menu entries, top to bottom. Each one is a clickable button.
    pub items: Vec<MainMenuItem>,
    /// Optional heading drawn above the items. Empty draws no heading.
    pub title: String,
    /// Show the menu as soon as the world loads. Off by default: the scene
    /// shows first and the toggle key opens the menu.
    pub initial: bool,
    /// InputKey that toggles the menu while the cursor is free. Empty binds no key.
    /// Only `"Escape"` is currently recognised by the runtime.
    pub toggle_key: String,
    /// RGBA fill drawn across the whole window behind the items. Defaults to
    /// opaque black: a fully opaque alpha (1.0) hides the scene completely, which
    /// lets the renderer skip the entire world render while the menu is open, so
    /// the frame costs only the menu overlay. Lower the alpha to keep the world
    /// visible behind a translucent fade (the world then keeps rendering); an
    /// alpha of 0 draws no backdrop at all.
    pub dim: [f32; 4],
    /// Horizontally center the menu and align it to the top of the window.
    /// When false, `x` is the column's center and `y` is the top of the first
    /// item.
    ///
    /// The menu is a screen overlay laid out against a fixed reference
    /// resolution and uniformly scaled to fill the window, so it keeps the same
    /// proportions at any window size. All pixel fields below are in that
    /// reference space, not raw window pixels.
    pub centered: bool,
    /// Column center x in reference-space pixels, used when `centered` is false.
    pub x: f32,
    /// Top of the first item in reference-space pixels, used when `centered` is
    /// false.
    pub y: f32,
    /// Width of each item's clickable region in pixels.
    pub button_width: f32,
    /// Height of each item's clickable region in pixels.
    pub button_height: f32,
    /// Pixels between adjacent items.
    pub row_gap: f32,
    /// [Font](#font) for the item text. Empty uses the built-in font.
    pub font: String,
    /// Pixel size of the item text when this menu emits its own built-in font
    /// (that is, when `font` is empty). Ignored when `font` names a
    /// [Font](#font), which carries its own size. In reference-space pixels.
    pub font_px: f32,
    /// Linear-space RGB color of the item text.
    pub text_color: [f32; 3],
    /// Scale applied to the item text.
    pub text_scale: f32,
    /// RGB color of an item's text while it is hovered.
    pub hover_color: [f32; 3],
    /// Multiplier applied to an item's text size while it is hovered. The
    /// default `1.0` keeps the size and position fixed, so only the color
    /// changes on hover; a value like `1.1` grows the hovered text by 10%.
    pub hover_scale: f32,
    /// Draw an in-engine arrow cursor while the menu is shown (the system
    /// cursor is hidden). When false the system cursor is used.
    pub cursor: bool,
    /// RGBA fill color of the arrow cursor. A contrasting outline is added
    /// automatically so it stays legible over any scene.
    pub cursor_color: [f32; 4],
    /// Arrow cursor height in pixels (its width follows the arrow's shape).
    pub cursor_size: f32,
    /// Which settings screen the `"settings"` item generates. `full` is the
    /// complete Video / Audio / Controls set a 3D world configures; `minimal`
    /// is the trimmed Video (window mode, resolution, vsync, frame rate) and
    /// Audio (volume) set that fits a world with nothing to render into (a
    /// visual-novel story, say), dropping the Controls tab and every
    /// scene-render group.
    pub settings_profile: SettingsProfile,
    /// Action fired by the settings screen's Back button, overriding the
    /// default (which returns to this menu). Setting it also generates the
    /// settings screen even when no item uses the `"settings"` convenience, so
    /// a caller that opens settings by its own action (a story, say) still gets
    /// the screen. Empty keeps the default Back-to-menu behavior.
    pub settings_back_action: String,
}

/// Which settings screen a [MainMenu](#mainmenu)'s `"settings"` item builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SettingsProfile {
    /// The complete Video / Audio / Controls settings, with the graphics
    /// quality preset and the Quality / Advanced render-feature groups.
    #[default]
    Full,
    /// A trimmed Video tab (window mode, resolution, vsync, frame rate) and an
    /// Audio tab (volume) only: no Controls tab, no graphics quality preset,
    /// and no scene-render groups. Suits a world that renders no 3D scene.
    Minimal,
}

/// One entry in a [MainMenu](#mainmenu).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MainMenuItem {
    /// Button text.
    pub label: String,
    /// Action fired on click. See [MainMenu](#mainmenu) for the vocabulary.
    pub action: String,
}

impl Default for MainMenu {
    fn default() -> Self {
        Self {
            items: vec![
                MainMenuItem {
                    label: "Return".to_string(),
                    action: "return".to_string(),
                },
                MainMenuItem {
                    label: "Settings".to_string(),
                    action: "settings".to_string(),
                },
                MainMenuItem {
                    label: "Quit".to_string(),
                    action: "quit".to_string(),
                },
            ],
            title: String::new(),
            initial: false,
            toggle_key: "Escape".to_string(),
            dim: [0.0, 0.0, 0.0, 1.0],
            centered: true,
            x: 640.0,
            y: 300.0,
            button_width: 360.0,
            button_height: 60.0,
            row_gap: 14.0,
            font: String::new(),
            font_px: 48.0,
            text_color: [0.85, 0.85, 0.85],
            text_scale: 1.1,
            hover_color: [1.0, 0.85, 0.3],
            hover_scale: 1.0,
            cursor: true,
            cursor_color: [1.0, 1.0, 1.0, 1.0],
            cursor_size: 22.0,
            settings_profile: SettingsProfile::Full,
            settings_back_action: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_menu_can_resume_configure_and_quit() {
        // The engine never injects a MainMenu, so a world that declares one with
        // no items still gets a usable pause menu out of the three defaults.
        let m = MainMenu::default();
        let actions: Vec<&str> = m.items.iter().map(|i| i.action.as_str()).collect();
        assert_eq!(actions, ["return", "settings", "quit"]);
        assert_eq!(m.items[0].label, "Return");
        assert_eq!(m.toggle_key, "Escape");
        assert_eq!(m.settings_profile, SettingsProfile::Full);
        // A pause menu is opened by its key, not shown at startup.
        assert!(!m.initial);
        assert!(m.centered);
        assert!(m.cursor);
    }

    #[test]
    fn an_authored_item_list_replaces_the_defaults_wholesale() {
        let m: MainMenu = serde_json::from_str(
            r#"{"title":"Ash","initial":true,"items":[{"label":"Play","action":"start"}],
                "settings_profile":"minimal","toggle_key":"Tab"}"#,
        )
        .unwrap();
        assert_eq!(m.items.len(), 1);
        assert_eq!(m.items[0].label, "Play");
        assert_eq!(m.items[0].action, "start");
        assert_eq!(m.title, "Ash");
        assert!(m.initial);
        assert_eq!(m.settings_profile, SettingsProfile::Minimal);
        assert_eq!(m.toggle_key, "Tab");
        // Layout the args did not mention keeps the schema defaults.
        assert_eq!((m.x, m.y), (640.0, 300.0));
        assert_eq!(m.button_width, 360.0);
    }

    #[test]
    fn a_blank_item_carries_neither_label_nor_action() {
        let item = MainMenuItem::default();
        assert!(item.label.is_empty());
        assert!(item.action.is_empty());
    }

    #[test]
    fn settings_profile_names_parse_in_lowercase() {
        assert_eq!(SettingsProfile::default(), SettingsProfile::Full);
        assert_eq!(
            serde_json::from_str::<SettingsProfile>(r#""full""#).unwrap(),
            SettingsProfile::Full
        );
        assert_eq!(
            serde_json::to_string(&SettingsProfile::Minimal).unwrap(),
            r#""minimal""#
        );
    }

    #[test]
    fn an_authored_menu_round_trips_through_postcard() {
        let m: MainMenu = serde_json::from_str(
            r#"{"items":[{"label":"Play","action":"start"},{"label":"Quit","action":"quit"}],
                "dim":[0,0,0,0.7],"font":"body","font_px":32,"hover_scale":1.2,
                "settings_back_action":"pause"}"#,
        )
        .unwrap();
        let bytes = postcard::to_allocvec(&m).unwrap();
        let back: MainMenu = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.items.len(), 2);
        assert_eq!(back.items[1].action, "quit");
        assert_eq!(back.dim, [0.0, 0.0, 0.0, 0.7]);
        assert_eq!(back.font, "body");
        assert_eq!(back.font_px, 32.0);
        assert_eq!(back.hover_scale, 1.2);
        assert_eq!(back.settings_back_action, "pause");
    }
}
