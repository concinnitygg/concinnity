// MainMenu expansion: Screen + Sprite (backdrop) + TextLabel +
// HitRegion per item + an optional Escape KeyBinding + an optional in-engine
// cursor Sprite, plus a generated settings sub-screen when an item asks for one.
//
// Everything is prefixed with the menu's own name so a generated element never
// collides with a hand-authored asset, and each element names the Screen it
// belongs to. The menu shows/hides at runtime purely as a Screen visibility
// flip; this pass adds no runtime behavior, only the assets the existing UI
// systems already drive.

mod rows;
pub(super) mod schema;
mod screen;
mod settings_tab;

#[cfg(test)]
mod tests;

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::overlay::UI_REFERENCE_SIZE;
use rows::settings_tabs;
use screen::{MenuMetrics, emit_menu_screen};
use settings_tab::emit_settings_tab;

use crate::authoring::spec::{AssetSpec, asset};
use crate::ui::expand::{Expand, ExpandContext};
use crate::world_builder::Diagnostic;
use concinnity_engine::components::{AuthoredAction, NamedAction};
use schema::{MainMenu, MenuItemAction};

// Top margin of a centered menu as a fraction of the reference height. The menu
// is top-aligned (not vertically centered) so the heading and tab bar hold a
// fixed position when switching between tabs with different row counts.
const TOP_MARGIN_FRAC: f32 = 0.07;

/// The settings tab a menu's `"settings"` item opens, and so the screen its
/// other tabs are reached from.
pub(crate) fn settings_entry_screen(menu_name: &str) -> String {
    format!("{menu_name}_settings_video")
}

/// The action an item fires: `"settings"` resolved to the settings screen the
/// menu generates.
pub(crate) fn item_action(menu_name: &str, action: &MenuItemAction) -> NamedAction {
    match action {
        MenuItemAction::Settings => AuthoredAction::Show(settings_entry_screen(menu_name).into()),
        MenuItemAction::Action(action) => action.clone(),
    }
}

/// Whether a menu generates a settings screen: an item opens one, or a Back
/// override is set for a caller that opens it by its own action (a story pause
/// menu, say).
pub(crate) fn generates_settings(menu: &MainMenu) -> bool {
    menu.items
        .iter()
        .any(|item| item.action == MenuItemAction::Settings)
        || menu.settings_back_action.is_some()
}

// An RGB accent lifted to an opaque RGBA fill: the active-tab underline marker
// and the scrollbar thumb draw the hover color at full alpha.
fn opaque(rgb: [f32; 3]) -> [f32; 4] {
    [rgb[0], rgb[1], rgb[2], 1.0]
}

// The in-engine cursor Sprite: a `follow_cursor` square the runtime tracks to
// the pointer while the menu is open.
fn cursor_sprite(name: &str, style: &MainMenu) -> AssetSpec {
    asset::sprite(
        name,
        [0.0, 0.0, style.cursor_size, style.cursor_size],
        style.cursor_color,
    )
    .set("follow_cursor", true)
}

impl Expand for MainMenu {
    fn expand(&self, name: &str, context: &ExpandContext) -> Result<Vec<AssetSpec>, Diagnostic> {
        // Menus lay out against a fixed reference canvas; the renderer uniformly
        // scales the overlay to the live window (see concinnity_engine::gfx::overlay),
        // so the declared Window size does not affect the menu layout.
        let (win_w, win_h) = (UI_REFERENCE_SIZE[0], UI_REFERENCE_SIZE[1]);
        let font_px = context.font_px(&self.font, self.font_px);
        Ok(expand_one(name, self, win_w, win_h, font_px))
    }
}

// Generate every asset for one menu: the main screen and its buttons, an optional
// Escape key binding, and (when an item resolves to the settings convenience) a
// settings sub-screen with a Back button.
fn expand_one(
    menu_name: &str,
    menu: &MainMenu,
    win_w: f32,
    win_h: f32,
    font_px: f32,
) -> Vec<AssetSpec> {
    let mut out = Vec::new();
    let wants_settings = generates_settings(menu);

    // Resolve the font. Use the user's font when set; otherwise emit a font for
    // this menu at `font_px` and reference it explicitly, because the row
    // geometry below is laid out in those pixels: leaving the labels font-less
    // would draw them at the built-in face's own size instead, inside buttons
    // measured for this one.
    let font_name = if menu.font.is_empty() {
        let name = format!("{menu_name}_font");
        out.push(asset::font(&name, u32::saturating_from(font_px)));
        name
    } else {
        menu.font.to_string()
    };

    // Resolve the per-menu convenience actions against this menu's name.
    let items: Vec<(String, NamedAction)> = menu
        .items
        .iter()
        .map(|item| (item.label.clone(), item_action(menu_name, &item.action)))
        .collect();

    out.extend(emit_menu_screen(
        menu_name,
        &menu.title,
        &items,
        menu,
        &font_name,
        MenuMetrics {
            win_w,
            win_h,
            font_px,
        },
        menu.initial,
    ));

    if !menu.toggle_key.is_empty() {
        out.push(
            AssetSpec::new(format!("{menu_name}_toggle"), "KeyBinding")
                .set("key", menu.toggle_key.as_str())
                .set("action", NamedAction::Toggle(menu_name.into())),
        );
    }

    if wants_settings {
        for (suffix, _) in settings_tabs(menu.settings_profile) {
            out.extend(emit_settings_tab(
                menu_name, suffix, menu, &font_name, win_w, win_h, font_px,
            ));
        }
    }

    out
}
