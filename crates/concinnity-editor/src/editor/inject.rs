// src/editor/inject.rs
//
// Runtime injection of the editor HUD elements into an already-compiled world.
// Runs between `App::load_blob` and `App::start`, so the injected components are
// indistinguishable from cooked ones -- and none of it is ever written back to
// the user's world.jsonl or blobs (the SAVE path serializes the authored entry
// list, not the live world). The elements are plain `Sprite` / `TextLabel` /
// `TextInput` components at reserved ids; the editor's `DebugHook` tick drives
// them each frame (see `hud.rs` / `panel.rs`). No editor-specific component or
// system is involved, so nothing here reaches the shipped runtime. (The two
// `TextInput` fields do bring in the engine's general text-input system, which
// is real runtime code, not editor-only.)

use super::{form_panel, hud, panel, preview, template_panel, templates, view};
use crate::assets::{Sprite, TextInput, TextLabel};
use crate::ecs::FontHandle;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;
use concinnity_templates::{AssetSpec, asset};

// Placeholder layout used only for frame 0; the tick re-anchors everything to
// the true window corner from the first frame's viewport. Sized to match the
// tick's geometry so the initial placement is already close.
const REF_W: f32 = 1280.0;
const SAVE_W: f32 = 88.0;
const VIEW_W: f32 = 132.0;
const GAP: f32 = 8.0;

// Reuse an existing HUD label's font for the button + field text. Every
// rendering world carries an injected `hud_font` (via the engine-default
// DebugHud) unless it explicitly opted out; falling back to `None` leaves the
// text to the renderer's default font.
fn hud_font(world: &World) -> Option<FontHandle> {
    world.query::<TextLabel>().find_map(|l| l.font)
}

// Inject the editor HUD: the floating Assets, edit-form, Preview, View, and
// Templates panels and the top bar (SAVE / View), all view-less window-space
// overlays. Injected once by the caller, before start. The overlay draws
// components in insertion order, so the panels go in FIRST and the top bar LAST:
// a panel dragged under the bar slides behind it, matching the hook's hit-test
// priority (top bar first). The edit form goes in after the Assets panel so it
// floats over the browse list it was opened from. (The hook also publishes a
// per-frame focus-stack draw layer that reorders overlapping panels; the
// injection order is the fallback when the layer map is empty.)
pub(crate) fn editor_hud(world: &mut World) {
    let font = hud_font(world);
    inject_panel(world, font);
    inject_form_panel(world, font);
    inject_preview(world, font);
    inject_view(world, font);
    inject_templates(world, font);
    inject_template_panel(world, font);
    inject_top_bar(world, font);
}

fn inject_top_bar(world: &mut World, font: Option<FontHandle>) {
    let save_rect = [REF_W - SAVE_W, 0.0, SAVE_W, hud::BTN_H];
    let view_rect = [save_rect[0] - GAP - VIEW_W, 0.0, VIEW_W, hud::BTN_H];

    world.add_component(button_sprite(
        hud::SAVE_BUTTON,
        save_rect,
        [0.82, 0.14, 0.16, 1.0],
        true,
    ));
    world.add_component(button_sprite(
        hud::VIEW_BUTTON,
        view_rect,
        [0.20, 0.34, 0.52, 1.0],
        true,
    ));
    world.add_component(centered_label(hud::SAVE_LABEL, "SAVE", save_rect, font));
    world.add_component(centered_label(hud::VIEW_LABEL, "View", view_rect, font));
}

// Inject the View panel's elements, hidden with placeholder geometry; the tick's
// `view::apply` positions and shows them when the panel is toggled on.
fn inject_view(world: &mut World, font: Option<FontHandle>) {
    let hidden = [0.0, 0.0, 0.0, 0.0];
    for id in view::all_sprite_ids() {
        world.add_component(button_sprite(id, hidden, [0.1, 0.1, 0.12, 1.0], false));
    }
    for id in view::all_label_ids() {
        world.add_component(row_label(id, "", hidden, font, false));
    }
}

// Inject the Templates panel's elements, hidden with placeholder geometry; the
// tick's `templates::apply` positions and shows them when the panel is toggled on.
fn inject_templates(world: &mut World, font: Option<FontHandle>) {
    let hidden = [0.0, 0.0, 0.0, 0.0];
    for id in templates::all_sprite_ids() {
        world.add_component(button_sprite(id, hidden, [0.1, 0.1, 0.12, 1.0], false));
    }
    for id in templates::all_label_ids() {
        world.add_component(row_label(id, "", hidden, font, false));
    }
}

// Inject the Template detail panel's elements, hidden with placeholder geometry;
// the tick's `template_panel::apply` positions and shows them once a template row
// is picked from the Templates list.
fn inject_template_panel(world: &mut World, font: Option<FontHandle>) {
    let hidden = [0.0, 0.0, 0.0, 0.0];
    for id in template_panel::all_sprite_ids() {
        world.add_component(button_sprite(id, hidden, [0.1, 0.1, 0.12, 1.0], false));
    }
    for id in template_panel::all_label_ids() {
        world.add_component(row_label(id, "", hidden, font, false));
    }
}

// Inject the Assets panel's elements, all starting hidden (the panel is closed
// at launch). The tick's `panel::apply` positions, tints, and shows only what the
// active mode needs each frame, so inject every element hidden with placeholder
// geometry / tint / content. Sourcing the ids from the panel's own id lists keeps
// this in lockstep with what the panel draws.
fn inject_panel(world: &mut World, font: Option<FontHandle>) {
    let hidden = [0.0, 0.0, 0.0, 0.0];
    for id in panel::all_sprite_ids() {
        world.add_component(button_sprite(id, hidden, [0.1, 0.1, 0.12, 1.0], false));
    }
    for id in panel::all_label_ids() {
        world.add_component(row_label(id, "", hidden, font, false));
    }
    // The combo's typed filter field (hidden; the panel shows + focuses it while
    // the combo is open).
    world.add_component(text_field(panel::FILTER_INPUT, "filter", font));
}

// Inject the edit-form panel's elements, all starting hidden (the form opens
// from a browse-list click or the "+" picker): its chrome + slot pools from its
// id lists, plus the name heading and the fixed pool of arg text inputs.
fn inject_form_panel(world: &mut World, font: Option<FontHandle>) {
    let hidden = [0.0, 0.0, 0.0, 0.0];
    for id in form_panel::all_sprite_ids() {
        world.add_component(button_sprite(id, hidden, [0.1, 0.1, 0.12, 1.0], false));
    }
    for id in form_panel::all_label_ids() {
        world.add_component(row_label(id, "", hidden, font, false));
    }
    world.add_component(text_field(form_panel::NAME_INPUT, "name", font));
    for j in 0..super::form::FIELD_POOL {
        world.add_component(text_field(form_panel::form_input(j), "", font));
    }
}

// Inject the Preview panel's elements, hidden with placeholder geometry; the
// tick's `preview::apply` positions and shows them from the first frame that has
// a `FrameInput`.
fn inject_preview(world: &mut World, font: Option<FontHandle>) {
    let hidden = [0.0, 0.0, 0.0, 0.0];
    for id in preview::all_sprite_ids() {
        world.add_component(button_sprite(id, hidden, [0.1, 0.1, 0.12, 1.0], false));
    }
    for id in preview::all_label_ids() {
        world.add_component(row_label(id, "", hidden, font, false));
    }
}

// Materialize a templates asset spec into a live component through the engine's
// own accept path (serde over the spec's args) -- the same conversion the cook
// pipeline runs on a world line. The reserved `asset_id` and the reused font are
// set by the caller afterward (neither is part of the spec's args).
fn materialize<T: serde::de::DeserializeOwned>(spec: AssetSpec) -> T {
    serde_json::from_value(crate::spec_args(&spec))
        .expect("editor HUD spec deserializes into its component")
}

fn button_sprite(id: AssetId, rect: [f32; 4], tint: [f32; 4], visible: bool) -> Sprite {
    let mut s: Sprite = materialize(asset::sprite("", rect, tint).set("visible", visible));
    s.asset_id = id;
    s
}

fn centered_label(
    id: AssetId,
    content: &str,
    rect: [f32; 4],
    font: Option<FontHandle>,
) -> TextLabel {
    let pos = [rect[0] + rect[2] * 0.5, rect[1] + hud::LABEL_TOP];
    let mut l: TextLabel = materialize(asset::text_label(
        "",
        content,
        pos,
        [1.0, 1.0, 1.0],
        "center",
    ));
    l.asset_id = id;
    l.font = font;
    l
}

fn row_label(
    id: AssetId,
    content: &str,
    rect: [f32; 4],
    font: Option<FontHandle>,
    visible: bool,
) -> TextLabel {
    let pos = [rect[0] + 12.0, rect[1] + 10.0];
    let mut l: TextLabel = materialize(
        asset::text_label("", content, pos, [0.9, 0.9, 0.92], "left").set("visible", visible),
    );
    l.asset_id = id;
    l.font = font;
    l
}

fn text_field(id: AssetId, placeholder: &str, font: Option<FontHandle>) -> TextInput {
    let mut t: TextInput = materialize(
        asset::text_input("", placeholder)
            .set("background", [0.14, 0.15, 0.20, 1.0])
            .set("max_len", 48u32)
            .set("visible", false),
    );
    t.asset_id = id;
    t.font = font;
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    // Injection adds the top-bar controls (visible) plus the Assets, Preview,
    // View, and Templates panels' chrome, row families, and the typed fields (all
    // hidden) -- every one a view-less overlay at a reserved id.
    #[test]
    fn injects_top_bar_and_hidden_panels() {
        let mut world = World::new_empty();
        editor_hud(&mut world);

        // Top-bar buttons are visible.
        for id in [hud::SAVE_BUTTON, hud::VIEW_BUTTON] {
            assert!(
                world
                    .query::<Sprite>()
                    .find(|s| s.asset_id == id)
                    .unwrap()
                    .visible,
                "{id:?} visible"
            );
        }

        // The View button is labelled "View".
        assert_eq!(
            world
                .query::<TextLabel>()
                .find(|l| l.asset_id == hud::VIEW_LABEL)
                .unwrap()
                .content,
            "View"
        );

        // Panel chrome + a representative row from each family start hidden, as do
        // the whole Preview, View, and Templates panels (the tick shows them on
        // demand).
        for id in [
            panel::PANEL_BG,
            panel::TITLE_BG,
            panel::PLUS_BG,
            panel::COMBO_BG,
            panel::MENU_BG,
            panel::list_row_bg(0),
            panel::combo_row_bg(0),
            preview::TITLE_BG,
            preview::ROW_BG,
            preview::CHECK_BOX,
            view::TITLE_BG,
            view::row_bg(0),
            view::check_box(0),
            templates::TITLE_BG,
            templates::row_bg(0),
            template_panel::PANEL_BG,
            template_panel::APPLY_BG,
            template_panel::row_bg(0),
        ] {
            assert!(
                !world
                    .query::<Sprite>()
                    .find(|s| s.asset_id == id)
                    .unwrap()
                    .visible,
                "{id:?} starts hidden"
            );
        }

        // Draw order is insertion order: the top bar goes in AFTER the panels so
        // a dragged panel slides behind it (matching the hook's hit-test order).
        let sprites: Vec<AssetId> = world.query::<Sprite>().map(|s| s.asset_id).collect();
        let pos = |id: AssetId| sprites.iter().position(|&x| x == id).unwrap();
        assert!(pos(panel::PANEL_BG) < pos(hud::SAVE_BUTTON));
        assert!(pos(preview::TITLE_BG) < pos(hud::SAVE_BUTTON));
        assert!(pos(view::TITLE_BG) < pos(hud::SAVE_BUTTON));
        assert!(pos(templates::TITLE_BG) < pos(hud::SAVE_BUTTON));
        assert!(pos(template_panel::PANEL_BG) < pos(hud::SAVE_BUTTON));

        // Both typed fields exist, hidden, and reference the reused font.
        let fields: Vec<AssetId> = world.query::<TextInput>().map(|t| t.asset_id).collect();
        assert!(fields.contains(&panel::FILTER_INPUT));
        assert!(fields.contains(&form_panel::NAME_INPUT));
        assert!(world.query::<TextInput>().all(|t| !t.visible));

        // View-less: window space, never overlay-scaled.
        assert!(world.query::<Sprite>().all(|s| s.view.is_none()));
        assert!(world.query::<TextLabel>().all(|l| l.view.is_none()));
        assert!(world.query::<TextInput>().all(|t| t.view.is_none()));
    }

    // The button + field text reuses whatever font an existing HUD label carries.
    #[test]
    fn reuses_an_existing_label_font() {
        let mut world = World::new_empty();
        world.add_component(TextLabel {
            asset_id: AssetId(7),
            font: Some(FontHandle(42)),
            ..Default::default()
        });
        editor_hud(&mut world);
        let save = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == hud::SAVE_LABEL)
            .unwrap();
        assert_eq!(save.font, Some(FontHandle(42)));
        let field = world
            .query::<TextInput>()
            .find(|t| t.asset_id == form_panel::NAME_INPUT)
            .unwrap();
        assert_eq!(field.font, Some(FontHandle(42)));
    }

    // The templates-spec-driven constructors materialize the same components the
    // hand-written struct literals used to, so routing them through templates did
    // not shift any field. Pins the spec builders against drift.
    #[test]
    fn constructors_materialize_expected_components() {
        use crate::assets::TextAlign;

        let s = button_sprite(
            AssetId(1),
            [10.0, 20.0, 100.0, 40.0],
            [1.0, 0.0, 0.0, 1.0],
            true,
        );
        assert_eq!((s.x, s.y, s.width, s.height), (10.0, 20.0, 100.0, 40.0));
        assert_eq!(s.tint, [1.0, 0.0, 0.0, 1.0]);
        assert!(s.visible);
        assert_eq!(s.asset_id, AssetId(1));

        let t = text_field(AssetId(2), "name", Some(FontHandle(9)));
        assert_eq!(t.placeholder, "name");
        assert_eq!(t.background, [0.14, 0.15, 0.20, 1.0]);
        assert_eq!(t.max_len, 48);
        assert!(!t.visible);
        assert_eq!(t.font, Some(FontHandle(9)));

        let c = centered_label(AssetId(3), "SAVE", [0.0, 0.0, 88.0, 88.0], None);
        assert_eq!(c.content, "SAVE");
        assert_eq!(c.align, TextAlign::Center);
        assert_eq!(c.x, 44.0, "centered on the button width");
        assert_eq!(c.color, [1.0, 1.0, 1.0]);
        assert!(c.visible);

        let r = row_label(AssetId(4), "row", [0.0, 0.0, 100.0, 40.0], None, false);
        assert_eq!(r.align, TextAlign::Left);
        assert_eq!(r.color, [0.9, 0.9, 0.92]);
        assert!(!r.visible);
    }
}
