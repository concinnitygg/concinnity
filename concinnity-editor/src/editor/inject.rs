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

use super::{hud, panel};
use crate::assets::{Sprite, TextInput, TextLabel};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;
use concinnity_templates::{AssetSpec, asset};

// Placeholder layout used only for frame 0; the tick re-anchors everything to
// the true window corner from the first frame's viewport. Sized to match the
// tick's geometry so the initial placement is already close.
const REF_W: f32 = 1280.0;
const SAVE_W: f32 = 88.0;
const ASSETS_W: f32 = 132.0;
const TPL_W: f32 = 132.0;
const GAP: f32 = 8.0;

// Reuse an existing HUD label's font for the button + field text. Every
// rendering world carries an injected `hud_font` (via the engine-default
// DebugHud) unless it explicitly opted out; falling back to `None` leaves the
// text to the renderer's default font.
fn hud_font(world: &World) -> Option<AssetId> {
    world.query::<TextLabel>().find_map(|l| l.font)
}

// Inject the editor HUD: the top bar (SAVE / Assets / Templates + capture
// checkbox + Templates dropdown rows) and the browse-and-add panel (its chrome,
// row families, and the two typed fields), all view-less window-space overlays.
// Injected once by the caller, before start.
pub(crate) fn editor_hud(world: &mut World) {
    let font = hud_font(world);
    inject_top_bar(world, font);
    inject_panel(world, font);
}

fn inject_top_bar(world: &mut World, font: Option<AssetId>) {
    let save_rect = [REF_W - SAVE_W, 0.0, SAVE_W, hud::BTN_H];
    let assets_rect = [save_rect[0] - GAP - ASSETS_W, 0.0, ASSETS_W, hud::BTN_H];
    let tpl_rect = [assets_rect[0] - GAP - TPL_W, 0.0, TPL_W, hud::BTN_H];

    world.add_component(button_sprite(
        hud::SAVE_BUTTON,
        save_rect,
        [0.82, 0.14, 0.16, 1.0],
        true,
    ));
    world.add_component(button_sprite(
        hud::ASSETS_BUTTON,
        assets_rect,
        [0.20, 0.34, 0.52, 1.0],
        true,
    ));
    world.add_component(button_sprite(
        hud::TPL_BUTTON,
        tpl_rect,
        [0.28, 0.24, 0.44, 1.0],
        true,
    ));
    world.add_component(centered_label(hud::SAVE_LABEL, "SAVE", save_rect, font));
    world.add_component(centered_label(
        hud::ASSETS_LABEL,
        "Assets",
        assets_rect,
        font,
    ));
    world.add_component(centered_label(hud::TPL_LABEL, "Templates", tpl_rect, font));

    // Capture checkbox: a row background, the box indicator, and its label.
    let check = hud::checkbox_rect(REF_W);
    world.add_component(button_sprite(
        hud::CHECK_BG,
        check,
        [0.12, 0.12, 0.15, 0.92],
        true,
    ));
    world.add_component(button_sprite(
        hud::CHECK_BOX,
        [check[0] + 8.0, check[1] + 7.0, 20.0, 20.0],
        [0.30, 0.30, 0.34, 1.0],
        true,
    ));
    world.add_component(row_label(
        hud::CHECK_LABEL,
        "Capture mouse",
        [check[0] + 24.0, check[1], check[2], check[3]],
        font,
        true,
    ));

    // Templates dropdown rows (hidden; the tick shows / labels them on open).
    for i in 0..hud::max_rows() {
        let rect = hud::dropdown_row_rect(REF_W, i);
        world.add_component(button_sprite(
            hud::dropdown_bg(i),
            rect,
            [0.14, 0.14, 0.17, 0.96],
            false,
        ));
        world.add_component(row_label(hud::dropdown_label(i), "", rect, font, false));
    }
}

// Inject the Assets panel's elements, all starting hidden (the panel is closed
// at launch). The tick's `panel::apply` positions, tints, and shows only what the
// active mode needs each frame, so inject every element hidden with placeholder
// geometry / tint / content. Sourcing the ids from the panel's own id lists keeps
// this in lockstep with what the panel draws.
fn inject_panel(world: &mut World, font: Option<AssetId>) {
    let hidden = [0.0, 0.0, 0.0, 0.0];
    for id in panel::all_sprite_ids() {
        world.add_component(button_sprite(id, hidden, [0.1, 0.1, 0.12, 1.0], false));
    }
    for id in panel::all_label_ids() {
        world.add_component(row_label(id, "", hidden, font, false));
    }
    // The typed fields (hidden; the panel shows + focuses them by mode): the
    // combo's filter field, the form's name field, and the add / edit form's
    // fixed pool of arg text inputs.
    world.add_component(text_field(panel::FILTER_INPUT, "filter", font));
    world.add_component(text_field(panel::NAME_INPUT, "name", font));
    for j in 0..super::form::FIELD_POOL {
        world.add_component(text_field(panel::form_input(j), "", font));
    }
}

// Materialize a templates asset spec into a live component through the engine's
// own accept path (serde over the spec's args) -- the same conversion the cook
// pipeline runs on a world line. The reserved `asset_id` and the reused font are
// set by the caller afterward (neither is part of the spec's args).
fn materialize<T: serde::de::DeserializeOwned>(spec: AssetSpec) -> T {
    serde_json::from_value(concinnity_app::spec_args(&spec))
        .expect("editor HUD spec deserializes into its component")
}

fn button_sprite(id: AssetId, rect: [f32; 4], tint: [f32; 4], visible: bool) -> Sprite {
    let mut s: Sprite = materialize(asset::sprite("", rect, tint).set("visible", visible));
    s.asset_id = id;
    s
}

fn centered_label(id: AssetId, content: &str, rect: [f32; 4], font: Option<AssetId>) -> TextLabel {
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
    font: Option<AssetId>,
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

fn text_field(id: AssetId, placeholder: &str, font: Option<AssetId>) -> TextInput {
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

    // Injection adds the top-bar controls (visible) plus the panel chrome, row
    // families, and the two typed fields (all hidden) -- every one a view-less
    // overlay at a reserved id.
    #[test]
    fn injects_top_bar_and_hidden_panel() {
        let mut world = World::new_empty();
        editor_hud(&mut world);

        // Top-bar buttons + checkbox are visible.
        for id in [
            hud::SAVE_BUTTON,
            hud::ASSETS_BUTTON,
            hud::TPL_BUTTON,
            hud::CHECK_BG,
            hud::CHECK_BOX,
        ] {
            assert!(
                world
                    .query::<Sprite>()
                    .find(|s| s.asset_id == id)
                    .unwrap()
                    .visible,
                "{id:?} visible"
            );
        }

        // The Assets button is labelled "Assets".
        assert_eq!(
            world
                .query::<TextLabel>()
                .find(|l| l.asset_id == hud::ASSETS_LABEL)
                .unwrap()
                .content,
            "Assets"
        );

        // Panel chrome + a representative row from each family start hidden.
        for id in [
            panel::PANEL_BG,
            panel::PLUS_BG,
            panel::COMBO_BG,
            panel::MENU_BG,
            panel::list_row_bg(0),
            panel::combo_row_bg(0),
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

        // Both typed fields exist, hidden, and reference the reused font.
        let fields: Vec<AssetId> = world.query::<TextInput>().map(|t| t.asset_id).collect();
        assert!(fields.contains(&panel::FILTER_INPUT));
        assert!(fields.contains(&panel::NAME_INPUT));
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
            font: Some(AssetId(42)),
            ..Default::default()
        });
        editor_hud(&mut world);
        let save = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == hud::SAVE_LABEL)
            .unwrap();
        assert_eq!(save.font, Some(AssetId(42)));
        let field = world
            .query::<TextInput>()
            .find(|t| t.asset_id == panel::NAME_INPUT)
            .unwrap();
        assert_eq!(field.font, Some(AssetId(42)));
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

        let t = text_field(AssetId(2), "name", Some(AssetId(9)));
        assert_eq!(t.placeholder, "name");
        assert_eq!(t.background, [0.14, 0.15, 0.20, 1.0]);
        assert_eq!(t.max_len, 48);
        assert!(!t.visible);
        assert_eq!(t.font, Some(AssetId(9)));

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
