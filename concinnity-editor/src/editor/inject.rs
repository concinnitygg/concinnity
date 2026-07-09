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
use crate::assets::{Sprite, TextAlign, TextInput, TextLabel};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

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
    for j in 0..super::form::MAX_FIELDS {
        world.add_component(text_field(panel::form_input(j), "", font));
    }
}

fn button_sprite(id: AssetId, rect: [f32; 4], tint: [f32; 4], visible: bool) -> Sprite {
    Sprite {
        asset_id: id,
        x: rect[0],
        y: rect[1],
        width: rect[2],
        height: rect[3],
        tint,
        visible,
        ..Default::default()
    }
}

fn centered_label(id: AssetId, content: &str, rect: [f32; 4], font: Option<AssetId>) -> TextLabel {
    TextLabel {
        asset_id: id,
        font,
        content: content.to_string(),
        x: rect[0] + rect[2] * 0.5,
        y: rect[1] + hud::LABEL_TOP,
        color: [1.0, 1.0, 1.0],
        align: TextAlign::Center,
        visible: true,
        ..Default::default()
    }
}

fn row_label(
    id: AssetId,
    content: &str,
    rect: [f32; 4],
    font: Option<AssetId>,
    visible: bool,
) -> TextLabel {
    TextLabel {
        asset_id: id,
        font,
        content: content.to_string(),
        x: rect[0] + 12.0,
        y: rect[1] + 10.0,
        color: [0.9, 0.9, 0.92],
        align: TextAlign::Left,
        visible,
        ..Default::default()
    }
}

fn text_field(id: AssetId, placeholder: &str, font: Option<AssetId>) -> TextInput {
    TextInput {
        asset_id: id,
        font,
        placeholder: placeholder.to_string(),
        background: [0.14, 0.15, 0.20, 1.0],
        max_len: 48,
        visible: false,
        ..Default::default()
    }
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
}
