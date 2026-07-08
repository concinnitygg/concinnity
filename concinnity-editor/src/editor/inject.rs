// src/editor/inject.rs
//
// Runtime injection of the editor HUD elements into an already-compiled world.
// Runs between `App::load_blob` and `App::start`, so the injected components are
// indistinguishable from cooked ones -- and none of it is ever written back to
// the user's world.jsonl or blobs (the SAVE path serializes the authored entry
// list, not the live world). The elements are plain `Sprite` / `TextLabel`
// components at reserved ids; the editor's `DebugHook` tick drives them each
// frame (see `hud.rs`). No editor-specific component or system is involved, so
// nothing here reaches the shipped runtime.

use super::hud;
use crate::assets::{Sprite, TextAlign, TextLabel};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// Placeholder layout used only for frame 0; the tick re-anchors everything to
// the true window corner from the first frame's viewport. Sized to match the
// tick's geometry so the initial placement is already close.
const REF_W: f32 = 1280.0;
const SAVE_W: f32 = 88.0;
const ADD_W: f32 = 132.0;
const GAP: f32 = 8.0;

// Reuse an existing HUD label's font for the button text. Every rendering world
// carries an injected `hud_font` (via the engine-default DebugHud) unless it
// explicitly opted out of the default HUDs; falling back to `None` leaves the
// labels to the renderer's default font.
fn hud_font(world: &World) -> Option<AssetId> {
    world.query::<TextLabel>().find_map(|l| l.font)
}

// Inject the editor HUD: the SAVE + Add buttons and the (initially hidden)
// add-asset dropdown rows, all view-less window-space overlays. Injected once by
// the caller, before start.
pub(crate) fn editor_hud(world: &mut World) {
    let font = hud_font(world);

    let save_rect = [REF_W - SAVE_W, 0.0, SAVE_W, hud::BTN_H];
    let add_rect = [REF_W - SAVE_W - GAP - ADD_W, 0.0, ADD_W, hud::BTN_H];

    world.add_component(button_sprite(
        hud::SAVE_BUTTON,
        save_rect,
        [0.82, 0.14, 0.16, 1.0],
        true,
    ));
    world.add_component(button_sprite(
        hud::ADD_BUTTON,
        add_rect,
        [0.20, 0.34, 0.52, 1.0],
        true,
    ));
    world.add_component(centered_label(hud::SAVE_LABEL, "SAVE", save_rect, font));
    world.add_component(centered_label(hud::ADD_LABEL, "Add", add_rect, font));

    // One hidden row per offered type; the tick shows/positions/colours them
    // while the dropdown is open. Row labels carry the type name.
    for (i, ty) in hud::ADD_TYPES.iter().enumerate() {
        let rect = hud::dropdown_row_rect(REF_W, i);
        world.add_component(button_sprite(
            hud::dropdown_bg(i),
            rect,
            [0.14, 0.14, 0.17, 0.96],
            false,
        ));
        world.add_component(row_label(hud::dropdown_label(i), ty, rect, font));
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

fn row_label(id: AssetId, content: &str, rect: [f32; 4], font: Option<AssetId>) -> TextLabel {
    TextLabel {
        asset_id: id,
        font,
        content: content.to_string(),
        x: rect[0] + 12.0,
        y: rect[1] + 10.0,
        color: [0.9, 0.9, 0.92],
        align: TextAlign::Left,
        // Hidden until the dropdown opens; the tick flips visibility.
        visible: false,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Injection adds the two buttons + their labels plus one hidden row (bg +
    // label) per offered type, all at reserved ids as view-less overlays.
    #[test]
    fn injects_buttons_and_hidden_rows() {
        let mut world = World::new_empty();
        editor_hud(&mut world);

        let n = hud::ADD_TYPES.len();
        assert_eq!(world.query::<Sprite>().count(), 2 + n, "buttons + row bgs");
        assert_eq!(
            world.query::<TextLabel>().count(),
            2 + n,
            "labels + row labels"
        );

        // Buttons are visible; every dropdown row starts hidden.
        assert!(
            world
                .query::<Sprite>()
                .find(|s| s.asset_id == hud::SAVE_BUTTON)
                .unwrap()
                .visible
        );
        for i in 0..n {
            assert!(
                !world
                    .query::<Sprite>()
                    .find(|s| s.asset_id == hud::dropdown_bg(i))
                    .unwrap()
                    .visible,
                "row {i} bg hidden"
            );
        }

        // Row labels carry the offered type names.
        for (i, ty) in hud::ADD_TYPES.iter().enumerate() {
            let lbl = world
                .query::<TextLabel>()
                .find(|l| l.asset_id == hud::dropdown_label(i))
                .unwrap();
            assert_eq!(&lbl.content, ty);
        }

        // View-less: window space, never overlay-scaled.
        assert!(world.query::<Sprite>().all(|s| s.view.is_none()));
        assert!(world.query::<TextLabel>().all(|l| l.view.is_none()));
    }

    // The button + row text reuses whatever font an existing HUD label carries.
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
        let row0 = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == hud::dropdown_label(0))
            .unwrap();
        assert_eq!(row0.font, Some(AssetId(42)));
    }
}
