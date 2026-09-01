// src/editor/worlds/loading.rs
//
// The start screen's loading cover: the black field and caption that stand
// over the preview area while the world the sidebar picked is compiled. The
// editor's own window comes up first and the listing with it, so the wait is
// spent on a screen the user can already read and click, and only the render
// side of it is covered.
//
// Styled after the engine's own scene-loading overlay
// (`concinnity-core/src/defaults/loading.rs`): a black field with the word
// over it. There is no progress bar, because there is no progress to report --
// the compile is one call, and a bar frozen mid-sweep reads as a hang.

use super::super::registry::ID_BASE;
use super::super::{theme, widget};
use crate::components::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// Reserved id family: the next free block after the shot fade's (0xD000).
const BASE: u32 = ID_BASE + 0xE000;
pub(crate) const COVER: AssetId = AssetId(BASE);
pub(crate) const CAPTION: AssetId = AssetId(BASE + 1);

// The same black the shots hand over through, so the cover gives way to the
// opening fade without a step in between.
const COVER_TINT: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const CAPTION_SCALE: f32 = 1.0;

// The caption a world's name earns, or the bare word while nothing is named.
pub(crate) fn caption(name: Option<&str>) -> String {
    match name {
        Some(name) => format!("Loading {name}"),
        None => "Loading".to_string(),
    }
}

// Cover `area` (the window the sidebar does not stand on) and caption it.
pub(crate) fn apply(world: &mut World, area: [f32; 4], name: Option<&str>) {
    widget::place_sprite(world, COVER, area, COVER_TINT, true);
    if let Some(label) = widget::label_mut(world, CAPTION) {
        label.x = area[0] + area[2] * 0.5;
        label.y = area[1] + area[3] * 0.5 - 10.0 * CAPTION_SCALE;
        label.align = TextAlign::Center;
        label.color = theme::LABEL_DIM;
        label.scale = CAPTION_SCALE;
        label.visible = true;
        label.content = caption(name);
    }
}

pub(crate) fn hide(world: &mut World) {
    widget::set_sprite_visible(world, COVER, false);
    widget::set_label_visible(world, CAPTION, false);
}

pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    vec![COVER]
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    vec![CAPTION]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Sprite, TextLabel};

    fn injected_world() -> World {
        let mut world = World::new();
        world.add_component(Sprite {
            asset_id: COVER,
            ..Default::default()
        });
        world.add_component(TextLabel {
            asset_id: CAPTION,
            ..Default::default()
        });
        world
    }

    #[test]
    fn the_cover_fills_its_area_and_names_what_is_loading() {
        let mut world = injected_world();
        apply(&mut world, [280.0, 0.0, 1000.0, 720.0], Some("bistro"));

        let cover = world.query::<Sprite>().next().expect("the cover");
        assert!(cover.visible);
        assert_eq!((cover.x, cover.y), (280.0, 0.0));
        assert_eq!((cover.width, cover.height), (1000.0, 720.0));
        assert_eq!(
            cover.tint, COVER_TINT,
            "opaque, so nothing half-built shows"
        );

        let label = world.query::<TextLabel>().next().expect("the caption");
        assert!(label.visible);
        assert_eq!(label.content, "Loading bistro");
        assert_eq!(label.align, TextAlign::Center);
        assert_eq!(
            label.x, 780.0,
            "centred in the covered area, not the window"
        );

        hide(&mut world);
        assert!(!world.query::<Sprite>().next().unwrap().visible);
        assert!(!world.query::<TextLabel>().next().unwrap().visible);
    }

    #[test]
    fn an_unnamed_world_still_says_what_is_happening() {
        assert_eq!(caption(None), "Loading");
    }
}
