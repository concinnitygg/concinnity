// src/editor/widget.rs
//
// Small shared helpers for the editor HUD's injected overlay elements. The whole
// HUD (top bar in `hud.rs`, Assets panel in `panel.rs`) is built from plain
// Sprite / TextLabel / TextInput components at reserved ids, driven each frame by
// the editor hook. Both modules repeatedly need to look up one element by its
// reserved id and mutate it; these keep that lookup -- and the identical
// place-a-sprite / point-in-rect logic -- in one place instead of two copies.

use super::theme;
use crate::assets::{Sprite, TextAlign, TextInput, TextLabel};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// Shared chrome for the floating editor panels (Assets, Preview): each has a
// draggable title bar of this height across its top. The bar shares the panel's
// background (no separate strip), so it reads as one unified surface.
pub(crate) const TITLE_H: f32 = 28.0;

// The title-bar close ("X") button: invisible until hovered, when a rounded
// square appears behind the always-drawn glyph. Shared so every floating
// panel's close button looks identical.
const CLOSE_TINT_HOVER: [f32; 4] = [0.46, 0.26, 0.28, 1.0];
const CLOSE_GLYPH_COLOR: [f32; 3] = [0.88, 0.89, 0.92];
// The hover square is inset from the TITLE_H bounding box so the pill hugs the
// glyph instead of filling the bar.
const CLOSE_INSET: f32 = 4.0;

// Draw a panel's rounded background surface (the unified chrome tint) framed by
// the shared hairline border.
pub(crate) fn place_panel(world: &mut World, id: AssetId, rect: [f32; 4]) {
    if let Some(s) = sprite_mut(world, id) {
        s.x = rect[0];
        s.y = rect[1];
        s.width = rect[2];
        s.height = rect[3];
        s.tint = theme::CHROME_TINT;
        s.corner_radius = theme::PANEL_RADIUS;
        s.border_width = theme::PANEL_BORDER_WIDTH;
        s.border_color = theme::PANEL_BORDER_TINT;
        s.visible = true;
    }
}

// Draw a panel's title-bar heading, left-aligned and vertically centered. The
// bar itself is the panel background showing through.
pub(crate) fn place_heading(world: &mut World, label: AssetId, rect: [f32; 4], heading: &str) {
    if let Some(l) = label_mut(world, label) {
        l.x = rect[0] + 10.0;
        l.y = rect[1] + rect[3] * 0.5 - theme::TEXT_HALF;
        l.align = TextAlign::Left;
        l.color = theme::HEADING;
        l.visible = true;
        l.content = heading.to_string();
    }
}

// The square close-button rect at the right end of a `title` bar rect.
pub(crate) fn close_rect(title: [f32; 4]) -> [f32; 4] {
    [title[0] + title[2] - TITLE_H, title[1], TITLE_H, TITLE_H]
}

// Draw a panel's title-bar close button at the right end of `title`: nothing
// but the glyph until `hovered`, when a rounded square appears behind it.
pub(crate) fn place_close(
    world: &mut World,
    bg: AssetId,
    label: AssetId,
    title: [f32; 4],
    hovered: bool,
) {
    let rect = close_rect(title);
    if hovered {
        let pill = [
            rect[0] + CLOSE_INSET,
            rect[1] + CLOSE_INSET,
            rect[2] - 2.0 * CLOSE_INSET,
            rect[3] - 2.0 * CLOSE_INSET,
        ];
        place_rounded(
            world,
            bg,
            pill,
            CLOSE_TINT_HOVER,
            theme::CONTROL_RADIUS,
            true,
        );
    } else {
        set_sprite_visible(world, bg, false);
    }
    if let Some(l) = label_mut(world, label) {
        l.x = rect[0] + rect[2] * 0.5;
        l.y = rect[1] + rect[3] * 0.5 - theme::TEXT_HALF;
        l.align = TextAlign::Center;
        l.color = if hovered {
            [1.0, 1.0, 1.0]
        } else {
            CLOSE_GLYPH_COLOR
        };
        l.visible = true;
        l.content = "X".to_string();
    }
}

// Clamp a dragged panel origin so the whole `size` panel stays on screen: a hard
// stop at every window edge. A panel larger than the window pins to the top-left.
pub(crate) fn clamp_origin(pos: [f32; 2], size: [f32; 2], viewport: [f32; 2]) -> [f32; 2] {
    [
        pos[0].clamp(0.0, (viewport[0] - size[0]).max(0.0)),
        pos[1].clamp(0.0, (viewport[1] - size[1]).max(0.0)),
    ]
}

// Find the reserved-id Sprite / TextLabel / TextInput to mutate (or read), or
// `None` if it was not injected -- a caller then simply no-ops, which is how a
// hidden / absent element is handled.
pub(crate) fn sprite_mut(world: &mut World, id: AssetId) -> Option<&mut Sprite> {
    world.query_mut::<Sprite>().find(|s| s.asset_id == id)
}
pub(crate) fn label_mut(world: &mut World, id: AssetId) -> Option<&mut TextLabel> {
    world.query_mut::<TextLabel>().find(|l| l.asset_id == id)
}
pub(crate) fn input_mut(world: &mut World, id: AssetId) -> Option<&mut TextInput> {
    world.query_mut::<TextInput>().find(|t| t.asset_id == id)
}
pub(crate) fn input(world: &World, id: AssetId) -> Option<&TextInput> {
    world.query::<TextInput>().find(|t| t.asset_id == id)
}

// Move + resize the Sprite with `id` to `rect` ([x, y, w, h]), set its tint +
// visibility. A no-op if the sprite is absent.
pub(crate) fn place_sprite(
    world: &mut World,
    id: AssetId,
    rect: [f32; 4],
    tint: [f32; 4],
    visible: bool,
) {
    place_rounded(world, id, rect, tint, 0.0, visible);
}

// `place_sprite` with rounded corners of the given `radius`.
pub(crate) fn place_rounded(
    world: &mut World,
    id: AssetId,
    rect: [f32; 4],
    tint: [f32; 4],
    radius: f32,
    visible: bool,
) {
    if let Some(s) = sprite_mut(world, id) {
        s.x = rect[0];
        s.y = rect[1];
        s.width = rect[2];
        s.height = rect[3];
        s.tint = tint;
        s.corner_radius = radius;
        s.border_width = 0.0;
        s.visible = visible;
    }
}

pub(crate) fn set_sprite_visible(world: &mut World, id: AssetId, visible: bool) {
    if let Some(s) = sprite_mut(world, id) {
        s.visible = visible;
    }
}
pub(crate) fn set_label_visible(world: &mut World, id: AssetId, visible: bool) {
    if let Some(l) = label_mut(world, id) {
        l.visible = visible;
    }
}

// Position + show a typed field, setting its focus explicitly. The focused field
// re-asserts `focused = true` every frame: the click that opened the mode (on a
// button, say) lands outside the field, so the engine's text-input system would
// otherwise blur it that frame; re-asserting keeps it typable. Non-focused
// fields are shown with `focused = false` so several fields can coexist without
// fighting for the keyboard. The content is not touched here (only on a
// transition), so what is typed stands.
pub(crate) fn show_field(world: &mut World, id: AssetId, rect: [f32; 4], focused: bool) {
    if let Some(t) = input_mut(world, id) {
        t.x = rect[0];
        t.y = rect[1];
        t.width = rect[2];
        t.height = rect[3];
        t.visible = true;
        t.focused = focused;
    }
}

// Hide + blur a typed field.
pub(crate) fn hide_field(world: &mut World, id: AssetId) {
    if let Some(t) = input_mut(world, id) {
        t.visible = false;
        t.focused = false;
    }
}

// Set a field's text + caret and give it focus (a mode transition; the hook
// calls this so the field is ready to type into immediately).
pub(crate) fn focus_field_with(world: &mut World, id: AssetId, content: &str) {
    if let Some(t) = input_mut(world, id) {
        t.content = content.to_string();
        t.caret = content.chars().count();
        t.focused = true;
        t.visible = true;
    }
}

// Seed a field's text + caret without changing focus (the layout decides focus
// each frame). Used to pre-fill form inputs on open / window scroll.
pub(crate) fn seed_field(world: &mut World, id: AssetId, content: &str) {
    if let Some(t) = input_mut(world, id) {
        t.content = content.to_string();
        t.caret = content.chars().count();
    }
}

// Read a field's current text.
pub(crate) fn field_text(world: &World, id: AssetId) -> String {
    input(world, id)
        .map(|t| t.content.clone())
        .unwrap_or_default()
}

// Whether `(x, y)` lies inside `rect` ([x, y, w, h], top-left origin). The pure
// geometry lives in the shared templates crate; re-exported here so the HUD and
// panel keep using `widget::point_in`.
pub(crate) use concinnity_templates::ui::point_in;

// A label's display text clipped to `max` characters with an ellipsis, for
// panels that show free-length strings (file paths, story lines) in
// fixed-width rows. Editable text scrolls in its TextInput instead.
pub(crate) fn clip_text(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let clipped: String = text.chars().take(max.saturating_sub(3)).collect();
    format!("{clipped}...")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world_with(ids: &[AssetId]) -> World {
        let mut world = World::new_empty();
        for &id in ids {
            world.add_component(Sprite {
                asset_id: id,
                ..Default::default()
            });
            world.add_component(TextLabel {
                asset_id: id,
                ..Default::default()
            });
            world.add_component(TextInput {
                asset_id: id,
                ..Default::default()
            });
        }
        world
    }

    #[test]
    fn clip_text_shortens_long_strings() {
        assert_eq!(clip_text("short", 10), "short");
        assert_eq!(clip_text("a very long line indeed", 10), "a very ...");
    }

    #[test]
    fn accessors_find_by_id_or_return_none() {
        let mut world = world_with(&[AssetId(1)]);
        assert!(sprite_mut(&mut world, AssetId(1)).is_some());
        assert!(label_mut(&mut world, AssetId(1)).is_some());
        assert!(input_mut(&mut world, AssetId(1)).is_some());
        assert!(input(&world, AssetId(1)).is_some());
        // An id that was never injected yields None (callers then no-op).
        assert!(sprite_mut(&mut world, AssetId(999)).is_none());
        assert!(input(&world, AssetId(999)).is_none());
    }

    #[test]
    fn place_sprite_sets_geometry_and_is_a_noop_when_absent() {
        let mut world = world_with(&[AssetId(1)]);
        place_sprite(
            &mut world,
            AssetId(1),
            [5.0, 6.0, 7.0, 8.0],
            [1.0, 0.0, 0.0, 1.0],
            true,
        );
        let s = world
            .query::<Sprite>()
            .find(|s| s.asset_id == AssetId(1))
            .unwrap();
        assert_eq!((s.x, s.y, s.width, s.height), (5.0, 6.0, 7.0, 8.0));
        assert_eq!(s.tint, [1.0, 0.0, 0.0, 1.0]);
        assert!(s.visible);
        // A missing id changes nothing and does not panic.
        place_sprite(&mut world, AssetId(2), [0.0; 4], [0.0; 4], true);
        assert_eq!(world.query::<Sprite>().count(), 1);
    }

    #[test]
    fn visibility_setters_toggle_the_right_element() {
        let mut world = world_with(&[AssetId(1)]);
        set_sprite_visible(&mut world, AssetId(1), false);
        set_label_visible(&mut world, AssetId(1), false);
        assert!(!world.query::<Sprite>().next().unwrap().visible);
        assert!(!world.query::<TextLabel>().next().unwrap().visible);
    }

    #[test]
    fn place_panel_draws_a_rounded_chrome_surface() {
        let mut world = world_with(&[AssetId(1)]);
        place_panel(&mut world, AssetId(1), [40.0, 60.0, 320.0, 400.0]);
        let bg = world
            .query::<Sprite>()
            .find(|s| s.asset_id == AssetId(1))
            .unwrap();
        assert!(bg.visible);
        assert_eq!(
            (bg.x, bg.y, bg.width, bg.height),
            (40.0, 60.0, 320.0, 400.0)
        );
        assert_eq!(bg.tint, theme::CHROME_TINT);
        assert_eq!(bg.corner_radius, theme::PANEL_RADIUS);
        assert_eq!(bg.border_width, theme::PANEL_BORDER_WIDTH);
        assert_eq!(bg.border_color, theme::PANEL_BORDER_TINT);
    }

    #[test]
    fn place_heading_positions_the_title_text() {
        let mut world = world_with(&[AssetId(2)]);
        place_heading(
            &mut world,
            AssetId(2),
            [40.0, 60.0, 320.0, TITLE_H],
            "Assets",
        );
        let l = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == AssetId(2))
            .unwrap();
        assert!(l.visible);
        assert_eq!(l.content, "Assets");
        assert_eq!(l.align, TextAlign::Left);
        assert_eq!(l.x, 50.0, "heading is inset from the panel's left edge");
    }

    // The close button is a square flush in the title bar's right end; its
    // background is hidden until hovered (the glyph always shows), and hovering
    // reveals a rounded square behind it.
    #[test]
    fn close_button_shows_its_square_only_on_hover() {
        let mut world = world_with(&[AssetId(1), AssetId(2)]);
        let title = [40.0, 60.0, 320.0, TITLE_H];
        let r = close_rect(title);
        assert_eq!(r[2], TITLE_H, "square");
        assert_eq!(
            r[0] + r[2],
            title[0] + title[2],
            "flush to the title bar's right edge"
        );
        // Not hovered: no background at all (the panel surface shows through).
        place_close(&mut world, AssetId(1), AssetId(2), title, false);
        let bg = |w: &World| {
            w.query::<Sprite>()
                .find(|s| s.asset_id == AssetId(1))
                .cloned()
                .unwrap()
        };
        assert!(!bg(&world).visible, "no square while idle");
        let glyph = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == AssetId(2))
            .unwrap();
        assert!(
            glyph.visible && glyph.content == "X",
            "the X glyph always shows"
        );
        // Hovered: a rounded square appears.
        place_close(&mut world, AssetId(1), AssetId(2), title, true);
        let hovered = bg(&world);
        assert!(hovered.visible, "the square shows on hover");
        assert!(hovered.corner_radius > 0.0, "and it is rounded");
    }

    // The clamp hard-stops a panel at every window edge: it can never be dragged
    // even partially off screen, and an oversized panel pins to the top-left.
    #[test]
    fn clamp_origin_keeps_the_whole_panel_on_screen() {
        let size = [320.0, 400.0];
        let vp = [1280.0, 720.0];
        assert_eq!(
            clamp_origin([100.0, 50.0], size, vp),
            [100.0, 50.0],
            "an in-bounds origin is untouched"
        );
        assert_eq!(clamp_origin([-40.0, -9000.0], size, vp), [0.0, 0.0]);
        assert_eq!(
            clamp_origin([2000.0, 700.0], size, vp),
            [vp[0] - size[0], vp[1] - size[1]],
            "stops with the panel's far edges on the window's"
        );
        // Taller than the window: pinned to the top rather than pushed off.
        assert_eq!(clamp_origin([10.0, 300.0], [320.0, 900.0], vp), [10.0, 0.0]);
    }
}
