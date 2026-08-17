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

// A panel's outer rect from its origin `o` and effective size `s`.
pub(crate) fn outer_rect(o: [f32; 2], s: [f32; 2]) -> [f32; 4] {
    [o[0], o[1], s[0], s[1]]
}

// The draggable title bar across a panel's top, at the effective width `w`.
pub(crate) fn title_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    [o[0], o[1], w, TITLE_H]
}

// The y where a panel's header row begins: just below the title bar, plus a
// per-panel `gap`.
pub(crate) fn header_y(o: [f32; 2], gap: f32) -> f32 {
    o[1] + TITLE_H + gap
}

// Position a left-aligned label at `pos`, setting its text, color, and
// visibility. A no-op if the label is absent.
pub(crate) fn place_left_label(
    world: &mut World,
    id: AssetId,
    pos: [f32; 2],
    content: &str,
    color: [f32; 3],
    visible: bool,
) {
    if let Some(l) = label_mut(world, id) {
        l.x = pos[0];
        l.y = pos[1];
        l.align = TextAlign::Left;
        l.color = color;
        l.visible = visible;
        l.content = content.to_string();
    }
}

// One line of the HUD font at the editor's text scale. The renderer does the
// real measuring; the editor needs this only to say how many lines a box holds.
pub(crate) const LINE_H: f32 = theme::TEXT_HALF * 2.0;

// How many lines of text fit in a box `h` pixels tall.
pub(crate) fn lines_in(h: f32) -> u32 {
    ((h / LINE_H).floor() as u32).max(1)
}

// A message bounded by the box that holds it: it wraps to `rect`'s width and
// stops after as many lines as `rect` is tall, ending in an ellipsis if there
// was more. Use this for text whose length is not known in advance -- a status
// line, an error, anything quoting a message from elsewhere -- so it can never
// draw outside its panel.
pub(crate) fn place_message(
    world: &mut World,
    id: AssetId,
    rect: [f32; 4],
    content: &str,
    color: [f32; 3],
    visible: bool,
) {
    let lines = lines_in(rect[3]);
    if let Some(l) = label_mut(world, id) {
        l.x = rect[0];
        l.y = rect[1];
        l.align = TextAlign::Left;
        l.color = color;
        l.visible = visible;
        l.wrap_width = rect[2].max(0.0);
        l.max_lines = lines;
        l.content = content.to_string();
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

// Clamp a dragged panel origin so the whole `size` panel stays on screen below
// the top bar: a hard stop at every window edge and at `top` (the bar's lower
// edge), so a panel can never slide under the bar. A panel too large for the
// region pins to its top-left corner.
pub(crate) fn clamp_origin(
    pos: [f32; 2],
    size: [f32; 2],
    viewport: [f32; 2],
    top: f32,
) -> [f32; 2] {
    [
        pos[0].clamp(0.0, (viewport[0] - size[0]).max(0.0)),
        pos[1].clamp(top, (viewport[1] - size[1]).max(top)),
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

// `place_rounded` with a stroke around it, for an element that reads as its own
// surface sitting on the panel rather than a tint over it.
pub(crate) fn place_bordered(
    world: &mut World,
    id: AssetId,
    rect: [f32; 4],
    tint: [f32; 4],
    border: [f32; 4],
    width: f32,
) {
    place_rounded(world, id, rect, tint, theme::CONTROL_RADIUS, true);
    if let Some(s) = sprite_mut(world, id) {
        s.border_color = border;
        s.border_width = width;
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

// Hide every listed element (the F1-hidden pass, or when a panel is toggled
// off). Fields are blurred as well, so a hidden field cannot keep keyboard
// focus.
pub(crate) fn hide_all(
    world: &mut World,
    sprites: &[AssetId],
    labels: &[AssetId],
    fields: &[AssetId],
) {
    for &id in sprites {
        set_sprite_visible(world, id, false);
    }
    for &id in labels {
        set_label_visible(world, id, false);
    }
    for &id in fields {
        hide_field(world, id);
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

// Whether `(x, y)` lies inside `rect` ([x, y, w, h], top-left origin). The
// top-left edge is inside; the bottom-right edge is outside.
pub(crate) fn point_in(x: f32, y: f32, rect: [f32; 4]) -> bool {
    x >= rect[0] && x < rect[0] + rect[2] && y >= rect[1] && y < rect[1] + rect[3]
}

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
    fn place_message_bounds_text_to_the_box_that_holds_it() {
        let mut world = World::new_empty();
        let id = AssetId(1);
        world.add_component(TextLabel {
            asset_id: id,
            ..Default::default()
        });
        // Two lines tall, so a long message wraps once and then stops.
        place_message(
            &mut world,
            id,
            [10.0, 20.0, 300.0, 2.0 * LINE_H],
            "a long message",
            [1.0; 3],
            true,
        );
        let l = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == id)
            .unwrap();
        assert_eq!((l.x, l.y), (10.0, 20.0));
        assert_eq!(l.wrap_width, 300.0);
        assert_eq!(l.max_lines, 2);
    }

    #[test]
    fn a_box_always_holds_at_least_one_line() {
        assert_eq!(lines_in(0.0), 1);
        assert_eq!(lines_in(LINE_H), 1);
        assert_eq!(lines_in(3.0 * LINE_H), 3);
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

    // The clamp hard-stops a panel at every window edge and at the top bar: it
    // can never be dragged even partially off screen or under the bar, and a
    // panel too tall for the region pins to the top-left below the bar.
    #[test]
    fn clamp_origin_keeps_the_whole_panel_on_screen() {
        let size = [320.0, 400.0];
        let vp = [1280.0, 720.0];
        let top = 40.0;
        assert_eq!(
            clamp_origin([100.0, 50.0], size, vp, top),
            [100.0, 50.0],
            "an in-bounds origin is untouched"
        );
        assert_eq!(
            clamp_origin([-40.0, -9000.0], size, vp, top),
            [0.0, top],
            "stops at the left edge and the bar's lower edge"
        );
        assert_eq!(
            clamp_origin([2000.0, 700.0], size, vp, top),
            [vp[0] - size[0], vp[1] - size[1]],
            "stops with the panel's far edges on the window's"
        );
        // Taller than the region below the bar: pinned under the bar, not off.
        assert_eq!(
            clamp_origin([10.0, 300.0], [320.0, 900.0], vp, top),
            [10.0, top]
        );
    }

    // The outer rect is the origin with the size as its width / height.
    #[test]
    fn outer_rect_is_origin_plus_size() {
        assert_eq!(
            outer_rect([40.0, 60.0], [320.0, 400.0]),
            [40.0, 60.0, 320.0, 400.0]
        );
    }

    // The title bar spans the given width at the origin, one bar tall.
    #[test]
    fn title_rect_spans_the_width_at_the_bar_height() {
        let t = title_rect([40.0, 60.0], 320.0);
        assert_eq!(t, [40.0, 60.0, 320.0, TITLE_H]);
        // The close button flushes to the bar's right edge (the paired helper).
        assert_eq!(close_rect(t)[0] + TITLE_H, t[0] + t[2]);
    }

    // The header row sits one title bar below the origin, plus the panel's gap.
    #[test]
    fn header_y_sits_below_the_title_bar_plus_gap() {
        assert_eq!(header_y([40.0, 60.0], 0.0), 60.0 + TITLE_H);
        assert_eq!(header_y([40.0, 60.0], 10.0), 60.0 + TITLE_H + 10.0);
    }

    // A left label takes the given position, text, and color; a missing id is a
    // no-op rather than a panic.
    #[test]
    fn place_left_label_positions_and_no_ops_when_absent() {
        let mut world = world_with(&[AssetId(1)]);
        place_left_label(
            &mut world,
            AssetId(1),
            [12.0, 34.0],
            "hi",
            [0.1, 0.2, 0.3],
            true,
        );
        let l = label_mut(&mut world, AssetId(1)).unwrap();
        assert_eq!((l.x, l.y), (12.0, 34.0));
        assert_eq!(l.align, TextAlign::Left);
        assert_eq!(l.color, [0.1, 0.2, 0.3]);
        assert!(l.visible && l.content == "hi");
        // A never-injected id changes nothing and does not panic.
        place_left_label(&mut world, AssetId(999), [0.0, 0.0], "x", [0.0; 3], true);
        assert_eq!(world.query::<TextLabel>().count(), 1);
    }

    #[test]
    fn point_in_includes_top_left_excludes_bottom_right() {
        let r = [10.0, 20.0, 100.0, 40.0];
        assert!(point_in(10.0, 20.0, r), "top-left corner is inside");
        assert!(point_in(50.0, 40.0, r), "interior is inside");
        assert!(!point_in(110.0, 40.0, r), "right edge is outside");
        assert!(!point_in(50.0, 60.0, r), "bottom edge is outside");
        assert!(!point_in(9.9, 40.0, r), "left of the rect is outside");
    }
}
