//! A list row's "..." menu for the editor HUD: three stacked dots at the row's
//! right end, and the menu of actions they open, floating below the row (or
//! above it when the panel has no room below). Plain `Sprite` / `TextLabel`
//! components at reserved ids, placed each frame by the owning panel, which
//! also routes the press.

use concinnity_core::ecs::World;
use concinnity_core::ecs::asset_id::AssetId;

use super::theme;
use super::widget::{self, place_rounded, point_in};

pub(crate) const DOT_SZ: f32 = 20.0;
pub(crate) const MAX_ITEMS: usize = 3;
const MENU_W: f32 = 132.0;
const ITEM_H: f32 = 26.0;
const ITEM_PAD: f32 = 10.0;

const DOT_BG_TINT: [f32; 4] = [0.30, 0.34, 0.46, 0.95];
const DOT_TINT: [f32; 4] = [0.90, 0.92, 0.96, 1.0];
const MENU_BG_TINT: [f32; 4] = [0.22, 0.23, 0.29, 1.0];
const ITEM_TINT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const DANGER_LABEL: [f32; 3] = [0.95, 0.60, 0.58];

// The reserved ids one row menu draws with.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MenuIds {
    pub dot_bg: AssetId,
    pub dots: [AssetId; 3],
    pub bg: AssetId,
    pub item_bgs: [AssetId; MAX_ITEMS],
    pub item_labels: [AssetId; MAX_ITEMS],
}

impl MenuIds {
    // Every sprite id in draw order: the dots, then the menu over them.
    pub(crate) fn sprites(&self) -> Vec<AssetId> {
        let mut ids = vec![self.dot_bg];
        ids.extend(self.dots);
        ids.push(self.bg);
        ids.extend(self.item_bgs);
        ids
    }

    pub(crate) fn labels(&self) -> Vec<AssetId> {
        self.item_labels.to_vec()
    }
}

// One action the menu offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Item {
    pub(crate) caption: &'static str,
    pub(crate) danger: bool,
}

// The dots at the right end of `row`, `inset` short of its edge.
pub(crate) fn dot_rect(row: [f32; 4], inset: f32) -> [f32; 4] {
    [
        row[0] + row[2] - inset - DOT_SZ,
        row[1] + (row[3] - DOT_SZ) * 0.5,
        DOT_SZ,
        DOT_SZ,
    ]
}

// The menu for `count` items off `row`, right-aligned to `right`: below the
// row, or above it when that would pass `bottom`. Returns the backing and one
// rect per item.
pub(crate) fn menu_rects(
    row: [f32; 4],
    right: f32,
    count: usize,
    bottom: f32,
) -> ([f32; 4], Vec<[f32; 4]>) {
    let h = count as f32 * ITEM_H;
    let below = row[1] + row[3];
    let top = match below + h <= bottom {
        true => below,
        false => row[1] - h,
    };
    let x = right - MENU_W;
    let items = (0..count)
        .map(|i| [x, top + i as f32 * ITEM_H, MENU_W, ITEM_H])
        .collect();
    ([x, top, MENU_W, h], items)
}

// The item under `(mx, my)`.
pub(crate) fn hit_item(mx: f32, my: f32, items: &[[f32; 4]]) -> Option<usize> {
    items.iter().position(|&r| point_in(mx, my, r))
}

// The three dots; the backing box shows only while `boxed` (hovered, or their
// menu is open).
pub(crate) fn place_dots(world: &mut World, ids: &MenuIds, d: [f32; 4], boxed: bool) {
    match boxed {
        true => place_rounded(
            world,
            ids.dot_bg,
            d,
            DOT_BG_TINT,
            theme::CONTROL_RADIUS,
            true,
        ),
        false => widget::set_sprite_visible(world, ids.dot_bg, false),
    }
    let (cx, cy) = (d[0] + d[2] * 0.5, d[1] + d[3] * 0.5);
    let (s, gap) = (3.5, 3.5);
    for (id, dy) in ids.dots.into_iter().zip([-gap - s, -s * 0.5, gap]) {
        widget::place_sprite(world, id, [cx - s * 0.5, cy + dy, s, s], DOT_TINT, true);
    }
}

pub(crate) fn hide_dots(world: &mut World, ids: &MenuIds) {
    widget::set_sprite_visible(world, ids.dot_bg, false);
    for id in ids.dots {
        widget::set_sprite_visible(world, id, false);
    }
}

// The open menu: its backing at `bg` and `items` at `rects`, the one under the
// cursor highlighted.
pub(crate) fn place_menu(
    world: &mut World,
    ids: &MenuIds,
    (bg, rects): (&[f32; 4], &[[f32; 4]]),
    items: &[Item],
    mouse: [f32; 2],
) {
    if let Some(sprite) = widget::sprite_mut(world, ids.bg) {
        sprite.x = bg[0];
        sprite.y = bg[1];
        sprite.width = bg[2];
        sprite.height = bg[3];
        sprite.tint = MENU_BG_TINT;
        sprite.corner_radius = theme::CONTROL_RADIUS;
        sprite.border_width = theme::PANEL_BORDER_WIDTH;
        sprite.border_color = theme::PANEL_BORDER_TINT;
        sprite.visible = true;
    }
    for slot in 0..MAX_ITEMS {
        let (bg_id, label_id) = (ids.item_bgs[slot], ids.item_labels[slot]);
        let (Some(item), Some(&r)) = (items.get(slot), rects.get(slot)) else {
            widget::set_sprite_visible(world, bg_id, false);
            widget::set_label_visible(world, label_id, false);
            continue;
        };
        let tint = match point_in(mouse[0], mouse[1], r) {
            true => theme::HOVER_TINT,
            false => ITEM_TINT,
        };
        place_rounded(world, bg_id, r, tint, theme::CONTROL_RADIUS, true);
        let color = match item.danger {
            true => DANGER_LABEL,
            false => theme::LABEL,
        };
        widget::place_left_label(
            world,
            label_id,
            [r[0] + ITEM_PAD, r[1] + r[3] * 0.5 - theme::TEXT_HALF],
            item.caption,
            color,
            true,
        );
    }
}

pub(crate) fn hide_menu(world: &mut World, ids: &MenuIds) {
    widget::set_sprite_visible(world, ids.bg, false);
    for slot in 0..MAX_ITEMS {
        widget::set_sprite_visible(world, ids.item_bgs[slot], false);
        widget::set_label_visible(world, ids.item_labels[slot], false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::{Sprite, TextLabel};

    const IDS: MenuIds = MenuIds {
        dot_bg: AssetId(1),
        dots: [AssetId(2), AssetId(3), AssetId(4)],
        bg: AssetId(5),
        item_bgs: [AssetId(6), AssetId(7), AssetId(10)],
        item_labels: [AssetId(8), AssetId(9), AssetId(11)],
    };
    const ROW: [f32; 4] = [0.0, 100.0, 400.0, 28.0];

    fn world() -> World {
        crate::test_support::injected_world(&IDS.sprites(), &IDS.labels(), &[])
    }

    #[test]
    fn the_menu_opens_below_the_row_unless_there_is_no_room() {
        let (bg, items) = menu_rects(ROW, 395.0, 2, 1000.0);
        assert_eq!(bg[1], ROW[1] + ROW[3]);
        assert_eq!(bg[0] + bg[2], 395.0);
        assert_eq!(items.len(), 2);
        assert_eq!(items[1][1], items[0][1] + ITEM_H);
        let (bg, _) = menu_rects(ROW, 395.0, 2, ROW[1] + ROW[3] + 10.0);
        assert_eq!(bg[1] + bg[3], ROW[1], "above the row");
        let d = dot_rect(ROW, 8.0);
        assert_eq!(d[0] + d[2], ROW[0] + ROW[2] - 8.0);
    }

    #[test]
    fn an_item_is_hit_by_its_own_rect() {
        let (_, items) = menu_rects(ROW, 395.0, 2, 1000.0);
        assert_eq!(
            hit_item(items[1][0] + 2.0, items[1][1] + 2.0, &items),
            Some(1)
        );
        assert_eq!(hit_item(0.0, 0.0, &items), None);
    }

    #[test]
    fn place_draws_the_items_given_and_hides_the_rest() {
        let mut world = world();
        let (bg, rects) = menu_rects(ROW, 395.0, 1, 1000.0);
        let items = [Item {
            caption: "Remove",
            danger: true,
        }];
        place_menu(&mut world, &IDS, (&bg, &rects), &items, [0.0, 0.0]);
        let first = world.get_by_id::<TextLabel>(IDS.item_labels[0]).unwrap();
        assert!(first.visible && first.content == "Remove");
        assert_eq!(first.color, DANGER_LABEL);
        assert!(!world.get_by_id::<Sprite>(IDS.item_bgs[1]).unwrap().visible);

        place_dots(&mut world, &IDS, dot_rect(ROW, 8.0), false);
        assert!(!world.get_by_id::<Sprite>(IDS.dot_bg).unwrap().visible);
        assert!(world.get_by_id::<Sprite>(IDS.dots[0]).unwrap().visible);
        hide_dots(&mut world, &IDS);
        hide_menu(&mut world, &IDS);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
    }
}
