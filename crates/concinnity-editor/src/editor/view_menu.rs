// src/editor/view_menu.rs
//
// The Display menu: a compact dropdown under the top bar's Display chip
// selecting the viewport view mode (radio rows) and the per-session show
// flags (toggle rows), plus the editor-side billboard toggle. Pure geometry
// and draw, on the same non-panel overlay pattern as `create_menu.rs`; the
// hook (`hook/view_menu_drive.rs`) owns the open state and routing.

use super::outlines::{Category, CategorySet};
use super::registry::ID_BASE;
use super::theme;
use super::widget::{self, point_in};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

pub(crate) use concinnity_core::gfx::view_modes::{ShowFlags, ViewMode};

// Reserved id family: the next free block after the create menu's (0x6000).
const BASE: u32 = ID_BASE + 0x7000;
pub(crate) const MENU_BG: AssetId = AssetId(BASE);
pub(crate) const HEADING: AssetId = AssetId(BASE + 1);
fn row_bg(i: usize) -> AssetId {
    AssetId(BASE + 0x10 + i as u32)
}
fn row_label(i: usize) -> AssetId {
    AssetId(BASE + 0x40 + i as u32)
}

const MENU_W: f32 = 170.0;
const ROW_H: f32 = 22.0;
const HEADING_H: f32 = 22.0;
const PAD: f32 = 6.0;
const MARGIN: f32 = 8.0;

// One menu row: a view-mode radio, a section heading, one show-flag toggle,
// the billboard-icons toggle (editor overlay sprites, not a render pass), or
// one always-on extent-outline category.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum MenuRow {
    Mode(ViewMode),
    Heading(&'static str),
    Flag(ShowFlags, &'static str),
    Billboards,
    Extent(Category, &'static str),
}

pub(crate) fn rows() -> Vec<MenuRow> {
    ViewMode::ALL
        .into_iter()
        .map(MenuRow::Mode)
        .chain(std::iter::once(MenuRow::Heading("Show")))
        .chain(
            ShowFlags::LABELED
                .into_iter()
                .map(|(f, label)| MenuRow::Flag(f, label)),
        )
        .chain(std::iter::once(MenuRow::Billboards))
        .chain(std::iter::once(MenuRow::Heading("Extents")))
        .chain(
            Category::LABELED
                .into_iter()
                .map(|(c, label)| MenuRow::Extent(c, label)),
        )
        .collect()
}

// Kept allocation-free: the layout, hit test, and per-frame hide all call it,
// and `rows_cover_every_mode_and_flag_once` pins it against `rows()`.
pub(crate) fn row_count() -> usize {
    ViewMode::ALL.len() + 1 + ShowFlags::LABELED.len() + 1 + 1 + Category::LABELED.len()
}

// The menu's top-left: right-aligned under the top bar, where the Display
// chip lives.
pub(crate) fn origin(vw: f32) -> [f32; 2] {
    [(vw - MENU_W - MARGIN).max(0.0), super::hud::BAR_H + 4.0]
}

pub(crate) fn size() -> [f32; 2] {
    [MENU_W, HEADING_H + row_count() as f32 * ROW_H + PAD * 2.0]
}

pub(crate) fn menu_rect(vw: f32) -> [f32; 4] {
    widget::outer_rect(origin(vw), size())
}

pub(crate) fn row_rect(vw: f32, i: usize) -> [f32; 4] {
    let o = origin(vw);
    [
        o[0] + PAD,
        o[1] + HEADING_H + PAD + i as f32 * ROW_H,
        MENU_W - PAD * 2.0,
        ROW_H,
    ]
}

pub(crate) fn hit_row(mx: f32, my: f32, vw: f32) -> Option<usize> {
    (0..row_count()).find(|&i| point_in(mx, my, row_rect(vw, i)))
}

pub(crate) fn over(mx: f32, my: f32, vw: f32) -> bool {
    point_in(mx, my, menu_rect(vw))
}

// The menu's selection state this frame, for the draw.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MenuState {
    pub mode: ViewMode,
    pub show: ShowFlags,
    pub billboards: bool,
    pub extents: CategorySet,
}

pub(crate) fn apply(world: &mut World, vw: f32, state: MenuState, mouse: [f32; 2]) {
    let o = origin(vw);
    widget::place_panel(world, MENU_BG, menu_rect(vw));
    widget::place_left_label(
        world,
        HEADING,
        [o[0] + PAD + 4.0, o[1] + 3.0],
        "View mode",
        theme::HEADING,
        true,
    );
    for (i, row) in rows().into_iter().enumerate() {
        let r = row_rect(vw, i);
        let hovered = point_in(mouse[0], mouse[1], r);
        let (caption, on, selectable) = match row {
            MenuRow::Mode(m) => (m.label().to_string(), state.mode == m, true),
            MenuRow::Heading(label) => (label.to_string(), false, false),
            MenuRow::Flag(f, label) => (label.to_string(), state.show.contains(f), true),
            MenuRow::Billboards => ("Billboards".to_string(), state.billboards, true),
            MenuRow::Extent(c, label) => (label.to_string(), state.extents.contains(c), true),
        };
        // A selected mode keeps the accent; other actionable rows light on
        // hover; the divider heading draws label-only.
        let (tint, bg_visible) = if on && matches!(row, MenuRow::Mode(_)) {
            (theme::ACCENT_TINT, true)
        } else {
            (theme::HOVER_TINT, hovered && selectable)
        };
        widget::place_rounded(world, row_bg(i), r, tint, theme::CONTROL_RADIUS, bg_visible);
        let color = match row {
            MenuRow::Heading(_) => theme::HEADING,
            MenuRow::Mode(_) => theme::LABEL,
            _ if on => theme::LABEL,
            _ => theme::LABEL_DIM,
        };
        let indent = if matches!(row, MenuRow::Heading(_)) {
            0.0
        } else {
            6.0
        };
        let text = match row {
            // Toggle rows carry an on/off marker; mode rows read as a radio
            // via the accent background.
            MenuRow::Flag(..) | MenuRow::Billboards | MenuRow::Extent(..) => {
                format!("{} {}", if on { "[x]" } else { "[ ]" }, caption)
            }
            _ => caption,
        };
        widget::place_left_label(
            world,
            row_label(i),
            [r[0] + indent, r[1] + (ROW_H - widget::LINE_H) * 0.5],
            &text,
            color,
            true,
        );
    }
}

pub(crate) fn hide(world: &mut World) {
    widget::set_sprite_visible(world, MENU_BG, false);
    widget::set_label_visible(world, HEADING, false);
    for i in 0..row_count() {
        widget::set_sprite_visible(world, row_bg(i), false);
        widget::set_label_visible(world, row_label(i), false);
    }
}

pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    std::iter::once(MENU_BG)
        .chain((0..row_count()).map(row_bg))
        .collect()
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    std::iter::once(HEADING)
        .chain((0..row_count()).map(row_label))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Sprite, TextLabel};

    fn injected_world() -> World {
        let mut world = World::new_empty();
        for id in all_sprite_ids() {
            world.add_component(Sprite {
                asset_id: id,
                ..Default::default()
            });
        }
        for id in all_label_ids() {
            world.add_component(TextLabel {
                asset_id: id,
                ..Default::default()
            });
        }
        world
    }

    fn state() -> MenuState {
        MenuState {
            mode: ViewMode::default(),
            show: ShowFlags::all(),
            billboards: true,
            extents: CategorySet::default(),
        }
    }

    fn label_of(world: &World, id: AssetId) -> String {
        world
            .query::<TextLabel>()
            .find(|l| l.asset_id == id)
            .map(|l| l.content.clone())
            .unwrap_or_default()
    }

    fn sprite_visible(world: &World, id: AssetId) -> bool {
        world
            .query::<Sprite>()
            .find(|s| s.asset_id == id)
            .is_some_and(|s| s.visible)
    }

    // The draw labels every row from `rows()`: mode rows read as plain
    // captions (the accent background is their radio), and every toggle row
    // carries its on/off marker.
    #[test]
    fn apply_labels_every_row_and_marks_the_toggles() {
        let vw = 1280.0;
        let mut world = injected_world();
        apply(&mut world, vw, state(), [0.0, 0.0]);

        assert_eq!(label_of(&world, HEADING), "View mode");
        for (i, row) in rows().into_iter().enumerate() {
            let text = label_of(&world, row_label(i));
            assert!(!text.is_empty(), "row {i} ({row:?}) drew no label");
            match row {
                MenuRow::Flag(..) | MenuRow::Billboards | MenuRow::Extent(..) => assert!(
                    text.starts_with("[x] ") || text.starts_with("[ ] "),
                    "row {i} ({row:?}) is a toggle but drew {text:?}"
                ),
                _ => assert!(
                    !text.starts_with('['),
                    "row {i} ({row:?}) is not a toggle but drew {text:?}"
                ),
            }
        }
    }

    // The markers follow the state: a cleared flag and an unset extent read
    // as off, and the selected mode is the only row keeping a background with
    // the cursor away from the menu.
    #[test]
    fn apply_reflects_the_selection_state() {
        let vw = 1280.0;
        let mut world = injected_world();
        let mut s = state();
        s.show = ShowFlags::all().toggled(ShowFlags::FOG);
        apply(&mut world, vw, s, [0.0, 0.0]);

        for (i, row) in rows().into_iter().enumerate() {
            let text = label_of(&world, row_label(i));
            match row {
                MenuRow::Flag(f, _) => assert_eq!(
                    text.starts_with("[x]"),
                    f != ShowFlags::FOG,
                    "flag row {i} marker disagrees with the state: {text:?}"
                ),
                // Nothing is selected in the default set.
                MenuRow::Extent(..) => assert!(text.starts_with("[ ]"), "{text:?}"),
                _ => {}
            }
        }

        let lit: Vec<usize> = (0..row_count())
            .filter(|&i| sprite_visible(&world, row_bg(i)))
            .collect();
        let selected = rows()
            .into_iter()
            .position(|r| r == MenuRow::Mode(s.mode))
            .expect("the selected mode is a row");
        assert_eq!(
            lit,
            vec![selected],
            "only the selected mode keeps a background off-hover"
        );
    }

    // A hovered actionable row lights up; a hovered heading stays flat,
    // because there is nothing there to click.
    #[test]
    fn apply_lights_a_hovered_row_but_never_a_heading() {
        let vw = 1280.0;
        let rows = rows();
        let flag = rows
            .iter()
            .position(|r| matches!(r, MenuRow::Flag(..)))
            .expect("a flag row");
        let heading = rows
            .iter()
            .position(|r| matches!(r, MenuRow::Heading(_)))
            .expect("a heading row");

        for (i, expected) in [(flag, true), (heading, false)] {
            let mut world = injected_world();
            let r = row_rect(vw, i);
            apply(&mut world, vw, state(), [r[0] + 2.0, r[1] + 2.0]);
            assert_eq!(
                sprite_visible(&world, row_bg(i)),
                expected,
                "row {i} ({:?}) hover background",
                rows[i]
            );
        }
    }

    // The per-frame hide blanks everything the draw shows, so a closed menu
    // leaves nothing painted over the viewport.
    #[test]
    fn hide_blanks_everything_apply_drew() {
        let mut world = injected_world();
        apply(&mut world, 1280.0, state(), [0.0, 0.0]);
        hide(&mut world);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
    }

    #[test]
    fn rows_cover_every_mode_and_flag_once() {
        let rows = rows();
        assert_eq!(rows.len(), row_count());
        for m in ViewMode::ALL {
            assert!(rows.contains(&MenuRow::Mode(m)), "{m:?}");
        }
        for (f, label) in ShowFlags::LABELED {
            assert!(rows.contains(&MenuRow::Flag(f, label)));
        }
        assert!(rows.contains(&MenuRow::Billboards));
        for (c, label) in Category::LABELED {
            assert!(rows.contains(&MenuRow::Extent(c, label)));
        }
    }

    #[test]
    fn geometry_stays_on_screen_and_rows_stack() {
        let vw = 1280.0;
        let rect = menu_rect(vw);
        assert!(rect[0] >= 0.0 && rect[0] + rect[2] <= vw);
        assert!(rect[1] >= super::super::hud::BAR_H);
        for i in 0..row_count() {
            let r = row_rect(vw, i);
            assert!(r[0] >= rect[0] && r[0] + r[2] <= rect[0] + rect[2]);
            if i > 0 {
                assert_eq!(r[1], row_rect(vw, i - 1)[1] + ROW_H);
            }
        }
    }

    #[test]
    fn hit_row_resolves_rows_and_misses_outside() {
        let vw = 1280.0;
        let r0 = row_rect(vw, 0);
        assert_eq!(hit_row(r0[0] + 2.0, r0[1] + 2.0, vw), Some(0),);
        let last = row_rect(vw, row_count() - 1);
        assert_eq!(
            hit_row(last[0] + 2.0, last[1] + 2.0, vw),
            Some(row_count() - 1),
        );
        assert_eq!(hit_row(5.0, 5.0, vw), None);
        assert!(over(r0[0], r0[1], vw));
        assert!(!over(5.0, 5.0, vw));
    }
}
