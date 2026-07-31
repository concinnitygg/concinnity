// src/editor/view_menu.rs
//
// The Display menu: a compact dropdown under the top bar's Display chip
// selecting the viewport view mode (radio rows) and the per-session show
// flags (toggle rows), plus the editor-side billboard toggle. Pure geometry
// and draw, on the same non-panel overlay pattern as `create_menu.rs`; the
// hook (`hook/view_menu_drive.rs`) owns the open state and routing.

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

// One menu row: a view-mode radio, the show-flags divider, one flag toggle,
// or the billboard-icons toggle (editor overlay sprites, not a render pass).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum MenuRow {
    Mode(ViewMode),
    ShowHeading,
    Flag(ShowFlags, &'static str),
    Billboards,
}

pub(crate) fn rows() -> Vec<MenuRow> {
    ViewMode::ALL
        .into_iter()
        .map(MenuRow::Mode)
        .chain(std::iter::once(MenuRow::ShowHeading))
        .chain(
            ShowFlags::LABELED
                .into_iter()
                .map(|(f, label)| MenuRow::Flag(f, label)),
        )
        .chain(std::iter::once(MenuRow::Billboards))
        .collect()
}

pub(crate) fn row_count() -> usize {
    ViewMode::ALL.len() + 1 + ShowFlags::LABELED.len() + 1
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
            MenuRow::ShowHeading => ("Show".to_string(), false, false),
            MenuRow::Flag(f, label) => (label.to_string(), state.show.contains(f), true),
            MenuRow::Billboards => ("Billboards".to_string(), state.billboards, true),
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
            MenuRow::ShowHeading => theme::HEADING,
            MenuRow::Mode(_) => theme::LABEL,
            _ if on => theme::LABEL,
            _ => theme::LABEL_DIM,
        };
        let indent = if matches!(row, MenuRow::ShowHeading) {
            0.0
        } else {
            6.0
        };
        let text = match row {
            // Toggle rows carry an on/off marker; mode rows read as a radio
            // via the accent background.
            MenuRow::Flag(..) | MenuRow::Billboards => {
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
