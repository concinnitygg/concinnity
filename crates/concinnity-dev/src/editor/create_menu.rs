// src/editor/create_menu.rs
//
// The viewport's right-click "Create here" menu: a floating list anchored at
// the cursor that adds an asset at the world position under the click. The
// offered types come from the registry (every blank-useful type that carries a
// world position -- the billboard-eligible set), plus a disclosure section
// listing the world's Prefabs (each creates a Prop instancing that prefab).
// Pure geometry, item derivation, and draw here; the open / click / commit
// flow lives in `hook/create_menu_drive.rs`.

use super::registry::ID_BASE;
use super::widget::{self, point_in};
use super::{billboards, panel, theme};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// Reserved id family: the next free block after the Content panel's (0x5000).
const BASE: u32 = ID_BASE + 0x6000;

pub(crate) const MENU_BG: AssetId = AssetId(BASE);
const HEADING: AssetId = AssetId(BASE + 1);

const fn row_bg(i: usize) -> AssetId {
    AssetId(BASE + 0x20 + i as u32)
}
const fn row_label(i: usize) -> AssetId {
    AssetId(BASE + 0x60 + i as u32)
}

// Row pool bound; items past it are clipped (a world with that many Prefabs
// still creates them through the Content panel or the console).
pub(crate) const MAX_ROWS: usize = 24;

const MENU_W: f32 = 190.0;
const ROW_H: f32 = 24.0;
const PAD: f32 = 4.0;
const HEADING_H: f32 = 22.0;
const INDENT: f32 = 14.0;

// One menu row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MenuItem {
    // A placeable registered type: picking it creates one at the click point.
    Type(&'static str),
    // The Prefab disclosure row; picking it unfolds / folds the Prefab list.
    PrefabHeader { open: bool },
    // A world Prefab: picking it creates a Prop instancing it.
    Prefab(String),
}

impl MenuItem {
    fn caption(&self) -> String {
        match self {
            MenuItem::Type(ty) => (*ty).to_string(),
            MenuItem::PrefabHeader { open: false } => "Prefab instance >".to_string(),
            MenuItem::PrefabHeader { open: true } => "Prefab instance v".to_string(),
            MenuItem::Prefab(name) => name.clone(),
        }
    }

    fn indented(&self) -> bool {
        matches!(self, MenuItem::Prefab(_))
    }
}

// The registered types the menu offers: every type the "+" picker adds blank
// (`useful_blank`) that also carries a world position (the billboard-eligible
// predicate), alphabetized. Singletons and renderable-config types are out by
// construction: the former are not in `add_types`, the latter carry no
// position.
pub(crate) fn placeable_types() -> Vec<&'static str> {
    let mut types: Vec<&'static str> = panel::add_types()
        .filter(|ty| billboards::eligible(ty))
        .collect();
    types.sort_unstable();
    types
}

// The authored arg key holding the type's world position ("position", or
// "centre" for the panel lights).
pub(crate) fn position_key(ty: &str) -> &'static str {
    if super::form::working_args(ty, None).contains_key("centre") {
        "centre"
    } else {
        "position"
    }
}

// The menu rows for this frame: the placeable types, then the Prefab section
// when the world has any. Clipped to the row pool.
pub(crate) fn items(prefabs: &[String], prefabs_open: bool) -> Vec<MenuItem> {
    let mut out: Vec<MenuItem> = placeable_types().into_iter().map(MenuItem::Type).collect();
    if !prefabs.is_empty() {
        out.push(MenuItem::PrefabHeader { open: prefabs_open });
        if prefabs_open {
            out.extend(prefabs.iter().cloned().map(MenuItem::Prefab));
        }
    }
    out.truncate(MAX_ROWS);
    out
}

pub(crate) fn size(rows: usize) -> [f32; 2] {
    [MENU_W, HEADING_H + rows as f32 * ROW_H + PAD * 2.0]
}

// The menu's top-left for `anchor` (the right-click position), kept fully on
// screen below the top bar.
pub(crate) fn origin(anchor: [f32; 2], rows: usize, vp: [f32; 2]) -> [f32; 2] {
    widget::clamp_origin(anchor, size(rows), vp, super::hud::BAR_H)
}

pub(crate) fn menu_rect(o: [f32; 2], rows: usize) -> [f32; 4] {
    widget::outer_rect(o, size(rows))
}

pub(crate) fn row_rect(o: [f32; 2], i: usize) -> [f32; 4] {
    [
        o[0] + PAD,
        o[1] + HEADING_H + PAD + i as f32 * ROW_H,
        MENU_W - PAD * 2.0,
        ROW_H,
    ]
}

// The row index under `(mx, my)`, or `None` for a press off every row (the
// caller pairs it with `over` to tell the menu's chrome from a click-away).
pub(crate) fn hit_row(mx: f32, my: f32, o: [f32; 2], rows: usize) -> Option<usize> {
    (0..rows).find(|&i| point_in(mx, my, row_rect(o, i)))
}

pub(crate) fn over(mx: f32, my: f32, o: [f32; 2], rows: usize) -> bool {
    point_in(mx, my, menu_rect(o, rows))
}

pub(crate) fn apply(world: &mut World, o: [f32; 2], items: &[MenuItem], mouse: [f32; 2]) {
    widget::place_panel(world, MENU_BG, menu_rect(o, items.len()));
    widget::place_left_label(
        world,
        HEADING,
        [o[0] + PAD + 4.0, o[1] + 3.0],
        "Create here",
        theme::HEADING,
        true,
    );
    for slot in 0..MAX_ROWS {
        match items.get(slot) {
            Some(item) => {
                let r = row_rect(o, slot);
                let hovered = point_in(mouse[0], mouse[1], r);
                widget::place_rounded(
                    world,
                    row_bg(slot),
                    r,
                    theme::HOVER_TINT,
                    theme::CONTROL_RADIUS,
                    hovered,
                );
                let indent = if item.indented() { INDENT } else { 0.0 };
                widget::place_left_label(
                    world,
                    row_label(slot),
                    [r[0] + 6.0 + indent, r[1] + (ROW_H - widget::LINE_H) * 0.5],
                    &widget::clip_text(&item.caption(), 24),
                    theme::LABEL,
                    true,
                );
            }
            None => {
                widget::set_sprite_visible(world, row_bg(slot), false);
                widget::set_label_visible(world, row_label(slot), false);
            }
        }
    }
}

pub(crate) fn hide(world: &mut World) {
    widget::set_sprite_visible(world, MENU_BG, false);
    widget::set_label_visible(world, HEADING, false);
    for slot in 0..MAX_ROWS {
        widget::set_sprite_visible(world, row_bg(slot), false);
        widget::set_label_visible(world, row_label(slot), false);
    }
}

pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    std::iter::once(MENU_BG)
        .chain((0..MAX_ROWS).map(row_bg))
        .collect()
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    std::iter::once(HEADING)
        .chain((0..MAX_ROWS).map(row_label))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // The offered set is registry-derived: position-carrying blank-useful
    // types are in; renderable, positionless, and singleton types are not.
    #[test]
    fn placeable_types_are_the_positioned_blank_useful_set() {
        let types = placeable_types();
        for expected in [
            "PointLight",
            "SpotLight",
            "TriggerVolume",
            "ReflectionProbe",
        ] {
            assert!(types.contains(&expected), "{expected} missing: {types:?}");
        }
        for excluded in ["Prop", "Sprite", "GraphicsConfig", "Window", "Material"] {
            assert!(!types.contains(&excluded), "{excluded} offered: {types:?}");
        }
        let mut sorted = types.clone();
        sorted.sort_unstable();
        assert_eq!(types, sorted, "alphabetized");
    }

    // Every offered type resolves a position key that its default args carry,
    // so the created entry's override always lands on a real field.
    #[test]
    fn every_placeable_type_has_its_position_key() {
        for ty in placeable_types() {
            let key = position_key(ty);
            let args = crate::editor::form::working_args(ty, None);
            assert!(args.contains_key(key), "{ty} lacks {key}");
        }
    }

    #[test]
    fn items_list_types_then_a_prefab_section() {
        let none = items(&[], false);
        assert!(none.iter().all(|i| matches!(i, MenuItem::Type(_))));

        let prefabs = vec!["door".to_string(), "crate".to_string()];
        let closed = items(&prefabs, false);
        assert_eq!(
            closed.last(),
            Some(&MenuItem::PrefabHeader { open: false }),
            "closed section is one header row"
        );
        assert!(!closed.iter().any(|i| matches!(i, MenuItem::Prefab(_))));

        let open = items(&prefabs, true);
        assert_eq!(
            open.iter()
                .filter(|i| matches!(i, MenuItem::Prefab(_)))
                .count(),
            2,
            "open section lists the world's prefabs"
        );
    }

    #[test]
    fn hit_row_resolves_rows_and_misses_outside() {
        let o = [100.0, 100.0];
        let rows = 5;
        let r1 = row_rect(o, 1);
        assert_eq!(hit_row(r1[0] + 2.0, r1[1] + 2.0, o, rows), Some(1));
        assert_eq!(hit_row(0.0, 0.0, o, rows), None);
        assert!(over(r1[0], r1[1], o, rows));
        assert!(!over(0.0, 0.0, o, rows));
        // Every row sits inside the menu rect.
        let m = menu_rect(o, rows);
        for i in 0..rows {
            let r = row_rect(o, i);
            assert!(r[1] + r[3] <= m[1] + m[3], "row {i} inside the menu");
        }
    }

    #[test]
    fn origin_clamps_fully_on_screen() {
        let vp = [800.0, 600.0];
        let o = origin([790.0, 590.0], 6, vp);
        let s = size(6);
        assert!(o[0] + s[0] <= vp[0]);
        assert!(o[1] + s[1] <= vp[1]);
        assert!(o[1] >= super::super::hud::BAR_H);
    }
}
