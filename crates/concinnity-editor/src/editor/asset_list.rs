// src/editor/asset_list.rs
//
// The editor's "grouped asset list": the world's assets shown under a type
// sub-header with the names indented and alphabetised. Both the Assets browse
// panel (`panel.rs`) and the Template detail panel (`template_panel.rs`) render
// this identical list, so the row model, the grouping, the row geometry / style,
// and the per-row + scrollbar draw all live here once. The two panels differ only
// in their extra chrome (the Assets panel adds hover / selection tints, a
// triple-dot menu, and delete); that stays in `panel.rs`, layered over this base.

use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

use super::theme;
use super::widget::{self, place_rounded, place_sprite};

// Visible rows in the body before it scrolls (shared by the grouped list and the
// Assets panel's combo option list, which reuses this window height).
pub(crate) const MAX_ROWS: usize = 12;

// Row geometry, in window pixels.
pub(crate) const ROW_H: f32 = 28.0;
pub(crate) const PAD: f32 = 8.0;
// Extra left inset for an asset name under its type sub-header.
pub(crate) const INDENT: f32 = 16.0;
pub(crate) const SCROLLBAR_W: f32 = 5.0;
pub(crate) const ROW_LABEL_TOP: f32 = ROW_H * 0.5 - theme::TEXT_HALF;

// Base tints / colours. Interactive tints (hover / selected) belong to the
// Assets panel; these are the shared baseline both lists draw from.
pub(crate) const ROW_TINT: [f32; 4] = [0.13, 0.13, 0.16, 0.0];
pub(crate) const TRACK_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.9];
pub(crate) const THUMB_TINT: [f32; 4] = [0.40, 0.44, 0.56, 0.95];
pub(crate) const LABEL: [f32; 3] = theme::LABEL;
pub(crate) const HEADER_LABEL: [f32; 3] = [0.58, 0.66, 0.80];

// One rendered browse-list row: a type sub-header, or an indented asset name that
// carries the index of its entry (used by the Assets panel's Delete / edit menu;
// unused by the read-only Template list).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ListRow {
    pub is_header: bool,
    pub text: String,
    // The `entries` index for a name row; `None` for a header.
    pub entry: Option<usize>,
}

// The `name` / `type` string of a world-line entry, if present.
fn entry_name(e: &serde_json::Value) -> Option<&str> {
    e.get("name").and_then(|v| v.as_str())
}
fn entry_type(e: &serde_json::Value) -> Option<&str> {
    e.get("type").and_then(|v| v.as_str())
}

// The distinct asset types present in `entries`, alphabetical.
fn distinct_types(entries: &[serde_json::Value]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for e in entries {
        if let Some(ty) = entry_type(e)
            && !out.iter().any(|s| s == ty)
        {
            out.push(ty.to_string());
        }
    }
    out.sort();
    out
}

// Group `entries` into browse rows: a sub-header per type (types alphabetical),
// then an indented row per asset name (names alphabetical within a type) carrying
// its `entries` index. `type_filter` keeps only the matching type's group.
pub(crate) fn grouped_rows(
    entries: &[serde_json::Value],
    type_filter: Option<&str>,
) -> Vec<ListRow> {
    let mut rows = Vec::new();
    for ty in distinct_types(entries) {
        if let Some(f) = type_filter
            && ty != f
        {
            continue;
        }
        let mut named: Vec<(usize, String)> = entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                let name = entry_name(e)?;
                (entry_type(e) == Some(ty.as_str())).then(|| (i, name.to_string()))
            })
            .collect();
        if named.is_empty() {
            continue;
        }
        named.sort_by(|a, b| a.1.cmp(&b.1));
        rows.push(ListRow {
            is_header: true,
            text: ty.clone(),
            entry: None,
        });
        for (i, name) in named {
            rows.push(ListRow {
                is_header: false,
                text: name,
                entry: Some(i),
            });
        }
    }
    rows
}

// Draw one grouped-list row into (`bg_id`, `label_id`) at `rect` with background
// `tint`: a type sub-header (no indent, header colour) or an indented asset name
// (name colour). The caller chooses the tint (a plain read-only row, or the
// Assets panel's hover / selected tint). A name row's background draws as a
// rounded highlight inset from the row rect; a header's spans the full row.
pub(crate) fn place_row(
    world: &mut World,
    bg_id: AssetId,
    label_id: AssetId,
    row: &ListRow,
    rect: [f32; 4],
    tint: [f32; 4],
) {
    if row.is_header {
        place_sprite(world, bg_id, rect, tint, true);
    } else {
        place_rounded(
            world,
            bg_id,
            theme::highlight_rect(rect),
            tint,
            theme::CONTROL_RADIUS,
            true,
        );
    }
    let (x_off, color) = if row.is_header {
        (PAD, HEADER_LABEL)
    } else {
        (PAD + INDENT, LABEL)
    };
    widget::place_message(
        world,
        label_id,
        [
            rect[0] + x_off,
            rect[1] + ROW_LABEL_TOP,
            (rect[2] - x_off - PAD).max(0.0),
            widget::LINE_H,
        ],
        &row.text,
        color,
        true,
    );
}

// A simple non-interactive scrollbar sizing the visible `window` against
// `total`, drawn down the right edge from `top_y`. `right_x` is the body's right
// edge; the bar sits just inside it. Shown only when the body overflows the window.
pub(crate) fn layout_scrollbar(
    world: &mut World,
    ids: (AssetId, AssetId),
    total: usize,
    scroll: usize,
    window: usize,
    right_x: f32,
    top_y: f32,
) {
    let (track_id, thumb_id) = ids;
    if total <= window {
        return;
    }
    let x = right_x - SCROLLBAR_W - 2.0;
    let track_h = window as f32 * ROW_H;
    place_rounded(
        world,
        track_id,
        [x, top_y, SCROLLBAR_W, track_h],
        TRACK_TINT,
        SCROLLBAR_W * 0.5,
        true,
    );
    let frac_visible = window as f32 / total as f32;
    let thumb_h = (track_h * frac_visible).max(20.0);
    let max_scroll = (total - window) as f32;
    let t = if max_scroll > 0.0 {
        scroll.min(total - window) as f32 / max_scroll
    } else {
        0.0
    };
    let thumb_y = top_y + t * (track_h - thumb_h);
    place_rounded(
        world,
        thumb_id,
        [x, thumb_y, SCROLLBAR_W, thumb_h],
        THUMB_TINT,
        SCROLLBAR_W * 0.5,
        true,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Sprite, TextLabel};

    fn entry(name: &str, ty: &str) -> serde_json::Value {
        serde_json::json!({"name": name, "type": ty, "args": {}})
    }

    // Types sort alphabetically; names sort alphabetically within their type; name
    // rows carry their original `entries` index (not the sorted position) so the
    // Assets panel's edit / delete still target the right entry.
    #[test]
    fn grouped_rows_sort_types_and_names_and_keep_indices() {
        let entries = vec![
            entry("zeta", "PointLight"),
            entry("beta", "Decal"),
            entry("alpha", "PointLight"),
        ];
        let rows = grouped_rows(&entries, None);
        assert!(rows[0].is_header && rows[0].text == "Decal");
        assert_eq!((rows[1].text.as_str(), rows[1].entry), ("beta", Some(1)));
        assert!(rows[2].is_header && rows[2].text == "PointLight");
        // alpha before zeta (alphabetical), each keeping its original index.
        assert_eq!((rows[3].text.as_str(), rows[3].entry), ("alpha", Some(2)));
        assert_eq!((rows[4].text.as_str(), rows[4].entry), ("zeta", Some(0)));
    }

    #[test]
    fn grouped_rows_filter_keeps_one_group() {
        let entries = vec![
            entry("a", "PointLight"),
            entry("b", "Decal"),
            entry("c", "PointLight"),
        ];
        let rows = grouped_rows(&entries, Some("PointLight"));
        assert_eq!(rows.len(), 3, "one header + two names");
        assert!(rows[0].is_header && rows[0].text == "PointLight");
    }

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
        }
        world
    }

    // A header row draws its caption at the base pad in the header colour; a name
    // row indents and uses the name colour.
    #[test]
    fn place_row_indents_names_and_colours_headers() {
        let mut world = world_with(&[AssetId(1), AssetId(2)]);
        let hdr = ListRow {
            is_header: true,
            text: "PointLight".into(),
            entry: None,
        };
        place_row(
            &mut world,
            AssetId(1),
            AssetId(1),
            &hdr,
            [10.0, 20.0, 300.0, ROW_H],
            ROW_TINT,
        );
        let l = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == AssetId(1))
            .unwrap();
        assert_eq!(l.x, 10.0 + PAD, "header caption sits at the base pad");
        assert_eq!(l.color, HEADER_LABEL);

        let name = ListRow {
            is_header: false,
            text: "lamp".into(),
            entry: Some(0),
        };
        place_row(
            &mut world,
            AssetId(2),
            AssetId(2),
            &name,
            [10.0, 20.0, 300.0, ROW_H],
            ROW_TINT,
        );
        let l = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == AssetId(2))
            .unwrap();
        assert_eq!(l.x, 10.0 + PAD + INDENT, "name caption is indented");
        assert_eq!(l.color, LABEL);
    }

    // The scrollbar draws only when the row total overflows the window.
    #[test]
    fn scrollbar_shows_only_on_overflow() {
        let mut world = world_with(&[AssetId(1), AssetId(2)]);
        for s in world.query_mut::<Sprite>() {
            s.visible = false;
        }
        // Fits the window: the bar is left untouched (still hidden).
        layout_scrollbar(
            &mut world,
            (AssetId(1), AssetId(2)),
            MAX_ROWS,
            0,
            MAX_ROWS,
            300.0,
            40.0,
        );
        assert!(
            world.query::<Sprite>().all(|s| !s.visible),
            "no bar when the list fits"
        );
        // Overflows: the track + thumb are shown.
        layout_scrollbar(
            &mut world,
            (AssetId(1), AssetId(2)),
            MAX_ROWS + 5,
            0,
            MAX_ROWS,
            300.0,
            40.0,
        );
        assert!(
            world.query::<Sprite>().all(|s| s.visible),
            "track + thumb show on overflow"
        );
    }
}
