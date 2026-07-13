// Settings-row plumbing shared by the per-frame SettingCommand drain and
// GraphicsSystem's init-time row captures: label / sprite writers, the
// action-string parsers, and the gray-out helpers for disabled rows.

use crate::assets::{HitRegion, Sprite, TextLabel};
use crate::ecs::PipelineContext;
use crate::ecs::asset_id::AssetId;

// Muted gray applied to the labels of a capability-disabled settings row, so it
// reads as unavailable next to the live rows.
pub(crate) const DISABLED_ROW_COLOR: [f32; 3] = [0.42, 0.42, 0.47];

// The full set of label ids to gray for a set of capability-gated rows: the
// gated value labels themselves (the fallback when a row is not in a scroll
// panel), plus every element of any scroll row that contains one of them, so a
// row dims as a whole (its name + value + stepper glyphs) rather than only its
// value. `rows` is each scroll row's element id list.
pub(crate) fn expand_dim_set(
    gated: &std::collections::HashSet<AssetId>,
    rows: &[Vec<AssetId>],
) -> std::collections::HashSet<AssetId> {
    let mut dim = gated.clone();
    for row in rows {
        if row.iter().any(|id| gated.contains(id)) {
            dim.extend(row.iter().copied());
        }
    }
    dim
}

// Gray a captured set of settings-row labels (or restore their authored
// colors), for a row disabled at runtime: the show_fps / show_vram rows under
// the "Display performance stats" master, and the Resolution row outside
// fullscreen. A free function taking the captured (id, color) list so it can
// run inside the settings drain, where the backend is a live borrow. The
// matching input inertness comes from the `DisabledSettingRows` resource,
// published after the drain.
pub(crate) fn set_rows_grayed(
    ctx: &mut PipelineContext,
    rows: &[(AssetId, [f32; 3])],
    grayed: bool,
) {
    for &(id, orig) in rows {
        let color = if grayed { DISABLED_ROW_COLOR } else { orig };
        for l in ctx.query_mut::<TextLabel>() {
            if l.asset_id == id {
                l.color = color;
                break;
            }
        }
    }
}

// The (label id, authored color) list of every settings row whose key is in
// `keys`, each expanded to its whole scroll row (background + name + value +
// glyphs) so the row grays as a unit; the authored colors drive the restore.
// Runs at init while the HitRegions / ScrollPanels are still present.
pub(crate) fn capture_row_labels(
    ctx: &mut PipelineContext,
    keys: &[&str],
) -> Vec<(AssetId, [f32; 3])> {
    // Collect the rows' value-label ids (every region of a row -- steppers'
    // prev/next or a dropdown's open -- references its value label).
    let mut anchors: std::collections::HashSet<AssetId> = std::collections::HashSet::new();
    for r in ctx.query::<HitRegion>() {
        let Some(rest) = r.action.strip_prefix("setting:") else {
            continue;
        };
        let key = rest.split(':').next().unwrap_or("");
        if keys.contains(&key)
            && let Some(label) = r.label
        {
            anchors.insert(label);
        }
    }
    if anchors.is_empty() {
        return Vec::new();
    }
    let rows: Vec<Vec<AssetId>> = ctx
        .query::<crate::assets::ScrollPanel>()
        .flat_map(|p| p.rows.iter().map(|r| r.elements.clone()))
        .collect();
    let dim = expand_dim_set(&anchors, &rows);
    ctx.query::<TextLabel>()
        .filter(|l| dim.contains(&l.asset_id))
        .map(|l| (l.asset_id, l.color))
        .collect()
}

// Overwrite the text of the TextLabel with the given id, if present.
pub(crate) fn set_label_content(ctx: &mut PipelineContext, id: AssetId, text: &str) {
    for l in ctx.query_mut::<TextLabel>() {
        if l.asset_id == id {
            l.content = text.to_string();
            break;
        }
    }
}

// Set a cycle row's value label from its init-captured id. Used to update a
// row other than the one that was clicked (the master preset relabels the
// quality toggles + render scale; a quality-toggle change relabels the master
// row). The menu's HitRegions are drained after init, so the row -> label map
// is captured once rather than re-queried here.
pub(crate) fn set_cached_row_label(
    labels: &std::collections::HashMap<String, AssetId>,
    ctx: &mut PipelineContext,
    key: &str,
    text: &str,
) {
    if let Some(&id) = labels.get(key) {
        set_label_content(ctx, id, text);
    }
}

// Move the Sprite with the given id to `x` (its left edge), if present. Used to
// slide a slider's handle along its track.
pub(crate) fn set_sprite_x(ctx: &mut PipelineContext, id: AssetId, x: f32) {
    for s in ctx.query_mut::<Sprite>() {
        if s.asset_id == id {
            s.x = x;
            break;
        }
    }
}

// The setting key of a slider drag action (`setting:<key>:drag`), or `None`.
pub(crate) fn slider_key_of(action: &str) -> Option<&str> {
    action
        .strip_prefix("setting:")?
        .strip_suffix(":drag")
        .filter(|k| !k.is_empty())
}

// The setting key of a key-rebind action (`setting:<key>:rebind`), or `None`.
pub(crate) fn rebind_key_of(action: &str) -> Option<&str> {
    action
        .strip_prefix("setting:")?
        .strip_suffix(":rebind")
        .filter(|k| !k.is_empty())
}

// The setting key of a cycle row's value-carrying region, or `None`. A stepper
// row emits a `:next` region (with a matching `:prev` sharing the same value
// label, so capturing `:next` alone maps the key once); a dropdown row emits a
// single `:open` region. Both carry the value label, so matching either maps
// each cycle key to its value label exactly once.
pub(crate) fn cycle_next_key_of(action: &str) -> Option<&str> {
    let rest = action.strip_prefix("setting:")?;
    rest.strip_suffix(":next")
        .or_else(|| rest.strip_suffix(":open"))
        .filter(|k| !k.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // A gated value label pulls in every element of the scroll row that holds
    // it (the row's background, name, value, and stepper glyphs), so the whole
    // row grays out; unrelated rows are untouched.
    #[test]
    fn dim_set_expands_a_gated_value_label_to_its_whole_row() {
        let value = AssetId(3);
        let gated: HashSet<AssetId> = [value].into_iter().collect();
        let rows = vec![
            // Row A: bg, name, prev_glyph, value, next_glyph (value is gated).
            vec![AssetId(1), AssetId(2), value, AssetId(4), AssetId(5)],
            // Row B: an unrelated row.
            vec![AssetId(10), AssetId(11)],
        ];
        let dim = expand_dim_set(&gated, &rows);
        for id in [1, 2, 3, 4, 5] {
            assert!(dim.contains(&AssetId(id)), "row A element {id} should dim");
        }
        assert!(!dim.contains(&AssetId(10)), "an unrelated row stays lit");
        assert!(!dim.contains(&AssetId(11)), "an unrelated row stays lit");
    }

    // With no scroll rows (a hand-authored menu outside a panel), only the gated
    // value label itself dims -- a graceful fallback, not a panic.
    #[test]
    fn dim_set_without_rows_falls_back_to_the_value_label() {
        let gated: HashSet<AssetId> = [AssetId(7)].into_iter().collect();
        assert_eq!(expand_dim_set(&gated, &[]), gated);
    }
}
