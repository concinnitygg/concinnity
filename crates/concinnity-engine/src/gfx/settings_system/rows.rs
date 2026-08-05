// Settings-row plumbing shared by the per-frame SettingCommand drain and
// GraphicsSystem's init-time row captures: label / sprite writers, the
// action-string parsers, and the gray-out helpers for disabled rows.

use crate::assets::{HitRegion, Sprite, TextLabel};
use crate::ecs::PipelineContext;
use crate::ecs::asset_id::AssetId;
use crate::gfx::setting_action;

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
        let Some((key, _)) = setting_action::parse(&r.action) else {
            continue;
        };
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
    crate::ecs::by_asset_id::set_text(ctx, id, text);
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
    crate::ecs::by_asset_id::update::<Sprite>(ctx, Some(id), |s| s.x = x);
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

    // Owns the storage a PipelineContext borrows from. The helpers under test
    // only touch components, so the blob / profile / resources stay empty.
    struct TestWorld {
        components: crate::ecs::ComponentStorage,
        blob: crate::blob::BlobData,
        profile: crate::gfx::profile::FrameProfile,
        resources: crate::ecs::Resources,
        scratch: crate::ecs::Arena,
    }

    impl TestWorld {
        fn new() -> Self {
            Self {
                components: crate::ecs::ComponentStorage::default(),
                blob: crate::blob::BlobData::new(vec![Some(Vec::new())]),
                profile: crate::gfx::profile::FrameProfile::default(),
                resources: crate::ecs::Resources::new(),
                scratch: crate::ecs::Arena::with_capacity(64 * 1024),
            }
        }

        fn push<C: crate::ecs::ComponentSlot>(&mut self, c: C) {
            self.components.push_typed(c);
        }

        fn ctx(&mut self) -> PipelineContext<'_> {
            PipelineContext {
                components: &mut self.components,
                blob: &mut self.blob,
                profile: &mut self.profile,
                resources: &mut self.resources,
                frame: crate::ecs::FrameContext::new(&self.scratch),
            }
        }
    }

    fn label(id: u32, color: [f32; 3]) -> TextLabel {
        TextLabel {
            asset_id: AssetId(id),
            color,
            ..Default::default()
        }
    }

    fn region(action: &str, label: Option<u32>) -> HitRegion {
        HitRegion {
            action: action.to_string(),
            label: label.map(AssetId),
            ..Default::default()
        }
    }

    // Writing a label's content hits the one matching id and leaves the rest
    // alone; an id with no label is a no-op rather than a panic.
    #[test]
    fn set_label_content_writes_only_the_matching_label() {
        let mut world = TestWorld::new();
        world.push(label(1, [1.0; 3]));
        world.push(label(2, [1.0; 3]));
        let mut ctx = world.ctx();

        set_label_content(&mut ctx, AssetId(2), "High");
        let contents: Vec<&str> = ctx
            .query::<TextLabel>()
            .map(|l| l.content.as_str())
            .collect();
        assert_eq!(contents, ["", "High"]);

        set_label_content(&mut ctx, AssetId(9), "Ultra");
        let contents: Vec<&str> = ctx
            .query::<TextLabel>()
            .map(|l| l.content.as_str())
            .collect();
        assert_eq!(contents, ["", "High"], "an absent id changes nothing");
    }

    // Moving a slider handle hits the one matching sprite; an absent id is a
    // no-op.
    #[test]
    fn set_sprite_x_moves_only_the_matching_sprite() {
        let mut world = TestWorld::new();
        world.push(Sprite {
            asset_id: AssetId(1),
            x: 0.0,
            ..Default::default()
        });
        world.push(Sprite {
            asset_id: AssetId(2),
            x: 0.0,
            ..Default::default()
        });
        let mut ctx = world.ctx();

        set_sprite_x(&mut ctx, AssetId(2), 42.0);
        assert_eq!(
            ctx.query::<Sprite>().map(|s| s.x).collect::<Vec<_>>(),
            [0.0, 42.0]
        );

        set_sprite_x(&mut ctx, AssetId(9), 99.0);
        assert_eq!(
            ctx.query::<Sprite>().map(|s| s.x).collect::<Vec<_>>(),
            [0.0, 42.0],
            "an absent id changes nothing"
        );
    }

    // Graying a captured row set recolors every listed label, and ungraying
    // restores each label's own authored color rather than a shared default.
    #[test]
    fn set_rows_grayed_grays_then_restores_authored_colors() {
        let authored = [[0.9, 0.9, 0.9], [0.2, 0.6, 1.0]];
        let mut world = TestWorld::new();
        world.push(label(1, authored[0]));
        world.push(label(2, authored[1]));
        let rows = [(AssetId(1), authored[0]), (AssetId(2), authored[1])];
        let mut ctx = world.ctx();

        set_rows_grayed(&mut ctx, &rows, true);
        assert!(
            ctx.query::<TextLabel>()
                .all(|l| l.color == DISABLED_ROW_COLOR)
        );

        set_rows_grayed(&mut ctx, &rows, false);
        assert_eq!(
            ctx.query::<TextLabel>()
                .map(|l| l.color)
                .collect::<Vec<_>>(),
            authored
        );
    }

    // A capture keys off the rows' regions and returns the whole scroll row's
    // labels with their authored colors, skipping unrelated rows, regions with
    // no label, and non-setting actions.
    #[test]
    fn capture_row_labels_returns_a_matching_rows_labels_and_colors() {
        let mut world = TestWorld::new();
        for id in [1, 2, 3, 4, 5, 20] {
            world.push(label(id, [id as f32 / 100.0; 3]));
        }
        world.push(region("setting:shadows:next", Some(3)));
        world.push(region("setting:other:next", Some(20)));
        world.push(region("quit", Some(1)));
        world.push(region("setting:shadows:prev", None));
        world.push(crate::assets::ScrollPanel {
            rows: vec![
                crate::assets::ScrollRow {
                    elements: vec![AssetId(1), AssetId(2), AssetId(3), AssetId(4), AssetId(5)],
                    ..Default::default()
                },
                crate::assets::ScrollRow {
                    elements: vec![AssetId(20)],
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        let mut ctx = world.ctx();

        let captured = capture_row_labels(&mut ctx, &["shadows"]);
        let expected: Vec<(AssetId, [f32; 3])> = [1, 2, 3, 4, 5]
            .into_iter()
            .map(|id| (AssetId(id), [id as f32 / 100.0; 3]))
            .collect();
        assert_eq!(captured, expected);
    }

    // A key no region carries captures nothing, so a row absent from a world's
    // menu simply has no gray-out set.
    #[test]
    fn capture_row_labels_without_a_matching_key_captures_nothing() {
        let mut world = TestWorld::new();
        world.push(label(1, [1.0; 3]));
        world.push(region("setting:shadows:next", Some(1)));
        let mut ctx = world.ctx();

        assert!(capture_row_labels(&mut ctx, &["resolution"]).is_empty());
    }
}
