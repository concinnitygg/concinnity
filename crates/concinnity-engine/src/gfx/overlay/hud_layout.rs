// Per-frame HUD label placement: LayoutContainer reflow and the DebugHud /
// StatHud chip anchoring. All of it needs the loaded font metrics, which is
// why it runs in the overlay build rather than the HUD systems themselves.

use crate::assets::{LabelBox, LayoutContainer, TextLabel};
use crate::ecs::PipelineContext;
use crate::ecs::asset_id::AssetId;
use crate::gfx::text;

// Reposition the labels owned by every visible `LayoutContainer`. This runs in
// the overlay build because measuring a label needs the loaded font metrics;
// the resolved origin is written back into each label so `build_text_calls`
// then draws it in place.
pub(super) fn apply_label_layout(
    ctx: &mut PipelineContext,
    loaded_fonts: &std::collections::HashMap<crate::ecs::FontHandle, text::LoadedFont>,
) {
    let containers: Vec<LayoutContainer> = ctx
        .query::<LayoutContainer>()
        .filter(|c| c.visible)
        .cloned()
        .collect();
    if containers.is_empty() {
        return;
    }
    // Measure every label once, keyed by id.
    let mut boxes: std::collections::HashMap<AssetId, LabelBox> = std::collections::HashMap::new();
    for label in ctx.query::<TextLabel>() {
        if let Some(b) = text::measure_label_box(label, loaded_fonts) {
            boxes.insert(label.asset_id, b);
        }
    }
    // Resolve placements, then write them back into the labels.
    let placements: Vec<_> = containers
        .iter()
        .flat_map(|c| c.layout(|id| boxes.get(&id).copied()))
        .collect();
    for p in placements {
        for label in ctx.query_mut::<TextLabel>() {
            if label.asset_id == p.id {
                label.x = p.x;
                label.y = p.y;
                break;
            }
        }
    }
}

// Anchor the DebugHud chips to the top-right of the window, stacked downward in
// id order (cursor, passes, camera). A blank chip (the HUD hidden, or a stat the
// backend cannot supply) reserves no space. Measured each frame so a chip that
// changes width -- the multi-line passes chip in particular -- re-anchors
// flush-right. HUD labels are literal window pixels (no overlay scaling), so
// this positions in window space directly.
//
// Timing: DebugHudSystem writes each chip's content AFTER the overlay build in
// the schedule, so the content present here is what DebugHudSystem wrote last
// tick -- which is exactly the content the draw list built in this same step
// renders. Measuring it (rather than this tick's not-yet-written content) is
// therefore correct: the measured width always matches the content being
// drawn, so the stack never mismatches within a frame.
pub(super) fn position_debug_hud(
    ctx: &mut PipelineContext,
    chip_ids: &[AssetId],
    loaded_fonts: &std::collections::HashMap<crate::ecs::FontHandle, text::LoadedFont>,
    win_w: f32,
) {
    if chip_ids.is_empty() || win_w <= 0.0 {
        return;
    }
    const MARGIN: f32 = 10.0;
    const GAP: f32 = 6.0;
    let mut y = MARGIN;
    for &id in chip_ids {
        // Measure the chip's box from its current (last-frame) content; skip a
        // blank chip so a hidden readout reserves no vertical space.
        let measured = ctx
            .query::<TextLabel>()
            .find(|l| l.asset_id == id && !l.content.is_empty())
            .and_then(|l| text::measure_label_box(l, loaded_fonts));
        let Some(b) = measured else {
            continue;
        };
        // Right-anchor the box (its left edge sits `pad` left of the text
        // origin) and place its top at the running y.
        let x = (win_w - MARGIN - b.w + b.pad).max(MARGIN);
        for l in ctx.query_mut::<TextLabel>() {
            if l.asset_id == id {
                l.x = x;
                l.y = y + b.top_inset;
                break;
            }
        }
        y += b.h + GAP;
    }
}

// Pack the StatHud chips (fps, vram, ev, edr) into a tight strip from the
// top-left of the window, left to right with a small gap. A blank chip (a
// readout hidden by the video settings, or a stat the world/display does not
// supply) reserves no width, so the strip stays as narrow as the live content
// and hidden chips leave no hole. Measured each frame so a chip that changes
// width re-packs its neighbours. HUD labels are literal window pixels (no
// overlay scaling), so this positions in window space directly.
pub(super) fn position_stat_hud(
    ctx: &mut PipelineContext,
    chip_ids: &[AssetId],
    loaded_fonts: &std::collections::HashMap<crate::ecs::FontHandle, text::LoadedFont>,
) {
    if chip_ids.is_empty() {
        return;
    }
    const MARGIN: f32 = 10.0;
    const GAP: f32 = 4.0;
    let mut x = MARGIN;
    for &id in chip_ids {
        // Measure the chip's box from its current (last-frame) content; skip a
        // blank chip so a hidden readout reserves no horizontal space.
        let measured = ctx
            .query::<TextLabel>()
            .find(|l| l.asset_id == id && !l.content.is_empty())
            .and_then(|l| text::measure_label_box(l, loaded_fonts));
        let Some(b) = measured else {
            continue;
        };
        // The box's left edge sits `pad` left of the text origin, so offset the
        // origin by `pad` to line the box up at the running x.
        for l in ctx.query_mut::<TextLabel>() {
            if l.asset_id == id {
                l.x = x + b.pad;
                l.y = MARGIN + b.top_inset;
                break;
            }
        }
        x += b.w + GAP;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Justify, LayoutRow, SpriteFit, TextAlign};
    use crate::blob::BlobData;
    use crate::ecs::{ComponentSlot, ComponentStorage, FontHandle, Resources};
    use crate::gfx::profile::FrameProfile;

    const FONT: FontHandle = FontHandle(0);
    // An authored position no layout pass should ever produce, so an untouched
    // label is unmistakable.
    const SENTINEL: f32 = -999.0;

    fn make_glyph(advance_px: f32) -> crate::build::font::GlyphMetrics {
        crate::build::font::GlyphMetrics {
            char_code: 0,
            atlas_x: 0,
            atlas_y: 0,
            atlas_w: 8,
            atlas_h: 12,
            advance_px,
            bearing_x: 0.0,
            bearing_y: 12.0,
        }
    }

    // A fixed-width synthetic font (every glyph 10px in a 16px em, caps 12px
    // tall) makes the measured boxes exact: a 1-line unpadded chip measures
    // 10px per char wide, 12px tall, with a -2px top inset.
    fn loaded_fonts() -> std::collections::HashMap<FontHandle, text::LoadedFont> {
        let metrics: std::collections::HashMap<u32, crate::build::font::GlyphMetrics> = ('a'..='z')
            .chain('A'..='Z')
            .map(|c| (c as u32, make_glyph(10.0)))
            .collect();
        let cap_px = text::derive_cap_px(&metrics, 16.0);
        std::collections::HashMap::from([(
            FONT,
            text::LoadedFont {
                atlas_slot: 0,
                cap_px,
                metrics,
                atlas_w: 128,
                atlas_h: 128,
                size_px: 16.0,
                supersample: 1.0,
            },
        )])
    }

    fn chip(id: AssetId, content: &str) -> TextLabel {
        TextLabel {
            asset_id: id,
            font: Some(FONT),
            content: content.to_string(),
            x: SENTINEL,
            y: SENTINEL,
            color: [1.0, 1.0, 1.0],
            scale: 1.0,
            centered: false,
            align: TextAlign::Left,
            fit: SpriteFit::Fit,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            screen: None,
        }
    }

    fn row(cols: &[AssetId]) -> LayoutRow {
        LayoutRow {
            cols: cols.to_vec(),
            justify: Justify::Left,
        }
    }

    // Owns the storage a PipelineContext borrows from. Label placement reads no
    // payloads, so the blob and resources stay empty.
    struct TestWorld {
        components: ComponentStorage,
        blob: BlobData,
        profile: FrameProfile,
        resources: Resources,
    }

    impl TestWorld {
        fn new() -> Self {
            Self {
                components: ComponentStorage::default(),
                blob: BlobData::new(vec![Some(Vec::new())]),
                profile: FrameProfile::default(),
                resources: Resources::new(),
            }
        }

        fn push<C: ComponentSlot>(&mut self, c: C) {
            self.components.push_typed(c);
        }

        fn ctx(&mut self) -> PipelineContext<'_> {
            PipelineContext {
                components: &mut self.components,
                blob: &mut self.blob,
                profile: &mut self.profile,
                resources: &mut self.resources,
            }
        }

        fn label_at(&mut self, id: AssetId) -> (f32, f32) {
            let ctx = self.ctx();
            let l = ctx
                .query::<TextLabel>()
                .find(|l| l.asset_id == id)
                .expect("label pushed");
            (l.x, l.y)
        }
    }

    // Rows lay out left to right from the container origin and stack downward;
    // the resolved origin is written back into each label.
    #[test]
    fn apply_label_layout_places_rows_from_the_container_origin() {
        let mut w = TestWorld::new();
        w.push(chip(AssetId(1), "aa"));
        w.push(chip(AssetId(2), "aaaa"));
        w.push(chip(AssetId(3), "aaaaaa"));
        w.push(LayoutContainer {
            x: 100.0,
            y: 50.0,
            rows: vec![row(&[AssetId(1), AssetId(2)]), row(&[AssetId(3)])],
            ..Default::default()
        });
        apply_label_layout(&mut w.ctx(), &loaded_fonts());
        assert_eq!(w.label_at(AssetId(1)), (100.0, 48.0));
        // The second chip clears the first's 20px box plus the 6px column gap.
        assert_eq!(w.label_at(AssetId(2)), (126.0, 48.0));
        // Row two drops by the first row's box height (12) plus the row gap.
        assert_eq!(w.label_at(AssetId(3)), (100.0, 66.0));
    }

    // A container that is not visible leaves its labels where they are.
    #[test]
    fn apply_label_layout_ignores_hidden_containers() {
        let mut w = TestWorld::new();
        w.push(chip(AssetId(1), "aa"));
        w.push(LayoutContainer {
            rows: vec![row(&[AssetId(1)])],
            visible: false,
            ..Default::default()
        });
        apply_label_layout(&mut w.ctx(), &loaded_fonts());
        assert_eq!(w.label_at(AssetId(1)), (SENTINEL, SENTINEL));
    }

    // A label that cannot be measured is dropped from the layout: it reserves no
    // width and keeps its own position, so its row neighbours close the gap.
    #[test]
    fn apply_label_layout_drops_labels_it_cannot_measure() {
        let mut w = TestWorld::new();
        w.push(chip(AssetId(1), "aa"));
        let mut hidden = chip(AssetId(2), "aaaa");
        hidden.visible = false;
        w.push(hidden);
        let mut orphan = chip(AssetId(3), "aaaa");
        orphan.font = Some(FontHandle(99));
        w.push(orphan);
        w.push(chip(AssetId(4), "aaaaaa"));
        w.push(LayoutContainer {
            x: 100.0,
            y: 50.0,
            rows: vec![row(&[AssetId(1), AssetId(2), AssetId(3), AssetId(4)])],
            ..Default::default()
        });
        apply_label_layout(&mut w.ctx(), &loaded_fonts());
        assert_eq!(w.label_at(AssetId(1)), (100.0, 48.0));
        // The last chip packs against the first, as if the two dropped ones were absent.
        assert_eq!(w.label_at(AssetId(4)), (126.0, 48.0));
        assert_eq!(w.label_at(AssetId(2)), (SENTINEL, SENTINEL));
        assert_eq!(w.label_at(AssetId(3)), (SENTINEL, SENTINEL));
    }

    // Chips anchor flush with the window's right margin, stacked downward in id
    // order, so a wider chip still lines its right edge up with the rest.
    #[test]
    fn position_debug_hud_right_anchors_chips_top_down() {
        let mut w = TestWorld::new();
        w.push(chip(AssetId(1), "aa"));
        w.push(chip(AssetId(2), "aaaa"));
        position_debug_hud(
            &mut w.ctx(),
            &[AssetId(1), AssetId(2)],
            &loaded_fonts(),
            1000.0,
        );
        assert_eq!(w.label_at(AssetId(1)), (970.0, 8.0));
        // Twice as wide, so it starts further left but ends on the same edge, one
        // box height (12) plus the gap (6) below.
        assert_eq!(w.label_at(AssetId(2)), (950.0, 26.0));
    }

    // A blank chip (a hidden readout, or a stat the backend cannot supply) and one
    // whose font is not loaded reserve no vertical space, so the chips below them
    // do not shift.
    #[test]
    fn position_debug_hud_skips_chips_it_cannot_measure() {
        let mut w = TestWorld::new();
        w.push(chip(AssetId(1), "aa"));
        w.push(chip(AssetId(2), ""));
        let mut orphan = chip(AssetId(3), "aaaa");
        orphan.font = Some(FontHandle(99));
        w.push(orphan);
        w.push(chip(AssetId(4), "aaaa"));
        position_debug_hud(
            &mut w.ctx(),
            &[AssetId(1), AssetId(2), AssetId(3), AssetId(4)],
            &loaded_fonts(),
            1000.0,
        );
        // The last chip lands where it would with only the first chip above it.
        assert_eq!(w.label_at(AssetId(4)), (950.0, 26.0));
        assert_eq!(w.label_at(AssetId(2)), (SENTINEL, SENTINEL));
        assert_eq!(w.label_at(AssetId(3)), (SENTINEL, SENTINEL));
    }

    // A chip wider than the window stops at the left margin instead of running off
    // the left edge.
    #[test]
    fn position_debug_hud_clamps_a_chip_wider_than_the_window() {
        let mut w = TestWorld::new();
        w.push(chip(AssetId(1), "aa"));
        position_debug_hud(&mut w.ctx(), &[AssetId(1)], &loaded_fonts(), 15.0);
        assert_eq!(w.label_at(AssetId(1)).0, 10.0);
    }

    // With no chips or no window there is nothing to anchor against.
    #[test]
    fn position_debug_hud_ignores_an_empty_chip_list_or_window() {
        let mut w = TestWorld::new();
        w.push(chip(AssetId(1), "aa"));
        position_debug_hud(&mut w.ctx(), &[], &loaded_fonts(), 1000.0);
        assert_eq!(w.label_at(AssetId(1)), (SENTINEL, SENTINEL));
        position_debug_hud(&mut w.ctx(), &[AssetId(1)], &loaded_fonts(), 0.0);
        assert_eq!(w.label_at(AssetId(1)), (SENTINEL, SENTINEL));
    }

    // Chips pack into a strip from the top-left, each box's left edge at the
    // running x (so the text origin is inset by the chip's own padding).
    #[test]
    fn position_stat_hud_packs_chips_left_to_right() {
        let mut w = TestWorld::new();
        let mut first = chip(AssetId(1), "aa");
        first.padding = 4.0;
        let mut second = chip(AssetId(2), "aaaa");
        second.padding = 4.0;
        w.push(first);
        w.push(second);
        position_stat_hud(&mut w.ctx(), &[AssetId(1), AssetId(2)], &loaded_fonts());
        // Box 1 at the margin, its origin 4px in.
        assert_eq!(w.label_at(AssetId(1)), (14.0, 12.0));
        // Box 2 clears box 1 (28px wide) plus the 4px gap.
        assert_eq!(w.label_at(AssetId(2)), (46.0, 12.0));
    }

    // A blank chip reserves no width, so the strip stays tight and leaves no hole.
    #[test]
    fn position_stat_hud_skips_blank_chips() {
        let mut w = TestWorld::new();
        w.push(chip(AssetId(1), "aa"));
        w.push(chip(AssetId(2), ""));
        w.push(chip(AssetId(3), "aaaa"));
        position_stat_hud(
            &mut w.ctx(),
            &[AssetId(1), AssetId(2), AssetId(3)],
            &loaded_fonts(),
        );
        assert_eq!(w.label_at(AssetId(1)), (10.0, 8.0));
        // The third chip lands where the second would have, had it any content.
        assert_eq!(w.label_at(AssetId(3)), (34.0, 8.0));
        assert_eq!(w.label_at(AssetId(2)), (SENTINEL, SENTINEL));
    }

    // With no chips there is nothing to pack.
    #[test]
    fn position_stat_hud_ignores_an_empty_chip_list() {
        let mut w = TestWorld::new();
        w.push(chip(AssetId(1), "aa"));
        position_stat_hud(&mut w.ctx(), &[], &loaded_fonts());
        assert_eq!(w.label_at(AssetId(1)), (SENTINEL, SENTINEL));
    }
}
