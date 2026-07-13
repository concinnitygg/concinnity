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
