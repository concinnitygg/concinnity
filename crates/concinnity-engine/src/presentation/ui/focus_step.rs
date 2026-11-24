// Keyboard and gamepad focus: moving it between regions and confirming the
// focused one.

use super::region_ops::{apply_value, region_covers_canvas, region_rect};
use super::{UiInputSystem, bound, focus, regions};
use crate::components::{
    Binding, NavDirection, SettingCommand, SettingOp, TextLabel, UiValue, ValueOp,
};
use crate::ecs::PipelineContext;
use crate::ecs::asset_id::AssetId;
use crate::settings::registry::SLIDER_STEP_FRACTION;
use alloc::vec::Vec;

impl UiInputSystem {
    // Advance the focus cursor for one directional pulse: Left/Right on a
    // focused value row adjust its setting in place; any other pulse moves the
    // focus to the nearest target, scrolling its panel row into the clip band.
    pub(super) fn step_focus(&mut self, dir: NavDirection, ctx: &mut PipelineContext<'_>) {
        // Follow-label regions with an empty label are hidden menu entries;
        // resolve their label contents in one pass so they never focus.
        let follow_labels: hashbrown::HashSet<AssetId> = self
            .regions
            .iter()
            .filter(|e| self.screens.receives_input(e.screen))
            .filter_map(|e| e.follow.map(|(id, _)| id))
            .collect();
        let empty_labels: hashbrown::HashSet<AssetId> = if follow_labels.is_empty() {
            hashbrown::HashSet::new()
        } else {
            follow_labels
                .iter()
                .copied()
                .filter(|&id| {
                    ctx.get_by_id::<TextLabel>(id)
                        .is_some_and(|l| l.content.is_empty())
                })
                .collect()
        };

        // The focusable candidates: the active screen's live regions minus
        // collapsed, disabled, hidden and full-canvas ones. Scrolled-out rows
        // stay so navigation reaches them.
        let candidates: Vec<focus::Candidate> = self
            .regions
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                let collapsed = e.scroll_row.is_some() && e.hidden;
                let follow_hidden = e.follow.is_some_and(|(id, _)| empty_labels.contains(&id));
                let full_canvas = e.canvas && region_covers_canvas(&e.region);
                self.screens.receives_input(e.screen)
                    && !collapsed
                    && !regions::setting_row_disabled(&self.disabled_rows_cache, &e.region)
                    && !follow_hidden
                    && !full_canvas
            })
            .map(|(i, e)| focus::Candidate {
                index: i,
                rect: region_rect(&e.region),
                action: e.region.action.clone(),
                slider: e.region.slider.clone(),
            })
            .collect();
        let targets = focus::targets(&candidates);

        // Re-anchor the stored rect to the focused region's live position (it
        // reflows with its panel) before navigating from it.
        if let Some(f) = self.focus.as_mut()
            && let Some(t) = targets.iter().find(|t| t.index == f.index)
        {
            f.rect = t.rect;
        }

        // Left/Right on a focused value row adjust its value in place: a
        // setting by the same Prev/Next ops the row's stepper arrows fire (a
        // setting slider steps by a fixed fraction of its range in the settings
        // apply), any other slider by that fraction directly.
        if matches!(dir, NavDirection::Left | NavDirection::Right)
            && let Some(f) = self.focus.as_ref()
            && let Some(t) = targets.iter().find(|t| t.index == f.index)
            && let Some(binding) = &t.adjusts
        {
            let back = dir == NavDirection::Left;
            adjust(binding, back, ctx);
            return;
        }

        let Some(next) = focus::navigate(&targets, self.focus.as_ref(), dir) else {
            return;
        };
        // A clamp onto a vanished region keeps the current anchor unchanged.
        let Some(rect) = targets.iter().find(|t| t.index == next).map(|t| t.rect) else {
            return;
        };
        self.focus = Some(focus::FocusRef { index: next, rect });

        // Scroll the focused row's panel so the row sits inside the clip band
        // (the layout solve at the end of the step clamps the offset).
        if let Some((pi, _)) = self.regions[next].scroll_row
            && let Some(panel) = self.panels.get_mut(pi)
        {
            let band = panel.band;
            let top = rect[1];
            let bottom = rect[1] + rect[3];
            if top < band[1] {
                panel.scroll -= band[1] - top;
            } else if bottom > band[1] + band[3] {
                panel.scroll += bottom - (band[1] + band[3]);
            }
        }
    }
}

// Step `binding` one notch forward, or back: a setting by one option (a
// slider by `SLIDER_STEP_FRACTION` in the settings apply), any other slider
// value by `SLIDER_STEP_FRACTION`, clamped to 0..1.
fn adjust(binding: &Binding, back: bool, ctx: &mut PipelineContext<'_>) {
    if let Binding::Setting(key) = binding {
        ctx.events_mut::<SettingCommand>().send(SettingCommand {
            setting: *key,
            op: if back {
                SettingOp::Prev
            } else {
                SettingOp::Next
            },
            persist: true,
        });
        return;
    }
    let Some(current) = bound::fraction(ctx, binding) else {
        return;
    };
    let step = if back {
        -SLIDER_STEP_FRACTION
    } else {
        SLIDER_STEP_FRACTION
    };
    let value = (current + step).clamp(0.0, 1.0);
    apply_value(binding, &ValueOp::Set(UiValue::Float(value)), ctx);
}
