// Keyboard and gamepad focus: moving it between regions and confirming the
// focused one.

use super::region_ops::{region_covers_canvas, region_rect};
use super::{RegionEntry, UiInputSystem, focus, regions};
use crate::components::{NavDirection, SettingCommand, SettingOp, TextLabel};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, Ref};
use alloc::vec::Vec;

impl UiInputSystem {
    // Whether SettingsSystem disabled this region's setting row at runtime
    // (mirrors the hit-test loop's gating).
    fn row_disabled(&self, entry: &RegionEntry) -> bool {
        regions::setting_row_disabled(&self.disabled_rows_cache, entry.region.action.as_ref())
    }

    // Advance the focus cursor for one directional pulse: Left/Right on a
    // focused value row adjust its setting in place; any other pulse moves the
    // focus to the nearest target, scrolling its panel row into the clip band.
    pub(super) fn step_focus(
        &mut self,
        dir: NavDirection,
        active_screen: Option<AssetId>,
        ctx: &mut PipelineContext<'_>,
    ) {
        // Follow-label regions with an empty label are hidden menu entries;
        // resolve their label contents in one pass so they never focus.
        let follow_labels: hashbrown::HashSet<AssetId> = self
            .regions
            .iter()
            .filter(|e| e.screen == active_screen)
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

        // The focusable candidates: the active screen's live regions, minus
        // collapsed rows, disabled rows, hidden follow-label entries, and
        // full-canvas "press anywhere" regions (those fire from the confirm
        // fallback, not a visible cursor). Out-of-band scrolled rows stay in,
        // so navigation reaches below the fold and scrolls to it.
        let candidates: Vec<focus::Candidate> = self
            .regions
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                let collapsed = e.scroll_row.is_some() && e.hidden;
                let follow_hidden = e.follow.is_some_and(|(id, _)| empty_labels.contains(&id));
                let full_canvas = e.screen.is_some() && region_covers_canvas(&e.region);
                e.screen == active_screen
                    && !collapsed
                    && !self.row_disabled(e)
                    && !follow_hidden
                    && !full_canvas
            })
            .map(|(i, e)| focus::Candidate {
                index: i,
                rect: region_rect(&e.region),
                action: e.region.action.clone(),
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

        // Left/Right on a focused value row adjust the setting in place, the
        // same Prev/Next ops the row's stepper arrows fire (a slider steps by
        // a fixed fraction of its range in the settings apply).
        if matches!(dir, NavDirection::Left | NavDirection::Right)
            && let Some(f) = self.focus.as_ref()
            && let Some(t) = targets.iter().find(|t| t.index == f.index)
            && let Some(key) = t.setting
        {
            let op = match dir {
                NavDirection::Left => SettingOp::Prev,
                _ => SettingOp::Next,
            };
            let value_label = self.regions[f.index].region.label;
            ctx.events_mut::<SettingCommand>().send(SettingCommand {
                setting: key,
                op,
                value_label: value_label.map(Ref::id),
                persist: true,
            });
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
