// The HitRegion pass: hover styling and click / confirm dispatch for every
// region, with the clicks that need the whole system (group toggles, rebind
// capture, dropdown opens) returned for the caller to apply.

use std::collections::{HashMap, HashSet};

use concinnity_core::components::{FrameInput, SettingVerb, SpriteFit, TextLabel, UiAction};
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{FrameVec, PipelineContext, StepResult};
use concinnity_core::gfx::overlay::OverlayTransform;
use concinnity_core::settings::SettingKey;

use super::intent::UiIntent;
use super::{
    OpenRequest, UiInputSystem, fire_action, point_in_rect, region_covers_canvas, region_rect,
    set_label_style,
};

// The per-frame state every region's hit-test reads.
pub(super) struct RegionFrame<'a> {
    // The topmost input-capturing screen; only its regions (or, with none,
    // the screen-less regions) react.
    active_screen: Option<AssetId>,
    // The scrollbar thumb is being dragged, so no region reacts.
    thumb_active: bool,
    // Reference-canvas mappings for screen-owned regions, chosen by `fit`.
    overlay: OverlayTransform,
    overlay_bottom: OverlayTransform,
    overlay_cover: OverlayTransform,
    // Each panel's content band (reference space), indexed like `panels`.
    panel_bands: FrameVec<'a, [f32; 4]>,
}

impl<'a> RegionFrame<'a> {
    // `overlay` is the default fit mapping; the bottom-anchored and cover
    // mappings a region may opt into derive from the same viewport.
    pub(super) fn new(
        viewport: [f32; 2],
        active_screen: Option<AssetId>,
        thumb_active: bool,
        overlay: OverlayTransform,
        panel_bands: FrameVec<'a, [f32; 4]>,
    ) -> Self {
        Self {
            active_screen,
            thumb_active,
            overlay,
            overlay_bottom: OverlayTransform::bottom_anchored_from_viewport(viewport),
            overlay_cover: OverlayTransform::cover_from_viewport(viewport),
            panel_bands,
        }
    }

    // The cursor in a region's own space: window pixels for a screen-less
    // region, reference space through its `fit` otherwise.
    fn cursor_for(&self, screen: Option<AssetId>, fit: SpriteFit, mx: f32, my: f32) -> (f32, f32) {
        if screen.is_none() {
            return (mx, my);
        }
        match fit {
            SpriteFit::Bottom => self.overlay_bottom.inverse(mx, my),
            SpriteFit::Cover => self.overlay_cover.inverse(mx, my),
            SpriteFit::Fit => self.overlay.inverse(mx, my),
        }
    }
}

// What the region pass asks the caller to do after it releases the regions.
#[derive(Default)]
pub(super) struct RegionOutcome {
    // A group header was clicked: flip that group on the active panel.
    pub(super) toggle_group: Option<usize>,
    // A rebind row was clicked: capture for this setting key and value label.
    pub(super) start_capture: Option<(SettingKey, Option<AssetId>)>,
    // A dropdown row was clicked: open its floating list.
    pub(super) start_open: Option<OpenRequest>,
    // A fired action ended the step (e.g. Quit); nothing else applies.
    pub(super) fired: Option<StepResult>,
}

// Why a region may be unable to hover or fire this frame.
#[derive(Debug, Default, Clone, Copy)]
struct RegionGate {
    thumb_active: bool,
    slider: bool,
    // The focus cursor is on this region (a focused slider still highlights).
    pad_focused: bool,
    screen_matches: bool,
    // Its scroll-content row is collapsed.
    collapsed_row: bool,
    // The engine disabled its setting row at runtime.
    disabled: bool,
    // It follows a label that is empty or gone.
    follow_inert: bool,
}

// Whether a region neither hovers nor fires this frame.
fn region_inert(gate: RegionGate) -> bool {
    gate.thumb_active
        || (gate.slider && !gate.pad_focused)
        || !gate.screen_matches
        || gate.collapsed_row
        || gate.disabled
        || gate.follow_inert
}

// A follow-label region's synced y and whether it is inert: it tracks its
// label and goes inert while the label is empty or missing.
fn follow_sync(
    follow: Option<(AssetId, f32)>,
    labels: &HashMap<AssetId, (f32, bool)>,
) -> (Option<f32>, bool) {
    let Some((label_id, offset)) = follow else {
        return (None, false);
    };
    match labels.get(&label_id) {
        Some(&(ly, empty)) => (Some(ly + offset), empty),
        None => (None, true),
    }
}

// Whether a region's setting row is in the runtime-disabled set.
pub(super) fn setting_row_disabled(
    disabled_rows: &HashSet<SettingKey>,
    action: Option<&UiAction>,
) -> bool {
    matches!(action, Some(UiAction::Setting { key, .. }) if disabled_rows.contains(key))
}

// A region's `(hovered, fire)`. While the focus cursor is set it owns the hover
// slot: the focused region styles and fires on confirm, and the mouse's region
// does neither. Otherwise the mouse hovers, a click fires, and an unfocused
// confirm may fall through to a full-canvas region.
fn hover_and_fire(
    has_focus: bool,
    pad_focused: bool,
    mouse_hovered: bool,
    clicked: bool,
    confirm: bool,
    fallback_fire: bool,
) -> (bool, bool) {
    if has_focus {
        (pad_focused, pad_focused && confirm)
    } else {
        (mouse_hovered, (mouse_hovered && clicked) || fallback_fire)
    }
}

impl UiInputSystem {
    // Refresh the owned copy of `DisabledSettingRows` only when the published
    // set changed, so the region pass reads it without a per-frame clone.
    pub(super) fn refresh_disabled_rows(&mut self, ctx: &PipelineContext) {
        let changed = match ctx.resource::<crate::ecs::DisabledSettingRows>() {
            Some(d) => d.0 != self.disabled_rows_cache,
            None => !self.disabled_rows_cache.is_empty(),
        };
        if changed {
            self.disabled_rows_cache = ctx
                .resource::<crate::ecs::DisabledSettingRows>()
                .map(|d| d.0.clone())
                .unwrap_or_default();
        }
    }

    // Resolve each followed label's (y, is-empty) in one query pass, so the
    // region pass reads a map instead of scanning every TextLabel per region.
    fn resolve_follow_labels(&mut self, ctx: &PipelineContext) {
        self.follow_labels.clear();
        if self.follow_label_ids.is_empty() {
            return;
        }
        for l in ctx.query::<TextLabel>() {
            if self.follow_label_ids.contains(&l.asset_id) {
                self.follow_labels
                    .entry(l.asset_id)
                    .or_insert((l.y, l.content.is_empty()));
            }
        }
    }

    // Hover-style and fire every region for one frame. An inert region that was
    // hovered has its label style restored so the hover never strands. A fired
    // action that ends the step returns at once, leaving later regions as they
    // were.
    pub(super) fn hit_test_regions(
        &mut self,
        input: &FrameInput,
        intent: &UiIntent,
        frame: RegionFrame,
        ctx: &mut PipelineContext,
    ) -> RegionOutcome {
        self.resolve_follow_labels(ctx);
        let (mx, my) = (input.mouse_x, input.mouse_y);
        let [vw, vh] = input.viewport;
        let focus_index = self.focus.as_ref().map(|f| f.index);
        let confirm_fallback = intent.confirm && focus_index.is_none();
        let mut confirm_used = false;
        let mut outcome = RegionOutcome::default();

        for (i, entry) in self.regions.iter_mut().enumerate() {
            let (follow_y, follow_inert) = follow_sync(entry.follow, &self.follow_labels);
            if let Some(y) = follow_y {
                entry.region.y = y;
            }
            let pad_focused = focus_index == Some(i);
            let gate = RegionGate {
                thumb_active: frame.thumb_active,
                slider: entry.slider_key.is_some(),
                pad_focused,
                screen_matches: entry.screen == frame.active_screen,
                collapsed_row: entry.scroll_row.is_some() && entry.hidden,
                disabled: setting_row_disabled(
                    &self.disabled_rows_cache,
                    entry.region.action.as_ref(),
                ),
                follow_inert,
            };
            if region_inert(gate) {
                if entry.was_hovered {
                    set_label_style(
                        ctx,
                        entry.region.label,
                        entry.original_color,
                        entry.original_scale,
                    );
                    entry.was_hovered = false;
                }
                continue;
            }

            // A region spanning the whole reference canvas covers the full
            // window, so a full-canvas advance region catches letterbox clicks.
            let full_window = entry.screen.is_some() && region_covers_canvas(&entry.region);
            let (qx, qy) = frame.cursor_for(entry.screen, entry.fit, mx, my);
            let r = &entry.region;
            let mut mouse_hovered = if full_window {
                mx >= 0.0 && mx < vw && my >= 0.0 && my < vh
            } else {
                point_in_rect(qx, qy, region_rect(r))
            };
            // A scroll-content region only counts as hovered inside its band, so
            // a row scrolled past the edge does not catch clicks over the chrome.
            if let Some((pi, _)) = entry.scroll_row
                && let Some(band) = frame.panel_bands.get(pi)
            {
                mouse_hovered = mouse_hovered && point_in_rect(qx, qy, *band);
            }
            let fallback_fire = confirm_fallback && full_window && !confirm_used;
            let (hovered, fire) = hover_and_fire(
                focus_index.is_some(),
                pad_focused,
                mouse_hovered,
                input.left_click,
                intent.confirm,
                fallback_fire,
            );

            if hovered && !entry.was_hovered {
                set_label_style(ctx, r.label, r.hover_color, r.hover_scale);
            } else if !hovered && entry.was_hovered {
                set_label_style(ctx, r.label, entry.original_color, entry.original_scale);
            }
            entry.was_hovered = hovered;

            if !fire {
                continue;
            }
            if fallback_fire {
                confirm_used = true;
            }
            match &r.action {
                None => {}
                Some(UiAction::GroupToggle(gid)) => outcome.toggle_group = Some(*gid),
                Some(UiAction::Setting {
                    key,
                    verb: SettingVerb::Rebind,
                }) => outcome.start_capture = Some((*key, r.label)),
                Some(UiAction::Setting {
                    key,
                    verb: SettingVerb::Open,
                }) => {
                    // Snapshot the control rect and the row's un-hovered value style.
                    outcome.start_open = Some(OpenRequest {
                        setting: *key,
                        value_label: r.label,
                        anchor: region_rect(r),
                        screen: entry.screen,
                        color: entry.original_color,
                        scale: entry.original_scale,
                    });
                }
                Some(action) => {
                    if let Some(result) = fire_action(action, r.label, ctx) {
                        outcome.fired = Some(result);
                        return outcome;
                    }
                }
            }
        }
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A live region: on the active screen and gated by nothing.
    fn live() -> RegionGate {
        RegionGate {
            screen_matches: true,
            ..Default::default()
        }
    }

    #[test]
    fn a_live_region_is_not_inert() {
        assert!(!region_inert(live()));
    }

    #[test]
    fn each_gate_makes_a_region_inert() {
        let gates = [
            RegionGate {
                thumb_active: true,
                ..live()
            },
            RegionGate {
                slider: true,
                ..live()
            },
            RegionGate {
                screen_matches: false,
                ..live()
            },
            RegionGate {
                collapsed_row: true,
                ..live()
            },
            RegionGate {
                disabled: true,
                ..live()
            },
            RegionGate {
                follow_inert: true,
                ..live()
            },
        ];
        for gate in gates {
            assert!(region_inert(gate), "{gate:?}");
        }
    }

    #[test]
    fn a_focused_slider_stays_live() {
        let gate = RegionGate {
            slider: true,
            pad_focused: true,
            ..live()
        };
        assert!(!region_inert(gate));
    }

    #[test]
    fn follow_sync_tracks_the_label_and_goes_inert_when_empty_or_missing() {
        let labels = HashMap::from([(AssetId(1), (40.0, false)), (AssetId(2), (60.0, true))]);
        assert_eq!(follow_sync(None, &labels), (None, false));
        assert_eq!(
            follow_sync(Some((AssetId(1), 5.0)), &labels),
            (Some(45.0), false)
        );
        assert_eq!(
            follow_sync(Some((AssetId(2), 5.0)), &labels),
            (Some(65.0), true)
        );
        assert_eq!(follow_sync(Some((AssetId(3), 5.0)), &labels), (None, true));
    }

    #[test]
    fn setting_row_disabled_matches_the_setting_key() {
        let rows = HashSet::from([SettingKey::ShowFps]);
        let row = |key| UiAction::Setting {
            key,
            verb: SettingVerb::Next,
        };
        assert!(setting_row_disabled(&rows, Some(&row(SettingKey::ShowFps))));
        assert!(!setting_row_disabled(&rows, Some(&row(SettingKey::Vsync))));
        assert!(!setting_row_disabled(&rows, Some(&UiAction::Quit)));
        assert!(!setting_row_disabled(&rows, None));
        assert!(!setting_row_disabled(
            &HashSet::new(),
            Some(&row(SettingKey::ShowFps))
        ));
    }

    #[test]
    fn mouse_hovers_and_clicks_without_focus() {
        assert_eq!(
            hover_and_fire(false, false, true, false, false, false),
            (true, false)
        );
        assert_eq!(
            hover_and_fire(false, false, true, true, false, false),
            (true, true)
        );
        assert_eq!(
            hover_and_fire(false, false, false, true, false, false),
            (false, false)
        );
    }

    #[test]
    fn unfocused_confirm_fires_only_through_the_fallback() {
        assert_eq!(
            hover_and_fire(false, false, true, false, true, false),
            (true, false)
        );
        assert_eq!(
            hover_and_fire(false, false, false, false, true, true),
            (false, true)
        );
    }

    #[test]
    fn focus_owns_the_hover_slot() {
        // The mouse's region neither hovers nor fires while focus is elsewhere.
        assert_eq!(
            hover_and_fire(true, false, true, true, true, true),
            (false, false)
        );
        assert_eq!(
            hover_and_fire(true, true, false, false, false, false),
            (true, false)
        );
        assert_eq!(
            hover_and_fire(true, true, false, false, true, false),
            (true, true)
        );
    }
}
