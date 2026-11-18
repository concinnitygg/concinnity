// The HitRegion pass: hover styling and click / confirm dispatch for every
// region, with the clicks that need the whole system (group toggles, rebind
// capture, dropdown opens) returned for the caller to apply.

use crate::ecs::{FrameVec, PipelineContext, Ref, StepResult};
use hashbrown::{HashMap, HashSet};

use crate::components::{FrameInput, HitRegion, SettingVerb, SpriteFit, TextLabel, UiAction};
use crate::ecs::asset_id::AssetId;
use crate::gfx::overlay::OverlayTransform;
use crate::settings::SettingKey;

use super::intent::UiIntent;
use super::{
    OpenRequest, UiInputSystem, fire_action, point_in_rect, region_covers_canvas, region_rect,
    set_label_style,
};

// The per-frame state every region's hit-test reads.
pub(super) struct RegionFrame<'a> {
    // The scrollbar thumb is being dragged, so no region reacts.
    thumb_active: bool,
    // Reference-canvas mappings for screen-owned regions, chosen by `fit`.
    overlay: OverlayTransform,
    overlay_bottom: OverlayTransform,
    overlay_cover: OverlayTransform,
    // Each panel's content band (reference space), indexed like `panels`.
    panel_bands: FrameVec<'a, [f32; 4]>,
    // A widget or popup took this frame's press, so no region fires from it.
    pub(super) click_taken: bool,
}

impl<'a> RegionFrame<'a> {
    // `overlay` is the default fit mapping; the bottom-anchored and cover
    // mappings a region may opt into derive from the same viewport.
    pub(super) fn new(
        viewport: [f32; 2],
        thumb_active: bool,
        overlay: OverlayTransform,
        panel_bands: FrameVec<'a, [f32; 4]>,
    ) -> Self {
        Self {
            thumb_active,
            overlay,
            overlay_bottom: OverlayTransform::bottom_anchored_from_viewport(viewport),
            overlay_cover: OverlayTransform::cover_from_viewport(viewport),
            panel_bands,
            click_taken: false,
        }
    }

    // The cursor in a region's own space: window pixels off the canvas,
    // reference space through its `fit` on it.
    fn cursor_for(&self, canvas: bool, fit: SpriteFit, mx: f32, my: f32) -> (f32, f32) {
        if !canvas {
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
    // A rebind row was clicked: capture for this setting key.
    pub(super) start_capture: Option<SettingKey>,
    // A dropdown row was clicked: open its floating list.
    pub(super) start_open: Option<OpenRequest>,
    // A fired action ended the step (e.g. Quit); nothing else applies.
    pub(super) fired: Option<StepResult>,
    // A region took this frame's press.
    pub(super) pressed: bool,
}

// Why a region may be unable to hover or fire this frame.
#[derive(Debug, Default, Clone, Copy)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent reasons a region may be inert"
)]
struct RegionGate {
    thumb_active: bool,
    slider: bool,
    // The focus cursor is on this region (a focused slider still highlights).
    pad_focused: bool,
    screen_matches: bool,
    // Its scroll-content row is collapsed.
    collapsed_row: bool,
    // SettingsSystem disabled its setting row at runtime.
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

// A follow-label region's synced position and whether it is inert: it tracks
// its label and goes inert while the label is empty or missing.
fn follow_sync(
    follow: Option<(AssetId, [f32; 2])>,
    labels: &HashMap<AssetId, ([f32; 2], bool)>,
) -> (Option<[f32; 2]>, bool) {
    let Some((label_id, offset)) = follow else {
        return (None, false);
    };
    match labels.get(&label_id) {
        Some(&(at, empty)) => (Some([at[0] + offset[0], at[1] + offset[1]]), empty),
        None => (None, true),
    }
}

// Whether a region's setting row is in the runtime-disabled set.
pub(super) fn setting_row_disabled(
    disabled_rows: &HashSet<SettingKey>,
    region: &HitRegion,
) -> bool {
    region
        .setting()
        .is_some_and(|key| disabled_rows.contains(&key))
}

// What drives a region this frame: the focus cursor while it is set, else the
// mouse.
#[derive(Debug, Clone, Copy)]
enum Driver {
    // `focused`: the cursor rests on this region.
    Focus {
        focused: bool,
        confirm: bool,
    },
    // `fallback_fire`: an unfocused confirm falling through to a full-canvas
    // region.
    Mouse {
        hovered: bool,
        clicked: bool,
        fallback_fire: bool,
    },
}

// A region's `(hovered, fire)`. The focused region styles and fires on
// confirm, and while focus is set the mouse's region does neither. Otherwise the
// mouse hovers, a click fires, and the fallback fires regardless.
fn hover_and_fire(driver: Driver) -> (bool, bool) {
    match driver {
        Driver::Focus { focused, confirm } => (focused, focused && confirm),
        Driver::Mouse {
            hovered,
            clicked,
            fallback_fire,
        } => (hovered, (hovered && clicked) || fallback_fire),
    }
}

impl UiInputSystem {
    // Refresh the owned copy of `DisabledSettingRows` only when the published
    // set changed, so the region pass reads it without a per-frame clone.
    pub(super) fn refresh_disabled_rows(&mut self, ctx: &PipelineContext<'_>) {
        let changed = match ctx.resource::<crate::settings::DisabledSettingRows>() {
            Some(d) => d.0 != self.disabled_rows_cache,
            None => !self.disabled_rows_cache.is_empty(),
        };
        if changed {
            self.disabled_rows_cache = ctx
                .resource::<crate::settings::DisabledSettingRows>()
                .map(|d| d.0.clone())
                .unwrap_or_default();
        }
    }

    // Resolve each followed label's (position, is-empty) once, so the region pass
    // reads a map instead of looking a label up per region.
    fn resolve_follow_labels(&mut self, ctx: &PipelineContext<'_>) {
        self.follow_labels.clear();
        for &id in &self.follow_label_ids {
            if let Some(l) = ctx.get_by_id::<TextLabel>(id) {
                self.follow_labels
                    .insert(id, ([l.x, l.y], l.content.is_empty()));
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
        frame: &RegionFrame<'_>,
        ctx: &mut PipelineContext<'_>,
    ) -> RegionOutcome {
        self.resolve_follow_labels(ctx);
        let (mx, my) = (input.mouse_x, input.mouse_y);
        let [vw, vh] = input.viewport;
        let focus_index = self.focus.as_ref().map(|f| f.index);
        let confirm_fallback = intent.confirm && focus_index.is_none();
        let mut confirm_used = false;
        let mut outcome = RegionOutcome::default();

        for (i, entry) in self.regions.iter_mut().enumerate() {
            let (follow_at, follow_inert) = follow_sync(entry.follow, &self.follow_labels);
            if let Some([x, y]) = follow_at {
                entry.region.x = x;
                entry.region.y = y;
            }
            let pad_focused = focus_index == Some(i);
            let gate = RegionGate {
                thumb_active: frame.thumb_active,
                slider: entry.region.slider.is_some(),
                pad_focused,
                screen_matches: self.screens.receives_input(entry.screen)
                    && !self.widgets.blocked(entry.screen),
                collapsed_row: entry.scroll_row.is_some() && entry.hidden,
                disabled: setting_row_disabled(&self.disabled_rows_cache, &entry.region),
                follow_inert,
            };
            if region_inert(gate) {
                if entry.was_hovered {
                    set_label_style(
                        ctx,
                        entry.region.label.map(Ref::id),
                        entry.original_color,
                        entry.original_scale,
                    );
                    entry.was_hovered = false;
                }
                continue;
            }

            // A region spanning the whole reference canvas covers the full
            // window, so a full-canvas advance region catches letterbox clicks.
            let full_window = entry.canvas && region_covers_canvas(&entry.region);
            let (qx, qy) = frame.cursor_for(entry.canvas, entry.fit, mx, my);
            let r = &entry.region;
            let mut mouse_hovered = if full_window {
                mx >= 0.0 && mx < vw && my >= 0.0 && my < vh
            } else {
                point_in_rect(qx, qy, region_rect(r))
            };
            // A region moves with its label, so it is reached where its
            // label's panel is.
            if mouse_hovered {
                let owner = r.label.and_then(|l| ctx.entity_of(l.id()));
                let own = entry.screen;
                let screens = &self.screens;
                let at = |s: Option<AssetId>| {
                    if s == own {
                        (qx, qy)
                    } else {
                        screens.cursor(s, &frame.overlay, mx, my)
                    }
                };
                mouse_hovered = self.widgets.stack.reaches_owned(own, owner, at);
            }
            // A scroll-content region only counts as hovered inside its band, so
            // a row scrolled past the edge does not catch clicks over the chrome.
            if let Some((pi, _)) = entry.scroll_row
                && let Some(band) = frame.panel_bands.get(pi)
            {
                mouse_hovered = mouse_hovered && point_in_rect(qx, qy, *band);
            }
            let fallback_fire = confirm_fallback && full_window && !confirm_used;
            let driver = if focus_index.is_some() {
                Driver::Focus {
                    focused: pad_focused,
                    confirm: intent.confirm,
                }
            } else {
                Driver::Mouse {
                    hovered: mouse_hovered,
                    clicked: input.left_click && !frame.click_taken,
                    fallback_fire,
                }
            };
            let (hovered, fire) = hover_and_fire(driver);

            if hovered && !entry.was_hovered {
                set_label_style(ctx, r.label.map(Ref::id), r.hover_color, r.hover_scale);
            } else if !hovered && entry.was_hovered {
                set_label_style(
                    ctx,
                    r.label.map(Ref::id),
                    entry.original_color,
                    entry.original_scale,
                );
            }
            entry.was_hovered = hovered;

            if !fire {
                continue;
            }
            if fallback_fire {
                confirm_used = true;
            }
            outcome.pressed = true;
            match &r.action {
                None => {}
                Some(UiAction::Setting {
                    key,
                    verb: SettingVerb::Rebind,
                }) => outcome.start_capture = Some(*key),
                Some(UiAction::Setting {
                    key,
                    verb: SettingVerb::Open,
                }) => {
                    // Snapshot the control rect and the row's un-hovered value style.
                    outcome.start_open = Some(OpenRequest {
                        setting: *key,
                        value_label: r.label.map(Ref::id),
                        anchor: region_rect(r),
                        screen: entry.screen,
                        color: entry.original_color,
                        scale: entry.original_scale,
                    });
                }
                Some(action) => {
                    if let Some(result) = fire_action(action, entry.id, ctx) {
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
    use crate::components::{Binding, ValueOp};

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
        let labels = HashMap::from([
            (AssetId(1), ([10.0, 40.0], false)),
            (AssetId(2), ([10.0, 60.0], true)),
        ]);
        let off = [-2.0, 5.0];
        assert_eq!(follow_sync(None, &labels), (None, false));
        assert_eq!(
            follow_sync(Some((AssetId(1), off)), &labels),
            (Some([8.0, 45.0]), false)
        );
        assert_eq!(
            follow_sync(Some((AssetId(2), off)), &labels),
            (Some([8.0, 65.0]), true)
        );
        assert_eq!(follow_sync(Some((AssetId(3), off)), &labels), (None, true));
    }

    #[test]
    fn setting_row_disabled_matches_the_setting_key() {
        let rows = HashSet::from([SettingKey::ShowFps, SettingKey::Exposure]);
        let row = |action: UiAction| HitRegion {
            action: Some(action),
            ..Default::default()
        };
        let step = |key| {
            row(UiAction::Value {
                binding: Binding::Setting(key),
                op: ValueOp::Step(1),
            })
        };
        let slider = HitRegion {
            slider: Some(Binding::Setting(SettingKey::Exposure)),
            ..Default::default()
        };
        assert!(setting_row_disabled(&rows, &step(SettingKey::ShowFps)));
        assert!(setting_row_disabled(&rows, &slider));
        assert!(!setting_row_disabled(&rows, &step(SettingKey::Vsync)));
        assert!(!setting_row_disabled(&rows, &row(UiAction::Quit)));
        assert!(!setting_row_disabled(&rows, &HitRegion::default()));
        assert!(!setting_row_disabled(
            &HashSet::new(),
            &step(SettingKey::ShowFps)
        ));
    }

    #[test]
    fn mouse_hovers_and_clicks_without_focus() {
        assert_eq!(
            hover_and_fire(Driver::Mouse {
                hovered: true,
                clicked: false,
                fallback_fire: false
            }),
            (true, false)
        );
        assert_eq!(
            hover_and_fire(Driver::Mouse {
                hovered: true,
                clicked: true,
                fallback_fire: false
            }),
            (true, true)
        );
        assert_eq!(
            hover_and_fire(Driver::Mouse {
                hovered: false,
                clicked: true,
                fallback_fire: false
            }),
            (false, false)
        );
    }

    #[test]
    fn unfocused_confirm_fires_only_through_the_fallback() {
        assert_eq!(
            hover_and_fire(Driver::Mouse {
                hovered: true,
                clicked: false,
                fallback_fire: false
            }),
            (true, false)
        );
        assert_eq!(
            hover_and_fire(Driver::Mouse {
                hovered: false,
                clicked: false,
                fallback_fire: true
            }),
            (false, true)
        );
    }

    #[test]
    fn focus_owns_the_hover_slot() {
        // The mouse's region neither hovers nor fires while focus is elsewhere.
        assert_eq!(
            hover_and_fire(Driver::Focus {
                focused: false,
                confirm: true
            }),
            (false, false)
        );
        assert_eq!(
            hover_and_fire(Driver::Focus {
                focused: true,
                confirm: false
            }),
            (true, false)
        );
        assert_eq!(
            hover_and_fire(Driver::Focus {
                focused: true,
                confirm: true
            }),
            (true, true)
        );
    }
}
