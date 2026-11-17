// A settings dropdown's open option list: the state while it is open, the
// input it takes over, and the view the renderer draws.

use super::panels::WHEEL_SCROLL_SPEED;
use super::region_ops::point_in_rect;
use super::{UiInputSystem, intent};
use crate::components::{FrameInput, NavDirection, SettingCommand, SettingOp, TextLabel};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{FontHandle, PipelineContext};
use crate::gfx::dropdown;
use crate::gfx::overlay::OverlayTransform;
use crate::protocol::{DropdownView, OpenDropdown};
use crate::settings;
use crate::settings::SettingKey;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use concinnity_core::math::SaturatingFrom;
use concinnity_core::platform::CompiledPlatforms;

// A settings dropdown whose floating option list is open. Owned by
// UiInputSystem: while set, the list overlays the menu and consumes the frame's
// input (a pick sends a SetIndex command; an outside click or Escape dismisses
// it; the wheel scrolls a list longer than the shown window). Published each
// frame as an `OpenDropdown` resource so the renderer can draw the list. The
// style fields mirror the row's value label so the list text matches it.
#[derive(Debug)]
pub(super) struct OpenDropdownState {
    // The setting the list picks a value for.
    pub(super) setting: SettingKey,
    // The row's value label, forwarded on the pick so the renderer refreshes
    // it.
    pub(super) value_label: Option<AssetId>,
    // The control button's rect `[x, y, w, h]` the list anchors to (reference
    // space for a screen-owned row, window pixels otherwise).
    pub(super) anchor: [f32; 4],
    // Option labels, top to bottom.
    pub(super) options: Vec<String>,
    // The currently-applied option (highlighted as selected).
    pub(super) selected: usize,
    // The scroll position of a list longer than the shown window, as a
    // fractional row offset (the wheel accumulates into it); `first()` rounds
    // and clamps it to the top shown option. 0 for a list that fits.
    scroll_rows: f32,
    // The option under the cursor this frame, if any (highlighted as hovered).
    pub(super) hovered: Option<usize>,
    // The grab offset (cursor y minus thumb top) while the list's scrollbar
    // thumb is being dragged, or `None`. Keeps the thumb from jumping under
    // the cursor on grab; a drag suppresses hover and the pick/dismiss click.
    pub(super) thumb_drag: Option<f32>,
    // The screen the row belongs to (drives reference-space vs window hit-testing
    // and rendering), or `None` for a screen-less row.
    pub(super) screen: Option<AssetId>,
    // Font / scale / color copied from the row's value label so the list text
    // matches the row (the un-hovered style, captured at open).
    pub(super) font: Option<FontHandle>,
    pub(super) scale: f32,
    pub(super) color: [f32; 3],
}

// A dropdown-row click captured during the hit-test loop, resolved into an
// `OpenDropdownState` after the loop (reading the value label's font + current
// text needs the ctx the loop borrows). The style is the row's un-hovered value
// style, snapshotted from the region entry at click time.
pub(super) struct OpenRequest {
    pub(super) setting: SettingKey,
    pub(super) value_label: Option<AssetId>,
    pub(super) anchor: [f32; 4],
    pub(super) screen: Option<AssetId>,
    pub(super) color: Option<[f32; 3]>,
    pub(super) scale: Option<f32>,
}

// The one-shot edges an open dropdown reacts to this frame: pick the hovered
// option, dismiss the list, or re-hover from a cursor move.
#[derive(Clone, Copy)]
pub(super) struct DropdownPulses {
    pub(super) confirm: bool,
    pub(super) escape: bool,
    pub(super) cursor_moved: bool,
}

impl OpenDropdownState {
    // The top shown option: the scroll accumulator rounded and clamped to the
    // windowable range. 0 for a list that fits.
    pub(super) fn first(&self) -> usize {
        usize::saturating_from(concinnity_core::math::round(self.scroll_rows).max(0.0))
            .min(dropdown::max_first(self.options.len()))
    }
}

impl DropdownPulses {
    pub(super) fn from_intent(intent: &intent::UiIntent) -> Self {
        Self {
            // Enter always picks inside the list, focused or not.
            confirm: intent.confirm || intent.enter_pressed,
            escape: intent.ui_escape,
            cursor_moved: intent.cursor_moved,
        }
    }
}

impl UiInputSystem {
    // Advance an open dropdown for one frame: track the option under the
    // cursor, and on a click pick it (a SetIndex command) or dismiss (a click
    // outside the list); Escape / back also dismiss. Nav pulses move the
    // selection highlight and confirm picks it, sharing the hover slot with
    // the cursor (whichever moved last wins). The wheel and a scrollbar-thumb
    // drag scroll the shown window of a list longer than
    // `dropdown::MAX_VISIBLE` (never dismissing). Clears `open_dropdown` when
    // the list closes.
    pub(super) fn step_open_dropdown(
        &mut self,
        input: &FrameInput,
        nav: Option<NavDirection>,
        pulses: DropdownPulses,
        ctx: &mut PipelineContext<'_>,
    ) {
        let DropdownPulses {
            confirm,
            escape,
            cursor_moved,
        } = pulses;
        let Some(state) = self.open_dropdown.as_mut() else {
            return;
        };
        let count = state.options.len();
        // Screen-owned rows hit-test in reference space; a screen-less row in window
        // pixels (matches the region hit-test in `step`).
        let overlay = OverlayTransform::from_viewport(input.viewport);
        let (qx, qy) = if state.screen.is_some() {
            overlay.inverse(input.mouse_x, input.mouse_y)
        } else {
            (input.mouse_x, input.mouse_y)
        };
        let max = dropdown::max_first(count) as f32;
        // Wheel: scroll the shown window (same feel as the settings panel:
        // wheel distance in pixels, converted to rows by the row height).
        if input.scroll_delta != 0.0 {
            let item_h = state.anchor[3].max(1.0);
            state.scroll_rows = (state.scroll_rows
                + input.scroll_delta * WHEEL_SCROLL_SPEED / item_h)
                .clamp(0.0, max);
        }
        let layout = dropdown::layout(state.anchor, count);

        // Scrollbar-thumb drag: a press on the thumb grabs it (keeping the
        // cursor's offset within it); a press elsewhere in the scrollbar strip
        // jumps the thumb there. While the button stays down the cursor's y
        // maps back to the window's scroll position, even outside the list.
        if input.left_button_down {
            if state.thumb_drag.is_none()
                && input.left_click
                && let Some(thumb) = dropdown::thumb_rect(&layout, state.first(), count)
                && let Some(track) = dropdown::track_rect(&layout, count)
            {
                if point_in_rect(qx, qy, thumb) {
                    state.thumb_drag = Some(qy - thumb[1]);
                } else if point_in_rect(qx, qy, track) {
                    state.thumb_drag = Some(thumb[3] / 2.0);
                }
            }
            if let Some(grab) = state.thumb_drag {
                state.scroll_rows = dropdown::first_for_thumb_top(&layout, count, qy - grab);
            }
        } else {
            state.thumb_drag = None;
        }
        let dragging = state.thumb_drag.is_some();

        let first = state.first();
        // Rows show options `first..`; hovered is the OPTION index. No hover
        // while the thumb is dragged (the drag owns the cursor); an idle
        // cursor keeps the pulse-driven selection instead of re-asserting the
        // stale position under it.
        if dragging {
            state.hovered = None;
        } else if cursor_moved || input.left_click || input.scroll_delta != 0.0 {
            state.hovered = dropdown::item_at(&layout, qx, qy).map(|row| first + row);
        }

        // Nav pulses move the selection highlight one option per pulse,
        // scrolling the shown window along with it.
        if let Some(dir) = nav
            && !dragging
        {
            let cur = state.hovered.unwrap_or(state.selected);
            let stepped = match dir {
                NavDirection::Up => cur.saturating_sub(1),
                NavDirection::Down => (cur + 1).min(count.saturating_sub(1)),
                _ => cur,
            };
            state.hovered = Some(stepped);
            let first = state.first();
            if stepped < first {
                state.scroll_rows = stepped as f32;
            } else if stepped >= first + dropdown::MAX_VISIBLE {
                state.scroll_rows = (stepped + 1 - dropdown::MAX_VISIBLE) as f32;
            }
        }

        // Escape / back dismiss without changing the value.
        if escape {
            self.open_dropdown = None;
            return;
        }
        // Confirm picks the highlighted option (or re-picks the current value
        // when nothing is highlighted yet), then closes.
        if confirm {
            let pick = state.hovered.unwrap_or(state.selected);
            let setting = state.setting;
            let value_label = state.value_label;
            self.open_dropdown = None;
            ctx.events_mut::<SettingCommand>().send(SettingCommand {
                setting,
                op: SettingOp::SetIndex(pick),
                value_label,
                persist: true,
            });
            return;
        }
        if input.left_click && !dragging {
            match state.hovered {
                // Pick the hovered option: send the absolute index, then close.
                Some(i) => {
                    let setting = state.setting;
                    let value_label = state.value_label;
                    self.open_dropdown = None;
                    ctx.events_mut::<SettingCommand>().send(SettingCommand {
                        setting,
                        op: SettingOp::SetIndex(i),
                        value_label,
                        persist: true,
                    });
                }
                // A click outside the list dismisses it (consumed here, so the
                // row behind it does not also react this frame).
                None => self.open_dropdown = None,
            }
        }
    }

    // Resolve a captured dropdown-row click into an open list: read the option
    // labels from the shared registry (or, for a runtime-enumerated setting
    // like `resolution`, from the resource its owner publishes) and the current
    // value from the row's value label to seed the selection. Returns `None`
    // (stays closed) when the setting has no options to offer.
    pub(super) fn build_open_dropdown(
        req: &OpenRequest,
        ctx: &PipelineContext<'_>,
    ) -> Option<OpenDropdownState> {
        let options: Vec<String> = if req.setting == SettingKey::GraphicsApi {
            ctx.resource::<CompiledPlatforms>()?
                .0
                .iter()
                .map(|p| p.label().to_string())
                .collect()
        } else if settings::is_dynamic_dropdown(req.setting) {
            // `resolution`'s modes, which the renderer publishes at init.
            let modes = ctx.resource::<crate::window::display_mode::DisplayModes>()?;
            if modes.0.is_empty() {
                return None;
            }
            modes
                .0
                .iter()
                .map(crate::window::display_mode::DisplayMode::label)
                .collect()
        } else {
            settings::options(req.setting)?
                .iter()
                .map(ToString::to_string)
                .collect()
        };
        // The value label's font (for the list text) and current content (to
        // mark the selected option).
        let (font, current) = req
            .value_label
            .and_then(|id| {
                ctx.get_by_id::<TextLabel>(id)
                    .map(|l| (l.font, l.content.clone()))
            })
            .unwrap_or((None, String::new()));
        let selected = options.iter().position(|o| *o == current).unwrap_or(0);
        Some(OpenDropdownState {
            setting: req.setting,
            value_label: req.value_label,
            anchor: req.anchor,
            // Open with the selection near the middle of a scrolled window.
            scroll_rows: dropdown::first_for_selected(selected, options.len()) as f32,
            options,
            selected,
            hovered: None,
            thumb_drag: None,
            screen: req.screen,
            font,
            scale: req.scale.unwrap_or(1.0),
            color: req.color.unwrap_or([1.0, 1.0, 1.0]),
        })
    }

    // Publish the current dropdown state as an `OpenDropdown` resource for
    // the renderer to draw next tick (`None` while closed). The published
    // view is updated in place: the option labels only change when a list
    // opens, so the steady-state frames of an open list refresh the cheap
    // fields without recloning the options.
    pub(super) fn publish_dropdown(&self, ctx: &mut PipelineContext<'_>) {
        let mut published = ctx.take_resource::<OpenDropdown>().unwrap_or_default();
        match &self.open_dropdown {
            None => published.0 = None,
            Some(s) => match published.0.as_mut() {
                Some(view) => {
                    view.anchor = s.anchor;
                    if view.options != s.options {
                        view.options.clone_from(&s.options);
                    }
                    view.selected = s.selected;
                    view.first = s.first();
                    view.hovered = s.hovered;
                    view.screen = s.screen;
                    view.font = s.font;
                    view.scale = s.scale;
                    view.color = s.color;
                }
                None => {
                    published.0 = Some(DropdownView {
                        anchor: s.anchor,
                        options: s.options.clone(),
                        selected: s.selected,
                        first: s.first(),
                        hovered: s.hovered,
                        screen: s.screen,
                        font: s.font,
                        scale: s.scale,
                        color: s.color,
                    });
                }
            },
        }
        ctx.insert_resource(published);
    }
}
