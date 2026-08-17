// HitRegion / Screen / KeyBinding input dispatch. An internal system (not a
// declarable asset): `World::start` constructs one whenever the world contains
// any `HitRegion`, `Screen`, or `KeyBinding`, then it processes hover/click,
// screen overlays, and key bindings each frame.

pub(crate) mod dropdown;
mod focus;
mod screen;
mod scroll_layout;

use crate::assets::{
    FrameInput, HitRegion, Key, KeyBinding, NavDirection, SceneCommand, Screen, ScreenCommand,
    ScreenShown, ScrollPanel, SettingCommand, SettingOp, Sprite, SpriteFit, StoryCommand,
    TextLabel,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult, System};
use crate::gfx::settings;
use concinnity_core::gfx::overlay::{OverlayTransform, UI_REFERENCE_SIZE};
use screen::{ScreenMeta, ScreenRegistry};
use scroll_layout::RowSpec;
use std::collections::HashMap;

// How many reference-space pixels one unit of scroll-wheel delta moves a panel.
const WHEEL_SCROLL_SPEED: f32 = 2.0;
// Shown in a rebind row's value label while it waits for the user to press a key.
const REBIND_PROMPT: &str = "Press a key...";
const PAD_REBIND_PROMPT: &str = "Press a button...";

// Per-hit-region bookkeeping stored after init().
#[derive(Debug)]
struct RegionEntry {
    region: HitRegion,
    // Original TextLabel color, captured at init() for hover-out restore.
    original_color: Option<[f32; 3]>,
    // Original TextLabel scale, captured at init() for hover-out restore.
    original_scale: Option<f32>,
    // Whether this region was hovered last frame (to detect transitions).
    was_hovered: bool,
    // The screen this region belongs to (derived from its name prefix at
    // init()), or `None` if it belongs to no screen. Regions in a screen only
    // fire while that screen is active; regions outside any screen only fire
    // when no screen is active.
    screen: Option<AssetId>,
    // For a slider drag region (action `setting:<key>:drag`), the setting key.
    // `None` for an ordinary click region. A slider region is driven by the
    // drag pass, not the click-to-fire path.
    slider_key: Option<String>,
    // The scroll panel + row this region belongs to, if it sits in a panel's
    // content band (resolved by position at init). Such a region reflows with
    // its row each frame and only fires while its row is shown and inside the
    // band. `None` for chrome (tab bar, Back) and non-panel regions.
    scroll_row: Option<(usize, usize)>,
    // The region's authored y, kept so the scroll reflow can set
    // `region.y = base_y + dy` from a fresh delta each frame.
    region_base_y: f32,
    // The collapsible group index this region's click toggles (action
    // `group:toggle:<gid>`), or `None`. A group-toggle region flips its panel's
    // group instead of firing an action.
    group_toggle: Option<usize>,
    // Set by the scroll reflow when this region's row is hidden (its group is
    // collapsed); a hidden region never hovers or fires.
    hidden: bool,
    // For a `follow_label` region: its label id and the captured `region.y -
    // label.y` offset. Each frame the region's y is re-synced to the label
    // (so a runtime-laid-out menu stays clickable) and the region is inert
    // while the label is empty (a hidden entry catches no clicks).
    follow: Option<(AssetId, f32)>,
    // How the region maps from the reference canvas to the window (matches the
    // sprite/label `fit`); a region spanning the whole canvas covers the full
    // window regardless.
    fit: SpriteFit,
}

// One row of a scroll panel: the elements that move together, their authored
// y's (snapshot at init so the reflow is `base + dy`), the row's top + height
// (for bucketing regions to rows), and the collapsible group it belongs to.
#[derive(Debug)]
struct RowState {
    elements: Vec<AssetId>,
    base_ys: Vec<f32>,
    base_y: f32,
    height: f32,
    group: Option<usize>,
}

// A collapsible group's runtime state: whether it is collapsed and the header
// label whose `+`/`-` prefix reflects it.
#[derive(Debug)]
struct GroupState {
    collapsed: bool,
    header: Option<AssetId>,
    title: String,
}

// Runtime state for one scroll panel, drained from a `ScrollPanel` at init.
#[derive(Debug)]
struct PanelState {
    screen: Option<AssetId>,
    // Content band [x, y, width, height] in reference space.
    band: [f32; 4],
    rows: Vec<RowState>,
    groups: Vec<GroupState>,
    thumb: Option<AssetId>,
    track: Option<AssetId>,
    track_x: f32,
    track_y: f32,
    track_w: f32,
    track_h: f32,
    // Current scroll offset (reference pixels), clamped by the solver.
    scroll: f32,
    // Last solve outputs, kept for the thumb-drag cursor->scroll mapping.
    content_height: f32,
    thumb_h: f32,
}

// One sprite's accumulated scroll-layout write; only the set fields apply.
#[derive(Debug, Default, Clone, Copy)]
struct SpriteUpdate {
    y: Option<f32>,
    height: Option<f32>,
    visible: Option<bool>,
}

// One label's accumulated scroll-layout write; only the set fields apply.
#[derive(Debug, Default)]
struct LabelUpdate {
    y: Option<f32>,
    visible: Option<bool>,
    content: Option<String>,
}

// Scratch buffers for the per-frame scroll-layout solve/apply, kept on the
// system so the pass reuses their capacity instead of reallocating each frame.
#[derive(Debug, Default)]
struct LayoutScratch {
    sprites: HashMap<AssetId, SpriteUpdate>,
    labels: HashMap<AssetId, LabelUpdate>,
    specs: Vec<RowSpec>,
    collapsed: Vec<bool>,
    // Per-panel `(active, row placements)` for the region reflow.
    solved_rows: Vec<(bool, Vec<scroll_layout::RowPlacement>)>,
}

// A settings dropdown whose floating option list is open. Owned by
// UiInputSystem: while set, the list overlays the menu and consumes the frame's
// input (a pick sends a SetIndex command; an outside click or Escape dismisses
// it; the wheel scrolls a list longer than the shown window). Published each
// frame as an `OpenDropdown` resource so GraphicsSystem can draw the list. The
// style fields mirror the row's value label so the list text matches it.
#[derive(Debug)]
struct OpenDropdownState {
    // The setting the list picks a value for (e.g. `"window_mode"`).
    setting: String,
    // The row's value label, forwarded on the pick so GraphicsSystem refreshes
    // it.
    value_label: Option<AssetId>,
    // The control button's rect `[x, y, w, h]` the list anchors to (reference
    // space for a screen-owned row, window pixels otherwise).
    anchor: [f32; 4],
    // Option labels, top to bottom.
    options: Vec<String>,
    // The currently-applied option (highlighted as selected).
    selected: usize,
    // The scroll position of a list longer than the shown window, as a
    // fractional row offset (the wheel accumulates into it); `first()` rounds
    // and clamps it to the top shown option. 0 for a list that fits.
    scroll_rows: f32,
    // The option under the cursor this frame, if any (highlighted as hovered).
    hovered: Option<usize>,
    // The grab offset (cursor y minus thumb top) while the list's scrollbar
    // thumb is being dragged, or `None`. Keeps the thumb from jumping under
    // the cursor on grab; a drag suppresses hover and the pick/dismiss click.
    thumb_drag: Option<f32>,
    // The screen the row belongs to (drives reference-space vs window hit-testing
    // and rendering), or `None` for a screen-less row.
    screen: Option<AssetId>,
    // Font / scale / color copied from the row's value label so the list text
    // matches the row (the un-hovered style, captured at open).
    font: Option<crate::ecs::FontHandle>,
    scale: f32,
    color: [f32; 3],
}

impl OpenDropdownState {
    // The top shown option: the scroll accumulator rounded and clamped to the
    // windowable range. 0 for a list that fits.
    fn first(&self) -> usize {
        (self.scroll_rows.round().max(0.0) as usize).min(dropdown::max_first(self.options.len()))
    }
}

// A dropdown-row click captured during the hit-test loop, resolved into an
// `OpenDropdownState` after the loop (reading the value label's font + current
// text needs the ctx the loop borrows). The style is the row's un-hovered value
// style, snapshotted from the region entry at click time.
struct OpenRequest {
    setting: String,
    value_label: Option<AssetId>,
    anchor: [f32; 4],
    screen: Option<AssetId>,
    color: Option<[f32; 3]>,
    scale: Option<f32>,
}

// An in-progress key rebind: a Controls-tab rebind row was clicked and is
// waiting for the user to press a key. The next `FrameInput.captured_key` binds
// it; Escape cancels and restores the row's previous value text.
#[derive(Debug)]
struct Capture {
    // The rebind setting key, e.g. `"key_forward"`.
    setting_key: String,
    // The value `TextLabel` showing the bound key (set to a prompt while
    // capturing; GraphicsSystem rewrites it after the bind).
    value_label: Option<AssetId>,
    // The label's text before capture began, restored if the user cancels.
    prev_text: String,
}

// HitRegion / Screen / KeyBinding input dispatch behavior. Constructed
// internally by `World::start` when the world declares any `HitRegion`,
// `Screen`, or `KeyBinding`; never a world-declared asset, so it carries no
// config.
#[derive(Debug)]
pub struct UiInputSystem {
    regions: Vec<RegionEntry>,
    bindings: Vec<KeyBinding>,
    screens: ScreenRegistry,
    // asset_id of UI elements (Sprite, TextLabel) by their owning screen.
    // Built at init() from `<screen_name>_*` name prefixes.
    sprites_by_screen: HashMap<AssetId, Vec<AssetId>>,
    labels_by_screen: HashMap<AssetId, Vec<AssetId>>,
    text_inputs_by_screen: HashMap<AssetId, Vec<AssetId>>,
    // Index (into `regions`) of the slider currently being dragged, or `None`.
    // Set on the press edge over a slider track, cleared on button release.
    dragging: Option<usize>,
    // Scroll panels in the world, drained at init. Driven each frame: collapse
    // state + scroll offset are solved into per-row positions written back onto
    // the elements + regions.
    panels: Vec<PanelState>,
    // `(panel index, grab offset)` while the scrollbar thumb is being dragged.
    // The grab offset keeps the thumb from jumping under the cursor on grab.
    thumb_drag: Option<(usize, f32)>,
    // A pending key rebind (a Controls-tab rebind row is capturing), or `None`.
    // While set, the menu consumes the frame for capture: the next pressed key
    // binds it and Escape cancels.
    capturing: Option<Capture>,
    // The open settings dropdown, or `None`. While set, its floating list
    // overlays the menu and consumes input until a pick / dismiss.
    open_dropdown: Option<OpenDropdownState>,
    // Cursor into the Events<ScreenCommand> queue. This system both sends (when a
    // `screen:*` action fires) and reads ScreenCommands, so a command fired this
    // frame is applied on the next, the same one-frame lag the old drain had.
    screen_cmd_cursor: crate::ecs::EventCursor,
    // Cached copy of the engine's `DisabledSettingRows`, refreshed only when the
    // published resource changes so the hit-test loop reads an owned set without
    // cloning the resource every frame (SettingsSystem republishes rarely).
    disabled_rows_cache: std::collections::HashSet<String>,
    // The gamepad/keyboard focus cursor, or `None` while the mouse drives the
    // menu. While set, it styles + fires in place of hover, and any cursor
    // movement dismisses it.
    focus: Option<focus::FocusRef>,
    // Last frame's cursor position, to detect the mouse movement that
    // dismisses focus.
    last_cursor: Option<(f32, f32)>,
    // Ids of the labels follow-label regions track, gathered at init so the
    // hit-test pass resolves them in one label query instead of one full scan
    // per region.
    follow_label_ids: std::collections::HashSet<AssetId>,
    // Scratch: each followed label's (y, is-empty) this frame.
    follow_labels: HashMap<AssetId, (f32, bool)>,
    // Scratch buffers for the scroll-layout solve/apply.
    layout: LayoutScratch,
}

impl UiInputSystem {
    // Empty dispatch state. The world's `HitRegion` / `Screen` / `KeyBinding`
    // components are drained into it in [`System::init`].
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
            bindings: Vec::new(),
            screens: ScreenRegistry::default(),
            sprites_by_screen: HashMap::new(),
            labels_by_screen: HashMap::new(),
            text_inputs_by_screen: HashMap::new(),
            dragging: None,
            panels: Vec::new(),
            thumb_drag: None,
            capturing: None,
            open_dropdown: None,
            screen_cmd_cursor: crate::ecs::EventCursor::default(),
            disabled_rows_cache: std::collections::HashSet::new(),
            focus: None,
            last_cursor: None,
            follow_label_ids: std::collections::HashSet::new(),
            follow_labels: HashMap::new(),
            layout: LayoutScratch::default(),
        }
    }
}

impl System for UiInputSystem {
    fn access(&self) -> crate::ecs::Access {
        crate::ecs::Access::new()
            .reads_components(crate::component_mask![crate::assets::FrameInput])
            .writes_components(crate::component_mask![
                crate::assets::TextLabel,
                crate::assets::Sprite,
                crate::assets::TextInput,
            ])
            .reads_resources(crate::resource_mask![
                crate::ecs::DisabledSettingRows,
                crate::ecs::DisplayModes,
            ])
            .writes_resources(crate::resource_mask![
                crate::ecs::OpenDropdown,
                crate::ecs::ScreenStack,
                crate::assets::ScreenCommand,
                crate::assets::ScreenShown,
                crate::assets::SettingCommand,
                crate::assets::SceneCommand,
                crate::assets::StoryCommand,
            ])
    }

    fn init(&mut self, ctx: &mut PipelineContext) {
        // Drain Screen assets, record each one's policies, and pick the one
        // flagged `initial` to open at world start.
        let mut initial: Option<AssetId> = None;
        for s in ctx.drain::<Screen>() {
            self.screens
                .register(s.asset_id, ScreenMeta::from_asset(&s));
            if s.initial && initial.is_none() {
                initial = Some(s.asset_id);
            }
        }

        // Drain KeyBindings: they aren't iterated each frame on the world,
        // we just match the pulse against this snapshot.
        self.bindings = ctx.drain::<KeyBinding>();

        // Drain HitRegions, capture per-region hover restore state, and
        // assign each region to a screen (or none) based on the resolved
        // `screen` field that the build pipeline writes from the name prefix.
        let hit_regions = ctx.drain::<HitRegion>();
        for region in hit_regions {
            // A region disabled by the engine (e.g. a capability-gated settings
            // row grayed out at init) is inert: dropping it here means it never
            // hovers, fires, drags, or reflows. Its labels are styled + reflowed
            // independently (by GraphicsSystem and the scroll panel).
            if region.disabled {
                continue;
            }
            let (original_color, original_scale) = match region.label {
                None => (None, None),
                Some(label_id) => ctx
                    .query::<TextLabel>()
                    .find(|l| l.asset_id == label_id)
                    .map(|l| (Some(l.color), Some(l.scale)))
                    .unwrap_or((None, None)),
            };
            let screen = region.screen;
            let slider_key = crate::gfx::setting_action::key_with_verb(&region.action, "drag")
                .map(str::to_string);
            let group_toggle = group_toggle_from_action(&region.action);
            let region_base_y = region.y;
            // A follow-label region captures the y offset to its label now, so
            // the runtime layout can move the label and the region tracks it.
            let follow = if region.follow_label {
                region.label.and_then(|lid| {
                    ctx.query::<TextLabel>()
                        .find(|l| l.asset_id == lid)
                        .map(|l| (lid, region.y - l.y))
                })
            } else {
                None
            };
            let fit = region.fit;
            self.regions.push(RegionEntry {
                region,
                original_color,
                original_scale,
                was_hovered: false,
                screen,
                slider_key,
                scroll_row: None,
                region_base_y,
                group_toggle,
                hidden: false,
                follow,
                fit,
            });
        }
        self.follow_label_ids = self
            .regions
            .iter()
            .filter_map(|e| e.follow.map(|(id, _)| id))
            .collect();

        // Build screen → UI-element maps by reading each Sprite/TextLabel's
        // resolved `screen` field (the build pipeline writes it from the
        // <screen>_* name prefix).
        for s in ctx.query::<Sprite>() {
            if let Some(screen_id) = s.screen {
                self.sprites_by_screen
                    .entry(screen_id)
                    .or_default()
                    .push(s.asset_id);
            }
        }
        for l in ctx.query::<TextLabel>() {
            if let Some(screen_id) = l.screen {
                self.labels_by_screen
                    .entry(screen_id)
                    .or_default()
                    .push(l.asset_id);
            }
        }
        for t in ctx.query::<crate::assets::TextInput>() {
            if let Some(screen_id) = t.screen {
                self.text_inputs_by_screen
                    .entry(screen_id)
                    .or_default()
                    .push(t.asset_id);
            }
        }

        // Drain ScrollPanels into runtime state and bucket the regions into
        // their rows (uses the regions drained just above).
        self.init_panels(ctx);

        // Screens start hidden: zero out the visibility of every screen-owned
        // Sprite and TextLabel.
        for ids in self.sprites_by_screen.values() {
            for &id in ids {
                for sp in ctx.query_mut::<Sprite>() {
                    if sp.asset_id == id {
                        sp.visible = false;
                        break;
                    }
                }
            }
        }
        for ids in self.labels_by_screen.values() {
            for &id in ids {
                for lbl in ctx.query_mut::<TextLabel>() {
                    if lbl.asset_id == id {
                        lbl.visible = false;
                        break;
                    }
                }
            }
        }
        for ids in self.text_inputs_by_screen.values() {
            for &id in ids {
                for ti in ctx.query_mut::<crate::assets::TextInput>() {
                    if ti.asset_id == id {
                        ti.visible = false;
                        break;
                    }
                }
            }
        }

        // Open the initial screen (if any) through the same transition path as
        // any navigation, so its elements show, its focus field focuses, and
        // the activation is announced (AudioCue music fires for the first
        // screen too). Publish the stack resource either way so the overlay
        // reads a fresh state from frame 0.
        if let Some(id) = initial
            && let Some(transition) = self.screens.apply(ScreenCommand::Show(id))
        {
            self.apply_transition(transition, ctx);
        }
        self.publish_screen_stack(ctx);

        // Solve the initial scroll layout so frame 0 already shows the right
        // collapsed/scrolled positions (a default-collapsed group starts shut).
        self.apply_scroll_layout(ctx);
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // Apply ScreenCommands sent last frame first, so a click last frame takes
        // effect before this frame's hit-testing reads `active`. Clone them out
        // of the queue to release the ctx borrow before apply_screen_command,
        // which needs &mut ctx.
        let screen_cmds: Vec<ScreenCommand> = match ctx.events::<ScreenCommand>() {
            Some(events) => events.read(&mut self.screen_cmd_cursor).cloned().collect(),
            None => Vec::new(),
        };
        for cmd in screen_cmds {
            self.apply_screen_command(cmd, ctx);
        }

        // Read (not drain) the per-frame input snapshot so this system can
        // coexist with Camera3DSystem (both query it; GraphicsSystem clears it
        // before the next push). Take the most recent if more than one exists.
        let input = match ctx.query::<FrameInput>().last().cloned() {
            Some(i) => i,
            None => return StepResult::Continue,
        };

        // While any visible TextInput has keyboard focus, typed keys belong to
        // the field: ordinary KeyBindings and the focus pulses are suspended
        // so typing cannot fire actions (screen toggles below stay live).
        let typing = ctx
            .query::<crate::assets::TextInput>()
            .any(|t| t.visible && t.focused);

        // The pad's menu pulses engage only while a capturing screen is
        // active; during play the same buttons keep their gameplay meanings.
        let screen_active = self.screens.top_capture().is_some();
        // Mouse movement dismisses the focus cursor: the menu returns to
        // hover-driven interaction until the next pulse.
        let cursor_moved = self
            .last_cursor
            .is_some_and(|(px, py)| (input.mouse_x - px).abs() + (input.mouse_y - py).abs() > 2.0);
        self.last_cursor = Some((input.mouse_x, input.mouse_y));
        if cursor_moved {
            self.focus = None;
        }
        // The keyboard arrows drive the same focus model as the pad pulse.
        let nav = if screen_active && !typing {
            input.nav.or(match input.captured_key {
                Some(Key::Up) => Some(NavDirection::Up),
                Some(Key::Down) => Some(NavDirection::Down),
                Some(Key::Left) => Some(NavDirection::Left),
                Some(Key::Right) => Some(NavDirection::Right),
                _ => None,
            })
        } else {
            None
        };
        // Confirm fires the focused control: the pad's South button, or Enter
        // while something is focused (an unfocused Enter still reaches the
        // KeyBindings, e.g. a story's advance binding).
        let enter_pressed = screen_active && !typing && input.captured_key == Some(Key::Enter);
        let enter_confirm = enter_pressed && self.focus.is_some();
        let confirm = (screen_active && !typing && input.confirm) || enter_confirm;
        // The pad's East button backs out like Escape while a screen is up.
        let ui_escape = input.escape || (input.back && screen_active);

        // An open settings dropdown's floating list overlays the menu and
        // consumes this frame: hover tracks the option under the cursor, a click
        // picks it (or, outside the list, dismisses), and Escape / a scroll close
        // it. Handled before the Escape keybinding + hit-test passes so it takes
        // priority (Escape closes the list rather than the menu, a click on an
        // option does not fall through to the row behind it).
        if self.open_dropdown.is_some() {
            // Enter always picks inside the list, focused or not.
            let pick = confirm || enter_pressed;
            self.step_open_dropdown(
                &input,
                nav,
                DropdownPulses {
                    confirm: pick,
                    escape: ui_escape,
                    cursor_moved,
                },
                ctx,
            );
            self.publish_dropdown(ctx);
            return StepResult::Continue;
        }

        // A pending rebind (a Controls-tab rebind row was clicked) consumes
        // the whole frame: the next pressed key (or gamepad button, for a
        // `pad_*` row) binds it, Escape cancels (and restores the row's
        // previous text), otherwise it keeps waiting. No clicks, hover, or
        // other key bindings fire while capturing; input of the other kind is
        // ignored, so a stray key press never lands in a button row.
        if self.capturing.is_some() {
            let wants_button = self
                .capturing
                .as_ref()
                .is_some_and(|c| c.setting_key.starts_with("pad_"));
            let op = if wants_button {
                input.captured_button.map(SettingOp::RebindButton)
            } else {
                input.captured_key.map(SettingOp::Rebind)
            };
            if input.escape {
                self.cancel_capture(ctx);
            } else if let Some(op) = op {
                let cap = self.capturing.take().expect("capturing is some");
                ctx.events_mut::<SettingCommand>().send(SettingCommand {
                    setting: cap.setting_key,
                    op,
                    value_label: cap.value_label,
                    persist: true,
                });
                // GraphicsSystem rewrites the value label to the new binding
                // when it reads the command next tick; the prompt shows until
                // then.
            }
            return StepResult::Continue;
        }

        // A Screen's `toggle_key` opens / closes it from anywhere, ahead of
        // ordinary KeyBindings and immune to the typing suppression (so a
        // console screen's own key still closes it while its field has
        // focus). A matched toggle consumes the key press. The pad's back
        // pulse rides the Escape name, so it pops toggled screens and fires
        // Escape bindings exactly like the key.
        let pressed_key = if ui_escape {
            Some("Escape".to_string())
        } else {
            input.captured_key.map(|k| k.name().to_string())
        };
        let toggled_key = pressed_key.as_deref().is_some_and(|name| {
            let toggles = self.screens.toggles_for_key(name);
            for id in &toggles {
                ctx.events_mut::<ScreenCommand>()
                    .send(ScreenCommand::Toggle(*id));
            }
            !toggles.is_empty()
        });

        // Handle KeyBindings before HitRegion clicks so an Esc-toggle-pause
        // beats a click that landed on the same frame. A binding scoped to a
        // screen only fires while that screen is on top of the stack. Escape is
        // matched separately (it is not a `Key` variant, so it never arrives as
        // a `captured_key`); every other binding matches the one-frame pressed
        // key by its canonical name -- e.g. a story's Space / Enter advance
        // bindings. Rebind capture and an open dropdown already returned above,
        // so this cannot steal a key those flows want; an Enter consumed as
        // confirm never doubles into an Enter binding.
        if !toggled_key
            && !typing
            && !enter_confirm
            && let Some(name) = pressed_key.as_deref()
        {
            let top = self.screens.top();
            for kb in &self.bindings {
                let scoped_out = kb.screen.is_some() && kb.screen != top;
                if kb.key == name && !kb.action.is_empty() && !scoped_out {
                    // KeyBindings carry no label (no settings row binds a key).
                    if let Some(result) = fire_action(&kb.action, None, ctx) {
                        return result;
                    }
                    break;
                }
            }
        }

        let mx = input.mouse_x;
        let my = input.mouse_y;
        let clicked = input.left_click;
        let down = input.left_button_down;
        // Regions gate on the topmost input-capturing screen (a passthrough
        // screen above it only draws): its regions fire; with no capturing
        // screen active, screen-less regions fire.
        let active_screen = self.screens.top_capture();
        // Screen-owned regions are overlay UI authored in the reference canvas and
        // scaled onto the window; map the live cursor back into reference space
        // before testing it against their (reference-space) rects. Screen-less
        // regions stay in window pixels (see crate::gfx::overlay).
        let overlay = OverlayTransform::from_viewport(input.viewport);
        // Alternate mappings a region may opt into via `fit` (bottom-anchored
        // dialog furniture); the fit transform above stays the default.
        let overlay_bottom = OverlayTransform::bottom_anchored_from_viewport(input.viewport);
        let overlay_cover = OverlayTransform::cover_from_viewport(input.viewport);
        let [vw, vh] = input.viewport;

        // Scroll-wheel + scrollbar-thumb input for the active screen's panel; both
        // adjust the panel's scroll offset (clamped later in the apply pass). A
        // thumb drag suppresses the slider + click passes so the gutter doesn't
        // double as a control.
        let thumb_active = self.handle_scroll_input(&input, active_screen, &overlay);

        // Per-panel bands (reference space), so a scroll-content region only
        // fires while the cursor is inside its panel window.
        let panel_bands: Vec<[f32; 4]> = self.panels.iter().map(|p| p.band).collect();

        // Slider drag pass. A slider's track region is driven here, not by the
        // click-to-fire loop below: the press edge (`clicked`) over a track
        // begins a drag, the held button (`down`) tracks the cursor each frame,
        // and release commits the final value. The dragged region is remembered
        // so the drag continues even when the cursor leaves the track.
        if !thumb_active && !down {
            // Release: commit the dragged slider's final position (persists).
            if let Some(i) = self.dragging.take()
                && self.regions[i].screen == active_screen
                && let Some(key) = self.regions[i].slider_key.clone()
            {
                // Slider tracks are overlay UI: map the cursor to reference space.
                let (qx, _) = overlay.inverse(mx, my);
                let r = &self.regions[i].region;
                let frac = ((qx - r.x) / r.width).clamp(0.0, 1.0);
                let label = r.label;
                ctx.events_mut::<SettingCommand>().send(SettingCommand {
                    setting: key,
                    op: SettingOp::SetFraction(frac),
                    value_label: label,
                    persist: true,
                });
            }
        } else if !thumb_active {
            // Slider tracks are overlay UI: map the cursor to reference space.
            let (qx, qy) = overlay.inverse(mx, my);
            for i in 0..self.regions.len() {
                if self.regions[i].screen != active_screen {
                    continue;
                }
                let Some(key) = self.regions[i].slider_key.clone() else {
                    continue;
                };
                let (rx, ry, rw, rh, label) = {
                    let r = &self.regions[i].region;
                    (r.x, r.y, r.width, r.height, r.label)
                };
                let over = qx >= rx && qx < rx + rw && qy >= ry && qy < ry + rh;
                if self.dragging.is_none() && clicked && over {
                    self.dragging = Some(i);
                }
                if self.dragging == Some(i) {
                    let frac = ((qx - rx) / rw).clamp(0.0, 1.0);
                    // In-progress: apply live but skip the disk write (persist
                    // only on release, above).
                    ctx.events_mut::<SettingCommand>().send(SettingCommand {
                        setting: key,
                        op: SettingOp::SetFraction(frac),
                        value_label: label,
                        persist: false,
                    });
                }
            }
        }

        // A group-toggle click recorded here is applied after the loop (the loop
        // borrows the regions mutably; the panels are mutated below).
        let mut toggle_group: Option<usize> = None;
        // A rebind-row click recorded here (setting key + value label) starts a
        // capture after the loop, for the same borrow reason.
        let mut start_capture: Option<(String, Option<AssetId>)> = None;
        // A dropdown-row click recorded here opens its floating list after the
        // loop (resolving the value label + options needs ctx, borrowed by the
        // loop).
        let mut start_open: Option<OpenRequest> = None;
        // Setting rows the engine disabled this frame (e.g. show_fps / show_vram
        // while the "Display performance stats" master is off): inert and grayed,
        // like the init-time capability gating but driven at runtime. Refresh the
        // owned cache only when the published set changes (a cheap set compare
        // otherwise), so the resource borrow ends before the mutable region loop
        // without cloning it every frame.
        let disabled_changed = match ctx.resource::<crate::ecs::DisabledSettingRows>() {
            Some(d) => d.0 != self.disabled_rows_cache,
            None => !self.disabled_rows_cache.is_empty(),
        };
        if disabled_changed {
            self.disabled_rows_cache = ctx
                .resource::<crate::ecs::DisabledSettingRows>()
                .map(|d| d.0.clone())
                .unwrap_or_default();
        }

        // A nav pulse moves the focus cursor (or adjusts the focused value
        // row); only pulse frames pay for the target derivation.
        if let Some(dir) = nav {
            self.step_focus(dir, active_screen, ctx);
        }
        // Confirm fires the focused region in the loop below; with no focus it
        // falls back to a full-canvas region ("press anywhere" advance).
        let focus_index = self.focus.as_ref().map(|f| f.index);
        let confirm_fallback = confirm && focus_index.is_none();
        let mut confirm_used = false;

        // Resolve each followed label's (y, is-empty) in one query pass, so the
        // loop below reads a map instead of scanning every TextLabel per region.
        self.follow_labels.clear();
        if !self.follow_label_ids.is_empty() {
            for l in ctx.query::<TextLabel>() {
                if self.follow_label_ids.contains(&l.asset_id) {
                    self.follow_labels
                        .entry(l.asset_id)
                        .or_insert((l.y, l.content.is_empty()));
                }
            }
        }
        let follow_labels = &self.follow_labels;

        let disabled_rows = &self.disabled_rows_cache;
        for (i, entry) in self.regions.iter_mut().enumerate() {
            // A region is inert this frame when it cannot hover or fire:
            //   - the scrollbar thumb is being dragged (no region reacts),
            //   - it is a slider track (driven by the drag pass above),
            //   - its screen is not the active one (behind an overlay, or screen-less
            //     while a screen is shown),
            //   - its scroll-content row is collapsed, or
            //   - the engine disabled its setting row at runtime (grayed).
            // Restore any hover styling first so a region hovered when it goes
            // inert (e.g. the clicked button whose screen is being hidden) does not
            // strand its hover color, then clear the hover flag and skip it.
            let disabled = !disabled_rows.is_empty()
                && entry
                    .region
                    .action
                    .strip_prefix("setting:")
                    .is_some_and(|rest| {
                        disabled_rows.contains(rest.split(':').next().unwrap_or(""))
                    });
            // A follow-label region tracks its label's y and goes inert while
            // the label is empty (a hidden menu entry catches no clicks).
            let follow_inert = if let Some((label_id, offset)) = entry.follow {
                match follow_labels.get(&label_id).copied() {
                    Some((ly, empty)) => {
                        entry.region.y = ly + offset;
                        empty
                    }
                    None => true,
                }
            } else {
                false
            };
            // A focused slider track stays in the pass so it shows the focus
            // highlight; its confirm dispatch is a recognised no-op and the
            // cursor cannot fire it while focus is set.
            let pad_focused = focus_index == Some(i);
            let inert = thumb_active
                || (entry.slider_key.is_some() && !pad_focused)
                || entry.screen != active_screen
                || (entry.scroll_row.is_some() && entry.hidden)
                || disabled
                || follow_inert;
            if inert {
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

            // Overlay (screen-owned) regions hit-test in reference space (through
            // the region's own `fit`); HUD regions in window pixels. A region
            // spanning the whole reference canvas covers the full window (so a
            // full-canvas advance region catches clicks in the letterbox too).
            let full_window = entry.screen.is_some() && region_covers_canvas(&entry.region);
            let (qx, qy) = if entry.screen.is_none() {
                (mx, my)
            } else {
                match entry.fit {
                    SpriteFit::Bottom => overlay_bottom.inverse(mx, my),
                    SpriteFit::Cover => overlay_cover.inverse(mx, my),
                    SpriteFit::Fit => overlay.inverse(mx, my),
                }
            };
            let group_toggle = entry.group_toggle;
            let r = &entry.region;
            let mut mouse_hovered = if full_window {
                mx >= 0.0 && mx < vw && my >= 0.0 && my < vh
            } else {
                qx >= r.x && qx < r.x + r.width && qy >= r.y && qy < r.y + r.height
            };
            // A scroll-content region only counts as hovered inside its band, so
            // a row scrolled past the edge does not catch clicks over the chrome.
            if let Some((pi, _)) = entry.scroll_row
                && let Some(band) = panel_bands.get(pi)
            {
                mouse_hovered = mouse_hovered && point_in_rect(qx, qy, *band);
            }
            // While the focus cursor is set it owns the hover slot: the
            // focused region styles + fires and the cursor's row does neither
            // (mouse movement clears the focus first, so this never masks a
            // live hover). Confirm with no focus falls through to a
            // full-canvas region, once.
            let hovered = if focus_index.is_some() {
                pad_focused
            } else {
                mouse_hovered
            };
            let fallback_fire = confirm_fallback && full_window && !confirm_used;
            let fire = if focus_index.is_some() {
                pad_focused && confirm
            } else {
                (mouse_hovered && clicked) || fallback_fire
            };

            // Apply hover styling on hover-in, restore the captured style on
            // hover-out.
            if hovered && !entry.was_hovered {
                set_label_style(ctx, r.label, r.hover_color, r.hover_scale);
            } else if !hovered && entry.was_hovered {
                set_label_style(ctx, r.label, entry.original_color, entry.original_scale);
            }

            entry.was_hovered = hovered;

            if fire {
                if fallback_fire {
                    confirm_used = true;
                }
                // A group header toggles its panel's group (handled after the
                // loop) instead of firing an action.
                if let Some(gid) = group_toggle {
                    toggle_group = Some(gid);
                } else if let Some(key) =
                    crate::gfx::setting_action::key_with_verb(&r.action, "rebind")
                {
                    // A rebind row enters capture (started after the loop)
                    // instead of firing an action immediately.
                    start_capture = Some((key.to_string(), r.label));
                } else if let Some(key) =
                    crate::gfx::setting_action::key_with_verb(&r.action, "open")
                {
                    // A dropdown row opens its floating list (started after the
                    // loop) instead of firing an action. Snapshot the control
                    // rect + the row's un-hovered value style now.
                    start_open = Some(OpenRequest {
                        setting: key.to_string(),
                        value_label: r.label,
                        anchor: [r.x, r.y, r.width, r.height],
                        screen: entry.screen,
                        color: entry.original_color,
                        scale: entry.original_scale,
                    });
                } else if !r.action.is_empty()
                    && let Some(result) = fire_action(&r.action, r.label, ctx)
                {
                    return result;
                }
            }
        }

        // Begin a rebind capture for a clicked rebind row: stash the value
        // label's current text (to restore on cancel) and show the prompt for
        // the input kind the row captures.
        if let Some((setting_key, value_label)) = start_capture {
            let prev_text = value_label
                .and_then(|id| {
                    ctx.query::<TextLabel>()
                        .find(|l| l.asset_id == id)
                        .map(|l| l.content.clone())
                })
                .unwrap_or_default();
            if let Some(id) = value_label {
                let prompt = if setting_key.starts_with("pad_") {
                    PAD_REBIND_PROMPT
                } else {
                    REBIND_PROMPT
                };
                crate::ecs::by_asset_id::set_text(ctx, id, prompt);
            }
            self.capturing = Some(Capture {
                setting_key,
                value_label,
                prev_text,
            });
        }

        // Open a dropdown for a clicked dropdown row: seed its list from the
        // shared option registry + the value label's current text, then take
        // over input from the next frame.
        if let Some(req) = start_open {
            self.open_dropdown = Self::build_open_dropdown(req, ctx);
        }

        // Apply a recorded group toggle to the active screen's panel, then solve
        // every panel so the next frame draws + hit-tests the reflowed layout.
        if let Some(gid) = toggle_group
            && let Some(panel) = self.panels.iter_mut().find(|p| p.screen == active_screen)
            && let Some(g) = panel.groups.get_mut(gid)
        {
            g.collapsed = !g.collapsed;
        }
        self.apply_scroll_layout(ctx);

        // Publish the current dropdown state (a just-opened list, or `None` when
        // closed) for GraphicsSystem to draw next tick.
        self.publish_dropdown(ctx);

        StepResult::Continue
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
    fn step_open_dropdown(
        &mut self,
        input: &FrameInput,
        nav: Option<NavDirection>,
        pulses: DropdownPulses,
        ctx: &mut PipelineContext,
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
        if !input.left_button_down {
            state.thumb_drag = None;
        } else {
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
            let setting = state.setting.clone();
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
                    let setting = state.setting.clone();
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
    fn build_open_dropdown(
        req: OpenRequest,
        ctx: &mut PipelineContext,
    ) -> Option<OpenDropdownState> {
        let options: Vec<String> =
            if concinnity_core::gfx::settings::is_dynamic_dropdown(&req.setting) {
                // Today the only dynamic dropdown is `resolution`, whose modes
                // GraphicsSystem publishes at init.
                let modes = ctx.resource::<crate::ecs::DisplayModes>()?;
                if modes.0.is_empty() {
                    return None;
                }
                modes.0.iter().map(|m| m.label()).collect()
            } else {
                settings::options(&req.setting)?
                    .iter()
                    .map(|s| s.to_string())
                    .collect()
            };
        // The value label's font (for the list text) and current content (to
        // mark the selected option).
        let (font, current) = req
            .value_label
            .and_then(|id| {
                ctx.query::<TextLabel>()
                    .find(|l| l.asset_id == id)
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
    // GraphicsSystem to draw next tick (`None` while closed).
    fn publish_dropdown(&self, ctx: &mut PipelineContext) {
        let screen = self
            .open_dropdown
            .as_ref()
            .map(|s| crate::ecs::DropdownView {
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
        ctx.insert_resource(crate::ecs::OpenDropdown(screen));
    }

    // Whether the engine disabled this region's setting row at runtime
    // (mirrors the hit-test loop's gating).
    fn row_disabled(&self, entry: &RegionEntry) -> bool {
        !self.disabled_rows_cache.is_empty()
            && entry
                .region
                .action
                .strip_prefix("setting:")
                .is_some_and(|rest| {
                    self.disabled_rows_cache
                        .contains(rest.split(':').next().unwrap_or(""))
                })
    }

    // Advance the focus cursor for one directional pulse: Left/Right on a
    // focused value row adjust its setting in place; any other pulse moves the
    // focus to the nearest target, scrolling its panel row into the clip band.
    fn step_focus(
        &mut self,
        dir: NavDirection,
        active_screen: Option<AssetId>,
        ctx: &mut PipelineContext,
    ) {
        // Follow-label regions with an empty label are hidden menu entries;
        // resolve their label contents in one pass so they never focus.
        let follow_labels: std::collections::HashSet<AssetId> = self
            .regions
            .iter()
            .filter(|e| e.screen == active_screen)
            .filter_map(|e| e.follow.map(|(id, _)| id))
            .collect();
        let empty_labels: std::collections::HashSet<AssetId> = if follow_labels.is_empty() {
            std::collections::HashSet::new()
        } else {
            ctx.query::<TextLabel>()
                .filter(|l| follow_labels.contains(&l.asset_id) && l.content.is_empty())
                .map(|l| l.asset_id)
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
                rect: [e.region.x, e.region.y, e.region.width, e.region.height],
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
            && let Some(key) = t.setting.clone()
        {
            let op = match dir {
                NavDirection::Left => SettingOp::Prev,
                _ => SettingOp::Next,
            };
            let value_label = self.regions[f.index].region.label;
            ctx.events_mut::<SettingCommand>().send(SettingCommand {
                setting: key,
                op,
                value_label,
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

    fn apply_screen_command(&mut self, cmd: ScreenCommand, ctx: &mut PipelineContext) {
        let Some(transition) = self.screens.apply(cmd) else {
            return;
        };
        // A stack change dismisses any open dropdown and the focus cursor so
        // neither lingers over a different screen.
        self.open_dropdown = None;
        self.focus = None;
        self.apply_transition(transition, ctx);
        self.publish_screen_stack(ctx);
    }

    // Enact one stack transition on the world: flip the entering / leaving
    // screens' element visibility, move keyboard focus to the new top's
    // `focus` field, and announce the newly-topmost screen (AudioCue).
    fn apply_transition(
        &mut self,
        transition: screen::ScreenTransition,
        ctx: &mut PipelineContext,
    ) {
        for id in &transition.hidden {
            self.set_screen_visibility(*id, false, ctx);
        }
        for id in &transition.shown {
            self.set_screen_visibility(*id, true, ctx);
        }
        if transition.top_changed {
            // Focus follows the top of the stack: blur every field, then focus
            // the new top's `focus` target (if it names one).
            let focus = transition
                .new_top
                .and_then(|id| self.screens.meta(id))
                .and_then(|m| m.focus);
            for ti in ctx.query_mut::<crate::assets::TextInput>() {
                ti.focused = Some(ti.asset_id) == focus;
            }
        }
        if let Some(top) = transition.new_top {
            ctx.events_mut::<ScreenShown>()
                .send(ScreenShown { screen: top });
        }
    }

    // Publish the stack's derived per-frame state (draw layers, world-pause,
    // input capture) for the overlay build and InputSystem, which read it a
    // frame later -- the same one-frame lag the visibility flips have.
    fn publish_screen_stack(&self, ctx: &mut PipelineContext) {
        ctx.insert_resource(crate::ecs::ScreenStack {
            layers: self.screens.layers(),
            pauses_world: self.screens.pauses_world(),
            captures_input: self.screens.captures_input(),
        });
    }

    // Cancel a pending rebind capture, restoring the row's previous value text.
    fn cancel_capture(&mut self, ctx: &mut PipelineContext) {
        if let Some(cap) = self.capturing.take()
            && let Some(id) = cap.value_label
        {
            let prev = cap.prev_text.clone();
            crate::ecs::by_asset_id::set_text(ctx, id, &prev);
        }
    }

    fn set_screen_visibility(&self, screen_id: AssetId, visible: bool, ctx: &mut PipelineContext) {
        if let Some(ids) = self.sprites_by_screen.get(&screen_id) {
            for &id in ids {
                for s in ctx.query_mut::<Sprite>() {
                    if s.asset_id == id {
                        s.visible = visible;
                        break;
                    }
                }
            }
        }
        if let Some(ids) = self.labels_by_screen.get(&screen_id) {
            for &id in ids {
                for l in ctx.query_mut::<TextLabel>() {
                    if l.asset_id == id {
                        l.visible = visible;
                        break;
                    }
                }
            }
        }
        if let Some(ids) = self.text_inputs_by_screen.get(&screen_id) {
            for &id in ids {
                for ti in ctx.query_mut::<crate::assets::TextInput>() {
                    if ti.asset_id == id {
                        ti.visible = visible;
                        break;
                    }
                }
            }
        }
    }

    // Drain the world's ScrollPanels into runtime state: snapshot each row
    // element's authored y (so the reflow is `base + dy`), translate the
    // `i32` group index into an `Option`, and bucket each HitRegion into the
    // panel row whose band its centre falls in (so the region reflows + gates
    // with that row). Runs once at init, after HitRegions are drained.
    fn init_panels(&mut self, ctx: &mut PipelineContext) {
        let panels = ctx.drain::<ScrollPanel>();
        if panels.is_empty() {
            return;
        }
        // Snapshot the authored y of every element any panel row references.
        let wanted: std::collections::HashSet<AssetId> = panels
            .iter()
            .flat_map(|p| p.rows.iter().flat_map(|r| r.elements.iter().copied()))
            .collect();
        let mut elem_y: HashMap<AssetId, f32> = HashMap::new();
        for s in ctx.query::<Sprite>() {
            if wanted.contains(&s.asset_id) {
                elem_y.insert(s.asset_id, s.y);
            }
        }
        for l in ctx.query::<TextLabel>() {
            if wanted.contains(&l.asset_id) {
                elem_y.insert(l.asset_id, l.y);
            }
        }

        for p in panels {
            let rows = p
                .rows
                .iter()
                .map(|r| {
                    let base_ys = r
                        .elements
                        .iter()
                        .map(|id| elem_y.get(id).copied().unwrap_or(r.base_y))
                        .collect();
                    RowState {
                        elements: r.elements.clone(),
                        base_ys,
                        base_y: r.base_y,
                        height: r.height,
                        group: (r.group >= 0).then_some(r.group as usize),
                    }
                })
                .collect();
            let groups = p
                .groups
                .iter()
                .map(|g| GroupState {
                    collapsed: g.collapsed,
                    header: g.header,
                    title: g.title.clone(),
                })
                .collect();
            self.panels.push(PanelState {
                screen: p.screen,
                band: [p.x, p.y, p.width, p.height],
                rows,
                groups,
                thumb: p.thumb,
                track: p.track,
                track_x: p.track_x,
                track_y: p.track_y,
                track_w: p.track_w,
                track_h: p.track_h,
                scroll: 0.0,
                content_height: 0.0,
                thumb_h: 0.0,
            });
        }

        // Bucket each panel-content region into its row by centre y. Only
        // content regions (a settings action or a group toggle) are bucketed;
        // chrome regions (tabs, Back -- `screen:show`) are left fixed even when an
        // overflow row's authored y reaches their position. Panels read
        // immutably while the regions are mutated (disjoint fields).
        let panels = &self.panels;
        for entry in self.regions.iter_mut() {
            let is_content =
                entry.region.action.starts_with("setting:") || entry.group_toggle.is_some();
            if !is_content {
                continue;
            }
            let cy = entry.region.y + entry.region.height * 0.5;
            'find: for (pi, panel) in panels.iter().enumerate() {
                if panel.screen != entry.screen {
                    continue;
                }
                for (ri, row) in panel.rows.iter().enumerate() {
                    if cy >= row.base_y && cy < row.base_y + row.height {
                        entry.scroll_row = Some((pi, ri));
                        break 'find;
                    }
                }
            }
        }
    }

    // The reference-space rectangle of a panel's scrollbar thumb at its current
    // scroll, or `None` if the panel does not overflow (no thumb to grab).
    fn thumb_rect(panel: &PanelState) -> Option<[f32; 4]> {
        if panel.content_height <= 0.0 || panel.thumb_h >= panel.track_h {
            return None;
        }
        let offset_frac = (panel.scroll / panel.content_height).clamp(0.0, 1.0);
        let thumb_y = panel.track_y + offset_frac * panel.track_h;
        Some([panel.track_x, thumb_y, panel.track_w, panel.thumb_h])
    }

    // Apply scroll-wheel + scrollbar-thumb input to the active screen's panel.
    // Returns true while the thumb is being dragged so the caller suppresses
    // the slider + click passes. The solver clamps the resulting scroll
    // offset. (The arrow keys move the focus cursor instead, which scrolls its
    // own row into view.)
    fn handle_scroll_input(
        &mut self,
        input: &FrameInput,
        active_screen: Option<AssetId>,
        overlay: &OverlayTransform,
    ) -> bool {
        let (qx, qy) = overlay.inverse(input.mouse_x, input.mouse_y);
        let active_panel = self.panels.iter().position(|p| p.screen == active_screen);

        // Wheel: scroll the active panel while the cursor is over its band.
        if input.scroll_delta != 0.0
            && let Some(pi) = active_panel
            && point_in_rect(qx, qy, self.panels[pi].band)
        {
            self.panels[pi].scroll += input.scroll_delta * WHEEL_SCROLL_SPEED;
        }

        // Thumb drag: begin on the press edge over the thumb, then map the
        // cursor's y to a scroll offset for the rest of the press.
        if !input.left_button_down {
            self.thumb_drag = None;
        } else {
            if self.thumb_drag.is_none()
                && input.left_click
                && let Some(pi) = active_panel
                && let Some(rect) = Self::thumb_rect(&self.panels[pi])
                && point_in_rect(qx, qy, rect)
            {
                self.thumb_drag = Some((pi, qy - rect[1]));
            }
            if let Some((pi, grab)) = self.thumb_drag {
                let panel = &mut self.panels[pi];
                let travel = (panel.track_h - panel.thumb_h).max(0.0);
                let max_scroll = (panel.content_height - panel.band[3]).max(0.0);
                if travel > 0.0 && max_scroll > 0.0 {
                    let thumb_top = (qy - grab).clamp(panel.track_y, panel.track_y + travel);
                    let frac = (thumb_top - panel.track_y) / travel;
                    panel.scroll = frac * max_scroll;
                }
            }
        }
        self.thumb_drag.is_some()
    }

    // Solve every panel's vertical layout and write the result back: element y +
    // visibility, region reflow + hidden flag, the scrollbar thumb position +
    // size, and each group header's `+`/`-` prefix. Only the active screen's panel
    // writes (an inactive screen's elements stay hidden by the screen system). Runs
    // at init and at the end of each step so the next frame draws + hit-tests the
    // reflowed positions consistently.
    fn apply_scroll_layout(&mut self, ctx: &mut PipelineContext) {
        if self.panels.is_empty() {
            return;
        }
        let active = self.screens.top_capture();

        // Accumulate component writes into the reused scratch maps, then apply
        // in one pass per component type.
        self.layout.sprites.clear();
        self.layout.labels.clear();
        self.layout.solved_rows.clear();

        for panel in self.panels.iter_mut() {
            let panel_active = panel.screen == active;
            self.layout.collapsed.clear();
            self.layout
                .collapsed
                .extend(panel.groups.iter().map(|g| g.collapsed));
            self.layout.specs.clear();
            self.layout.specs.extend(panel.rows.iter().map(|r| RowSpec {
                height: r.height,
                group: r.group,
            }));
            let solved = scroll_layout::solve(
                &self.layout.specs,
                &self.layout.collapsed,
                panel.band[3],
                panel.scroll,
            );
            panel.scroll = solved.scroll;
            panel.content_height = solved.content_height;
            panel.thumb_h = solved.thumb_frac * panel.track_h;

            if panel_active {
                for (ri, row) in panel.rows.iter().enumerate() {
                    let pl = solved.rows[ri];
                    for (k, id) in row.elements.iter().enumerate() {
                        let y = row.base_ys[k] + pl.dy;
                        let s = self.layout.sprites.entry(*id).or_default();
                        s.y = Some(y);
                        s.visible = Some(pl.visible);
                        let l = self.layout.labels.entry(*id).or_default();
                        l.y = Some(y);
                        l.visible = Some(pl.visible);
                    }
                }
                let scrollable = solved.scrollable();
                if let Some(thumb) = panel.thumb {
                    let thumb_y = panel.track_y + solved.thumb_offset_frac * panel.track_h;
                    let s = self.layout.sprites.entry(thumb).or_default();
                    s.y = Some(thumb_y);
                    s.height = Some(panel.thumb_h);
                    s.visible = Some(scrollable);
                }
                if let Some(track) = panel.track {
                    self.layout.sprites.entry(track).or_default().visible = Some(scrollable);
                }
                for g in &panel.groups {
                    if let Some(h) = g.header {
                        let prefix = if g.collapsed { "+ " } else { "- " };
                        self.layout.labels.entry(h).or_default().content =
                            Some(format!("{prefix}{}", g.title));
                    }
                }
            }
            self.layout.solved_rows.push((panel_active, solved.rows));
        }

        // Reflow each panel-owned region in memory (positions the click loop
        // hit-tests against next frame).
        for entry in self.regions.iter_mut() {
            if let Some((pi, ri)) = entry.scroll_row
                && let Some((panel_active, rows)) = self.layout.solved_rows.get(pi)
                && *panel_active
            {
                let pl = rows[ri];
                entry.region.y = entry.region_base_y + pl.dy;
                entry.hidden = !pl.visible;
            }
        }

        // Apply the accumulated component writes.
        for s in ctx.query_mut::<Sprite>() {
            if let Some(u) = self.layout.sprites.get(&s.asset_id) {
                if let Some(y) = u.y {
                    s.y = y;
                }
                if let Some(h) = u.height {
                    s.height = h;
                }
                if let Some(vis) = u.visible {
                    s.visible = vis;
                }
            }
        }
        for l in ctx.query_mut::<TextLabel>() {
            if let Some(u) = self.layout.labels.get_mut(&l.asset_id) {
                if let Some(y) = u.y {
                    l.y = y;
                }
                if let Some(vis) = u.visible {
                    l.visible = vis;
                }
                if let Some(content) = u.content.take() {
                    l.content = content;
                }
            }
        }
    }
}

// The collapsible-group index of a group-toggle action (`group:toggle:<gid>`),
// or `None`. A region with `Some` here flips its panel's group instead of
// firing an action.
fn group_toggle_from_action(action: &str) -> Option<usize> {
    action.strip_prefix("group:toggle:")?.parse::<usize>().ok()
}

// The one-shot edges an open dropdown reacts to this frame: pick the hovered
// option, dismiss the list, or re-hover from a cursor move.
struct DropdownPulses {
    confirm: bool,
    escape: bool,
    cursor_moved: bool,
}

// Whether a point lies inside an `[x, y, width, height]` rectangle.
fn point_in_rect(x: f32, y: f32, rect: [f32; 4]) -> bool {
    x >= rect[0] && x < rect[0] + rect[2] && y >= rect[1] && y < rect[1] + rect[3]
}

// Whether a region spans the whole reference canvas (a full-screen "click
// anywhere" region). Such a region covers the entire live window, including any
// letterbox margin, rather than only the fitted canvas rect.
fn region_covers_canvas(r: &HitRegion) -> bool {
    r.x <= 0.0
        && r.y <= 0.0
        && r.x + r.width >= UI_REFERENCE_SIZE[0]
        && r.y + r.height >= UI_REFERENCE_SIZE[1]
}

// Write the given color + scale onto a region's referenced label, if any.
// Drives hover-in (hover style), hover-out (captured style), and the restore
// applied when a hovered region goes inert (its screen hides, its row collapses,
// or it is disabled) so its hover styling never strands on the label.
fn set_label_style(
    ctx: &mut PipelineContext,
    label: Option<AssetId>,
    color: Option<[f32; 3]>,
    scale: Option<f32>,
) {
    let Some(label_id) = label else {
        return;
    };
    for lbl in ctx.query_mut::<TextLabel>() {
        if lbl.asset_id == label_id {
            if let Some(c) = color {
                lbl.color = c;
            }
            if let Some(s) = scale {
                lbl.scale = s;
            }
            break;
        }
    }
}

// Parse and execute an action string. Returns Some(StepResult) when the
// action produces an engine-level result (e.g. Quit), None otherwise. `label`
// is the firing region's referenced TextLabel (the value display for a
// settings row), forwarded so GraphicsSystem can update it.
fn fire_action(
    action: &str,
    label: Option<AssetId>,
    ctx: &mut PipelineContext,
) -> Option<StepResult> {
    if action == "quit" {
        return Some(StepResult::Stop);
    }
    if let Some(scene_ref) = action.strip_prefix("scene:") {
        // The build rewrites `scene:<name>` to `scene:<id>` so the target is
        // a plain integer here (see concinnity_cook::pipeline::resolve_scene_refs).
        match scene_ref.parse::<u32>() {
            Ok(id) => {
                ctx.events_mut::<SceneCommand>().send(SceneCommand {
                    scene: AssetId(id),
                    transition: "FadeBlack".to_string(),
                });
                // Dismiss every open screen on a scene change: the user has
                // chosen a new context, so the whole overlay stack clears.
                ctx.events_mut::<ScreenCommand>().send(ScreenCommand::Clear);
            }
            Err(_) => tracing::warn!("UiInputSystem: unresolved scene action '{}'", action),
        }
        return None;
    }
    if action == "screen:hide" {
        ctx.events_mut::<ScreenCommand>().send(ScreenCommand::Hide);
        return None;
    }
    if let Some(screen_ref) = action.strip_prefix("screen:show:") {
        match screen_ref.parse::<u32>() {
            Ok(id) => ctx
                .events_mut::<ScreenCommand>()
                .send(ScreenCommand::Show(AssetId(id))),
            Err(_) => tracing::warn!("UiInputSystem: unresolved screen action '{}'", action),
        }
        return None;
    }
    if let Some(screen_ref) = action.strip_prefix("screen:toggle:") {
        match screen_ref.parse::<u32>() {
            Ok(id) => ctx
                .events_mut::<ScreenCommand>()
                .send(ScreenCommand::Toggle(AssetId(id))),
            Err(_) => tracing::warn!("UiInputSystem: unresolved screen action '{}'", action),
        }
        return None;
    }
    if let Some(screen_ref) = action.strip_prefix("screen:push:") {
        match screen_ref.parse::<u32>() {
            Ok(id) => ctx
                .events_mut::<ScreenCommand>()
                .send(ScreenCommand::Push(AssetId(id))),
            Err(_) => tracing::warn!("UiInputSystem: unresolved screen action '{}'", action),
        }
        return None;
    }
    // story:start | story:advance | story:choose:<i> | the quick-row and
    // slot-overlay controls -- the story system reads the StoryCommand and
    // moves through its compiled graph.
    if let Some(rest) = action.strip_prefix("story:") {
        let cmd = match rest {
            "start" => Some(StoryCommand::Start),
            "continue" => Some(StoryCommand::Continue),
            "advance" => Some(StoryCommand::Advance),
            "auto" => Some(StoryCommand::ToggleAuto),
            "skip" => Some(StoryCommand::ToggleSkip),
            "log" => Some(StoryCommand::ToggleLog),
            "save" => Some(StoryCommand::OpenSave),
            "load" => Some(StoryCommand::OpenLoad),
            "pause" => Some(StoryCommand::TogglePause),
            "settings" => Some(StoryCommand::OpenSettings),
            "settings_back" => Some(StoryCommand::CloseSettings),
            _ => match rest.split_once(':') {
                Some(("choose", i)) => i.parse::<usize>().ok().map(StoryCommand::Choose),
                Some(("slot", i)) => i.parse::<usize>().ok().map(StoryCommand::Slot),
                _ => None,
            },
        };
        match cmd {
            Some(cmd) => ctx.events_mut::<StoryCommand>().send(cmd),
            None => tracing::warn!("UiInputSystem: malformed story action '{}'", action),
        }
        return None;
    }
    // setting:<key>:next|prev -- cycle a graphics setting. GraphicsSystem
    // reads the SettingCommand to apply, persist, and refresh the value label.
    if let Some(rest) = action.strip_prefix("setting:") {
        match rest.rsplit_once(':') {
            Some((key, "next")) | Some((key, "prev")) if !key.is_empty() => {
                let op = if rest.ends_with(":prev") {
                    SettingOp::Prev
                } else {
                    SettingOp::Next
                };
                ctx.events_mut::<SettingCommand>().send(SettingCommand {
                    setting: key.to_string(),
                    op,
                    value_label: label,
                    // A cycle is one discrete change: always persisted.
                    persist: true,
                });
            }
            // Slider drags, key rebinds, and dropdown opens are driven by their
            // own passes (the drag pass, the capture flow, the dropdown pass),
            // not the click-to-fire path, so they never reach here from a
            // HitRegion click; recognise them so a stray binding does not log a
            // false "malformed" warning.
            Some((key, "drag")) | Some((key, "rebind")) | Some((key, "open"))
                if !key.is_empty() => {}
            _ => tracing::warn!("UiInputSystem: malformed setting action '{}'", action),
        }
        return None;
    }
    tracing::warn!("UiInputSystem: unknown action '{}'", action);
    None
}

#[cfg(test)]
mod tests {
    // UiInputSystem is internal: each test seeds the gating components
    // (HitRegion / Screen / KeyBinding) before `world.start()`, which constructs
    // the system from them via the build schedule.
    use super::*;
    use crate::assets::{HitRegion, ScrollGroup, ScrollRow, TextLabel};
    use crate::ecs::World;

    fn make_frame_input(mx: f32, my: f32, clicked: bool) -> FrameInput {
        FrameInput {
            mouse_x: mx,
            mouse_y: my,
            left_click: clicked,
            ..Default::default()
        }
    }

    // The ScreenCommand UiInputSystem sent this step, read with a fresh cursor so
    // the system's own cursor (which applies them a frame later) is untouched.
    // Returns the first if several were sent.
    fn produced_screen_command(world: &World) -> Option<ScreenCommand> {
        let mut cursor = crate::ecs::EventCursor::default();
        world
            .events::<ScreenCommand>()
            .and_then(|e| e.read(&mut cursor).next().cloned())
    }

    // Every SettingCommand the system sent, read with a fresh cursor (in send
    // order). GraphicsSystem applies these, but these tests run UiInputSystem
    // alone, so they inspect the queue directly via .first()/.last()/.is_empty().
    fn produced_setting_commands(world: &World) -> Vec<SettingCommand> {
        let mut cursor = crate::ecs::EventCursor::default();
        world
            .events::<SettingCommand>()
            .map(|e| e.read(&mut cursor).cloned().collect())
            .unwrap_or_default()
    }

    // A screen-owned TextLabel used as a scroll-panel element.
    fn panel_label(id: u32, y: f32, screen: AssetId, content: &str) -> TextLabel {
        TextLabel {
            asset_id: AssetId(id),
            font: None,
            content: content.to_string(),
            x: 0.0,
            y,
            color: [1.0, 1.0, 1.0],
            scale: 1.0,
            centered: false,
            align: crate::assets::TextAlign::Left,
            fit: crate::assets::SpriteFit::Fit,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            screen: Some(screen),
            wrap_width: 0.0,
            max_lines: 0,
        }
    }

    fn label_field<T>(world: &World, id: AssetId, f: impl Fn(&TextLabel) -> T) -> T {
        world
            .query::<TextLabel>()
            .find(|l| l.asset_id == id)
            .map(f)
            .unwrap()
    }

    #[test]
    fn hover_applies_and_restores_label_style() {
        let mut world = World::new_empty();

        world.add_component(TextLabel {
            asset_id: AssetId(1),
            font: None,
            content: "Hello".to_string(),
            x: 0.0,
            y: 0.0,
            color: [1.0, 1.0, 1.0],
            scale: 1.0,
            centered: false,
            align: crate::assets::TextAlign::Left,
            fit: crate::assets::SpriteFit::Fit,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            screen: None,
            wrap_width: 0.0,
            max_lines: 0,
        });
        world.add_component(HitRegion {
            x: 10.0,
            y: 10.0,
            width: 100.0,
            height: 40.0,
            label: Some(AssetId(1)),
            hover_color: Some([1.0, 0.0, 0.0]),
            hover_scale: Some(2.0),
            action: String::new(),
            drag_handle: None,
            screen: None,
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.start().unwrap();

        // Hover over the region.
        world.add_component(make_frame_input(50.0, 30.0, false));
        world.step();

        // Label should be styled.
        let lbl_color = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == AssetId(1))
            .map(|l| l.color)
            .unwrap();
        assert_eq!(lbl_color, [1.0, 0.0, 0.0]);

        // Move cursor away.
        world.add_component(make_frame_input(0.0, 0.0, false));
        world.step();

        let lbl_color_after = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == AssetId(1))
            .map(|l| l.color)
            .unwrap();
        assert_eq!(lbl_color_after, [1.0, 1.0, 1.0]);
    }

    // Clicking a menu button hovers its label (hover color) and switches away to
    // another screen the same frame. The next frame the button's screen is hidden;
    // its hover color must be restored, not stranded, so it is not still
    // highlighted when its screen is shown again.
    #[test]
    fn hover_style_restored_when_region_screen_is_hidden() {
        let mut world = World::new_empty();
        let menu = AssetId(80);
        let settings = AssetId(81);
        world.add_component(Screen {
            asset_id: menu,
            initial: true,
            fade_in_secs: 0.0,
            ..Default::default()
        });
        world.add_component(Screen {
            asset_id: settings,
            initial: false,
            fade_in_secs: 0.0,
            ..Default::default()
        });
        // The menu's "Settings" label + its hit region (screen-owned).
        world.add_component(TextLabel {
            asset_id: AssetId(1),
            font: None,
            content: "Settings".to_string(),
            x: 0.0,
            y: 0.0,
            color: [1.0, 1.0, 1.0],
            scale: 1.0,
            centered: false,
            align: crate::assets::TextAlign::Left,
            fit: crate::assets::SpriteFit::Fit,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            screen: Some(menu),
            wrap_width: 0.0,
            max_lines: 0,
        });
        world.add_component(HitRegion {
            x: 10.0,
            y: 10.0,
            width: 100.0,
            height: 40.0,
            label: Some(AssetId(1)),
            hover_color: Some([1.0, 0.85, 0.3]),
            hover_scale: Some(1.0),
            action: "screen:show:81".to_string(),
            drag_handle: None,
            screen: Some(menu),
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.start().unwrap();

        // Hover + click the Settings button (identity overlay at viewport [0,0]):
        // the label takes the hover color and the click sends Show(settings).
        world.add_component(FrameInput {
            mouse_x: 50.0,
            mouse_y: 30.0,
            left_click: true,
            ..Default::default()
        });
        world.step();
        assert_eq!(
            label_field(&world, AssetId(1), |l| l.color),
            [1.0, 0.85, 0.3],
            "hovered button takes the hover color"
        );

        // Next frame applies Show(settings): the menu (and its Settings label) is
        // hidden. The hover color must be restored despite the screen being hidden.
        world.add_component(FrameInput::default());
        world.step();
        assert_eq!(
            label_field(&world, AssetId(1), |l| l.color),
            [1.0, 1.0, 1.0],
            "hover color restored when the region's screen is hidden"
        );
    }

    // A screen-owned dropdown row (window_mode has three options, so its
    // `:open` region opens a floating list). Clicking the control opens the list
    // (published as an OpenDropdown resource, no command yet); clicking an option
    // sends a SetIndex command and closes.
    fn dropdown_world() -> (World, AssetId) {
        let screen = AssetId(9);
        let mut world = World::new_empty();
        world.add_component(Screen {
            asset_id: screen,
            initial: true,
            ..Default::default()
        });
        // The row's value label (screen-owned), currently "Windowed" (option 0).
        world.add_component(TextLabel {
            asset_id: AssetId(1),
            font: None,
            content: "Windowed".to_string(),
            x: 0.0,
            y: 0.0,
            color: [0.85, 0.85, 0.85],
            scale: 1.0,
            centered: false,
            align: crate::assets::TextAlign::Left,
            fit: crate::assets::SpriteFit::Fit,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            screen: Some(screen),
            wrap_width: 0.0,
            max_lines: 0,
        });
        // The control button whose click opens the list.
        world.add_component(HitRegion {
            x: 400.0,
            y: 100.0,
            width: 200.0,
            height: 40.0,
            label: Some(AssetId(1)),
            hover_color: Some([1.0, 0.85, 0.3]),
            hover_scale: Some(1.0),
            action: "setting:window_mode:open".to_string(),
            drag_handle: None,
            screen: Some(screen),
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.start().unwrap();
        (world, screen)
    }

    fn dropdown_is_open(world: &World) -> bool {
        world
            .resource::<crate::ecs::OpenDropdown>()
            .and_then(|d| d.0.as_ref())
            .is_some()
    }

    #[test]
    fn dropdown_opens_then_picks_an_option() {
        let (mut world, _) = dropdown_world();

        // Click the control button (default viewport is identity, so window
        // coords are reference coords): the list opens, no command yet.
        world.add_component(make_frame_input(500.0, 120.0, true));
        world.step();
        assert!(produced_setting_commands(&world).is_empty());
        let open = world.resource::<crate::ecs::OpenDropdown>().unwrap();
        let dv = open.0.as_ref().expect("dropdown should be open");
        assert_eq!(dv.options.len(), 3);
        assert_eq!(dv.selected, 0, "current value 'Windowed' is option 0");

        // The list opens below the button (rows at y = 140, 180, 220, height 40).
        // Click the second option ("Borderless") at y = 200.
        world.add_component(make_frame_input(500.0, 200.0, true));
        world.step();
        let cmds = produced_setting_commands(&world);
        assert_eq!(cmds.len(), 1);
        assert_eq!(cmds[0].setting, "window_mode");
        assert!(matches!(cmds[0].op, SettingOp::SetIndex(1)));
        assert!(!dropdown_is_open(&world), "picking closes the list");
    }

    // A dropdown over a runtime-enumerated list longer than the shown window
    // (a `resolution` row with 20 display modes): opening centers the current
    // selection, the wheel scrolls the window instead of dismissing, and a
    // click picks the OPTION under the row (not the raw row index).
    fn scrolled_dropdown_world() -> World {
        let screen = AssetId(9);
        let mut world = World::new_empty();
        world.add_component(Screen {
            asset_id: screen,
            initial: true,
            ..Default::default()
        });
        // 20 modes, 1000x100 (0Hz) .. 1000x2000 (0Hz); the row's value label
        // currently shows the 11th (index 10).
        let modes: Vec<crate::gfx::display_mode::DisplayMode> = (1..=20)
            .map(|i| crate::gfx::display_mode::DisplayMode {
                width: 1000,
                height: i * 100,
                refresh_hz: 0,
            })
            .collect();
        world.add_component(TextLabel {
            asset_id: AssetId(1),
            font: None,
            content: modes[10].label(),
            x: 0.0,
            y: 0.0,
            color: [0.85, 0.85, 0.85],
            scale: 1.0,
            centered: false,
            align: crate::assets::TextAlign::Left,
            fit: crate::assets::SpriteFit::Fit,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            screen: Some(screen),
            wrap_width: 0.0,
            max_lines: 0,
        });
        world.add_component(HitRegion {
            x: 400.0,
            y: 100.0,
            width: 200.0,
            height: 40.0,
            label: Some(AssetId(1)),
            hover_color: Some([1.0, 0.85, 0.3]),
            hover_scale: Some(1.0),
            action: "setting:resolution:open".to_string(),
            drag_handle: None,
            screen: Some(screen),
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.start().unwrap();
        world.insert_resource(crate::ecs::DisplayModes(modes));
        world
    }

    #[test]
    fn dropdown_scrolls_instead_of_dismissing() {
        let mut world = scrolled_dropdown_world();

        // Open: the full option list is carried, the window starts centered on
        // the selection (10 - MAX_VISIBLE/2).
        world.add_component(make_frame_input(500.0, 120.0, true));
        world.step();
        let center = 10 - dropdown::MAX_VISIBLE / 2;
        {
            let open = world.resource::<crate::ecs::OpenDropdown>().unwrap();
            let dv = open.0.as_ref().expect("dropdown should be open");
            assert_eq!(dv.options.len(), 20);
            assert_eq!(dv.selected, 10);
            assert_eq!(dv.first, center);
        }

        // Wheel: the list stays open and the window moves (40px * speed 2.0 /
        // 40px rows = +2 rows), clamped to the scrollable range.
        world.add_component(FrameInput {
            mouse_x: 500.0,
            mouse_y: 200.0,
            scroll_delta: 40.0,
            ..Default::default()
        });
        world.step();
        {
            let open = world.resource::<crate::ecs::OpenDropdown>().unwrap();
            let dv = open.0.as_ref().expect("scrolling must not dismiss");
            assert_eq!(dv.first, center + 2);
        }

        // Click the top shown row (y 140..180): the pick is the OPTION at the
        // scrolled window's top, not row 0 of the full list.
        world.add_component(make_frame_input(500.0, 150.0, true));
        world.step();
        let cmds = produced_setting_commands(&world);
        assert_eq!(cmds.len(), 1);
        assert_eq!(cmds[0].setting, "resolution");
        assert!(matches!(cmds[0].op, SettingOp::SetIndex(i) if i == center + 2));
        assert!(!dropdown_is_open(&world));
    }

    #[test]
    fn dropdown_outside_click_dismisses_without_command() {
        let (mut world, _) = dropdown_world();

        world.add_component(make_frame_input(500.0, 120.0, true));
        world.step();
        assert!(dropdown_is_open(&world));

        // Click far from both the list and the button: it dismisses, no command.
        world.add_component(make_frame_input(50.0, 600.0, true));
        world.step();
        assert!(produced_setting_commands(&world).is_empty());
        assert!(!dropdown_is_open(&world));
    }

    // The open list's first shown option, from the published OpenDropdown
    // resource.
    fn dropdown_first(world: &World) -> usize {
        world
            .resource::<crate::ecs::OpenDropdown>()
            .and_then(|d| d.0.as_ref())
            .map(|dv| dv.first)
            .expect("dropdown should be open")
    }

    // Dragging the open list's scrollbar thumb scrolls the window: a press on
    // the thumb neither picks nor dismisses, the held cursor's y drives the
    // window, and after release a click picks again.
    #[test]
    fn dropdown_thumb_drag_scrolls_window_without_picking() {
        let mut world = scrolled_dropdown_world();

        // Open: 20 options, window starts at first = 6 (selection centered).
        // The list is [400, 140, 200, 320]; the 28px thumb sits at y = 286
        // (halfway along its 292px travel, first 6 of max 12).
        world.add_component(make_frame_input(500.0, 120.0, true));
        world.step();
        assert_eq!(dropdown_first(&world), 6);

        // Press on the thumb (thumb x 584..598): the drag begins, nothing is
        // picked even though an option row sits under the cursor.
        world.add_component(FrameInput {
            mouse_x: 590.0,
            mouse_y: 290.0,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        });
        world.step();
        assert!(dropdown_is_open(&world), "a thumb press must not pick");
        assert!(produced_setting_commands(&world).is_empty());
        assert_eq!(dropdown_first(&world), 6, "grab does not jump the window");

        // Drag past the bottom of the list: the window clamps to the end.
        world.add_component(FrameInput {
            mouse_x: 590.0,
            mouse_y: 600.0,
            left_button_down: true,
            ..Default::default()
        });
        world.step();
        assert!(dropdown_is_open(&world));
        assert_eq!(dropdown_first(&world), 12);

        // Release, then click the top shown row: picking works again and maps
        // to the dragged window (option 12).
        world.add_component(make_frame_input(590.0, 400.0, false));
        world.step();
        world.add_component(make_frame_input(500.0, 150.0, true));
        world.step();
        let cmds = produced_setting_commands(&world);
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0].op, SettingOp::SetIndex(12)));
        assert!(!dropdown_is_open(&world));
    }

    // A press on the scrollbar strip beside the thumb jumps the window there
    // (and starts a drag) instead of picking the option row behind it.
    #[test]
    fn dropdown_track_click_jumps_instead_of_picking() {
        let mut world = scrolled_dropdown_world();
        world.add_component(make_frame_input(500.0, 120.0, true));
        world.step();
        assert_eq!(dropdown_first(&world), 6);

        // Press near the top of the strip (x 582..600), well above the thumb.
        world.add_component(FrameInput {
            mouse_x: 590.0,
            mouse_y: 145.0,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        });
        world.step();
        assert!(dropdown_is_open(&world), "a track press must not pick");
        assert!(produced_setting_commands(&world).is_empty());
        assert_eq!(dropdown_first(&world), 0, "the window jumps to the press");
    }

    // The open list's highlighted option, from the published resource.
    fn dropdown_hovered(world: &World) -> Option<usize> {
        world
            .resource::<crate::ecs::OpenDropdown>()
            .and_then(|d| d.0.as_ref())
            .expect("dropdown should be open")
            .hovered
    }

    // Nav pulses (arrow keys / pad) move the open list's selection highlight,
    // scrolling the shown window along with it, and Enter picks the
    // highlighted option. The cursor stays put throughout, so the idle hover
    // never re-asserts the row under it.
    #[test]
    fn dropdown_nav_moves_selection_and_enter_picks() {
        let mut world = scrolled_dropdown_world();
        world.add_component(make_frame_input(500.0, 120.0, true));
        world.step();
        assert_eq!(dropdown_first(&world), 6);

        let pulse_down = || FrameInput {
            mouse_x: 500.0,
            mouse_y: 120.0,
            captured_key: Some(Key::Down),
            ..Default::default()
        };

        // Down steps from the current value (10); the window (6..14) holds.
        world.add_component(pulse_down());
        world.step();
        assert!(dropdown_is_open(&world), "nav must not dismiss");
        assert_eq!(dropdown_hovered(&world), Some(11));
        assert_eq!(dropdown_first(&world), 6);

        // Walking to option 14 crosses the window's bottom edge: the window
        // follows the selection.
        for _ in 0..3 {
            world.add_component(pulse_down());
            world.step();
        }
        assert_eq!(dropdown_hovered(&world), Some(14));
        assert_eq!(dropdown_first(&world), 7);

        // Enter picks the highlighted option and closes the list.
        world.add_component(FrameInput {
            mouse_x: 500.0,
            mouse_y: 120.0,
            captured_key: Some(Key::Enter),
            ..Default::default()
        });
        world.step();
        let cmds = produced_setting_commands(&world);
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0].op, SettingOp::SetIndex(14)));
        assert!(!dropdown_is_open(&world));
    }

    // When a region's hover_scale equals its label's scale (what the generated
    // settings menu emits), hovering changes only the color: the label keeps its
    // size, so it does not grow or shift out of its row. This is the runtime end
    // of the build-side `default_menu_hover_is_color_only` guarantee.
    #[test]
    fn hover_with_matching_scale_changes_color_only() {
        let mut world = World::new_empty();

        world.add_component(TextLabel {
            asset_id: AssetId(1),
            font: None,
            content: "Vsync".to_string(),
            x: 0.0,
            y: 0.0,
            color: [0.85, 0.85, 0.85],
            scale: 0.66,
            centered: false,
            align: crate::assets::TextAlign::Left,
            fit: crate::assets::SpriteFit::Fit,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            screen: None,
            wrap_width: 0.0,
            max_lines: 0,
        });
        world.add_component(HitRegion {
            x: 10.0,
            y: 10.0,
            width: 100.0,
            height: 40.0,
            label: Some(AssetId(1)),
            hover_color: Some([1.0, 0.85, 0.3]),
            // Matches the label's scale, so hover must not resize it.
            hover_scale: Some(0.66),
            action: String::new(),
            drag_handle: None,
            screen: None,
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.start().unwrap();

        world.add_component(make_frame_input(50.0, 30.0, false));
        world.step();

        let lbl = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == AssetId(1))
            .map(|l| (l.color, l.scale))
            .unwrap();
        assert_eq!(lbl.0, [1.0, 0.85, 0.3], "hover should change color");
        assert_eq!(lbl.1, 0.66, "hover must not change the label scale");
    }

    #[test]
    fn click_pushes_scene_command() {
        let mut world = World::new_empty();

        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: "scene:3".to_string(),
            drag_handle: None,
            screen: None,
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.start().unwrap();

        world.add_component(make_frame_input(50.0, 50.0, true));
        world.step();

        let has_cmd = world
            .events::<SceneCommand>()
            .is_some_and(|e| !e.is_empty());
        assert!(has_cmd);
    }

    #[test]
    fn quit_action_returns_stop() {
        let mut world = World::new_empty();

        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: "quit".to_string(),
            drag_handle: None,
            screen: None,
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.start().unwrap();

        world.add_component(make_frame_input(50.0, 50.0, true));
        let result = world.step();
        assert_eq!(result, StepResult::Stop);
    }

    // Showing a screen makes its sprites visible and hides them again on Hide.
    #[test]
    fn screen_show_and_hide_toggles_sprite_visibility() {
        let mut world = World::new_empty();

        let screen_id = AssetId(10);
        world.add_component(Screen {
            asset_id: screen_id,
            initial: false,
            fade_in_secs: 0.0,
            ..Default::default()
        });
        world.add_component(Sprite {
            asset_id: AssetId(11),
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            texture: None,
            tint: [0.0, 0.0, 0.0, 0.5],
            follow_cursor: false,
            visible: true, // intentionally true to confirm init hides it
            screen: Some(screen_id),
            fit: crate::assets::SpriteFit::Fit,
            corner_radius: 0.0,
            border_width: 0.0,
            border_color: [0.0, 0.0, 0.0, 1.0],
        });
        world.start().unwrap();

        // init() hides screen elements.
        let visible_after_init = world
            .query::<Sprite>()
            .find(|s| s.asset_id == AssetId(11))
            .map(|s| s.visible)
            .unwrap();
        assert!(!visible_after_init, "screen starts hidden after init");

        // Show the screen.
        world
            .events_mut::<ScreenCommand>()
            .send(ScreenCommand::Show(screen_id));
        world.add_component(FrameInput::default());
        world.step();

        let visible_after_show = world
            .query::<Sprite>()
            .find(|s| s.asset_id == AssetId(11))
            .map(|s| s.visible)
            .unwrap();
        assert!(visible_after_show, "screen sprite is visible after Show");

        // Hide it again.
        world
            .events_mut::<ScreenCommand>()
            .send(ScreenCommand::Hide);
        world.add_component(FrameInput::default());
        world.step();

        let visible_after_hide = world
            .query::<Sprite>()
            .find(|s| s.asset_id == AssetId(11))
            .map(|s| s.visible)
            .unwrap();
        assert!(!visible_after_hide, "screen sprite is hidden after Hide");
    }

    // A screen-owned region is overlay UI authored in the reference canvas; when
    // the window differs from the reference the region is scaled, and the live
    // cursor must be mapped back into reference space to hit it. At a 2x
    // viewport, a click at the scaled on-screen rect fires; a click at the raw
    // reference coordinates (which no longer overlap the scaled rect) does not.
    fn frame_input_at(mx: f32, my: f32, viewport: [f32; 2]) -> FrameInput {
        FrameInput {
            mouse_x: mx,
            mouse_y: my,
            left_click: true,
            viewport,
            ..Default::default()
        }
    }

    fn overlay_region_world() -> World {
        let mut world = World::new_empty();
        let screen_id = AssetId(30);
        world.add_component(Screen {
            asset_id: screen_id,
            initial: true,
            fade_in_secs: 0.0,
            ..Default::default()
        });
        // Reference-space rect [200,400] x [200,260].
        world.add_component(HitRegion {
            x: 200.0,
            y: 200.0,
            width: 200.0,
            height: 60.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: "scene:7".to_string(),
            drag_handle: None,
            screen: Some(screen_id),
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.start().unwrap();
        world
    }

    #[test]
    fn screen_owned_region_hit_tests_in_reference_space_when_scaled() {
        // 2x reference viewport: the reference center (300,230) maps on-screen
        // to (600,460). A click there inverse-maps back inside the rect → fires.
        let mut world = overlay_region_world();
        world.add_component(frame_input_at(600.0, 460.0, [2560.0, 1440.0]));
        world.step();
        assert!(
            world
                .events::<SceneCommand>()
                .is_some_and(|e| !e.is_empty()),
            "click at the scaled rect should fire the region"
        );

        // A click at the raw reference coordinates lands outside the scaled
        // rect at 2x, so it must not fire.
        let mut world = overlay_region_world();
        world.add_component(frame_input_at(300.0, 230.0, [2560.0, 1440.0]));
        world.step();
        assert!(
            world.events::<SceneCommand>().is_none_or(|e| e.is_empty()),
            "click at the unscaled coords should miss the scaled region"
        );
    }

    // While a screen is active, underlying scene HitRegions don't fire.
    #[test]
    fn hit_region_filtered_when_view_is_active() {
        let mut world = World::new_empty();

        let screen_id = AssetId(20);
        world.add_component(Screen {
            asset_id: screen_id,
            initial: false,
            fade_in_secs: 0.0,
            ..Default::default()
        });
        // A scene-level region (no screen) that would normally fire.
        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: "scene:7".to_string(),
            drag_handle: None,
            screen: None,
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.start().unwrap();

        // Show the screen, then click where the scene-region is.
        world
            .events_mut::<ScreenCommand>()
            .send(ScreenCommand::Show(screen_id));
        world.add_component(make_frame_input(50.0, 50.0, true));
        world.step();

        let has_cmd = world
            .events::<SceneCommand>()
            .is_some_and(|e| !e.is_empty());
        assert!(
            !has_cmd,
            "scene-level region should not fire while screen is active"
        );
    }

    #[test]
    fn fire_action_dispatches_view_variants() {
        // screen:hide → ScreenCommand::Hide
        let mut world = World::new_empty();
        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: "screen:hide".to_string(),
            drag_handle: None,
            screen: None,
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.start().unwrap();
        world.add_component(make_frame_input(50.0, 50.0, true));
        world.step();
        assert!(matches!(
            produced_screen_command(&world),
            Some(ScreenCommand::Hide)
        ));

        // screen:show:42 → ScreenCommand::Show(42)
        let mut world = World::new_empty();
        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: "screen:show:42".to_string(),
            drag_handle: None,
            screen: None,
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.start().unwrap();
        world.add_component(make_frame_input(50.0, 50.0, true));
        world.step();
        let cmd = produced_screen_command(&world);
        assert!(matches!(cmd, Some(ScreenCommand::Show(AssetId(42)))));

        // screen:toggle:43 → ScreenCommand::Toggle(43)
        let mut world = World::new_empty();
        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: "screen:toggle:43".to_string(),
            drag_handle: None,
            screen: None,
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.start().unwrap();
        world.add_component(make_frame_input(50.0, 50.0, true));
        world.step();
        let cmd = produced_screen_command(&world);
        assert!(matches!(cmd, Some(ScreenCommand::Toggle(AssetId(43)))));
    }

    #[test]
    fn fire_action_dispatches_setting_with_value_label() {
        // setting:vsync:next → SettingCommand carrying the region's label as the
        // value-label to update, and the parsed direction.
        let mut world = World::new_empty();
        let value_label = AssetId(99);
        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            label: Some(value_label),
            hover_color: None,
            hover_scale: None,
            action: "setting:vsync:next".to_string(),
            drag_handle: None,
            screen: None,
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.start().unwrap();
        world.add_component(make_frame_input(50.0, 50.0, true));
        world.step();
        let cmd = produced_setting_commands(&world)
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(cmd.setting, "vsync");
        assert_eq!(cmd.op, SettingOp::Next);
        assert_eq!(cmd.value_label, Some(value_label));

        // The :prev suffix parses to the reverse direction. The default
        // HitRegion is 100x40, so click within those bounds.
        let mut world = World::new_empty();
        world.add_component(HitRegion {
            action: "setting:vsync:prev".to_string(),
            ..Default::default()
        });
        world.start().unwrap();
        world.add_component(make_frame_input(50.0, 20.0, true));
        world.step();
        let cmd = produced_setting_commands(&world)
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(cmd.op, SettingOp::Prev);
    }

    // A region the engine disabled (e.g. a capability-gated settings row grayed
    // out at init) is inert: clicking where it sits fires nothing.
    #[test]
    fn disabled_region_does_not_fire() {
        let mut world = World::new_empty();
        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: "setting:ray_traced_reflections:next".to_string(),
            drag_handle: None,
            screen: None,
            disabled: true,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.start().unwrap();

        world.add_component(make_frame_input(50.0, 50.0, true));
        world.step();
        assert!(
            produced_setting_commands(&world).is_empty(),
            "a disabled region must not fire its action"
        );
    }

    // A row disabled at runtime via the `DisabledSettingRows` resource (e.g. the
    // show_fps row while the "Display performance stats" master is off) is inert,
    // even though its HitRegion was enabled at init. This is the runtime twin of
    // the init-time capability gating above.
    #[test]
    fn runtime_disabled_setting_row_does_not_fire() {
        let mut world = World::new_empty();
        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: "setting:show_fps:next".to_string(),
            drag_handle: None,
            screen: None,
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.start().unwrap();

        // Master off: the show_fps row is in the runtime-disabled set, so a click
        // over it fires nothing.
        world.insert_resource(crate::ecs::DisabledSettingRows(
            ["show_fps".to_string()].into_iter().collect(),
        ));
        world.add_component(make_frame_input(50.0, 50.0, true));
        world.step();
        assert!(
            produced_setting_commands(&world).is_empty(),
            "a runtime-disabled row must not fire its action"
        );
    }

    #[test]
    fn slider_drag_pushes_set_fraction_then_persists_on_release() {
        let mut world = World::new_empty();
        let value_label = AssetId(7);
        world.add_component(HitRegion {
            x: 100.0,
            y: 0.0,
            width: 200.0,
            height: 40.0,
            label: Some(value_label),
            hover_color: None,
            hover_scale: None,
            action: "setting:exposure:drag".to_string(),
            drag_handle: Some(AssetId(8)),
            screen: None,
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.start().unwrap();

        // Press at x=150 (25% across the [100, 300) track) with the button held:
        // a live, non-persisting fraction.
        world.add_component(FrameInput {
            mouse_x: 150.0,
            mouse_y: 20.0,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        });
        world.step();
        let cmd = produced_setting_commands(&world)
            .into_iter()
            .last()
            .unwrap();
        assert_eq!(cmd.setting, "exposure");
        assert!(matches!(cmd.op, SettingOp::SetFraction(f) if (f - 0.25).abs() < 1.0e-5));
        assert_eq!(cmd.value_label, Some(value_label));
        assert!(
            !cmd.persist,
            "an in-progress drag applies live but does not persist"
        );

        // Release at x=250 (75%): the button up commits the final value and persists.
        world.add_component(FrameInput {
            mouse_x: 250.0,
            mouse_y: 20.0,
            left_click: false,
            left_button_down: false,
            ..Default::default()
        });
        world.step();
        let cmd = produced_setting_commands(&world)
            .into_iter()
            .last()
            .unwrap();
        assert!(matches!(cmd.op, SettingOp::SetFraction(f) if (f - 0.75).abs() < 1.0e-5));
        assert!(cmd.persist, "release commits and persists the final value");
    }

    // A group-toggle click collapses the group's body rows (hiding their
    // elements) and flips the header's `+`/`-` prefix; the body's click region
    // then goes inert.
    #[test]
    fn group_toggle_collapses_body_and_updates_header() {
        let mut world = World::new_empty();
        let screen = AssetId(50);
        let (header, body) = (AssetId(51), AssetId(52));
        world.add_component(Screen {
            asset_id: screen,
            initial: true,
            fade_in_secs: 0.0,
            ..Default::default()
        });
        world.add_component(panel_label(51, 100.0, screen, "- Adv"));
        world.add_component(panel_label(52, 140.0, screen, "Body"));
        // Header click region (toggles group 0).
        world.add_component(HitRegion {
            x: 0.0,
            y: 100.0,
            width: 300.0,
            height: 40.0,
            label: Some(header),
            hover_color: None,
            hover_scale: None,
            action: "group:toggle:0".to_string(),
            drag_handle: None,
            screen: Some(screen),
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        // Body click region (a settings action; a content region, so it is
        // bucketed into its row and gated by the collapse).
        world.add_component(HitRegion {
            x: 0.0,
            y: 140.0,
            width: 300.0,
            height: 40.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: "setting:vsync:next".to_string(),
            drag_handle: None,
            screen: Some(screen),
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.add_component(ScrollPanel {
            screen: Some(screen),
            x: 0.0,
            y: 100.0,
            width: 300.0,
            height: 100.0,
            rows: vec![
                ScrollRow {
                    elements: vec![header],
                    base_y: 100.0,
                    height: 40.0,
                    group: -1,
                },
                ScrollRow {
                    elements: vec![body],
                    base_y: 140.0,
                    height: 40.0,
                    group: 0,
                },
            ],
            groups: vec![ScrollGroup {
                collapsed: false,
                header: Some(header),
                title: "Adv".to_string(),
            }],
            thumb: None,
            track: None,
            track_x: 0.0,
            track_y: 0.0,
            track_w: 0.0,
            track_h: 0.0,
        });
        world.start().unwrap();

        // Expanded after init: body shown, header reads "- Adv".
        assert!(label_field(&world, body, |l| l.visible));
        assert_eq!(label_field(&world, header, |l| l.content.clone()), "- Adv");

        // Click the header to collapse.
        world.add_component(make_frame_input(10.0, 120.0, true));
        world.step();
        assert!(!label_field(&world, body, |l| l.visible), "body hides");
        assert_eq!(label_field(&world, header, |l| l.content.clone()), "+ Adv");

        // The body's region is now inert: clicking where it was fires nothing.
        world.add_component(make_frame_input(10.0, 160.0, true));
        world.step();
        assert!(
            produced_setting_commands(&world).is_empty(),
            "a collapsed row's region does not fire"
        );
    }

    // The mouse wheel, with the cursor over the panel band, scrolls the content
    // up: the top row's element moves up by wheel-delta * speed (clamped).
    #[test]
    fn wheel_scrolls_panel_content() {
        let mut world = World::new_empty();
        let screen = AssetId(60);
        let e0 = AssetId(61);
        world.add_component(Screen {
            asset_id: screen,
            initial: true,
            fade_in_secs: 0.0,
            ..Default::default()
        });
        world.add_component(panel_label(61, 0.0, screen, "Row0"));
        // Three 40px rows (120px) in a 60px band -> overflows by 60px.
        world.add_component(ScrollPanel {
            screen: Some(screen),
            x: 0.0,
            y: 0.0,
            width: 300.0,
            height: 60.0,
            rows: vec![
                ScrollRow {
                    elements: vec![e0],
                    base_y: 0.0,
                    height: 40.0,
                    group: -1,
                },
                ScrollRow {
                    elements: vec![],
                    base_y: 40.0,
                    height: 40.0,
                    group: -1,
                },
                ScrollRow {
                    elements: vec![],
                    base_y: 80.0,
                    height: 40.0,
                    group: -1,
                },
            ],
            groups: vec![],
            thumb: None,
            track: None,
            track_x: 0.0,
            track_y: 0.0,
            track_w: 0.0,
            track_h: 0.0,
        });
        world.start().unwrap();
        assert_eq!(label_field(&world, e0, |l| l.y), 0.0);

        // Wheel down with the cursor inside the band: scroll = 10 * speed (2.0)
        // = 20 (within the 60px max), so the top row moves up by 20.
        world.add_component(FrameInput {
            mouse_x: 10.0,
            mouse_y: 10.0,
            scroll_delta: 10.0,
            ..Default::default()
        });
        world.step();
        assert_eq!(label_field(&world, e0, |l| l.y), -20.0);

        // Wheel far past the end clamps to the max (60px up), not further.
        world.add_component(FrameInput {
            mouse_x: 10.0,
            mouse_y: 10.0,
            scroll_delta: 1000.0,
            ..Default::default()
        });
        world.step();
        assert_eq!(label_field(&world, e0, |l| l.y), -60.0);
    }

    // A panel whose content overflows its band, with scrollbar-track geometry
    // so the thumb is grabbable: three 40px rows (120px) in a 60px band, the
    // track beside it (thumb = 30px, travel = 30px, max scroll = 60px). Each
    // row carries a setting region so focus navigation has targets.
    fn scrollbar_panel_world() -> (World, AssetId) {
        let mut world = World::new_empty();
        let screen = AssetId(60);
        let e0 = AssetId(61);
        world.add_component(Screen {
            asset_id: screen,
            initial: true,
            fade_in_secs: 0.0,
            ..Default::default()
        });
        world.add_component(panel_label(61, 0.0, screen, "Row0"));
        for (i, y) in [0.0, 40.0, 80.0].into_iter().enumerate() {
            world.add_component(HitRegion {
                x: 0.0,
                y,
                width: 300.0,
                height: 40.0,
                label: None,
                hover_color: None,
                hover_scale: None,
                action: format!("setting:row{i}:next"),
                drag_handle: None,
                screen: Some(screen),
                disabled: false,
                follow_label: false,
                fit: crate::assets::SpriteFit::Fit,
            });
        }
        world.add_component(ScrollPanel {
            screen: Some(screen),
            x: 0.0,
            y: 0.0,
            width: 300.0,
            height: 60.0,
            rows: vec![
                ScrollRow {
                    elements: vec![e0],
                    base_y: 0.0,
                    height: 40.0,
                    group: -1,
                },
                ScrollRow {
                    elements: vec![],
                    base_y: 40.0,
                    height: 40.0,
                    group: -1,
                },
                ScrollRow {
                    elements: vec![],
                    base_y: 80.0,
                    height: 40.0,
                    group: -1,
                },
            ],
            groups: vec![],
            thumb: None,
            track: None,
            track_x: 305.0,
            track_y: 0.0,
            track_w: 8.0,
            track_h: 60.0,
        });
        world.start().unwrap();
        (world, e0)
    }

    // Dragging the panel's scrollbar thumb scrolls the content: a press on the
    // thumb grabs it and the held cursor's y maps onto the scroll range.
    #[test]
    fn thumb_drag_scrolls_panel_content() {
        let (mut world, e0) = scrollbar_panel_world();
        assert_eq!(label_field(&world, e0, |l| l.y), 0.0);

        // Press on the thumb (at rest: [305, 0, 8, 30]).
        world.add_component(FrameInput {
            mouse_x: 306.0,
            mouse_y: 5.0,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        });
        world.step();
        assert_eq!(label_field(&world, e0, |l| l.y), 0.0);

        // Drag halfway along the 30px travel: scroll = half of the 60px max.
        world.add_component(FrameInput {
            mouse_x: 306.0,
            mouse_y: 20.0,
            left_button_down: true,
            ..Default::default()
        });
        world.step();
        assert_eq!(label_field(&world, e0, |l| l.y), -30.0);

        // Drag far past the end: the scroll clamps to the max.
        world.add_component(FrameInput {
            mouse_x: 306.0,
            mouse_y: 500.0,
            left_button_down: true,
            ..Default::default()
        });
        world.step();
        assert_eq!(label_field(&world, e0, |l| l.y), -60.0);

        // Release: the drag ends and the cursor no longer moves the content.
        world.add_component(make_frame_input(306.0, 5.0, false));
        world.step();
        assert_eq!(label_field(&world, e0, |l| l.y), -60.0);
    }

    // Nav pulses (arrow keys / pad) walk the panel's rows and scroll the
    // focused row into the clip band, wrapping back to the top past the end.
    #[test]
    fn nav_focus_walks_panel_rows_and_scrolls_them_into_view() {
        let (mut world, e0) = scrollbar_panel_world();
        let pulse_down = || FrameInput {
            captured_key: Some(Key::Down),
            ..Default::default()
        };

        // First pulse focuses the top row: everything already in view.
        world.add_component(pulse_down());
        world.step();
        assert_eq!(label_field(&world, e0, |l| l.y), 0.0);

        // The second row's bottom (80) pokes past the 60px band: the panel
        // scrolls the overflow (20px) into view.
        world.add_component(pulse_down());
        world.step();
        assert_eq!(label_field(&world, e0, |l| l.y), -20.0);

        // The third row scrolls fully in (its bottom sat 40px past the band).
        world.add_component(pulse_down());
        world.step();
        assert_eq!(label_field(&world, e0, |l| l.y), -60.0);

        // Past the end the focus wraps to the top row, scrolling back up.
        world.add_component(pulse_down());
        world.step();
        assert_eq!(label_field(&world, e0, |l| l.y), 0.0);
    }

    // A rebind row: a value TextLabel showing the current key + a HitRegion over
    // it whose action enters capture mode.
    fn rebind_world() -> (World, AssetId) {
        let mut world = World::new_empty();
        let value = AssetId(7);
        world.add_component(TextLabel {
            asset_id: value,
            font: None,
            content: "W".to_string(),
            x: 0.0,
            y: 0.0,
            color: [1.0, 1.0, 1.0],
            scale: 1.0,
            centered: false,
            align: crate::assets::TextAlign::Left,
            fit: crate::assets::SpriteFit::Fit,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            screen: None,
            wrap_width: 0.0,
            max_lines: 0,
        });
        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 40.0,
            label: Some(value),
            hover_color: None,
            hover_scale: None,
            action: "setting:key_forward:rebind".to_string(),
            drag_handle: None,
            screen: None,
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.start().unwrap();
        (world, value)
    }

    // Clicking a rebind row enters capture (the value shows the prompt and no
    // command fires); the next pressed key binds it via a Rebind SettingCommand.
    #[test]
    fn rebind_click_captures_then_binds_next_key() {
        use crate::assets::Key;
        let (mut world, value) = rebind_world();

        // Click the rebind row: enters capture, value shows the prompt, and no
        // command is pushed yet.
        world.add_component(make_frame_input(50.0, 20.0, true));
        world.step();
        assert_eq!(
            label_field(&world, value, |l| l.content.clone()),
            REBIND_PROMPT
        );
        assert!(
            produced_setting_commands(&world).is_empty(),
            "no command until a key is pressed"
        );

        // Press a key: it binds, pushing a Rebind command carrying the key.
        world.add_component(FrameInput {
            captured_key: Some(Key::Q),
            ..Default::default()
        });
        world.step();
        let cmd = produced_setting_commands(&world)
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(cmd.setting, "key_forward");
        assert_eq!(cmd.value_label, Some(value));
        assert!(matches!(cmd.op, SettingOp::Rebind(Key::Q)));
        assert!(cmd.persist);
    }

    // Escape while capturing cancels and restores the row's previous value text.
    #[test]
    fn rebind_escape_cancels_and_restores() {
        let (mut world, value) = rebind_world();
        world.add_component(make_frame_input(50.0, 20.0, true));
        world.step();
        assert_eq!(
            label_field(&world, value, |l| l.content.clone()),
            REBIND_PROMPT
        );

        // Escape cancels: the previous text returns and nothing is bound.
        world.add_component(FrameInput {
            escape: true,
            ..Default::default()
        });
        world.step();
        assert_eq!(label_field(&world, value, |l| l.content.clone()), "W");
        assert!(produced_setting_commands(&world).is_empty());
    }

    // A captured key with no active capture binds nothing.
    #[test]
    fn captured_key_without_capture_is_ignored() {
        use crate::assets::Key;
        let (mut world, _value) = rebind_world();
        world.add_component(FrameInput {
            captured_key: Some(Key::Q),
            ..Default::default()
        });
        world.step();
        assert!(produced_setting_commands(&world).is_empty());
    }

    // The ScreenShown announcements the cursor has not yet consumed, read the
    // way a real consumer (AudioCue) does: incrementally, before the queue's
    // two-frame retention retires them.
    fn shown_views(world: &World, cursor: &mut crate::ecs::EventCursor) -> Vec<AssetId> {
        world
            .events::<ScreenShown>()
            .map(|e| e.read(cursor).map(|s| s.screen).collect())
            .unwrap_or_default()
    }

    // Both the initial screen at start and a Show navigation announce the newly
    // active screen, so screen-triggered consumers (AudioCue) hear every screen.
    #[test]
    fn view_activation_emits_view_shown() {
        let mut world = World::new_empty();
        let first = AssetId(80);
        let second = AssetId(81);
        for (id, initial) in [(first, true), (second, false)] {
            world.add_component(Screen {
                asset_id: id,
                initial,
                fade_in_secs: 0.0,
                ..Default::default()
            });
        }
        world.start().unwrap();
        let mut cursor = crate::ecs::EventCursor::default();
        assert_eq!(shown_views(&world, &mut cursor), vec![first]);

        world
            .events_mut::<ScreenCommand>()
            .send(ScreenCommand::Show(second));
        world.step();
        assert_eq!(shown_views(&world, &mut cursor), vec![second]);

        // Showing the already-active screen is a no-op: no repeat announcement.
        world
            .events_mut::<ScreenCommand>()
            .send(ScreenCommand::Show(second));
        world.step();
        assert!(shown_views(&world, &mut cursor).is_empty());
    }

    #[test]
    fn escape_key_binding_fires_action() {
        let mut world = World::new_empty();

        let screen_id = AssetId(50);
        world.add_component(Screen {
            asset_id: screen_id,
            initial: false,
            fade_in_secs: 0.0,
            ..Default::default()
        });
        world.add_component(KeyBinding {
            key: "Escape".to_string(),
            action: "screen:toggle:50".to_string(),
            ..Default::default()
        });
        world.start().unwrap();

        // Press Escape.
        world.add_component(FrameInput {
            escape: true,
            ..Default::default()
        });
        world.step();

        let cmd = produced_screen_command(&world);
        assert!(matches!(cmd, Some(ScreenCommand::Toggle(AssetId(50)))));
    }

    // Every StoryCommand the system sent this step, in send order.
    fn produced_story_commands(world: &World) -> Vec<StoryCommand> {
        let mut cursor = crate::ecs::EventCursor::default();
        world
            .events::<StoryCommand>()
            .map(|e| e.read(&mut cursor).cloned().collect())
            .unwrap_or_default()
    }

    // A non-Escape KeyBinding fires when its key is pressed (surfaced as the
    // one-frame captured_key), so a story's Space / Enter advance bindings work.
    #[test]
    fn pressed_key_binding_fires_action() {
        for key in [Key::Space, Key::Enter] {
            let mut world = World::new_empty();
            world.add_component(KeyBinding {
                key: key.name().to_string(),
                action: "story:advance".to_string(),
                ..Default::default()
            });
            world.start().unwrap();

            world.add_component(FrameInput {
                captured_key: Some(key),
                ..Default::default()
            });
            world.step();

            assert_eq!(
                produced_story_commands(&world),
                vec![StoryCommand::Advance],
                "{key:?} should advance",
            );
        }
    }

    // A pressed key with no matching binding fires nothing (the arrow keys a
    // scrollable list reads must not be swallowed by a phantom action).
    #[test]
    fn pressed_key_without_a_binding_fires_nothing() {
        let mut world = World::new_empty();
        world.add_component(KeyBinding {
            key: "Space".to_string(),
            action: "story:advance".to_string(),
            ..Default::default()
        });
        world.start().unwrap();

        world.add_component(FrameInput {
            captured_key: Some(Key::Down),
            ..Default::default()
        });
        world.step();

        assert!(produced_story_commands(&world).is_empty());
    }

    // Escape toggles the menu; Settings is reached by a Show (a sub-screen). After
    // visiting Settings, escaping back to the menu and escaping again must return
    // to the world, not back into Settings: a Toggle that opens a screen must not
    // record the outgoing screen as the dismiss target.
    #[test]
    fn escape_from_menu_returns_to_world_after_visiting_a_subview() {
        let mut world = World::new_empty();
        let menu = AssetId(60);
        let settings = AssetId(61);
        world.add_component(Screen {
            asset_id: menu,
            initial: false,
            fade_in_secs: 0.0,
            ..Default::default()
        });
        world.add_component(Screen {
            asset_id: settings,
            initial: false,
            fade_in_secs: 0.0,
            ..Default::default()
        });
        // One sprite per screen, to observe which screen is active by visibility.
        for (id, screen) in [(70u32, menu), (71u32, settings)] {
            world.add_component(Sprite {
                asset_id: AssetId(id),
                x: 0.0,
                y: 0.0,
                width: 10.0,
                height: 10.0,
                texture: None,
                tint: [0.0, 0.0, 0.0, 1.0],
                follow_cursor: false,
                visible: false,
                screen: Some(screen),
                fit: crate::assets::SpriteFit::Fit,
                corner_radius: 0.0,
                border_width: 0.0,
                border_color: [0.0, 0.0, 0.0, 1.0],
            });
        }
        world.add_component(KeyBinding {
            key: "Escape".to_string(),
            action: "screen:toggle:60".to_string(),
            ..Default::default()
        });
        world.start().unwrap();

        let shown = |w: &World, id: u32| {
            w.query::<Sprite>()
                .find(|s| s.asset_id == AssetId(id))
                .map(|s| s.visible)
                .unwrap()
        };
        // A frame that presses Escape (fires the toggle keybinding).
        fn esc(w: &mut World) {
            w.add_component(FrameInput {
                escape: true,
                ..Default::default()
            });
            w.step();
        }
        // A frame that applies the screen command queued the previous frame.
        fn settle(w: &mut World) {
            w.add_component(FrameInput::default());
            w.step();
        }

        // World -> Escape -> menu.
        esc(&mut world);
        settle(&mut world);
        assert!(shown(&world, 70) && !shown(&world, 71), "menu opens");

        // Menu -> click Settings (a Show) -> settings sub-screen.
        world
            .events_mut::<ScreenCommand>()
            .send(ScreenCommand::Show(settings));
        settle(&mut world);
        assert!(!shown(&world, 70) && shown(&world, 71), "settings shown");

        // Settings -> Escape -> back to the menu.
        esc(&mut world);
        settle(&mut world);
        assert!(shown(&world, 70) && !shown(&world, 71), "back to the menu");

        // Menu -> Escape -> world (regression: previously returned to Settings).
        esc(&mut world);
        settle(&mut world);
        assert!(
            !shown(&world, 70) && !shown(&world, 71),
            "escape from the menu returns to the world, not back into Settings"
        );
    }

    // A Screen's `toggle_key` opens it from anywhere and closes it again,
    // without any KeyBinding.
    #[test]
    fn toggle_key_opens_and_closes_a_screen() {
        let mut world = World::new_empty();
        world.add_component(Screen {
            asset_id: AssetId(80),
            toggle_key: "Backtick".to_string(),
            ..Default::default()
        });
        world.add_component(Sprite {
            asset_id: AssetId(81),
            width: 10.0,
            height: 10.0,
            visible: true,
            screen: Some(AssetId(80)),
            ..Default::default()
        });
        world.start().unwrap();
        let shown = |w: &World| {
            w.query::<Sprite>()
                .find(|s| s.asset_id == AssetId(81))
                .unwrap()
                .visible
        };
        assert!(!shown(&world), "screens start hidden");

        let backtick = |w: &mut World| {
            w.add_component(FrameInput {
                captured_key: Some(Key::Backtick),
                ..Default::default()
            });
            w.step();
            // Settle frame: apply the command queued above.
            w.add_component(FrameInput::default());
            w.step();
        };
        backtick(&mut world);
        assert!(shown(&world), "toggle key opens the screen");
        backtick(&mut world);
        assert!(!shown(&world), "toggle key closes it again");
    }

    // A screen's `focus` TextInput gains keyboard focus when the screen
    // reaches the top of the stack, and loses it when the screen closes.
    #[test]
    fn focus_field_follows_the_top_screen() {
        let mut world = World::new_empty();
        world.add_component(Screen {
            asset_id: AssetId(90),
            toggle_key: "Backtick".to_string(),
            focus: Some(AssetId(91)),
            ..Default::default()
        });
        world.add_component(crate::assets::TextInput {
            asset_id: AssetId(91),
            visible: true,
            screen: Some(AssetId(90)),
            ..Default::default()
        });
        world.start().unwrap();
        let focused = |w: &World| {
            w.query::<crate::assets::TextInput>()
                .find(|t| t.asset_id == AssetId(91))
                .map(|t| (t.visible, t.focused))
                .unwrap()
        };
        assert_eq!(focused(&world), (false, false), "hidden until shown");

        let backtick = |w: &mut World| {
            w.add_component(FrameInput {
                captured_key: Some(Key::Backtick),
                ..Default::default()
            });
            w.step();
            w.add_component(FrameInput::default());
            w.step();
        };
        backtick(&mut world);
        assert_eq!(focused(&world), (true, true), "shown and focused");
        backtick(&mut world);
        assert_eq!(focused(&world), (false, false), "blurred on close");
    }

    // While a visible TextInput has keyboard focus, ordinary KeyBindings are
    // suspended so typing cannot fire actions.
    #[test]
    fn keybindings_are_suspended_while_typing() {
        let mut world = World::new_empty();
        world.add_component(Screen {
            asset_id: AssetId(100),
            ..Default::default()
        });
        world.add_component(KeyBinding {
            key: "T".to_string(),
            action: "screen:toggle:100".to_string(),
            ..Default::default()
        });
        let mut field = crate::assets::TextInput {
            asset_id: AssetId(101),
            visible: true,
            ..Default::default()
        };
        field.focused = true;
        world.add_component(field);
        world.start().unwrap();

        world.add_component(FrameInput {
            captured_key: Some(Key::T),
            ..Default::default()
        });
        world.step();
        assert!(
            produced_screen_command(&world).is_none(),
            "typed key does not fire the binding"
        );

        // Blur the field: the same key now fires the binding.
        for ti in world.query_mut::<crate::assets::TextInput>() {
            ti.focused = false;
        }
        world.add_component(FrameInput {
            captured_key: Some(Key::T),
            ..Default::default()
        });
        world.step();
        assert!(matches!(
            produced_screen_command(&world),
            Some(ScreenCommand::Toggle(AssetId(100)))
        ));
    }

    // A KeyBinding scoped to a screen fires only while that screen is on top.
    #[test]
    fn scoped_keybinding_fires_only_while_its_screen_is_top() {
        let mut world = World::new_empty();
        world.add_component(Screen {
            asset_id: AssetId(110),
            ..Default::default()
        });
        world.add_component(Screen {
            asset_id: AssetId(111),
            ..Default::default()
        });
        world.add_component(KeyBinding {
            key: "Space".to_string(),
            action: "screen:show:111".to_string(),
            screen: Some(AssetId(110)),
        });
        world.start().unwrap();

        // No screen on top: the scoped binding stays quiet.
        world.add_component(FrameInput {
            captured_key: Some(Key::Space),
            ..Default::default()
        });
        world.step();
        assert!(produced_screen_command(&world).is_none());

        // Open its screen; the binding now fires.
        world
            .events_mut::<ScreenCommand>()
            .send(ScreenCommand::Show(AssetId(110)));
        world.add_component(FrameInput::default());
        world.step();
        world.add_component(FrameInput {
            captured_key: Some(Key::Space),
            ..Default::default()
        });
        world.step();
        let mut cursor = crate::ecs::EventCursor::default();
        let sent: Vec<ScreenCommand> = world
            .events::<ScreenCommand>()
            .map(|e| e.read(&mut cursor).cloned().collect())
            .unwrap_or_default();
        assert!(
            sent.iter()
                .any(|c| matches!(c, ScreenCommand::Show(AssetId(111)))),
            "scoped binding fires on top: {sent:?}"
        );
    }

    // `screen:push:` stacks a screen over the current one: both stay visible,
    // and hiding the pushed screen reveals the one beneath.
    #[test]
    fn push_stacks_over_the_current_screen() {
        let mut world = World::new_empty();
        for (screen, sprite) in [(120u32, 130u32), (121, 131)] {
            world.add_component(Screen {
                asset_id: AssetId(screen),
                ..Default::default()
            });
            world.add_component(Sprite {
                asset_id: AssetId(sprite),
                width: 10.0,
                height: 10.0,
                visible: true,
                screen: Some(AssetId(screen)),
                ..Default::default()
            });
        }
        world.start().unwrap();
        let shown = |w: &World, id: u32| {
            w.query::<Sprite>()
                .find(|s| s.asset_id == AssetId(id))
                .unwrap()
                .visible
        };
        let settle = |w: &mut World| {
            w.add_component(FrameInput::default());
            w.step();
        };
        world
            .events_mut::<ScreenCommand>()
            .send(ScreenCommand::Show(AssetId(120)));
        settle(&mut world);
        world
            .events_mut::<ScreenCommand>()
            .send(ScreenCommand::Push(AssetId(121)));
        settle(&mut world);
        assert!(
            shown(&world, 130) && shown(&world, 131),
            "both screens visible while stacked"
        );
        // The stack resource carries both layers, in stack order.
        {
            let stack = world.resource::<crate::ecs::ScreenStack>().unwrap();
            assert_eq!(stack.layers[&AssetId(120)], 1);
            assert_eq!(stack.layers[&AssetId(121)], 2);
            assert!(stack.pauses_world && stack.captures_input);
        }
        world
            .events_mut::<ScreenCommand>()
            .send(ScreenCommand::Hide);
        settle(&mut world);
        assert!(
            shown(&world, 130) && !shown(&world, 131),
            "hide pops the pushed screen, revealing the one beneath"
        );
    }

    // A gating component (here a Screen) spawns the internal UiInputSystem.
    #[test]
    fn ui_component_spawns_internal_system() {
        let mut world = World::new_empty();
        world.add_component(Screen {
            asset_id: AssetId(1),
            initial: false,
            fade_in_secs: 0.0,
            ..Default::default()
        });
        world.start().unwrap();

        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["UiInputSystem"]);
    }

    // No HitRegion / Screen / KeyBinding means no UiInputSystem.
    #[test]
    fn no_ui_components_means_no_system() {
        let mut world = World::new_empty();
        world.start().unwrap();
        assert!(world.systems().is_empty());
    }

    // A panel whose content fits its track has no thumb to draw.
    fn scroll_panel(scroll: f32, content_height: f32, thumb_h: f32) -> PanelState {
        PanelState {
            screen: None,
            band: [0.0, 0.0, 100.0, 200.0],
            rows: Vec::new(),
            groups: Vec::new(),
            thumb: None,
            track: None,
            track_x: 300.0,
            track_y: 50.0,
            track_w: 8.0,
            track_h: 200.0,
            scroll,
            content_height,
            thumb_h,
        }
    }

    // The thumb sits proportionally down its track, at the panel's scroll
    // fraction of the content.
    #[test]
    fn thumb_rect_tracks_the_scroll_fraction() {
        let top = UiInputSystem::thumb_rect(&scroll_panel(0.0, 400.0, 100.0)).unwrap();
        assert_eq!(
            top,
            [300.0, 50.0, 8.0, 100.0],
            "unscrolled sits at track top"
        );

        // Scrolled a quarter of the content: a quarter down the 200px track.
        let mid = UiInputSystem::thumb_rect(&scroll_panel(100.0, 400.0, 100.0)).unwrap();
        assert_eq!(mid, [300.0, 100.0, 8.0, 100.0]);
    }

    // Content that fits (a thumb as tall as its track) draws no thumb, and
    // neither does a panel that has not been solved yet.
    #[test]
    fn thumb_rect_absent_when_there_is_nothing_to_scroll() {
        assert!(UiInputSystem::thumb_rect(&scroll_panel(0.0, 400.0, 200.0)).is_none());
        assert!(UiInputSystem::thumb_rect(&scroll_panel(0.0, 0.0, 50.0)).is_none());
    }

    // A scroll offset past the content pins the thumb to the track's end rather
    // than sliding it off.
    #[test]
    fn thumb_rect_clamps_an_overscrolled_offset() {
        let rect = UiInputSystem::thumb_rect(&scroll_panel(9999.0, 400.0, 100.0)).unwrap();
        assert_eq!(rect[1], 250.0, "pinned at track_y + track_h");
    }

    // A menu screen (id 90) with two buttons at y 100 / 200 whose labels take a
    // hover color, plus a second screen (91) the first button navigates to.
    fn focus_menu_world() -> World {
        let mut world = World::new_empty();
        let menu = AssetId(90);
        world.add_component(Screen {
            asset_id: menu,
            initial: true,
            fade_in_secs: 0.0,
            ..Default::default()
        });
        world.add_component(Screen {
            asset_id: AssetId(91),
            initial: false,
            fade_in_secs: 0.0,
            ..Default::default()
        });
        for (id, y, action) in [
            (1u32, 100.0f32, "screen:show:91"),
            (2, 200.0, "screen:hide"),
        ] {
            world.add_component(panel_label(id, y, menu, "Btn"));
            world.add_component(HitRegion {
                x: 0.0,
                y,
                width: 200.0,
                height: 40.0,
                label: Some(AssetId(id)),
                hover_color: Some([1.0, 0.85, 0.3]),
                hover_scale: Some(1.0),
                action: action.to_string(),
                drag_handle: None,
                screen: Some(menu),
                disabled: false,
                follow_label: false,
                fit: crate::assets::SpriteFit::Fit,
            });
        }
        world.start().unwrap();
        world
    }

    // A nav pulse focuses the first button (styled like hover), further pulses
    // walk the list, and confirm fires the focused button's action.
    #[test]
    fn nav_focuses_buttons_and_confirm_fires_the_focused_one() {
        let mut world = focus_menu_world();

        world.add_component(FrameInput {
            nav: Some(NavDirection::Down),
            ..Default::default()
        });
        world.step();
        assert_eq!(
            label_field(&world, AssetId(1), |l| l.color),
            [1.0, 0.85, 0.3],
            "first pulse focuses + styles the top button"
        );

        // The next pulse moves focus to the second button; the first restores.
        world.add_component(FrameInput {
            nav: Some(NavDirection::Down),
            ..Default::default()
        });
        world.step();
        assert_eq!(
            label_field(&world, AssetId(1), |l| l.color),
            [1.0, 1.0, 1.0]
        );
        assert_eq!(
            label_field(&world, AssetId(2), |l| l.color),
            [1.0, 0.85, 0.3]
        );

        // Back up to the first, then confirm: its action fires.
        world.add_component(FrameInput {
            nav: Some(NavDirection::Up),
            ..Default::default()
        });
        world.step();
        world.add_component(FrameInput {
            confirm: true,
            ..Default::default()
        });
        world.step();
        assert!(matches!(
            produced_screen_command(&world),
            Some(ScreenCommand::Show(AssetId(91)))
        ));
    }

    // Mouse movement dismisses the focus cursor: the focused label restores
    // and a later confirm no longer fires anything.
    #[test]
    fn mouse_movement_dismisses_focus() {
        let mut world = focus_menu_world();

        world.add_component(FrameInput {
            nav: Some(NavDirection::Down),
            ..Default::default()
        });
        world.step();
        assert_eq!(
            label_field(&world, AssetId(1), |l| l.color),
            [1.0, 0.85, 0.3]
        );

        // The cursor moves (well away from both buttons): focus clears.
        world.add_component(make_frame_input(600.0, 600.0, false));
        world.step();
        assert_eq!(
            label_field(&world, AssetId(1), |l| l.color),
            [1.0, 1.0, 1.0],
            "moving the mouse restores the focused label"
        );

        world.add_component(FrameInput {
            mouse_x: 600.0,
            mouse_y: 600.0,
            confirm: true,
            ..Default::default()
        });
        world.step();
        assert!(
            produced_screen_command(&world).is_none(),
            "no focus, nothing fires"
        );
    }

    // A settings screen with a stepper row and a slider row: Left/Right on the
    // focused row send Prev/Next for its setting instead of moving focus.
    #[test]
    fn focus_left_right_adjusts_value_rows() {
        let mut world = World::new_empty();
        let screen = AssetId(95);
        world.add_component(Screen {
            asset_id: screen,
            initial: true,
            fade_in_secs: 0.0,
            ..Default::default()
        });
        world.add_component(panel_label(1, 100.0, screen, "Vsync"));
        world.add_component(panel_label(2, 200.0, screen, "50"));
        // The stepper's two regions (prev + next) share the row.
        for (x, suffix) in [(200.0, "prev"), (260.0, "next")] {
            world.add_component(HitRegion {
                x,
                y: 100.0,
                width: 50.0,
                height: 40.0,
                label: Some(AssetId(1)),
                hover_color: Some([1.0, 0.85, 0.3]),
                hover_scale: Some(1.0),
                action: format!("setting:vsync:{suffix}"),
                drag_handle: None,
                screen: Some(screen),
                disabled: false,
                follow_label: false,
                fit: crate::assets::SpriteFit::Fit,
            });
        }
        world.add_component(HitRegion {
            x: 200.0,
            y: 200.0,
            width: 110.0,
            height: 40.0,
            label: Some(AssetId(2)),
            hover_color: Some([1.0, 0.85, 0.3]),
            hover_scale: Some(1.0),
            action: "setting:exposure:drag".to_string(),
            drag_handle: None,
            screen: Some(screen),
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.start().unwrap();

        // Focus the stepper row (its two regions read as one target).
        world.add_component(FrameInput {
            nav: Some(NavDirection::Down),
            ..Default::default()
        });
        world.step();
        assert_eq!(
            label_field(&world, AssetId(1), |l| l.color),
            [1.0, 0.85, 0.3]
        );

        // Right cycles forward, Left back; focus stays on the row.
        for (dir, expect_prev) in [(NavDirection::Right, false), (NavDirection::Left, true)] {
            world.add_component(FrameInput {
                nav: Some(dir),
                ..Default::default()
            });
            world.step();
            let cmds = produced_setting_commands(&world);
            let cmd = cmds.last().unwrap();
            assert_eq!(cmd.setting, "vsync");
            assert_eq!(
                matches!(cmd.op, SettingOp::Prev),
                expect_prev,
                "Left sends Prev, Right sends Next"
            );
            assert!(cmd.persist);
        }
        assert_eq!(
            label_field(&world, AssetId(1), |l| l.color),
            [1.0, 0.85, 0.3],
            "adjusting keeps the row focused"
        );

        // Down reaches the slider row (styled too); Right steps its setting.
        world.add_component(FrameInput {
            nav: Some(NavDirection::Down),
            ..Default::default()
        });
        world.step();
        assert_eq!(
            label_field(&world, AssetId(2), |l| l.color),
            [1.0, 0.85, 0.3]
        );
        world.add_component(FrameInput {
            nav: Some(NavDirection::Right),
            ..Default::default()
        });
        world.step();
        let cmds = produced_setting_commands(&world);
        let cmd = cmds.last().unwrap();
        assert_eq!(cmd.setting, "exposure");
        assert!(matches!(cmd.op, SettingOp::Next));
    }

    // The pad's back pulse mirrors Escape only while a screen is active: with
    // nothing open it must NOT open an Escape-toggled menu, and with the menu
    // open it closes it.
    #[test]
    fn back_mirrors_escape_only_while_a_screen_is_active() {
        let mut world = World::new_empty();
        let menu = AssetId(97);
        world.add_component(Screen {
            asset_id: menu,
            initial: false,
            fade_in_secs: 0.0,
            toggle_key: "Escape".to_string(),
            ..Default::default()
        });
        world.start().unwrap();

        // Back with no screen active: nothing opens.
        world.add_component(FrameInput {
            back: true,
            ..Default::default()
        });
        world.step();
        assert!(produced_screen_command(&world).is_none());

        // Escape opens the menu (toggle); a frame applies it.
        world.add_component(FrameInput {
            escape: true,
            ..Default::default()
        });
        world.step();
        world.add_component(FrameInput::default());
        world.step();
        assert!(
            world
                .resource::<crate::ecs::ScreenStack>()
                .is_some_and(|s| s.captures_input),
            "escape toggles the menu open"
        );

        // Back now mirrors Escape: the toggle pops the menu.
        world.add_component(FrameInput {
            back: true,
            ..Default::default()
        });
        world.step();
        world.add_component(FrameInput::default());
        world.step();
        assert!(
            world
                .resource::<crate::ecs::ScreenStack>()
                .is_some_and(|s| !s.captures_input),
            "back closes the open menu"
        );
    }

    // Confirm with no focus fires a full-canvas "press anywhere" region (a
    // story stage's advance region), which is itself excluded from focus.
    #[test]
    fn confirm_without_focus_fires_a_full_canvas_region() {
        let mut world = World::new_empty();
        let stage = AssetId(98);
        world.add_component(Screen {
            asset_id: stage,
            initial: true,
            fade_in_secs: 0.0,
            ..Default::default()
        });
        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: UI_REFERENCE_SIZE[0],
            height: UI_REFERENCE_SIZE[1],
            label: None,
            hover_color: None,
            hover_scale: None,
            action: "story:advance".to_string(),
            drag_handle: None,
            screen: Some(stage),
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.start().unwrap();

        world.add_component(FrameInput {
            confirm: true,
            ..Default::default()
        });
        world.step();
        let mut cursor = crate::ecs::EventCursor::default();
        let advanced = world.events::<StoryCommand>().is_some_and(|e| {
            e.read(&mut cursor)
                .into_iter()
                .any(|c| *c == StoryCommand::Advance)
        });
        assert!(advanced, "South advances the stage without a focus cursor");
    }

    // While a pad rebind row is capturing, the East button is a bindable
    // button, not a back pulse: it binds instead of cancelling.
    #[test]
    fn pad_capture_binds_east_instead_of_backing_out() {
        let mut world = World::new_empty();
        let mut label = panel_label(7, 0.0, AssetId(0), "South");
        label.screen = None;
        world.add_component(label);
        world.add_component(HitRegion {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 40.0,
            label: Some(AssetId(7)),
            hover_color: None,
            hover_scale: None,
            action: "setting:pad_jump:rebind".to_string(),
            drag_handle: None,
            screen: None,
            disabled: false,
            follow_label: false,
            fit: crate::assets::SpriteFit::Fit,
        });
        world.start().unwrap();

        // Click the row: capture begins.
        world.add_component(make_frame_input(50.0, 20.0, true));
        world.step();

        // Press East (which also raises the back pulse): it binds.
        world.add_component(FrameInput {
            captured_button: Some(crate::assets::GamepadButton::East),
            back: true,
            ..Default::default()
        });
        world.step();
        let cmds = produced_setting_commands(&world);
        assert_eq!(cmds.len(), 1);
        assert_eq!(cmds[0].setting, "pad_jump");
        assert!(matches!(
            cmds[0].op,
            SettingOp::RebindButton(crate::assets::GamepadButton::East)
        ));
    }
}
