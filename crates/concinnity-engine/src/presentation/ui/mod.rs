//! `HitRegion` / Screen / `KeyBinding` input dispatch. An internal system (not a
//! declarable asset): the host's system table builds one whenever the world
//! contains any `HitRegion`, `Screen`, or `KeyBinding`, then it processes
//! hover/click, screen overlays, and key bindings each frame.

pub(crate) mod authored_action;
mod capture;
mod dropdown_list;
mod focus;
mod focus_step;
pub(crate) mod hit_region;
mod intent;
mod key_bindings;
pub(crate) mod layout_container;
mod panels;
mod region_ops;
mod regions;
pub(crate) mod screen;
pub(crate) mod screen_command;
mod screen_flow;
pub(crate) mod screen_shown;
mod scroll_layout;
pub(crate) mod scroll_panel;
mod slider;
pub(crate) mod sprite;
pub(crate) mod text_input;
pub(crate) mod text_label;
pub(crate) mod ui_action;

use crate::ecs::{
    Access, EventCursor, PipelineContext, Ref, StepResult, System, World, by_asset_id,
};

use crate::protocol::{OpenDropdown, ScreenStack};
use alloc::vec::Vec;

use crate::components::{
    FrameInput, HitRegion, Identity, KeyBinding, SceneCommand, Screen, ScreenCommand, ScreenShown,
    SettingCommand, SettingVerb, Sprite, SpriteFit, StoryCommand, TextInput, TextLabel, UiAction,
};
use crate::ecs::asset_id::AssetId;
use crate::gfx::overlay::OverlayTransform;
use concinnity_core::platform::CompiledPlatforms;
use hashbrown::HashMap;
use regions::RegionFrame;
use screen::{ScreenMeta, ScreenRegistry};

use crate::settings::{DisabledSettingRows, SettingKey};
use dropdown_list::{DropdownPulses, OpenDropdownState, OpenRequest};
use panels::{LayoutScratch, PanelState};
use region_ops::{fire_action, point_in_rect, region_covers_canvas, region_rect, set_label_style};

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
    // For a slider drag region (a `Setting` action with the `Drag` verb), the
    // setting key.
    // `None` for an ordinary click region. A slider region is driven by the
    // drag pass, not the click-to-fire path.
    slider_key: Option<SettingKey>,
    // The scroll panel + row this region belongs to, if it sits in a panel's
    // content band (resolved by position at init). Such a region reflows with
    // its row each frame and only fires while its row is shown and inside the
    // band. `None` for chrome (tab bar, Back) and non-panel regions.
    scroll_row: Option<(usize, usize)>,
    // The region's authored y, kept so the scroll reflow can set
    // `region.y = base_y + dy` from a fresh delta each frame.
    region_base_y: f32,
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

/// `HitRegion` / Screen / `KeyBinding` input dispatch behavior. Built by the host's
/// system table when the world declares any `HitRegion`, `Screen`, or
/// `KeyBinding`; never a world-declared asset, so it carries no config.
#[derive(Debug)]
pub(crate) struct UiInputSystem {
    regions: Vec<RegionEntry>,
    bindings: Vec<KeyBinding>,
    screens: ScreenRegistry,
    // Asset ids of UI elements (Sprite, TextLabel) by their owning screen.
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
    capturing: Option<capture::Capture>,
    // The open settings dropdown, or `None`. While set, its floating list
    // overlays the menu and consumes input until a pick / dismiss.
    open_dropdown: Option<OpenDropdownState>,
    // Cursor into the Events<ScreenCommand> queue. This system both sends (when a
    // `screen:*` action fires) and reads ScreenCommands, so a command fired this
    // frame is applied on the next.
    screen_cmd_cursor: EventCursor,
    // Cached copy of SettingsSystem's `DisabledSettingRows`, refreshed only when the
    // published resource changes so the hit-test loop reads an owned set without
    // cloning the resource every frame (SettingsSystem republishes rarely).
    disabled_rows_cache: hashbrown::HashSet<SettingKey>,
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
    follow_label_ids: hashbrown::HashSet<AssetId>,
    // Scratch: each followed label's (y, is-empty) this frame.
    follow_labels: HashMap<AssetId, (f32, bool)>,
    // Scratch buffers for the scroll-layout solve/apply.
    layout: LayoutScratch,
}

impl UiInputSystem {
    /// The system for a world that declares any `HitRegion`, `Screen` or
    /// `KeyBinding`.
    pub(crate) fn for_world(world: &World) -> Option<Self> {
        let needs = world.query::<HitRegion>().next().is_some()
            || world.query::<Screen>().next().is_some()
            || world.query::<KeyBinding>().next().is_some();
        needs.then(Self::new)
    }

    /// Empty dispatch state. The world's `HitRegion` / `Screen` / `KeyBinding`
    /// components are drained into it in [`System::init`].
    pub(crate) fn new() -> Self {
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
            screen_cmd_cursor: EventCursor::default(),
            disabled_rows_cache: hashbrown::HashSet::new(),
            focus: None,
            last_cursor: None,
            follow_label_ids: hashbrown::HashSet::new(),
            follow_labels: HashMap::new(),
            layout: LayoutScratch::default(),
        }
    }
}

impl Default for UiInputSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl System for UiInputSystem {
    fn access(&self) -> Access {
        Access::new()
            .reads_components(crate::component_mask![FrameInput, Identity])
            .writes_components(crate::component_mask![TextLabel, Sprite, TextInput])
            .reads_resources(crate::resource_mask![
                crate::ecs::EntityById,
                DisabledSettingRows,
                crate::window::display_mode::DisplayModes,
                CompiledPlatforms,
            ])
            .writes_resources(crate::resource_mask![
                OpenDropdown,
                ScreenStack,
                ScreenCommand,
                ScreenShown,
                SettingCommand,
                SceneCommand,
                StoryCommand,
                crate::protocol::ScreenshotRequest,
                crate::protocol::HudHidden,
            ])
    }

    fn init(&mut self, ctx: &mut PipelineContext<'_>) {
        // Drain Screen assets, record each one's policies, and pick the one
        // flagged `initial` to open at world start.
        let mut initial: Option<AssetId> = None;
        for (id, s) in ctx.drain_with_ids::<Screen>() {
            let Some(id) = id else { continue };
            self.screens.register(id, ScreenMeta::from_asset(&s));
            if s.initial && initial.is_none() {
                initial = Some(id);
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
            // A region SettingsSystem disabled (e.g. a capability-gated settings
            // row grayed out at init) is inert: dropping it here means it never
            // hovers, fires, drags, or reflows. Its labels are styled + reflowed
            // independently (by the renderer and the scroll panel).
            if region.disabled {
                continue;
            }
            let (original_color, original_scale) = match region.label {
                None => (None, None),
                Some(label_id) => ctx
                    .get_by_id::<TextLabel>(label_id.id())
                    .map_or((None, None), |l| (Some(l.color), Some(l.scale))),
            };
            let screen = region.screen;
            let slider_key = match region.action {
                Some(UiAction::Setting {
                    key,
                    verb: SettingVerb::Drag,
                }) => Some(key),
                _ => None,
            };
            let region_base_y = region.y;
            // A follow-label region captures the y offset to its label now, so
            // the runtime layout can move the label and the region tracks it.
            let follow = if region.follow_label {
                region.label.and_then(|lid| {
                    ctx.get_by_id::<TextLabel>(lid.id())
                        .map(|l| (lid.id(), region.y - l.y))
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
                screen: screen.map(Ref::id),
                slider_key,
                scroll_row: None,
                region_base_y,
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
        for (_, s, identity) in ctx.join2::<Sprite, Identity>() {
            if let Some(screen_id) = s.screen {
                self.sprites_by_screen
                    .entry(screen_id.id())
                    .or_default()
                    .push(identity.id());
            }
        }
        for (_, l, identity) in ctx.join2::<TextLabel, Identity>() {
            if let Some(screen_id) = l.screen {
                self.labels_by_screen
                    .entry(screen_id.id())
                    .or_default()
                    .push(identity.id());
            }
        }
        for (_, t, identity) in ctx.join2::<TextInput, Identity>() {
            if let Some(screen_id) = t.screen {
                self.text_inputs_by_screen
                    .entry(screen_id.id())
                    .or_default()
                    .push(identity.id());
            }
        }

        // Drain ScrollPanels into runtime state and bucket the regions into
        // their rows (uses the regions drained just above).
        self.init_panels(ctx);

        // Screens start hidden: zero out the visibility of every screen-owned
        // Sprite and TextLabel.
        for &id in self.sprites_by_screen.values().flatten() {
            by_asset_id::update::<Sprite>(ctx, Some(id), |sp| sp.visible = false);
        }
        for &id in self.labels_by_screen.values().flatten() {
            by_asset_id::update::<TextLabel>(ctx, Some(id), |l| l.visible = false);
        }
        for &id in self.text_inputs_by_screen.values().flatten() {
            by_asset_id::update::<TextInput>(ctx, Some(id), |ti| ti.visible = false);
        }

        // Open the initial screen (if any) through the same transition path as
        // any navigation, so its elements show, its focus field focuses, and
        // the activation is announced (AudioCue music fires for the first
        // screen too). Publish the stack resource either way so the overlay
        // reads a fresh state from frame 0.
        if let Some(id) = initial
            && let Some(transition) = self.screens.apply(ScreenCommand::Show(id))
        {
            self.apply_transition(&transition, ctx);
        }
        self.publish_screen_stack(ctx);

        // Solve the initial scroll layout so frame 0 already shows the right
        // collapsed/scrolled positions (a default-collapsed group starts shut).
        self.apply_scroll_layout(ctx);
    }

    fn step(&mut self, ctx: &mut PipelineContext<'_>) -> StepResult {
        self.apply_pending_screen_commands(ctx);

        // Read (not drain) the latest input snapshot: Camera3DSystem queries it
        // too, and the renderer clears it before the next push.
        let Some(input) = ctx.query::<FrameInput>().last().cloned() else {
            return StepResult::Continue;
        };

        // While any visible TextInput has keyboard focus, typed keys belong to
        // the field: ordinary KeyBindings and the focus pulses are suspended.
        let typing = ctx.query::<TextInput>().any(|t| t.visible && t.focused);
        let intent = intent::frame_intent(
            &input,
            typing,
            self.screens.top_capture().is_some(),
            self.focus.is_some(),
            self.last_cursor,
        );
        // Mouse movement dismisses the focus cursor: the menu returns to
        // hover-driven interaction until the next pulse.
        self.last_cursor = Some((input.mouse_x, input.mouse_y));
        if intent.cursor_moved {
            self.focus = None;
        }

        // An open dropdown's floating list consumes the frame ahead of the
        // key bindings and regions, so Escape closes the list rather than the
        // menu and a pick never falls through to the row behind it.
        if self.open_dropdown.is_some() {
            let pulses = DropdownPulses::from_intent(&intent);
            self.step_open_dropdown(&input, intent.nav, pulses, ctx);
            self.publish_dropdown(ctx);
            return StepResult::Continue;
        }
        if self.capturing.is_some() {
            return self.step_rebind_capture(&input, ctx);
        }
        if let Some(result) = self.dispatch_key_bindings(&intent, typing, ctx) {
            return result;
        }

        let active_screen = self.screens.top_capture();
        // Screen-owned UI is authored in the reference canvas (crate::gfx::overlay).
        let overlay = OverlayTransform::from_viewport(input.viewport);
        // A thumb drag suppresses the slider and region passes so the gutter
        // does not double as a control.
        let thumb_active = self.handle_scroll_input(&input, active_screen, &overlay);
        if !thumb_active {
            self.step_slider_drag(&input, active_screen, &overlay, ctx);
        }
        self.refresh_disabled_rows(ctx);
        if let Some(dir) = intent.nav {
            self.step_focus(dir, active_screen, ctx);
        }

        let bands = ctx.frame.collect(self.panels.iter().map(|p| p.band));
        let frame = RegionFrame::new(input.viewport, active_screen, thumb_active, overlay, bands);
        let outcome = self.hit_test_regions(&input, &intent, &frame, ctx);
        if let Some(result) = outcome.fired {
            return result;
        }
        if let Some((setting_key, value_label)) = outcome.start_capture {
            self.begin_capture(setting_key, value_label, ctx);
        }
        if let Some(req) = outcome.start_open {
            self.open_dropdown = Self::build_open_dropdown(&req, ctx);
        }
        if let Some(gid) = outcome.toggle_group {
            self.toggle_group(active_screen, gid);
        }
        // Solve every panel so the next frame draws and hit-tests the reflowed
        // layout, then publish the dropdown state (a just-opened list, or none).
        self.apply_scroll_layout(ctx);
        self.publish_dropdown(ctx);
        StepResult::Continue
    }
}

#[cfg(test)]
mod tests;
