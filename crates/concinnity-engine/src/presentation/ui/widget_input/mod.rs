// Widget input: the popups take the frame's pointer and Escape first (an open
// dialog shuts out the rest of its screen), then each widget under the pointer
// or holding the keyboard reacts. What a widget changes or tells the app is
// gathered as effects and applied once the widgets are released.
//   mod.rs       the pass, its state, and the effects
//   popups.rs    Menu, Modal, Toasts
//   items.rs     List, Grid
//   controls.rs  Tabs, Checkbox, Button, ValueSlider
//   panels.rs    movable and resizable root layouts
//   scrolls.rs   scrolling layouts
//   fields.rs    TextField focus
//   text_area.rs TextArea

mod controls;
mod fields;
mod items;
mod panels;
mod popups;
mod scrolls;
mod text_area;

use alloc::string::String;
use alloc::vec::Vec;

use crate::components::{
    Binding, FrameInput, Identity, InputKey, Screen, Toast, UiAction, UiEvent, UiState, UiValue,
    ValueOp, VariableValues,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{Entity, EventCursor, PipelineContext, Ref, StepResult};
use crate::gfx::overlay::{OverlayTransform, UI_REFERENCE_SIZE};
use crate::presentation::text_edit::LocalClipboard;
use crate::settings::SettingValues;

use super::UiInputSystem;
use super::bound::Values;
use super::panel_stack::PanelStack;
use super::region_ops::{apply_value, fire_action};
use super::screen::ScreenRegistry;
use super::widgets::point_in;

// Seconds between two presses on one row that make a double click.
const DOUBLE_CLICK_S: f64 = 0.4;
// Pixels a list scrolls per wheel unit.
const WHEEL_PX: f32 = 3.0;

// What a widget's input does once the widgets are released.
#[derive(Debug)]
enum Effect {
    // Change the value a binding names.
    Set(Binding, ValueOp),
    // Send a `UiEvent` named so, from the widget `Entity`.
    Emit(String, Entity, Option<UiValue>),
    // Fire an action, from the widget `Entity`.
    Fire(UiAction, Entity),
}

// What the widgets took of the frame's input, so nothing under them also
// reacts.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct WidgetOutcome {
    // The press landed on a widget or popup.
    pub(super) click_taken: bool,
    // Escape closed a popup or gave back the keyboard.
    pub(super) escape_taken: bool,
    // The wheel scrolled a widget, so no layout around it scrolls too.
    pub(super) wheel_taken: bool,
    // The keys that moved a widget, which key bindings and focus leave be.
    pub(super) keys: TakenKeys,
}

// The list and grid keys a widget used this frame.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct TakenKeys(u8);

impl TakenKeys {
    // The keys a list or grid moves its selection with.
    const KEYS: [InputKey; 7] = [
        InputKey::Up,
        InputKey::Down,
        InputKey::Home,
        InputKey::End,
        InputKey::Enter,
        InputKey::Left,
        InputKey::Right,
    ];

    // Mark `key` used, when it is one a list takes. Returns whether it is.
    pub(super) fn take(&mut self, key: InputKey) -> bool {
        let Some(i) = Self::KEYS.iter().position(|k| *k == key) else {
            return false;
        };
        self.0 |= 1 << i;
        true
    }

    // Whether the key named `name` was used.
    pub(super) fn has_name(self, name: &str) -> bool {
        Self::KEYS
            .iter()
            .enumerate()
            .any(|(i, k)| self.0 & (1 << i) != 0 && k.name() == name)
    }

    // Whether a focus move in `dir` would come from a used key.
    pub(super) fn has_nav(self, dir: crate::components::NavDirection) -> bool {
        use crate::components::NavDirection;
        match dir {
            NavDirection::Up => self.has_name(InputKey::Up.name()),
            NavDirection::Down => self.has_name(InputKey::Down.name()),
            NavDirection::Left => self.has_name(InputKey::Left.name()),
            NavDirection::Right => self.has_name(InputKey::Right.name()),
        }
    }
}

// The widget pass's state across frames.
#[derive(Debug, Default)]
pub(super) struct WidgetInput {
    // Seconds since the first frame, for double clicks and text presses.
    clock: f64,
    // The list row or grid cell pressed last, and when, for a double click.
    last_press: Option<(Entity, usize, f64)>,
    // The list or grid the keyboard moves: the one clicked last.
    active_list: Option<Entity>,
    // The item a held press may drag out.
    item_drag: Option<items::PendingDrag>,
    // The text area a held press is dragging over.
    dragging_text: Option<Entity>,
    // The panel a held press is moving or sizing.
    pub(super) panel_drag: Option<panels::PanelDrag>,
    // The scrolling layout whose scrollbar thumb a held press is dragging,
    // and the grab offset from the thumb's top.
    layout_thumb: Option<(Entity, f32)>,
    // The value slider a held press is dragging, and the number it last set.
    held_slider: Option<controls::HeldSlider>,
    // The list or grid whose scrollbar thumb a held press is dragging, and
    // the grab offset from the thumb's top.
    list_thumb: Option<(Entity, f32)>,
    toast_cursor: EventCursor,
    // Scratch: this frame's toasts, and the widgets' effects.
    incoming: Vec<Toast>,
    effects: Vec<Effect>,
    // Scratch: the screens an open dialog shuts.
    modal_screens: Vec<Option<AssetId>>,
    // The clipboard text areas copy to and paste from without a host one.
    clipboard: LocalClipboard,
    // The floating panels and the elements each holds.
    pub(super) stack: PanelStack,
}

impl WidgetInput {
    // Whether the rest of `screen` is shut behind an open dialog.
    pub(super) fn blocked(&self, screen: Option<AssetId>) -> bool {
        self.modal_screens.contains(&screen)
    }

    // Whether an open dialog shuts out a key binding of `screen`: one on no
    // screen reaches every screen, so any open dialog shuts it out.
    pub(super) fn shuts_keys(&self, screen: Option<AssetId>) -> bool {
        match screen {
            None => !self.modal_screens.is_empty(),
            Some(_) => self.blocked(screen),
        }
    }
}

// The pointer and frame facts every widget's input reads.
pub(super) struct Pointer<'a> {
    input: &'a FrameInput,
    // Escape, or the pad's back while a screen is open.
    escape: bool,
    overlay: &'a OverlayTransform,
    screens: &'a ScreenRegistry,
    modal_screens: &'a [Option<AssetId>],
    stack: &'a PanelStack,
    // The press has been taken by a widget or popup already.
    taken: bool,
}

impl Pointer<'_> {
    // The cursor in `screen`'s layout space.
    pub(super) fn at(&self, screen: Option<Ref<Screen>>) -> (f32, f32) {
        self.screens.cursor(
            screen.map(Ref::id),
            self.overlay,
            self.input.mouse_x,
            self.input.mouse_y,
        )
    }

    // The cursor as `entity` of `screen` sees it: off every rect while a
    // panel in front of the element's own covers the point.
    pub(super) fn at_element(&self, screen: Option<Ref<Screen>>, entity: Entity) -> (f32, f32) {
        let (x, y) = self.at(screen);
        self.seen_by(screen, entity, x, y)
    }

    // `(x, y)` as `entity` of `screen` sees it.
    pub(super) fn seen_by(
        &self,
        screen: Option<Ref<Screen>>,
        entity: Entity,
        x: f32,
        y: f32,
    ) -> (f32, f32) {
        // An element a scrolling layout clips is reached only inside it.
        if self
            .stack
            .clip_of(entity)
            .is_some_and(|clip| !point_in(x, y, clip))
        {
            return (f32::INFINITY, f32::INFINITY);
        }
        let own = screen.map(Ref::id);
        let at = |s: Option<AssetId>| {
            if s == own {
                (x, y)
            } else {
                self.screens
                    .cursor(s, self.overlay, self.input.mouse_x, self.input.mouse_y)
            }
        };
        if self.stack.reaches(own, entity, at) {
            (x, y)
        } else {
            (f32::INFINITY, f32::INFINITY)
        }
    }

    // Whether an open dialog shuts out `screen`.
    fn blocked(&self, screen: Option<Ref<Screen>>) -> bool {
        self.modal_screens.contains(&screen.map(Ref::id))
    }

    // Whether a widget of `screen` takes input now.
    pub(super) fn live(&self, screen: Option<Ref<Screen>>) -> bool {
        let id = screen.map(Ref::id);
        self.screens.receives_input(id) && !self.modal_screens.contains(&id)
    }

    // The size of the space a widget of `screen` lays out in.
    fn area(&self, screen: Option<Ref<Screen>>) -> [f32; 2] {
        if self.screens.on_canvas(screen.map(Ref::id)) {
            UI_REFERENCE_SIZE
        } else {
            self.input.viewport
        }
    }

    // Whether this frame's press is still free for a widget.
    pub(super) fn click(&self) -> bool {
        self.input.left_click && !self.taken
    }
}

// A field or text area a hidden layout holds gives up the keyboard, so typing
// never lands in something the user cannot see.
fn blur_hidden(stack: &PanelStack, ctx: &mut PipelineContext<'_>) {
    for (entity, field) in ctx.query_mut_with_entity::<crate::components::TextField>() {
        if field.focused && (stack.hides(entity) || stack.shuts(entity, field.screen.map(Ref::id)))
        {
            field.focused = false;
        }
    }
    for (entity, area) in ctx.query_mut_with_entity::<crate::components::TextArea>() {
        if area.focused && (stack.hides(entity) || stack.shuts(entity, area.screen.map(Ref::id))) {
            area.focused = false;
        }
    }
}

// Bring the frontmost panel under a press in front of the others. The stack
// keeps this frame's order, so the press itself still reaches what was in front.
fn raise_panel(
    stack: &PanelStack,
    screens: &ScreenRegistry,
    modal_screens: &[Option<AssetId>],
    input: &FrameInput,
    overlay: &OverlayTransform,
    ctx: &mut PipelineContext<'_>,
) {
    let pressed = stack.front_under(|screen| {
        (screens.receives_input(screen) && !modal_screens.contains(&screen))
            .then(|| screens.cursor(screen, overlay, input.mouse_x, input.mouse_y))
    });
    if let Some(root) = pressed
        && let Some(rank) = stack.raised_rank(root)
        && let Some(layout) = ctx.get_mut::<crate::components::Layout>(root)
    {
        layout.rank = rank;
    }
}

// The resources bindings read, taken out of the context while the widgets are
// borrowed and put back after.
#[derive(Default)]
struct TakenValues {
    state: Option<UiState>,
    vars: Option<VariableValues>,
    settings: Option<SettingValues>,
}

impl TakenValues {
    fn take(ctx: &mut PipelineContext<'_>) -> Self {
        Self {
            state: ctx.take_resource::<UiState>(),
            vars: ctx.take_resource::<VariableValues>(),
            settings: ctx.take_resource::<SettingValues>(),
        }
    }

    fn values(&self) -> Values<'_> {
        Values {
            state: self.state.as_ref(),
            vars: self.vars.as_ref(),
            settings: self.settings.as_ref(),
        }
    }

    fn put_back(self, ctx: &mut PipelineContext<'_>) {
        if let Some(state) = self.state {
            ctx.insert_resource(state);
        }
        if let Some(vars) = self.vars {
            ctx.insert_resource(vars);
        }
        if let Some(settings) = self.settings {
            ctx.insert_resource(settings);
        }
    }
}

impl UiInputSystem {
    // Drive every widget for one frame. Returns what the widgets took of the
    // input, and a result when a fired action ends the step.
    pub(super) fn step_widgets(
        &mut self,
        input: &FrameInput,
        overlay: &OverlayTransform,
        intent: &super::intent::UiIntent,
        typing: bool,
        ctx: &mut PipelineContext<'_>,
    ) -> (WidgetOutcome, Option<StepResult>) {
        let dt = ctx
            .resource::<crate::protocol::FrameTime>()
            .map_or(0.0, |t| t.dt.max(0.0));
        let w = &mut self.widgets;
        w.clock += f64::from(dt);
        w.incoming.clear();
        if let Some(events) = ctx.events::<Toast>() {
            w.incoming.extend(events.read(&mut w.toast_cursor).cloned());
        }
        let taken = TakenValues::take(ctx);
        let values = taken.values();
        let mut out = WidgetOutcome::default();
        popups::find_modals(ctx, values, &mut w.modal_screens);
        w.stack.rebuild(ctx);
        blur_hidden(&w.stack, ctx);
        if input.left_click || input.right_click {
            raise_panel(
                &w.stack,
                &self.screens,
                &w.modal_screens,
                input,
                overlay,
                ctx,
            );
        }
        let mut pointer = Pointer {
            input,
            escape: intent.ui_escape,
            overlay,
            screens: &self.screens,
            modal_screens: &w.modal_screens,
            stack: &w.stack,
            taken: false,
        };
        popups::step_toasts(ctx, &pointer, &w.incoming, dt, &mut w.effects, &mut out);
        pointer.taken = out.click_taken;
        popups::step_menus(ctx, &pointer, values, &mut w.effects, &mut out);
        pointer.taken = out.click_taken;
        popups::step_modals(ctx, &pointer, values, &mut w.effects, &mut out);
        pointer.taken = out.click_taken;
        panels::step_panel_drag(ctx, &pointer, &mut w.panel_drag, &mut out);
        pointer.taken = out.click_taken;
        let lists = items::ItemsInput {
            clock: w.clock,
            last_press: &mut w.last_press,
            active: &mut w.active_list,
            thumb: &mut w.list_thumb,
            drag: &mut w.item_drag,
        };
        items::step_item_widgets(
            ctx,
            &pointer,
            values,
            typing,
            lists,
            &mut w.effects,
            &mut out,
        );
        pointer.taken = out.click_taken;
        controls::step_tabs(ctx, &pointer, &mut w.effects, &mut out);
        pointer.taken = out.click_taken;
        controls::step_checkboxes(ctx, &pointer, values, &mut w.effects, &mut out);
        pointer.taken = out.click_taken;
        controls::step_value_sliders(
            ctx,
            &pointer,
            values,
            &mut w.held_slider,
            &mut w.effects,
            &mut out,
        );
        pointer.taken = out.click_taken;
        controls::step_buttons(ctx, &pointer, &mut w.effects, &mut out);
        pointer.taken = out.click_taken;
        let text = text_area::AreaInput {
            clock: w.clock,
            dragging: &mut w.dragging_text,
            clipboard: &mut w.clipboard,
        };
        text_area::step_text_areas(ctx, &pointer, values, text, &mut w.effects, &mut out);
        pointer.taken = out.click_taken;
        // After every widget that scrolls itself, so the one under the pointer
        // takes the wheel before a layout around it does.
        scrolls::step_scrolls(ctx, &pointer, &mut w.layout_thumb, &mut out);
        pointer.taken = out.click_taken;
        fields::step_text_fields(ctx, &pointer, &mut out);
        taken.put_back(ctx);
        let result = self.apply_effects(ctx);
        (out, result)
    }

    // Apply the effects the widgets gathered, in order. A fired action that
    // ends the step stops the rest.
    fn apply_effects(&mut self, ctx: &mut PipelineContext<'_>) -> Option<StepResult> {
        let mut effects = core::mem::take(&mut self.widgets.effects);
        let mut result = None;
        for effect in effects.drain(..) {
            match effect {
                Effect::Set(binding, op) => apply_value(&binding, &op, ctx),
                Effect::Emit(name, entity, value) => {
                    let element = ctx.get::<Identity>(entity).map(|i| i.id());
                    ctx.events_mut::<UiEvent>().send(UiEvent {
                        name,
                        element,
                        value,
                    });
                }
                Effect::Fire(action, entity) => {
                    let element = ctx.get::<Identity>(entity).map(|i| i.id());
                    if result.is_none() {
                        result = fire_action(&action, element, ctx);
                    }
                }
            }
        }
        self.widgets.effects = effects;
        result
    }
}

// Whether `world` declares any widget, so it needs the UI input system.
pub(super) fn any_widget(world: &crate::ecs::World) -> bool {
    use crate::components::{
        Button, Checkbox, Grid, List, Menu, Modal, Tabs, TextArea, Toasts, ValueSlider,
    };
    world.query::<List>().next().is_some()
        || world.query::<ValueSlider>().next().is_some()
        || world.query::<Grid>().next().is_some()
        || world.query::<Button>().next().is_some()
        || world.query::<Tabs>().next().is_some()
        || world.query::<Checkbox>().next().is_some()
        || world.query::<TextArea>().next().is_some()
        || world.query::<Menu>().next().is_some()
        || world.query::<Modal>().next().is_some()
        || world.query::<Toasts>().next().is_some()
}

// Queue the event `name` from `entity` carrying `value`, unless the name is
// empty.
fn emit(effects: &mut Vec<Effect>, name: &str, entity: Entity, value: Option<UiValue>) {
    if !name.is_empty() {
        effects.push(Effect::Emit(name.into(), entity, value));
    }
}

// Queue setting `binding` to `value`, when there is a binding.
fn set(effects: &mut Vec<Effect>, binding: Option<&Binding>, value: UiValue) {
    if let Some(b) = binding {
        effects.push(Effect::Set(b.clone(), ValueOp::Set(value)));
    }
}

// A row index as the whole number a binding holds.
fn index_value(i: usize) -> UiValue {
    UiValue::Int(i32::try_from(i).unwrap_or(i32::MAX))
}
