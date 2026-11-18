// Widget input: the popups take the frame's pointer and Escape first (an open
// dialog shuts out the rest of its screen), then each widget under the pointer
// or holding the keyboard reacts. What a widget changes or tells the app is
// gathered as effects and applied once the widgets are released.
//   mod.rs       the pass, its state, and the effects
//   popups.rs    Menu, Modal, Toasts
//   controls.rs  List, Tabs, Checkbox
//   panels.rs    movable and resizable root layouts
//   fields.rs    TextField focus
//   text_area.rs TextArea

mod controls;
mod fields;
mod panels;
mod popups;
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
use super::region_ops::{apply_value, fire_action};
use super::screen::ScreenRegistry;

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
    // The keys that moved a widget, which key bindings and focus leave be.
    pub(super) keys: TakenKeys,
}

// The list keys a widget used this frame.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct TakenKeys(u8);

impl TakenKeys {
    // The keys a list moves its selection with.
    const KEYS: [InputKey; 5] = [
        InputKey::Up,
        InputKey::Down,
        InputKey::Home,
        InputKey::End,
        InputKey::Enter,
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
            NavDirection::Left | NavDirection::Right => false,
        }
    }
}

// The widget pass's state across frames.
#[derive(Debug, Default)]
pub(super) struct WidgetInput {
    // Seconds since the first frame, for double clicks and text presses.
    clock: f64,
    // The list row pressed last, and when, for a double click.
    last_press: Option<(Entity, usize, f64)>,
    // The list the keyboard moves: the one clicked last.
    active_list: Option<Entity>,
    // The text area a held press is dragging over.
    dragging_text: Option<Entity>,
    // The panel a held press is moving or sizing.
    pub(super) panel_drag: Option<panels::PanelDrag>,
    // The list whose scrollbar thumb a held press is dragging, and the grab
    // offset from the thumb's top.
    list_thumb: Option<(Entity, f32)>,
    toast_cursor: EventCursor,
    // Scratch: this frame's toasts, and the widgets' effects.
    incoming: Vec<Toast>,
    effects: Vec<Effect>,
    // Scratch: the screens an open dialog shuts.
    modal_screens: Vec<Option<AssetId>>,
    // The clipboard text areas copy to and paste from without a host one.
    clipboard: LocalClipboard,
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
        let mut pointer = Pointer {
            input,
            escape: intent.ui_escape,
            overlay,
            screens: &self.screens,
            modal_screens: &w.modal_screens,
            taken: false,
        };
        popups::step_toasts(ctx, &pointer, &w.incoming, dt, &mut out);
        pointer.taken = out.click_taken;
        popups::step_menus(ctx, &pointer, values, &mut w.effects, &mut out);
        pointer.taken = out.click_taken;
        popups::step_modals(ctx, &pointer, values, &mut w.effects, &mut out);
        pointer.taken = out.click_taken;
        panels::step_panel_drag(ctx, &pointer, &mut w.panel_drag, &mut out);
        pointer.taken = out.click_taken;
        let lists = controls::ListInput {
            clock: w.clock,
            last_press: &mut w.last_press,
            active: &mut w.active_list,
            thumb: &mut w.list_thumb,
        };
        controls::step_lists(
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
        let text = text_area::AreaInput {
            clock: w.clock,
            dragging: &mut w.dragging_text,
            clipboard: &mut w.clipboard,
        };
        text_area::step_text_areas(ctx, &pointer, values, text, &mut w.effects, &mut out);
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
    use crate::components::{Checkbox, List, Menu, Modal, Tabs, TextArea, Toasts};
    world.query::<List>().next().is_some()
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
