// The input of the widgets that show a list's items, a List's rows or a Grid's
// cells: scrolled by the wheel and the scrollbar, hovered, clicked, dragged
// out, and moved by the keyboard.

use alloc::vec::Vec;

use crate::components::{Binding, Grid, InputKey, List, Screen, UiItem};
use crate::ecs::{ComponentSlot, Entity, PipelineContext, Ref};

use super::super::bound::Values;
use super::super::widgets::list::RowPart;
use super::super::widgets::point_in;
use super::super::widgets::scroll_view::ScrollView;
use super::{DOUBLE_CLICK_S, Effect, Pointer, WHEEL_PX, WidgetOutcome, emit, index_value, set};

// Pixels a press travels before it drags its item out.
const DRAG_PX: f32 = 4.0;

// A press on an item that drags it out once it travels: the widget, the item,
// and where the press began.
#[derive(Debug, Clone, Copy)]
pub(super) struct PendingDrag {
    entity: Entity,
    item: usize,
    at: (f32, f32),
}

// The item-widget state the pass keeps across frames.
pub(super) struct ItemsInput<'a> {
    pub(super) clock: f64,
    pub(super) last_press: &'a mut Option<(Entity, usize, f64)>,
    pub(super) active: &'a mut Option<Entity>,
    pub(super) thumb: &'a mut Option<(Entity, f32)>,
    pub(super) drag: &'a mut Option<PendingDrag>,
}

// The events an item widget sends.
struct ItemEvents<'a> {
    select: &'a str,
    activate: &'a str,
    context: &'a str,
    extend: &'a str,
    drag: &'a str,
}

// What a press on an item does beyond selecting it.
enum Press<'a> {
    Select,
    // Send this event with the item's index and nothing else.
    Only(&'a str),
}

// A widget that shows the items of a list in the UI state.
trait ItemWidget: ComponentSlot {
    fn shown(&self) -> bool;
    fn screen(&self) -> Option<Ref<Screen>>;
    fn items(&self) -> &str;
    fn selected(&self) -> Option<&Binding>;
    fn rect(&self) -> [f32; 4];
    fn view(&self, count: usize) -> ScrollView;
    fn set_scroll(&mut self, scroll: f32);
    fn set_hovered(&mut self, hovered: Option<usize>);
    fn item_at(&self, x: f32, y: f32, count: usize) -> Option<usize>;
    fn events(&self) -> ItemEvents<'_>;
    // What a left press at `x` on `item` does.
    fn press(&self, _x: f32, _item: &UiItem) -> Press<'_> {
        Press::Select
    }
    // Whether the widget moves its selection with `key`.
    fn takes(&self, _key: InputKey) -> bool {
        true
    }
    // The item `key` moves the selection to from `current`.
    fn step(&self, key: InputKey, current: Option<usize>, count: usize) -> Option<usize>;
    fn reveal(&mut self, i: usize, count: usize);
}

impl ItemWidget for List {
    fn shown(&self) -> bool {
        self.visible
    }
    fn screen(&self) -> Option<Ref<Screen>> {
        self.screen
    }
    fn items(&self) -> &str {
        &self.items
    }
    fn selected(&self) -> Option<&Binding> {
        self.selected.as_ref()
    }
    fn rect(&self) -> [f32; 4] {
        Self::rect(self)
    }
    fn view(&self, count: usize) -> ScrollView {
        Self::view(self, count)
    }
    fn set_scroll(&mut self, scroll: f32) {
        self.scroll = scroll;
    }
    fn set_hovered(&mut self, hovered: Option<usize>) {
        self.hovered = hovered;
    }
    fn item_at(&self, x: f32, y: f32, count: usize) -> Option<usize> {
        self.row_at(x, y, count)
    }
    fn events(&self) -> ItemEvents<'_> {
        ItemEvents {
            select: &self.on_select,
            activate: &self.on_activate,
            context: &self.on_context,
            extend: &self.on_extend,
            drag: &self.on_drag,
        }
    }
    fn press(&self, x: f32, item: &UiItem) -> Press<'_> {
        match self.part_at(x, item) {
            RowPart::Expander => Press::Only(&self.on_expand),
            RowPart::Toggle(t) => Press::Only(&self.toggles[t].on_toggle),
            RowPart::Body => Press::Select,
        }
    }
    fn takes(&self, key: InputKey) -> bool {
        !matches!(key, InputKey::Left | InputKey::Right)
    }
    fn step(&self, key: InputKey, current: Option<usize>, count: usize) -> Option<usize> {
        match key {
            InputKey::Down => Some(current.map_or(0, |i| (i + 1).min(count - 1))),
            InputKey::Up => Some(current.map_or(0, |i| i.saturating_sub(1))),
            InputKey::Home => Some(0),
            InputKey::End => Some(count - 1),
            _ => current,
        }
    }
    fn reveal(&mut self, i: usize, count: usize) {
        Self::reveal(self, i, count);
    }
}

impl ItemWidget for Grid {
    fn shown(&self) -> bool {
        self.visible
    }
    fn screen(&self) -> Option<Ref<Screen>> {
        self.screen
    }
    fn items(&self) -> &str {
        &self.items
    }
    fn selected(&self) -> Option<&Binding> {
        self.selected.as_ref()
    }
    fn rect(&self) -> [f32; 4] {
        Self::rect(self)
    }
    fn view(&self, count: usize) -> ScrollView {
        Self::view(self, count)
    }
    fn set_scroll(&mut self, scroll: f32) {
        self.scroll = scroll;
    }
    fn set_hovered(&mut self, hovered: Option<usize>) {
        self.hovered = hovered;
    }
    fn item_at(&self, x: f32, y: f32, count: usize) -> Option<usize> {
        self.cell_at(x, y, count)
    }
    fn events(&self) -> ItemEvents<'_> {
        ItemEvents {
            select: &self.on_select,
            activate: &self.on_activate,
            context: &self.on_context,
            extend: &self.on_extend,
            drag: &self.on_drag,
        }
    }
    fn step(&self, key: InputKey, current: Option<usize>, count: usize) -> Option<usize> {
        let columns = self.columns(count);
        let last = count - 1;
        match key {
            InputKey::Right => Some(current.map_or(0, |i| (i + 1).min(last))),
            InputKey::Left => Some(current.map_or(0, |i| i.saturating_sub(1))),
            InputKey::Down => Some(current.map_or(0, |i| (i + columns).min(last))),
            InputKey::Up => Some(current.map_or(0, |i| i.saturating_sub(columns))),
            InputKey::Home => Some(0),
            InputKey::End => Some(last),
            _ => current,
        }
    }
    fn reveal(&mut self, i: usize, count: usize) {
        Self::reveal(self, i, count);
    }
}

pub(super) fn step_item_widgets(
    ctx: &mut PipelineContext<'_>,
    pointer: &Pointer<'_>,
    values: Values<'_>,
    typing: bool,
    state: ItemsInput<'_>,
    effects: &mut Vec<Effect>,
    out: &mut WidgetOutcome,
) {
    let ItemsInput {
        clock,
        last_press,
        active,
        thumb,
        drag,
    } = state;
    let mut state = Stepping {
        clock,
        last_press,
        active,
        thumb,
        drag,
        clicked_any: false,
    };
    step_each::<List>(ctx, pointer, values, typing, &mut state, effects, out);
    step_each::<Grid>(ctx, pointer, values, typing, &mut state, effects, out);
    if !pointer.input.left_button_down {
        *state.drag = None;
    }
    // A click on anything else lets the keyboard go.
    if pointer.click() && !state.clicked_any && !pointer.taken {
        *state.active = None;
    }
}

struct Stepping<'a> {
    clock: f64,
    last_press: &'a mut Option<(Entity, usize, f64)>,
    active: &'a mut Option<Entity>,
    thumb: &'a mut Option<(Entity, f32)>,
    drag: &'a mut Option<PendingDrag>,
    clicked_any: bool,
}

fn step_each<W: ItemWidget>(
    ctx: &mut PipelineContext<'_>,
    pointer: &Pointer<'_>,
    values: Values<'_>,
    typing: bool,
    state: &mut Stepping<'_>,
    effects: &mut Vec<Effect>,
    out: &mut WidgetOutcome,
) {
    for (entity, widget) in ctx.query_mut_with_entity::<W>() {
        let items = values.state.map_or(&[][..], |s| s.list(widget.items()));
        let count = items.len();
        let view = widget.view(count);
        widget.set_scroll(view.scroll.clamp(0.0, view.max_scroll()));
        if !widget.shown() || !pointer.live(widget.screen()) {
            widget.set_hovered(None);
            continue;
        }
        step_drag(widget, entity, pointer, state.drag, effects);
        let (x, y) = pointer.at_element(widget.screen(), entity);
        let over = point_in(x, y, widget.rect());
        let hovered = over.then(|| widget.item_at(x, y, count)).flatten();
        widget.set_hovered(hovered);
        let view = widget.view(count);
        if let Some((grabbed, grab)) = *state.thumb
            && grabbed == entity
        {
            if pointer.input.left_button_down {
                widget.set_scroll(view.scroll_for_thumb(y - grab));
                out.click_taken = true;
                continue;
            }
            *state.thumb = None;
        }
        if over && pointer.input.scroll_delta != 0.0 {
            widget.set_scroll(view.wheeled(pointer.input.scroll_delta * WHEEL_PX));
        }
        let selected = widget.selected().and_then(|b| values.int(b));
        // The keys move a selection the widget keeps in a binding; with none,
        // there is nothing for them to move from.
        if *state.active == Some(entity) && !typing && widget.selected().is_some() {
            step_keys(widget, entity, pointer, count, selected, out, effects);
        }
        let right = pointer.input.right_click && over && !pointer.taken;
        let left = pointer.click() && over;
        if !left && !right {
            continue;
        }
        out.click_taken = true;
        state.clicked_any = true;
        *state.active = Some(entity);
        if left
            && let Some((track, bar)) = view.scrollbar()
            && point_in(x, y, track)
        {
            // A press on the thumb grabs it where it was pressed; one on the
            // track beside it centers the thumb there first.
            let grab = if point_in(x, y, bar) {
                y - bar[1]
            } else {
                bar[3] * 0.5
            };
            widget.set_scroll(view.scroll_for_thumb(y - grab));
            *state.thumb = Some((entity, grab));
            continue;
        }
        let Some(i) = hovered else {
            continue;
        };
        let events = widget.events();
        if right {
            set(effects, widget.selected(), index_value(i));
            emit(effects, events.context, entity, Some(index_value(i)));
            continue;
        }
        if let Press::Only(event) = widget.press(x, &items[i]) {
            emit(effects, event, entity, Some(index_value(i)));
            continue;
        }
        let input = pointer.input;
        let extend = (input.shift || input.ctrl || input.cmd) && !events.extend.is_empty();
        if extend {
            emit(effects, events.extend, entity, Some(index_value(i)));
            *state.last_press = None;
            continue;
        }
        set(effects, widget.selected(), index_value(i));
        emit(effects, events.select, entity, Some(index_value(i)));
        if !events.drag.is_empty() {
            *state.drag = Some(PendingDrag {
                entity,
                item: i,
                at: pointer.at(widget.screen()),
            });
        }
        let double = matches!(*state.last_press, Some((e, at, when))
            if e == entity && at == i && state.clock - when <= DOUBLE_CLICK_S);
        if double {
            emit(effects, events.activate, entity, Some(index_value(i)));
            *state.last_press = None;
        } else {
            *state.last_press = Some((entity, i, state.clock));
        }
    }
}

// Send the drag of the item a held press on `widget` began on, once the
// pointer has traveled far enough from where it began.
fn step_drag<W: ItemWidget>(
    widget: &W,
    entity: Entity,
    pointer: &Pointer<'_>,
    drag: &mut Option<PendingDrag>,
    effects: &mut Vec<Effect>,
) {
    let Some(pending) = *drag else {
        return;
    };
    if pending.entity != entity || !pointer.input.left_button_down {
        return;
    }
    let (x, y) = pointer.at(widget.screen());
    let (dx, dy) = (x - pending.at.0, y - pending.at.1);
    if dx * dx + dy * dy >= DRAG_PX * DRAG_PX {
        emit(
            effects,
            widget.events().drag,
            entity,
            Some(index_value(pending.item)),
        );
        *drag = None;
    }
}

// The arrows move the active widget's selection, Home and End jump it, and
// Enter activates it; each one pressed is marked `taken`.
fn step_keys<W: ItemWidget>(
    widget: &mut W,
    entity: Entity,
    pointer: &Pointer<'_>,
    count: usize,
    selected: Option<i32>,
    out: &mut WidgetOutcome,
    effects: &mut Vec<Effect>,
) {
    if count == 0 {
        return;
    }
    let current = selected.and_then(|i| usize::try_from(i).ok());
    let mut next = current;
    for key in pointer.input.pressed_keys() {
        if !widget.takes(key) || !out.keys.take(key) {
            continue;
        }
        if key == InputKey::Enter {
            if let Some(i) = next {
                emit(
                    effects,
                    widget.events().activate,
                    entity,
                    Some(index_value(i)),
                );
            }
        } else {
            next = widget.step(key, next, count);
        }
    }
    if let Some(i) = next
        && next != current
    {
        widget.reveal(i, count);
        set(effects, widget.selected(), index_value(i));
        emit(
            effects,
            widget.events().select,
            entity,
            Some(index_value(i)),
        );
    }
}
