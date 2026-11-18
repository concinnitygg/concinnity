// The controls' input: a List scrolled, hovered, clicked, and moved by the
// keyboard; a Tabs strip and a Checkbox clicked.

use alloc::vec::Vec;

use crate::components::{Button, Checkbox, InputKey, List, Tabs, UiValue, ValueOp};
use crate::ecs::{Entity, PipelineContext};

use super::super::bound::Values;
use super::super::widgets::list::EXPANDER_W;
use super::super::widgets::point_in;
use super::{DOUBLE_CLICK_S, Effect, Pointer, WHEEL_PX, WidgetOutcome, emit, index_value, set};

// The list state the pass keeps across frames.
pub(super) struct ListInput<'a> {
    pub(super) clock: f64,
    pub(super) last_press: &'a mut Option<(Entity, usize, f64)>,
    pub(super) active: &'a mut Option<Entity>,
    pub(super) thumb: &'a mut Option<(Entity, f32)>,
}

pub(super) fn step_lists(
    ctx: &mut PipelineContext<'_>,
    pointer: &Pointer<'_>,
    values: Values<'_>,
    typing: bool,
    state: ListInput<'_>,
    effects: &mut Vec<Effect>,
    out: &mut WidgetOutcome,
) {
    let ListInput {
        clock,
        last_press,
        active,
        thumb,
    } = state;
    let mut clicked_any = false;
    for (entity, list) in ctx.query_mut_with_entity::<List>() {
        let items = values.state.map_or(&[][..], |s| s.list(&list.items));
        let count = items.len();
        list.scroll = list.scroll.clamp(0.0, list.max_scroll(count));
        if !list.visible || !pointer.live(list.screen) {
            list.hovered = None;
            continue;
        }
        let (x, y) = pointer.at_element(list.screen, entity);
        let over = point_in(x, y, list.rect());
        list.hovered = over.then(|| list.row_at(x, y, count)).flatten();
        if let Some((grabbed, grab)) = *thumb
            && grabbed == entity
        {
            if pointer.input.left_button_down {
                list.scroll = list.scroll_for_thumb(y - grab, count);
                out.click_taken = true;
                continue;
            }
            *thumb = None;
        }
        if over && pointer.input.scroll_delta != 0.0 {
            list.scroll = (list.scroll - pointer.input.scroll_delta * WHEEL_PX)
                .clamp(0.0, list.max_scroll(count));
        }
        let selected = list.selected.as_ref().and_then(|b| values.int(b));
        if *active == Some(entity) && !typing {
            step_list_keys(
                list,
                entity,
                pointer,
                count,
                selected,
                &mut out.keys,
                effects,
            );
        }
        let right = pointer.input.right_click && over && !pointer.taken;
        let left = pointer.click() && over;
        if !left && !right {
            continue;
        }
        out.click_taken = true;
        clicked_any = true;
        *active = Some(entity);
        if left
            && let Some((track, bar)) = list.scrollbar(count)
            && point_in(x, y, track)
        {
            // A press on the thumb grabs it where it was pressed; one on the
            // track beside it centers the thumb there first.
            let grab = if point_in(x, y, bar) {
                y - bar[1]
            } else {
                bar[3] * 0.5
            };
            list.scroll = list.scroll_for_thumb(y - grab, count);
            *thumb = Some((entity, grab));
            continue;
        }
        let Some(i) = list.hovered else {
            continue;
        };
        let item = &items[i];
        let ex = list.expander_x(item.depth);
        if right {
            set(effects, list.selected.as_ref(), index_value(i));
            emit(effects, &list.on_context, entity, Some(index_value(i)));
            continue;
        }
        if item.expandable && x >= ex && x < ex + EXPANDER_W {
            emit(effects, &list.on_expand, entity, Some(index_value(i)));
            continue;
        }
        set(effects, list.selected.as_ref(), index_value(i));
        emit(effects, &list.on_select, entity, Some(index_value(i)));
        let double = matches!(*last_press, Some((e, row, at))
            if e == entity && row == i && clock - at <= DOUBLE_CLICK_S);
        if double {
            emit(effects, &list.on_activate, entity, Some(index_value(i)));
            *last_press = None;
        } else {
            *last_press = Some((entity, i, clock));
        }
    }
    // A click on anything else lets the keyboard go.
    if pointer.click() && !clicked_any && !pointer.taken {
        *active = None;
    }
}

// Up and Down move the active list's selection, Home and End jump it, and
// Enter activates it; each one pressed is marked `taken`.
fn step_list_keys(
    list: &mut List,
    entity: Entity,
    pointer: &Pointer<'_>,
    count: usize,
    selected: Option<i32>,
    taken: &mut super::TakenKeys,
    effects: &mut Vec<Effect>,
) {
    if count == 0 {
        return;
    }
    let current = selected.and_then(|i| usize::try_from(i).ok());
    let mut next = current;
    for key in pointer.input.pressed_keys() {
        if !taken.take(key) {
            continue;
        }
        next = match key {
            InputKey::Down => Some(next.map_or(0, |i| (i + 1).min(count - 1))),
            InputKey::Up => Some(next.map_or(0, |i| i.saturating_sub(1))),
            InputKey::Home => Some(0),
            InputKey::End => Some(count - 1),
            InputKey::Enter => {
                if let Some(i) = next {
                    emit(effects, &list.on_activate, entity, Some(index_value(i)));
                }
                next
            }
            _ => next,
        };
    }
    if let Some(i) = next
        && next != current
    {
        list.reveal(i, count);
        set(effects, list.selected.as_ref(), index_value(i));
        emit(effects, &list.on_select, entity, Some(index_value(i)));
    }
}

pub(super) fn step_tabs(
    ctx: &mut PipelineContext<'_>,
    pointer: &Pointer<'_>,
    effects: &mut Vec<Effect>,
    out: &mut WidgetOutcome,
) {
    for (entity, tabs) in ctx.query_mut_with_entity::<Tabs>() {
        if !tabs.visible || !pointer.live(tabs.screen) {
            tabs.hovered = None;
            continue;
        }
        let (x, y) = pointer.at_element(tabs.screen, entity);
        tabs.hovered = tabs.tab_at(x, y);
        if !pointer.click() {
            continue;
        }
        if let Some(i) = tabs.hovered {
            out.click_taken = true;
            set(effects, tabs.selected.as_ref(), index_value(i));
            emit(effects, &tabs.on_select, entity, Some(index_value(i)));
        }
    }
}

pub(super) fn step_checkboxes(
    ctx: &mut PipelineContext<'_>,
    pointer: &Pointer<'_>,
    values: Values<'_>,
    effects: &mut Vec<Effect>,
    out: &mut WidgetOutcome,
) {
    for (entity, check) in ctx.query_mut_with_entity::<Checkbox>() {
        if !check.visible || !pointer.live(check.screen) {
            check.hovered = false;
            continue;
        }
        let (x, y) = pointer.at_element(check.screen, entity);
        check.hovered = point_in(x, y, check.rect());
        if !(pointer.click() && check.hovered) || check.disabled {
            continue;
        }
        out.click_taken = true;
        let Some(binding) = &check.checked else {
            continue;
        };
        let now = !values.bool(binding).unwrap_or(false);
        effects.push(Effect::Set(binding.clone(), ValueOp::Flip));
        emit(effects, &check.on_change, entity, Some(UiValue::Bool(now)));
    }
}

pub(super) fn step_buttons(
    ctx: &mut PipelineContext<'_>,
    pointer: &Pointer<'_>,
    effects: &mut Vec<Effect>,
    out: &mut WidgetOutcome,
) {
    for (entity, button) in ctx.query_mut_with_entity::<Button>() {
        if !button.visible || !pointer.live(button.screen) {
            button.hovered = false;
            continue;
        }
        let (x, y) = pointer.at_element(button.screen, entity);
        button.hovered = point_in(x, y, button.rect());
        if !(pointer.click() && button.hovered) || button.disabled {
            continue;
        }
        out.click_taken = true;
        if let Some(action) = &button.action {
            effects.push(Effect::Fire(action.clone(), entity));
        }
    }
}
