// The controls' input: a Tabs strip, a Checkbox, and a Button clicked.

use alloc::vec::Vec;

use crate::components::{Button, Checkbox, Tabs, UiValue, ValueOp};
use crate::ecs::PipelineContext;

use super::super::bound::Values;
use super::super::widgets::point_in;
use super::{Effect, Pointer, WidgetOutcome, emit, index_value, set};

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
