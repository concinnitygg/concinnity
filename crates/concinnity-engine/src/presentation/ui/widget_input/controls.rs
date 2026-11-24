// The controls' input: a Tabs strip, a Checkbox, and a Button clicked, and a
// ValueSlider dragged.

use alloc::string::String;
use alloc::vec::Vec;

use crate::components::{Button, Checkbox, Tabs, UiValue, ValueOp, ValueSlider};
use crate::ecs::{Entity, PipelineContext};

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

// A value slider a held press is dragging: the slider, the number it last
// set, and the event letting go sends.
#[derive(Debug)]
pub(super) struct HeldSlider {
    entity: Entity,
    value: f32,
    on_commit: String,
}

// Each value slider: a press on its track grabs it, and while the button stays
// down the number follows the cursor along the track. A held slider that
// lets go, or that the press can no longer reach, commits the number it set.
pub(super) fn step_value_sliders(
    ctx: &mut PipelineContext<'_>,
    pointer: &Pointer<'_>,
    values: Values<'_>,
    held: &mut Option<HeldSlider>,
    effects: &mut Vec<Effect>,
    out: &mut WidgetOutcome,
) {
    let mut reached = false;
    for (entity, slider) in ctx.query_mut_with_entity::<ValueSlider>() {
        let grabbed = held.as_ref().is_some_and(|h| h.entity == entity);
        if !slider.visible || !pointer.live(slider.screen) {
            slider.hovered = false;
            continue;
        }
        let (x, y) = pointer.at_element(slider.screen, entity);
        slider.hovered = point_in(x, y, slider.grab_rect());
        if !grabbed {
            if !(pointer.click() && slider.hovered) || slider.disabled {
                continue;
            }
            let start = slider
                .value
                .as_ref()
                .and_then(|b| values.number(b))
                .unwrap_or(slider.min);
            *held = Some(HeldSlider {
                entity,
                value: start,
                on_commit: slider.on_commit.clone(),
            });
        }
        reached = true;
        out.click_taken = true;
        let Some(h) = held.as_mut() else {
            continue;
        };
        if !pointer.input.left_button_down {
            emit(effects, &h.on_commit, entity, Some(UiValue::Float(h.value)));
            *held = None;
            continue;
        }
        // While held the cursor drives the number even off the slider.
        let (x, _) = pointer.at(slider.screen);
        let value = slider.value_at(x);
        if (value - h.value).abs() > f32::EPSILON || !grabbed {
            if let Some(binding) = &slider.value {
                effects.push(Effect::Set(
                    binding.clone(),
                    ValueOp::Set(UiValue::Float(value)),
                ));
            }
            emit(
                effects,
                &slider.on_change,
                entity,
                Some(UiValue::Float(value)),
            );
            h.value = value;
        }
    }
    if !reached && let Some(h) = held.take() {
        emit(
            effects,
            &h.on_commit,
            h.entity,
            Some(UiValue::Float(h.value)),
        );
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
