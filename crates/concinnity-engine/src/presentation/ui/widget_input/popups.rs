// The popups' input: toasts dismissed by a click, an open menu picked from or
// closed, an open dialog's buttons pressed or Escape closing it.

use alloc::vec::Vec;

use crate::components::{Menu, Modal, Toast, Toasts, UiValue};
use crate::ecs::PipelineContext;
use crate::ecs::asset_id::AssetId;

use super::super::bound::Values;
use super::{Effect, Pointer, WidgetOutcome, emit, index_value, point_in, set};

// The screens an open dialog shuts this frame.
pub(super) fn find_modals(
    ctx: &PipelineContext<'_>,
    values: Values<'_>,
    out: &mut Vec<Option<AssetId>>,
) {
    out.clear();
    for modal in ctx.query::<Modal>() {
        if is_open(modal.open.as_ref(), values) {
            out.push(modal.screen.map(crate::ecs::Ref::id));
        }
    }
}

fn is_open(open: Option<&crate::components::Binding>, values: Values<'_>) -> bool {
    open.and_then(|b| values.bool(b)).unwrap_or(false)
}

// Add the frame's toasts to every stack, count their time off, and dismiss
// the card a click lands on, unless it shows progress. A click on the line
// counting the waiting cards reports it.
pub(super) fn step_toasts(
    ctx: &mut PipelineContext<'_>,
    pointer: &Pointer<'_>,
    incoming: &[Toast],
    dt: f32,
    effects: &mut Vec<Effect>,
    out: &mut WidgetOutcome,
) {
    for (entity, toasts) in ctx.query_mut_with_entity::<Toasts>() {
        for toast in incoming {
            toasts.push(toast);
        }
        toasts.tick(dt);
        let screen = toasts.screen.map(crate::ecs::Ref::id);
        if !toasts.visible
            || pointer.stack.shuts(entity, screen)
            || !pointer.click()
            || !pointer
                .screens
                .receives_input(toasts.screen.map(crate::ecs::Ref::id))
        {
            continue;
        }
        let (x, y) = pointer.at(toasts.screen);
        let area = pointer.area(toasts.screen);
        if let Some(i) = toasts.card_at(x, y, area) {
            out.click_taken = true;
            if toasts.shown[i].progress.is_none() {
                let card = toasts.shown.remove(i);
                emit(
                    effects,
                    &toasts.on_dismiss,
                    entity,
                    Some(UiValue::Text(card.key)),
                );
            }
        } else if toasts.more_rect(area).is_some_and(|r| point_in(x, y, r)) {
            out.click_taken = true;
            emit(effects, &toasts.on_more, entity, None);
        }
    }
}

// An open menu: placed when it opens, hovered, picked from, or closed by a
// click outside it or Escape. It takes the frame's press either way.
pub(super) fn step_menus(
    ctx: &mut PipelineContext<'_>,
    pointer: &Pointer<'_>,
    values: Values<'_>,
    effects: &mut Vec<Effect>,
    out: &mut WidgetOutcome,
) {
    for (entity, menu) in ctx.query_mut_with_entity::<Menu>() {
        if !is_open(menu.open.as_ref(), values) {
            menu.at = None;
            menu.hovered = None;
            continue;
        }
        let count = menu.entry_count(values.state);
        let (x, y) = pointer.at(menu.screen);
        // An open menu follows its `x` and `y`, or stays where the cursor
        // opened it, and stays on screen as its entries grow, such as when a
        // disclosure unfolds.
        let at = if menu.at_cursor {
            menu.at.unwrap_or([x, y])
        } else {
            [menu.x, menu.y]
        };
        menu.open_at(at, count, pointer.area(menu.screen));
        if !pointer.live(menu.screen)
            || pointer
                .stack
                .shuts(entity, menu.screen.map(crate::ecs::Ref::id))
        {
            continue;
        }
        menu.hovered = menu.row_at(x, y, count);
        if pointer.escape {
            close_menu(menu, effects);
            out.escape_taken = true;
            continue;
        }
        if pointer.input.right_click && !pointer.taken {
            // A press on the menu stays with it. One elsewhere moves a menu
            // that follows the cursor to it, and closes any other; the press
            // goes on to what is under it, so a list there selects the row the
            // menu now names.
            if point_in(x, y, menu.rect(count)) {
                out.click_taken = true;
            } else if menu.at_cursor {
                menu.open_at([x, y], count, pointer.area(menu.screen));
                menu.hovered = None;
            } else {
                close_menu(menu, effects);
            }
            continue;
        }
        if !pointer.click() {
            continue;
        }
        out.click_taken = true;
        let Some(i) = menu.hovered else {
            close_menu(menu, effects);
            continue;
        };
        let stay_open = if menu.items.is_empty() {
            match menu.entries.get(i) {
                Some(entry) if !entry.disabled => {
                    if let Some(action) = &entry.action {
                        effects.push(Effect::Fire(action.clone(), entity));
                    }
                    entry.stay_open
                }
                _ => continue,
            }
        } else {
            let Some(item) = values
                .state
                .and_then(|s| s.list(&menu.items).get(i))
                .filter(|item| !item.dimmed)
            else {
                continue;
            };
            emit(effects, &menu.on_pick, entity, Some(index_value(i)));
            item.expandable
        };
        if !stay_open {
            close_menu(menu, effects);
        }
    }
}

fn close_menu(menu: &mut Menu, effects: &mut Vec<Effect>) {
    set(effects, menu.open.as_ref(), UiValue::Bool(false));
    menu.at = None;
    menu.hovered = None;
}

// An open dialog: its buttons hovered and pressed, Escape closing it. It takes
// the frame's press and Escape either way, so a click on the dim around it
// does not reach the screen behind.
pub(super) fn step_modals(
    ctx: &mut PipelineContext<'_>,
    pointer: &Pointer<'_>,
    values: Values<'_>,
    effects: &mut Vec<Effect>,
    out: &mut WidgetOutcome,
) {
    for (entity, modal) in ctx.query_mut_with_entity::<Modal>() {
        if !is_open(modal.open.as_ref(), values) {
            modal.hovered = None;
            continue;
        }
        if !pointer
            .screens
            .receives_input(modal.screen.map(crate::ecs::Ref::id))
        {
            continue;
        }
        let (x, y) = pointer.at(modal.screen);
        modal.hovered = modal.button_at(x, y);
        if pointer.escape {
            set(effects, modal.open.as_ref(), UiValue::Bool(false));
            out.escape_taken = true;
            continue;
        }
        if !pointer.click() {
            continue;
        }
        out.click_taken = true;
        if let Some(i) = modal.hovered {
            if let Some(action) = modal.buttons.get(i).and_then(|b| b.action.clone()) {
                effects.push(Effect::Fire(action, entity));
            }
            set(effects, modal.open.as_ref(), UiValue::Bool(false));
        }
    }
}
