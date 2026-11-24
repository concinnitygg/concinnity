// Text field focus: a press no popup or widget took focuses the field under
// it (the topmost), any press blurs the rest, and a field on a screen an open
// dialog shuts out lets go of the keyboard.

use crate::components::{SpriteFit, TextField};
use crate::ecs::{Entity, PipelineContext};
use crate::gfx::overlay::OverlayTransform;

use super::super::widgets::point_in;
use super::{Pointer, WidgetOutcome};

// The cursor in a field's layout space, through its own `fit` on the canvas.
fn cursor_in(field: &TextField, pointer: &Pointer<'_>) -> (f32, f32) {
    let input = pointer.input;
    if !pointer
        .screens
        .on_canvas(field.screen.map(crate::ecs::Ref::id))
    {
        return (input.mouse_x, input.mouse_y);
    }
    let overlay = match field.fit {
        SpriteFit::Bottom => OverlayTransform::bottom_anchored_from_viewport(input.viewport),
        SpriteFit::Cover => OverlayTransform::cover_from_viewport(input.viewport),
        SpriteFit::Fit => *pointer.overlay,
    };
    overlay.inverse(input.mouse_x, input.mouse_y)
}

pub(super) fn step_text_fields(
    ctx: &mut PipelineContext<'_>,
    pointer: &Pointer<'_>,
    out: &mut WidgetOutcome,
) {
    let press = pointer.input.left_click;
    let mut hit: Option<Entity> = None;
    if pointer.click() {
        for (entity, field) in ctx.query_with_entity::<TextField>() {
            let (x, y) = cursor_in(field, pointer);
            let (x, y) = pointer.seen_by(field.screen, entity, x, y);
            let rect = [field.x, field.y, field.width, field.height];
            if field.visible && pointer.live(field.screen) && point_in(x, y, rect) {
                hit = Some(entity);
            }
        }
    }
    for (entity, field) in ctx.query_mut_with_entity::<TextField>() {
        if press {
            let focus = Some(entity) == hit;
            if focus && !field.focused {
                field.caret = field.content.chars().count();
            }
            field.focused = focus;
        } else if field.focused && pointer.blocked(field.screen) {
            field.focused = false;
        }
    }
    if hit.is_some() {
        out.click_taken = true;
    }
}
