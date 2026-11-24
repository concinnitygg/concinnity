// Canvases: a press on a tagged shape reports it, and a pannable canvas is
// dragged by a press on its empty space and scrolled by the wheel.

use alloc::vec::Vec;

use crate::components::{Canvas, UiShape, UiValue};
use crate::ecs::{Entity, PipelineContext};

use super::super::bound::Values;
use super::super::widgets::point_in;
use super::{Effect, Pointer, WHEEL_PX, WidgetOutcome, emit};

// How far one wheel notch pans a canvas.
const WHEEL_PAN: f32 = WHEEL_PX * 10.0;

// A canvas a held press is panning, and where the cursor was last frame.
#[derive(Debug)]
pub(super) struct HeldCanvas {
    entity: Entity,
    last: (f32, f32),
}

pub(super) fn step_canvases(
    ctx: &mut PipelineContext<'_>,
    pointer: &Pointer<'_>,
    values: Values<'_>,
    held: &mut Option<HeldCanvas>,
    effects: &mut Vec<Effect>,
    out: &mut WidgetOutcome,
) {
    let mods = pointer.input.ctrl || pointer.input.cmd || pointer.input.shift;
    for (entity, canvas) in ctx.query_mut_with_entity::<Canvas>() {
        let shapes = values.state.map_or(&[][..], |s| s.shapes(&canvas.shapes));
        // Shapes that shrank, or a canvas that grew, leave less to pan over.
        canvas.pan = canvas.clamp_pan(canvas.pan, shapes);
        if !canvas.visible || canvas.passthrough || !pointer.live(canvas.screen) {
            canvas.hovered = None;
            if held.as_ref().is_some_and(|h| h.entity == entity) {
                *held = None;
            }
            continue;
        }
        let (x, y) = pointer.at_element(canvas.screen, entity);
        canvas.hovered = canvas.tag_at(x, y, shapes);
        if let Some(h) = held.as_mut().filter(|h| h.entity == entity) {
            if !pointer.input.left_button_down {
                *held = None;
                continue;
            }
            // While held the cursor drags the shapes even off the canvas.
            let (cx, cy) = pointer.at(canvas.screen);
            let pan = [
                canvas.pan[0] - (cx - h.last.0),
                canvas.pan[1] - (cy - h.last.1),
            ];
            canvas.pan = canvas.clamp_pan(pan, shapes);
            h.last = (cx, cy);
            continue;
        }
        let inside = point_in(x, y, canvas.rect());
        if inside && canvas.pannable && pointer.input.scroll_delta != 0.0 && !out.wheel_taken {
            let step = pointer.input.scroll_delta * WHEEL_PAN;
            if let Some(pan) = wheel_pan(canvas, shapes, step, pointer.input.shift) {
                canvas.pan = pan;
                out.wheel_taken = true;
            }
        }
        if !inside || !pointer.click() {
            continue;
        }
        out.click_taken = true;
        match canvas.hovered {
            Some(tag) => {
                let event = if mods && !canvas.on_extend.is_empty() {
                    &canvas.on_extend
                } else {
                    &canvas.on_press
                };
                let value = Some(UiValue::Int(i32::try_from(tag).unwrap_or(i32::MAX)));
                emit(effects, event, entity, value);
            }
            None if canvas.pannable => {
                *held = Some(HeldCanvas {
                    entity,
                    last: (x, y),
                });
            }
            None => {}
        }
    }
}

// Where a wheel `step` pans `canvas`: down, or across with Shift, falling back
// to the other axis when that one has nowhere to go. `None` when the pan is
// already at the end of its room.
fn wheel_pan(canvas: &Canvas, shapes: &[UiShape], step: f32, across: bool) -> Option<[f32; 2]> {
    let room = canvas.max_pan(shapes);
    let primary = usize::from(!across);
    let axis = if room[primary] > 0.0 {
        primary
    } else {
        1 - primary
    };
    let mut pan = canvas.pan;
    pan[axis] -= step;
    let pan = canvas.clamp_pan(pan, shapes);
    ((pan[axis] - canvas.pan[axis]).abs() > f32::EPSILON).then_some(pan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn wide() -> (Canvas, Vec<UiShape>) {
        let canvas = Canvas {
            width: 100.0,
            height: 50.0,
            pannable: true,
            ..Default::default()
        };
        let shapes = vec![UiShape {
            rect: [0.0, 0.0, 300.0, 40.0],
            ..UiShape::default()
        }];
        (canvas, shapes)
    }

    // A row wider than the canvas scrolls sideways under the plain wheel, as
    // it has no height to scroll.
    #[test]
    fn the_wheel_pans_the_axis_with_room() {
        let (canvas, shapes) = wide();
        assert_eq!(wheel_pan(&canvas, &shapes, -30.0, false), Some([30.0, 0.0]));
        assert_eq!(wheel_pan(&canvas, &shapes, -30.0, true), Some([30.0, 0.0]));
    }

    // At the end of its room the wheel does not pan, so it is left to what
    // lies under the canvas.
    #[test]
    fn the_wheel_stops_at_the_end() {
        let (mut canvas, shapes) = wide();
        canvas.pan = [200.0, 0.0];
        assert_eq!(wheel_pan(&canvas, &shapes, -30.0, false), None);
        assert_eq!(wheel_pan(&canvas, &shapes, 30.0, false), Some([170.0, 0.0]));
    }
}
