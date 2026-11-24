// Transient layouts: a press outside a shown layout naming an `on_dismiss`
// event sends it; the press goes no further only when it lands outside every
// such layout, so one open over another still takes its own presses.

use alloc::vec::Vec;

use crate::components::Layout;
use crate::ecs::{PipelineContext, Ref};

use super::super::widgets::point_in;
use super::{Effect, Pointer, WidgetOutcome, emit};

pub(super) fn step_dismissals(
    ctx: &PipelineContext<'_>,
    pointer: &Pointer<'_>,
    effects: &mut Vec<Effect>,
    out: &mut WidgetOutcome,
) {
    let pressed = pointer.input.left_click || pointer.input.right_click;
    if !pressed || pointer.taken {
        return;
    }
    let mut any = false;
    let mut inside_one = false;
    for (entity, layout) in ctx.query_with_entity::<Layout>() {
        if layout.on_dismiss.is_empty() || !layout.visible {
            continue;
        }
        let Some(rect) = layout.rect else {
            continue;
        };
        if !pointer.live(layout.screen) || pointer.stack.shuts(entity, layout.screen.map(Ref::id)) {
            continue;
        }
        any = true;
        // Inside is where the press reaches the layout: not where a panel in
        // front of its own covers it, nor where a scrolling layout clips it.
        let (x, y) = pointer.at_element(layout.screen, entity);
        if point_in(x, y, rect) {
            inside_one = true;
        } else {
            emit(effects, &layout.on_dismiss, entity, None);
        }
    }
    if any && !inside_one {
        out.click_taken = true;
    }
}
