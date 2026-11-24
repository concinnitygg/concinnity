// Scrolling layouts: the wheel over one scrolls its children unless a widget
// under the pointer took it, the innermost of nested ones first, and a press on
// its scrollbar drags the thumb.

use crate::components::Layout;
use crate::ecs::{Entity, PipelineContext};

use super::super::widgets::point_in;
use super::super::widgets::scroll_view::ScrollView;
use super::{Pointer, WHEEL_PX, WidgetOutcome};

// The scroll a layout's children sit at, as a view over them.
fn view(layout: &Layout, rect: [f32; 4]) -> ScrollView {
    ScrollView {
        rect,
        content: layout.content_height,
        scroll: layout.scrolled,
    }
}

// The innermost scrolling layout under the pointer that has children to
// scroll: of the ones under it, the one with the smallest rect, which nested
// ones are inside.
fn wheel_target(ctx: &PipelineContext<'_>, pointer: &Pointer<'_>) -> Option<Entity> {
    let mut best: Option<(Entity, f32)> = None;
    for (entity, layout) in ctx.query_with_entity::<Layout>() {
        if !layout.scroll || !layout.visible || !pointer.live(layout.screen) {
            continue;
        }
        let Some(rect) = layout.rect else {
            continue;
        };
        let (x, y) = pointer.at_element(layout.screen, entity);
        let area = rect[2] * rect[3];
        if point_in(x, y, rect)
            && view(layout, rect).scrolls()
            && best.is_none_or(|(_, smallest)| area < smallest)
        {
            best = Some((entity, area));
        }
    }
    best.map(|(entity, _)| entity)
}

pub(super) fn step_scrolls(
    ctx: &mut PipelineContext<'_>,
    pointer: &Pointer<'_>,
    thumb: &mut Option<(Entity, f32)>,
    out: &mut WidgetOutcome,
) {
    let wheeled = (pointer.input.scroll_delta != 0.0 && !out.wheel_taken)
        .then(|| wheel_target(ctx, pointer))
        .flatten();
    for (entity, layout) in ctx.query_mut_with_entity::<Layout>() {
        if !layout.scroll || !layout.visible || !pointer.live(layout.screen) {
            continue;
        }
        let Some(rect) = layout.rect else {
            continue;
        };
        let view = view(layout, rect);
        let (x, y) = pointer.at_element(layout.screen, entity);
        if let Some((grabbed, grab)) = *thumb
            && grabbed == entity
        {
            if pointer.input.left_button_down {
                layout.scrolled = view.scroll_for_thumb(y - grab);
                out.click_taken = true;
                continue;
            }
            *thumb = None;
        }
        if wheeled == Some(entity) {
            layout.scrolled = view.wheeled(pointer.input.scroll_delta * WHEEL_PX);
            out.wheel_taken = true;
        }
        if let Some((track, bar)) = view.scrollbar()
            && pointer.click()
            && !out.click_taken
            && point_in(x, y, track)
        {
            // A press on the thumb grabs it where it was pressed; one on the
            // track beside it centers the thumb there first.
            let grab = if point_in(x, y, bar) {
                y - bar[1]
            } else {
                bar[3] * 0.5
            };
            layout.scrolled = view.scroll_for_thumb(y - grab);
            *thumb = Some((entity, grab));
            out.click_taken = true;
        }
    }
}
