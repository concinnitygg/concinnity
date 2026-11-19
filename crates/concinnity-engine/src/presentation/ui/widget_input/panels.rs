// Floating panels: a press on a movable root layout's top band drags it, one
// on a resizable root's right or bottom edge sizes it, until the button lets
// go. A press starts a drag only when nothing else under it took it.

use crate::components::{FrameInput, Layout, ResizeEdges};
use crate::ecs::{Entity, PipelineContext};
use crate::gfx::overlay::UI_REFERENCE_SIZE;

use super::super::UiInputSystem;
use super::super::screen::ScreenRegistry;
use super::super::widgets::point_in;
use super::{Pointer, WidgetOutcome};

// How much of a panel stays on its screen, so it can always be dragged back.
const KEEP: f32 = 40.0;

// What a press on a panel grabbed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Grip {
    // The panel, to move it.
    Move,
    // Its right edge, its bottom edge, or both.
    Resize(ResizeEdges),
}

// A panel drag in progress: what was grabbed, and the cursor, offset, and rect
// it started from.
#[derive(Debug, Clone, Copy)]
pub(in crate::presentation::ui) struct PanelDrag {
    entity: Entity,
    grip: Grip,
    from: [f32; 2],
    offset: [f32; 2],
    rect: [f32; 4],
}

// What a press at `(x, y)` grabs of a root laid out at its rect, if anything.
pub(super) fn grip_at(root: &Layout, x: f32, y: f32) -> Option<Grip> {
    let rect = root.rect?;
    if !point_in(x, y, rect) {
        return None;
    }
    if let Some(edges) = root.resize_grip(x, y) {
        return Some(Grip::Resize(edges));
    }
    (root.movable && y < rect[1] + root.grab).then_some(Grip::Move)
}

// The offset and size a drag that has moved the cursor by `moved` gives a root
// on a screen of size `area`. A move keeps the top band on the screen; a
// resize keeps the edges it did not grab where they were, whatever the root's
// anchor.
fn dragged(d: &PanelDrag, root: &Layout, moved: [f32; 2], area: [f32; 2]) -> ([f32; 2], [f32; 2]) {
    let [x, y, w, h] = d.rect;
    match d.grip {
        Grip::Move => {
            let nx = (x + moved[0]).clamp(KEEP - w, area[0] - KEEP);
            let ny = (y + moved[1]).clamp(0.0, (area[1] - root.grab).max(0.0));
            ([d.offset[0] + nx - x, d.offset[1] + ny - y], [w, h])
        }
        Grip::Resize(ResizeEdges { right, bottom }) => {
            let grow = |on: bool, axis: usize, len: f32| {
                if on {
                    (len + moved[axis]).max(root.min_size[axis])
                } else {
                    len
                }
            };
            let size = [grow(right, 0, w), grow(bottom, 1, h)];
            // An anchored root sits `fraction` of its size back from its
            // anchor, so growing it moves its near edge; the offset undoes that.
            let fr = root.anchor.fractions();
            let offset = [
                d.offset[0] + fr[0] * (size[0] - w),
                d.offset[1] + fr[1] * (size[1] - h),
            ];
            (offset, size)
        }
    }
}

// The size of the space a root of `screen` lays out in.
fn area(screens: &ScreenRegistry, root: &Layout, input: &FrameInput) -> [f32; 2] {
    if screens.on_canvas(root.screen.map(crate::ecs::Ref::id)) {
        UI_REFERENCE_SIZE
    } else {
        input.viewport
    }
}

// Carry a panel drag on while the button is held, and end it on release. A
// drag in progress takes the frame's pointer.
pub(super) fn step_panel_drag(
    ctx: &mut PipelineContext<'_>,
    pointer: &Pointer<'_>,
    drag: &mut Option<PanelDrag>,
    out: &mut WidgetOutcome,
) {
    let Some(d) = *drag else {
        return;
    };
    let Some(root) = ctx.get_mut::<Layout>(d.entity) else {
        *drag = None;
        return;
    };
    let (x, y) = pointer.at(root.screen);
    let moved = [x - d.from[0], y - d.from[1]];
    let space = area(pointer.screens, root, pointer.input);
    let (offset, size) = dragged(&d, root, moved, space);
    root.offset = offset;
    if let Grip::Resize(edges) = d.grip {
        root.size = Some(size);
        root.resizing = Some(edges);
    }
    if !pointer.input.left_button_down {
        root.resizing = None;
        *drag = None;
    }
    out.click_taken = true;
}

impl UiInputSystem {
    // Start a panel drag on a press nothing else took: the topmost panel, the
    // last laid out, under the press.
    pub(in crate::presentation::ui) fn start_panel_drag(
        &mut self,
        input: &FrameInput,
        overlay: &crate::gfx::overlay::OverlayTransform,
        ctx: &PipelineContext<'_>,
    ) {
        if !input.left_click || self.widgets.panel_drag.is_some() {
            return;
        }
        let mut grabbed = None;
        for (entity, root) in ctx.query_with_entity::<Layout>() {
            let Some(rect) = root.rect else {
                continue;
            };
            let screen = root.screen.map(crate::ecs::Ref::id);
            if !(root.movable || root.resizable)
                || !root.visible
                || !self.screens.receives_input(screen)
                || self.widgets.blocked(screen)
            {
                continue;
            }
            let (x, y) = self
                .screens
                .cursor(screen, overlay, input.mouse_x, input.mouse_y);
            let at = |s| {
                self.screens
                    .cursor(s, overlay, input.mouse_x, input.mouse_y)
            };
            if self.widgets.stack.root_over(screen, at) != Some(entity) {
                continue;
            }
            if let Some(grip) = grip_at(root, x, y) {
                grabbed = Some(PanelDrag {
                    entity,
                    grip,
                    from: [x, y],
                    offset: root.offset,
                    rect,
                });
            }
        }
        self.widgets.panel_drag = grabbed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::LayoutAnchor;

    fn panel(movable: bool, resizable: bool) -> Layout {
        Layout {
            movable,
            resizable,
            rect: Some(RECT),
            ..Default::default()
        }
    }

    const RECT: [f32; 4] = [100.0, 100.0, 300.0, 200.0];
    const AREA: [f32; 2] = [1280.0, 720.0];

    fn drag(grip: Grip) -> PanelDrag {
        PanelDrag {
            entity: Entity::dangling(),
            grip,
            from: [0.0, 0.0],
            offset: [0.0, 0.0],
            rect: RECT,
        }
    }

    #[test]
    fn the_top_band_moves_and_the_edges_resize() {
        let both = panel(true, true);
        assert_eq!(grip_at(&both, 150.0, 110.0), Some(Grip::Move));
        assert_eq!(grip_at(&both, 150.0, 200.0), None, "below the band");
        assert_eq!(
            grip_at(&both, 398.0, 200.0),
            Some(Grip::Resize(ResizeEdges {
                right: true,
                bottom: false
            }))
        );
        assert_eq!(
            grip_at(&both, 398.0, 298.0),
            Some(Grip::Resize(ResizeEdges {
                right: true,
                bottom: true
            }))
        );
        assert_eq!(grip_at(&both, 50.0, 110.0), None, "outside");
        assert_eq!(grip_at(&panel(false, false), 150.0, 110.0), None);
        assert_eq!(grip_at(&panel(true, false), 398.0, 298.0), None);
    }

    #[test]
    fn a_resize_keeps_the_edges_it_did_not_grab_whatever_the_anchor() {
        let corner = Grip::Resize(ResizeEdges {
            right: true,
            bottom: true,
        });
        for (anchor, offset) in [
            (LayoutAnchor::TopLeft, [0.0, 0.0]),
            (LayoutAnchor::Center, [25.0, 10.0]),
            (LayoutAnchor::BottomRight, [50.0, 20.0]),
        ] {
            let root = Layout {
                anchor,
                ..panel(true, true)
            };
            let got = dragged(&drag(corner), &root, [50.0, 20.0], AREA);
            assert_eq!(got, (offset, [350.0, 220.0]), "{anchor:?}");
        }
    }

    #[test]
    fn a_moved_panel_keeps_its_band_on_the_screen() {
        let root = panel(true, false);
        let (offset, _) = dragged(&drag(Grip::Move), &root, [-1000.0, -1000.0], AREA);
        assert_eq!(offset, [KEEP - 300.0 - 100.0, -100.0]);
        let (offset, _) = dragged(&drag(Grip::Move), &root, [5000.0, 5000.0], AREA);
        assert_eq!(
            offset,
            [AREA[0] - KEEP - 100.0, AREA[1] - root.grab - 100.0]
        );
    }
}
