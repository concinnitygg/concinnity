// Per-frame layout: every root Layout is solved and its elements moved into
// place. A label sizes by its measured text, which needs the loaded font
// metrics, so the pass runs in the overlay build rather than a system of its
// own.

use alloc::vec::Vec;

use crate::components::{
    Button, Checkbox, Identity, LabelBox, Layout, List, Sprite, Tabs, TextAlign, TextArea,
    TextField, TextLabel,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{Entity, PipelineContext, Ref};
use crate::gfx::overlay::UI_REFERENCE_SIZE;
use crate::presentation::ui::layout::{LayoutSolve, LayoutSource, label_present};
use crate::protocol::ScreenSpaces;
use crate::render::overlay::text;
use hashbrown::{HashMap, HashSet};

// Reused across frames so the pass allocates nothing in steady state.
#[derive(Debug, Default)]
pub(super) struct LayoutScratch {
    // Every layout held by another, so the rest are roots.
    nested: HashSet<AssetId>,
    // Each laid-out label's measured box this frame, `None` for one that
    // takes no space.
    boxes: HashMap<AssetId, Option<LabelBox>>,
    solve: LayoutSolve,
    // Each root's rect this frame.
    roots: Vec<(Entity, Option<[f32; 4]>)>,
}

// The world as the solver reads it: layouts by id, and each element's size.
struct WorldLayouts<'a, 'w> {
    ctx: &'a PipelineContext<'w>,
    boxes: &'a HashMap<AssetId, Option<LabelBox>>,
}

impl LayoutSource for WorldLayouts<'_, '_> {
    fn layout(&self, id: AssetId) -> Option<&Layout> {
        self.ctx.get_by_id::<Layout>(id)
    }

    fn element(&self, id: AssetId) -> Option<[f32; 2]> {
        if let Some(measured) = self.boxes.get(&id) {
            return measured.map(|b| [b.w, b.h]);
        }
        if let Some(s) = self.ctx.get_by_id::<Sprite>(id) {
            return s.visible.then_some([s.width, s.height]);
        }
        if let Some(t) = self.ctx.get_by_id::<TextField>(id) {
            return t.visible.then_some([t.width, t.height]);
        }
        if let Some(w) = self.ctx.get_by_id::<List>(id) {
            return w.visible.then_some([w.width, w.height]);
        }
        if let Some(w) = self.ctx.get_by_id::<Tabs>(id) {
            return w.visible.then_some([w.width, w.height]);
        }
        if let Some(w) = self.ctx.get_by_id::<Checkbox>(id) {
            return w.visible.then_some([w.width, w.height]);
        }
        if let Some(w) = self.ctx.get_by_id::<Button>(id) {
            return w.visible.then_some([w.width, w.height]);
        }
        let w = self.ctx.get_by_id::<TextArea>(id)?;
        w.visible.then_some([w.width, w.height])
    }
}

// Solve every root layout and write its elements' rects back: a sprite or a
// field takes its whole rect, a label the anchor its alignment reads, inset to
// its text origin.
pub(super) fn apply_layouts(
    ctx: &mut PipelineContext<'_>,
    fonts: &text::FontSet,
    viewport: [f32; 2],
    scratch: &mut LayoutScratch,
) {
    if ctx.query::<Layout>().next().is_none() {
        return;
    }
    scratch.nested.clear();
    scratch.boxes.clear();
    scratch.solve.clear();
    scratch.roots.clear();
    for layout in ctx.query::<Layout>() {
        for item in layout.children.iter().filter_map(|c| c.item) {
            let id = item.id();
            scratch.nested.insert(id);
            if let Some(label) = ctx.get_by_id::<TextLabel>(id) {
                let measured = label_present(label)
                    .then(|| text::measure_label_box(label, fonts))
                    .flatten();
                scratch.boxes.insert(id, measured);
            }
        }
    }
    let no_spaces = ScreenSpaces::default();
    let spaces = ctx.resource::<ScreenSpaces>().unwrap_or(&no_spaces);
    let src = WorldLayouts {
        ctx,
        boxes: &scratch.boxes,
    };
    for (entity, root) in ctx.query_with_entity::<Layout>() {
        let id = ctx.get::<Identity>(entity).map(|i| i.id());
        if id.is_some_and(|id| scratch.nested.contains(&id)) {
            continue;
        }
        let area = if spaces.on_canvas(root.screen.map(Ref::id)) {
            UI_REFERENCE_SIZE
        } else {
            viewport
        };
        let rect = scratch.solve.solve_root(&src, id, root, area);
        scratch.roots.push((entity, rect));
    }
    for &(entity, rect) in &scratch.roots {
        if let Some(root) = ctx.get_mut::<Layout>(entity)
            && root.rect != rect
        {
            root.rect = rect;
        }
    }
    for &(id, [x, y, w, h]) in &scratch.solve.placed {
        if let Some(b) = scratch.boxes.get(&id).copied().flatten() {
            if let Some(l) = ctx.get_mut_by_id::<TextLabel>(id) {
                l.x = label_anchor(l.align, x, w, b.pad);
                l.y = y + b.top_inset;
            }
        } else if let Some(s) = ctx.get_mut_by_id::<Sprite>(id) {
            (s.x, s.y, s.width, s.height) = (x, y, w, h);
        } else if let Some(t) = ctx.get_mut_by_id::<TextField>(id) {
            (t.x, t.y, t.width, t.height) = (x, y, w, h);
        } else if let Some(t) = ctx.get_mut_by_id::<List>(id) {
            (t.x, t.y, t.width, t.height) = (x, y, w, h);
        } else if let Some(t) = ctx.get_mut_by_id::<Tabs>(id) {
            (t.x, t.y, t.width, t.height) = (x, y, w, h);
        } else if let Some(t) = ctx.get_mut_by_id::<Checkbox>(id) {
            (t.x, t.y, t.width, t.height) = (x, y, w, h);
        } else if let Some(t) = ctx.get_mut_by_id::<Button>(id) {
            (t.x, t.y, t.width, t.height) = (x, y, w, h);
        } else if let Some(t) = ctx.get_mut_by_id::<TextArea>(id) {
            (t.x, t.y, t.width, t.height) = (x, y, w, h);
        }
    }
}

// The x a label of `align` anchors at to draw its text in a box at `x`, `w`
// wide, `pad` inside each edge.
fn label_anchor(align: TextAlign, x: f32, w: f32, pad: f32) -> f32 {
    match align {
        TextAlign::Left => x + pad,
        TextAlign::Center => x + w / 2.0,
        TextAlign::Right => x + w - pad,
    }
}

#[cfg(test)]
mod tests;
