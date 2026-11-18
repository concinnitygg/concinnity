// The floating panels of a frame, in the order they stack: which panel each
// element belongs to, which panel is on top at a point, and the draw rank that
// lifts a panel's elements over the panels beneath it. Also which elements a
// hidden layout hides.
//
// A panel covers its own screen and every screen layered beneath it, or on its
// layer (screens sharing a layer cover each other), each tested in that
// screen's own cursor space, so a press on a panel never reaches an element
// under it on a lower screen.

use alloc::vec::Vec;

use crate::components::{Identity, Layout};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{Entity, PipelineContext, Ref};
use crate::presentation::ui::widgets::point_in;
use crate::protocol::ScreenStack;
use hashbrown::{HashMap, HashSet};

// One floating panel: a movable or resizable root layout as last laid out.
#[derive(Debug, Clone, Copy)]
struct Stacked {
    root: Entity,
    screen: Option<AssetId>,
    // Its screen's draw layer.
    layer: i32,
    rect: [f32; 4],
    rank: u32,
}

// Reused across frames, so a steady frame allocates nothing.
#[derive(Debug, Default)]
pub(crate) struct PanelStack {
    // Back to front.
    panels: Vec<Stacked>,
    // The panel (an index into `panels`) each element and nested layout of a
    // panel belongs to.
    owners: HashMap<Entity, usize>,
    // Every element and layout inside a hidden layout.
    hidden: HashSet<Entity>,
    // The rect each element inside a scrolling layout is clipped to.
    clips: HashMap<Entity, [f32; 4]>,
    // Every layout another layout holds, so the rest are roots.
    nested: HashSet<AssetId>,
    // Each layered screen's draw layer, read from the screen stack.
    layers: Vec<(AssetId, i32)>,
    // Scratch for walking a layout tree.
    walk: Vec<AssetId>,
}

impl PanelStack {
    // Gather this frame's panels and the elements each holds.
    pub(crate) fn rebuild(&mut self, ctx: &PipelineContext<'_>) {
        self.panels.clear();
        self.owners.clear();
        self.hidden.clear();
        self.clips.clear();
        self.nested.clear();
        self.layers.clear();
        let stack = ctx.resource::<ScreenStack>();
        if let Some(stack) = stack {
            self.layers
                .extend(stack.layers.iter().map(|(&id, &layer)| (id, layer)));
        }
        for layout in ctx.query::<Layout>() {
            self.nested
                .extend(layout.children.iter().filter_map(|c| c.item).map(Ref::id));
        }
        for layout in ctx.query::<Layout>().filter(|l| !l.visible) {
            self.walk_tree(ctx, layout, |stack, entity| stack.hidden.insert(entity));
        }
        for layout in ctx.query::<Layout>().filter(|l| l.scroll && l.visible) {
            let Some(rect) = layout.rect else {
                continue;
            };
            self.walk_tree(ctx, layout, |stack, entity| {
                let clip = match stack.clips.get(&entity) {
                    Some(&outer) => intersect(outer, rect),
                    None => rect,
                };
                stack.clips.insert(entity, clip);
                true
            });
        }
        for (root, layout) in ctx.query_with_entity::<Layout>() {
            if !(layout.movable || layout.resizable) || !layout.visible {
                continue;
            }
            let id = ctx.get::<Identity>(root).map(|i| i.id());
            if id.is_some_and(|id| self.nested.contains(&id)) {
                continue;
            }
            let screen = layout.screen.map(Ref::id);
            let open = screen.is_none_or(|s| stack.is_none_or(|st| st.open.contains(&s)));
            let Some(rect) = layout.rect.filter(|_| open) else {
                continue;
            };
            self.panels.push(Stacked {
                root,
                screen,
                layer: self.layer_of(screen),
                rect,
                rank: layout.rank,
            });
        }
        // Higher screens in front; on a screen, higher ranks, and equal ranks
        // keep the order the panels were added in.
        self.panels.sort_by_key(|p| (p.layer, p.rank));
        for i in 0..self.panels.len() {
            let root = self.panels[i].root;
            self.owners.insert(root, i);
            if let Some(layout) = ctx.get::<Layout>(root) {
                self.walk_tree(ctx, layout, |stack, entity| {
                    stack.owners.insert(entity, i);
                    true
                });
            }
        }
    }

    // Visit every element and layout under `layout`; `visit` returns whether
    // to go on into what an element holds.
    fn walk_tree(
        &mut self,
        ctx: &PipelineContext<'_>,
        layout: &Layout,
        mut visit: impl FnMut(&mut Self, Entity) -> bool,
    ) {
        self.walk.clear();
        self.walk
            .extend(layout.children.iter().filter_map(|c| c.item).map(Ref::id));
        while let Some(id) = self.walk.pop() {
            let Some(entity) = ctx.entity_of(id) else {
                continue;
            };
            if !visit(self, entity) {
                continue;
            }
            if let Some(nested) = ctx.get::<Layout>(entity) {
                self.walk
                    .extend(nested.children.iter().filter_map(|c| c.item).map(Ref::id));
            }
        }
    }

    // The draw layer of `screen`: 0 for a screen-less element or an
    // unlayered screen.
    fn layer_of(&self, screen: Option<AssetId>) -> i32 {
        screen
            .and_then(|s| self.layers.iter().find(|(id, _)| *id == s))
            .map_or(0, |&(_, layer)| layer)
    }

    // The rect a scrolling layout clips `entity` to, if one holds it.
    pub(crate) fn clip_of(&self, entity: Entity) -> Option<[f32; 4]> {
        self.clips.get(&entity).copied()
    }

    // Whether a hidden layout holds `entity`, so it neither draws nor takes
    // input.
    pub(crate) fn hides(&self, entity: Entity) -> bool {
        self.hidden.contains(&entity)
    }

    // The draw rank of `entity`'s panel: 0 outside every panel, else its place
    // in the stack from 1 at the back.
    pub(crate) fn rank_of(&self, entity: Entity) -> u32 {
        self.owners
            .get(&entity)
            .map_or(0, |&i| u32::try_from(i + 1).unwrap_or(u32::MAX))
    }

    // The frontmost panel covering an element of `screen` where the pointer
    // is: one on that screen or a screen above it, the pointer in each
    // panel's own screen space by `at`.
    fn top_over(
        &self,
        screen: Option<AssetId>,
        at: impl Fn(Option<AssetId>) -> (f32, f32),
    ) -> Option<usize> {
        let layer = self.layer_of(screen);
        self.panels.iter().rposition(|p| {
            let (x, y) = at(p.screen);
            p.layer >= layer && point_in(x, y, p.rect)
        })
    }

    // The root of the frontmost panel covering an element of `screen` at the
    // pointer.
    pub(crate) fn root_over(
        &self,
        screen: Option<AssetId>,
        at: impl Fn(Option<AssetId>) -> (f32, f32),
    ) -> Option<Entity> {
        self.top_over(screen, at).map(|i| self.panels[i].root)
    }

    // Whether the pointer reaches `entity` of `screen`: no panel covers it
    // there, or the frontmost one that does holds it.
    pub(crate) fn reaches(
        &self,
        screen: Option<AssetId>,
        entity: Entity,
        at: impl Fn(Option<AssetId>) -> (f32, f32),
    ) -> bool {
        self.reaches_owned(screen, Some(entity), at)
    }

    // `reaches` for an element known by the entity it moves with, if any; one
    // with none is reached only where no panel covers it.
    pub(crate) fn reaches_owned(
        &self,
        screen: Option<AssetId>,
        entity: Option<Entity>,
        at: impl Fn(Option<AssetId>) -> (f32, f32),
    ) -> bool {
        if entity.is_some_and(|e| self.hides(e)) {
            return false;
        }
        self.top_over(screen, at)
            .is_none_or(|top| entity.is_some_and(|e| self.owners.get(&e) == Some(&top)))
    }

    // The frontmost panel under the pointer on any screen. `at` gives the
    // pointer in a screen's space, or `None` for a screen that takes no input
    // now.
    pub(crate) fn front_under(
        &self,
        at: impl Fn(Option<AssetId>) -> Option<(f32, f32)>,
    ) -> Option<Entity> {
        self.panels
            .iter()
            .rev()
            .find(|p| at(p.screen).is_some_and(|(x, y)| point_in(x, y, p.rect)))
            .map(|p| p.root)
    }

    // The rank that brings `root` in front of every other panel on its
    // screen's layer, or `None` when it is in front already.
    pub(crate) fn raised_rank(&self, root: Entity) -> Option<u32> {
        let layer = self.panels.iter().find(|p| p.root == root)?.layer;
        let front = self.panels.iter().rev().find(|p| p.layer == layer)?;
        (front.root != root).then(|| front.rank.saturating_add(1))
    }
}

// The overlap of two `[x, y, width, height]` rects, empty when they miss.
pub(crate) fn intersect(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let x = a[0].max(b[0]);
    let y = a[1].max(b[1]);
    let right = (a[0] + a[2]).min(b[0] + b[2]);
    let bottom = (a[1] + a[3]).min(b[1] + b[3]);
    [x, y, (right - x).max(0.0), (bottom - y).max(0.0)]
}

#[cfg(test)]
mod tests;
