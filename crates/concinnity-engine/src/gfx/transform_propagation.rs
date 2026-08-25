// src/gfx/transform_propagation.rs
//
// Resolving each entity's world matrix from its Transform and its Parent chain,
// and writing the result back to its GlobalTransform. GraphicsSystem runs this
// every frame before it builds a draw list.
//
// The cache holds a slot table rather than entity-keyed maps: every entity that
// owns a Transform gets a dense slot, and parents, depths, children, locals and
// world matrices are parallel arrays indexed by it. A full resolve orders the
// slots shallowest-first (a counting sort on depth) so one pass composes the
// whole hierarchy.
//
// Between full resolves the pass is incremental. The Transform column stamps a
// change tick per row, so the entities written since the last resolve are
// recoverable without scanning their values; each one re-walks only its own
// subtree. A whole-column write, a row added or removed, any Parent edit, or a
// dirty set large enough that the subtree walks would overlap all fall back to
// the full resolve.

use std::collections::HashMap;

use crate::components::{GlobalTransform, Parent, Transform};
use crate::ecs::{ColumnTicks, Entity, MAX_CHANGE_AGE, PipelineContext, Tick};
use crate::gfx::draw_list::mat4_mul;

pub(crate) type WorldMatrix = [[f32; 4]; 4];

// Slot sentinel for "no parent" and for an entity the slot table does not hold.
const NO_SLOT: u32 = u32::MAX;

// Depth sentinels. CYCLIC marks an entity whose ancestor chain loops, or that
// descends from one: it resolves to its own local matrix. VISITING marks a slot
// the depth walk is currently descending through, which is how it detects the
// loop; UNVISITED is the initial state.
const CYCLIC: u32 = u32::MAX;
const VISITING: u32 = u32::MAX - 1;
const UNVISITED: u32 = u32::MAX - 2;

// A moved entity re-walks its own subtree, so past a column-size fraction of
// this divisor the walks overlap enough that one ordered full resolve is the
// cheaper pass. A single moved entity always stays incremental.
const DIRTY_BUDGET_DIVISOR: usize = 8;

// The source-column tick stamps observed at the last resolve. `transform`
// carries all four so the next frame can tell a targeted write (which the
// per-row stamps describe) from a whole-column write or a row add/remove
// (which they do not).
#[derive(Clone, Copy)]
struct SourceTicks {
    transform: ColumnTicks,
    parent: Tick,
}

// Reused scratch plus change-tracking for the per-frame transform propagation.
// GraphicsSystem owns one and passes it to `propagate_transforms_cached` each
// frame: every buffer is refilled in place (no per-frame allocation once they
// reach steady-state capacity), propagation is skipped entirely on frames where
// neither the Transform nor the Parent column changed since the last recompute,
// and on frames where only some Transforms were written only those entities'
// subtrees are recomposed.
#[derive(Default)]
pub(crate) struct TransformCache {
    // Slot tables, indexed by slot and all the same length: one slot per entity
    // owning a Transform, in Transform-column order.
    entity: Vec<Entity>,
    local: Vec<WorldMatrix>,
    world: Vec<WorldMatrix>,
    parent: Vec<u32>,
    depth: Vec<u32>,
    // Slots shallowest-first, so one forward pass composes every chain.
    order: Vec<u32>,
    // Children in compressed-row form: `child_list[child_start[s]..child_start[s + 1]]`
    // are slot `s`'s children. `child_start` has one entry per slot plus a tail.
    child_start: Vec<u32>,
    child_list: Vec<u32>,
    // Entity index -> slot, NO_SLOT where the entity owns no Transform.
    slot_of: Vec<u32>,
    // Last pass that recomputed each slot, against `pass`, so a subtree already
    // covered by a shallower dirty ancestor is not walked twice.
    visited: Vec<u32>,
    pass: u32,
    // Per-pass scratch: the depth walk's ancestor path, the dirty slots, the
    // subtree walk's stack, and a u32 run shared by the counting sort and the
    // child-index fill (each clears it before use).
    path: Vec<u32>,
    dirty: Vec<u32>,
    stack: Vec<u32>,
    offsets: Vec<u32>,
    last: Option<SourceTicks>,
}

impl TransformCache {
    // The slot holding `entity`'s transform, or None when it owns none. The
    // generation check rejects a handle whose index was recycled.
    fn slot_of(&self, entity: Entity) -> Option<u32> {
        let slot = *self.slot_of.get(entity.index() as usize)?;
        if slot == NO_SLOT {
            return None;
        }
        (self.entity[slot as usize] == entity).then_some(slot)
    }

    // Rebuild the whole slot table from the live Transform + Parent columns:
    // one slot per Transform, each slot's parent slot, its depth, its children,
    // and the shallowest-first order that composes them.
    fn resolve(&mut self, ctx: &PipelineContext) {
        self.entity.clear();
        self.local.clear();
        for (entity, transform) in ctx.query_with_entity::<Transform>() {
            self.entity.push(entity);
            self.local.push(transform.model_matrix());
        }
        let slots = self.entity.len();

        let widest = self
            .entity
            .iter()
            .map(|e| e.index() as usize)
            .max()
            .map_or(0, |i| i + 1);
        self.slot_of.clear();
        self.slot_of.resize(widest, NO_SLOT);
        for (slot, entity) in self.entity.iter().enumerate() {
            self.slot_of[entity.index() as usize] = slot as u32;
        }

        // A Parent naming an entity with no Transform composes against nothing,
        // so its child resolves as a root.
        self.parent.clear();
        self.parent.resize(slots, NO_SLOT);
        for (entity, parent) in ctx.query_with_entity::<Parent>() {
            if let Some(slot) = self.slot_of(entity) {
                self.parent[slot as usize] = self.slot_of(parent.0).unwrap_or(NO_SLOT);
            }
        }

        self.compute_depths();
        self.order_by_depth();
        self.build_children();

        self.world.clear();
        self.world.resize(slots, crate::gfx::draw_list::IDENTITY4);
        for i in 0..self.order.len() {
            let slot = self.order[i] as usize;
            self.world[slot] = self.compose(slot);
        }

        self.visited.clear();
        self.visited.resize(slots, 0);
        self.pass = 0;
    }

    // One slot's world matrix from its parent's, which the shallowest-first
    // order guarantees is already resolved. A root, and anything caught in (or
    // hanging off) a parent cycle, keeps its local matrix.
    fn compose(&self, slot: usize) -> WorldMatrix {
        match (self.depth[slot], self.parent[slot]) {
            (CYCLIC, _) | (_, NO_SLOT) => self.local[slot],
            (_, parent) => mat4_mul(self.world[parent as usize], self.local[slot]),
        }
    }

    // Depth of every slot: 0 for a root, parent depth + 1 otherwise, CYCLIC for
    // a slot whose ancestor chain loops. Iterative over an explicit ancestor
    // path, so a deep hierarchy cannot overflow the stack, and memoized, so
    // each slot is resolved once no matter how many descendants ask for it.
    fn compute_depths(&mut self) {
        self.depth.clear();
        self.depth.resize(self.local.len(), UNVISITED);
        for start in 0..self.depth.len() {
            if self.depth[start] != UNVISITED {
                continue;
            }
            self.path.clear();
            let mut current = start;
            // The depth to assign the last entry pushed onto the path.
            let base = loop {
                match self.depth[current] {
                    UNVISITED => {}
                    // Already on this path, or known to hang off a loop.
                    VISITING | CYCLIC => break CYCLIC,
                    known => break known + 1,
                }
                self.depth[current] = VISITING;
                self.path.push(current as u32);
                match self.parent[current] {
                    NO_SLOT => break 0,
                    parent => current = parent as usize,
                }
            };
            let mut depth = base;
            for &slot in self.path.iter().rev() {
                self.depth[slot as usize] = depth;
                if depth != CYCLIC {
                    depth += 1;
                }
            }
        }
    }

    // Order the slots shallowest-first by counting sort, with the cyclic slots
    // in a bucket of their own past the deepest real level.
    fn order_by_depth(&mut self) {
        let deepest = self
            .depth
            .iter()
            .copied()
            .filter(|&d| d != CYCLIC)
            .max()
            .unwrap_or(0) as usize;
        let cyclic_bucket = deepest + 1;
        self.offsets.clear();
        self.offsets.resize(cyclic_bucket + 1, 0);
        for &depth in &self.depth {
            let bucket = if depth == CYCLIC {
                cyclic_bucket
            } else {
                depth as usize
            };
            self.offsets[bucket] += 1;
        }
        let mut offset = 0;
        for count in &mut self.offsets {
            let bucket = *count;
            *count = offset;
            offset += bucket;
        }
        self.order.clear();
        self.order.resize(self.depth.len(), 0);
        for slot in 0..self.depth.len() {
            let bucket = if self.depth[slot] == CYCLIC {
                cyclic_bucket
            } else {
                self.depth[slot] as usize
            };
            self.order[self.offsets[bucket] as usize] = slot as u32;
            self.offsets[bucket] += 1;
        }
    }

    // Invert `parent` into the compressed-row child index the subtree walk
    // descends. Counts per parent, prefix-sums them into row starts, then
    // places each slot under its parent.
    fn build_children(&mut self) {
        let slots = self.parent.len();
        self.child_start.clear();
        self.child_start.resize(slots + 1, 0);
        for &parent in &self.parent {
            if parent != NO_SLOT {
                self.child_start[parent as usize + 1] += 1;
            }
        }
        for i in 0..slots {
            self.child_start[i + 1] += self.child_start[i];
        }
        self.child_list.clear();
        self.child_list.resize(self.child_start[slots] as usize, 0);
        // Walk the starts forward as rows fill, then restore them.
        self.offsets.clear();
        self.offsets.extend_from_slice(&self.child_start[..slots]);
        for slot in 0..slots {
            let parent = self.parent[slot];
            if parent == NO_SLOT {
                continue;
            }
            let at = self.offsets[parent as usize] as usize;
            self.offsets[parent as usize] += 1;
            self.child_list[at] = slot as u32;
        }
    }

    // Write every slot's world matrix into its GlobalTransform. An entity that
    // owns a Transform but no GlobalTransform is simply skipped.
    fn write_all(&self, ctx: &mut PipelineContext) {
        for (slot, &entity) in self.entity.iter().enumerate() {
            if let Some(global) = ctx.get_mut::<GlobalTransform>(entity) {
                global.0 = self.world[slot];
            }
        }
    }

    // Recompose only the subtrees under the Transforms written since `since`.
    // Returns false when the per-row stamps cannot carry the pass -- an unknown
    // entity, or a dirty set large enough that a full resolve is cheaper -- and
    // the caller falls back.
    fn resolve_incremental(&mut self, ctx: &mut PipelineContext, since: Tick) -> bool {
        let budget = (self.entity.len() / DIRTY_BUDGET_DIVISOR).max(1);
        self.dirty.clear();
        for (entity, transform) in ctx.changed_rows::<Transform>(since) {
            if self.dirty.len() >= budget {
                return false;
            }
            let Some(slot) = self.slot_of(entity) else {
                return false;
            };
            self.local[slot as usize] = transform.model_matrix();
            self.dirty.push(slot);
        }

        // Shallowest first, so an ancestor's walk subsumes any dirty descendant
        // rather than recomputing it against a stale parent and again after.
        self.dirty
            .sort_unstable_by_key(|&slot| self.depth[slot as usize]);
        self.pass = self.pass.wrapping_add(1);
        if self.pass == 0 {
            self.visited.fill(0);
            self.pass = 1;
        }
        for i in 0..self.dirty.len() {
            self.walk_subtree(ctx, self.dirty[i]);
        }
        true
    }

    // Recompose `seed` and everything under it, writing each GlobalTransform as
    // it goes. Slots already recomposed this pass are skipped, so seeds that
    // share an ancestor cost one walk between them.
    fn walk_subtree(&mut self, ctx: &mut PipelineContext, seed: u32) {
        self.stack.clear();
        self.stack.push(seed);
        while let Some(slot) = self.stack.pop() {
            let slot = slot as usize;
            if self.visited[slot] == self.pass {
                continue;
            }
            self.visited[slot] = self.pass;
            self.world[slot] = self.compose(slot);
            if let Some(global) = ctx.get_mut::<GlobalTransform>(self.entity[slot]) {
                global.0 = self.world[slot];
            }
            let (start, end) = (
                self.child_start[slot] as usize,
                self.child_start[slot + 1] as usize,
            );
            self.stack.extend_from_slice(&self.child_list[start..end]);
        }
    }

    // The resolved matrices keyed by entity, for the one-shot callers that want
    // a lookup table rather than the slot arrays.
    fn world_map(&self) -> HashMap<Entity, WorldMatrix> {
        self.entity
            .iter()
            .copied()
            .zip(self.world.iter().copied())
            .collect()
    }
}

// Resolve each entity's world matrix from its Transform and Parent chain.
// Returns an entity -> world matrix map, built through a throwaway
// `TransformCache`; the per-frame path owns a cache and calls
// `propagate_transforms_cached` instead so the buffers survive across frames.
pub(crate) fn resolve_world_matrices(ctx: &PipelineContext) -> HashMap<Entity, WorldMatrix> {
    let mut cache = TransformCache::default();
    cache.resolve(ctx);
    cache.world_map()
}

// Resolve and write every entity's GlobalTransform in one shot, with no cache
// carried across calls. The reparent recompose and the load-time pass use this;
// the per-frame path uses `propagate_transforms_cached`.
pub(crate) fn propagate_transforms(ctx: &mut PipelineContext) {
    let mut cache = TransformCache::default();
    cache.resolve(ctx);
    cache.write_all(ctx);
}

// Per-frame transform propagation against the reused scratch in `cache`.
// Writes each entity's GlobalTransform from its Transform + Parent chain,
// exactly as `propagate_transforms`, but does the least work the change ticks
// allow: nothing at all when neither source column moved, a walk of just the
// moved entities' subtrees when only targeted Transform writes landed, and a
// full ordered resolve otherwise.
pub(crate) fn propagate_transforms_cached(ctx: &mut PipelineContext, cache: &mut TransformCache) {
    let transform = ctx.column_ticks::<Transform>();
    let parent = ctx.changed_tick::<Parent>();

    if let Some(last) = cache.last {
        if last.transform.changed == transform.changed && last.parent == parent {
            return;
        }
        // Per-row stamps describe the change only while no whole-column write,
        // row add/remove, or Parent edit has happened since the last resolve,
        // and only while the two ticks are still within the wrap-relative
        // window a row comparison is valid over.
        let targeted_only = last.parent == parent
            && last.transform.bulk == transform.bulk
            && last.transform.structural == transform.structural
            && transform
                .changed
                .get()
                .wrapping_sub(last.transform.changed.get())
                <= MAX_CHANGE_AGE;
        if targeted_only && cache.resolve_incremental(ctx, last.transform.changed) {
            cache.last = Some(SourceTicks { transform, parent });
            return;
        }
    }

    cache.resolve(ctx);
    cache.write_all(ctx);
    cache.last = Some(SourceTicks { transform, parent });
}

// Re-parent an entity at runtime: detach it from its current parent (if any),
// attach it under `new_parent` (or leave it a root when `None`), keep both
// parents' Children lists in sync, and recompose world matrices so the new
// chain shows up immediately. Entity-keyed throughout, so it is invariant to
// component-column order. Driven by ReparentRequest events the GraphicsSystem
// drains each step.
pub(crate) fn reparent(ctx: &mut PipelineContext, child: Entity, new_parent: Option<Entity>) {
    use crate::components::Children;

    // Drop the old parent edge and unlist the child from that parent.
    if let Some(old) = ctx.remove::<Parent>(child)
        && let Some(siblings) = ctx.get_mut::<Children>(old.0)
    {
        siblings.0.retain(|&e| e != child);
    }

    // Attach under the new parent (None leaves it a root). The Parent column is
    // free of `child` here (just removed), so the insert never duplicates.
    if let Some(parent) = new_parent {
        ctx.insert(child, Parent(parent));
        match ctx.get_mut::<Children>(parent) {
            Some(kids) => {
                if !kids.0.contains(&child) {
                    kids.0.push(child);
                }
            }
            None => ctx.insert(parent, Children(concinnity_memory::InlineVec::one(child))),
        }
    }

    propagate_transforms(ctx);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::BlobData;
    use crate::components::Children;
    use crate::ecs::{Arena, ComponentStorage, FrameContext, Resources};
    use crate::gfx::draw_list::IDENTITY4;
    use crate::gfx::profile::FrameProfile;

    // The pieces a PipelineContext borrows, owned so a test can build one
    // context and hold it for the whole body.
    struct TestWorld {
        components: ComponentStorage,
        blob: BlobData,
        profile: FrameProfile,
        resources: Resources,
        scratch: Arena,
    }

    impl TestWorld {
        fn new() -> TestWorld {
            TestWorld {
                components: ComponentStorage::default(),
                blob: BlobData::empty(),
                profile: FrameProfile::default(),
                resources: Resources::new(),
                scratch: Arena::with_capacity(64 * 1024),
            }
        }

        fn ctx(&mut self) -> PipelineContext<'_> {
            PipelineContext {
                components: &mut self.components,
                blob: &mut self.blob,
                profile: &mut self.profile,
                resources: &mut self.resources,
                frame: FrameContext::new(&self.scratch),
            }
        }
    }

    fn translate(x: f32) -> Transform {
        Transform {
            position: [x, 0.0, 0.0],
            rotation_deg: [0.0; 3],
            scale: [1.0; 3],
        }
    }

    // An entity carrying the Transform propagation reads and the
    // GlobalTransform it writes, optionally parented.
    fn spawn(ctx: &mut PipelineContext, t: Transform, parent: Option<Entity>) -> Entity {
        let entity = ctx.components.spawn();
        ctx.insert(entity, t);
        ctx.insert(entity, GlobalTransform::default());
        if let Some(p) = parent {
            ctx.insert(entity, Parent(p));
        }
        entity
    }

    fn global(ctx: &PipelineContext, entity: Entity) -> WorldMatrix {
        ctx.get::<GlobalTransform>(entity).unwrap().0
    }

    // A chain of `depth` entities each parented to the one above, translated a
    // unit apart, returned root-first.
    fn chain(ctx: &mut PipelineContext, depth: usize) -> Vec<Entity> {
        let mut chain = Vec::with_capacity(depth);
        let mut parent = None;
        for i in 0..depth {
            let entity = spawn(ctx, translate(i as f32 + 1.0), parent);
            chain.push(entity);
            parent = Some(entity);
        }
        chain
    }

    // propagate_transforms composes each entity's GlobalTransform from its parent
    // chain: a root's world matrix is its local, a child's is parent_world * local.
    #[test]
    fn propagate_transforms_composes_parent_then_child() {
        let parent_t = Transform {
            position: [1.0, 2.0, 3.0],
            rotation_deg: [0.0, 30.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        };
        let child_t = Transform {
            position: [0.0, 0.0, 1.0],
            rotation_deg: [10.0, 0.0, 5.0],
            scale: [2.0, 2.0, 2.0],
        };

        let mut world = TestWorld::new();
        let mut ctx = world.ctx();
        let parent_e = spawn(&mut ctx, parent_t, None);
        let child_e = spawn(&mut ctx, child_t, Some(parent_e));

        propagate_transforms(&mut ctx);

        assert_eq!(
            global(&ctx, parent_e),
            parent_t.model_matrix(),
            "root world = local"
        );
        assert_eq!(
            global(&ctx, child_e),
            mat4_mul(parent_t.model_matrix(), child_t.model_matrix()),
            "child world = parent_world * local"
        );
    }

    // The cached per-frame path resolves the same parent-then-child composition
    // as the uncached `propagate_transforms`.
    #[test]
    fn cached_propagation_matches_the_uncached_path() {
        let parent_t = Transform {
            position: [1.0, 2.0, 3.0],
            rotation_deg: [0.0, 30.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        };
        let child_t = Transform {
            position: [0.0, 0.0, 1.0],
            rotation_deg: [10.0, 0.0, 5.0],
            scale: [2.0, 2.0, 2.0],
        };

        let mut world = TestWorld::new();
        let mut ctx = world.ctx();
        let parent_e = spawn(&mut ctx, parent_t, None);
        let child_e = spawn(&mut ctx, child_t, Some(parent_e));

        let mut cache = TransformCache::default();
        propagate_transforms_cached(&mut ctx, &mut cache);

        assert_eq!(global(&ctx, parent_e), parent_t.model_matrix());
        assert_eq!(
            global(&ctx, child_e),
            mat4_mul(parent_t.model_matrix(), child_t.model_matrix())
        );
    }

    // The cached path skips the resolve (and the GlobalTransform writes) on
    // frames where no Transform / Parent changed, and recomputes once one does.
    #[test]
    fn cached_propagation_skips_until_a_transform_changes() {
        let mut world = TestWorld::new();
        let mut ctx = world.ctx();
        let t0 = translate(1.0);
        let e = spawn(&mut ctx, t0, None);

        let mut cache = TransformCache::default();
        propagate_transforms_cached(&mut ctx, &mut cache);
        assert_eq!(global(&ctx, e), t0.model_matrix());

        // A GlobalTransform write does not dirty the Transform column, so the
        // next pass must skip and leave the (deliberately corrupted) value.
        ctx.get_mut::<GlobalTransform>(e).unwrap().0 = IDENTITY4;
        propagate_transforms_cached(&mut ctx, &mut cache);
        assert_eq!(
            global(&ctx, e),
            IDENTITY4,
            "unchanged Transform => propagation skipped"
        );

        // Mutating the Transform dirties its row; the next pass recomputes.
        let t1 = Transform {
            position: [0.0, 5.0, 0.0],
            rotation_deg: [0.0; 3],
            scale: [1.0; 3],
        };
        *ctx.get_mut::<Transform>(e).unwrap() = t1;
        propagate_transforms_cached(&mut ctx, &mut cache);
        assert_eq!(
            global(&ctx, e),
            t1.model_matrix(),
            "changed Transform => propagation recomputed"
        );
    }

    // A moved entity carries its whole subtree with it: every descendant's
    // GlobalTransform recomposes against the new ancestor world matrix.
    #[test]
    fn moving_a_root_recomposes_its_whole_subtree() {
        let mut world = TestWorld::new();
        let mut ctx = world.ctx();
        let links = chain(&mut ctx, 4);

        let mut cache = TransformCache::default();
        propagate_transforms_cached(&mut ctx, &mut cache);

        let moved = translate(100.0);
        *ctx.get_mut::<Transform>(links[0]).unwrap() = moved;
        propagate_transforms_cached(&mut ctx, &mut cache);

        // Each link's world matrix is its ancestors composed left to right.
        let mut expected = moved.model_matrix();
        assert_eq!(global(&ctx, links[0]), expected);
        for (depth, &link) in links.iter().enumerate().skip(1) {
            let local = translate(depth as f32 + 1.0).model_matrix();
            expected = mat4_mul(expected, local);
            assert_eq!(global(&ctx, link), expected, "link at depth {depth}");
        }
    }

    // The pass is genuinely incremental: an entity outside the moved subtree is
    // not rewritten, so a deliberately corrupted sibling stays corrupted.
    #[test]
    fn an_untouched_subtree_is_not_rewritten() {
        let mut world = TestWorld::new();
        let mut ctx = world.ctx();
        let root_a = spawn(&mut ctx, translate(1.0), None);
        let child_a = spawn(&mut ctx, translate(2.0), Some(root_a));
        let root_b = spawn(&mut ctx, translate(3.0), None);
        let child_b = spawn(&mut ctx, translate(4.0), Some(root_b));

        let mut cache = TransformCache::default();
        propagate_transforms_cached(&mut ctx, &mut cache);

        ctx.get_mut::<GlobalTransform>(child_b).unwrap().0 = IDENTITY4;
        *ctx.get_mut::<Transform>(root_a).unwrap() = translate(50.0);
        propagate_transforms_cached(&mut ctx, &mut cache);

        assert_eq!(
            global(&ctx, child_a),
            mat4_mul(
                translate(50.0).model_matrix(),
                translate(2.0).model_matrix()
            ),
            "the moved root's subtree recomposed"
        );
        assert_eq!(
            global(&ctx, child_b),
            IDENTITY4,
            "the other tree was never walked"
        );
        assert_eq!(global(&ctx, root_b), translate(3.0).model_matrix());
    }

    // An ancestor and one of its descendants dirtied in the same frame: the
    // descendant must end up composed against the ancestor's NEW world matrix,
    // not the stale one it would read if the walks ran deepest-first.
    #[test]
    fn a_dirty_ancestor_and_descendant_resolve_against_the_new_parent() {
        let mut world = TestWorld::new();
        let mut ctx = world.ctx();
        let links = chain(&mut ctx, 3);

        let mut cache = TransformCache::default();
        propagate_transforms_cached(&mut ctx, &mut cache);

        // Dirty the deepest link first, so column order alone would resolve it
        // before its ancestor.
        *ctx.get_mut::<Transform>(links[2]).unwrap() = translate(7.0);
        *ctx.get_mut::<Transform>(links[0]).unwrap() = translate(9.0);
        propagate_transforms_cached(&mut ctx, &mut cache);

        let root = translate(9.0).model_matrix();
        let mid = mat4_mul(root, translate(2.0).model_matrix());
        assert_eq!(global(&ctx, links[0]), root);
        assert_eq!(global(&ctx, links[1]), mid);
        assert_eq!(
            global(&ctx, links[2]),
            mat4_mul(mid, translate(7.0).model_matrix())
        );
    }

    // Adding an entity moves the Transform column's structural tick, which the
    // per-row stamps cannot describe, so the pass falls back to a full resolve
    // and the new entity is picked up.
    #[test]
    fn a_spawned_entity_forces_a_full_resolve() {
        let mut world = TestWorld::new();
        let mut ctx = world.ctx();
        let root = spawn(&mut ctx, translate(1.0), None);

        let mut cache = TransformCache::default();
        propagate_transforms_cached(&mut ctx, &mut cache);

        let late = spawn(&mut ctx, translate(5.0), Some(root));
        propagate_transforms_cached(&mut ctx, &mut cache);

        assert_eq!(
            global(&ctx, late),
            mat4_mul(translate(1.0).model_matrix(), translate(5.0).model_matrix()),
            "the entity added after the last resolve composed correctly"
        );
    }

    // Despawning likewise moves the structural tick, and the swap-remove that
    // frees the row must not leave the survivors pointing at the wrong slots.
    #[test]
    fn a_despawned_entity_leaves_the_survivors_correct() {
        let mut world = TestWorld::new();
        let mut ctx = world.ctx();
        let root = spawn(&mut ctx, translate(1.0), None);
        let doomed = spawn(&mut ctx, translate(2.0), Some(root));
        let kept = spawn(&mut ctx, translate(3.0), Some(root));

        let mut cache = TransformCache::default();
        propagate_transforms_cached(&mut ctx, &mut cache);

        ctx.despawn(doomed);
        *ctx.get_mut::<Transform>(root).unwrap() = translate(20.0);
        propagate_transforms_cached(&mut ctx, &mut cache);

        assert_eq!(
            global(&ctx, kept),
            mat4_mul(
                translate(20.0).model_matrix(),
                translate(3.0).model_matrix()
            )
        );
    }

    // A whole-column write leaves no per-row stamps to read, so the pass must
    // fall back and pick up every entity's new value.
    #[test]
    fn a_whole_column_write_falls_back_to_a_full_resolve() {
        let mut world = TestWorld::new();
        let mut ctx = world.ctx();
        let root = spawn(&mut ctx, translate(1.0), None);
        let child = spawn(&mut ctx, translate(2.0), Some(root));

        let mut cache = TransformCache::default();
        propagate_transforms_cached(&mut ctx, &mut cache);

        for t in ctx.query_mut::<Transform>() {
            t.position[1] += 3.0;
        }
        propagate_transforms_cached(&mut ctx, &mut cache);

        let shifted = |x: f32| Transform {
            position: [x, 3.0, 0.0],
            rotation_deg: [0.0; 3],
            scale: [1.0; 3],
        };
        assert_eq!(global(&ctx, root), shifted(1.0).model_matrix());
        assert_eq!(
            global(&ctx, child),
            mat4_mul(shifted(1.0).model_matrix(), shifted(2.0).model_matrix())
        );
    }

    // Enough entities dirtied to blow the budget: the pass gives up on the
    // subtree walks and full-resolves, which must reach every one of them.
    #[test]
    fn a_dirty_set_past_the_budget_falls_back_and_still_resolves() {
        let mut world = TestWorld::new();
        let mut ctx = world.ctx();
        let entities: Vec<Entity> = (0..16)
            .map(|i| spawn(&mut ctx, translate(i as f32), None))
            .collect();

        let mut cache = TransformCache::default();
        propagate_transforms_cached(&mut ctx, &mut cache);

        // 16 / DIRTY_BUDGET_DIVISOR is 2, so moving half is well past it.
        for (i, &e) in entities.iter().enumerate().take(8) {
            *ctx.get_mut::<Transform>(e).unwrap() = translate(100.0 + i as f32);
        }
        propagate_transforms_cached(&mut ctx, &mut cache);

        for (i, &e) in entities.iter().enumerate() {
            let expected = if i < 8 {
                translate(100.0 + i as f32)
            } else {
                translate(i as f32)
            };
            assert_eq!(global(&ctx, e), expected.model_matrix(), "entity {i}");
        }
    }

    // A Parent naming an entity that owns no Transform has nothing to compose
    // against, so the child resolves as a root.
    #[test]
    fn a_parent_without_a_transform_leaves_the_child_a_root() {
        let mut world = TestWorld::new();
        let mut ctx = world.ctx();
        let bare = ctx.components.spawn();
        ctx.insert(bare, Children(concinnity_memory::InlineVec::new()));
        let child = spawn(&mut ctx, translate(4.0), Some(bare));

        propagate_transforms(&mut ctx);
        assert_eq!(global(&ctx, child), translate(4.0).model_matrix());
    }

    // resolve_world_matrices breaks a parent cycle: mutually-parented entities
    // fall back to their own local matrix rather than looping forever.
    #[test]
    fn resolve_world_matrices_breaks_parent_cycle() {
        let mut world = TestWorld::new();
        let mut ctx = world.ctx();
        let a_t = translate(1.0);
        let b_t = Transform {
            position: [0.0, 2.0, 0.0],
            rotation_deg: [0.0; 3],
            scale: [1.0; 3],
        };

        // a parents b and b parents a: a cycle with no root.
        let a = ctx.components.spawn();
        ctx.insert(a, a_t);
        let b = ctx.components.spawn();
        ctx.insert(b, b_t);
        ctx.insert(a, Parent(b));
        ctx.insert(b, Parent(a));

        let resolved = resolve_world_matrices(&ctx);
        assert_eq!(resolved.len(), 2);
        // Neither resolved through the chain, so each keeps its own local matrix.
        assert_eq!(resolved.get(&a).copied(), Some(a_t.model_matrix()));
        assert_eq!(resolved.get(&b).copied(), Some(b_t.model_matrix()));
    }

    // An entity hanging off a cycle inherits the fallback: with no resolvable
    // ancestor world matrix it keeps its own local rather than composing
    // against a matrix that was never resolved.
    #[test]
    fn an_entity_below_a_cycle_falls_back_to_its_local() {
        let mut world = TestWorld::new();
        let mut ctx = world.ctx();
        let a = spawn(&mut ctx, translate(1.0), None);
        let b = spawn(&mut ctx, translate(2.0), Some(a));
        ctx.insert(a, Parent(b));
        let below = spawn(&mut ctx, translate(3.0), Some(b));

        propagate_transforms(&mut ctx);

        assert_eq!(global(&ctx, a), translate(1.0).model_matrix());
        assert_eq!(global(&ctx, b), translate(2.0).model_matrix());
        assert_eq!(global(&ctx, below), translate(3.0).model_matrix());
    }

    // A hierarchy deeper than any recursion budget: the depth walk and the
    // resolve are both iterative, so this composes rather than overflowing.
    // Each link steps a single unit, so every partial sum along the chain is an
    // exact f32 and the deepest world matrix can be asserted outright.
    #[test]
    fn a_very_deep_chain_resolves_iteratively() {
        const DEPTH: usize = 20_000;

        let mut world = TestWorld::new();
        let mut ctx = world.ctx();
        let mut links = Vec::with_capacity(DEPTH);
        let mut parent = None;
        for _ in 0..DEPTH {
            let entity = spawn(&mut ctx, translate(1.0), parent);
            links.push(entity);
            parent = Some(entity);
        }

        propagate_transforms(&mut ctx);

        assert_eq!(global(&ctx, links[DEPTH - 1])[3][0], DEPTH as f32);
    }

    #[test]
    fn reparent_recomposes_child_world_matrix_and_relists() {
        let (a_t, b_t, child_t) = (translate(10.0), translate(-5.0), translate(1.0));

        let mut world = TestWorld::new();
        let mut ctx = world.ctx();
        let a = spawn(&mut ctx, a_t, None);
        let b = spawn(&mut ctx, b_t, None);
        let child = spawn(&mut ctx, child_t, None);

        // Attach under A: the child's world matrix composes A x local, and A
        // lists it.
        reparent(&mut ctx, child, Some(a));
        let under_a = global(&ctx, child);
        assert_eq!(
            under_a,
            mat4_mul(a_t.model_matrix(), child_t.model_matrix())
        );
        assert_eq!(ctx.get::<Children>(a).unwrap().0, vec![child]);

        // Move under B: world matrix recomposes against B, A unlists it.
        reparent(&mut ctx, child, Some(b));
        let under_b = global(&ctx, child);
        assert_eq!(
            under_b,
            mat4_mul(b_t.model_matrix(), child_t.model_matrix())
        );
        assert_ne!(under_a, under_b, "the child actually moved");
        assert!(
            ctx.get::<Children>(a).unwrap().0.is_empty(),
            "A unlisted the child"
        );
        assert_eq!(ctx.get::<Children>(b).unwrap().0, vec![child]);
        assert_eq!(ctx.get::<Parent>(child).unwrap().0, b);

        // Detach to a root: no Parent, world matrix is its own local.
        reparent(&mut ctx, child, None);
        assert_eq!(global(&ctx, child), child_t.model_matrix());
        assert!(ctx.get::<Parent>(child).is_none(), "child is now a root");
        assert!(
            ctx.get::<Children>(b).unwrap().0.is_empty(),
            "B unlisted the child"
        );
    }
}
