//! `SpawnSystem`: the per-frame entity churn. Ticks Lifetime countdowns and
//! Spawner cadences, and drains the runtime `DespawnRequest` / `ReparentRequest` /
//! `SpawnRequest` events. GPU slot decisions come from the `RenderSlots`
//! allocator; the backend effects are recorded into the frame's op queue and
//! replayed at submission. Subtree removal and draw-slot retirement live in
//! `despawn.rs`, show/hide in `visibility.rs`, and copying a placement in
//! `template.rs`.
//!
//! Scheduled before the renderer so a despawned entity is
//! already gone from the `GlobalTransform` x `RenderHandle` join when transforms
//! are pushed (it contributes nothing to any pass this same frame), and so a
//! spawn reuses slots freed this same frame before the backend grows its draw
//! list. The world clock (Lifetime + Spawner) freezes while a menu is open
//! (`MenuActive`, published by `OverlaySystem` earlier this tick).

use super::{despawn, template, visibility};
use crate::components::{
    DespawnRequest, EntityTarget, ReparentRequest, SkeletonPose, SpawnRequest, VisibilityRequest,
};
use crate::ecs::{
    Entity, EventCursor, FrameTime, MenuActive, PipelineContext, StepResult, System, World,
};
use crate::gfx::render_types::{DrawIndex, SkinnedIndex};
use crate::render::ops::RenderOps;
use crate::render::queues::ActiveRenderQueues;
use crate::render::render_mode::RenderMode;
use crate::render::render_slots::RenderSlots;
use crate::transform::propagation;

// Resolve a request target to a live entity. A named target goes through the
// world's name index; an entity-addressed one is already what the caller means,
// including entities that never had a name.
fn resolve_target(ctx: &PipelineContext<'_>, target: EntityTarget) -> Option<Entity> {
    match target {
        EntityTarget::Name(name) => ctx.entity_of(name),
        EntityTarget::Entity(entity) => Some(entity),
    }
}

// Allocate a destination draw slot and record the backend clone for it: the
// `clone_slot` seam `template::spawn_from_template` drives. The slot index is
// known immediately (`RenderSlots` owns allocation); the GPU copy applies when
// the frame's ops replay. A clone that fails at replay leaves a hidden slot,
// logged there.
fn record_clone_static(
    ops: &mut RenderOps,
    slots: &mut RenderSlots,
    src: DrawIndex,
    model: [[f32; 4]; 4],
) -> Option<DrawIndex> {
    let dst = slots.allocate_draw();
    ops.record(move |backend| {
        if let Err(e) = backend.clone_static_draw_object(src, model, dst) {
            tracing::warn!("SpawnSystem: draw-slot clone of {} failed: {}", src, e);
        }
    });
    Some(dst.slot())
}

// Claim a pre-reserved skinned instance and record its reveal: the
// `acquire_slot` seam `template::spawn_skinned_from_template` drives. `None`
// when the template's reserve is exhausted.
fn record_skinned_claim(
    ops: &mut RenderOps,
    slots: &mut RenderSlots,
    template: SkinnedIndex,
    model: [[f32; 4]; 4],
) -> Option<SkinnedIndex> {
    let instance = slots.claim_skinned(template)?;
    ops.record(move |backend| backend.reveal_skinned_instance(instance, model));
    Some(instance)
}

/// The per-frame entity churn: Lifetime and Spawner ticks plus the runtime
/// despawn, show/hide, reparent and spawn requests.
#[derive(Debug, Default)]
pub struct SpawnSystem {
    // Cursors into the Events<DespawnRequest>, Events<ReparentRequest> and
    // Events<SpawnRequest> queues.
    despawn_cmd_cursor: EventCursor,
    reparent_cmd_cursor: EventCursor,
    spawn_cmd_cursor: EventCursor,
    // Cursor into the Events<VisibilityRequest> queue (runtime show/hide:
    // Behavior show/hide nodes).
    visibility_cmd_cursor: EventCursor,
}

impl SpawnSystem {
    /// The system for a world that runs with a window.
    pub fn for_world(world: &World) -> Option<Self> {
        RenderMode::renders_in(world).then(Self::new)
    }

    /// A system with every request cursor at the start of its queue.
    pub fn new() -> Self {
        Self::default()
    }
}

impl System for SpawnSystem {
    fn step(&mut self, ctx: &mut PipelineContext<'_>) -> StepResult {
        // The recording surfaces graphics init published; absent means
        // graphics never inited, so there are no draw slots to churn. Taken
        // out for the drain (see `ActiveRenderQueues`) so `ctx` stays freely
        // borrowable.
        let Some(mut queues) = ActiveRenderQueues::take(ctx.resources) else {
            return StepResult::Continue;
        };
        self.drain(ctx, &mut queues.ops, &mut queues.slots);
        ActiveRenderQueues::put(ctx.resources, queues);
        StepResult::Continue
    }
}

impl SpawnSystem {
    fn drain(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        ops: &mut RenderOps,
        slots: &mut RenderSlots,
    ) {
        // Per-frame delta for the time-based ticks. Clamped to non-negative so
        // a clock reset never rushes an expiry.
        let dt = ctx
            .resource::<FrameTime>()
            .copied()
            .unwrap_or_default()
            .dt
            .max(0.0);
        // Copied out of the context so the event drains below can hold scratch
        // while the cascades they feed take `ctx` mutably.
        let frame = ctx.frame;
        // The menu state OverlaySystem published earlier this tick.
        let menu_active = ctx.resource::<MenuActive>().is_some_and(|m| m.0);

        // Timed despawn: decrement every Lifetime by this frame's dt and
        // despawn the entities whose countdown reached zero, through the
        // same cascade a DespawnRequest uses. This is the churn that
        // returns draw slots to the free list for the spawn drain below
        // to recycle. Frozen while a menu is open so the world clock
        // (timed despawns + cadence spawns below) truly pauses.
        if !menu_active {
            let expired = template::tick_lifetimes(ctx, dt);
            for &entity in &expired {
                despawn::despawn_subtree(ctx, ops, slots, entity);
            }
        }

        // Runtime entity despawn: drain DespawnRequest events, resolve
        // each name to its entity, hide that entity's draw slots, and
        // remove it (and its descendants) from the ECS. Done before
        // the renderer's transform push so a despawned entity is already
        // gone from the GlobalTransform x RenderHandle join this frame and
        // contributes nothing to any pass.
        let despawn_targets = match ctx.events::<DespawnRequest>() {
            Some(events) => {
                frame.collect(events.read(&mut self.despawn_cmd_cursor).map(|r| r.target))
            }
            None => frame.collect([]),
        };
        for &target in &despawn_targets {
            if let Some(entity) = resolve_target(ctx, target) {
                despawn::despawn_subtree(ctx, ops, slots, entity);
            }
        }

        // Runtime show/hide: drain VisibilityRequest events, resolve each
        // name to its entity, and switch its subtree's Hidden tags and draw
        // slots. After the despawn drain so a request naming a just-removed
        // entity simply finds nothing to switch.
        let vis_reqs = match ctx.events::<VisibilityRequest>() {
            Some(events) => frame.collect(events.read(&mut self.visibility_cmd_cursor).copied()),
            None => frame.collect([]),
        };
        for &req in &vis_reqs {
            if let Some(entity) = resolve_target(ctx, req.target) {
                visibility::set_subtree_visibility(ctx, ops, entity, req.visible);
            }
        }

        // Runtime re-parenting: drain ReparentRequest events, resolve the
        // child + parent names to entities, and re-point the child's
        // Parent edge (recomposing world matrices). After the despawn
        // drain so a reparent naming a just-removed entity simply finds
        // nothing to move.
        let reparents = match ctx.events::<ReparentRequest>() {
            Some(events) => frame.collect(events.read(&mut self.reparent_cmd_cursor).copied()),
            None => frame.collect([]),
        };
        for &req in &reparents {
            let Some(child) = resolve_target(ctx, req.child) else {
                continue;
            };
            let parent = req.parent.and_then(|p| resolve_target(ctx, p));
            // A named-but-unresolved parent skips, so a typo never
            // silently detaches the child to a root.
            if req.parent.is_some() && parent.is_none() {
                continue;
            }
            propagation::reparent(ctx, child, parent);
        }

        // Runtime entity spawn: drain SpawnRequest events, resolve each
        // template name to its entity, and instantiate a copy at the
        // requested transform. Each cloned draw slot reuses one freed by
        // an earlier despawn / Lifetime expiry before the backend grows
        // its draw_objects, so steady spawn/despawn churn does not leak
        // slots. After the despawn / reparent drains so a spawn can reuse
        // slots freed this same frame.
        let spawn_reqs = match ctx.events::<SpawnRequest>() {
            Some(events) => frame.collect(events.read(&mut self.spawn_cmd_cursor).copied()),
            None => frame.collect([]),
        };
        for &req in &spawn_reqs {
            let Some(template) = ctx.entity_of(req.template) else {
                continue;
            };
            // A skinned template (a SkeletonPose entity) claims a
            // pre-reserved instance slot; a static one clones a draw
            // slot. Dispatch on which the template carries.
            if ctx.get::<SkeletonPose>(template).is_some() {
                template::spawn_skinned_from_template(
                    ctx,
                    template,
                    req.name,
                    req.transform,
                    req.lifetime_secs,
                    |tmpl, model| record_skinned_claim(ops, slots, tmpl, model),
                );
            } else {
                template::spawn_from_template(
                    ctx,
                    template,
                    req.name,
                    req.transform,
                    req.lifetime_secs,
                    |src, model| record_clone_static(ops, slots, src, model),
                );
            }
        }

        // Cadence-driven spawn: advance every Spawner's clock and
        // instantiate the copies now due, at the spawner's position.
        // Transient (unnamed) and Lifetime-bounded, so a steady spawner
        // churns through recycled draw slots. After the SpawnRequest
        // drain so both spawn paths reuse slots freed this frame. Frozen
        // while a menu is open so spawner clocks do not advance behind
        // the pause.
        let due_spawns = if menu_active {
            frame.collect([])
        } else {
            template::tick_spawners(ctx, dt)
        };
        for &due in &due_spawns {
            let Some(template) = ctx.entity_of(due.template) else {
                continue;
            };
            if ctx.get::<SkeletonPose>(template).is_some() {
                template::spawn_skinned_from_template(
                    ctx,
                    template,
                    None,
                    due.transform,
                    due.lifetime,
                    |tmpl, model| record_skinned_claim(ops, slots, tmpl, model),
                );
            } else {
                template::spawn_from_template(
                    ctx,
                    template,
                    None,
                    due.transform,
                    due.lifetime,
                    |src, model| record_clone_static(ops, slots, src, model),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::World;
    use crate::ecs::asset_id::AssetId;

    // Build an isolated PipelineContext over a fresh world, with no backend.
    fn run<R>(body: impl FnOnce(&mut PipelineContext<'_>) -> R) -> R {
        body(&mut World::new().context())
    }

    #[test]
    fn an_entity_target_resolves_to_itself_without_a_name_index() {
        run(|ctx| {
            let entity = ctx.components.spawn();
            assert_eq!(
                resolve_target(ctx, EntityTarget::Entity(entity)),
                Some(entity)
            );
        });
    }

    #[test]
    fn a_named_target_resolves_through_the_index() {
        run(|ctx| {
            let entity = ctx.components.spawn();
            ctx.identify(entity, AssetId(7));
            assert_eq!(
                resolve_target(ctx, EntityTarget::Name(AssetId(7))),
                Some(entity)
            );
            assert_eq!(resolve_target(ctx, EntityTarget::Name(AssetId(8))), None);
        });
    }

    #[test]
    fn a_named_target_without_a_name_index_resolves_to_nothing() {
        run(|ctx| {
            assert_eq!(resolve_target(ctx, EntityTarget::Name(AssetId(7))), None);
        });
    }
}
