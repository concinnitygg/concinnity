// src/spawn/mod.rs
//
// SpawnSystem: the per-frame entity churn. Ticks Lifetime countdowns and
// Spawner cadences, and drains the runtime DespawnRequest / ReparentRequest /
// SpawnRequest events. GPU slot decisions come from the engine's `RenderSlots`
// allocator; the backend effects are recorded into the frame's op queue and
// replayed at submission:
//   mod.rs      system + the per-frame drains
//   template.rs instantiate a copy of a placement (static or skinned)
//   despawn.rs  subtree removal + draw-slot retirement
//
// Scheduled immediately before GraphicsSystem so a despawned entity is
// already gone from the GlobalTransform x RenderHandle join when transforms
// are pushed (it contributes nothing to any pass this same frame), and so a
// spawn reuses slots freed this same frame before the backend grows its draw
// list. The world clock (Lifetime + Spawner) freezes while a menu is open
// (`MenuActive`, published by OverlaySystem earlier this tick).

use crate::components::{
    DespawnRequest, EntityTarget, ReparentRequest, SpawnRequest, VisibilityRequest,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{ActiveRenderQueues, PipelineContext, StepResult, System};
use crate::gfx::ops::RenderOps;
use crate::gfx::render_slots::RenderSlots;
use crate::gfx::transform_propagation;
use std::time::Instant;

// Resolve a request target to a live entity. A named target goes through the
// world's name index; an entity-addressed one is already what the caller means,
// including entities that never had a name.
fn resolve_target(ctx: &PipelineContext, target: EntityTarget) -> Option<crate::ecs::Entity> {
    match target {
        EntityTarget::Name(name) => resolve_name(ctx, name),
        EntityTarget::Entity(entity) => Some(entity),
    }
}

// The entity a decomposed name resolves to. Borrows the name index only for
// the lookup, so the caller is free to take `&mut ctx` immediately after.
fn resolve_name(ctx: &PipelineContext, name: AssetId) -> Option<crate::ecs::Entity> {
    ctx.resource::<crate::ecs::decompose::EntityByName>()?
        .0
        .get(&name)
        .copied()
}

mod despawn;
mod template;
mod visibility;

// Allocate a destination draw slot and record the backend clone for it: the
// `clone_slot` seam `template::spawn_from_template` drives. The slot index is
// known immediately (the engine owns allocation); the GPU copy applies when
// the frame's ops replay. A clone that fails at replay leaves a hidden slot,
// logged there.
fn record_clone_static(
    ops: &mut RenderOps,
    slots: &mut RenderSlots,
    src: usize,
    model: [[f32; 4]; 4],
) -> Option<usize> {
    use crate::gfx::draw_slot::SlotAlloc;
    let dst = slots.allocate_draw();
    let idx = match dst {
        SlotAlloc::Reuse(i) | SlotAlloc::Append(i) => i,
    };
    ops.record(move |backend| {
        if let Err(e) = backend.clone_static_draw_object(src, model, dst) {
            tracing::warn!("SpawnSystem: draw-slot clone of {} failed: {}", src, e);
        }
    });
    Some(idx)
}

// Claim a pre-reserved skinned instance and record its reveal: the
// `acquire_slot` seam `template::spawn_skinned_from_template` drives. `None`
// when the template's reserve is exhausted, exactly like the old backend
// claim.
fn record_skinned_claim(
    ops: &mut RenderOps,
    slots: &mut RenderSlots,
    template: usize,
    model: [[f32; 4]; 4],
) -> Option<usize> {
    let instance = slots.claim_skinned(template)?;
    ops.record(move |backend| backend.reveal_skinned_instance(instance, model));
    Some(instance)
}

#[derive(Debug, Default)]
pub struct SpawnSystem {
    // Cursor into the Events<DespawnRequest> queue (runtime entity despawn:
    // cn debug `despawn`, and gameplay-driven removal once that path exists).
    despawn_cmd_cursor: crate::ecs::EventCursor,
    // Cursor into the Events<ReparentRequest> queue (runtime re-parenting:
    // cn debug `reparent`, and gameplay-driven moves once that path exists).
    reparent_cmd_cursor: crate::ecs::EventCursor,
    // Cursor into the Events<SpawnRequest> queue (runtime entity spawn: cn debug
    // `spawn`, and gameplay-driven spawning once that path exists).
    spawn_cmd_cursor: crate::ecs::EventCursor,
    // Cursor into the Events<VisibilityRequest> queue (runtime show/hide:
    // Behavior show/hide nodes).
    visibility_cmd_cursor: crate::ecs::EventCursor,
    // Clock base and the cumulative elapsed seconds at the previous step, so
    // each step derives the per-frame dt for the Lifetime / Spawner ticks.
    start_time: Option<Instant>,
    prev_elapsed: f32,
}

impl SpawnSystem {
    pub fn new() -> Self {
        Self::default()
    }
}

impl System for SpawnSystem {
    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
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
    fn drain(&mut self, ctx: &mut PipelineContext, ops: &mut RenderOps, slots: &mut RenderSlots) {
        let elapsed = self
            .start_time
            .get_or_insert_with(Instant::now)
            .elapsed()
            .as_secs_f32();
        // Per-frame delta for the time-based ticks. Clamped to non-negative so
        // a clock reset never rushes an expiry.
        let dt = (elapsed - self.prev_elapsed).max(0.0);
        self.prev_elapsed = elapsed;
        // Copied out of the context so the event drains below can hold scratch
        // while the cascades they feed take `ctx` mutably.
        let frame = ctx.frame;
        // The menu state OverlaySystem published earlier this tick.
        let menu_active = ctx
            .resource::<crate::ecs::MenuActive>()
            .map(|m| m.0)
            .unwrap_or(false);

        // Timed despawn: decrement every Lifetime by this frame's dt and
        // despawn the entities whose countdown reached zero, through the
        // same cascade a DespawnRequest uses. This is the churn that
        // returns draw slots to the free list for the spawn drain below
        // to recycle. Frozen while a menu is open so the world clock
        // (timed despawns + cadence spawns below) truly pauses.
        if !menu_active {
            let expired = template::tick_lifetimes(ctx, dt);
            for &entity in expired.iter() {
                despawn::despawn_subtree(ctx, ops, slots, entity);
            }
        }

        // Runtime entity despawn: drain DespawnRequest events, resolve
        // each name to its entity, hide that entity's draw slots, and
        // remove it (and its descendants) from the ECS. Done before
        // GraphicsSystem's transform push so a despawned entity is already
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
            transform_propagation::reparent(ctx, child, parent);
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
        for &req in spawn_reqs.iter() {
            let Some(template) = resolve_name(ctx, req.template) else {
                continue;
            };
            // A skinned template (a SkeletonPose entity) claims a
            // pre-reserved instance slot; a static one clones a draw
            // slot. Dispatch on which the template carries.
            if ctx
                .get::<crate::components::SkeletonPose>(template)
                .is_some()
            {
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
        for &due in due_spawns.iter() {
            let Some(template) = resolve_name(ctx, due.template) else {
                continue;
            };
            if ctx
                .get::<crate::components::SkeletonPose>(template)
                .is_some()
            {
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
