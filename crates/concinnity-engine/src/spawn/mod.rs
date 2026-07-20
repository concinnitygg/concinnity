// src/spawn/mod.rs
//
// SpawnSystem: the per-frame entity churn. Ticks Lifetime countdowns and
// Spawner cadences, and drains the runtime DespawnRequest / ReparentRequest /
// SpawnRequest events, retiring and recycling GPU draw slots through the
// world's parked render backend:
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

use crate::assets::{DespawnRequest, ReparentRequest, SpawnRequest};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{ActiveRenderBackend, PipelineContext, StepResult, System};
use crate::gfx::backend::RenderBackend;
use crate::gfx::draw_list;
use std::time::Instant;

mod despawn;
mod template;

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
        // No parked backend (graphics failed, or the editor transplanted it
        // away): no draw slots to retire or clone, so the churn waits.
        let Some(mut backend) = ActiveRenderBackend::take(ctx.resources) else {
            return StepResult::Continue;
        };
        self.drain(ctx, backend.as_mut());
        ActiveRenderBackend::put(ctx.resources, backend);
        StepResult::Continue
    }
}

impl SpawnSystem {
    fn drain(&mut self, ctx: &mut PipelineContext, backend: &mut dyn RenderBackend) {
        let elapsed = self
            .start_time
            .get_or_insert_with(Instant::now)
            .elapsed()
            .as_secs_f32();
        // Per-frame delta for the time-based ticks. Clamped to non-negative so
        // a clock reset never rushes an expiry.
        let dt = (elapsed - self.prev_elapsed).max(0.0);
        self.prev_elapsed = elapsed;
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
            for entity in expired {
                despawn::despawn_subtree(ctx, backend, entity);
            }
        }

        // Runtime entity despawn: drain DespawnRequest events, resolve
        // each name to its entity, hide that entity's draw slots, and
        // remove it (and its descendants) from the ECS. Done before
        // GraphicsSystem's transform push so a despawned entity is already
        // gone from the GlobalTransform x RenderHandle join this frame and
        // contributes nothing to any pass.
        let despawn_names: Vec<AssetId> = match ctx.events::<DespawnRequest>() {
            Some(events) => events
                .read(&mut self.despawn_cmd_cursor)
                .into_iter()
                .map(|r| r.name)
                .collect(),
            None => Vec::new(),
        };
        if !despawn_names.is_empty() {
            // Clone the name index out so the ctx borrow ends before the
            // despawns, which take &mut ctx.
            let by_name = ctx
                .resource::<crate::ecs::decompose::EntityByName>()
                .map(|n| n.0.clone())
                .unwrap_or_default();
            for name in despawn_names {
                if let Some(&entity) = by_name.get(&name) {
                    despawn::despawn_subtree(ctx, backend, entity);
                }
            }
        }

        // Runtime re-parenting: drain ReparentRequest events, resolve the
        // child + parent names to entities, and re-point the child's
        // Parent edge (recomposing world matrices). After the despawn
        // drain so a reparent naming a just-removed entity simply finds
        // nothing to move.
        let reparents: Vec<ReparentRequest> = match ctx.events::<ReparentRequest>() {
            Some(events) => events
                .read(&mut self.reparent_cmd_cursor)
                .into_iter()
                .copied()
                .collect(),
            None => Vec::new(),
        };
        if !reparents.is_empty() {
            let by_name = ctx
                .resource::<crate::ecs::decompose::EntityByName>()
                .map(|n| n.0.clone())
                .unwrap_or_default();
            for req in reparents {
                let Some(&child) = by_name.get(&req.child) else {
                    continue;
                };
                let parent = req.parent.and_then(|p| by_name.get(&p).copied());
                // A named-but-unresolved parent skips, so a typo never
                // silently detaches the child to a root.
                if req.parent.is_some() && parent.is_none() {
                    continue;
                }
                draw_list::reparent(ctx, child, parent);
            }
        }

        // Runtime entity spawn: drain SpawnRequest events, resolve each
        // template name to its entity, and instantiate a copy at the
        // requested transform. Each cloned draw slot reuses one freed by
        // an earlier despawn / Lifetime expiry before the backend grows
        // its draw_objects, so steady spawn/despawn churn does not leak
        // slots. After the despawn / reparent drains so a spawn can reuse
        // slots freed this same frame.
        let spawn_reqs: Vec<SpawnRequest> = match ctx.events::<SpawnRequest>() {
            Some(events) => events
                .read(&mut self.spawn_cmd_cursor)
                .into_iter()
                .copied()
                .collect(),
            None => Vec::new(),
        };
        if !spawn_reqs.is_empty() {
            let by_name = ctx
                .resource::<crate::ecs::decompose::EntityByName>()
                .map(|n| n.0.clone())
                .unwrap_or_default();
            for req in spawn_reqs {
                let Some(&template) = by_name.get(&req.template) else {
                    continue;
                };
                // A skinned template (a SkeletonPose entity) claims a
                // pre-reserved instance slot; a static one clones a draw
                // slot. Dispatch on which the template carries.
                if ctx.get::<crate::assets::SkeletonPose>(template).is_some() {
                    template::spawn_skinned_from_template(
                        ctx,
                        template,
                        req.name,
                        req.transform,
                        req.lifetime_secs,
                        |tmpl, model| backend.spawn_skinned_instance(tmpl, model),
                    );
                } else {
                    template::spawn_from_template(
                        ctx,
                        template,
                        req.name,
                        req.transform,
                        req.lifetime_secs,
                        |src, model| backend.clone_static_draw_object(src, model).ok(),
                    );
                }
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
            Vec::new()
        } else {
            template::tick_spawners(ctx, dt)
        };
        if !due_spawns.is_empty() {
            let by_name = ctx
                .resource::<crate::ecs::decompose::EntityByName>()
                .map(|n| n.0.clone())
                .unwrap_or_default();
            for due in due_spawns {
                let Some(&template) = by_name.get(&due.template) else {
                    continue;
                };
                if ctx.get::<crate::assets::SkeletonPose>(template).is_some() {
                    template::spawn_skinned_from_template(
                        ctx,
                        template,
                        None,
                        due.transform,
                        due.lifetime,
                        |tmpl, model| backend.spawn_skinned_instance(tmpl, model),
                    );
                } else {
                    template::spawn_from_template(
                        ctx,
                        template,
                        None,
                        due.transform,
                        due.lifetime,
                        |src, model| backend.clone_static_draw_object(src, model).ok(),
                    );
                }
            }
        }
    }
}
