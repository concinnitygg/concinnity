// src/spawn/visibility.rs
//
// Runtime show/hide: switch a subtree's draw slots off and on without
// despawning anything. Driven by the VisibilityRequest events SpawnSystem
// drains each step. The Hidden tag records the state, and the scene-switch
// visibility snapshot skips tagged entities, so a scene jump never relights
// a hidden entity.

use crate::assets::{Hidden, RenderHandle};
use crate::ecs::{Entity, PipelineContext};
use crate::gfx::backend::RenderBackend;

pub(super) fn set_subtree_visibility(
    ctx: &mut PipelineContext,
    backend: &mut dyn RenderBackend,
    root: Entity,
    visible: bool,
) {
    for entity in super::despawn::collect_subtree(ctx, root) {
        if visible {
            ctx.remove::<Hidden>(entity);
        } else if ctx.get::<Hidden>(entity).is_none() {
            ctx.insert(entity, Hidden);
        }
        // Clone the slot list out so the immutable borrow ends before the
        // next entity's tag toggle.
        let slots: concinnity_memory::InlineVec<u32> = ctx
            .get::<RenderHandle>(entity)
            .map(|h| h.draws.clone())
            .unwrap_or_default();
        for slot in slots {
            backend.update_visibility(slot as usize, visible);
        }
    }
}
