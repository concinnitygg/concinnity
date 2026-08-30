// The PhysicsConfig a world with physics content simulates on. The schedule
// gate falls back to the same values when the component is absent, so injecting
// it changes no simulation; what it changes is that the settings are a
// component tooling can read and an editor can write.

use crate::components::{PhysicsConfig, PropBody, RigidBody, TriggerVolume};
use crate::ecs::PipelineContext;
use crate::resource::SkinnedMeshTable;

pub(super) fn inject(ctx: &mut PipelineContext) {
    if ctx.query::<PhysicsConfig>().next().is_some() || !has_physics_content(ctx) {
        return;
    }
    ctx.push(PhysicsConfig::default());
}

// Mirrors the physics schedule gate, minus the PhysicsConfig arm the caller
// has already ruled out: the two must agree on which worlds run physics, or a
// world would receive a config it never simulates with.
fn has_physics_content(ctx: &PipelineContext) -> bool {
    ctx.query::<RigidBody>().next().is_some()
        || ctx.query::<PropBody>().next().is_some()
        || ctx.query::<TriggerVolume>().next().is_some()
        || ctx
            .resource::<SkinnedMeshTable>()
            .is_some_and(SkinnedMeshTable::has_capsule)
}
