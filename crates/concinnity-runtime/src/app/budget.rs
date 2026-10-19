//! The role the frame threads take from the published thread budget.

use concinnity_core::budget::ThreadBudget;
use concinnity_core::components::FramePriority;
use concinnity_core::ecs::World;
use concinnity_host::thread::ThreadRole;

/// The role the render and simulation threads take for `world`: its published
/// budget's, or the default priority's before `Runtime::start` publishes one.
pub(crate) fn main_thread_role(world: &World) -> ThreadRole {
    ThreadRole::Frame(
        world
            .resource::<ThreadBudget>()
            .map_or_else(FramePriority::default, |b| b.frame_priority),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_world_without_a_budget_takes_the_default_role() {
        let mut world = World::new();
        assert_eq!(
            main_thread_role(&world),
            ThreadRole::Frame(FramePriority::default())
        );
        world.insert_resource(ThreadBudget::compute(4, 3, 0, FramePriority::Normal));
        assert_eq!(
            main_thread_role(&world),
            ThreadRole::Frame(FramePriority::Normal)
        );
    }
}
