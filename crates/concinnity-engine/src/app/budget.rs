//! The host half of the process budgets: measuring the machine for
//! `concinnity_core::budget` and mapping the thread budget onto thread roles.

use concinnity_core::budget::ThreadBudget;
use concinnity_core::components::FramePriority;
use concinnity_core::ecs::World;
use concinnity_host::thread::ThreadRole;

// The thread budget for this machine's core count and the job pool's default.
pub(crate) fn thread_budget(
    job_threads_override: u32,
    frame_priority: FramePriority,
) -> ThreadBudget {
    let total_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    ThreadBudget::compute(
        total_cores,
        concinnity_host::thread::jobs::default_threads(),
        job_threads_override,
        frame_priority,
    )
}

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
    fn auto_thread_budget_leaves_a_core_for_the_main_thread() {
        let tb = thread_budget(0, FramePriority::default());
        assert_eq!(tb.job_threads, tb.total_cores.saturating_sub(1).max(1));
        assert!(tb.job_threads >= 1);
    }

    #[test]
    fn a_world_without_a_budget_takes_the_default_role() {
        let mut world = World::new();
        assert_eq!(
            main_thread_role(&world),
            ThreadRole::Frame(FramePriority::default())
        );
        world.insert_resource(thread_budget(0, FramePriority::Normal));
        assert_eq!(
            main_thread_role(&world),
            ThreadRole::Frame(FramePriority::Normal)
        );
    }
}
