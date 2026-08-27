// What this crate lends `concinnity_core::behavior::BehaviorSystem`: the two
// things the system needs that only a host has.
//
//   save.rs   a state file under the host's save directory
//   eval.rs   the job pool a tick's evaluation fans out across
//
// The system itself, the VM it runs, and everything a tick does to the world
// are in concinnity-core; the gate in `ecs::schedule` builds it with both of
// these attached.

pub(crate) mod eval;
pub(crate) mod save;

#[cfg(test)]
mod tests;

use concinnity_core::behavior::BehaviorSystem;

// The behavior system as this host runs it: evaluation on the job pool, state
// in a file under the save directory when one exists.
pub(crate) fn build() -> BehaviorSystem {
    let system = BehaviorSystem::new().with_scheduler(Box::new(eval::Pool));
    match save::FileStore::new() {
        Some(store) => system.with_store(Box::new(store)),
        None => system,
    }
}
