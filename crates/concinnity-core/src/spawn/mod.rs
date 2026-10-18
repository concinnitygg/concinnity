//! Runtime entity churn: instantiating a copy of an authored placement, the
//! two clocks that drive it, and the system that drains the runtime spawn,
//! despawn, reparent and show/hide requests.
//!
//! What a copy carries and when one is due is the world's business and lives
//! in `template`. Allocating the copy's draw slot and retiring it afterwards go
//! through the `RenderSlots` allocator, reached through the `clone_slot` /
//! `acquire_slot` closures the builders take.

mod despawn;
mod system;
mod template;
mod visibility;

pub use system::SpawnSystem;
pub use template::{
    DueSpawn, spawn_from_template, spawn_skinned_from_template, tick_lifetimes, tick_spawners,
};
