//! Runtime entity churn: instantiating a copy of an authored placement, and
//! the two clocks that drive it.
//!
//! What a copy carries and when one is due is the world's business and lives
//! here. Allocating the copy's draw slot and retiring it afterwards are a
//! renderer's, and reach this through the `clone_slot` / `acquire_slot`
//! closures a caller passes in.

mod template;

pub use template::{
    DueSpawn, spawn_from_template, spawn_skinned_from_template, tick_lifetimes, tick_spawners,
};
