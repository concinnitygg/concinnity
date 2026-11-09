//! Runtime entity churn as the world sees it: the spawn, despawn, reparent
//! and show/hide requests, the two clocks that make copies due, and
//! instantiating a copy of an authored placement.
//!
//! What a copy carries and when one is due is the world's business and lives
//! in `template`. Allocating the copy's draw slot and retiring it afterwards
//! belong to the renderer, which drains the requests
//! ([`SpawnSystem`](crate::render::spawn::SpawnSystem)) and reaches `template`
//! through the `clone_slot` / `acquire_slot` closures the builders take.

mod template;

pub use template::{
    DueSpawn, spawn_from_template, spawn_skinned_from_template, tick_lifetimes, tick_spawners,
};

pub(crate) mod despawn_request;
pub(crate) mod hidden;
pub(crate) mod lifetime;
pub(crate) mod prop;
pub(crate) mod prop_instance;
pub(crate) mod reparent_request;
pub(crate) mod spawn_request;
pub(crate) mod spawner;
pub(crate) mod visibility_request;
