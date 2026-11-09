//! The renderer's half of runtime entity churn: draining the world's spawn,
//! despawn, reparent and show/hide requests into draw-slot allocations and
//! the backend calls that clone, reveal, hide and retire them.

mod despawn;
mod system;
#[cfg(test)]
mod tests;
mod visibility;

pub use system::SpawnSystem;
