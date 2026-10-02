//! The CPU side of the scene acceleration structure a backend ray-traces
//! reflections against: which geometry its BLAS cover, the order its TLAS
//! instances and geometry table follow, when it is updated and how, and the
//! rings its per-frame rebuilds reuse.
//!
//! A backend keeps one [`AccelBook`] beside its device structures. Each frame it
//! asks the book for a plan, records the API work the plan calls for, and
//! commits the result back, so the policy and the instance layout the trace
//! depends on are the same on every backend while the build inputs, barriers
//! and residency stay with the API.

mod book;
mod plan;
mod rings;

pub use book::{AccelBook, HeadRefresh, InstanceBlas, SeedSet};
pub use plan::{
    EmptyHead, FailureStreak, RefreshMode, RtStep, RtUpdate, RtUpdatePlan, StreakChange,
    empty_head, seed_wanted,
};
pub use rings::{FrameRing, ScratchRing, SlotLiveness, StaticRing};
