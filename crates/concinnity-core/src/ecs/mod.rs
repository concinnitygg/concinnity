// src/ecs/mod.rs
//
// The ECS surface, re-exported wholesale from concinnity-types, plus the one
// piece that cannot live there: the build-time name -> dense id interner. The
// interner keeps a per-thread table (`thread_local!` + `Once`), which is std,
// so it belongs to this crate even though the identity types it produces are
// vocabulary.
pub use concinnity_types::ecs::*;

pub mod asset_id;
mod name_interner;
