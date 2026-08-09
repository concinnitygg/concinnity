// src/ecs/mod.rs
//
// The one piece of the ECS that cannot live in the vocabulary crate: the
// build-time name -> dense id interner. It keeps a per-thread table
// (`thread_local!` + `Once`), which is std, so it belongs to this crate even
// though the identity types it produces are vocabulary.
pub mod asset_id;
mod name_interner;
