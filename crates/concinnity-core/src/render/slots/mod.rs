//! Ownership of backend draw and skinned-instance slots, the per-slot dirty
//! tracking, and the CPU half of repacking the shared geometry buffers.

pub mod draw_slot;
pub(crate) mod frame_dirty;
pub mod geometry_repack;
pub mod render_slots;
pub mod rewrites;
pub(crate) mod skinned_pool;
pub(crate) mod skinned_slots;
pub(crate) mod table;
