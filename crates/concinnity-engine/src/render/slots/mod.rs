//! Ownership of backend draw and skinned-instance slots, the per-slot dirty
//! tracking, and the CPU half of repacking the shared geometry buffers.

#[cfg(test)]
mod bench;
pub mod draw_slot;
pub(crate) mod frame_dirty;
pub mod geometry_repack;
pub(crate) mod render_handle;
pub mod render_slots;
pub mod rewrites;
pub(crate) mod skinned_pool;
pub(crate) mod skinned_slots;
pub(crate) mod table;
