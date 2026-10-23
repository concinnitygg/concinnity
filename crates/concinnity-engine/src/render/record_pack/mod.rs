//! Packing the per-frame cull records -- each draw's `GpuObjectData` and
//! `GpuDrawArgs` -- in contiguous chunks that can run on separate threads.
//!
//! A chunk owns a disjoint run of records in both outputs and in the
//! model-history tracker, so chunks share nothing and every record is written
//! by exactly one of them. A record's bytes depend only on its own draw, so a
//! pack split any number of ways writes what a serial pack writes.
//!
//! Records that do not change from frame to frame (the instance tail) are
//! written into each ring slot once per change instead, tracked by
//! [`StaticSlots`].

mod chunk;
mod pack;
mod static_slots;

pub use chunk::{MAX_PACK_CHUNKS, MIN_CHUNK_RECORDS, chunk_count, chunk_range};
pub use pack::{PackChunk, PackView, StaticPack, pool_object_record};
pub use static_slots::StaticSlots;
