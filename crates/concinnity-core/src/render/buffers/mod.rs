//! GPU buffer memory every backend manages the same way: range and block
//! sub-allocation, growth policy, frame-retired frees, per-frame upload rings
//! and recorded staging uploads.

pub mod block_alloc;
pub mod block_pool;
pub mod growth;
pub mod range_alloc;
pub mod retire_pool;
pub mod staging;
pub mod staging_uploads;
pub mod upload_ring;
