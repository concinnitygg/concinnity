//! Placement policy for device memory: which block and what byte offset a
//! resource occupies. `block_alloc` stacks core's `RangeAllocator`, which places
//! within one span, into a pool of blocks so a backend spends its
//! allocation-count budget on blocks rather than on resources. `staging` rings the CPU-visible bytes a
//! recorded upload copies out of.
//!
//! Pure policy: no device handles, no API calls. The backend owns the blocks and
//! performs the bind, which is why frees are keyed on a retire frame rather than
//! released here.

pub(crate) mod block_alloc;
#[cfg(any(backend_dx, backend_vk))]
pub(crate) mod staging;
