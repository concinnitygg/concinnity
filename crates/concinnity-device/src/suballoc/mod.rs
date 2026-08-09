// src/suballoc/mod.rs
//
// Placement policy for device memory: which block and what byte offset a
// resource occupies. `range_alloc` places within one span; `block_alloc` stacks
// it into a pool of blocks so a backend spends its allocation-count budget on
// blocks rather than on resources.
//
// Pure policy: no device handles, no API calls. The backend owns the blocks and
// performs the bind, which is why frees are keyed on a retire frame rather than
// released here.
pub mod block_alloc;
pub mod range_alloc;
