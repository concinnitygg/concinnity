//! Wiring tests: worlds started against the runtime's public system table,
//! checking what only the assembled schedule can show.

// The allocation pins read the heap counters, so this binary tracks its heap.
concinnity_core::install_global_allocator!();

mod alloc_budget;
mod consumed_columns;
mod controllers;
mod determinism;
mod systems;
