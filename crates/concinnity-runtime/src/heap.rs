//! The tracked heap the runtime reports on.
//!
//! `#[global_allocator]` is a per-program item: exactly one may exist per
//! program, so a library that declares one decides for every binary that links
//! it. The `concinnity` facade does exactly that, behind its default-on
//! `heap-tracking` feature, so an application gets heap figures without
//! declaring anything; this crate and the others under it leave the choice
//! open, so their own test and benchmark binaries each install it:
//!
//! ```text
//! concinnity_core::install_global_allocator!();
//! ```
//!
//! It is optional. A binary without it runs correctly on Rust's default
//! allocator, and every consumer of `concinnity_core::memory::stats()` already reads
//! an `Option`: crash reports ship without heap figures, and drift detection
//! (`concinnity_engine::budget::drift`) reports nothing rather than guessing. A host embedding the
//! engine is entitled to that trade, so nothing complains about it at startup.
//!
//! The shipped binaries pin the allocator with a unit test, which is what
//! catches its removal -- see `the_shipped_player_tracks_its_own_heap` beside
//! the player binary.

// The engine's own test binary is a binary too, and unit tests and in-crate
// benchmarks read allocation counts, so it installs the allocator here.
#[cfg(test)]
concinnity_core::install_global_allocator!();

#[cfg(test)]
mod tests {
    #[test]
    fn the_test_binary_tracks_its_own_heap() {
        const MIB: usize = 1 << 20;
        let held: Vec<u8> = std::hint::black_box(vec![0; MIB]);

        let stats =
            concinnity_core::memory::stats().expect("the test binary installs the allocator");
        assert!(stats.alloc_count > 0);
        assert!(
            stats.live_bytes >= MIB as u64,
            "live bytes {} does not cover a megabyte this test is holding",
            stats.live_bytes
        );
        assert!(stats.peak_bytes >= stats.live_bytes);
        drop(held);
    }
}
