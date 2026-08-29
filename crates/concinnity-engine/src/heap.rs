// src/heap.rs
//
// The tracked heap the engine reports on.
//
// `#[global_allocator]` is a per-program item, so it belongs to the binary
// rather than to a library in its dependency graph. Exactly one may exist per
// program, so a library that declares one decides for every binary that links
// it, and forbids any of them from choosing another -- including the test and
// benchmark binaries that link the engine to measure it. A Concinnity binary
// that wants heap figures therefore installs the tracking allocator itself, at
// its crate root:
//
//     concinnity_core::install_global_allocator!();
//
// It is optional. A binary without it runs correctly on Rust's default
// allocator, and every consumer of `concinnity_core::memory::stats()` already reads
// an `Option`: crash reports ship without heap figures, and drift detection
// (`app::mem_drift`) reports nothing rather than guessing. A host embedding the
// engine is entitled to that trade, so nothing complains about it at startup.
//
// The binaries that do report heap figures each pin their own declaration with
// a unit test, which is what catches its removal -- see
// `the_shipped_player_tracks_its_own_heap` beside the player binary.

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
