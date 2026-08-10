// src/heap.rs
//
// The tracked heap the engine reports on, and the check that the binary
// running it actually installed one.
//
// `#[global_allocator]` is a per-program item, so it belongs to the binary
// rather than to a library in its dependency graph. Exactly one may exist per
// program, so a library that declares one decides for every binary that links
// it, and forbids any of them from choosing another -- including the test and
// benchmark binaries that link the engine to measure it. Each Concinnity
// binary therefore installs the tracking allocator itself, at its crate root:
//
//     concinnity_memory::install_global_allocator!();
//
// Nothing forces that declaration at compile time, and a binary without it
// runs correctly on Rust's default allocator while reporting no memory at all:
// crash reports lose their heap figures, and long-session drift detection
// (`app::mem_drift`) goes dark, since it cannot separate heap growth from
// growth outside the heap without a heap figure. `verify_installed` turns that
// silent loss into a warning at startup.

// Warn when the running binary installed no tracking allocator. Called from
// `app::run::init_logging`, which every host runs before anything else; a
// process has allocated many times by then, so `None` means "not installed"
// rather than "nothing allocated yet".
pub(crate) fn verify_installed() {
    if concinnity_memory::stats().is_none() {
        tracing::warn!(
            "this binary installed no tracking #[global_allocator]: crash reports \
             carry no heap figures and memory drift detection is off. Declare \
             concinnity_memory::TrackingAlloc in the binary's main.rs."
        );
    }
}

// The engine's own test binary is a binary too, and unit tests and in-crate
// benchmarks read allocation counts, so it installs the allocator here.
#[cfg(test)]
concinnity_memory::install_global_allocator!();

#[cfg(test)]
mod tests {
    #[test]
    fn the_test_binary_tracks_its_own_heap() {
        const MIB: usize = 1 << 20;
        let held: Vec<u8> = std::hint::black_box(vec![0; MIB]);

        let stats = concinnity_memory::stats().expect("the test binary installs the allocator");
        assert!(stats.alloc_count > 0);
        assert!(
            stats.live_bytes >= MIB as u64,
            "live bytes {} does not cover a megabyte this test is holding",
            stats.live_bytes
        );
        assert!(stats.peak_bytes >= stats.live_bytes);
        drop(held);
    }

    // The startup check passes here precisely because the block above ran, so
    // it exercises the installed path; the uninstalled path is what every
    // binary's own allocator test guards.
    #[test]
    fn verify_installed_accepts_a_tracked_binary() {
        super::verify_installed();
        assert!(concinnity_memory::stats().is_some());
    }
}
