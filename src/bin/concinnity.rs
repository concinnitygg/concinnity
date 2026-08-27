//! concinnity: the dev CLI binary.
//!
//! The command tree, its dispatch, and every subcommand live in the
//! concinnity-cli library. This target is the binary that links them, and the
//! one place the tracking allocator is declared -- a global allocator belongs to
//! a final artifact, not to a library every consumer would inherit it from.

concinnity_memory::install_global_allocator!();

fn main() -> std::io::Result<()> {
    concinnity_cli::run()
}

#[cfg(test)]
mod tests {
    // `cn debug` and `cn editor` report heap figures through the editor's health
    // output, which the probe harness reads. Nothing forces the declaration at
    // the top of this file to exist, so this is what catches its removal:
    // without it the CLI would run correctly while reporting no memory at all.
    #[test]
    fn the_dev_cli_tracks_its_own_heap() {
        const MIB: usize = 1 << 20;

        let before = concinnity_memory::stats()
            .expect("this binary declares the tracking allocator")
            .alloc_count;
        let held: Vec<u8> = core::hint::black_box(vec![0; MIB]);
        let after = concinnity_memory::stats().expect("the allocator stays installed");

        assert!(
            after.alloc_count > before,
            "allocation count did not move ({before} -> {}) across a megabyte",
            after.alloc_count
        );
        assert!(after.peak_bytes >= after.live_bytes);
        drop(core::hint::black_box(held));
    }
}
