//! concinnity-cli: the `concinnity` dev CLI binary.
//!
//! Owns the clap command tree + dispatch (entry.rs) and the subcommand
//! implementations (cli/), driving the dev-session entry points the
//! concinnity-editor library exposes and the concinnity-cook compile pipeline.

concinnity_memory::install_global_allocator!();

mod cli;
mod entry;

fn main() -> std::io::Result<()> {
    concinnity_engine::crash::install();
    entry::run()
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
