// src/heap.rs
//
// The process allocator every Concinnity binary runs on.
//
// The engine owns allocation policy rather than leaving it to whichever binary
// links it. That is the AAA position and the one this engine is built for: a
// console budget is fixed, cert is real, and the engine's own streaming already
// decides residency from byte budgets in production. An allocator chosen
// per-binary would mean the shipped player, the dev CLI and every example could
// each be measuring something different, and a new binary could silently ship
// measuring nothing.
//
// `#[global_allocator]` is a link-time item: a binary gets this by depending on
// the engine, with nothing to opt into and nothing to forget. Exactly one may
// exist per program, so no binary in the graph may declare its own.
//
// `TrackingAlloc` is generic over what it wraps, which is what makes this the
// right home rather than merely a workable one: the day there is an in-house
// allocator to wrap, the change is the type argument below and every binary
// picks it up.

#[global_allocator]
static ALLOC: concinnity_memory::TrackingAlloc<std::alloc::System> =
    concinnity_memory::TrackingAlloc::new(std::alloc::System);

#[cfg(test)]
mod tests {
    // Linking the engine is what installs the allocator, so this holds for any
    // binary that does -- the shipped player included, which is where it
    // matters and where nothing else could assert it.
    #[test]
    fn linking_the_engine_installs_the_tracking_allocator() {
        const MIB: usize = 1 << 20;
        let held: Vec<u8> = vec![0; MIB];

        let stats = concinnity_memory::stats().expect("the engine installs the allocator");
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
