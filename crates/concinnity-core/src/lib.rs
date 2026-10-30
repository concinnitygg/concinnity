//! concinnity-core: the `no_std` foundation every Concinnity crate shares.
//!
//! The scalar, vector and rotation math the engine is expressed in
//! ([`math`]), the allocation layer it measures and arenas its frames through
//! ([`memory`]), how a host lends its threads to work that can split
//! ([`fanout`]), and the small primitives with no better owner ([`fnv`],
//! [`utc`]). Nothing here knows about worlds, components or rendering; that
//! is `concinnity-engine`, which builds on this crate.
#![cfg_attr(
    test,
    expect(
        clippy::float_cmp,
        reason = "tests assert exact, deterministic float results"
    )
)]
#![no_std]

extern crate alloc;
#[cfg(any(test, feature = "test-support"))]
extern crate std;

#[cfg(any(test, feature = "test-support"))]
pub mod approx;
#[cfg(any(test, feature = "test-support"))]
pub mod bench;
pub mod fanout;
pub mod fnv;
pub mod math;
pub mod memory;
pub mod utc;

// The test binary counts its own heap, so the allocation benchmarks and the
// memory tests read live counters.
#[cfg(test)]
install_global_allocator!();
