//! Derives the hash of the payload format helpers in `src/build`, which
//! concinnity-cook folds into every payload cache key. A cached payload is a
//! function of the code that produced it, and that code is split across the two
//! crates, so the half living here is published as a constant rather than read
//! out of this directory by the consumer's build script -- a registry checkout
//! of concinnity-cook has no sibling copy of these files.

use concinnity_toolchain::hash_sources;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let hash = hash_sources(&[manifest.join("src/build")]);

    let out =
        PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR")).join("build_source_hash.rs");
    std::fs::write(
        &out,
        format!(
            "/// Hash of the payload format helpers in `build`, folded into the\n\
             /// cook's payload cache key so a change here misses instead of\n\
             /// replaying bytes an older version of this code produced.\n\
             pub const BUILD_SOURCE_HASH: u32 = {hash:#010x};\n"
        ),
    )
    .expect("write build_source_hash.rs");
}
