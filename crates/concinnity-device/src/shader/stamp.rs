//! What a compiled engine shader artifact is a function of beyond its source
//! text: the code that compiles it and the shader set it was assembled from.

use concinnity_core::fnv::Fnv1a;
use concinnity_engine::render::shader::embedded::SOURCES_HASH;

// SHADER_COMPILE_SOURCE_HASH: derived by build.rs from the modules that decide
// how an artifact is produced.
include!(concat!(env!("OUT_DIR"), "/shader_compile_source_hash.rs"));

/// What the content-addressed shader cache records as the toolchain behind an
/// engine SPIR-V or DXIL artifact. A metallib is keyed on the Metal toolchain's
/// release instead, which the cache segment's dxc stamp does not cover.
pub(crate) const COMPILER_TAG: &str = "hlsl";

/// Names everything a shipped shader file must match to be served: the compile
/// code here, the compile flags in `concinnity-shader`, the compiler tag and the
/// embedded engine shaders. An export compiled by a `concinnity` whose stamp
/// differs from the player's is refused.
pub const SHIPPED_SHADER_STAMP: u64 = {
    let mut hash = Fnv1a::new();
    hash.write(&SHADER_COMPILE_SOURCE_HASH.to_le_bytes());
    hash.write(&concinnity_shader::SOURCE_HASH.to_le_bytes());
    hash.write(COMPILER_TAG.as_bytes());
    hash.write(&SOURCES_HASH.to_le_bytes());
    hash.finish()
};
