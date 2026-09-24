// The runtime half of the single-source shader compile: one call into dxc per
// target, and spirv-cross after it on the Metal leg.
//
// Every backend reaches the compiler the same way -- assemble the source, then
// hand it to dxc -- so the call is made once, here, rather than three times.
// The build script makes the same call in
// `concinnity_toolchain::shader_artifacts`, and the two must agree: an artifact
// the build embedded is taken whenever its source digest matches, so a runtime
// compile of the same text under different flags would be a different shader
// under the same name.

use concinnity_core::platform::Platform;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_shader::HlslTarget;

/// What the content-addressed shader cache records as the toolchain behind an
/// engine SPIR-V or DXIL artifact. A metallib is keyed on the Metal toolchain's
/// release instead, which the cache segment's dxc stamp does not cover.
#[cfg(any(backend_vk, backend_dx))]
pub(crate) const COMPILER_TAG: &str = "hlsl";

/// The shader toolchain this binary reaches, for the cache's whole-segment
/// invalidation: an upgraded dxc discards every entry an older one wrote.
pub(crate) fn toolchain_id() -> &'static str {
    concinnity_shader::compiler_id()
}

/// One entry point of `source`, compiled to `target`. A Metal library holds
/// exactly one entry, since a variant binds only the resources it reads.
pub(crate) fn compile(
    file: &str,
    entry: &str,
    source: &str,
    target: HlslTarget,
) -> RenderResult<Vec<u8>> {
    let work = crate::shader::compiler_work::dir().map_err(RenderError::Other)?;
    concinnity_shader::compile(&job(file, entry, source, target), work.path())
        .map_err(RenderError::ShaderCompile)
}

/// One entry point of `source` in the form the cook stores for `platform`
/// ([`HlslTarget::cooked`]), for a world's shader whose cooked artifact no
/// longer matches the engine template it was built against. The world's text
/// is authored, so a dxc warning is logged as the cook logs it rather than
/// failing the compile.
pub(crate) fn cooked(
    platform: Platform,
    file: &str,
    entry: &str,
    source: &str,
) -> RenderResult<Vec<u8>> {
    let work = crate::shader::compiler_work::dir().map_err(RenderError::Other)?;
    let target = HlslTarget::cooked(platform);
    let compiled =
        concinnity_shader::compile_with_warnings(&job(file, entry, source, target), work.path())
            .map_err(RenderError::ShaderCompile)?;
    if let Some(warnings) = compiled.warnings {
        tracing::warn!("world shader {file}: compiling '{entry}':\n{warnings}");
    }
    Ok(compiled.artifact)
}

fn job<'a>(
    file: &'a str,
    entry: &'a str,
    source: &'a str,
    target: HlslTarget,
) -> concinnity_shader::HlslJob<'a> {
    concinnity_shader::HlslJob {
        source,
        file_name: file,
        entry,
        target,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // An upgraded dxc must miss, so the id carries its release.
    #[test]
    fn the_toolchain_id_names_the_compiler() {
        assert!(toolchain_id().starts_with("dxc"), "{}", toolchain_id());
    }
}
