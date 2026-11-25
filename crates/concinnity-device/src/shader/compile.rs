// The runtime half of the single-source shader compile: one call into dxc per
// target, and spirv-cross after it on the Metal leg.
//
// Every backend reaches the compiler the same way -- assemble the source, then
// hand it to dxc -- so the call is made once, here, rather than three times.
// An export's shipped artifacts come from this same call, which is what lets a
// player take one whenever its source digest matches.

use concinnity_core::platform::Platform;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_shader::HlslTarget;

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
