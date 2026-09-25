// Hot-reload's half of the single-source shader assembly.
//
// The assembly itself lives in `concinnity_core::render::shader_source`, below the
// backends, so this crate's runtime compile and the build script's precompile
// cannot key different text for the same program. What stays here is the one
// part that needs a filesystem: under hot-reload a shader's checkout copy wins
// over the embedded one, so an edit is picked up without a rebuild.

use concinnity_core::platform::Platform;
use concinnity_core::render::shader_programs::Variant;
use concinnity_core::render::shader_source::Splice;

/// The exact source text a program compiles for `platform`. `file` names the
/// shader under core's `src/render/shaders/`; under `hot_reload` its checkout
/// copy is preferred, for the shader and for every fragment spliced into it.
/// `splices` fills markers no file in the tree can, which today is only a
/// world's own distance field.
pub(crate) fn assemble(
    hot_reload: bool,
    platform: Platform,
    file: &str,
    defines: &[(&str, &str)],
    splices: &[Splice<'_>],
) -> String {
    let resolve = |f: &str| hot_reload.then(|| from_checkout(f)).flatten();
    concinnity_core::render::shader_source::assemble_with_splices(
        file, platform, defines, resolve, splices,
    )
}

/// The exact source text variant `v` compiles for `platform`, preferring the
/// checkout's copies under `hot_reload`.
pub(crate) fn assemble_variant(hot_reload: bool, platform: Platform, v: Variant<'_>) -> String {
    assemble(hot_reload, platform, v.program.file, &v.defines(), &[])
}

// The checkout's copy of `file`, leaked so it can join the embedded texts under
// one `&'static str` resolver. Hot-reload is a development path that recompiles
// a bounded set of shaders on edit, so the leak is bounded by how many distinct
// shader revisions one session touches.
pub(crate) fn from_checkout(file: &str) -> Option<&'static str> {
    let path = format!(
        "{}/../concinnity-core/src/render/shaders/{}",
        env!("CARGO_MANIFEST_DIR"),
        file
    );
    match std::fs::read_to_string(&path) {
        Ok(text) => Some(Box::leak(text.into_boxed_str())),
        Err(e) => {
            tracing::debug!("hot-reload: falling back to embedded {file} ({e})");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hot-reload off is the shipped path: the embedded copy, fragments spliced,
    // defines leading.
    #[test]
    fn the_embedded_body_is_assembled_with_defines_leading() {
        let src = assemble(false, Platform::Vulkan, "fog.hlsl", &[("A", "1")], &[]);
        assert!(src.starts_with("#define CN_BACKEND_VULKAN 1\n#define A 1\n"));
        assert!(!src.contains("{POST_COMMON}"));
    }

    // A hot-reload read that finds nothing must not lose the program: the
    // embedded copy stands in, so a shipped binary with hot-reload on still
    // compiles the shader it shipped with.
    #[test]
    fn a_missing_hot_reload_file_falls_back_to_embedded() {
        assert_eq!(
            assemble(
                true,
                Platform::Metal,
                "definitely_not_a_shader.hlsl",
                &[("A", "1")],
                &[]
            ),
            "#define CN_BACKEND_METAL 1\n#define A 1\n"
        );
    }

    // The two paths agree for a shader whose checkout copy is unchanged, which
    // is what keeps a hot-reload session sharing cache artifacts with a normal
    // run rather than recompiling everything on the first frame.
    #[test]
    fn hot_reload_matches_the_embedded_assembly_for_an_unedited_shader() {
        assert_eq!(
            assemble(true, Platform::DirectX, "fog.hlsl", &[("A", "1")], &[]),
            assemble(false, Platform::DirectX, "fog.hlsl", &[("A", "1")], &[])
        );
    }
}
