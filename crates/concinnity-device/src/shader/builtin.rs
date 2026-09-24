// Where a built-in program's artifact comes from: the copy the build script
// embedded when it was built from this exact source, else the shader cache's,
// else a compile. Every backend fetches its built-ins through this one path.
//
// The match is on the source digest rather than on hot-reload being off, and
// that distinction is the whole point. `cn debug` and `cn editor` both run with
// hot-reload on, so a mode check would leave the two binaries a developer
// actually runs compiling every shader at startup, and needing dxc to do it.
// Comparing digests means an unedited shader takes the embedded artifact in
// every build and an edited one is recompiled in all of them.

use std::borrow::Cow;

#[cfg(any(backend_dx, backend_vk))]
use concinnity_core::platform::Platform;
use concinnity_core::render::error::RenderResult;
#[cfg(any(backend_dx, backend_vk))]
use concinnity_core::render::shader_programs::{ShaderProgram, Variant};
use concinnity_core::render::shader_source::source_digest;
#[cfg(any(backend_dx, backend_vk))]
use concinnity_shader::HlslTarget;

use super::cache::{self, Key};

/// A build script's embedded artifact: the digest of the source it was built
/// from, and its bytes.
pub(crate) type Embedded = (u64, &'static [u8]);

/// One built-in program's artifact. `embedded` is the build script's copy for
/// the program, with the digest of the source it was built from. `key` is
/// `None` where the cache cannot key the toolchain's output, which then
/// compiles every time. `label` names the program in errors and the miss log.
pub(crate) fn fetch(
    label: &str,
    source: &str,
    embedded: Option<Embedded>,
    key: Option<&Key<'_>>,
    compile: impl FnOnce() -> RenderResult<Vec<u8>>,
) -> RenderResult<Cow<'static, [u8]>> {
    if let Some((digest, bytes)) = embedded
        && digest == source_digest(source)
    {
        return Ok(Cow::Borrowed(bytes));
    }
    let compiled = match key {
        Some(key) => cache::cached(key, label, compile),
        None => compile(),
    };
    compiled.map(Cow::Owned).map_err(|e| e.context(label))
}

/// A declaration at the variant a backend compiles. A program that reads the
/// main pass's depth compiles through `ShaderProgram::at` with the host's sample
/// count; every other program has one variant and compiles as itself.
#[cfg(any(backend_dx, backend_vk))]
pub(crate) trait Program {
    fn variant(&self) -> Variant<'_>;
}

#[cfg(any(backend_dx, backend_vk))]
impl Program for ShaderProgram {
    fn variant(&self) -> Variant<'_> {
        debug_assert!(
            !self.msaa,
            "{} reads the sample count; compile it through `at`",
            self.label
        );
        self.at(false)
    }
}

#[cfg(any(backend_dx, backend_vk))]
impl Program for Variant<'_> {
    fn variant(&self) -> Variant<'_> {
        *self
    }
}

/// What one backend's built-in compile varies by: the backend the source is
/// assembled for, the target dxc emits for a program, and the build script's
/// embedded artifacts.
#[cfg(any(backend_dx, backend_vk))]
pub(crate) struct Backend {
    pub platform: Platform,
    pub target: fn(&ShaderProgram) -> HlslTarget,
    pub embedded: fn(&str) -> Option<Embedded>,
}

#[cfg(any(backend_dx, backend_vk))]
impl Backend {
    /// The exact source text `v` compiles.
    pub(crate) fn source(&self, v: Variant<'_>, hot_reload: bool) -> String {
        super::source::assemble_variant(hot_reload, self.platform, v)
    }

    /// The shader-cache key for `source`, which is what a compile the embedded
    /// artifacts missed is stored under.
    pub(crate) fn cache_key<'a>(&self, v: Variant<'_>, source: &'a str) -> Key<'a> {
        Key {
            compiler: super::compile::COMPILER_TAG,
            source,
            entry: v.program.entry,
            target: (self.target)(v.program).name(),
        }
    }

    /// `v`'s artifact, embedded, cached or compiled (see [`fetch`]).
    pub(crate) fn compile(&self, v: Variant<'_>, hot_reload: bool) -> RenderResult<Vec<u8>> {
        let source = self.source(v, hot_reload);
        let name = v.artifact_name();
        let key = self.cache_key(v, &source);
        fetch(&name, &source, (self.embedded)(&name), Some(&key), || {
            super::compile::compile(
                v.program.file,
                v.program.entry,
                &source,
                (self.target)(v.program),
            )
        })
        .map(Cow::into_owned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::render::error::RenderError;

    const SOURCE: &str = "#define A 1\nBODY\n";
    static EMBEDDED: &[u8] = b"embedded bytes";

    fn no_compile() -> RenderResult<Vec<u8>> {
        panic!("a matching embedded artifact reached the compiler")
    }

    #[test]
    fn a_matching_embedded_artifact_is_taken_without_a_compile() {
        let embedded = Some((source_digest(SOURCE), EMBEDDED));
        let bytes = fetch("p", SOURCE, embedded, None, no_compile).unwrap();
        assert!(matches!(bytes, Cow::Borrowed(b) if b == EMBEDDED));
    }

    // An edited source digests differently, so the embedded copy of the
    // unedited one must not shadow it.
    #[test]
    fn an_edited_source_compiles_instead_of_taking_the_embedded_copy() {
        let embedded = Some((source_digest("#define A 0\nBODY\n"), EMBEDDED));
        let bytes = fetch("p", SOURCE, embedded, None, || Ok(b"fresh".to_vec())).unwrap();
        assert_eq!(&*bytes, b"fresh");
    }

    #[test]
    fn a_program_with_nothing_embedded_compiles() {
        let bytes = fetch("p", SOURCE, None, None, || Ok(b"fresh".to_vec())).unwrap();
        assert!(matches!(bytes, Cow::Owned(_)));
    }

    #[test]
    fn a_failed_compile_names_the_program() {
        let err = fetch("the_program", SOURCE, None, None, || {
            Err(RenderError::ShaderCompile("bad".into()))
        })
        .unwrap_err();
        assert!(err.to_string().contains("the_program"), "{err}");
    }
}
