// Where a built-in program's artifact comes from: the copy an export shipped
// when it was compiled from this exact source, else the shader cache's, else a
// compile, which the cache then keeps for the next launch. Every backend
// fetches its built-ins through this one path.
//
// The match is on the source digest rather than on hot-reload being off: an
// unedited shader takes the shipped artifact whichever way it was assembled,
// and an edited one is recompiled.

use std::borrow::Cow;

#[cfg(any(backend_dx, backend_vk))]
use concinnity_core::platform::Platform;
use concinnity_engine::render::backend::error::RenderResult;
#[cfg(any(backend_dx, backend_vk))]
use concinnity_engine::render::shader::programs::{ShaderProgram, Variant};
use concinnity_engine::render::shader::source::source_digest;
#[cfg(any(backend_dx, backend_vk))]
use concinnity_shader::HlslTarget;

use super::cache::{self, Key};

use super::shipped::Shipped;
#[cfg(any(backend_dx, backend_vk))]
use super::shipped::{Compile, Planned};

/// One built-in program's artifact. `shipped` is an export's copy for the
/// program, with the digest of the source it was compiled from. `key` is
/// `None` where the cache cannot key the toolchain's output, which then
/// compiles every time. `label` names the program in errors and the miss log.
pub(crate) fn fetch(
    label: &str,
    source: &str,
    shipped: Option<Shipped>,
    key: Option<&Key<'_>>,
    compile: impl FnOnce() -> RenderResult<Vec<u8>>,
) -> RenderResult<Cow<'static, [u8]>> {
    if let Some((digest, bytes)) = shipped
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
/// assembled for, and the target dxc emits for a program.
#[cfg(any(backend_dx, backend_vk))]
pub(crate) struct Backend {
    pub platform: Platform,
    pub target: fn(&ShaderProgram) -> HlslTarget,
}

#[cfg(any(backend_dx, backend_vk))]
impl Backend {
    /// The exact source text `v` compiles.
    pub(crate) fn source(&self, v: Variant<'_>, hot_reload: bool) -> String {
        super::source::assemble_variant(hot_reload, self.platform, v)
    }

    /// The shader-cache key for `source`, which is what a compile the shipped
    /// artifacts missed is stored under.
    pub(crate) fn cache_key<'a>(&self, v: Variant<'_>, source: &'a str) -> Key<'a> {
        Key {
            compiler: super::stamp::COMPILER_TAG,
            source,
            entry: v.program.entry,
            target: (self.target)(v.program).name(),
        }
    }

    /// `v` as a player fetches it: its artifact name, the source it compiles
    /// and how.
    pub(crate) fn planned(&self, v: Variant<'_>, hot_reload: bool) -> Planned {
        Planned {
            name: v.artifact_name(),
            source: self.source(v, hot_reload),
            compile: Compile::Hlsl {
                file: v.program.file,
                entry: v.program.entry,
                target: (self.target)(v.program),
            },
        }
    }

    /// `v`'s artifact, shipped, cached or compiled (see [`fetch`]).
    pub(crate) fn compile(&self, v: Variant<'_>, hot_reload: bool) -> RenderResult<Vec<u8>> {
        let planned = self.planned(v, hot_reload);
        let key = self.cache_key(v, &planned.source);
        fetch(
            &planned.name,
            &planned.source,
            planned.shipped(self.platform),
            Some(&key),
            || planned.compile(),
        )
        .map(Cow::into_owned)
    }

    /// Every variant `table` declares, as a player fetches it, for an export
    /// to compile and ship.
    pub(crate) fn plan(
        &self,
        table: &concinnity_engine::render::shader::programs::Table,
    ) -> Vec<Planned> {
        table.variants().map(|v| self.planned(v, false)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::render::backend::error::RenderError;

    const SOURCE: &str = "#define A 1\nBODY\n";
    static SHIPPED: &[u8] = b"shipped bytes";

    fn no_compile() -> RenderResult<Vec<u8>> {
        panic!("a matching shipped artifact reached the compiler")
    }

    #[test]
    fn a_matching_shipped_artifact_is_taken_without_a_compile() {
        let shipped = Some((source_digest(SOURCE), SHIPPED));
        let bytes = fetch("p", SOURCE, shipped, None, no_compile).unwrap();
        assert!(matches!(bytes, Cow::Borrowed(b) if b == SHIPPED));
    }

    // An edited source digests differently, so the shipped copy of the
    // unedited one must not shadow it.
    #[test]
    fn an_edited_source_compiles_instead_of_taking_the_shipped_copy() {
        let shipped = Some((source_digest("#define A 0\nBODY\n"), SHIPPED));
        let bytes = fetch("p", SOURCE, shipped, None, || Ok(b"fresh".to_vec())).unwrap();
        assert_eq!(&*bytes, b"fresh");
    }

    #[test]
    fn a_program_with_nothing_shipped_compiles() {
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
