//! Which single-source program a fullscreen post pass runs, and how many
//! resources that program declares.
//!
//! The binding count is declared here, beside the program identity, rather than
//! read back from the shader at runtime. Reflection is the obvious alternative
//! and the wrong one: the shipped renderer compiles no shaders (the cook emits
//! every artifact ahead of time), so asking slangc for a layout at init would
//! reintroduce the runtime compiler this engine spent the shader arc removing.
//! A declared constant costs nothing at runtime and is still checked against the
//! source: the texture count is scanned straight out of the embedded `.slang`
//! by this module's own tests, and the constant size is pinned to the block that
//! `shader_layout`'s reflection mirrors already hold against the same shader.
//! That keeps the single source the contract on every host while leaving the
//! shipped path compiler-free.
//!
//! Each backend maps a [`PostProgram`] to its own program table entry
//! (`{vulkan,directx}/slang_builtins.rs`, `metal/slang_builtins.rs`); the counts
//! below are what all three build their layouts from, so a host cannot invent a
//! binding model the others do not share.

/// A fullscreen post-pass fragment program. The vertex stage is always the one
/// shared `fullscreen_vertex`, which builds its triangle from the vertex id, so
/// a program names only its fragment half.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PostProgram {
    /// `taa_fragment_main` from `taa.slang`: the temporal resolve.
    TaaResolve,
}

/// What a post program binds: the resource counts every backend derives its
/// descriptor layout from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PostProgramBindings {
    /// Sampled textures the fragment declares, at slots `0..textures`. Each is a
    /// combined `Sampler2D` in the source, which slangc lowers to a texture plus
    /// a sampler at the same index, so this is also the sampler count.
    pub textures: usize,
    /// Bytes of push / root constants the fragment declares. Zero means the pass
    /// binds no constants at all.
    pub constants: usize,
}

impl PostProgram {
    /// The pass name a backend reports when this program's pipeline fails to
    /// build.
    pub const fn label(self) -> &'static str {
        match self {
            PostProgram::TaaResolve => "taa resolve",
        }
    }

    /// The resource counts this program declares.
    pub const fn bindings(self) -> PostProgramBindings {
        match self {
            // scene, velocity, history; a single `float history_valid`.
            PostProgram::TaaResolve => PostProgramBindings {
                textures: 3,
                constants: 4,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::shaders;
    use crate::render::uniforms::TaaParams;

    // Top-level `Sampler2D<...> name;` declarations in a shader's source: the
    // sampled sources it binds. A `Sampler2D` inside a parameter list (a helper
    // taking one) is not a declaration, so a match must run from the start of a
    // statement to its semicolon with no call parenthesis in between.
    fn declared_sources(src: &str) -> usize {
        src.lines()
            .map(|line| line.trim())
            .filter(|line| {
                let Some(rest) = line.strip_prefix("[[").and_then(|l| l.split_once("]]")) else {
                    return line.starts_with("Sampler2D<") && line.ends_with(';');
                };
                let decl = rest.1.trim();
                decl.starts_with("Sampler2D<") && decl.ends_with(';')
            })
            .count()
    }

    #[test]
    fn a_helper_taking_a_sampler_is_not_a_declaration() {
        // Negative control: the shared post helpers take a `Sampler2D` by
        // parameter, and counting those would inflate every program's count.
        assert_eq!(
            declared_sources("void combined_dims(Sampler2D<float4> s, out uint w)\n{\n}\n"),
            0
        );
        assert_eq!(
            declared_sources("[[vk::binding(0, 0)]] Sampler2D<float4> t;"),
            1
        );
        assert_eq!(declared_sources("Sampler2D<float4> t;"), 1);
    }

    #[test]
    fn the_declared_texture_count_matches_the_shader_source() {
        // The declaration this table carries is what all three backends build
        // their descriptor layouts from, so it has to be the source's count and
        // not a number somebody typed.
        let (program, file) = (PostProgram::TaaResolve, "taa.slang");
        let src = shaders::embedded(file).expect("the program's source is embedded");
        assert_eq!(
            declared_sources(src),
            program.bindings().textures,
            "{file} declares a different number of sources than {program:?} does"
        );
    }

    #[test]
    fn the_declared_constant_size_matches_the_uploaded_block() {
        // The other half of the contract. The block itself is pinned to the
        // shader by `shader_layout`'s reflection mirrors, so pinning the
        // declaration to the block reaches the source through them.
        assert_eq!(
            PostProgram::TaaResolve.bindings().constants,
            core::mem::size_of::<TaaParams>()
        );
    }
}
