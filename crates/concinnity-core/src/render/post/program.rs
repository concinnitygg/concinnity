//! Which single-source program a fullscreen post pass runs, and how many
//! resources that program declares.
//!
//! The binding count is declared here, beside the program identity, rather than
//! read back from the shader at runtime. Reflection is the obvious alternative
//! and the wrong one: the shipped renderer compiles no shaders (the cook emits
//! every artifact ahead of time), so asking slangc for a layout at init would
//! reintroduce the runtime compiler this engine spent the shader arc removing.
//! A declared constant costs nothing at runtime and is still checked against the
//! source: the texture count and the probe-set declaration are scanned straight
//! out of the embedded `.slang` by this module's own tests, and the constant
//! size is pinned to the block that `shader_layout`'s reflection mirrors already
//! hold against the same shader. That keeps the single source the contract on
//! every host while leaving the shipped path compiler-free.
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
    /// `ssr_resolve_fragment` from `ssr.slang`: the screen-space reflection
    /// ray-march.
    SsrResolve,
    /// `ssgi_gather_fragment` from `ssgi.slang`: the indirect-light hemisphere
    /// gather.
    SsgiGather,
    /// `ssgi_composite_fragment` from `ssgi.slang`: the depth-aware blur the
    /// gathered term is blended into the scene through.
    SsgiComposite,
}

/// What a post program binds: the resource counts every backend derives its
/// descriptor layout from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PostProgramBindings {
    /// Sampled textures the fragment declares, at slots `0..textures`. Each is a
    /// combined `Sampler2D` or `SamplerCube` in the source, which slangc lowers
    /// to a texture plus a sampler at the same index, so this is also the
    /// sampler count.
    pub textures: usize,
    /// Bytes of push / root constants the fragment declares. Zero means the pass
    /// binds no constants at all.
    pub constants: usize,
    /// Whether the fragment reads the world's reflection-probe set: the probe
    /// records plus the cube array, laid out after the declared sources.
    pub probes: bool,
}

impl PostProgram {
    /// The pass name a backend reports when this program's pipeline fails to
    /// build.
    pub const fn label(self) -> &'static str {
        match self {
            PostProgram::TaaResolve => "taa resolve",
            PostProgram::SsrResolve => "ssr resolve",
            PostProgram::SsgiGather => "ssgi gather",
            PostProgram::SsgiComposite => "ssgi composite",
        }
    }

    /// The resource counts this program declares.
    pub const fn bindings(self) -> PostProgramBindings {
        match self {
            // scene, velocity, history; a single `float history_valid`.
            PostProgram::TaaResolve => PostProgramBindings {
                textures: 3,
                constants: 4,
                probes: false,
            },
            // scene, normal+depth, roughness, prefilter cube; `SsrParams`.
            PostProgram::SsrResolve => PostProgramBindings {
                textures: 4,
                constants: 144,
                probes: true,
            },
            // scene (gather) or gathered term (composite), normal+depth;
            // `SsgiParams`.
            PostProgram::SsgiGather | PostProgram::SsgiComposite => PostProgramBindings {
                textures: 2,
                constants: 32,
                probes: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::render_types::{SsgiParams, SsrParams};
    use crate::render::shaders;
    use crate::render::uniforms::TaaParams;
    use alloc::vec::Vec;

    // Every program, the source it compiles from, and the defines that select
    // its variant.
    const PROGRAMS: [(PostProgram, &str, &[&str]); 4] = [
        (PostProgram::TaaResolve, "taa.slang", &[]),
        (PostProgram::SsrResolve, "ssr.slang", &[]),
        (PostProgram::SsgiGather, "ssgi.slang", &["SSGI_GATHER"]),
        (
            PostProgram::SsgiComposite,
            "ssgi.slang",
            &["SSGI_COMPOSITE"],
        ),
    ];

    // The source lines a compile with `defines` keeps: `#if defined(..)`,
    // `#elif defined(..)`, `#ifdef`, `#ifndef`, `#else` and `#endif` are
    // evaluated, so a variant's declarations are counted and its siblings' are
    // not. A macro the shader defines for itself counts as undefined, which is
    // what every guard in the post sources tests for.
    fn active_lines<'s>(src: &'s str, defines: &[&str]) -> Vec<&'s str> {
        // Per open conditional: whether its enclosing region is live, whether a
        // branch has been taken, and whether the current branch is live.
        let mut stack: Vec<(bool, bool, bool)> = Vec::new();
        let defined = |cond: &str| {
            let name = cond
                .trim()
                .trim_start_matches("defined")
                .trim_matches(|c| c == '(' || c == ')' || c == ' ');
            defines.contains(&name)
        };
        let live = |stack: &[(bool, bool, bool)]| stack.last().is_none_or(|s| s.2);
        let mut out = Vec::new();
        for line in src.lines().map(str::trim) {
            if let Some(cond) = line.strip_prefix("#if ") {
                let parent = live(&stack);
                let taken = parent && defined(cond);
                stack.push((parent, taken, taken));
            } else if let Some(name) = line.strip_prefix("#ifdef ") {
                let parent = live(&stack);
                let taken = parent && defined(name);
                stack.push((parent, taken, taken));
            } else if let Some(name) = line.strip_prefix("#ifndef ") {
                let parent = live(&stack);
                let taken = parent && !defined(name);
                stack.push((parent, taken, taken));
            } else if let Some(cond) = line.strip_prefix("#elif ") {
                if let Some(top) = stack.last_mut() {
                    let now = top.0 && !top.1 && defined(cond);
                    top.1 |= now;
                    top.2 = now;
                }
            } else if line.starts_with("#else") {
                if let Some(top) = stack.last_mut() {
                    top.2 = top.0 && !top.1;
                    top.1 = true;
                }
            } else if line.starts_with("#endif") {
                stack.pop();
            } else if live(&stack) {
                out.push(line);
            }
        }
        out
    }

    // Top-level single sampled-source declarations: `Sampler2D<...> name;` or
    // `SamplerCube<...> name;`, optionally behind a `[[...]]` attribute. A
    // sampler inside a parameter list (a helper taking one) is not a
    // declaration, and neither is an array, which is a probe set's cubes rather
    // than a slot-indexed source.
    fn declared_sources(lines: &[&str]) -> usize {
        lines
            .iter()
            .filter(|line| {
                let decl = match line.strip_prefix("[[").and_then(|l| l.split_once("]]")) {
                    Some((_, rest)) => rest.trim(),
                    None => line,
                };
                (decl.starts_with("Sampler2D<") || decl.starts_with("SamplerCube<"))
                    && decl.ends_with(';')
                    && !decl.contains('[')
            })
            .count()
    }

    // Whether the kept lines declare the reflection-probe records.
    fn declares_probes(lines: &[&str]) -> bool {
        lines
            .iter()
            .any(|line| line.contains("ConstantBuffer<ProbeSet>"))
    }

    #[test]
    fn a_helper_taking_a_sampler_is_not_a_declaration() {
        // Negative control: the shared post helpers take a `Sampler2D` by
        // parameter, and counting those would inflate every program's count.
        assert_eq!(
            declared_sources(&["void combined_dims(Sampler2D<float4> s, out uint w)"]),
            0
        );
        assert_eq!(
            declared_sources(&["[[vk::binding(0, 0)]] Sampler2D<float4> t;"]),
            1
        );
        assert_eq!(declared_sources(&["SamplerCube<float4> c;"]), 1);
        assert_eq!(
            declared_sources(&["[[vk::binding(8, 1)]] SamplerCube<float4> cubes[MAX_PROBES];"]),
            0
        );
    }

    #[test]
    fn only_the_selected_variant_is_kept() {
        let src = "#if defined(A)\na;\n#elif defined(B)\nb;\n#else\nc;\n#endif\nd;";
        assert_eq!(active_lines(src, &["A"]), ["a;", "d;"]);
        assert_eq!(active_lines(src, &["B"]), ["b;", "d;"]);
        assert_eq!(active_lines(src, &[]), ["c;", "d;"]);
        let nested = "#ifndef X\n#if defined(A)\na;\n#endif\nn;\n#endif";
        assert_eq!(active_lines(nested, &["X", "A"]), Vec::<&str>::new());
        assert_eq!(active_lines(nested, &["A"]), ["a;", "n;"]);
    }

    #[test]
    fn the_declared_texture_count_matches_the_shader_source() {
        // The declaration this table carries is what all three backends build
        // their descriptor layouts from, so it has to be the source's count and
        // not a number somebody typed.
        for (program, file, defines) in PROGRAMS {
            let src = shaders::embedded(file).expect("the program's source is embedded");
            let lines = active_lines(src, defines);
            assert_eq!(
                declared_sources(&lines),
                program.bindings().textures,
                "{file} {defines:?} declares a different number of sources than {program:?} does"
            );
        }
    }

    #[test]
    fn the_declared_probe_set_matches_the_shader_source() {
        for (program, file, defines) in PROGRAMS {
            let src = shaders::embedded(file).expect("the program's source is embedded");
            assert_eq!(
                declares_probes(&active_lines(src, defines)),
                program.bindings().probes,
                "{file} {defines:?} disagrees with {program:?} about the probe set"
            );
        }
    }

    #[test]
    fn the_declared_constant_size_matches_the_uploaded_block() {
        // The other half of the contract. The blocks themselves are pinned to
        // the shaders by `shader_layout`'s reflection mirrors, so pinning the
        // declaration to the block reaches the source through them.
        use core::mem::size_of;
        let blocks = [
            (PostProgram::TaaResolve, size_of::<TaaParams>()),
            (PostProgram::SsrResolve, size_of::<SsrParams>()),
            (PostProgram::SsgiGather, size_of::<SsgiParams>()),
            (PostProgram::SsgiComposite, size_of::<SsgiParams>()),
        ];
        for (program, size) in blocks {
            assert_eq!(program.bindings().constants, size, "{program:?}");
        }
    }

    #[test]
    fn every_program_has_a_distinct_label() {
        for (i, (a, ..)) in PROGRAMS.iter().enumerate() {
            for (b, ..) in &PROGRAMS[i + 1..] {
                assert_ne!(a.label(), b.label());
            }
        }
    }
}
