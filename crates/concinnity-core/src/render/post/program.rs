//! Which single-source program a fullscreen post pass runs, and how many
//! resources that program declares.
//!
//! The binding count is declared here, beside the program identity, rather than
//! read back from the shader at runtime. Reflection is the obvious alternative
//! and the wrong one: the shipped renderer compiles no shaders (the cook emits
//! every artifact ahead of time), so asking a compiler for a layout at init would
//! put a compiler back on the shipped path.
//! A declared constant costs nothing at runtime and is still checked against the
//! source: the texture count and the probe-set declaration are scanned straight
//! out of the embedded source by this module's own tests, and the constant
//! size is pinned to the block that `shader_layout`'s reflection mirrors already
//! hold against the same shader. That keeps the single source the contract on
//! every host while leaving the shipped path compiler-free.
//!
//! Every backend compiles a [`PostProgram`] from the one shared declaration
//! [`PostProgram::program`] names, and builds its layout from the counts below,
//! so a host cannot invent a binding model the others do not share.

use crate::render::shader_programs::ShaderProgram;
use crate::render::shader_programs::shared::{SSGI_COMPOSITE, SSGI_GATHER, SSR_RESOLVE, TAA_FRAG};

/// A fullscreen post-pass fragment program. The vertex stage is always
/// `fullscreen_vertex`, which builds its triangle from the vertex id, so a
/// program names only its fragment half; a backend pairs it with its own
/// `FULLSCREEN_VERT`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PostProgram {
    /// `taa_fragment_main` from `taa.hlsl`: the temporal resolve.
    TaaResolve,
    /// `ssr_resolve_fragment` from `ssr.hlsl`: the screen-space reflection
    /// ray-march.
    SsrResolve,
    /// `ssgi_gather_fragment` from `ssgi.hlsl`: the indirect-light hemisphere
    /// gather.
    SsgiGather,
    /// `ssgi_composite_fragment` from `ssgi.hlsl`: the depth-aware blur the
    /// gathered term is blended into the scene through.
    SsgiComposite,
}

/// What a post program binds: the resource counts every backend derives its
/// descriptor layout from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PostProgramBindings {
    /// Sampled textures the fragment declares, each with a sampler of its own,
    /// so this is also the sampler count. D3D and Metal take source `i` at
    /// texture and sampler slot `i`; Vulkan binds the textures at `0..textures`
    /// and their samplers after them, at `textures..2 * textures`.
    pub textures: usize,
    /// Bytes of push / root constants the fragment declares. Zero means the pass
    /// binds no constants at all.
    pub constants: usize,
    /// Whether the fragment reads the world's reflection-probe set: the probe
    /// records plus the cube array, laid out after the declared sources, and the
    /// main camera's cluster grid that bins them (its params and lists).
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

    /// The shared declaration this program's fragment compiles from.
    pub const fn program(self) -> &'static ShaderProgram {
        match self {
            PostProgram::TaaResolve => &TAA_FRAG,
            PostProgram::SsrResolve => &SSR_RESOLVE,
            PostProgram::SsgiGather => &SSGI_GATHER,
            PostProgram::SsgiComposite => &SSGI_COMPOSITE,
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
    fn programs() -> [(PostProgram, &'static str, &'static [&'static str]); 4] {
        [
            PostProgram::TaaResolve,
            PostProgram::SsrResolve,
            PostProgram::SsgiGather,
            PostProgram::SsgiComposite,
        ]
        .map(|p| (p, p.program().file, p.program().gates))
    }

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

    // A declaration's text with every leading `[[...]]` attribute removed.
    fn without_attributes(line: &str) -> &str {
        let mut rest = line.trim();
        while let Some((_, tail)) = rest.strip_prefix("[[").and_then(|l| l.split_once("]]")) {
            rest = tail.trim();
        }
        rest
    }

    // Top-level single sampled-source declarations. A source is a texture and
    // its sampler, so the texture half (`Texture2D<...>` / `TextureCube<...>`)
    // is what counts and its `SamplerState` sibling would double the tally. A
    // texture inside a parameter list (a helper taking one) is not a
    // declaration, and neither is an array or the probe set's cube array, which
    // are not slot-indexed sources.
    fn declared_sources(lines: &[&str]) -> usize {
        const HEADS: [&str; 2] = ["Texture2D<", "TextureCube<"];
        lines
            .iter()
            .filter(|line| {
                let decl = without_attributes(line);
                HEADS.iter().any(|head| decl.starts_with(head))
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

    // Whether the kept lines declare the cluster grid's params.
    fn declares_cluster_grid(lines: &[&str]) -> bool {
        lines
            .iter()
            .any(|line| line.contains("ConstantBuffer<ClusterParams>"))
    }

    #[test]
    fn an_array_is_not_a_source() {
        assert_eq!(
            declared_sources(&["TextureCube<float4> c : register(t0);"]),
            1
        );
        assert_eq!(
            declared_sources(&[
                "[[vk::binding(8, 1)]] TextureCubeArray<float4> cubes : register(t4);"
            ]),
            0
        );
        assert_eq!(
            declared_sources(&["Texture2D<float4> pool[] : register(t0, space1);"]),
            0
        );
    }

    #[test]
    fn a_texture_sampler_pair_counts_once() {
        // A source declares a texture and a sampler; counting the sampler too
        // would double every program's texture count. A helper taking a texture
        // by parameter is not a declaration.
        let pair = [
            "[[vk::binding(0, 0)]] Texture2D<float4> scene : register(t0);",
            "[[vk::binding(1, 0)]] SamplerState scene_samp : register(s0);",
        ];
        assert_eq!(declared_sources(&pair), 1);
        assert_eq!(
            declared_sources(&["float2 size(Texture2D<float4> t) { return 0.0; }"]),
            0
        );
    }

    // The set-0 `vk::binding` numbers of the kept declarations that start with
    // one of `heads`, in declaration order.
    fn set0_bindings(lines: &[&str], heads: &[&str]) -> Vec<u32> {
        lines
            .iter()
            .filter(|line| {
                heads
                    .iter()
                    .any(|h| without_attributes(line).starts_with(h))
            })
            .filter_map(|line| {
                let args = line.split_once("vk::binding(")?.1.split_once(')')?.0;
                let (binding, set) = args.split_once(',')?;
                (set.trim() == "0").then(|| binding.trim().parse().ok())?
            })
            .collect()
    }

    #[test]
    fn every_program_binds_its_textures_then_their_samplers() {
        // Vulkan builds each program's set 0 from the texture count alone: the
        // images at 0..n, then their samplers in the same order at n..2n.
        for (program, file, defines) in programs() {
            let src = shaders::embedded(file).expect("the program's source is embedded");
            let lines = active_lines(src, defines);
            let n = program.bindings().textures as u32;
            assert_eq!(
                set0_bindings(&lines, &["Texture2D<", "TextureCube<"]),
                (0..n).collect::<Vec<_>>(),
                "{file} {defines:?}"
            );
            assert_eq!(
                set0_bindings(&lines, &["SamplerState "]),
                (n..2 * n).collect::<Vec<_>>(),
                "{file} {defines:?}"
            );
        }
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
        for (program, file, defines) in programs() {
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
        for (program, file, defines) in programs() {
            let src = shaders::embedded(file).expect("the program's source is embedded");
            assert_eq!(
                declares_probes(&active_lines(src, defines)),
                program.bindings().probes,
                "{file} {defines:?} disagrees with {program:?} about the probe set"
            );
        }
    }

    #[test]
    fn a_probe_reading_program_declares_the_cluster_grid() {
        // Every backend binds the cluster grid beside the probe set, so the two
        // declarations travel together.
        for (program, file, defines) in programs() {
            let src = shaders::embedded(file).expect("the program's source is embedded");
            assert_eq!(
                declares_cluster_grid(&active_lines(src, defines)),
                program.bindings().probes,
                "{file} {defines:?} disagrees with {program:?} about the cluster grid"
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
        let programs = programs();
        for (i, (a, ..)) in programs.iter().enumerate() {
            for (b, ..) in &programs[i + 1..] {
                assert_ne!(a.label(), b.label());
            }
        }
    }
}
