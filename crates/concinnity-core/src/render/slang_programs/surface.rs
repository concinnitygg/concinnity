//! What a world `Shader` compiles to, on every backend.
//!
//! A world Shader defines two hooks, `transform` and `shade`, and the engine's
//! own main-pass entries call them. So a world shader compiles as the engine's
//! main-pass programs do, from `main_bindless.slang`, with the world's files
//! spliced at the hook markers in place of the engine's defaults. The cook
//! iterates this table to compile a Shader ahead of time and each renderer
//! iterates it to find what the cook left.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::platform::Platform;
use crate::render::slang_source;
use crate::render::uniforms::{BINDLESS_POOL_SIZE, MAX_PROBES};

/// The marker the world's `vertex` file is spliced at.
pub const VERTEX_MARKER: &str = "{SURFACE_VERTEX}";
/// The marker the world's `fragment` file is spliced at.
pub const FRAGMENT_MARKER: &str = "{SURFACE_FRAGMENT}";

/// Whether an entry runs at the vertex or the fragment stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Places a vertex; carries the world's `transform`.
    Vertex,
    /// Shades a surface; carries the world's `shade`.
    Fragment,
}

/// One entry point of one main-pass file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Program {
    /// The shader file the entry compiles from.
    pub file: &'static str,
    /// Entry point name, as the source spells it.
    pub entry: &'static str,
    /// Which stage it compiles for.
    pub stage: Stage,
}

/// Every entry a world Shader compiles: the bindless pair, on every host.
pub const ALL: &[Program] = &[
    Program {
        file: "main_bindless.slang",
        entry: "vertex_main_bindless",
        stage: Stage::Vertex,
    },
    Program {
        file: "main_bindless.slang",
        entry: "fragment_main_bindless",
        stage: Stage::Fragment,
    },
];

/// The entry named `entry`.
pub fn program(entry: &str) -> Option<&'static Program> {
    ALL.iter().find(|p| p.entry == entry)
}

/// The entries a host compiles for a world Shader.
pub fn programs(_platform: Platform) -> impl Iterator<Item = &'static Program> {
    ALL.iter()
}

/// How a host's entries group into artifacts. Metal takes the pair as one
/// library: slangc emits one MSL translation unit and the runtime wants one
/// library to pull both functions out of. The other two hosts take one
/// artifact per entry.
pub fn groups(platform: Platform) -> Vec<Vec<&'static Program>> {
    if platform == Platform::Metal {
        return alloc::vec![ALL.iter().collect()];
    }
    ALL.iter().map(|p| alloc::vec![p]).collect()
}

/// The world's two files, as text.
#[derive(Debug, Clone, Copy)]
pub struct Sources<'a> {
    /// The `vertex` file, when declared.
    pub vertex: Option<&'a str>,
    /// The `fragment` file.
    pub fragment: &'a str,
}

impl<'a> Sources<'a> {
    /// The splices that put the declared files in place of the engine's
    /// default hooks. An undeclared vertex file leaves the default.
    pub fn splices(&self) -> Vec<(&'static str, &'a str)> {
        let mut out = Vec::with_capacity(2);
        if let Some(v) = self.vertex {
            out.push((VERTEX_MARKER, v));
        }
        out.push((FRAGMENT_MARKER, self.fragment));
        out
    }
}

/// The variant defines for one entry on one host. `pool_size` and
/// `probe_count` are the bindless texture-pool length and the probe cube
/// array length the Vulkan host declares; the cook bakes the ceilings and a
/// device that cannot seat them recompiles, exactly as the engine's own
/// bindless programs do. Metal and DirectX bind fixed counts.
pub fn defines(
    _program: &Program,
    platform: Platform,
    pool_size: usize,
    probe_count: usize,
) -> Vec<(&'static str, String)> {
    let probes = ("MAX_PROBES", MAX_PROBES.to_string());
    match platform {
        Platform::Metal => alloc::vec![
            ("METAL_ABI", "1".into()),
            ("POOL_SIZE", BINDLESS_POOL_SIZE.to_string()),
            probes,
        ],
        Platform::Hlsl => alloc::vec![("DXIL_ABI", "1".into()), probes],
        Platform::Glsl => alloc::vec![
            ("POOL_SIZE", pool_size.to_string()),
            ("MAX_PROBES", probe_count.to_string()),
        ],
    }
}

/// The exact source text one entry compiles for one host with the world's
/// files spliced in. `resolve` lets a hot-reload build prefer the checkout's
/// copy of the templates over the embedded ones.
pub fn source_with(
    program: &Program,
    platform: Platform,
    pool_size: usize,
    probe_count: usize,
    sources: &Sources<'_>,
    resolve: impl Fn(&str) -> Option<&'static str>,
) -> String {
    let defines = defines(program, platform, pool_size, probe_count);
    let defines: Vec<(&str, &str)> = defines.iter().map(|(k, v)| (*k, v.as_str())).collect();
    slang_source::assemble_with_splices(program.file, &defines, resolve, &sources.splices())
}

/// The same source from the embedded templates alone.
pub fn source(
    program: &Program,
    platform: Platform,
    pool_size: usize,
    probe_count: usize,
    sources: &Sources<'_>,
) -> String {
    source_with(
        program,
        platform,
        pool_size,
        probe_count,
        sources,
        crate::render::shaders::embedded,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHADE: &str =
        "float4 shade(VertexOut in, GpuObjectData od) { return float4(1.0, 0.0, 1.0, 1.0); }";

    fn fragment_only() -> Sources<'static> {
        Sources {
            vertex: None,
            fragment: SHADE,
        }
    }

    // Every host compiles each entry it names exactly once, whichever way it
    // groups them; a dropped entry is a pipeline that cannot be built at load.
    #[test]
    fn a_grouping_covers_every_entry_of_the_host_once() {
        for platform in [Platform::Metal, Platform::Hlsl, Platform::Glsl] {
            let mut grouped: Vec<&str> =
                groups(platform).iter().flatten().map(|p| p.entry).collect();
            grouped.sort_unstable();
            let mut listed: Vec<&str> = programs(platform).map(|p| p.entry).collect();
            listed.sort_unstable();
            assert_eq!(grouped, listed, "{platform:?}");
        }
    }

    // The bindless pair is the whole table on every host; Metal takes it as one
    // library and the other two one artifact per entry.
    #[test]
    fn the_pair_is_the_whole_table_and_metal_groups_it() {
        for platform in [Platform::Metal, Platform::Hlsl, Platform::Glsl] {
            let entries: Vec<&str> = programs(platform).map(|p| p.entry).collect();
            assert_eq!(
                entries,
                ["vertex_main_bindless", "fragment_main_bindless"],
                "{platform:?}"
            );
        }
        let metal = groups(Platform::Metal);
        assert_eq!(metal.len(), 1);
        assert_eq!(metal[0].len(), 2);
        for platform in [Platform::Hlsl, Platform::Glsl] {
            assert_eq!(groups(platform).len(), 2, "{platform:?}");
            assert!(
                groups(platform).iter().all(|g| g.len() == 1),
                "{platform:?}"
            );
        }
    }

    #[test]
    fn every_entry_is_found_by_name_and_names_are_unique() {
        for p in ALL {
            assert_eq!(program(p.entry).map(|q| q.entry), Some(p.entry));
        }
        assert!(program("no_such_entry").is_none());
        let mut names: Vec<&str> = ALL.iter().map(|p| p.entry).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), ALL.len());
    }

    // The fragment file replaces the engine's default `shade`; an undeclared
    // vertex file leaves the default `transform` in place.
    #[test]
    fn the_world_fragment_replaces_the_default_and_the_vertex_default_stays() {
        let frag = program("fragment_main_bindless").unwrap();
        let src = source(
            frag,
            Platform::Metal,
            BINDLESS_POOL_SIZE,
            MAX_PROBES,
            &fragment_only(),
        );
        assert!(src.contains(SHADE));
        assert!(
            !src.contains("return shade_surface(in, od);"),
            "default shade replaced"
        );
        assert!(!src.contains(FRAGMENT_MARKER) && !src.contains(VERTEX_MARKER));
        assert!(src.contains("return project_vertex(model, pos, normal, tangent, color, uv);"));

        let both = Sources {
            vertex: Some(
                "VertexOut transform(float4x4 m, float3 p, float3 n, float3 t, float3 c, float2 uv) { return project_vertex(m, p, n, t, c, uv); }",
            ),
            fragment: SHADE,
        };
        let src = source(frag, Platform::Metal, BINDLESS_POOL_SIZE, MAX_PROBES, &both);
        assert!(src.contains("VertexOut transform(float4x4 m,"));
        assert!(!src.contains("return project_vertex(model, pos, normal, tangent, color, uv);"));
    }

    // The bindless file compiles both stages from one variant, so both hooks
    // land in it and both stages assemble to identical text.
    #[test]
    fn the_pair_assembles_to_one_text() {
        let vert = program("vertex_main_bindless").unwrap();
        let frag = program("fragment_main_bindless").unwrap();
        let a = source(
            vert,
            Platform::Metal,
            BINDLESS_POOL_SIZE,
            MAX_PROBES,
            &fragment_only(),
        );
        let b = source(
            frag,
            Platform::Metal,
            BINDLESS_POOL_SIZE,
            MAX_PROBES,
            &fragment_only(),
        );
        assert_eq!(a, b);
        assert!(a.contains(SHADE));
        assert!(a.contains("#define METAL_ABI 1"));
    }

    // Each host's defines are the ones its own program table bakes, and the
    // Vulkan pool size is whatever the caller declares.
    #[test]
    fn defines_follow_the_host() {
        let frag = program("fragment_main_bindless").unwrap();
        let names =
            |platform, pool| -> Vec<(&str, String)> { defines(frag, platform, pool, MAX_PROBES) };
        assert_eq!(
            names(Platform::Metal, 0),
            [
                ("METAL_ABI", "1".to_string()),
                ("POOL_SIZE", "1024".to_string()),
                ("MAX_PROBES", "8".to_string())
            ]
        );
        assert_eq!(
            names(Platform::Hlsl, 0),
            [
                ("DXIL_ABI", "1".to_string()),
                ("MAX_PROBES", "8".to_string())
            ]
        );
        assert_eq!(
            names(Platform::Glsl, 37),
            [
                ("POOL_SIZE", "37".to_string()),
                ("MAX_PROBES", "8".to_string())
            ]
        );
        // Only the Vulkan host reads the probe count the device seats.
        assert_eq!(
            defines(frag, Platform::Glsl, 37, 4),
            [
                ("POOL_SIZE", "37".to_string()),
                ("MAX_PROBES", "4".to_string())
            ]
        );
        assert_eq!(
            defines(frag, Platform::Metal, 0, 4),
            defines(frag, Platform::Metal, 0, MAX_PROBES)
        );
    }
}
