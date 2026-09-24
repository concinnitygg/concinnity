//! What a world `Shader` compiles to, on every backend.
//!
//! A world Shader defines two hooks, `transform` and `shade`, and the engine's
//! own main-pass entries call them. So a world shader compiles as the engine's
//! main-pass programs do, from `main_bindless.hlsl`, with the world's files
//! spliced at the hook markers in place of the engine's defaults. The cook
//! iterates this table to compile a Shader ahead of time and each renderer
//! iterates it to find what the cook left.

use alloc::string::String;
use alloc::vec::Vec;

use crate::platform::Platform;
use crate::render::shader_source;

/// The marker the world's `vertex` file is spliced at.
pub const VERTEX_MARKER: &str = "{SURFACE_VERTEX}";
/// The marker the world's `fragment` file is spliced at.
pub const FRAGMENT_MARKER: &str = "{SURFACE_FRAGMENT}";

/// One entry point of one main-pass file. The vertex entry carries the
/// world's `transform`, the fragment entry its `shade`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Program {
    /// The shader file the entry compiles from.
    pub file: &'static str,
    /// Entry point name, as the source spells it.
    pub entry: &'static str,
}

/// Every entry a world Shader compiles: the bindless pair, on every host.
pub const ALL: &[Program] = &[
    Program {
        file: "main_bindless.hlsl",
        entry: "vertex_main_bindless",
    },
    Program {
        file: "main_bindless.hlsl",
        entry: "fragment_main_bindless",
    },
];

/// The entry named `entry`.
pub fn program(entry: &str) -> Option<&'static Program> {
    ALL.iter().find(|p| p.entry == entry)
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

/// The exact source text one entry compiles for one host with the world's
/// files spliced in. `resolve` lets a hot-reload build prefer the checkout's
/// copy of the templates over the embedded ones. No define varies it: every
/// array the pass binds is unsized, so one text serves every device.
pub fn source_with(
    program: &Program,
    platform: Platform,
    sources: &Sources<'_>,
    resolve: impl Fn(&str) -> Option<&'static str>,
) -> String {
    shader_source::assemble_with_splices(program.file, platform, &[], resolve, &sources.splices())
}

/// The same source from the embedded templates alone.
pub fn source(program: &Program, platform: Platform, sources: &Sources<'_>) -> String {
    source_with(program, platform, sources, crate::render::shaders::embedded)
}

#[cfg(test)]
mod tests {
    use super::super::declared::{Row, assert_rows_are_sound};
    use super::*;

    const SHADE: &str =
        "float4 shade(VertexOut v, GpuObjectData od) { return float4(1.0, 0.0, 1.0, 1.0); }";

    fn fragment_only() -> Sources<'static> {
        Sources {
            vertex: None,
            fragment: SHADE,
        }
    }

    #[test]
    fn every_program_is_sound_on_every_host() {
        for platform in Platform::ALL {
            let rows: Vec<Row> = ALL
                .iter()
                .map(|p| Row {
                    label: String::from(p.entry),
                    file: p.file,
                    entry: p.entry,
                    defines: Vec::new(),
                    source: source(p, platform, &fragment_only()),
                })
                .collect();
            assert_rows_are_sound(&alloc::format!("surface {platform:?}"), &rows);
        }
    }

    // The bindless pair is the whole table.
    #[test]
    fn the_pair_is_the_whole_table() {
        let entries: Vec<&str> = ALL.iter().map(|p| p.entry).collect();
        assert_eq!(entries, ["vertex_main_bindless", "fragment_main_bindless"]);
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
        let src = source(frag, Platform::Metal, &fragment_only());
        assert!(src.contains(SHADE));
        assert!(
            !src.contains("return shade_surface(v, od);"),
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
        let src = source(frag, Platform::Metal, &both);
        assert!(src.contains("VertexOut transform(float4x4 m,"));
        assert!(!src.contains("return project_vertex(model, pos, normal, tangent, color, uv);"));
    }

    // The bindless file compiles both stages from one variant, so both hooks
    // land in it and both stages assemble to identical text.
    #[test]
    fn the_pair_assembles_to_one_text() {
        let vert = program("vertex_main_bindless").unwrap();
        let frag = program("fragment_main_bindless").unwrap();
        let a = source(vert, Platform::Metal, &fragment_only());
        let b = source(frag, Platform::Metal, &fragment_only());
        assert_eq!(a, b);
        assert!(a.contains(SHADE));
        assert!(a.starts_with("#define CN_BACKEND_METAL 1\n"));
    }
}
