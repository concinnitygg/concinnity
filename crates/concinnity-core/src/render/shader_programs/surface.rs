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
use crate::render::shader_source::{self, Splice};

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

/// One of the world's files: the path it was read from and its text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceFile<'a> {
    /// The path a compiler's diagnostics name the file by. It rides the
    /// assembled text in a `#line` directive, so it is part of what an
    /// artifact's source digest covers.
    pub path: &'a str,
    /// The file's text.
    pub text: &'a str,
}

/// The world's two files.
#[derive(Debug, Clone, Copy)]
pub struct Sources<'a> {
    /// The `vertex` file, when declared.
    pub vertex: Option<SourceFile<'a>>,
    /// The `fragment` file.
    pub fragment: SourceFile<'a>,
}

impl<'a> Sources<'a> {
    /// The splices that put the declared files in place of the engine's
    /// default hooks, each fenced under its own path. An undeclared vertex file
    /// leaves the default.
    pub fn splices(&self) -> Vec<Splice<'a>> {
        let splice = |marker, file: SourceFile<'a>| Splice::from_file(marker, file.text, file.path);
        let mut out = Vec::with_capacity(2);
        if let Some(v) = self.vertex {
            out.push(splice(VERTEX_MARKER, v));
        }
        out.push(splice(FRAGMENT_MARKER, self.fragment));
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
    const TRANSFORM: &str = "VertexOut transform(float4x4 m, float3 p, float3 n, float3 t, float3 c, float2 uv)\n\
        {\n    return project_vertex(m, p, n, t, c, uv);\n}\n";

    fn file<'a>(path: &'a str, text: &'a str) -> SourceFile<'a> {
        SourceFile { path, text }
    }

    fn fragment_only() -> Sources<'static> {
        Sources {
            vertex: None,
            fragment: file("shaders/magenta.hlsl", SHADE),
        }
    }

    fn vertex_and_fragment() -> Sources<'static> {
        Sources {
            vertex: Some(file("shaders/sway.hlsl", TRANSFORM)),
            fragment: file("shaders/magenta.hlsl", SHADE),
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

        let src = source(frag, Platform::Metal, &vertex_and_fragment());
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

    // Each line of `src` that is not a `#line` directive, with the file and
    // line a compiler numbers it by.
    fn numbered<'s>(src: &'s str, file: &str) -> Vec<(String, usize, &'s str)> {
        let mut current = String::from(file);
        let mut line = 1;
        let mut out = Vec::new();
        for text in src.lines() {
            if let Some(rest) = text.strip_prefix("#line ") {
                let (number, path) = rest.split_once(' ').expect("a line and a path");
                line = number.parse().expect("a line number");
                current = String::from(path.trim_matches('"'));
                continue;
            }
            out.push((current.clone(), line, text));
            line += 1;
        }
        out
    }

    // Against the real template: the world's lines number from 1 under their
    // own paths, and every template line keeps the number it has with the
    // markers left in place, which is what a compiler reported before.
    fn assert_numbering(sources: &Sources<'_>) {
        for program in ALL {
            let fenced = source(program, Platform::Vulkan, sources);
            let markers_kept: Vec<Splice<'_>> = sources
                .splices()
                .iter()
                .map(|s| Splice::inline(s.marker, s.marker))
                .collect();
            let unspliced = shader_source::assemble_with_splices(
                program.file,
                Platform::Vulkan,
                &[],
                crate::render::shaders::embedded,
                &markers_kept,
            );
            let template: Vec<&str> = unspliced.lines().collect();
            let mut spliced = alloc::collections::BTreeMap::<String, Vec<&str>>::new();
            for (path, line, text) in numbered(&fenced, program.file) {
                if path == program.file {
                    assert!(
                        template[line - 1].ends_with(text),
                        "{}:{line} reads {text:?}, template has {:?}",
                        program.file,
                        template[line - 1]
                    );
                } else {
                    let lines = spliced.entry(path).or_default();
                    assert_eq!(line, lines.len() + 1, "numbered from 1");
                    lines.push(text);
                }
            }
            let mut want = alloc::collections::BTreeMap::new();
            want.insert(
                String::from(sources.fragment.path),
                sources.fragment.text.lines().collect::<Vec<_>>(),
            );
            if let Some(v) = sources.vertex {
                want.insert(String::from(v.path), v.text.lines().collect());
            }
            assert_eq!(spliced, want, "{}", program.entry);
        }
    }

    #[test]
    fn a_fragment_only_shader_numbers_its_file_and_the_template() {
        assert_numbering(&fragment_only());
    }

    #[test]
    fn a_vertex_and_fragment_shader_numbers_both_files_and_the_template() {
        assert_numbering(&vertex_and_fragment());
    }
}
