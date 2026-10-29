//! What a world `Shader` compiles to, on every backend.
//!
//! A world Shader defines two hooks, `transform` and `shade`, and the engine's
//! own main-pass entries call them. So a world shader compiles as the engine's
//! main-pass programs do, from `main_bindless.hlsl`, with the world's files
//! spliced at the hook markers in place of the engine's defaults. The G-buffer
//! pre-pass compiles from the same file under `SURFACE_PREPASS`, so the
//! world's `transform` places its depth and motion too. The cook iterates this
//! table to compile a Shader ahead of time and each renderer iterates it to
//! find what the cook left.

use alloc::string::String;
use alloc::vec::Vec;

use crate::platform::Platform;
use crate::render::shader::source::{self, SourceFile, Splice};

/// The marker the world's `vertex` file is spliced at.
const VERTEX_MARKER: &str = "{SURFACE_VERTEX}";
/// The marker the world's `fragment` file is spliced at.
const FRAGMENT_MARKER: &str = "{SURFACE_FRAGMENT}";

/// The gate the G-buffer pre-pass entries compile under.
pub const PREPASS_GATE: &str = "SURFACE_PREPASS";

/// The gate the main fragment compiles under to shade the ground under grass
/// and a far band as grass canopy. Without it (or [`GROUND_GATE`]) the band is
/// a constant 0 and the grass code folds away.
pub const CANOPY_GATE: &str = "SURFACE_CANOPY";

/// The gate the main fragment compiles under to shade the ground under grass
/// alone, for draws wholly nearer than their far band.
pub const GROUND_GATE: &str = "SURFACE_GRASS_GROUND";

/// The gate a grass fragment compiles under to light a planet's ground from
/// its own sky where it lies far round the planet or the camera is high.
pub const DAYLIGHT_GATE: &str = "SURFACE_DAYLIGHT";

/// What a main fragment shades of the ground a surface grows grass on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SurfaceVariant {
    /// Nothing: the surface grows no grass.
    #[default]
    Plain,
    /// The ground under the grass (litter, patches, the leaf's shade), for a
    /// draw that cannot reach its far band.
    Ground,
    /// [`Self::Ground`], lit from a planet's own sky.
    GroundDaylight,
    /// The ground under the grass and the canopy past the blades.
    Canopy,
    /// [`Self::Canopy`], lit from a planet's own sky.
    CanopyDaylight,
}

impl SurfaceVariant {
    /// Every variant, the plain one first.
    pub const ALL: [Self; 5] = [
        Self::Plain,
        Self::Ground,
        Self::GroundDaylight,
        Self::Canopy,
        Self::CanopyDaylight,
    ];

    /// Whether the fragment shades the ground under grass.
    pub fn shades_ground(self) -> bool {
        self != Self::Plain
    }

    /// Whether the fragment lights the ground from a planet's own sky.
    pub fn daylight(self) -> bool {
        matches!(self, Self::GroundDaylight | Self::CanopyDaylight)
    }

    /// Whether the fragment shades the canopy past the blades.
    pub fn canopy(self) -> bool {
        matches!(self, Self::Canopy | Self::CanopyDaylight)
    }
}

/// One entry point of one main-pass file. The vertex entries carry the
/// world's `transform`, the main fragment entry its `shade`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Program {
    /// The shader file the entry compiles from.
    pub file: &'static str,
    /// Entry point name, as the source spells it.
    pub entry: &'static str,
    /// Variant gates, each injected as `#define <gate> 1`.
    pub gates: &'static [&'static str],
}

/// The bindless main-pass vertex entry.
pub const MAIN_VERTEX: Program = Program {
    file: "main_bindless.hlsl",
    entry: "vertex_main_bindless",
    gates: &[],
};
/// The bindless main-pass fragment entry.
pub const MAIN_FRAGMENT: Program = Program {
    file: "main_bindless.hlsl",
    entry: "fragment_main_bindless",
    gates: &[],
};
/// The bindless main-pass fragment entry that shades a far band as canopy.
pub const MAIN_FRAGMENT_CANOPY: Program = Program {
    file: "main_bindless.hlsl",
    entry: "fragment_main_bindless",
    gates: &[CANOPY_GATE],
};
/// [`MAIN_FRAGMENT_CANOPY`] lit from a planet's own sky.
pub const MAIN_FRAGMENT_CANOPY_DAYLIGHT: Program = Program {
    file: "main_bindless.hlsl",
    entry: "fragment_main_bindless",
    gates: &[CANOPY_GATE, DAYLIGHT_GATE],
};
/// The bindless main-pass fragment entry that shades the ground under grass
/// without the canopy.
const MAIN_FRAGMENT_GROUND: Program = Program {
    file: "main_bindless.hlsl",
    entry: "fragment_main_bindless",
    gates: &[GROUND_GATE],
};
/// [`MAIN_FRAGMENT_GROUND`] lit from a planet's own sky.
const MAIN_FRAGMENT_GROUND_DAYLIGHT: Program = Program {
    file: "main_bindless.hlsl",
    entry: "fragment_main_bindless",
    gates: &[GROUND_GATE, DAYLIGHT_GATE],
};
/// The G-buffer pre-pass vertex entry.
pub const PREPASS_VERTEX: Program = Program {
    file: "main_bindless.hlsl",
    entry: "vertex_prepass_bindless",
    gates: &[PREPASS_GATE],
};
/// The G-buffer pre-pass fragment entry.
pub const PREPASS_FRAGMENT: Program = Program {
    file: "main_bindless.hlsl",
    entry: "fragment_prepass_bindless",
    gates: &[PREPASS_GATE],
};

/// Every program a world Shader compiles, on every host: the bindless main
/// pair, the canopy and ground main fragments with and without a planet's own
/// sky, and the pre-pass pair.
pub const ALL: &[Program] = &[
    MAIN_VERTEX,
    MAIN_FRAGMENT,
    MAIN_FRAGMENT_CANOPY,
    MAIN_FRAGMENT_CANOPY_DAYLIGHT,
    MAIN_FRAGMENT_GROUND,
    MAIN_FRAGMENT_GROUND_DAYLIGHT,
    PREPASS_VERTEX,
    PREPASS_FRAGMENT,
];

/// The main fragment that shades `variant`.
pub fn main_fragment(variant: SurfaceVariant) -> &'static Program {
    match variant {
        SurfaceVariant::Plain => &MAIN_FRAGMENT,
        SurfaceVariant::Ground => &MAIN_FRAGMENT_GROUND,
        SurfaceVariant::GroundDaylight => &MAIN_FRAGMENT_GROUND_DAYLIGHT,
        SurfaceVariant::Canopy => &MAIN_FRAGMENT_CANOPY,
        SurfaceVariant::CanopyDaylight => &MAIN_FRAGMENT_CANOPY_DAYLIGHT,
    }
}

impl Program {
    /// The variant defines: each gate as `1`.
    pub fn defines(&self) -> Vec<(&'static str, &'static str)> {
        self.gates.iter().map(|g| (*g, "1")).collect()
    }
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
        let mut out = Vec::with_capacity(2);
        if let Some(v) = self.vertex {
            out.push(Splice::of(VERTEX_MARKER, v));
        }
        out.push(Splice::of(FRAGMENT_MARKER, self.fragment));
        out
    }
}

/// The exact source text one entry compiles for one host with the world's
/// files spliced in. `resolve` lets a hot-reload build prefer the checkout's
/// copy of the templates over the embedded ones. Only the entry's gates vary
/// it: every array the pass binds is unsized, so one text serves every device.
pub fn source_with(
    program: &Program,
    platform: Platform,
    sources: &Sources<'_>,
    resolve: impl Fn(&str) -> Option<&'static str>,
) -> String {
    source::assemble_with_splices(
        program.file,
        platform,
        &program.defines(),
        resolve,
        &sources.splices(),
    )
}

/// The same source from the embedded templates alone.
pub fn source(program: &Program, platform: Platform, sources: &Sources<'_>) -> String {
    source_with(
        program,
        platform,
        sources,
        crate::render::shader::embedded::embedded,
    )
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
                    label: alloc::format!("{}{:?}", p.entry, p.gates),
                    file: p.file,
                    entry: p.entry,
                    defines: p.defines(),
                    source: source(p, platform, &fragment_only()),
                })
                .collect();
            assert_rows_are_sound(&alloc::format!("surface {platform:?}"), &rows);
        }
    }

    // The main pair, the grass fragments and the pre-pass pair are the whole
    // table, each gated only by what it compiles.
    #[test]
    fn the_main_pair_grass_fragments_and_prepass_pair_are_the_whole_table() {
        let rows: Vec<(&str, &[&str])> = ALL.iter().map(|p| (p.entry, p.gates)).collect();
        assert_eq!(
            rows,
            [
                ("vertex_main_bindless", &[][..]),
                ("fragment_main_bindless", &[][..]),
                ("fragment_main_bindless", &[CANOPY_GATE][..]),
                ("fragment_main_bindless", &[CANOPY_GATE, DAYLIGHT_GATE][..]),
                ("fragment_main_bindless", &[GROUND_GATE][..]),
                ("fragment_main_bindless", &[GROUND_GATE, DAYLIGHT_GATE][..]),
                ("vertex_prepass_bindless", &[PREPASS_GATE][..]),
                ("fragment_prepass_bindless", &[PREPASS_GATE][..]),
            ]
        );
    }

    // The pre-pass runs the world's vertex hook, so the world's vertex file is
    // in its text, and the gate leads the text past the backend define.
    #[test]
    fn the_prepass_carries_the_world_vertex_hook_under_its_gate() {
        for program in [&PREPASS_VERTEX, &PREPASS_FRAGMENT] {
            let src = source(program, Platform::Vulkan, &vertex_and_fragment());
            assert!(src.starts_with("#define CN_BACKEND_VULKAN 1\n#define SURFACE_PREPASS 1\n"));
            assert!(src.contains("VertexOut transform(float4x4 m,"));
            assert!(src.contains(&alloc::format!("{}(", program.entry)));
        }
        let main = source(&MAIN_VERTEX, Platform::Vulkan, &vertex_and_fragment());
        assert!(!main.contains("#define SURFACE_PREPASS"));
    }

    // Each grass fragment carries its own gates alone, and differs from the
    // plain fragment by those gates only.
    #[test]
    fn each_grass_fragment_carries_only_its_gates() {
        let body = |src: &str| -> Vec<String> {
            src.lines()
                .filter(|l| !l.starts_with("#line ") && !l.starts_with("#define "))
                .map(String::from)
                .collect()
        };
        let plain = source(&MAIN_FRAGMENT, Platform::Metal, &fragment_only());
        assert_eq!(main_fragment(SurfaceVariant::Plain), &MAIN_FRAGMENT);
        for variant in SurfaceVariant::ALL {
            let program = main_fragment(variant);
            let src = source(program, Platform::Metal, &fragment_only());
            for (gate, on) in [
                (CANOPY_GATE, variant.canopy()),
                (GROUND_GATE, variant.shades_ground() && !variant.canopy()),
                (DAYLIGHT_GATE, variant.daylight()),
            ] {
                let define = alloc::format!("#define {gate} 1\n");
                assert_eq!(src.contains(&define), on, "{variant:?} {gate}");
            }
            assert_eq!(
                body(&src),
                body(&plain),
                "{variant:?}: the gates are the only difference"
            );
        }
    }

    #[test]
    fn programs_are_unique_by_entry_and_gates() {
        let mut keys: Vec<(&str, &[&str])> = ALL.iter().map(|p| (p.entry, p.gates)).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), ALL.len());
    }

    // The fragment file replaces the engine's default `shade`; an undeclared
    // vertex file leaves the default `transform` in place.
    #[test]
    fn the_world_fragment_replaces_the_default_and_the_vertex_default_stays() {
        let frag = &MAIN_FRAGMENT;
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
        let a = source(&MAIN_VERTEX, Platform::Metal, &fragment_only());
        let b = source(&MAIN_FRAGMENT, Platform::Metal, &fragment_only());
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
            let unspliced = source::assemble_with_splices(
                program.file,
                Platform::Vulkan,
                &program.defines(),
                crate::render::shader::embedded::embedded,
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
