//! What the raymarched SDF volume pass compiles, on every backend.
//!
//! All three backends compile the same six entries out of `raymarch.hlsl`,
//! differing only in the backend define the assembler leads the source with and
//! in what dxc is asked to emit. The cook iterates it to compile a volume's
//! field ahead of time and each renderer iterates it to find what the cook left.

use alloc::string::String;

use crate::components::SdfVolume;
use crate::platform::Platform;
use crate::render::shader_source::{self, SourceFile, Splice};

/// The shader file every entry below compiles from.
pub const FILE: &str = "raymarch.hlsl";

/// The marker a volume's authored distance field is spliced at.
pub const BODY_MARKER: &str = "{SDF_BODY}";

/// The helper an authored `shade` calls to read the scene behind the surface.
pub const SCENE_TAP: &str = "sampleSceneRefracted";

/// Whether an authored distance field reads the scene behind the surface.
///
/// The tap is the only way in: the scene snapshot is reached through this
/// helper and is named by no other declaration a field can see. A renderer
/// copies the frame's color target for the pass only when some visible volume
/// answers `true` here, so a world of opaque volumes pays nothing.
///
/// A field that spells the name in a comment reads as tapping. That is the
/// conservative direction, and the cost of being wrong is the copy this
/// existed to skip rather than a black refraction.
pub fn field_taps_scene(field: &str) -> bool {
    field.contains(SCENE_TAP)
}

/// Which of the three draws an entry belongs to. A volume compiles one of the
/// first two, plus the third when it casts shadows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// An opaque surface writing color and depth.
    Surface,
    /// A participating medium blended over the scene.
    Volumetric,
    /// A depth-only caster marched from the light side.
    Shadow,
}

impl Family {
    /// The variant define selecting this family.
    pub fn define(self) -> &'static str {
        match self {
            Family::Surface => "RAYMARCH_SURFACE",
            Family::Volumetric => "RAYMARCH_VOLUMETRIC",
            Family::Shadow => "RAYMARCH_SHADOW",
        }
    }
}

/// One entry point of one family: a vertex entry rasterizing the bounding-box
/// proxy, or a fragment entry marching the field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Program {
    /// Entry point name, as the source spells it.
    pub entry: &'static str,
    /// The draw it belongs to.
    pub family: Family,
}

/// Every entry `raymarch.hlsl` declares.
pub const ALL: &[Program] = &[
    Program {
        entry: "raymarch_vertex",
        family: Family::Surface,
    },
    Program {
        entry: "raymarch_fragment",
        family: Family::Surface,
    },
    Program {
        entry: "raymarch_volumetric_vertex",
        family: Family::Volumetric,
    },
    Program {
        entry: "raymarch_volumetric_fragment",
        family: Family::Volumetric,
    },
    Program {
        entry: "raymarch_shadow_vertex",
        family: Family::Shadow,
    },
    Program {
        entry: "raymarch_shadow_fragment",
        family: Family::Shadow,
    },
];

/// The families a volume draws with: its own, plus the shadow caster when it
/// casts one. A volumetric medium never casts, so the pair is exclusive.
pub fn families(volumetric: bool, cast_shadows: bool) -> impl Iterator<Item = Family> {
    let own = if volumetric {
        Family::Volumetric
    } else {
        Family::Surface
    };
    let shadow = (cast_shadows && !volumetric).then_some(Family::Shadow);
    core::iter::once(own).chain(shadow)
}

/// Every entry a volume with these flags needs compiled.
pub fn programs(volumetric: bool, cast_shadows: bool) -> impl Iterator<Item = &'static Program> {
    families(volumetric, cast_shadows).flat_map(|f| ALL.iter().filter(move |p| p.family == f))
}

/// The flags that decide which pipelines a volume draws with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VolumeFlags {
    /// The volume is a participating medium rather than a surface.
    pub volumetric: bool,
    /// The volume asks for a shadow caster.
    pub cast_shadows: bool,
}

impl VolumeFlags {
    /// The flags `volume` declares.
    pub fn of(volume: &SdfVolume) -> Self {
        Self {
            volumetric: volume.volumetric,
            cast_shadows: volume.cast_shadows,
        }
    }

    /// Whether the volume draws a shadow caster. A medium never does, whatever
    /// its asset says.
    pub fn casts(self) -> bool {
        self.families().any(|f| f == Family::Shadow)
    }

    /// Every family the volume draws with.
    pub fn families(self) -> impl Iterator<Item = Family> {
        families(self.volumetric, self.cast_shadows)
    }
}

/// The exact source text one family compiles for one host, with `field` spliced
/// in as the world's distance field. The field is fenced by `#line` directives
/// naming its path, so a compiler reports its lines against the author's file.
/// `resolve` lets a hot-reload build prefer the checkout's copy of the template
/// over the embedded one.
pub fn source_with(
    family: Family,
    platform: Platform,
    field: SourceFile<'_>,
    resolve: impl Fn(&str) -> Option<&'static str>,
) -> String {
    shader_source::assemble_with_splices(
        FILE,
        platform,
        &[(family.define(), "1")],
        resolve,
        &[Splice::of(BODY_MARKER, field)],
    )
}

/// The same source from the embedded templates alone.
pub fn source(family: Family, platform: Platform, field: SourceFile<'_>) -> String {
    source_with(family, platform, field, crate::render::shaders::embedded)
}

#[cfg(test)]
mod tests {
    use super::super::declared::{Row, assert_rows_are_sound};
    use super::*;
    use alloc::vec::Vec;

    const PATH: &str = "fields/blob.hlsl";

    fn field(text: &str) -> SourceFile<'_> {
        SourceFile { path: PATH, text }
    }

    // A medium never casts, so a volumetric volume that also sets
    // `cast_shadows` builds no caster.
    #[test]
    fn only_a_casting_surface_volume_draws_a_shadow_caster() {
        let flags = |volumetric, cast_shadows| VolumeFlags {
            volumetric,
            cast_shadows,
        };
        assert!(flags(false, true).casts());
        assert!(!flags(false, false).casts());
        assert!(!flags(true, true).casts());
        let families: Vec<Family> = flags(true, true).families().collect();
        assert_eq!(families, [Family::Volumetric]);
        let volume = SdfVolume {
            cast_shadows: true,
            ..SdfVolume::default()
        };
        assert_eq!(VolumeFlags::of(&volume), flags(false, true));
    }

    #[test]
    fn a_surface_volume_compiles_its_own_pair_and_nothing_else() {
        let entries: Vec<&str> = programs(false, false).map(|p| p.entry).collect();
        assert_eq!(entries, ["raymarch_vertex", "raymarch_fragment"]);
    }

    #[test]
    fn a_casting_surface_volume_adds_the_shadow_pair() {
        let entries: Vec<&str> = programs(false, true).map(|p| p.entry).collect();
        assert_eq!(
            entries,
            [
                "raymarch_vertex",
                "raymarch_fragment",
                "raymarch_shadow_vertex",
                "raymarch_shadow_fragment"
            ]
        );
    }

    // A medium is integrated, not surfaced, so it has no depth to cast from.
    // The asset validation forces `cast_shadows` off for one; this makes the
    // table agree even if an authored volume sets both.
    #[test]
    fn a_volumetric_volume_never_compiles_a_shadow_caster() {
        for cast_shadows in [false, true] {
            let entries: Vec<&str> = programs(true, cast_shadows).map(|p| p.entry).collect();
            assert_eq!(
                entries,
                ["raymarch_volumetric_vertex", "raymarch_volumetric_fragment"]
            );
        }
    }

    // A family's source leads with the backend define, then selects the
    // family.
    #[test]
    fn the_source_leads_with_the_backend_then_the_family() {
        let src = source(Family::Shadow, Platform::DirectX, field("// field"));
        assert!(src.starts_with("#define CN_BACKEND_DIRECTX 1\n#define RAYMARCH_SHADOW 1\n"));
        let src = source(Family::Volumetric, Platform::Vulkan, field("// field"));
        assert!(src.starts_with("#define CN_BACKEND_VULKAN 1\n#define RAYMARCH_VOLUMETRIC 1\n"));
    }

    #[test]
    fn every_program_is_sound_on_every_host() {
        let map = "float map(float3 p, SdfParams q, float t) { return 1.0; }";
        for platform in Platform::ALL {
            let rows: Vec<Row> = ALL
                .iter()
                .map(|p| Row {
                    label: String::from(p.entry),
                    file: FILE,
                    entry: p.entry,
                    defines: alloc::vec![(p.family.define(), "1")],
                    source: source(p.family, platform, field(map)),
                })
                .collect();
            assert_rows_are_sound(&alloc::format!("raymarch {platform:?}"), &rows);
        }
        let text = crate::render::shaders::embedded(FILE).expect("raymarch.hlsl");
        assert!(text.contains(BODY_MARKER), "{FILE} carries no body marker");
    }

    // The tap the detection looks for is the one the helpers declare, so a
    // rename in the shader fails here rather than by silently making every
    // refractive volume read a stale scene.
    #[test]
    fn the_scene_tap_is_declared_by_the_helpers() {
        let text =
            crate::render::shaders::embedded("raymarch_common.hlsl").expect("raymarch_common");
        assert!(text.contains(SCENE_TAP), "{SCENE_TAP} declares nothing");
    }

    // A field that never names the tap is a field the scene copy can skip,
    // which is the whole of the saving.
    #[test]
    fn only_a_field_naming_the_tap_reads_as_tapping() {
        let opaque = "float map(float3 p, SdfParams q, float t) { return 1.0; }";
        assert!(!field_taps_scene(opaque));
        let refractive = "s.transmitted = sampleSceneRefracted(frag_uv, normal, 0.05);";
        assert!(field_taps_scene(refractive));
    }

    // The engine template names the tap in its own declaration, so the flag
    // has to come from the authored field alone: assembled source would read
    // as tapping for every volume in every world.
    #[test]
    fn the_assembled_source_is_not_what_the_flag_reads() {
        let opaque = "float map(float3 p, SdfParams q, float t) { return 1.0; }";
        let src = source(Family::Surface, Platform::Metal, field(opaque));
        assert!(src.contains(SCENE_TAP), "the template declares the tap");
        assert!(!field_taps_scene(opaque));
    }

    // The field reaches the assembled source and the defines lead it, on every
    // host. A family's source must also differ per host, or two backends would
    // share a cache entry for different binding layouts.
    #[test]
    fn the_field_is_spliced_and_the_hosts_assemble_differently() {
        let map = "float map(float3 p, SdfParams q, float t) { return 1.0; }";
        let mut seen = Vec::new();
        for platform in Platform::ALL {
            let src = source(Family::Surface, platform, field(map));
            assert!(src.contains(map), "{platform:?} lost the field");
            assert!(!src.contains(BODY_MARKER), "{platform:?} left the marker");
            assert!(src.starts_with("#define "), "{platform:?} defines lead");
            seen.push(shader_source::source_digest(&src));
        }
        seen.dedup();
        assert_eq!(seen.len(), 3, "two hosts assemble identical source");
    }

    // Two fields are two sources, which is what keeps the content-addressed
    // cache from serving one world's volume the artifact of another's.
    #[test]
    fn two_fields_assemble_to_two_digests() {
        let a = source(Family::Surface, Platform::Metal, field("// one"));
        let b = source(Family::Surface, Platform::Metal, field("// two"));
        assert_ne!(
            shader_source::source_digest(&a),
            shader_source::source_digest(&b)
        );
    }

    // The path rides the assembled text, so a renderer reassembling a stored
    // artifact's source has to use the path the cook compiled it under.
    #[test]
    fn the_same_field_under_two_paths_assembles_to_two_digests() {
        let text = "// one";
        let here = source(Family::Surface, Platform::Metal, field(text));
        let there = source(
            Family::Surface,
            Platform::Metal,
            SourceFile {
                path: "elsewhere/blob.hlsl",
                text,
            },
        );
        assert_ne!(
            shader_source::source_digest(&here),
            shader_source::source_digest(&there)
        );
    }

    // Against the real template, in every family: the field's lines number
    // from 1 under its own path, and every template line keeps the number it
    // has with the marker left in place, which is what a compiler reported
    // before the field was fenced.
    #[test]
    fn the_field_numbers_from_its_own_first_line_and_the_template_keeps_its_own() {
        let text = "float map(float3 p, SdfParams q, float t)\n{\n    return 1.0;\n}\n";
        for family in [Family::Surface, Family::Volumetric, Family::Shadow] {
            let fenced = source(family, Platform::Vulkan, field(text));
            let unspliced = shader_source::assemble_with_splices(
                FILE,
                Platform::Vulkan,
                &[(family.define(), "1")],
                crate::render::shaders::embedded,
                &[Splice::inline(BODY_MARKER, BODY_MARKER)],
            );
            let template: Vec<&str> = unspliced.lines().collect();
            let mut spliced = Vec::new();
            let mut current = FILE;
            let mut line = 1;
            for row in fenced.lines() {
                if let Some(rest) = row.strip_prefix("#line ") {
                    let (number, path) = rest.split_once(' ').expect("a line and a path");
                    line = number.parse().expect("a line number");
                    current = path.trim_matches('"');
                    continue;
                }
                if current == FILE {
                    assert!(
                        template[line - 1].ends_with(row),
                        "{family:?} {FILE}:{line} reads {row:?}, template has {:?}",
                        template[line - 1]
                    );
                } else {
                    assert_eq!(current, PATH);
                    spliced.push((line, row));
                }
                line += 1;
            }
            let want: Vec<(usize, &str)> =
                text.lines().enumerate().map(|(i, l)| (i + 1, l)).collect();
            assert_eq!(spliced, want, "{family:?}");
        }
    }
}
