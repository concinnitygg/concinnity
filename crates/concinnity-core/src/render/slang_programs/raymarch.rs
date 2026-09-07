//! What the raymarched SDF volume pass compiles, on every backend.
//!
//! The other two tables here are per-backend because the backends compile
//! different sets. This one is shared: all three compile the same six entries
//! out of `raymarch.slang`, differing only in the ABI define and in what slangc
//! is asked to emit. The cook iterates it to compile a volume's field ahead of
//! time and each renderer iterates it to find what the cook left.

use alloc::string::String;

use crate::platform::Platform;
use crate::render::slang_source;

/// The shader file every entry below compiles from.
pub const FILE: &str = "raymarch.slang";

/// The marker a volume's authored distance field is spliced at.
pub const BODY_MARKER: &str = "{SDF_BODY}";

/// The helper an authored `shade` calls to read the scene behind the surface.
pub const SCENE_TAP: &str = "sampleSceneRefracted";

/// Whether an authored distance field reads the scene behind the surface.
///
/// The tap is the only way in: the scene snapshot is reached through this
/// helper and is named by no other declaration a field can see. A renderer
/// copies the frame's colour target for the pass only when some visible volume
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
    /// An opaque surface writing colour and depth.
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

/// Whether an entry runs at the vertex or the fragment stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Rasterises the bounding-box proxy.
    Vertex,
    /// Marches the field and writes the draw's output.
    Fragment,
}

/// One entry point of one family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Program {
    /// Entry point name, as the source spells it.
    pub entry: &'static str,
    /// Which stage it compiles for.
    pub stage: Stage,
    /// The draw it belongs to.
    pub family: Family,
}

/// Every entry `raymarch.slang` declares.
pub const ALL: &[Program] = &[
    Program {
        entry: "raymarch_vertex",
        stage: Stage::Vertex,
        family: Family::Surface,
    },
    Program {
        entry: "raymarch_fragment",
        stage: Stage::Fragment,
        family: Family::Surface,
    },
    Program {
        entry: "raymarch_volumetric_vertex",
        stage: Stage::Vertex,
        family: Family::Volumetric,
    },
    Program {
        entry: "raymarch_volumetric_fragment",
        stage: Stage::Fragment,
        family: Family::Volumetric,
    },
    Program {
        entry: "raymarch_shadow_vertex",
        stage: Stage::Vertex,
        family: Family::Shadow,
    },
    Program {
        entry: "raymarch_shadow_fragment",
        stage: Stage::Fragment,
        family: Family::Shadow,
    },
];

/// The ABI define naming a host's binding layout. Vulkan takes the source's
/// `#else` branch and so needs none, which is why this is an `Option`.
pub fn abi_define(platform: Platform) -> Option<&'static str> {
    match platform {
        Platform::Metal => Some("RAYMARCH_METAL"),
        Platform::Hlsl => Some("RAYMARCH_DXIL"),
        Platform::Glsl => None,
    }
}

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

/// The variant defines for one family on one host.
pub fn defines(
    family: Family,
    platform: Platform,
) -> alloc::vec::Vec<(&'static str, &'static str)> {
    let mut out = alloc::vec::Vec::with_capacity(2);
    if let Some(abi) = abi_define(platform) {
        out.push((abi, "1"));
    }
    out.push((family.define(), "1"));
    out
}

/// The exact source text one family compiles for one host, with `field` spliced
/// in as the world's distance field. `resolve` lets a hot-reload build prefer
/// the checkout's copy of the template over the embedded one.
pub fn source_with(
    family: Family,
    platform: Platform,
    field: &str,
    resolve: impl Fn(&str) -> Option<&'static str>,
) -> String {
    slang_source::assemble_with_splices(
        FILE,
        &defines(family, platform),
        resolve,
        &[(BODY_MARKER, field)],
    )
}

/// The same source from the embedded templates alone.
pub fn source(family: Family, platform: Platform, field: &str) -> String {
    source_with(family, platform, field, crate::render::shaders::embedded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

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

    // Vulkan takes the source's `#else` branch, so its only define is the
    // family; the other two name their binding block as well.
    #[test]
    fn the_defines_name_the_abi_only_where_the_source_has_a_branch_for_it() {
        assert_eq!(
            defines(Family::Surface, Platform::Metal),
            [("RAYMARCH_METAL", "1"), ("RAYMARCH_SURFACE", "1")]
        );
        assert_eq!(
            defines(Family::Shadow, Platform::Hlsl),
            [("RAYMARCH_DXIL", "1"), ("RAYMARCH_SHADOW", "1")]
        );
        assert_eq!(
            defines(Family::Volumetric, Platform::Glsl),
            [("RAYMARCH_VOLUMETRIC", "1")]
        );
    }

    // Every entry the table names appears in the source it claims to come from,
    // so a renamed entry point fails here rather than at a renderer's init.
    #[test]
    fn every_entry_the_table_names_is_declared_in_the_source() {
        let text = crate::render::shaders::embedded(FILE).expect("raymarch.slang");
        for program in ALL {
            assert!(
                text.contains(program.entry),
                "{} names no entry in {FILE}",
                program.entry
            );
        }
        assert!(text.contains(BODY_MARKER), "{FILE} carries no body marker");
    }

    // The tap the detection looks for is the one the helpers declare, so a
    // rename in the shader fails here rather than by silently making every
    // refractive volume read a stale scene.
    #[test]
    fn the_scene_tap_is_declared_by_the_helpers() {
        let text =
            crate::render::shaders::embedded("raymarch_common.slang").expect("raymarch_common");
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
        let src = source(Family::Surface, Platform::Metal, opaque);
        assert!(src.contains(SCENE_TAP), "the template declares the tap");
        assert!(!field_taps_scene(opaque));
    }

    // The field reaches the assembled source and the defines lead it, on every
    // host. A family's source must also differ per host, or two backends would
    // share a cache entry for different binding layouts.
    #[test]
    fn the_field_is_spliced_and_the_hosts_assemble_differently() {
        let field = "float map(float3 p, SdfParams q, float t) { return 1.0; }";
        let mut seen = Vec::new();
        for platform in [Platform::Metal, Platform::Hlsl, Platform::Glsl] {
            let src = source(Family::Surface, platform, field);
            assert!(src.contains(field), "{platform:?} lost the field");
            assert!(!src.contains(BODY_MARKER), "{platform:?} left the marker");
            assert!(src.starts_with("#define "), "{platform:?} defines lead");
            seen.push(slang_source::source_digest(&src));
        }
        seen.dedup();
        assert_eq!(seen.len(), 3, "two hosts assemble identical source");
    }

    // Two fields are two sources, which is what keeps the content-addressed
    // cache from serving one world's volume the artifact of another's.
    #[test]
    fn two_fields_assemble_to_two_digests() {
        let a = source(Family::Surface, Platform::Metal, "// one");
        let b = source(Family::Surface, Platform::Metal, "// two");
        assert_ne!(
            slang_source::source_digest(&a),
            slang_source::source_digest(&b)
        );
    }
}
