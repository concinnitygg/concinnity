//! Source assembly for the single-source `.slang` engine shaders.
//!
//! Every consumer that compiles one of `src/shaders/*.slang` assembles the same
//! text the same way: the file's body, its `{...}` fragment markers replaced,
//! and the program's variant defines injected ahead of it. Both the fragments
//! and the defines ride the text rather than an include path or a command line,
//! so the backends' content-addressed shader cache keys them -- which is what
//! keeps two pool sizes, two Hi-Z variants, or two revisions of a shared helper
//! from ever sharing an artifact.
//!
//! It lives here, below the device backends, because the assembly is what they
//! and the build script have to agree on: a build script's precompile and a
//! renderer's runtime compile must produce byte-identical source for the
//! content-addressed cache to be sound across them. Keeping one implementation
//! is what makes that true by construction rather than by review.
//!
//! Reading a shader off disk is not part of it. Hot-reload wants the checkout's
//! copy to win over the embedded one, and that is a `std` filesystem concern in
//! a `no_std` crate -- so `assemble_with` takes a resolver and the device crate
//! supplies the one that reads disk first.

use alloc::borrow::Cow;
use alloc::string::{String, ToString};

use crate::render::shaders;

/// The shared fragments, as (marker, file). A shader carrying a marker has the
/// named file's text spliced in at that point.
///
/// Three of them come in pairs, because a shader's resource bindings sit
/// between the halves: MAIN_TYPES / PROBE_TYPES / RT_TYPES declare the records
/// a binding names, and MAIN_SHADING / PROBE_COMMON / RT_TRACE the code that
/// reads the bound resources.
///
/// PARTICLE_TYPES is the one shared by two halves of a *system* rather than of
/// a shader: the simulation kernel writes the pool the render pair reads.
///
/// The order is load-bearing where one fragment carries another's marker: the
/// splice runs the table once, so a nested marker is only replaced if its own
/// row comes later.
pub const FRAGMENTS: &[(&str, &str)] = &[
    ("{POST_COMMON}", "post_common.slang"),
    // MAIN_TYPES leads OBJECT_COMMON because it carries that marker itself:
    // the object record belongs with the rest of the main pass's vocabulary,
    // and a fragment spliced after its own marker has been passed would land
    // unreplaced.
    ("{MAIN_TYPES}", "main_types.slang"),
    ("{OBJECT_COMMON}", "object_common.slang"),
    ("{PROBE_TYPES}", "probe_types.slang"),
    ("{PROBE_COMMON}", "probe_common.slang"),
    ("{RT_TYPES}", "rt_types.slang"),
    ("{RT_TRACE}", "rt_trace.slang"),
    ("{PARTICLE_TYPES}", "particle_types.slang"),
    // RAYMARCH_TYPES leads LIGHT_TYPES for the reason MAIN_TYPES leads
    // OBJECT_COMMON: it carries that marker itself, and the main pass's own
    // splice carries it too, so the light records land whichever half asked.
    ("{RAYMARCH_TYPES}", "raymarch_types.slang"),
    ("{LIGHT_TYPES}", "light_types.slang"),
    ("{RAYMARCH_COMMON}", "raymarch_common.slang"),
    ("{MAIN_SHADING}", "main_shading.slang"),
    // SHADOW_BIAS trails both halves that carry it: the cascade compare
    // offset is shared by the main pass and the raymarched surfaces, and a
    // row placed ahead of theirs would leave the marker unreplaced.
    ("{SHADOW_BIAS}", "shadow_bias.slang"),
    // The two hooks a world Shader defines, with the engine's own shading as
    // the default. A world compile passes its files as caller splices for the
    // same markers, which take precedence over these rows.
    ("{SURFACE_VERTEX}", "surface_vertex_default.slang"),
    ("{SURFACE_FRAGMENT}", "surface_fragment_default.slang"),
];

/// Prepend a `#define` line per `(name, value)` pair. The defines become part
/// of the source text on purpose: the shader cache keys on the assembled
/// source, so two pool sizes can never share an artifact.
pub fn inject_defines(source: &str, defines: &[(&str, &str)]) -> String {
    if defines.is_empty() {
        return source.to_string();
    }
    let mut out = String::with_capacity(source.len() + defines.len() * 32);
    for (name, value) in defines {
        out.push_str("#define ");
        out.push_str(name);
        out.push(' ');
        out.push_str(value);
        out.push('\n');
    }
    out.push_str(source);
    out
}

/// The exact source text a program compiles, from the embedded shaders alone.
/// `file` names the `.slang` under `src/shaders/`.
pub fn assemble(file: &str, defines: &[(&str, &str)]) -> String {
    assemble_with(file, defines, shaders::embedded)
}

/// Digest of one assembled source text, identifying the artifact a build script
/// compiled from it.
///
/// A precompiled artifact is only usable if the source still matches the one it
/// was built from, and the name alone cannot say so: hot-reload exists precisely
/// to compile an edited shader, and a build-time artifact keyed by name would
/// shadow the edit. Comparing digests makes the embedded copy a content hit --
/// used whenever the text is unchanged, skipped the moment it is not, in any
/// build and under any flag.
///
/// FNV-1a, because it is over a few kilobytes on a path that then either does
/// nothing or invokes a compiler; the cost has to disappear next to both.
pub fn source_digest(source: &str) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x1000_0000_01b3;
    let mut hash = OFFSET;
    for byte in source.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// The same assembly against a caller-supplied resolver, which is how the
/// device crate lets a hot-reload build prefer the checkout's copy of a shader
/// (and of every fragment it splices) over the embedded one. A resolver that
/// returns `None` falls back to the embedded text, so a missing file loses the
/// edit rather than the shader.
pub fn assemble_with(
    file: &str,
    defines: &[(&str, &str)],
    resolve: impl Fn(&str) -> Option<&'static str>,
) -> String {
    assemble_with_splices(file, defines, resolve, &[])
}

/// The same assembly with caller-supplied text spliced in as well, for a shader
/// whose source is only complete once something outside the shader tree is
/// known. The raymarched SDF volumes are the case: a world authors the distance
/// field, so `{SDF_BODY}` is filled from the volume's payload rather than from
/// a file the table can name.
///
/// A caller's splice wins over a [`FRAGMENTS`] row for the same marker, which
/// is how a world Shader's hooks replace the engine's default ones; the
/// caller's splices also run again after the table, so a fragment may carry
/// one of their markers and still have it filled. Like the defines and the
/// shared fragments, they ride the text, which is what makes the
/// content-addressed shader cache key two worlds' shaders apart.
pub fn assemble_with_splices(
    file: &str,
    defines: &[(&str, &str)],
    resolve: impl Fn(&str) -> Option<&'static str>,
    splices: &[(&str, &str)],
) -> String {
    let mut spliced = read(file, &resolve);
    // The table fills every marker the caller does not claim, then the
    // caller's text lands once, after it, so a table marker spelled inside a
    // world file is never expanded and a marker a fragment introduces is
    // still reached.
    for (marker, fragment_file) in FRAGMENTS {
        if spliced.contains(marker) && !splices.iter().any(|(m, _)| m == marker) {
            let text = read(fragment_file, &resolve);
            spliced = Cow::Owned(spliced.replace(marker, &text));
        }
    }
    spliced = splice_all(spliced, splices);
    inject_defines(&spliced, defines)
}

fn splice_all<'a>(mut text: Cow<'a, str>, splices: &[(&str, &str)]) -> Cow<'a, str> {
    for (marker, fill) in splices {
        if text.contains(marker) {
            text = Cow::Owned(text.replace(marker, fill));
        }
    }
    text
}

fn read(file: &str, resolve: &impl Fn(&str) -> Option<&'static str>) -> Cow<'static, str> {
    match resolve(file).or_else(|| shaders::embedded(file)) {
        Some(text) => Cow::Borrowed(text),
        // A name no table carries: leave it empty rather than panicking in a
        // renderer. The compile that follows reports the real error.
        None => Cow::Borrowed(""),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn defines_lead_the_body() {
        assert_eq!(
            inject_defines("BODY\n", &[("A", "1"), ("B", "2")]),
            "#define A 1\n#define B 2\nBODY\n"
        );
        assert_eq!(inject_defines("BODY\n", &[]), "BODY\n");
    }

    // The marker is replaced by the shared text, and the replacement lands in
    // the assembled source rather than behind an include path -- which is what
    // makes the content-addressed cache key cover it.
    #[test]
    fn the_post_common_marker_is_spliced_into_the_body() {
        let src = assemble_with("x.slang", &[], |f| {
            (f == "x.slang").then_some("A\n{POST_COMMON}\nB\n")
        });
        assert!(!src.contains("{POST_COMMON}"));
        assert!(src.contains("float2 combined_size("));
        assert!(src.starts_with("A\n") && src.ends_with("\nB\n"));
    }

    // The probe and ray-tracing fragments each splice in two halves, and the
    // order matters: the record declarations have to precede the helpers that
    // read them, because a shader puts its resource bindings between the two.
    #[test]
    fn the_paired_fragments_splice_records_before_helpers() {
        let body = "{PROBE_TYPES}\n{PROBE_COMMON}\n{RT_TYPES}\n{RT_TRACE}\n";
        let src = assemble_with("x.slang", &[], |f| (f == "x.slang").then_some(body));
        for (marker, _) in FRAGMENTS {
            assert!(!src.contains(marker), "unspliced {marker}");
        }
        assert!(src.find("struct ProbeSet") < src.find("float3 probe_set_specular("));
        assert!(src.find("struct RtGeomEntry") < src.find("bool rt_trace_reflection("));
    }

    // A resolver that answers wins over the embedded copy; one that declines
    // falls back to it. This is the whole of what hot-reload needs from here.
    #[test]
    fn the_resolver_overrides_the_embedded_copy_and_declining_falls_back() {
        let overridden = assemble_with("fog.slang", &[], |f| {
            (f == "fog.slang").then_some("OVERRIDDEN\n")
        });
        assert_eq!(overridden, "OVERRIDDEN\n");
        assert_eq!(
            assemble_with("fog.slang", &[], |_| None),
            assemble("fog.slang", &[])
        );
    }

    // A caller-supplied splice fills a marker no file in the table names, which
    // is how a world's own distance field reaches the raymarch source.
    #[test]
    fn a_caller_splice_fills_a_marker_the_table_does_not_name() {
        let src = assemble_with_splices(
            "x.slang",
            &[],
            |f| (f == "x.slang").then_some("A\n{SDF_BODY}\nB\n"),
            &[("{SDF_BODY}", "float map() { return 1.0; }")],
        );
        assert_eq!(src, "A\nfloat map() { return 1.0; }\nB\n");
    }

    // The caller's splices run after the table's, so a shared fragment may
    // carry one of their markers. Nothing does today; the ordering is what
    // keeps that from becoming a silent unreplaced marker if one ever does.
    #[test]
    fn a_caller_splice_reaches_a_marker_inside_a_shared_fragment() {
        let src = assemble_with_splices(
            "x.slang",
            &[],
            |f| match f {
                "x.slang" => Some("{POST_COMMON}\n"),
                "post_common.slang" => Some("frag {SDF_BODY} end"),
                _ => None,
            },
            &[("{SDF_BODY}", "FILLED")],
        );
        assert_eq!(src, "frag FILLED end\n");
    }

    // A world file lands verbatim: a table marker spelled inside it (in a
    // comment, say) is not expanded, and its text is never rescanned.
    #[test]
    fn a_caller_splice_is_not_rescanned_for_table_markers() {
        let src = assemble_with_splices(
            "x.slang",
            &[],
            |f| (f == "x.slang").then_some("{SURFACE_FRAGMENT}\n"),
            &[(
                "{SURFACE_FRAGMENT}",
                "// see {MAIN_TYPES} and {SURFACE_VERTEX}\n",
            )],
        );
        assert_eq!(src, "// see {MAIN_TYPES} and {SURFACE_VERTEX}\n\n");
    }

    // A caller splice for a marker the table also names replaces the table's
    // default: a world Shader's hooks land where the engine's own would.
    #[test]
    fn a_caller_splice_wins_over_a_table_row_for_the_same_marker() {
        let body = "{SURFACE_VERTEX}\n{SURFACE_FRAGMENT}\n";
        let src = assemble_with_splices(
            "x.slang",
            &[],
            |f| (f == "x.slang").then_some(body),
            &[("{SURFACE_FRAGMENT}", "WORLD SHADE")],
        );
        assert!(src.contains("WORLD SHADE"));
        assert!(
            !src.contains("float4 shade(VertexOut in, GpuObjectData od)\n{"),
            "default shade replaced"
        );
        assert!(
            src.contains("VertexOut transform("),
            "default transform kept"
        );
    }

    // The two halves of the raymarch splice land in the order the source needs:
    // the records a binding names, then the body that reads them, then the
    // world's own field after both. The light records arrive through the types
    // half, so a raymarch shader never declares them itself.
    #[test]
    fn the_raymarch_source_assembles_records_then_body_then_the_world_field() {
        let src = assemble_with_splices(
            "raymarch.slang",
            &[("RAYMARCH_METAL", "1"), ("RAYMARCH_SURFACE", "1")],
            shaders::embedded,
            &[("{SDF_BODY}", "// the world's field")],
        );
        for (marker, _) in FRAGMENTS {
            assert!(!src.contains(marker), "unspliced {marker}");
        }
        assert!(!src.contains("{SDF_BODY}"));
        let types = src.find("struct SdfVolumeUniforms").expect("records");
        let lights = src.find("struct LightUniforms").expect("light records");
        let body = src.find("RayHit coneRaymarch(").expect("marcher");
        let field = src.find("// the world's field").expect("world field");
        let entry = src
            .find("RaymarchFragOut raymarch_fragment(")
            .expect("entry");
        assert!(lights < types, "light records precede the volume block");
        assert!(types < body, "records precede the body that reads them");
        assert!(
            body < field,
            "the field is declared after the helpers call it"
        );
        assert!(field < entry, "the entry points come last");
    }

    // Both halves of the light-record splice resolve: the main pass carries it
    // through its own types fragment and the raymarch pass through its. A
    // shader that declared these itself would be the drift the split removes.
    #[test]
    fn both_passes_take_the_light_records_from_one_fragment() {
        for file in ["main_bindless.slang", "raymarch.slang"] {
            let src = assemble(file, &[]);
            assert!(!src.contains("{LIGHT_TYPES}"), "{file} left the marker");
            assert_eq!(
                src.matches("struct LightUniforms").count(),
                1,
                "{file} declares the light block other than once"
            );
        }
    }

    // A body without the marker keeps its text byte for byte, so the splice
    // cannot perturb the key of a program that does not use it.
    #[test]
    fn a_body_without_the_marker_is_untouched() {
        let src = assemble_with("x.slang", &[], |f| (f == "x.slang").then_some("BODY\n"));
        assert_eq!(src, "BODY\n");
    }

    // Every fragment the table names has to exist, or a shader carrying its
    // marker would silently splice in nothing.
    #[test]
    fn every_fragment_the_table_names_is_embedded() {
        for (marker, file) in FRAGMENTS {
            assert!(
                shaders::embedded(file).is_some(),
                "{marker} names a missing {file}"
            );
        }
    }

    // The lookup and the table are the same set, and every name is unique --
    // a duplicate would make `embedded` return whichever came first.
    #[test]
    fn the_source_table_is_a_unique_set_the_lookup_covers() {
        let mut names: Vec<&str> = shaders::SOURCES.iter().map(|(n, _)| *n).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count, "duplicate shader name");
        for (name, text) in shaders::SOURCES {
            assert_eq!(shaders::embedded(name), Some(*text));
        }
        assert_eq!(shaders::embedded("not_a_shader.slang"), None);
    }
}
