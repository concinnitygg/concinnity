// Source assembly for the single-source `.slang` engine shaders.
//
// Every backend that compiles one of `src/shaders/*.slang` assembles the same
// text the same way: the file (from disk under hot-reload so an edit wins,
// from the embedded copy otherwise), its `{...}` fragment markers replaced, and
// the program's variant defines injected ahead of it. Both the fragments and
// the defines ride the text rather than an include path or a command line, so
// `shader_cache` keys them -- which is what keeps two pool sizes, two Hi-Z
// variants, or two revisions of a shared helper from ever sharing an artifact.
//
// It lives here rather than in a backend module because the assembly is what
// the backends have to agree on: a build script's precompile and a renderer's
// runtime compile must produce byte-identical source for the content-addressed
// cache to be sound across them. `SLANG_SHADER_FRAGMENTS` in the device build
// script is the build-time half of the same table, exactly as
// `METAL_SHADER_FRAGMENTS` mirrors the `.metal` splice.

use std::borrow::Cow;

// The shared fragments, as (marker, file, embedded copy). `SLANG_SHADER_FRAGMENTS`
// in the device build script is the build-time half of this table; the two must
// agree or a build script and a renderer would key different text for the same
// program.
//
// Two of them come in pairs, because a shader's resource bindings sit between
// the halves: PROBE_TYPES / RT_TYPES declare the records a binding names, and
// PROBE_COMMON / RT_TRACE the code that reads the bound resources.
const FRAGMENTS: &[(&str, &str, &str)] = &[
    (
        "{POST_COMMON}",
        "post_common.slang",
        include_str!("shaders/post_common.slang"),
    ),
    (
        "{OBJECT_COMMON}",
        "object_common.slang",
        include_str!("shaders/object_common.slang"),
    ),
    (
        "{PROBE_TYPES}",
        "probe_types.slang",
        include_str!("shaders/probe_types.slang"),
    ),
    (
        "{PROBE_COMMON}",
        "probe_common.slang",
        include_str!("shaders/probe_common.slang"),
    ),
    (
        "{RT_TYPES}",
        "rt_types.slang",
        include_str!("shaders/rt_types.slang"),
    ),
    (
        "{RT_TRACE}",
        "rt_trace.slang",
        include_str!("shaders/rt_trace.slang"),
    ),
];

// The exact source text a program compiles. `file` names the `.slang` under
// `src/shaders/`; `embedded` is its `include_str!` copy, used when hot-reload
// is off and as the fallback when the disk read fails (a shipped binary has no
// checkout to read from).
pub(crate) fn assemble(
    hot_reload: bool,
    file: &str,
    embedded: &'static str,
    defines: &[(&str, &str)],
) -> String {
    let mut spliced = read(hot_reload, file, embedded);
    for (marker, fragment_file, fragment) in FRAGMENTS {
        if spliced.contains(marker) {
            let text = read(hot_reload, fragment_file, fragment);
            spliced = Cow::Owned(spliced.replace(marker, &text));
        }
    }
    concinnity_slang::inject_defines(&spliced, defines)
}

// One `.slang` file's text: the checkout's copy under hot-reload, the embedded
// copy otherwise (and whenever the disk read fails).
fn read(hot_reload: bool, file: &str, embedded: &'static str) -> Cow<'static, str> {
    if !hot_reload {
        return Cow::Borrowed(embedded);
    }
    let path = format!("{}/src/shaders/{}", env!("CARGO_MANIFEST_DIR"), file);
    match std::fs::read_to_string(&path) {
        Ok(s) => Cow::Owned(s),
        Err(e) => {
            tracing::debug!("hot-reload: falling back to embedded {file} ({e})");
            Cow::Borrowed(embedded)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defines_lead_the_embedded_body() {
        let src = assemble(
            false,
            "nonexistent.slang",
            "BODY\n",
            &[("A", "1"), ("B", "2")],
        );
        assert_eq!(src, "#define A 1\n#define B 2\nBODY\n");
    }

    #[test]
    fn no_defines_is_the_body_verbatim() {
        assert_eq!(assemble(false, "x.slang", "BODY\n", &[]), "BODY\n");
    }

    // A hot-reload read that cannot find the file must not lose the program:
    // the embedded copy stands in, so a shipped binary with hot-reload on still
    // compiles the shader it shipped with.
    #[test]
    fn a_missing_hot_reload_file_falls_back_to_embedded() {
        let src = assemble(
            true,
            "definitely_not_a_shader.slang",
            "BODY\n",
            &[("A", "1")],
        );
        assert_eq!(src, "#define A 1\nBODY\n");
    }

    // The marker is replaced by the shared text, and the replacement lands in
    // the assembled source rather than behind an include path -- which is what
    // makes the content-addressed cache key cover it.
    #[test]
    fn the_post_common_marker_is_spliced_into_the_body() {
        let src = assemble(false, "x.slang", "A\n{POST_COMMON}\nB\n", &[]);
        assert!(!src.contains("{POST_COMMON}"));
        assert!(src.contains("float2 combined_size("));
        assert!(src.starts_with("A\n") && src.ends_with("\nB\n"));
    }

    // The per-object record, the object-id reconstruction and the normal matrix
    // all arrive from one fragment, so the three passes that stride the object
    // buffer cannot drift apart in their declaration of it.
    #[test]
    fn the_object_common_marker_is_spliced_into_the_body() {
        let src = assemble(false, "x.slang", "A\n{OBJECT_COMMON}\nB\n", &[]);
        assert!(!src.contains("{OBJECT_COMMON}"));
        assert!(src.contains("struct GpuObjectData"));
        assert!(src.contains("uint object_instance_index("));
        assert!(src.contains("float3x3 normal_matrix("));
    }

    // A body carrying both markers gets both, so the table is a loop rather
    // than a single special case.
    #[test]
    fn every_marker_in_one_body_is_spliced() {
        let src = assemble(false, "x.slang", "{POST_COMMON}\n{OBJECT_COMMON}\n", &[]);
        assert!(src.contains("float2 combined_size("));
        assert!(src.contains("struct GpuObjectData"));
    }

    // The probe and ray-tracing fragments each splice in two halves, and the
    // order matters: the record declarations have to precede the helpers that
    // read them, because a shader puts its resource bindings between the two.
    #[test]
    fn the_paired_fragments_splice_records_before_helpers() {
        let src = assemble(
            false,
            "x.slang",
            "{PROBE_TYPES}\n{PROBE_COMMON}\n{RT_TYPES}\n{RT_TRACE}\n",
            &[],
        );
        for marker in [
            "{PROBE_TYPES}",
            "{PROBE_COMMON}",
            "{RT_TYPES}",
            "{RT_TRACE}",
        ] {
            assert!(!src.contains(marker), "unspliced {marker}");
        }
        assert!(src.find("struct ProbeSet") < src.find("float3 probe_set_specular("));
        assert!(src.find("struct RtGeomEntry") < src.find("bool rt_trace_reflection("));
    }

    // A shader with no marker keeps its text byte for byte, so the splice
    // cannot perturb the key of a program that does not use it.
    #[test]
    fn a_body_without_the_marker_is_untouched() {
        assert_eq!(assemble(false, "x.slang", "BODY\n", &[]), "BODY\n");
    }
}
