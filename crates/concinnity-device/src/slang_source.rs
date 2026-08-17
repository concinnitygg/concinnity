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

const POST_COMMON_SLANG: &str = include_str!("shaders/post_common.slang");
const POST_COMMON_MARKER: &str = "{POST_COMMON}";

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
    let base = read(hot_reload, file, embedded);
    let spliced = if base.contains(POST_COMMON_MARKER) {
        Cow::Owned(base.replace(
            POST_COMMON_MARKER,
            &read(hot_reload, "post_common.slang", POST_COMMON_SLANG),
        ))
    } else {
        base
    };
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
        assert!(!src.contains(POST_COMMON_MARKER));
        assert!(src.contains("float2 combined_size("));
        assert!(src.starts_with("A\n") && src.ends_with("\nB\n"));
    }

    // A shader with no marker keeps its text byte for byte, so the splice
    // cannot perturb the key of a program that does not use it.
    #[test]
    fn a_body_without_the_marker_is_untouched() {
        assert_eq!(assemble(false, "x.slang", "BODY\n", &[]), "BODY\n");
    }
}
