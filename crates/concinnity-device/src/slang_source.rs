// Source assembly for the single-source `.slang` engine shaders.
//
// Every backend that compiles one of `src/shaders/*.slang` assembles the same
// text the same way: the file (from disk under hot-reload so an edit wins,
// from the embedded copy otherwise) with the program's variant defines
// injected ahead of it. The defines ride the text rather than the command line
// so `shader_cache` keys them, which is what keeps two pool sizes or two Hi-Z
// variants from ever sharing an artifact.
//
// It lives here rather than in a backend module because the assembly is what
// the backends have to agree on: a build script's precompile and a renderer's
// runtime compile must produce byte-identical source for the content-addressed
// cache to be sound across them.

use std::borrow::Cow;

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
    let base: Cow<'static, str> = if hot_reload {
        let path = format!("{}/src/shaders/{}", env!("CARGO_MANIFEST_DIR"), file);
        match std::fs::read_to_string(&path) {
            Ok(s) => Cow::Owned(s),
            Err(e) => {
                tracing::debug!("hot-reload: falling back to embedded {file} ({e})");
                Cow::Borrowed(embedded)
            }
        }
    } else {
        Cow::Borrowed(embedded)
    };
    concinnity_slang::inject_defines(&base, defines)
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
}
