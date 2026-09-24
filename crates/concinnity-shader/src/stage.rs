//! The pipeline stage an entry point declares, read off its
//! `[shader("...")]` attribute.
//!
//! dxc needs a `-T` profile on every invocation, and the stage half of it is a
//! property of the entry point rather than of the caller. Keeping it in the
//! source is what lets a program table name an entry point and nothing else;
//! the shader-model half follows from the target (see `HlslTarget`).

/// A pipeline stage, as the `[shader("...")]` attribute spells it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// `[shader("vertex")]`.
    Vertex,
    /// `[shader("pixel")]`.
    Pixel,
    /// `[shader("compute")]`.
    Compute,
}

impl Stage {
    /// The dxc profile for this stage at shader model `major.minor`.
    #[must_use]
    pub fn profile(self, major: u32, minor: u32) -> String {
        let prefix = match self {
            Stage::Vertex => "vs",
            Stage::Pixel => "ps",
            Stage::Compute => "cs",
        };
        format!("{prefix}_{major}_{minor}")
    }

    fn parse(word: &str) -> Option<Self> {
        match word {
            "vertex" => Some(Stage::Vertex),
            "pixel" => Some(Stage::Pixel),
            "compute" => Some(Stage::Compute),
            _ => None,
        }
    }
}

/// The stage `entry` declares in `source`.
///
/// Errs when the entry point carries no `[shader("...")]` attribute, which is
/// the only way a program table can name an entry the source does not export.
pub fn stage_of(source: &str, entry: &str) -> Result<Stage, String> {
    for (stage, name) in declared_stages(source) {
        if name == entry {
            return Ok(stage);
        }
    }
    Err(format!(
        "no `[shader(\"...\")]` entry point named `{entry}`: the stage of an \
         engine shader entry point is declared on the entry point itself"
    ))
}

// Every `[shader("...")]`-attributed entry point in `source`, as (stage, name).
// Attributes between the shader attribute and the signature are skipped, so a
// compute kernel's `[numthreads(...)]` does not read as the entry name.
fn declared_stages(source: &str) -> Vec<(Stage, &str)> {
    const MARKER: &str = "[shader(\"";
    let mut found = Vec::new();
    let mut at = 0;
    while let Some(start) = source[at..].find(MARKER) {
        let open = at + start + MARKER.len();
        let Some(end) = source[open..].find('"') else {
            break;
        };
        let stage = Stage::parse(&source[open..open + end]);
        at = open + end;
        if let Some(stage) = stage
            && let Some(name) = entry_name_after(source, at)
        {
            found.push((stage, name));
        }
    }
    found
}

// The declared name of the function following the attribute at `at`: the
// identifier immediately before the parameter list, skipping any further
// bracketed attributes on the way.
fn entry_name_after(source: &str, at: usize) -> Option<&str> {
    let bytes = source.as_bytes();
    let mut i = at;
    let mut last_ident: Option<(usize, usize)> = None;
    while i < bytes.len() {
        match bytes[i] {
            b'[' => {
                let mut depth = 0i32;
                while i < bytes.len() {
                    match bytes[i] {
                        b'[' => depth += 1,
                        b']' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    i += 1;
                }
                last_ident = None;
            }
            b'(' => return last_ident.map(|(s, e)| &source[s..e]),
            // A body or a statement terminator before any parameter list means
            // the attribute did not lead an entry point.
            b'{' | b'}' | b';' => return None,
            c if c.is_ascii_alphanumeric() || c == b'_' => {
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                last_ident = Some((start, i));
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = "\
[shader(\"vertex\")]\n\
VOut text_vertex_main(VIn v)\n{\n    return (VOut)0;\n}\n\
[shader(\"pixel\")]\n\
float4 text_fragment_main(VOut i) : SV_Target\n{\n    return 0;\n}\n\
[shader(\"compute\")]\n\
[numthreads(64, 1, 1)]\n\
void cull_kernel(uint3 id : SV_DispatchThreadID)\n{\n}\n";

    #[test]
    fn each_entry_point_reports_the_stage_it_declares() {
        assert_eq!(stage_of(SOURCE, "text_vertex_main"), Ok(Stage::Vertex));
        assert_eq!(stage_of(SOURCE, "text_fragment_main"), Ok(Stage::Pixel));
        assert_eq!(stage_of(SOURCE, "cull_kernel"), Ok(Stage::Compute));
    }

    // `[numthreads(...)]` sits between the attribute and the signature, and its
    // own parameter list would otherwise read as the entry point.
    #[test]
    fn an_attribute_between_the_marker_and_the_signature_is_skipped() {
        assert_eq!(declared_stages(SOURCE)[2], (Stage::Compute, "cull_kernel"));
    }

    #[test]
    fn an_entry_the_source_does_not_export_errs_naming_it() {
        let err = stage_of(SOURCE, "no_such_entry").unwrap_err();
        assert!(err.contains("no_such_entry"), "{err}");
    }

    // A stage spelling HLSL does not have is not silently taken for another.
    #[test]
    fn an_unknown_stage_spelling_declares_nothing() {
        assert!(declared_stages("[shader(\"fragment\")]\nfloat4 f() { return 0; }").is_empty());
    }

    #[test]
    fn a_profile_carries_the_stage_and_the_shader_model() {
        assert_eq!(Stage::Vertex.profile(6, 0), "vs_6_0");
        assert_eq!(Stage::Pixel.profile(6, 5), "ps_6_5");
        assert_eq!(Stage::Compute.profile(6, 0), "cs_6_0");
    }
}
