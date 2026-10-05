//! Which functions a failed compile called without a definition.
//!
//! dxc words this differently per target. The SPIR-V leg (Vulkan, and Metal
//! through it) reports a located `found undefined function` whose caret sits
//! on the call; DXIL reports an unlocated `External function used in
//! non-library profile` naming the function by its mangled symbol.

use super::{Severity, parse};

const SPIRV_MESSAGE: &str = "found undefined function";
const DXIL_MESSAGE: &str = "External function used in non-library profile: ";

/// The name of every function `output` reports as called but never defined,
/// in the order dxc printed them.
pub fn undefined_functions(output: &str) -> Vec<String> {
    let mut names: Vec<String> = parse(output)
        .iter()
        .filter(|d| d.severity == Severity::Error && d.message == SPIRV_MESSAGE)
        .filter_map(|d| called_at_caret(&d.context))
        .collect();
    names.extend(
        output
            .lines()
            .filter_map(|line| line.split_once(DXIL_MESSAGE))
            .filter_map(|(_, symbol)| demangled_name(symbol.trim())),
    );
    names
}

// The identifier under the caret, read from the excerpt above it. The caret is
// aligned with the excerpt as printed, which is not the header's column once a
// tab has been expanded.
fn called_at_caret(context: &str) -> Option<String> {
    let mut lines = context.lines();
    let excerpt = lines.next()?;
    let at = lines.next()?.chars().position(|c| c == '^')?;
    let name: String = excerpt.chars().skip(at).take_while(is_ident).collect();
    (!name.is_empty()).then_some(name)
}

// `\01?shade@@YA...` is MSVC's mangling of a global function `shade`; an
// unmangled symbol is its own name.
fn demangled_name(symbol: &str) -> Option<String> {
    let symbol = symbol.strip_prefix("\\01").unwrap_or(symbol);
    let name: String = match symbol.strip_prefix('?') {
        Some(mangled) => mangled.chars().take_while(|&c| c != '@').collect(),
        None => symbol.chars().take_while(is_ident).collect(),
    };
    (!name.is_empty() && name.chars().all(|c| is_ident(&c))).then_some(name)
}

fn is_ident(c: &char) -> bool {
    c.is_ascii_alphanumeric() || *c == '_'
}

#[cfg(test)]
mod tests {
    use super::*;

    // The SPIR-V leg's wording, which a Vulkan or Metal compile prints.
    const SPIRV: &str = "hlsl: dxc failed for main_bindless.hlsl:
main_bindless.hlsl:1540:12: error: found undefined function
    return shade(v, od);
           ^

";

    // The DXIL wording, which a D3D12 compile prints.
    const DXIL: &str = "hlsl: dxc failed for main_bindless.hlsl:
error: External function used in non-library profile: \
\\01?shade@@YA?AV?$vector@M$03@@UVertexOut@@UGpuObjectData@@@Z

";

    #[test]
    fn the_spirv_wording_names_the_function_under_the_caret() {
        assert_eq!(undefined_functions(SPIRV), ["shade"]);
    }

    #[test]
    fn the_dxil_wording_names_the_demangled_function() {
        assert_eq!(undefined_functions(DXIL), ["shade"]);
    }

    // dxc expands a tab in the excerpt, so the caret, not the header's column,
    // is what lines up with the name.
    #[test]
    fn the_caret_is_read_against_the_excerpt_as_printed() {
        let output = "frag.hlsl:2:5: error: found undefined function
        float d = map(p, params, time);
                  ^
";
        assert_eq!(undefined_functions(output), ["map"]);
    }

    #[test]
    fn every_undefined_function_is_named_in_order() {
        let output = format!(
            "{SPIRV}error: External function used in non-library profile: \
             ?sampleVolume@@YA?AUVolumeSample@@V?$vector@M$02@@USdfParams@@M@Z\n"
        );
        assert_eq!(undefined_functions(&output), ["shade", "sampleVolume"]);
    }

    #[test]
    fn an_unmangled_symbol_is_its_own_name() {
        let output = "error: External function used in non-library profile: transform\n";
        assert_eq!(undefined_functions(output), ["transform"]);
    }

    // Other errors, a caretless excerpt, and a symbol that does not demangle
    // to a plain name report nothing.
    #[test]
    fn other_failures_name_no_function() {
        assert!(undefined_functions("").is_empty());
        assert!(
            undefined_functions(
                "frag.hlsl:2:16: error: expected ';' at end of declaration\n  float y = 1.0\n               ^\n"
            )
            .is_empty()
        );
        assert!(undefined_functions("frag.hlsl:2:5: error: found undefined function\n").is_empty());
        assert!(
            undefined_functions(
                "error: External function used in non-library profile: \\01??$f@M@@YAMM@Z\n"
            )
            .is_empty()
        );
    }
}
