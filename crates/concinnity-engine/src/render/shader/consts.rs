//! Reads a constant's value back out of an engine shader, so a test can lock
//! the shader's copy of a size or threshold to the Rust one it mirrors.

/// The literal a `static const <type> <name> = <literal>;` or
/// `#define <name> <literal>` line in `source` assigns, with any `u` suffix
/// dropped. Panics when no such line exists: a renamed constant is the drift
/// the caller is checking for.
pub(crate) fn literal<'a>(source: &'a str, name: &str) -> &'a str {
    source
        .lines()
        .find_map(|line| {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("#define ") {
                let (n, value) = rest.split_once(char::is_whitespace)?;
                return (n == name).then(|| value.trim());
            }
            let rest = line.strip_prefix("static const ")?;
            let (decl, value) = rest.split_once('=')?;
            let n = decl.split_whitespace().last()?;
            (n == name).then(|| value.split_once(';').map_or(value, |v| v.0).trim())
        })
        .unwrap_or_else(|| panic!("shader declares no constant `{name}`"))
        .trim_end_matches('u')
}

/// [`literal`] as an unsigned integer.
pub(crate) fn uint(source: &str, name: &str) -> usize {
    let text = literal(source, name);
    let parsed = match text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        Some(hex) => usize::from_str_radix(hex, 16),
        None => text.parse(),
    };
    parsed.unwrap_or_else(|e| panic!("`{name} = {text}` is not an integer: {e}"))
}

/// [`literal`] as a float.
pub(crate) fn float(source: &str, name: &str) -> f32 {
    let text = literal(source, name);
    text.parse()
        .unwrap_or_else(|e| panic!("`{name} = {text}` is not a float: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = "\
static const uint  A = 4u;
static const float B   = 0.25;
static const int   C = 48;
#define D 32
static const uint E = 0xFFu;
";

    #[test]
    fn reads_each_declaration_form() {
        assert_eq!(uint(SRC, "A"), 4);
        assert_eq!(float(SRC, "B"), 0.25);
        assert_eq!(uint(SRC, "C"), 48);
        assert_eq!(uint(SRC, "D"), 32);
        assert_eq!(uint(SRC, "E"), 255);
    }

    #[test]
    #[should_panic(expected = "no constant `Z`")]
    fn a_missing_constant_panics() {
        literal(SRC, "Z");
    }
}
