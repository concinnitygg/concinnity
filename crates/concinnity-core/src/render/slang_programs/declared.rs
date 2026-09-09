//! The scan behind each backend table's completeness check.
//!
//! `ALL` is what both the build script and the bundle export iterate, so a
//! program declared here but left off it compiles at renderer init on every
//! host instead of being carried in the binary. Nothing observes that at
//! runtime beyond a line in the log, and a warm shader cache hides even that,
//! so the declarations are read from the module's own source text.

use alloc::string::String;
use alloc::vec::Vec;

/// Names declared as `pub static <NAME>: SlangProgram` in `src`.
pub(super) fn statics(src: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in src.lines() {
        if let Some(rest) = line.trim_start().strip_prefix("pub static ")
            && let Some((name, tail)) = rest.split_once(':')
            && tail.trim_start().starts_with("SlangProgram")
        {
            names.push(String::from(name.trim()));
        }
    }
    names
}

/// Names listed in the `pub static ALL` table in `src`.
pub(super) fn table(src: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut inside = false;
    for line in src.lines() {
        let line = line.trim();
        if line.starts_with("pub static ALL") {
            inside = true;
        } else if inside {
            if line == "];" {
                break;
            }
            if let Some(name) = line.strip_prefix('&').and_then(|l| l.strip_suffix(',')) {
                names.push(String::from(name));
            }
        }
    }
    names
}

/// Assert the declarations and the table name the same set of programs.
pub(super) fn assert_table_is_complete(src: &str) {
    let mut declared = statics(src);
    let mut listed = table(src);
    assert!(!declared.is_empty(), "no declarations found");
    declared.sort();
    listed.sort();
    assert_eq!(
        declared, listed,
        "the declarations and ALL disagree; a program left off the table compiles at renderer init"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
pub static A: SlangProgram = SlangProgram {\n\
};\n\
pub static B: SlangProgram = SlangProgram {\n\
};\n\
pub static ALL: &[&SlangProgram] = &[\n\
    &A,\n\
    &B,\n\
];\n\
pub static AFTER: SlangProgram = SlangProgram {\n";

    #[test]
    fn the_scan_reads_declarations_and_the_table() {
        assert_eq!(statics(SAMPLE), ["A", "B", "AFTER"]);
        assert_eq!(table(SAMPLE), ["A", "B"]);
    }

    #[test]
    #[should_panic(expected = "disagree")]
    fn an_unlisted_declaration_fails() {
        assert_table_is_complete(SAMPLE);
    }

    #[test]
    fn a_complete_table_passes() {
        assert_table_is_complete(
            &SAMPLE.replace("pub static AFTER: SlangProgram = SlangProgram {\n", ""),
        );
    }
}
