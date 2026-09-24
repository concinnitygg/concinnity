//! The checks every program table runs.
//!
//! Completeness: `ALL` is what the build script iterates, so a program declared
//! but left off it compiles at renderer init on every host instead of being
//! carried in the binary. Nothing observes that at runtime beyond a line in the
//! log, and a warm shader cache hides even that, so the declarations are read
//! from the module's own source text.
//!
//! Soundness: every row assembles to a compilable source that holds its entry,
//! and no two rows share an artifact key or compile the same program.

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::render::{shader_source, shaders};

/// One program as the soundness check sees it: the key its artifact is filed
/// under, what it compiles, and the source it assembles to.
pub(super) struct Row {
    pub label: String,
    pub file: &'static str,
    pub entry: &'static str,
    pub defines: Vec<(&'static str, &'static str)>,
    pub source: String,
}

/// Assert every row of one table is sound:
///
/// - its file is embedded, so the assembly does not come up empty;
/// - every [`shader_source::FRAGMENTS`] marker is spliced, since an unreplaced
///   one reaches the compiler as a syntax error at renderer init;
/// - its entry point appears in the assembled source, so a rename on one side
///   fails here rather than at a pipeline build;
/// - its label is unique, since the label keys the precompiled artifact and a
///   shared one would hand one variant's bytes to another;
/// - no other row compiles the same (file, entry, defines).
pub(super) fn assert_rows_are_sound(table: &str, rows: &[Row]) {
    assert!(!rows.is_empty(), "{table}: no programs");
    let mut labels = BTreeSet::new();
    let mut programs = BTreeSet::new();
    for row in rows {
        let at = format!("{table} {}", row.label);
        assert!(
            shaders::embedded(row.file).is_some(),
            "{at}: no embedded {}",
            row.file
        );
        for (marker, _) in shader_source::FRAGMENTS {
            assert!(!row.source.contains(marker), "{at}: unspliced {marker}");
        }
        assert!(
            row.source.contains(&format!(" {}(", row.entry)),
            "{at}: entry {} not found in {}",
            row.entry,
            row.file
        );
        assert!(labels.insert(row.label.clone()), "{at}: label shared");
        assert!(
            programs.insert((row.file, row.entry, row.defines.clone())),
            "{at}: another row compiles the same program"
        );
    }
}

/// Names declared as `pub static <NAME>: ShaderProgram` in `src`.
pub(super) fn statics(src: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in src.lines() {
        if let Some(rest) = line.trim_start().strip_prefix("pub static ")
            && let Some((name, tail)) = rest.split_once(':')
            && tail.trim_start().starts_with("ShaderProgram")
        {
            names.push(String::from(name.trim()));
        }
    }
    names
}

/// Names listed in the `pub static ALL` table in `src`.
pub(super) fn table(src: &str) -> Vec<String> {
    let Some(start) = src.find("pub static ALL") else {
        return Vec::new();
    };
    let body = &src[start..];
    let body = &body[body.find("= &[").map_or(0, |i| i + 4)..body.find("];").unwrap_or(body.len())];
    body.split(',')
        .filter_map(|name| name.trim().strip_prefix('&'))
        .map(String::from)
        .collect()
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
pub static A: ShaderProgram = ShaderProgram {\n\
};\n\
pub static B: ShaderProgram = ShaderProgram {\n\
};\n\
pub static ALL: &[&ShaderProgram] = &[\n\
    &A,\n\
    &B,\n\
];\n\
pub static AFTER: ShaderProgram = ShaderProgram {\n";

    #[test]
    fn a_one_line_table_is_read_too() {
        assert_eq!(
            table("pub static ALL: &[&ShaderProgram] = &[&A, &B];\n"),
            ["A", "B"]
        );
    }

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

    fn row(label: &str, entry: &'static str, source: &str) -> Row {
        Row {
            label: String::from(label),
            file: "fog.hlsl",
            entry,
            defines: Vec::new(),
            source: String::from(source),
        }
    }

    #[test]
    fn sound_rows_pass() {
        assert_rows_are_sound(
            "t",
            &[
                row("a", "one", "void one() {}"),
                row("b", "two", "void two() {}"),
            ],
        );
    }

    #[test]
    #[should_panic(expected = "label shared")]
    fn a_shared_label_fails() {
        assert_rows_are_sound(
            "t",
            &[
                row("a", "one", "void one() {}"),
                row("a", "two", "void two() {}"),
            ],
        );
    }

    #[test]
    #[should_panic(expected = "same program")]
    fn a_duplicate_program_fails() {
        assert_rows_are_sound(
            "t",
            &[
                row("a", "one", "void one() {}"),
                row("b", "one", "void one() {}"),
            ],
        );
    }

    #[test]
    #[should_panic(expected = "unspliced")]
    fn an_unspliced_marker_fails() {
        assert_rows_are_sound("t", &[row("a", "one", "{POST_COMMON}\nvoid one() {}")]);
    }

    #[test]
    #[should_panic(expected = "not found")]
    fn a_missing_entry_fails() {
        assert_rows_are_sound("t", &[row("a", "one", "void other() {}")]);
    }

    #[test]
    #[should_panic(expected = "no embedded")]
    fn a_missing_file_fails() {
        let mut r = row("a", "one", "void one() {}");
        r.file = "not_a_shader.hlsl";
        assert_rows_are_sound("t", &[r]);
    }

    #[test]
    fn a_complete_table_passes() {
        assert_table_is_complete(
            &SAMPLE.replace("pub static AFTER: ShaderProgram = ShaderProgram {\n", ""),
        );
    }
}
