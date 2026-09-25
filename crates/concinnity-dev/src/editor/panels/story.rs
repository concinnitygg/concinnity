//! The data half of the Story panel: the starter story a "+ Create story"
//! writes, and the gutter marker a parse error pins to its line. The text
//! itself is edited in a code text area (`text_area`).

use crate::editor::text_area::markers::{GutterMarker, Severity};

// The marker for a story parse error that names its line (`line N: ...`, N
// counted from 1 over the whole source). Frontmatter errors count their lines
// within the frontmatter, so they stay on the status line alone.
pub(crate) fn error_marker(error: &str) -> Option<GutterMarker> {
    let rest = error.strip_prefix("line ")?;
    let (number, message) = rest.split_once(": ")?;
    let line: usize = number.parse().ok()?;
    Some(GutterMarker {
        line: line.checked_sub(1)?,
        severity: Severity::Error,
        message: message.to_string(),
    })
}

// The starter story a "+ Create story" writes: minimal but exercising the
// format's main constructs (frontmatter with a declared character, a node
// heading, an attributed line, plain narration). Pinned parseable by test.
pub(crate) const STARTER_STORY: &str = "---
title: Untitled Story
characters:
  narrator: Narrator
---

# start

**narrator:** Once upon a time...

The story begins here.
";

#[cfg(test)]
mod tests {
    use super::*;

    // A real parse error lands on the line it names.
    #[test]
    fn a_parse_error_marks_its_line() {
        let broken = STARTER_STORY.replace("The story begins here.", "[jump](#nowhere) and prose");
        let error = concinnity_cook::build_only::validate_story_source(&broken).unwrap_err();
        let marker = error_marker(&error).expect("the error names a line");
        let line = broken.lines().nth(marker.line).unwrap();
        assert!(line.contains("#nowhere"), "{error} -> {line:?}");
        assert_eq!(marker.severity, Severity::Error);
        assert!(!marker.message.starts_with("line"));
    }

    #[test]
    fn errors_without_a_source_line_mark_nothing() {
        assert_eq!(
            error_marker("frontmatter line 2: expected `key: value`"),
            None
        );
        assert_eq!(error_marker("line zero: nonsense"), None);
        assert_eq!(error_marker("line 0: out of range"), None);
    }

    // The starter story must parse with the real story pipeline; a format
    // change breaks this test instead of shipping a broken template.
    #[test]
    fn starter_story_parses() {
        concinnity_cook::build_only::validate_story_source(STARTER_STORY)
            .expect("the starter story template parses");
    }
}
