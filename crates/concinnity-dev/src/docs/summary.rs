// One-line summaries of the authorable types, for the editor's type pickers.
//
// The text is the same first paragraph the reference index shows, in plain
// form: a picker row is a label, not markdown, so link syntax and code spans
// are unwrapped to the words they carry.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use super::prose::{first_paragraph, strip_rust_blocks};
use super::reference::registry_assets;

/// One-line summary per authorable type, keyed by its registry name. Types
/// whose rustdoc opens with something other than prose are absent rather than
/// present and empty.
pub(crate) fn type_summaries() -> &'static BTreeMap<&'static str, String> {
    static SUMMARIES: OnceLock<BTreeMap<&'static str, String>> = OnceLock::new();
    SUMMARIES.get_or_init(|| {
        registry_assets()
            .iter()
            .filter_map(|a| {
                let summary = plain_text(&first_paragraph(&strip_rust_blocks(a.schema.doc)));
                (!summary.is_empty()).then_some((a.name, summary))
            })
            .collect()
    })
}

// Markdown a rustdoc summary can carry, reduced to the words a label shows:
// `[text](link)` and `[text]` become `text`, and code spans lose their
// backticks.
fn plain_text(doc: &str) -> String {
    let mut out = String::with_capacity(doc.len());
    let mut rest = doc;
    while let Some(open) = rest.find('[') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let Some(close) = after.find(']') else {
            out.push_str(after);
            return out.replace('`', "");
        };
        out.push_str(&after[..close]);
        rest = &after[close + 1..];
        // A reference link carries its target in the parentheses that follow.
        if rest.starts_with('(')
            && let Some(end) = rest.find(')')
        {
            rest = &rest[end + 1..];
        }
    }
    out.push_str(rest);
    out.replace('`', "")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_unwraps_links_and_code_spans() {
        assert_eq!(plain_text("A light."), "A light.");
        assert_eq!(
            plain_text("Instances a [Prefab](Prefab.md) at a point."),
            "Instances a Prefab at a point."
        );
        assert_eq!(plain_text("Sits beside [Prop]."), "Sits beside Prop.");
        assert_eq!(
            plain_text("Reads `radius` in meters."),
            "Reads radius in meters."
        );
    }

    #[test]
    fn plain_text_keeps_unbalanced_brackets_readable() {
        assert_eq!(plain_text("An array [0, 1"), "An array 0, 1");
    }

    #[test]
    fn every_summary_is_one_plain_line() {
        let summaries = type_summaries();
        assert!(
            summaries.contains_key("PointLight"),
            "the registry's documented types should carry summaries"
        );
        for (name, summary) in summaries {
            assert!(
                !summary.contains('\n'),
                "{name} summary spans lines: {summary}"
            );
            assert!(!summary.contains('`'), "{name} summary keeps a code span");
            assert!(
                !summary.contains("]("),
                "{name} summary keeps a markdown link"
            );
        }
    }
}
