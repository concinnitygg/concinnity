// Markdown page assembly. `reference` renders the bodies; this turns one into a
// whole page, and the entry list into the index.

/// Leading marker on every generated page. A docs viewer strips it before
/// rendering; it warns a human (or an AI) editing the file by hand.
pub(crate) const AUTOGEN_MARKER: &str = "<!-- Auto-generated - do not edit. -->";

// An entry in the index table of contents: a type's name (and route) plus its
// one-line summary.
pub(super) struct IndexEntry {
    pub name: String,
    pub summary: String,
}

// The link target for a documented type: a sibling `.md` file. Relative and
// self-contained, so the pages cross-link correctly browsed as plain markdown;
// a docs viewer is free to rewrite the suffix to its own routes.
fn doc_link(name: &str) -> String {
    format!("{name}.md")
}

// Assemble a full page: the auto-generated marker, the `# Name` heading, then
// the body (description plus the generated Parameters/Values section).
pub(super) fn render_page(name: &str, body: &str) -> String {
    let mut out = String::new();
    out.push_str(AUTOGEN_MARKER);
    out.push_str("\n\n# ");
    out.push_str(name);
    out.push_str("\n\n");
    out.push_str(body.trim());
    while out.ends_with('\n') {
        out.pop();
    }
    out.push('\n');
    out
}

// Render the index page: an alphabetical list of every asset, then a list of
// the referenced value types and enums, each linking to its own page.
pub(super) fn render_index(assets: &[IndexEntry], ref_types: &[IndexEntry]) -> String {
    let mut out = String::new();
    out.push_str(AUTOGEN_MARKER);
    out.push_str("\n\n# Assets\n\n");
    for a in assets {
        out.push_str(&index_line(a));
    }
    if !ref_types.is_empty() {
        out.push_str("\n## Reference types\n\n");
        for t in ref_types {
            out.push_str(&index_line(t));
        }
    }
    while out.ends_with('\n') {
        out.pop();
    }
    out.push('\n');
    out
}

fn index_line(e: &IndexEntry) -> String {
    let summary = e.summary.trim();
    if summary.is_empty() {
        format!("- [{}]({})\n", e.name, doc_link(&e.name))
    } else {
        format!("- [{}]({}) - {}\n", e.name, doc_link(&e.name), summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_page_has_marker_then_h1_then_body() {
        let page = render_page("Prop", "A prop.\n\n## Parameters\n\n- `x`: A float.");
        let mut lines = page.lines();
        assert_eq!(lines.next(), Some(AUTOGEN_MARKER));
        assert_eq!(lines.next(), Some(""));
        assert_eq!(lines.next(), Some("# Prop"));
        assert!(page.contains("A prop."));
        assert!(page.contains("## Parameters"));
        assert!(page.ends_with('\n'));
        assert!(!page.ends_with("\n\n"));
    }

    fn idx(name: &str, summary: &str) -> IndexEntry {
        IndexEntry {
            name: name.to_string(),
            summary: summary.to_string(),
        }
    }

    #[test]
    fn render_index_lists_assets_then_reference_types() {
        let assets = vec![idx("Camera3D", "A camera."), idx("Prop", "A prop.")];
        let refs = vec![idx("PropCollider", "A collision volume.")];
        let md = render_index(&assets, &refs);
        assert!(md.starts_with(AUTOGEN_MARKER));
        assert!(md.contains("# Assets"));
        assert!(md.contains("- [Camera3D](Camera3D.md) - A camera."));
        assert!(md.contains("- [Prop](Prop.md) - A prop."));
        assert!(md.contains("## Reference types"));
        assert!(md.contains("- [PropCollider](PropCollider.md) - A collision volume."));
        let assets_pos = md.find("# Assets").unwrap();
        let refs_pos = md.find("## Reference types").unwrap();
        assert!(assets_pos < refs_pos);
    }

    #[test]
    fn render_index_omits_reference_section_when_empty() {
        let md = render_index(&[idx("Prop", "A prop.")], &[]);
        assert!(!md.contains("Reference types"));
        assert!(md.ends_with('\n'));
    }
}
