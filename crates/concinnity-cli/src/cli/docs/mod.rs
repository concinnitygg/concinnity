// `cn docs`: write the asset reference pages under docs/assets/.
//
// The prose is rustdoc, serde keys, and `Default` literals, none of which
// survive compilation, so the reference is read from the engine's own asset
// sources each time this runs: `schema` parses them, `reference` joins the two
// trees over the authoring registry and renders each body, `page` assembles the
// pages. That makes this a command for a checkout of the engine, which is the
// only place the pages are regenerated.
//
// The pages are committed to the repository; `committed_pages_are_current` fails
// when they drift from the sources.

mod page;
mod reference;
mod render;
mod schema;

use page::{AUTOGEN_MARKER, IndexEntry, render_index, render_page};
use reference::AssetDoc;

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

// Where the pages land, relative to the directory given on the command line.
const PAGES_DIR: &str = "docs/assets";

// The whole reference as markdown, keyed by page file name (`Prop.md`,
// `index.md`).
fn pages(docs: &[AssetDoc]) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for d in docs {
        out.insert(
            format!("{}.md", d.type_name),
            render_page(&d.type_name, &d.full_doc),
        );
    }

    let index = |reference_types: bool| -> Vec<IndexEntry> {
        docs.iter()
            .filter(|d| d.is_reference_type == reference_types)
            .map(|d| IndexEntry {
                name: d.type_name.clone(),
                summary: d.summary.clone(),
            })
            .collect()
    };
    out.insert(
        "index.md".to_string(),
        render_index(&index(false), &index(true)),
    );
    out
}

// Write the pages under `<root>/docs/assets`, defaulting to the current
// directory. `<root>` is also the engine checkout the prose is read from.
// Unchanged pages are left alone, so running this on an up-to-date tree touches
// nothing.
pub(crate) fn docs(root: Option<&str>) -> io::Result<()> {
    let engine_root = PathBuf::from(root.unwrap_or("."));
    let pages = pages(&reference::build(&engine_root)?);
    let dir = engine_root.join(PAGES_DIR);
    let (written, removed) = write_pages(&dir, &pages)?;

    println!(
        "{} asset pages in {} ({written} written, {removed} removed)",
        pages.len(),
        dir.display()
    );
    Ok(())
}

// Put `pages` on disk in `dir`, pruning the generated pages no longer among
// them. Returns how many were written and how many pruned. Unchanged pages are
// left alone, so running this on an up-to-date tree touches nothing.
fn write_pages(dir: &Path, pages: &BTreeMap<String, String>) -> io::Result<(usize, usize)> {
    fs::create_dir_all(dir)?;

    let mut written = 0usize;
    for (file, content) in pages {
        let path = dir.join(file);
        if fs::read_to_string(&path).ok().as_deref() == Some(content.as_str()) {
            continue;
        }
        fs::write(&path, content)?;
        written += 1;
    }
    Ok((written, remove_stale_pages(dir, pages)?))
}

// Drop generated pages no longer in the reference (a renamed or deleted asset).
// Only files carrying the auto-generated marker are removed, so a hand-authored
// page dropped in the directory survives.
fn remove_stale_pages(dir: &Path, keep: &BTreeMap<String, String>) -> io::Result<usize> {
    let mut removed = 0;
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if keep.contains_key(name) {
            continue;
        }
        if fs::read_to_string(&path).is_ok_and(|s| s.starts_with(AUTOGEN_MARKER)) {
            fs::remove_file(&path)?;
            removed += 1;
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The repository root, two levels above this crate: the engine checkout the
    // reference is read out of.
    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn reference_pages() -> BTreeMap<String, String> {
        pages(&reference::build(&repo_root()).expect("read the asset sources"))
    }

    // The committed pages are what the asset sources render to. A failure means
    // an asset's rustdoc or args changed without a `cn docs` run.
    #[test]
    fn committed_pages_are_current() {
        let dir = repo_root().join(PAGES_DIR);
        for (file, expected) in &reference_pages() {
            let path = dir.join(file);
            let on_disk = fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {}: {e}; run `cn docs`", path.display()));
            assert_eq!(
                &on_disk, expected,
                "{PAGES_DIR}/{file} is out of date; run `cn docs`"
            );
        }
    }

    // No generated page lingers for a type the reference no longer covers.
    #[test]
    fn no_generated_page_is_orphaned() {
        let pages = reference_pages();
        for entry in fs::read_dir(repo_root().join(PAGES_DIR)).expect("read the pages directory") {
            let path = entry.expect("directory entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap()
                .to_string();
            let generated = fs::read_to_string(&path).is_ok_and(|s| s.starts_with(AUTOGEN_MARKER));
            assert!(
                !generated || pages.contains_key(&name),
                "{PAGES_DIR}/{name} is generated but no longer in the reference; run `cn docs`"
            );
        }
    }

    // Writing into a fresh directory produces the whole page set; a stale
    // generated page is pruned on the next run and a hand-authored one is not.
    #[test]
    fn writing_is_complete_and_prunes_only_generated_pages() {
        let dir = std::env::temp_dir().join("cn-docs-write-test");
        fs::remove_dir_all(&dir).ok();

        let pages: BTreeMap<String, String> = ["Prop", "Texture"]
            .iter()
            .map(|n| (format!("{n}.md"), render_page(n, "A body.")))
            .collect();
        assert_eq!(write_pages(&dir, &pages).expect("first run"), (2, 0));
        for (file, content) in &pages {
            assert_eq!(
                &fs::read_to_string(dir.join(file)).expect("written"),
                content
            );
        }

        // A second run over an unchanged tree touches nothing.
        assert_eq!(write_pages(&dir, &pages).expect("second run"), (0, 0));

        fs::write(dir.join("Gone.md"), format!("{AUTOGEN_MARKER}\n\n# Gone\n")).unwrap();
        fs::write(dir.join("notes.md"), "hand written\n").unwrap();
        // Non-Markdown neighbours are skipped outright, marker or not.
        fs::write(dir.join("diagram.png"), format!("{AUTOGEN_MARKER}\n")).unwrap();
        assert_eq!(write_pages(&dir, &pages).expect("third run"), (0, 1));
        assert!(!dir.join("Gone.md").exists(), "stale page should be pruned");
        assert!(
            dir.join("notes.md").exists(),
            "hand-authored page should stay"
        );
        assert!(dir.join("diagram.png").exists(), "non-page should stay");

        fs::remove_dir_all(&dir).ok();
    }

    fn describe<'a>(docs: &'a [AssetDoc], type_name: &str) -> Option<&'a AssetDoc> {
        docs.iter()
            .find(|d| d.type_name.eq_ignore_ascii_case(type_name))
    }

    #[test]
    fn every_documented_type_is_found_by_name() {
        let docs = reference::build(&repo_root()).expect("read the asset sources");
        let d = describe(&docs, "Texture").expect("Texture should be documented");
        assert_eq!(d.type_name, "Texture");
        assert!(d.full_doc.contains(&d.summary));
        assert!(describe(&docs, "texture").is_some());
        assert!(describe(&docs, "TEXTURE").is_some());
        assert!(describe(&docs, "NotARealAsset").is_none());

        // A nested value type an asset embeds (Prop.collider) is documented in
        // its own right, not just inlined into the asset that embeds it.
        let d = describe(&docs, "PropCollider").expect("PropCollider should be documented");
        assert!(d.is_reference_type);
    }

    // The extraction resolved real prose for every type. An empty summary means
    // the asset sources went unread, which would otherwise surface as a page set
    // of bare titles.
    #[test]
    fn every_type_resolved_documentation() {
        let docs = reference::build(&repo_root()).expect("read the asset sources");
        assert!(docs.len() > 50, "suspiciously small reference");
        for d in &docs {
            assert!(!d.summary.is_empty(), "{} has no summary", d.type_name);
            assert!(
                !d.summary.contains('\n'),
                "{}'s summary spans multiple lines: {:?}",
                d.type_name,
                d.summary
            );
        }
    }

    #[test]
    fn pages_cover_every_type_plus_the_index() {
        let docs = reference::build(&repo_root()).expect("read the asset sources");
        let pages = pages(&docs);
        assert_eq!(pages.len(), docs.len() + 1);
        assert!(pages.contains_key("index.md"));
        for d in &docs {
            let page = &pages[&format!("{}.md", d.type_name)];
            assert!(page.starts_with(AUTOGEN_MARKER));
            assert!(page.contains(&format!("# {}", d.type_name)));
        }
    }

    // No `](#anchor)` cross-reference survives into a page's prose: every one is
    // rewritten to a relative `Name.md` link. Code spans and fenced blocks are
    // exempt, since they never render as links and a doc may legitimately show
    // anchor syntax verbatim (StoryImport documents its own Markdown dialect).
    #[test]
    fn no_in_page_anchor_links_remain() {
        for d in &reference::build(&repo_root()).expect("read the asset sources") {
            assert!(
                !prose_only(&d.full_doc).contains("](#"),
                "{} still has an in-page anchor link outside code: {:?}",
                d.type_name,
                d.full_doc
            );
        }
    }

    // Strip fenced code blocks and inline code spans, leaving the prose that
    // renders as markdown.
    fn prose_only(doc: &str) -> String {
        let mut out = String::new();
        let mut in_fence = false;
        for line in doc.lines() {
            if line.trim_start().starts_with("```") {
                in_fence = !in_fence;
                continue;
            }
            if in_fence {
                continue;
            }
            // Drop the content of `...` spans; an unpaired backtick keeps the
            // rest of the line, which errs toward checking more, not less.
            let mut parts = line.split('`');
            out.push_str(parts.next().unwrap_or(""));
            while let (Some(_code), Some(prose)) = (parts.next(), parts.next()) {
                out.push_str(prose);
            }
            out.push('\n');
        }
        out
    }
}
