// `cn docs`: write the asset reference pages under docs/assets/.
//
// The prose is rustdoc, serde keys, and `Default` literals, none of which
// survive compilation, so the reference is read from the engine's own asset
// sources each time this runs: `schema` parses them, `reference` joins the two
// trees over the authoring registry and renders each body, `page` assembles the
// pages. That makes this a command for a checkout of the engine, which is the
// only place the pages are regenerated.
//
// The pages are committed to the repository. Whether they still match the
// sources is a question about a checkout, not about this code, so it belongs to
// a repository check rather than a unit test: a test that reads the committed
// pages passes or fails on files no test wrote.

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
/// Regenerate the asset reference pages under `docs/assets/`, read out of the
/// engine's own schema sources.
///
/// `root` is the engine checkout to read from; `None` uses the working
/// directory.
pub fn docs(root: Option<&str>) -> io::Result<()> {
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

    // A vocabulary the test wrote, in the shape the extractor reads: rustdoc on
    // the struct and on each field, and a `Default` impl for the defaults the
    // parameter table renders.
    //
    // Reading the engine's own sources instead would tie these to whichever
    // assets it happens to declare, and would assert nothing a source edit
    // could not silently satisfy: the anchor-link check below only means
    // something because this vocabulary contains an anchor link.
    const SOURCES: &str = r#"
        /// A widget in the world.
        ///
        /// The shape it embeds is a [collider](#widgetcollider).
        pub struct Widget {
            /// The mesh to draw.
            pub mesh: String,
            /// The shape it collides with.
            pub collider: Option<WidgetCollider>,
        }
        impl Default for Widget {
            fn default() -> Self {
                Self { mesh: "cube".to_string(), collider: None }
            }
        }

        /// A collider shape a widget embeds.
        pub struct WidgetCollider {
            /// Half the box's size on each axis.
            pub half_extents: [f32; 3],
        }
        impl Default for WidgetCollider {
            fn default() -> Self {
                Self { half_extents: [0.5, 0.5, 0.5] }
            }
        }

        /// A gadget that makes noise.
        pub struct Gadget {
            /// How loud, from silent to full.
            pub volume: f32,
        }
        impl Default for Gadget {
            fn default() -> Self {
                Self { volume: 1.0 }
            }
        }

        /// Engine bookkeeping no world declares.
        pub struct Internal {
            /// A counter.
            pub ticks: u32,
        }
        impl Default for Internal {
            fn default() -> Self {
                Self { ticks: 0 }
            }
        }
    "#;

    // The reference over `SOURCES`, and the tree it was read from. The tree is
    // returned so it outlives the borrow-free `Vec` the caller works with.
    fn synthetic_reference() -> (concinnity_testing::TempTree, Vec<AssetDoc>) {
        let tree = concinnity_testing::TempTree::new();
        tree.write("schema/vocabulary.rs", SOURCES);
        // A non-Rust neighbour the walk must skip.
        tree.write("schema/notes.md", "not rust");

        let components = [
            reference::ComponentMeta::pass_through("Widget", "External"),
            reference::ComponentMeta::pass_through("Gadget", "External"),
            // Never declared in a world, so it must get no page.
            reference::ComponentMeta::pass_through("Internal", "RuntimeOnly"),
        ];
        let docs = reference::build_from(&[tree.join("schema")], &components)
            .expect("the synthetic sources parse");
        (tree, docs)
    }

    // Writing into a fresh directory produces the whole page set; a stale
    // generated page is pruned on the next run and a hand-authored one is not.
    #[test]
    fn writing_is_complete_and_prunes_only_generated_pages() {
        let tree = concinnity_testing::TempTree::new();
        let dir = tree.path();

        let pages: BTreeMap<String, String> = ["Prop", "Texture"]
            .iter()
            .map(|n| (format!("{n}.md"), render_page(n, "A body.")))
            .collect();
        assert_eq!(write_pages(dir, &pages).expect("first run"), (2, 0));
        for (file, content) in &pages {
            assert_eq!(
                &fs::read_to_string(dir.join(file)).expect("written"),
                content
            );
        }

        // A second run over an unchanged tree touches nothing.
        assert_eq!(write_pages(dir, &pages).expect("second run"), (0, 0));

        fs::write(dir.join("Gone.md"), format!("{AUTOGEN_MARKER}\n\n# Gone\n")).unwrap();
        fs::write(dir.join("notes.md"), "hand written\n").unwrap();
        // Non-Markdown neighbours are skipped outright, marker or not.
        fs::write(dir.join("diagram.png"), format!("{AUTOGEN_MARKER}\n")).unwrap();
        assert_eq!(write_pages(dir, &pages).expect("third run"), (0, 1));
        assert!(!dir.join("Gone.md").exists(), "stale page should be pruned");
        assert!(
            dir.join("notes.md").exists(),
            "hand-authored page should stay"
        );
        assert!(dir.join("diagram.png").exists(), "non-page should stay");
    }

    fn describe<'a>(docs: &'a [AssetDoc], type_name: &str) -> Option<&'a AssetDoc> {
        docs.iter()
            .find(|d| d.type_name.eq_ignore_ascii_case(type_name))
    }

    // An asset is found by name whatever its casing, a type a field embeds is
    // documented in its own right rather than only inlined, and a RuntimeOnly
    // component -- one no world declares -- gets no page at all.
    #[test]
    fn every_documented_type_is_found_by_name() {
        let (_tree, docs) = synthetic_reference();

        let d = describe(&docs, "Widget").expect("Widget should be documented");
        assert_eq!(d.type_name, "Widget");
        assert!(d.full_doc.contains(&d.summary));
        assert!(describe(&docs, "widget").is_some());
        assert!(describe(&docs, "WIDGET").is_some());
        assert!(describe(&docs, "NotARealAsset").is_none());

        let embedded = describe(&docs, "WidgetCollider").expect("the embedded type is documented");
        assert!(embedded.is_reference_type);
        assert!(!d.is_reference_type, "an asset is not a reference type");

        assert!(
            describe(&docs, "Internal").is_none(),
            "a RuntimeOnly component is engine-internal and gets no page"
        );
    }

    // Every entry resolved real prose. An empty summary means the sources went
    // unread, which would otherwise surface as a page set of bare titles.
    #[test]
    fn every_type_resolved_documentation() {
        let (_tree, docs) = synthetic_reference();
        assert_eq!(docs.len(), 3, "two assets and the type they embed");

        for d in &docs {
            assert!(!d.summary.is_empty(), "{} has no summary", d.type_name);
            assert!(
                !d.summary.contains('\n'),
                "{}'s summary spans multiple lines: {:?}",
                d.type_name,
                d.summary
            );
        }

        // The summary is the first paragraph, not the whole body.
        let widget = describe(&docs, "Widget").expect("Widget");
        assert_eq!(widget.summary, "A widget in the world.");

        // A field's own prose and its default both reach the parameter table.
        assert!(
            widget.full_doc.contains("The mesh to draw."),
            "{:?}",
            widget.full_doc
        );
        assert!(widget.full_doc.contains("cube"), "{:?}", widget.full_doc);
    }

    #[test]
    fn pages_cover_every_type_plus_the_index() {
        let (_tree, docs) = synthetic_reference();
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
    //
    // `SOURCES` contains such a link, so this fails if the rewriting stops
    // happening -- not only if some asset's prose happens to carry one.
    #[test]
    fn no_in_page_anchor_links_remain() {
        let (_tree, docs) = synthetic_reference();
        let widget = describe(&docs, "Widget").expect("Widget");

        assert!(
            widget.full_doc.contains("](WidgetCollider.md)"),
            "the anchor was rewritten to a relative page link: {:?}",
            widget.full_doc
        );
        for d in &docs {
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
