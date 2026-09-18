// `cn docs`: write the asset reference pages under docs/assets/.
//
// The prose, keys, types and defaults all come from the schema the derives
// compile into the engine (`concinnity_core::ecs::schema`, behind the `schema`
// feature): `reference` walks it from the authoring registry and renders each
// body, `defaults` serializes what an omitted key reads as, and `page`
// assembles the pages. Nothing is read from the engine's sources, so any build
// of `cn` regenerates the same pages.
//
// The pages are committed to the repository. Whether they still match the
// sources is a question about a checkout, not about this code, so it belongs to
// a repository check rather than a unit test: a test that reads the committed
// pages passes or fails on files no test wrote.

mod defaults;
mod page;
mod prose;
mod reference;
mod render;

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

/// Regenerate the asset reference pages under `docs/assets/` from the
/// authored schema compiled into this build.
///
/// `root` is the directory the pages go under; `None` uses the working
/// directory. Unchanged pages are left alone, so running this on an
/// up-to-date tree touches nothing.
pub fn docs(root: Option<&str>) -> io::Result<()> {
    let pages = pages(&reference::build(&reference::registry_assets()));
    let dir = PathBuf::from(root.unwrap_or(".")).join(PAGES_DIR);
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
    use concinnity_core::ecs::schema::{
        Body, FieldDefault, FieldSchema, FieldType, TypeSchema, ValueSchema,
    };
    use std::collections::BTreeMap as Map;

    // A vocabulary the test wrote, in the shape the derives emit: a doc on the
    // type and on each field, and serde's defaults for the parameter table.
    //
    // The engine's own schema instead would tie these to whichever assets it
    // happens to declare, and would assert nothing a doc edit could not
    // silently satisfy: the anchor-link check below only means something
    // because this vocabulary contains an anchor link.
    static WIDGET: TypeSchema = TypeSchema {
        name: "Widget",
        doc: "A widget in the world.\n\nThe shape it embeds is a [collider](#widgetcollider).",
        body: Body::Fields(&[
            FieldSchema {
                key: "mesh",
                doc: "The mesh\nto draw.",
                ty: || FieldType::Str,
                default: FieldDefault::Value(|| Some(Box::new("cube"))),
            },
            FieldSchema {
                key: "collider",
                doc: "The shape it collides with.",
                ty: || FieldType::Optional(|| FieldType::Nested(&WIDGET_COLLIDER)),
                default: FieldDefault::Null,
            },
            FieldSchema {
                key: "target",
                doc: "What it points at.",
                ty: || FieldType::Reference(&[]),
                default: FieldDefault::Value(|| Some(Box::new(""))),
            },
        ]),
        default: None,
    };

    static WIDGET_COLLIDER: TypeSchema = TypeSchema {
        name: "WidgetCollider",
        doc: "A collider shape a widget embeds.",
        body: Body::Fields(&[FieldSchema {
            key: "half_extents",
            doc: "Half the box's size on each axis.",
            ty: || FieldType::Array {
                elem: || FieldType::Float,
                len: Some(3),
            },
            default: FieldDefault::Container,
        }]),
        default: Some(|| Some(Box::new(Map::from([("half_extents", [0.5_f32; 3])])))),
    };

    static GADGET: TypeSchema = TypeSchema {
        name: "GadgetArgs",
        doc: "A gadget that makes noise.",
        body: Body::Fields(&[
            FieldSchema {
                key: "volume",
                doc: "How loud, from silent to full.",
                ty: || FieldType::Float,
                default: FieldDefault::Value(|| Some(Box::new(1.0_f32))),
            },
            FieldSchema {
                key: "mode",
                doc: "How it plays.",
                ty: || FieldType::Enum(&MODE),
                default: FieldDefault::Required,
            },
            FieldSchema {
                key: "widget",
                doc: "The widget it sits on.",
                ty: || FieldType::Nested(&WIDGET),
                default: FieldDefault::Required,
            },
        ]),
        default: None,
    };

    static MODE: TypeSchema = TypeSchema {
        name: "Mode",
        doc: "",
        body: Body::Values(&[
            ValueSchema {
                name: "once",
                doc: "Plays once",
            },
            ValueSchema {
                name: "loop",
                doc: "",
            },
        ]),
        default: None,
    };

    fn synthetic_reference() -> Vec<AssetDoc> {
        reference::build(&[
            reference::AssetEntry {
                name: "Widget",
                schema: &WIDGET,
            },
            reference::AssetEntry {
                name: "Gadget",
                schema: &GADGET,
            },
        ])
    }

    fn describe<'a>(docs: &'a [AssetDoc], type_name: &str) -> &'a AssetDoc {
        docs.iter()
            .find(|d| d.type_name == type_name)
            .unwrap_or_else(|| panic!("{type_name} is documented"))
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
        // Non-Markdown neighbors are skipped outright, marker or not.
        fs::write(dir.join("diagram.png"), format!("{AUTOGEN_MARKER}\n")).unwrap();
        assert_eq!(write_pages(dir, &pages).expect("third run"), (0, 1));
        assert!(!dir.join("Gone.md").exists(), "stale page should be pruned");
        assert!(
            dir.join("notes.md").exists(),
            "hand-authored page should stay"
        );
        assert!(dir.join("diagram.png").exists(), "non-page should stay");
    }

    // Assets are named by registry name, not by their args schema's; a nested
    // struct and a documented enum a field reaches get pages of their own; and
    // a field embedding another asset links to that asset rather than to a
    // second page for its schema.
    #[test]
    fn assets_and_the_types_their_fields_reach_are_documented() {
        let docs = synthetic_reference();
        let names: Vec<(&str, bool)> = docs
            .iter()
            .map(|d| (d.type_name.as_str(), d.is_reference_type))
            .collect();
        assert_eq!(
            names,
            [
                ("Gadget", false),
                ("Widget", false),
                ("Mode", true),
                ("WidgetCollider", true),
            ]
        );
        let gadget = describe(&docs, "Gadget");
        assert!(
            gadget
                .full_doc
                .contains("- `widget`: A [Widget](Widget.md) object. The widget it sits on."),
            "{}",
            gadget.full_doc
        );
        assert!(
            gadget
                .full_doc
                .contains("- `mode`: A string (see [Mode](Mode.md)). How it plays."),
            "{}",
            gadget.full_doc
        );
        let mode = describe(&docs, "Mode");
        assert_eq!(mode.summary, "A set of named string values.");
        assert!(
            mode.full_doc.contains("- `once`: Plays once.\n- `loop`"),
            "{}",
            mode.full_doc
        );
    }

    // A field's prose, type and default reach its bullet: a literal default, a
    // default read off the containing type's `Default`, and an optional with
    // none. A reference defaulting to no name states no default.
    #[test]
    fn each_field_states_its_type_doc_and_default() {
        let docs = synthetic_reference();
        let widget = describe(&docs, "Widget");
        assert_eq!(widget.summary, "A widget in the world.");
        for line in [
            "- `mesh`: A string. The mesh to draw. Defaults to `\"cube\"`.",
            "- `collider`: A [WidgetCollider](WidgetCollider.md) object. The shape it collides with. Optional.",
        ] {
            assert!(
                widget.full_doc.contains(line),
                "{line}\n{}",
                widget.full_doc
            );
        }
        let collider = describe(&docs, "WidgetCollider");
        assert!(
            collider.full_doc.contains(
                "- `half_extents`: An array of 3 floats. Half the box's size on each axis. Defaults to `[0.5, 0.5, 0.5]`."
            ),
            "{}",
            collider.full_doc
        );
        assert!(
            describe(&docs, "Gadget")
                .full_doc
                .contains("Defaults to `1.0`.")
        );
    }

    #[test]
    fn pages_cover_every_type_plus_the_index() {
        let docs = synthetic_reference();
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
    // The synthetic vocabulary contains such a link, so this fails if the
    // rewriting stops happening -- not only if some asset's prose happens to
    // carry one.
    #[test]
    fn no_in_page_anchor_links_remain() {
        let docs = synthetic_reference();
        let widget = describe(&docs, "Widget");

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

    // The registry documents every type a world may declare, the build-only
    // ones included, and none of the components only a running world mints.
    #[test]
    fn the_registry_lists_every_authorable_asset() {
        let assets = reference::registry_assets();
        let has = |name: &str| assets.iter().any(|a| a.name == name);
        assert!(has("Prop") && has("Prefab") && has("Texture"));
        assert!(!has("Transform"));
        let room = assets.iter().find(|a| a.name == "Room").expect("Room");
        assert_eq!(room.schema.name, "RoomArgs");
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
