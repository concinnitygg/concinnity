// The asset reference, embedded.
//
// `build.rs` reads the rustdoc off every authorable asset's schema struct,
// pairs it with the args metadata in the authoring registry, and bakes the
// result into a static table. The extraction happens once, at build time, from
// the tree being built, so the prose and the registry can never disagree and
// nothing here reads a source file at runtime.
//
// That makes the reference usable wherever an asset type needs explaining: the
// cook pipeline's type discovery, `cn docs` writing the markdown pages, and any
// authoring or agentic tool that wants a type's documentation on demand.

#![no_std]

extern crate alloc;

// The test harness links std; name it so `#[cfg(test)]` modules can use
// std-pathed helpers. The library target pulls in nothing beyond core + alloc.
#[cfg(test)]
extern crate std;

mod page;

pub use page::{AUTOGEN_MARKER, IndexEntry, render_index, render_page};

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

include!(concat!(env!("OUT_DIR"), "/assets_doc.rs"));

/// Look up a type's documentation by NAME (case-insensitive). Finds both
/// authorable assets and the reference types (nested value types, documented
/// enums) they embed.
pub fn describe(type_name: &str) -> Option<&'static AssetDoc> {
    ASSET_DOCS
        .iter()
        .find(|d| d.type_name.eq_ignore_ascii_case(type_name))
}

/// A type's one-line summary, for a listing that has room for a description but
/// not a whole page.
pub fn summary(type_name: &str) -> Option<&'static str> {
    describe(type_name).map(|d| d.summary)
}

/// Every authorable asset, in the order the reference lists them. Excludes the
/// reference types, which document what assets embed rather than what a world
/// can declare.
pub fn assets() -> impl Iterator<Item = &'static AssetDoc> {
    ASSET_DOCS.iter().filter(|d| !d.is_reference_type)
}

/// The whole reference as markdown, keyed by page file name (`Prop.md`,
/// `index.md`). What `cn docs` writes to disk.
pub fn pages() -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for d in ASSET_DOCS {
        out.insert(
            format!("{}.md", d.type_name),
            render_page(d.type_name, d.full_doc),
        );
    }

    let index = |reference_types: bool| -> Vec<IndexEntry> {
        ASSET_DOCS
            .iter()
            .filter(|d| d.is_reference_type == reference_types)
            .map(|d| IndexEntry {
                name: d.type_name.to_string(),
                summary: d.summary.to_string(),
            })
            .collect()
    };
    out.insert(
        "index.md".to_string(),
        render_index(&index(false), &index(true)),
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_finds_assets_case_insensitively() {
        let d = describe("Texture").expect("Texture should be documented");
        assert_eq!(d.type_name, "Texture");
        assert!(!d.summary.is_empty());
        assert!(d.full_doc.contains(d.summary));
        assert!(describe("texture").is_some());
        assert!(describe("TEXTURE").is_some());
        assert!(describe("NotARealAsset").is_none());
    }

    // A nested value type an asset embeds (Prop.collider) is documented in its
    // own right, not just inlined into the asset that embeds it.
    #[test]
    fn describe_finds_reference_types() {
        let d = describe("PropCollider").expect("PropCollider should be documented");
        assert!(d.is_reference_type);
        assert!(!assets().any(|a| a.type_name == "PropCollider"));
    }

    // The extraction resolved real prose for every type. An empty summary means
    // the asset sources went unread, which would otherwise surface as a page set
    // of bare titles.
    #[test]
    fn every_type_resolved_documentation() {
        assert!(ASSET_DOCS.len() > 50, "suspiciously small reference");
        for d in ASSET_DOCS {
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
        let pages = pages();
        assert_eq!(pages.len(), ASSET_DOCS.len() + 1);
        assert!(pages.contains_key("index.md"));
        for d in ASSET_DOCS {
            let page = &pages[&format!("{}.md", d.type_name)];
            assert!(page.starts_with(AUTOGEN_MARKER));
            assert!(page.contains(&format!("# {}", d.type_name)));
        }
    }

    // No `](#anchor)` cross-reference survives into a page's prose: every one is
    // rewritten to a relative `Name.md` link at build time. Code spans and
    // fenced blocks are exempt, since they never render as links and a doc may
    // legitimately show anchor syntax verbatim (StoryImport documents its own
    // Markdown dialect).
    #[test]
    fn no_in_page_anchor_links_remain() {
        for d in ASSET_DOCS {
            assert!(
                !prose_only(d.full_doc).contains("](#"),
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
