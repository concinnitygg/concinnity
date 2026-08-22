// `cn docs`: write the asset reference pages under docs/assets/.
//
// The reference itself is embedded in the binary by concinnity-docs, extracted
// from the asset sources when that crate was built. This command only puts it on
// disk, so it needs no source tree and cannot disagree with the registry.
//
// The pages are committed to the repository; `committed_pages_are_current` fails
// when they drift from the embedded reference.

use concinnity_docs::AUTOGEN_MARKER;
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

// Where the pages land, relative to the directory given on the command line.
const PAGES_DIR: &str = "docs/assets";

// Write the pages under `<root>/docs/assets`, defaulting to the current
// directory. Unchanged pages are left alone, so running this on an up-to-date
// tree touches nothing.
pub(crate) fn docs(root: Option<&str>) -> io::Result<()> {
    let dir = PathBuf::from(root.unwrap_or(".")).join(PAGES_DIR);
    fs::create_dir_all(&dir)?;

    let pages = concinnity_docs::pages();
    let mut written = 0usize;
    for (file, content) in &pages {
        let path = dir.join(file);
        if fs::read_to_string(&path).ok().as_deref() == Some(content.as_str()) {
            continue;
        }
        fs::write(&path, content)?;
        written += 1;
    }
    let removed = remove_stale_pages(&dir, &pages)?;

    println!(
        "{} asset pages in {} ({written} written, {removed} removed)",
        pages.len(),
        dir.display()
    );
    Ok(())
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

    // The repository root, two levels above this crate.
    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    // The committed pages are what the embedded reference renders to. A failure
    // means an asset's rustdoc or args changed without a `cn docs` run.
    #[test]
    fn committed_pages_are_current() {
        let dir = repo_root().join(PAGES_DIR);
        for (file, expected) in &concinnity_docs::pages() {
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
        let pages = concinnity_docs::pages();
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

    // Writing into a fresh directory produces the whole reference; a stale
    // generated page is pruned on the next run and a hand-authored one is not.
    #[test]
    fn writing_is_complete_and_prunes_only_generated_pages() {
        let root = std::env::temp_dir().join("cn-docs-write-test");
        fs::remove_dir_all(&root).ok();
        docs(Some(root.to_str().unwrap())).expect("first run writes the pages");

        let dir = root.join(PAGES_DIR);
        for (file, content) in &concinnity_docs::pages() {
            assert_eq!(
                &fs::read_to_string(dir.join(file)).expect("page written"),
                content
            );
        }

        fs::write(dir.join("Gone.md"), format!("{AUTOGEN_MARKER}\n\n# Gone\n")).unwrap();
        fs::write(dir.join("notes.md"), "hand written\n").unwrap();
        // Non-Markdown neighbours are skipped outright, marker or not.
        fs::write(dir.join("diagram.png"), format!("{AUTOGEN_MARKER}\n")).unwrap();
        docs(Some(root.to_str().unwrap())).expect("second run");
        assert!(!dir.join("Gone.md").exists(), "stale page should be pruned");
        assert!(
            dir.join("notes.md").exists(),
            "hand-authored page should stay"
        );
        assert!(dir.join("diagram.png").exists(), "non-page should stay");

        fs::remove_dir_all(&root).ok();
    }
}
