//! Which reload subjects a filesystem event touches. The catalogs store paths
//! as the build resolved them (often relative), while the watcher reports them
//! absolute and, on macOS, through the canonical `/private` prefix, so both
//! sides are normalized before they are compared.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

// Every watched file under its normalized path, with the subject reading it. A
// file two subjects share appears once per subject.
#[derive(Debug)]
pub(super) struct FileIndex<K>(Vec<(PathBuf, K)>);

impl<K> Default for FileIndex<K> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<K: Ord + Clone> FileIndex<K> {
    pub(super) fn new<'a>(files: impl IntoIterator<Item = (&'a str, K)>) -> Self {
        Self(
            files
                .into_iter()
                .map(|(path, key)| (normalize(Path::new(path)), key))
                .collect(),
        )
    }

    // The subjects an event on `paths` recompiles: those reading a file still
    // there. A deleted file has nothing to compile, and the rebuild that
    // follows drops its subject from the catalog; a save, atomic or not,
    // leaves the file at its path.
    pub(super) fn to_recompile<'a>(
        &self,
        paths: impl IntoIterator<Item = &'a PathBuf>,
    ) -> BTreeSet<K> {
        self.touched(paths.into_iter().filter(|p| p.exists()))
    }

    // The subjects reading any of `paths`.
    fn touched<'a>(&self, paths: impl IntoIterator<Item = &'a PathBuf>) -> BTreeSet<K> {
        let mut touched = BTreeSet::new();
        for path in paths {
            let path = normalize(path);
            touched.extend(
                self.0
                    .iter()
                    .filter(|(file, _)| *file == path)
                    .map(|(_, key)| key.clone()),
            );
        }
        touched
    }
}

// An absolute path with symlinks resolved as far as the filesystem allows. A
// file an atomic save just removed cannot be canonicalized, so its parent is
// instead; a path whose directory is gone stays as it was, made absolute.
fn normalize(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }
    if let (Some(parent), Some(name)) = (path.parent(), path.file_name())
        && let Ok(parent) = parent.canonicalize()
    {
        return parent.join(name);
    }
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touched(index: &FileIndex<u32>, paths: &[&str]) -> Vec<u32> {
        let paths: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
        index.touched(&paths).into_iter().collect()
    }

    // A save marks just the subjects reading that file, and a file two
    // subjects share marks both.
    #[test]
    fn a_save_marks_only_the_subjects_reading_the_file() {
        let index = FileIndex::new([
            ("/cn-none/lit.hlsl", 1),
            ("/cn-none/sway.hlsl", 1),
            ("/cn-none/reeds.hlsl", 2),
            ("/cn-none/sway.hlsl", 2),
            ("/cn-none/rock.hlsl", 3),
        ]);
        assert_eq!(touched(&index, &["/cn-none/rock.hlsl"]), [3]);
        assert_eq!(touched(&index, &["/cn-none/sway.hlsl"]), [1, 2]);
        assert_eq!(
            touched(&index, &["/cn-none/lit.hlsl", "/cn-none/rock.hlsl"]),
            [1, 3]
        );
        assert!(touched(&index, &["/cn-none/engine.hlsl"]).is_empty());
    }

    // A catalog path relative to the working directory matches the absolute
    // path the watcher reports for the same file.
    #[test]
    fn a_relative_catalog_path_matches_its_absolute_event_path() {
        let index = FileIndex::new([("cn-none/water.hlsl", 4)]);
        let absolute = std::env::current_dir().unwrap().join("cn-none/water.hlsl");
        assert_eq!(
            index.touched(&[absolute]).into_iter().collect::<Vec<_>>(),
            [4]
        );
    }

    // An event on a file that is gone (its subject deleted it) recompiles
    // nothing, while one on a file still there recompiles every subject
    // reading it.
    #[test]
    fn a_removed_file_recompiles_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let (lit, water) = (dir.path().join("lit.hlsl"), dir.path().join("water.hlsl"));
        std::fs::write(&lit, "").unwrap();
        std::fs::write(&water, "").unwrap();
        let index = FileIndex::new([
            (lit.to_str().unwrap(), "lit_a"),
            (lit.to_str().unwrap(), "lit_b"),
            (water.to_str().unwrap(), "water"),
        ]);
        std::fs::remove_file(&water).unwrap();
        assert!(index.to_recompile([&water]).is_empty());
        assert_eq!(
            index
                .to_recompile([&lit, &water])
                .into_iter()
                .collect::<Vec<_>>(),
            ["lit_a", "lit_b"]
        );
    }

    #[test]
    fn an_empty_index_touches_nothing() {
        let index = FileIndex::<u32>::default();
        assert!(touched(&index, &["/cn-none/lit.hlsl"]).is_empty());
    }
}
