//! A temporary directory that cleans itself up, and the writes into it.

use std::path::{Path, PathBuf};

/// A private directory for one test's files, deleted when the value drops.
///
/// Every path a test writes should come from one of these. A test that builds
/// its own path under the system temp directory leaves that tree behind on
/// every run, and two runs of the same test share it.
///
/// ```
/// # use concinnity_testing::TempTree;
/// let tree = TempTree::new();
/// let path = tree.write("greeting.txt", b"hello");
/// assert_eq!(std::fs::read(&path).unwrap(), b"hello");
/// ```
pub struct TempTree(tempfile::TempDir);

impl TempTree {
    /// Create the directory.
    ///
    /// # Panics
    ///
    /// If the system temporary directory cannot be written.
    #[expect(
        clippy::new_without_default,
        reason = "creation touches the filesystem and panics, which Default should not"
    )]
    pub fn new() -> Self {
        Self(tempfile::tempdir().expect("a temporary directory is created"))
    }

    /// The root of the tree.
    pub fn path(&self) -> &Path {
        self.0.path()
    }

    /// `name` resolved against the root, without creating anything.
    ///
    /// `name` may name a nested path; nothing is created for it here.
    pub fn join(&self, name: &str) -> PathBuf {
        self.path().join(name)
    }

    /// Create a directory under the root, with its parents, and return it.
    ///
    /// # Panics
    ///
    /// If the directory cannot be created.
    pub fn dir(&self, name: &str) -> PathBuf {
        let path = self.join(name);
        std::fs::create_dir_all(&path).expect("a directory under the tree is created");
        path
    }

    /// Write `bytes` to `name` under the root and return the path.
    ///
    /// Parent directories are created, so `name` may be nested.
    ///
    /// # Panics
    ///
    /// If the file cannot be written.
    pub fn write(&self, name: &str, bytes: impl AsRef<[u8]>) -> PathBuf {
        write_into(self.path(), name, bytes)
    }

    /// [`write`](Self::write), with the path as a `String` for the APIs that
    /// take one.
    ///
    /// # Panics
    ///
    /// If the file cannot be written, or its path is not UTF-8.
    pub fn write_path(&self, name: &str, bytes: impl AsRef<[u8]>) -> String {
        utf8(&self.write(name, bytes))
    }

    /// The root as a `String`, for the APIs that take one.
    ///
    /// # Panics
    ///
    /// If the path is not UTF-8.
    pub fn root_path(&self) -> String {
        utf8(self.path())
    }
}

/// Write `bytes` to `name` under `dir` and return the path.
///
/// What [`TempTree::write`] does, for a test that already holds a directory
/// from somewhere else. Parent directories are created, so `name` may be
/// nested.
///
/// # Panics
///
/// If the file cannot be written.
pub fn write_into(dir: &Path, name: &str, bytes: impl AsRef<[u8]>) -> PathBuf {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("the file's parent directory is created");
    }
    std::fs::write(&path, bytes).expect("a file under the directory is written");
    path
}

/// `path` as a `String`.
///
/// # Panics
///
/// If the path is not UTF-8. A temporary directory built by this crate always
/// is, so a failure here is the host's temporary path, not the test's.
pub fn utf8(path: &Path) -> String {
    path.to_str().expect("the path is UTF-8").to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_written_file_reads_back_and_the_tree_reports_its_path() {
        let tree = TempTree::new();
        let path = tree.write("a.bin", [1u8, 2, 3]);

        assert_eq!(std::fs::read(&path).expect("read back"), [1, 2, 3]);
        assert!(path.starts_with(tree.path()));
        assert_eq!(tree.join("a.bin"), path);
    }

    #[test]
    fn a_nested_write_creates_its_parents() {
        let tree = TempTree::new();
        let path = tree.write("deep/er/still.txt", "content");

        assert_eq!(
            std::fs::read_to_string(&path).expect("read back"),
            "content"
        );
    }

    #[test]
    fn dir_creates_a_directory_and_is_idempotent() {
        let tree = TempTree::new();
        let first = tree.dir("nested/inner");
        let second = tree.dir("nested/inner");

        assert_eq!(first, second);
        assert!(first.is_dir());
    }

    #[test]
    fn the_string_forms_name_the_same_paths() {
        let tree = TempTree::new();
        let path = tree.write_path("a.txt", "x");

        assert_eq!(path, utf8(&tree.join("a.txt")));
        assert!(path.starts_with(&tree.root_path()));
    }

    #[test]
    fn write_into_targets_any_directory() {
        let tree = TempTree::new();
        let path = write_into(tree.path(), "nested/a.txt", "x");

        assert_eq!(std::fs::read_to_string(&path).expect("read back"), "x");
        assert_eq!(path, tree.join("nested/a.txt"));
    }

    #[test]
    fn the_directory_is_gone_once_the_tree_drops() {
        let path = {
            let tree = TempTree::new();
            tree.write("a.txt", "x");
            tree.path().to_path_buf()
        };

        assert!(!path.exists(), "the tree deletes itself");
    }
}
