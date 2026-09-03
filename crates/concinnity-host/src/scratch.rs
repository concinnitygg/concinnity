//! Scratch paths in the machine's temporary directory.
//!
//! An external compiler that will not take a pipe needs a file to read or
//! write, and that file belongs to one call: nothing reads it afterwards, and
//! two calls must never share it. Concurrent runs of one tool -- a parallel
//! build, a test suite spread over processes, two exports of one project --
//! collide on any name picked by hand, one resetting a path the other is
//! part-way through writing.
//!
//! So a name here carries the process id and a counter, and is unique for as
//! long as it names anything. [`Scratch`] removes it on drop, which is the half
//! a hand-rolled path keeps getting wrong: the tool that fails is the one that
//! returns early, and its intermediates are what stay behind.
//!
//! This is the only place in the workspace that names the temporary directory.
//! `tests/file_access_discipline.rs` holds the rest of the workspace to that.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// A path in the machine's temporary directory, unique to this call.
///
/// Nothing is created: the caller writes the file, or hands the path to the
/// tool that writes it. The unique part leads, so `name` keeps its extension
/// for a tool that reads one.
///
/// Prefer [`Scratch`], which removes the path again. Reach for this only where
/// something else already owns that.
///
/// ```
/// let path = concinnity_host::scratch::path("shader.air");
/// assert!(path.to_string_lossy().ends_with("shader.air"));
/// assert_ne!(path, concinnity_host::scratch::path("shader.air"));
/// ```
pub fn path(name: &str) -> PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "cn-{}-{}-{name}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ))
}

/// A scratch path that goes away when this value does.
///
/// Dropping it removes the path, so a tool that fails part-way leaves nothing
/// behind and no caller has to remember an error path.
///
/// ```
/// let scratch = concinnity_host::scratch::Scratch::file("notes.txt");
/// std::fs::write(scratch.path(), b"working").unwrap();
/// let path = scratch.path().to_path_buf();
/// drop(scratch);
/// assert!(!path.exists());
/// ```
pub struct Scratch {
    path: PathBuf,
    directory: bool,
}

impl Scratch {
    /// A file path, unique to this call. Nothing is written yet.
    pub fn file(name: &str) -> Self {
        Self {
            path: path(name),
            directory: false,
        }
    }

    /// A directory, unique to this call and created empty.
    ///
    /// # Errors
    ///
    /// If the directory cannot be created.
    pub fn dir(name: &str) -> std::io::Result<Self> {
        // The parent is the temporary directory, which is already there, so
        // this is one call rather than a walk up the path.
        let path = path(name);
        std::fs::create_dir(&path)?;
        Ok(Self {
            path,
            directory: true,
        })
    }

    /// The path itself.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = if self.directory {
            std::fs::remove_dir_all(&self.path)
        } else {
            std::fs::remove_file(&self.path)
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_is_never_handed_out_twice() {
        let first = path("thing");
        let second = path("thing");

        assert_ne!(first, second, "two calls must not share a path");
        assert!(
            first.parent().is_some_and(|p| second.starts_with(p)),
            "both live in the temporary directory"
        );
    }

    // A tool that reads an extension has to find it, so the unique part leads.
    #[test]
    fn a_name_keeps_its_extension() {
        let path = path("My-Game.iconset");

        assert_eq!(path.extension().and_then(|e| e.to_str()), Some("iconset"));
    }

    // A sibling process sweeping its own leftovers must not match ours.
    #[test]
    fn a_name_carries_the_process_that_made_it() {
        let path = path("thing");

        assert!(
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(&format!("cn-{}-", std::process::id()))),
            "got {}",
            path.display()
        );
    }

    #[test]
    fn a_file_goes_when_its_scratch_does() {
        let scratch = Scratch::file("leftover");
        std::fs::write(scratch.path(), b"work").expect("write");
        let path = scratch.path().to_path_buf();

        assert!(path.is_file());
        drop(scratch);
        assert!(!path.exists(), "the file went with the guard");
    }

    // The whole tree goes, not just the directory: what a tool leaves inside is
    // exactly what nothing else would clean up.
    #[test]
    fn a_directory_goes_with_everything_in_it() {
        let scratch = Scratch::dir("work").expect("create");
        std::fs::write(scratch.path().join("inner"), b"work").expect("write");
        let path = scratch.path().to_path_buf();

        assert!(path.is_dir());
        drop(scratch);
        assert!(!path.exists(), "the tree went with the guard");
    }

    // Dropping is not conditional on the tool having succeeded, which is the
    // whole reason the path is owned rather than remembered.
    #[test]
    fn a_path_that_was_never_written_drops_quietly() {
        let scratch = Scratch::file("never-written");
        let path = scratch.path().to_path_buf();

        drop(scratch);
        assert!(!path.exists());
    }
}
