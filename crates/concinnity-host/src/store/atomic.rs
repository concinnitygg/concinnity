//! Replacing a file whole, for the state tree's regenerable containers.
//!
//! A cache segment is read by one process while another writes it -- an editor
//! showing a world while a build cooks it is routine -- so a writer must never
//! be observed part-way through. The bytes go to a process-unique temp file
//! beside the target and are renamed over it, which the filesystem publishes
//! atomically: a reader opens the old file or the new one.
//!
//! The write is a closure rather than a byte slice, so a producer holding its
//! image in memory and one streaming a payload it never materializes both go
//! through the same publish.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// Replace `path` with whatever `write` emits, creating the directory above it
/// if needed. Reports whether the file was replaced; a failure anywhere leaves
/// the existing file untouched and no temp behind.
pub fn replace(path: &Path, write: impl FnOnce(&mut BufWriter<File>) -> io::Result<()>) -> bool {
    let Some(dir) = path.parent() else {
        return false;
    };
    if fs::create_dir_all(dir).is_err() {
        return false;
    }
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    if emit(&tmp, write).is_err() || fs::rename(&tmp, path).is_err() {
        let _ = fs::remove_file(&tmp);
        return false;
    }
    true
}

// The temp file's whole life: created, written through a buffer, flushed, and
// closed before the rename can publish it.
fn emit(tmp: &Path, write: impl FnOnce(&mut BufWriter<File>) -> io::Result<()>) -> io::Result<()> {
    let mut out = BufWriter::new(File::create(tmp)?);
    write(&mut out)?;
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_write_replaces_the_file_and_creates_its_directory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache").join("1");

        assert!(replace(&path, |out| out.write_all(b"first")));
        assert_eq!(fs::read(&path).unwrap(), b"first");

        assert!(replace(&path, |out| out.write_all(b"second")));
        assert_eq!(fs::read(&path).unwrap(), b"second");

        let leftovers = fs::read_dir(path.parent().unwrap())
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "tmp"))
            .count();
        assert_eq!(leftovers, 0, "temp files must not survive a write");
    }

    // A producer that fails part-way leaves the previous file in place: the
    // reader's guarantee is that it sees one whole version or another.
    #[test]
    fn a_failed_write_keeps_the_previous_file_and_no_temp() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("1");
        assert!(replace(&path, |out| out.write_all(b"kept")));

        assert!(!replace(&path, |out| {
            out.write_all(b"partial")?;
            Err(io::Error::other("producer gave up"))
        }));
        assert_eq!(fs::read(&path).unwrap(), b"kept");
        let leftovers = fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "tmp"))
            .count();
        assert_eq!(leftovers, 0, "a failed write leaves no temp behind");
    }

    // Best-effort: a directory that cannot be created drops the write rather
    // than failing whatever produced the bytes.
    #[test]
    fn a_directory_that_cannot_be_created_drops_the_write() {
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("blocker");
        fs::write(&blocker, b"a file, not a directory").unwrap();

        let path = blocker.join("cache").join("1");
        assert!(!replace(&path, |out| out.write_all(b"bytes")));
        assert!(!path.exists());
    }
}
