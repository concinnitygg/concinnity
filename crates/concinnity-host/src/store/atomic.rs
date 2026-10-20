//! Replacing a file whole, for the state tree's regenerable containers.
//!
//! A cache segment is read by one process while another writes it -- an editor
//! showing a world while a build cooks it is routine -- so a writer must never
//! be observed part-way through. The bytes go to a temp file unique to the write
//! beside the target and are renamed over it, which the filesystem publishes
//! atomically: a reader opens the old file or the new one.
//!
//! The write is a closure rather than a byte slice, so a producer holding its
//! image in memory and one streaming a payload it never materializes both go
//! through the same publish.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

// Distinguishes the temps of concurrent writes within this process.
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

/// Replace `path` with whatever `write` emits, creating the directory above it
/// if needed. A failure anywhere leaves the existing file untouched and no temp
/// behind.
#[expect(
    clippy::let_underscore_must_use,
    reason = "the publish error is what the caller needs; removing the temp is best effort"
)]
pub fn replace(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> io::Result<()>,
) -> io::Result<()> {
    let dir = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no parent"))?;
    fs::create_dir_all(dir)?;
    let tmp = temp_path(path);
    let published = emit(&tmp, write).and_then(|()| fs::rename(&tmp, path));
    if published.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    published
}

// A temp beside `path` no other write uses: `<name>.<pid>.<n>.tmp`.
fn temp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    let n = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    name.push(format!(".{}.{n}.tmp", std::process::id()));
    path.with_file_name(name)
}

/// [`replace`] with `bytes`, synced to the device before the rename, so a crash
/// right after leaves the new file rather than an empty one.
pub fn replace_durable(path: &Path, bytes: &[u8]) -> io::Result<()> {
    replace(path, |out| {
        out.write_all(bytes)?;
        out.flush()?;
        out.get_ref().sync_all()
    })
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
        let path = dir.path().join("nested").join("file");

        replace(&path, |out| out.write_all(b"first")).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"first");

        replace(&path, |out| out.write_all(b"second")).unwrap();
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
        replace(&path, |out| out.write_all(b"kept")).unwrap();

        assert!(
            replace(&path, |out| {
                out.write_all(b"partial")?;
                Err(io::Error::other("producer gave up"))
            })
            .is_err()
        );
        assert_eq!(fs::read(&path).unwrap(), b"kept");
        let leftovers = fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "tmp"))
            .count();
        assert_eq!(leftovers, 0, "a failed write leaves no temp behind");
    }

    #[test]
    fn a_durable_write_replaces_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("state");
        replace_durable(&path, b"old").unwrap();
        replace_durable(&path, b"new").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new");
    }

    // Siblings that differ only by extension, written at once, each land whole.
    #[test]
    fn concurrent_writes_to_siblings_differing_by_extension_stay_apart() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("state.cbor");
        let b = dir.path().join("state.json");
        std::thread::scope(|scope| {
            for (path, byte) in [(&a, b'a'), (&b, b'b')] {
                scope.spawn(move || {
                    for _ in 0..50 {
                        replace_durable(path, &[byte; 64]).unwrap();
                    }
                });
            }
        });
        assert_eq!(fs::read(&a).unwrap(), [b'a'; 64]);
        assert_eq!(fs::read(&b).unwrap(), [b'b'; 64]);
        assert!(temp_path(&a).to_string_lossy().contains("state.cbor."));
        assert_ne!(temp_path(&a), temp_path(&a));
    }

    // A directory that cannot be created fails the write and creates nothing.
    #[test]
    fn a_directory_that_cannot_be_created_fails_the_write() {
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("blocker");
        fs::write(&blocker, b"a file, not a directory").unwrap();

        let path = blocker.join("nested").join("file");
        assert!(replace(&path, |out| out.write_all(b"bytes")).is_err());
        assert!(!path.exists());
    }
}
