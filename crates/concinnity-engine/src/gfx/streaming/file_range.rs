// Byte-range reads for the disk-backed streaming sources.
//
// A streamed payload is re-read from disk on demand rather than held
// RAM-resident, so both the texture source (reading its blob file) and the mesh
// source (reading its geometry scratch file) need the same seek-and-read.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

// Read `len` bytes from `path` starting at absolute `file_offset`.
//
// Errors are human-readable strings: they reach the renderer through the
// background worker, which has nothing to act on a typed error with.
pub(super) fn read_at(path: &str, file_offset: u64, len: u64) -> Result<Vec<u8>, String> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {}", path, e))?;
    file.seek(SeekFrom::Start(file_offset))
        .map_err(|e| format!("seek {} in {}: {}", file_offset, path, e))?;
    let mut bytes = vec![0u8; len as usize];
    file.read_exact(&mut bytes)
        .map_err(|e| format!("read {} bytes from {}: {}", len, path, e))?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn scratch(bytes: &[u8]) -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payload").to_string_lossy().into_owned();
        File::create(&path).unwrap().write_all(bytes).unwrap();
        (dir, path)
    }

    #[test]
    fn reads_a_range_from_the_middle() {
        let (_dir, path) = scratch(b"hello world");
        assert_eq!(read_at(&path, 6, 5).unwrap(), b"world");
    }

    #[test]
    fn reads_a_zero_length_range() {
        let (_dir, path) = scratch(b"hello");
        assert!(read_at(&path, 2, 0).unwrap().is_empty());
    }

    #[test]
    fn errors_when_the_range_runs_past_the_end() {
        let (_dir, path) = scratch(b"short");
        let e = read_at(&path, 0, 99).unwrap_err();
        assert!(e.starts_with("read 99 bytes from "), "{}", e);
    }

    #[test]
    fn errors_when_the_offset_is_past_the_end() {
        let (_dir, path) = scratch(b"short");
        assert!(read_at(&path, 99, 1).is_err());
    }

    #[test]
    fn errors_on_a_missing_file() {
        let e = read_at("/nonexistent/cn/payload", 0, 1).unwrap_err();
        assert!(e.starts_with("open /nonexistent/cn/payload: "), "{}", e);
    }
}
