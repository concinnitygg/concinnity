// CBOR-on-disk store shared by the engine's small persisted files: the settings
// store, story saves, and behavior fired-flag saves. All three want the same
// shape -- read-or-default on a missing/corrupt file, and a write that creates
// the parent directory.

use std::path::Path;

// Read `path` as CBOR. `None` when the file is absent; `None` with a warning
// naming `what` when it is present but unreadable, so a truncated or
// incompatible file starts fresh rather than failing the caller.
pub(crate) fn read<T: serde::de::DeserializeOwned>(path: &Path, what: &str) -> Option<T> {
    let bytes = std::fs::read(path).ok()?;
    match ciborium::from_reader(&bytes[..]) {
        Ok(value) => Some(value),
        Err(e) => {
            tracing::warn!("{what} unreadable, starting fresh: {e}");
            None
        }
    }
}

// Write `value` to `path` as CBOR, creating the parent directory as needed.
pub(crate) fn write<T: serde::Serialize>(path: &Path, value: &T) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut bytes = Vec::new();
    ciborium::into_writer(value, &mut bytes).map_err(std::io::Error::other)?;
    std::fs::write(path, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read_round_trips_through_a_created_directory() {
        let dir = std::env::temp_dir().join("cn_cbor_file_round_trip/nested");
        let path = dir.join("store");
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());

        write(&path, &vec![1u32, 2, 3]).expect("write creates the parent dir");
        assert_eq!(read::<Vec<u32>>(&path, "test store"), Some(vec![1, 2, 3]));

        let _ = std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
    }

    #[test]
    fn a_missing_or_corrupt_file_reads_as_none() {
        let dir = std::env::temp_dir().join("cn_cbor_file_corrupt");
        let path = dir.join("store");
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(read::<Vec<u32>>(&path, "test store"), None, "missing file");

        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, b"not cbor at all").unwrap();
        assert_eq!(read::<Vec<u32>>(&path, "test store"), None, "corrupt file");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
