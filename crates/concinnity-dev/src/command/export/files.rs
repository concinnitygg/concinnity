//! Copying, archiving and permissions for the files a bundle holds.

use std::path::{Path, PathBuf};
use std::{fs, io};

// Remove `dir` if it exists, then create it fresh, so re-exports are clean.
pub(super) fn reset_dir(dir: &Path) -> io::Result<()> {
    if dir.exists() {
        fs::remove_dir_all(dir)?;
    }
    fs::create_dir_all(dir)
}

// Copy every compiled blob (the integer-named files) from the build's data
// directory into `data_dst`, skipping the shader-compile intermediates the
// build leaves there (named after their asset). Returns the number copied.
// Copy the built blobs into the bundle as the `data` entry the player looks
// for, and report how many were copied.
//
// A world that fits in one blob ships as a single file named `data`; one that
// overflows ships as a `data/` directory holding `0`, `1`, ... Overflow blobs
// are always siblings of blob 0 named by index, so the single-file form can
// only carry a world that has none -- the player refuses the mismatch rather
// than reading `1` and `2` out of the folder it was launched from.
// Remove `path`, a file or a directory, when it exists.
pub(super) fn remove_any(path: &Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() => fs::remove_dir_all(path),
        Ok(_) => fs::remove_file(path),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

pub(super) fn copy_blobs(data_src: &Path, data_dst: &Path) -> io::Result<usize> {
    let blobs = blobs_in(data_src)?;
    if blobs.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("no compiled blobs in {}", data_src.display()),
        ));
    }
    // Whatever the previous export left here: the two forms occupy the same
    // name, so a shrinking world would otherwise write a file over a directory.
    remove_any(data_dst)?;

    if let [(_, only)] = blobs.as_slice() {
        if let Some(parent) = data_dst.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(only, data_dst)?;
        return Ok(1);
    }

    fs::create_dir_all(data_dst)?;
    for (index, path) in &blobs {
        fs::copy(path, data_dst.join(index.to_string()))?;
    }
    Ok(blobs.len())
}

// Every blob file in `dir`, paired with its index and ordered by it.
pub(super) fn blobs_in(dir: &Path) -> io::Result<Vec<(u32, PathBuf)>> {
    let mut blobs = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        if let Some(index) = blob_index(&name) {
            blobs.push((index, entry.path()));
        }
    }
    blobs.sort_by_key(|(index, _)| *index);
    Ok(blobs)
}

// A blob file is named by its integer index with no extension (blob_path uses
// `index.to_string()`); everything else in data/ is build scratch.
pub(super) fn blob_index(name: &str) -> Option<u32> {
    if name.is_empty() || !name.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    name.parse().ok()
}

#[cfg(unix)]
pub(super) fn make_executable(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = fs::metadata(path)?.permissions();
    perms.set_mode(0o755);
    fs::set_permissions(path, perms)
}

#[cfg(not(unix))]
#[expect(
    clippy::unnecessary_wraps,
    reason = "matches the Unix version, which can fail"
)]
pub(super) fn make_executable(_path: &Path) -> io::Result<()> {
    Ok(())
}

// Recursively copy the directory tree at `src` to `dst`.
pub(super) fn copy_tree(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_tree(&from, &to)?;
        } else {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

// Zip the tree at `src_dir` under a single top-level `<top>/` folder, marking
// the player executable (`exe_rel`, relative to `src_dir`) executable so it
// stays runnable after extraction on Unix. Files are added in sorted order for
// a reproducible archive.
pub(super) fn zip_tree(
    src_dir: &Path,
    top: &str,
    exe_rel: &str,
    zip_path: &Path,
) -> io::Result<()> {
    use std::io::Write;
    use zip::write::SimpleFileOptions;

    let mut files = Vec::new();
    collect_files(src_dir, &mut files)?;
    files.sort();

    let file = fs::File::create(zip_path)?;
    let mut zw = zip::ZipWriter::new(file);
    for path in files {
        let rel = path
            .strip_prefix(src_dir)
            .map_err(io::Error::other)?
            .to_string_lossy()
            .replace('\\', "/");
        let mode = if rel == exe_rel { 0o755 } else { 0o644 };
        // Marked as made on Unix: zip defaults to DOS on a Windows host, and
        // unzip ignores the mode bits of a DOS entry, which would drop the
        // executable bit.
        let options = SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .system(zip::System::Unix)
            .unix_permissions(mode);
        zw.start_file(format!("{top}/{rel}"), options)
            .map_err(io::Error::other)?;
        let bytes = fs::read(&path)?;
        zw.write_all(&bytes)?;
    }
    zw.finish().map_err(io::Error::other)?;
    Ok(())
}

// Collect every file under `dir` (recursively) into `out`.
pub(super) fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else {
            out.push(path);
        }
    }
    Ok(())
}
