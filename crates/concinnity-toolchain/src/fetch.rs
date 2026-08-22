// First-run asset fetching for the workspace examples.
//
// An example that needs a large asset pack downloads and unpacks it on first
// run rather than at build time, so `cargo build` never touches the network.
// The policy (where the pack lives, what overrides the source) stays with the
// caller; this module owns the mechanics.

use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

/// A zipped asset pack an example needs on disk before it can run.
pub struct ZipFetch<'a> {
    /// Downloaded when the pack is missing and no local archive is given.
    pub url: &'a str,
    /// Directory the archive is unpacked into, created if absent.
    pub extract_to: &'a Path,
    /// Directory the sentinels are resolved against.
    pub root: &'a Path,
    /// Files that must all exist for the pack to count as present.
    pub sentinels: &'a [&'a str],
    /// An already-downloaded archive to unpack instead of fetching one.
    pub local_archive: Option<PathBuf>,
}

/// Whether every sentinel exists under `root`.
pub fn present(root: &Path, sentinels: &[&str]) -> bool {
    sentinels.iter().all(|rel| root.join(rel).is_file())
}

/// Download and unpack `fetch` unless its sentinels are already on disk.
pub fn ensure(fetch: &ZipFetch) -> io::Result<()> {
    if present(fetch.root, fetch.sentinels) {
        return Ok(());
    }

    let (archive, downloaded) = match &fetch.local_archive {
        Some(local) => {
            eprintln!("using local archive: {}", local.display());
            (local.clone(), false)
        }
        None => {
            eprintln!("assets not found, downloading from {}", fetch.url);
            let tmp = std::env::temp_dir().join("concinnity_asset_download");
            download(fetch.url, &tmp)?;
            (tmp, true)
        }
    };

    let mut header = [0u8; 4];
    {
        let mut f = std::fs::File::open(&archive)?;
        f.read_exact(&mut header)?;
    }
    if !looks_like_zip(&header) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "downloaded archive is not a ZIP (first bytes {header:02x?}). The download URL \
                 may have changed format; fetch the pack manually and unpack it into {}.",
                fetch.extract_to.display()
            ),
        ));
    }

    eprintln!("extracting into {} ...", fetch.extract_to.display());
    std::fs::create_dir_all(fetch.extract_to)?;
    extract_zip(&archive, fetch.extract_to)?;

    // Only a download created the temp file; a caller-supplied archive is left
    // where it was.
    if downloaded {
        let _ = std::fs::remove_file(&archive);
    }

    if !present(fetch.root, fetch.sentinels) {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "extraction finished but the expected files are still missing; the archive layout \
             may differ from what the sentinels describe.",
        ));
    }

    eprintln!("assets ready.");
    Ok(())
}

// Stream a URL to a file, printing coarse progress. Streams to disk rather than
// memory because these archives run to gigabytes.
fn download(url: &str, dest: &Path) -> io::Result<()> {
    let resp = ureq::get(url)
        .call()
        .map_err(|e| io::Error::other(format!("request failed: {e}")))?;

    let status = resp.status();
    if !status.is_success() {
        return Err(io::Error::other(format!("server returned HTTP {status}")));
    }

    let total: Option<u64> = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok());

    let mut reader = resp.into_body().into_reader();
    let mut file = std::fs::File::create(dest)?;

    let mut buf = vec![0u8; 1 << 20];
    let mut downloaded: u64 = 0;
    let mut last_report: u64 = 0;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        downloaded += n as u64;
        if downloaded - last_report >= 64 << 20 {
            report_progress(downloaded, total);
            last_report = downloaded;
        }
    }
    report_progress(downloaded, total);
    eprintln!();
    Ok(())
}

fn report_progress(downloaded: u64, total: Option<u64>) {
    let mib = |b: u64| b as f64 / (1 << 20) as f64;
    match total {
        Some(t) if t > 0 => eprint!(
            "\r  downloaded {:.0} / {:.0} MiB ({:.0}%)   ",
            mib(downloaded),
            mib(t),
            downloaded as f64 / t as f64 * 100.0
        ),
        _ => eprint!("\r  downloaded {:.0} MiB   ", mib(downloaded)),
    }
    let _ = io::stderr().flush();
}

// Unpack a ZIP archive into a destination directory, preserving its internal
// paths.
fn extract_zip(archive: &Path, dest: &Path) -> io::Result<()> {
    let file = std::fs::File::open(archive)?;
    let mut zip = zip::ZipArchive::new(file)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("not a valid zip: {e}")))?;
    zip.extract(dest)
        .map_err(|e| io::Error::other(format!("extract failed: {e}")))?;
    Ok(())
}

// True when an archive begins with the ZIP local-file-header magic ("PK\x03\x04").
fn looks_like_zip(header: &[u8]) -> bool {
    header.starts_with(&[0x50, 0x4b, 0x03, 0x04])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zip_magic_is_recognised() {
        assert!(looks_like_zip(&[0x50, 0x4b, 0x03, 0x04, 0x14]));
        // gzip and tar magics, and a short slice, are not ZIPs.
        assert!(!looks_like_zip(&[0x1f, 0x8b, 0x08, 0x00]));
        assert!(!looks_like_zip(&[0x50, 0x4b]));
        assert!(!looks_like_zip(&[]));
    }

    #[test]
    fn present_requires_every_sentinel() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        let nested = root.join("assets/pack");
        std::fs::create_dir_all(&nested).unwrap();
        let sentinels = ["assets/pack/one.bin", "assets/pack/two.bin"];

        assert!(!present(root, &sentinels), "no files yet");

        std::fs::write(nested.join("one.bin"), b"x").unwrap();
        assert!(!present(root, &sentinels), "one alone is not enough");

        std::fs::write(nested.join("two.bin"), b"x").unwrap();
        assert!(present(root, &sentinels), "both sentinels present");
    }
}
