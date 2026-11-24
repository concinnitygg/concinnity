//! The small file moves every install shares.

use std::fs;
use std::path::{Path, PathBuf};

use crate::{Context, Result};

/// Put `built` at `payload`, which only exists once the copy is whole.
///
/// # Errors
///
/// When the copy or the move into place fails.
pub fn install(built: &Path, payload: &Path) -> Result {
    if let Some(parent) = payload.parent() {
        fs::create_dir_all(parent).context(parent.display())?;
    }
    let mut incoming = payload.as_os_str().to_owned();
    incoming.push(".incoming");
    fs::copy(built, &incoming).context(format!("copy {}", built.display()))?;
    fs::rename(&incoming, payload).context(payload.display())
}

/// Remove `dir` if it is there, ignoring what was not.
pub fn remove_dir_quietly(dir: &Path) {
    let _removed = fs::remove_dir_all(dir);
}

/// Where an install is assembled before it replaces `target` whole.
pub fn staging_dir(target: &Path) -> PathBuf {
    let name = target.file_name().unwrap_or_default().to_string_lossy();
    target.with_file_name(format!(".{name}.incoming"))
}

/// The last component of `path`, for messages.
pub fn file_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_sits_beside_the_target() {
        assert_eq!(
            staging_dir(Path::new("/vendor/dxc-1.9.2607-linux-x86_64")),
            Path::new("/vendor/.dxc-1.9.2607-linux-x86_64.incoming")
        );
    }

    #[test]
    fn install_creates_the_payload_directory() {
        let dir = tempfile::tempdir().unwrap();
        let built = dir.path().join("built.dll");
        fs::write(&built, b"dll").unwrap();
        let payload = dir.path().join("sdk/bin/out.dll");
        install(&built, &payload).unwrap();
        assert_eq!(fs::read(&payload).unwrap(), b"dll");
        assert!(!dir.path().join("sdk/bin/out.dll.incoming").exists());
    }
}
