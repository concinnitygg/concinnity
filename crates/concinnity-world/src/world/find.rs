/// The world source file's conventional name.
pub const WORLD_JSONL: &str = "world.jsonl";
/// Locate a world JSONL file.
///
/// If `name` is given, returns the state root's `worlds/<name>.jsonl` when it
/// exists. If `name` is None, returns the most recently modified `.jsonl` in
/// `worlds/`. Falls back to `world.jsonl` in the current directory and then
/// walks up parent directories, which is also the whole search when no state
/// root is installed.
pub fn find_world_jsonl(name: Option<&str>) -> std::io::Result<String> {
    let worlds_dir = crate::paths::worlds_dir();

    if let Some(n) = name {
        let path = worlds_dir
            .as_deref()
            .map(|d| d.join(format!("{}.jsonl", n)));
        if let Some(path) = &path
            && path.exists()
        {
            return Ok(path.to_string_lossy().into_owned());
        }
        return Err(named_world_not_found(n, path.as_deref()));
    }

    // No name given: pick the most recently modified world in `worlds/`.
    if let Some(worlds_dir) = worlds_dir.filter(|d| d.is_dir()) {
        let mut best: Option<(std::time::SystemTime, std::path::PathBuf)> = None;
        if let Ok(entries) = std::fs::read_dir(&worlds_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                    continue;
                }
                if let Ok(meta) = std::fs::metadata(&path) {
                    let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
                    if best.as_ref().map(|(t, _)| mtime > *t).unwrap_or(true) {
                        best = Some((mtime, path));
                    }
                }
            }
        }
        if let Some((_, path)) = best {
            return Ok(path.to_string_lossy().into_owned());
        }
    }

    // Fall back to world.jsonl in cwd or any parent directory.
    let mut dir = std::env::current_dir()?;
    loop {
        let candidate = dir.join(WORLD_JSONL);
        if candidate.exists() {
            return Ok(candidate.to_string_lossy().into_owned());
        }
        match dir.parent() {
            Some(parent) => dir = parent.to_path_buf(),
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "no world found: run `cn fetch-world` or create `{}`",
                        WORLD_JSONL,
                    ),
                ));
            }
        }
    }
}

// The error for a named world that could not be located. Split out so both
// misses -- a state root holding no such world, and no state root at all -- are
// testable without touching the process-global anchor.
fn named_world_not_found(name: &str, path: Option<&std::path::Path>) -> std::io::Error {
    let message = match path {
        Some(p) => format!("world '{}' not found at {}", name, p.display()),
        None => format!("world '{}' not found: no project state directory", name),
    };
    std::io::Error::new(std::io::ErrorKind::NotFound, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The named-miss branches are exercised through the pure error builder: the
    // lookup itself reads the process-global path anchors and walks up from the
    // cwd, both of which are shared with other tests in this binary (paths.rs
    // owns the global mutation), so redirecting them would race those tests.
    #[test]
    fn a_missing_named_world_names_the_file_it_looked_for() {
        let path = std::path::Path::new("/proj/worlds/cn_test_no_such_world.jsonl");
        let err = named_world_not_found("cn_test_no_such_world", Some(path));
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        let msg = err.to_string();
        assert!(msg.contains("cn_test_no_such_world"), "message was: {msg}");
        assert!(msg.contains(".jsonl"), "message was: {msg}");
    }

    // With no state root there is no file to name, so the message says what is
    // missing instead of pointing at a path nobody chose.
    #[test]
    fn a_named_world_without_a_state_root_says_so() {
        let err = named_world_not_found("main", None);
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        let msg = err.to_string();
        assert!(msg.contains("main"), "message was: {msg}");
        assert!(
            msg.contains("no project state directory"),
            "message was: {msg}"
        );
    }

    // The lookup still fails cleanly for a name nothing on this host provides.
    #[test]
    fn missing_named_world_is_a_not_found_error() {
        let err = find_world_jsonl(Some("cn_test_no_such_world")).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(err.to_string().contains("cn_test_no_such_world"));
    }
}
