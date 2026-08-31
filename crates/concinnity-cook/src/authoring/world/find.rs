/// The world source file's conventional name.
pub const WORLD_JSONL: &str = "world.jsonl";
/// Locate a world JSONL file, searching `worlds_dir` first.
///
/// If `name` is given, returns `worlds_dir/<name>.jsonl` when it exists. If
/// `name` is None, returns the most recently modified `.jsonl` there. Falls
/// back to `world.jsonl` in the current directory and then walks up parent
/// directories, which is also the whole search when the caller has no
/// `worlds/` to offer.
pub fn find_world_jsonl(
    worlds_dir: Option<&std::path::Path>,
    name: Option<&str>,
) -> std::io::Result<String> {
    if let Some(n) = name {
        let path = worlds_dir.map(|d| d.join(format!("{}.jsonl", n)));
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
        if let Ok(entries) = std::fs::read_dir(worlds_dir) {
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
// misses -- a `worlds/` holding no such world, and no `worlds/` at all -- read
// the same way to a caller.
fn named_world_not_found(name: &str, path: Option<&std::path::Path>) -> std::io::Error {
    let message = match path {
        Some(p) => format!("world '{}' not found at {}", name, p.display()),
        None => format!("world '{}' not found: no worlds directory", name),
    };
    std::io::Error::new(std::io::ErrorKind::NotFound, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The named-miss branches are exercised through the pure error builder: the
    // lookup itself walks up from the working directory, which is shared with
    // every other test in this binary, so moving it would race them.
    #[test]
    fn a_missing_named_world_names_the_file_it_looked_for() {
        let path = std::path::Path::new("/proj/worlds/cn_test_no_such_world.jsonl");
        let err = named_world_not_found("cn_test_no_such_world", Some(path));
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        let msg = err.to_string();
        assert!(msg.contains("cn_test_no_such_world"), "message was: {msg}");
        assert!(msg.contains(".jsonl"), "message was: {msg}");
    }

    // With no `worlds/` there is no file to name, so the message says what is
    // missing instead of pointing at a path nobody chose.
    #[test]
    fn a_named_world_without_a_worlds_directory_says_so() {
        let err = named_world_not_found("main", None);
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        let msg = err.to_string();
        assert!(msg.contains("main"), "message was: {msg}");
        assert!(msg.contains("no worlds directory"), "message was: {msg}");
    }

    // A named world is looked for in the directory the caller gave and nowhere
    // else: a miss there never falls through to the cwd walk, which would open
    // some other world under the name the caller asked for.
    #[test]
    fn a_named_world_is_looked_for_only_in_the_directory_given() {
        let tree = concinnity_testing::TempTree::new();
        let err = find_world_jsonl(Some(tree.path()), Some("cn_test_no_such_world")).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        let msg = err.to_string();
        assert!(msg.contains("cn_test_no_such_world"), "message was: {msg}");
        assert!(
            msg.contains(&tree.path().display().to_string()),
            "message was: {msg}"
        );
    }

    // An unnamed lookup takes the most recently modified world in the directory
    // given, so `cn build` with no argument opens whatever was last edited.
    #[test]
    fn an_unnamed_lookup_takes_the_newest_world_in_the_directory() {
        let tree = concinnity_testing::TempTree::new();
        let older = tree.path().join("older.jsonl");
        let newer = tree.path().join("newer.jsonl");
        std::fs::write(&older, "{}").unwrap();
        std::fs::write(&newer, "{}").unwrap();
        // `write` order does not guarantee distinct mtimes at this resolution;
        // set them explicitly so the pick is the one under test.
        let base = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000);
        set_mtime(&older, base);
        set_mtime(&newer, base + std::time::Duration::from_secs(60));

        let found = find_world_jsonl(Some(tree.path()), None).unwrap();
        assert_eq!(std::path::Path::new(&found), newer);
    }

    fn set_mtime(path: &std::path::Path, at: std::time::SystemTime) {
        let file = std::fs::File::options().write(true).open(path).unwrap();
        file.set_modified(at).unwrap();
    }
}
