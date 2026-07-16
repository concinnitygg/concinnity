// src/editor/file_dialog.rs
//
// The native file picker behind the Import panel's Browse button, and the
// project-relative rewrite its result needs. `pick_import_file` is a thin
// wrapper over `rfd` (NSOpenPanel on macOS, IFileDialog on Windows, the XDG
// desktop portal on Linux) and is the one part here that cannot be tested
// headlessly -- it opens a modal OS dialog. Everything done with the result
// goes through `project_path`, which is pure but for its canonicalization and
// is unit tested. Nothing in this module reaches a shipped game: the runtime
// never links the editor crate.

use std::path::{Path, PathBuf};

// Open a native "open file" dialog rooted at `start`, filtered to the
// extensions `entry_from_path` resolves. Returns `None` when the user cancels.
//
// Blocks the calling thread -- and so the editor's frame loop -- until the
// dialog closes; it is modal and frontmost, so the still window behind it is
// expected rather than a hang. Must be called from the main thread (macOS
// requires NSOpenPanel there), which the editor hook's tick already runs on.
pub(crate) fn pick_import_file(start: &Path) -> Option<PathBuf> {
    let groups = crate::authoring::IMPORT_EXTENSION_GROUPS;
    // An "everything" filter first so the user never has to guess a category,
    // then one per group for narrowing.
    let all: Vec<&str> = groups
        .iter()
        .flat_map(|(_, exts)| exts.iter().copied())
        .collect();
    let mut dialog = rfd::FileDialog::new()
        .set_title("Import a file")
        .set_directory(start)
        .add_filter("All supported", &all);
    for (name, exts) in groups {
        dialog = dialog.add_filter(*name, exts);
    }
    dialog.pick_file()
}

// The text to put in the Import panel's path field for a picked file: relative
// to `root` (the project directory) when the file sits inside it, so
// world.jsonl stays portable across machines; the absolute path otherwise,
// which resolves on this machine but will not survive the project moving.
//
// Both sides are canonicalized first, so a symlinked root still matches (macOS
// hands out `/tmp` for `/private/tmp`, and a picked file comes back resolved).
pub(crate) fn project_path(picked: &Path, root: &Path) -> String {
    let picked = canonical_or_self(picked);
    let root = canonical_or_self(root);
    let rel = picked.strip_prefix(&root).unwrap_or(&picked);
    to_slash(rel)
}

// The resolved path, or the original when it cannot be resolved (a path that
// no longer exists, a permission error); relativizing is best-effort.
fn canonical_or_self(p: &Path) -> PathBuf {
    p.canonicalize().unwrap_or_else(|_| p.to_path_buf())
}

// Path text for world.jsonl. Windows separators become `/` so a world authored
// there still resolves on macOS / Linux (Windows accepts forward slashes); a
// Unix path is left alone, where `\` is a legal filename character.
fn to_slash(p: &Path) -> String {
    let s = p.to_string_lossy().into_owned();
    if cfg!(windows) {
        s.replace('\\', "/")
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A file inside the project comes back relative, so the world is portable.
    #[test]
    fn project_path_relativizes_inside_the_project() {
        let root = tempfile::tempdir().unwrap();
        let assets = root.path().join("assets");
        std::fs::create_dir_all(&assets).unwrap();
        let file = assets.join("crate.glb");
        std::fs::write(&file, b"glb").unwrap();
        assert_eq!(project_path(&file, root.path()), "assets/crate.glb");
    }

    // A file outside the project keeps its absolute path: there is no portable
    // way to name it, and an absolute path at least resolves here.
    #[test]
    fn project_path_keeps_outside_files_absolute() {
        let root = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let file = other.path().join("far.glb");
        std::fs::write(&file, b"glb").unwrap();
        let text = project_path(&file, root.path());
        assert!(
            std::path::Path::new(&text).is_absolute(),
            "an outside file stays absolute: {text}"
        );
        assert!(text.ends_with("far.glb"));
    }

    // The root and the picked path may name the same directory differently (a
    // symlinked root, `.` segments); canonicalizing both keeps the relative
    // rewrite working. macOS's `/tmp` -> `/private/tmp` is exactly this case.
    #[test]
    fn project_path_matches_through_symlinks_and_dot_segments() {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("scene.glb");
        std::fs::write(&file, b"glb").unwrap();
        // The same file reached via a `.` segment still relativizes.
        let dotted = root.path().join(".").join("scene.glb");
        assert_eq!(project_path(&dotted, root.path()), "scene.glb");

        // And via a symlink to the project root.
        let link_home = tempfile::tempdir().unwrap();
        let link = link_home.path().join("link");
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(root.path(), &link).unwrap();
            assert_eq!(
                project_path(&link.join("scene.glb"), root.path()),
                "scene.glb",
                "a symlinked path resolves to the same project file"
            );
        }
        #[cfg(not(unix))]
        let _ = link;
    }

    // A picked file that has since vanished cannot canonicalize; the rewrite
    // falls back to the raw path instead of panicking.
    #[test]
    fn project_path_falls_back_when_a_path_cannot_resolve() {
        let root = tempfile::tempdir().unwrap();
        let missing = root.path().join("gone.glb");
        let text = project_path(&missing, root.path());
        assert!(text.ends_with("gone.glb"), "got: {text}");
    }
}
