//! Copies of the files an entry owns, for a duplicate that must not share
//! them: every field its type marks as an owned file (the registry's
//! `owned_file_fields`) is copied beside the original as `<stem>_copy.<ext>`,
//! numbered past any file already there, and the entry is pointed at the copy.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use concinnity_cook::authoring::field_path::{retarget_leaves, string_leaves};
use concinnity_cook::authoring::registry::RegisteredType;

// One owned file an entry declared, and what became of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Copied {
    // Copied: the declared original and the declared copy the entry now names.
    To { from: String, to: String },
    // Not copied, with why; the entry still names the original.
    Failed { from: String, reason: String },
}

// Every distinct owned file `entry` declares, in field order.
pub(crate) fn declared_files(entry: &serde_json::Value) -> Vec<String> {
    let Some(ty) = entry
        .get("type")
        .and_then(|t| t.as_str())
        .and_then(RegisteredType::parse)
    else {
        return Vec::new();
    };
    let Some(args) = entry.get("args") else {
        return Vec::new();
    };
    let mut out: Vec<String> = Vec::new();
    for field in ty.owned_file_fields() {
        for (_, declared) in string_leaves(args, field) {
            if !declared.is_empty() && !out.iter().any(|d| d == declared) {
                out.push(declared.to_string());
            }
        }
    }
    out
}

// Where the copy of `path` goes: `<stem>_copy.<ext>` beside it, then
// `<stem>_copy_1.<ext>` and on, past every name `taken` reports.
pub(crate) fn copy_path(path: &Path, taken: impl Fn(&Path) -> bool) -> PathBuf {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy())
        .unwrap_or_default();
    let ext = path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()));
    let ext = ext.unwrap_or_default();
    let named = |suffix: String| path.with_file_name(format!("{stem}_copy{suffix}{ext}"));
    std::iter::once(named(String::new()))
        .chain((1..).map(|i| named(format!("_{i}"))))
        .find(|p| !taken(p))
        .expect("an unbounded search finds a free name")
}

// `declared` naming the file `name` in the same place.
pub(crate) fn with_file_name(declared: &str, name: &str) -> String {
    match declared.rfind(['/', '\\']) {
        Some(i) => format!("{}{name}", &declared[..=i]),
        None => name.to_string(),
    }
}

// Copy every owned file `clone` declares and point it at the copies. A file
// is resolved through `resolve` (as the build finds it); a bare name also
// counts as taken wherever `find` finds it, since it resolves by name alone.
pub(crate) fn copy_owned(
    clone: &mut serde_json::Value,
    resolve: impl Fn(&str) -> String,
    find: impl Fn(&str) -> bool,
) -> Vec<Copied> {
    let mut done: BTreeMap<String, String> = BTreeMap::new();
    let mut out = Vec::new();
    for from in declared_files(clone) {
        let source = PathBuf::from(resolve(&from));
        let bare = !from.contains(['/', '\\']);
        let target = copy_path(&source, |p| {
            p.exists() || (bare && p.file_name().is_some_and(|n| find(&n.to_string_lossy())))
        });
        let name = target.file_name().unwrap_or_default().to_string_lossy();
        let to = with_file_name(&from, &name);
        match std::fs::copy(&source, &target) {
            Ok(_) => {
                done.insert(from.clone(), to.clone());
                out.push(Copied::To { from, to });
            }
            Err(e) => out.push(Copied::Failed {
                from,
                reason: e.to_string(),
            }),
        }
    }
    retarget(clone, &done);
    out
}

// Point every owned-file field of `entry` naming a key of `moves` at its value.
fn retarget(entry: &mut serde_json::Value, moves: &BTreeMap<String, String>) {
    let Some(ty) = entry
        .get("type")
        .and_then(|t| t.as_str())
        .and_then(RegisteredType::parse)
    else {
        return;
    };
    let Some(args) = entry.get_mut("args") else {
        return;
    };
    for field in ty.owned_file_fields() {
        for (from, to) in moves {
            retarget_leaves(args, field, from, Some(to));
        }
    }
}

// The toast after a duplicate that copied or failed to copy files.
pub(crate) fn copied_message(copied: &[Copied]) -> Option<String> {
    let names = |p: &str| p.rsplit(['/', '\\']).next().unwrap_or(p).to_string();
    let made: Vec<String> = copied
        .iter()
        .filter_map(|c| match c {
            Copied::To { from, to } => Some(format!("{} to {}", names(from), names(to))),
            Copied::Failed { .. } => None,
        })
        .collect();
    let failed: Vec<String> = copied
        .iter()
        .filter_map(|c| match c {
            Copied::Failed { from, reason } => Some(format!("{from}: {reason}")),
            Copied::To { .. } => None,
        })
        .collect();
    let mut out = Vec::new();
    if !made.is_empty() {
        out.push(format!("Copied {}", made.join(", ")));
    }
    if !failed.is_empty() {
        out.push(format!(
            "could not copy {}; the copy still reads the original",
            failed.join("; ")
        ));
    }
    (!out.is_empty()).then(|| out.join("; "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn owned_files_are_the_marked_fields_once_each() {
        let shader = json!({"type": "Shader", "args": {
            "$id": "s", "fragment": "a.hlsl", "vertex": "a.hlsl",
        }});
        assert_eq!(declared_files(&shader), ["a.hlsl"]);
        let story = json!({"type": "StoryImport", "args": {"source": "story/tale.md"}});
        assert_eq!(declared_files(&story), ["story/tale.md"]);
        let mesh = json!({"type": "Mesh", "args": {"source": "rock.glb"}});
        assert!(declared_files(&mesh).is_empty(), "an import is not owned");
    }

    #[test]
    fn a_copy_is_named_beside_the_original_past_taken_names() {
        let original = Path::new("/cn-none/shaders/water.hlsl");
        let taken = [
            PathBuf::from("/cn-none/shaders/water_copy.hlsl"),
            PathBuf::from("/cn-none/shaders/water_copy_1.hlsl"),
        ];
        assert_eq!(
            copy_path(original, |p| taken.iter().any(|t| t == p)),
            Path::new("/cn-none/shaders/water_copy_2.hlsl")
        );
        assert_eq!(
            copy_path(Path::new("/cn-none/notes"), |_| false),
            Path::new("/cn-none/notes_copy")
        );
    }

    #[test]
    fn a_declared_path_keeps_its_directory() {
        assert_eq!(
            with_file_name("assets/s/a.hlsl", "b.hlsl"),
            "assets/s/b.hlsl"
        );
        assert_eq!(with_file_name("a.hlsl", "b.hlsl"), "b.hlsl");
    }

    // Each owned file is copied once, never over an existing file, and the
    // clone names the copies; an unowned path is left alone.
    #[test]
    fn copying_points_the_clone_at_fresh_copies() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("w.hlsl"), "fragment").unwrap();
        std::fs::write(root.join("w_copy.hlsl"), "someone else's").unwrap();
        std::fs::write(root.join("sway.hlsl"), "vertex").unwrap();
        let declared = |name: &str| root.join(name).to_string_lossy().into_owned();
        let mut clone = json!({"type": "Shader", "args": {
            "$id": "w_1", "fragment": declared("w.hlsl"), "vertex": declared("sway.hlsl"),
        }});
        let copied = copy_owned(&mut clone, str::to_string, |_| false);
        assert_eq!(copied.len(), 2);
        assert_eq!(clone["args"]["fragment"], declared("w_copy_1.hlsl"));
        assert_eq!(clone["args"]["vertex"], declared("sway_copy.hlsl"));
        let read = |n: &str| std::fs::read_to_string(root.join(n)).unwrap();
        assert_eq!(read("w_copy_1.hlsl"), "fragment");
        assert_eq!(read("w_copy.hlsl"), "someone else's", "never overwritten");
        assert_eq!(read("sway_copy.hlsl"), "vertex");
        let message = copied_message(&copied).unwrap();
        assert!(message.contains("w.hlsl to w_copy_1.hlsl"), "{message}");
    }

    // A bare name resolves by name anywhere under the assets, so a name used
    // elsewhere there is taken too; a missing original is reported and the
    // clone keeps naming it.
    #[test]
    fn a_bare_name_skips_names_found_elsewhere_and_a_missing_file_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let shaders = dir.path().join("shaders");
        std::fs::create_dir_all(&shaders).unwrap();
        std::fs::write(shaders.join("blob.hlsl"), "sdf").unwrap();
        let mut sdf = json!({"type": "SdfVolume", "args": {"fragment_shader": "blob.hlsl"}});
        let resolve = |d: &str| shaders.join(d).to_string_lossy().into_owned();
        let copied = copy_owned(&mut sdf, resolve, |name| name == "blob_copy.hlsl");
        assert_eq!(
            copied,
            [Copied::To {
                from: "blob.hlsl".into(),
                to: "blob_copy_1.hlsl".into()
            }]
        );
        assert_eq!(sdf["args"]["fragment_shader"], "blob_copy_1.hlsl");

        let mut gone = json!({"type": "StoryImport", "args": {"source": "/cn-none/tale.md"}});
        let copied = copy_owned(&mut gone, str::to_string, |_| false);
        assert!(matches!(&copied[..], [Copied::Failed { .. }]));
        assert_eq!(gone["args"]["source"], "/cn-none/tale.md");
        assert!(copied_message(&copied).unwrap().contains("could not copy"));
    }
}
