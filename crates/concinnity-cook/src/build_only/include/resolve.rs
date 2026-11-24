//! Resolving `Include` lines: each is replaced by the entries of the file it
//! names, recursively.
//!
//! Every file reads its own relative paths from its library root (see
//! [`concinnity_host::store::source`]): an `Include` line names a file from it,
//! and the build sees every other file path an entry holds rebased onto the
//! build's root, so a library included from a project names its files as it
//! does on its own.

use std::path::{Path, PathBuf};

use concinnity_host::store::source;
use serde_json::Value;

use super::schema::Include;
use crate::authoring::field_path::map_leaves;
use crate::authoring::registry::RegisteredType;
use crate::authoring::world::{ID_KEY, parse_world_jsonl};

/// One entry of a world whose includes are resolved, with the file it was
/// read from.
#[derive(Debug, Clone, PartialEq)]
pub struct SourcedEntry {
    /// The entry, as a `{"type", "args"}` object.
    pub entry: Value,
    /// The included file the entry was read from; `None` for an entry of the
    /// top-level file itself.
    pub file: Option<PathBuf>,
    /// The root the entry's own paths resolve against: its file's.
    pub root: Option<PathBuf>,
}

/// Whether `entry` is an `Include` line.
pub fn is_include(entry: &Value) -> bool {
    entry.get("type").and_then(Value::as_str) == Some(RegisteredType::Include.as_str())
}

/// The root the relative paths in `file` resolve against: its library root
/// within the build's root `within`, or `fallback` (the root of whatever
/// reached it) when it has none. With no file, or no build root to look within,
/// it is `fallback`.
pub fn file_root(
    file: Option<&Path>,
    fallback: Option<&Path>,
    within: Option<&Path>,
) -> Option<PathBuf> {
    file.zip(within)
        .and_then(|(file, within)| source::marked_root(file, within))
        .or_else(|| fallback.map(Path::to_path_buf))
}

/// Rewrite every file path `entry` holds from relative to `from` to relative to
/// `to`, so it names the same file from the build's root.
pub fn rebase_files(entry: &mut Value, from: &Path, to: &Path) {
    let Some(ty) = entry
        .get("type")
        .and_then(Value::as_str)
        .and_then(RegisteredType::parse)
    else {
        return;
    };
    let Some(args) = entry.get_mut("args") else {
        return;
    };
    for path in ty.file_fields() {
        map_leaves(args, path, &mut |p| source::rebase(p, from, to));
    }
}

/// Replace every `Include` in `entries` with the entries of the file it names,
/// recursively: the world the build sees, with every file path rebased onto
/// `root`.
///
/// `world_file` is the file `entries` were read from, named in errors and
/// seeding the cycle check. `root` is the build's root, and the fallback for a
/// file with no library root of its own; with neither, only an absolute path
/// resolves.
///
/// # Errors
///
/// Returns an error if an `Include` has malformed args, its file cannot be read
/// or parsed, or the includes form a cycle.
pub fn resolve_includes(
    entries: Vec<Value>,
    world_file: Option<&Path>,
    root: Option<&Path>,
) -> Result<Vec<Value>, String> {
    let sourced = Walk::new(world_file, root, Mode::Build).run(entries, world_file)?;
    Ok(sourced.into_iter().map(|s| s.entry).collect())
}

/// `entries` with every include resolved as [`resolve_includes`] does, but
/// each `Include` line kept in place ahead of the entries it inlines, every
/// entry tagged with the file it came from and that file's root, and every
/// entry as its file spells it, with no path rebased.
///
/// An `Include` line only counts toward the labels of its own type, so the
/// `<Type>#<n>` labels over this list are the ones the build gives.
///
/// # Errors
///
/// Returns an error under the same conditions as [`resolve_includes`].
pub fn with_includes(
    entries: Vec<Value>,
    world_file: Option<&Path>,
    root: Option<&Path>,
) -> Result<Vec<SourcedEntry>, String> {
    Walk::new(world_file, root, Mode::Authored).run(entries, world_file)
}

// What a walk produces: the build's view, or the authored one.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    // Include lines replaced, file paths rebased onto the build's root.
    Build,
    // Include lines kept, every entry as its file spells it.
    Authored,
}

struct Walk<'a> {
    root: Option<&'a Path>,
    mode: Mode,
    // The files being read, outermost first, as canonical paths beside the
    // spelling an error shows.
    stack: Vec<(PathBuf, PathBuf)>,
}

impl<'a> Walk<'a> {
    fn new(world_file: Option<&Path>, root: Option<&'a Path>, mode: Mode) -> Self {
        let stack = world_file
            .and_then(|f| Some((std::fs::canonicalize(f).ok()?, f.to_path_buf())))
            .into_iter()
            .collect();
        Self { root, mode, stack }
    }

    fn run(
        mut self,
        entries: Vec<Value>,
        world_file: Option<&Path>,
    ) -> Result<Vec<SourcedEntry>, String> {
        let mut out = Vec::with_capacity(entries.len());
        let file_root = file_root(world_file, self.root, self.root);
        self.walk(entries, world_file, file_root.as_deref(), None, &mut out)?;
        Ok(out)
    }

    // `file` is where `entries` were read from and `file_root` the root its
    // paths resolve against; `included` is the same file when it was reached
    // through an `Include`, which is what the entries are tagged with.
    fn walk(
        &mut self,
        entries: Vec<Value>,
        file: Option<&Path>,
        file_root: Option<&Path>,
        included: Option<&Path>,
        out: &mut Vec<SourcedEntry>,
    ) -> Result<(), String> {
        let sourced = |entry| SourcedEntry {
            entry,
            file: included.map(Path::to_path_buf),
            root: file_root.map(Path::to_path_buf),
        };
        let rebase = match (self.mode, file_root, self.root) {
            (Mode::Build, Some(from), Some(to)) => Some((from, to)),
            _ => None,
        };
        for mut entry in entries {
            if !is_include(&entry) {
                if let Some((from, to)) = rebase {
                    rebase_files(&mut entry, from, to);
                }
                out.push(sourced(entry));
                continue;
            }
            let target = target_path(&entry, file, file_root)?;
            let at =
                |msg: String| format!("Include '{}'{}: {msg}", target.display(), in_file(file));
            let canonical = std::fs::canonicalize(&target).map_err(|e| at(e.to_string()))?;
            if let Some(start) = self.stack.iter().position(|(c, _)| *c == canonical) {
                let chain: Vec<String> = self.stack[start..]
                    .iter()
                    .map(|(_, shown)| shown.display().to_string())
                    .chain([target.display().to_string()])
                    .collect();
                return Err(format!("Include cycle: {}", chain.join(" -> ")));
            }
            let text = std::fs::read_to_string(&target).map_err(|e| at(e.to_string()))?;
            let nested = parse_world_jsonl(&text).map_err(|e| {
                e.0.iter()
                    .map(|line| format!("{}: {line}", target.display()))
                    .collect::<Vec<_>>()
                    .join("\n")
            })?;
            if self.mode == Mode::Authored {
                out.push(sourced(entry));
            }
            let nested_root = self::file_root(Some(&target), file_root, self.root);
            self.stack.push((canonical, target.clone()));
            self.walk(
                nested,
                Some(&target),
                nested_root.as_deref(),
                Some(&target),
                out,
            )?;
            self.stack.pop();
        }
        Ok(())
    }
}

// The file an `Include` line in `file` names, joined onto that file's root.
fn target_path(
    entry: &Value,
    file: Option<&Path>,
    file_root: Option<&Path>,
) -> Result<PathBuf, String> {
    let at = |msg: String| format!("Include{}: {msg}", in_file(file));
    let mut args = entry.get("args").cloned().unwrap_or(Value::Null);
    if args.is_null() {
        args = Value::Object(serde_json::Map::default());
    }
    if args.get(ID_KEY).is_some() {
        return Err(at(format!(
            "an Include takes no `{ID_KEY}`: it is replaced by the entries it names"
        )));
    }
    let include: Include = serde_path_to_error::deserialize(args)
        .map_err(|e| at(format!("invalid args: `{}`: {}", e.path(), e.inner())))?;
    if include.path.is_empty() {
        return Err(at("`path` is empty".to_string()));
    }
    if file_root.is_none() && Path::new(&include.path).is_relative() {
        return Err(at(format!(
            "'{}' is relative, and there is no root to resolve it against",
            include.path
        )));
    }
    Ok(source::resolve(&include.path, file_root))
}

fn in_file(file: Option<&Path>) -> String {
    file.map(|f| format!(" in {}", f.display()))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn write(dir: &Path, name: &str, contents: &str) -> PathBuf {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&path, contents).unwrap();
        path
    }

    fn include(path: &str) -> Value {
        json!({"type": "Include", "args": {"path": path}})
    }

    fn window(id: &str) -> Value {
        json!({"type": "Window", "args": {"$id": id}})
    }

    fn ids(entries: &[Value]) -> Vec<&str> {
        entries
            .iter()
            .map(|e| e["args"]["$id"].as_str().unwrap_or("-"))
            .collect()
    }

    // Resolve `entries` as read from `world` under the content root `dir`.
    fn resolve(entries: Vec<Value>, dir: &Path, world: &Path) -> Result<Vec<Value>, String> {
        resolve_includes(entries, Some(world), Some(dir))
    }

    #[test]
    fn entries_without_includes_pass_through() {
        let out = resolve_includes(vec![window("a")], None, None).unwrap();
        assert_eq!(out, [window("a")]);
    }

    #[test]
    fn an_include_is_replaced_in_place_by_its_file() {
        let dir = tempfile::tempdir().unwrap();
        let world = write(dir.path(), "world.jsonl", "");
        write(
            dir.path(),
            "chunk.jsonl",
            "[\"Window\",{\"$id\":\"b\"}]\n[\"Window\",{\"$id\":\"c\"}]\n",
        );
        let out = resolve(
            vec![window("a"), include("chunk.jsonl"), window("d")],
            dir.path(),
            &world,
        )
        .unwrap();
        assert_eq!(ids(&out), ["a", "b", "c", "d"]);
    }

    // A path is taken from the content root at every depth, never from the
    // directory of the file the line is in.
    #[test]
    fn every_include_resolves_against_the_content_root() {
        let dir = tempfile::tempdir().unwrap();
        let world = write(dir.path(), "worlds/world.jsonl", "");
        write(
            dir.path(),
            "parts/outer.jsonl",
            "[\"Window\",{\"$id\":\"outer\"}]\n[\"Include\",{\"path\":\"leaves/leaf.jsonl\"}]\n",
        );
        write(
            dir.path(),
            "leaves/leaf.jsonl",
            "[\"Window\",{\"$id\":\"leaf\"}]\n",
        );
        write(
            dir.path(),
            "parts/leaves/leaf.jsonl",
            "[\"Window\",{\"$id\":\"beside\"}]\n",
        );
        let out = resolve(vec![include("parts/outer.jsonl")], dir.path(), &world).unwrap();
        assert_eq!(ids(&out), ["outer", "leaf"]);
    }

    #[test]
    fn a_path_beside_the_including_file_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let world = write(dir.path(), "worlds/world.jsonl", "");
        write(dir.path(), "worlds/chunk.jsonl", "[\"Scene\"]\n");
        let err = resolve(vec![include("chunk.jsonl")], dir.path(), &world).unwrap_err();
        assert!(err.contains("chunk.jsonl"), "{err}");
    }

    #[test]
    fn an_absolute_path_resolves_without_a_root() {
        let dir = tempfile::tempdir().unwrap();
        let chunk = write(dir.path(), "chunk.jsonl", "[\"Window\",{\"$id\":\"x\"}]\n");
        let out = resolve_includes(vec![include(chunk.to_str().unwrap())], None, None).unwrap();
        assert_eq!(ids(&out), ["x"]);
    }

    #[test]
    fn a_relative_path_needs_a_root() {
        let dir = tempfile::tempdir().unwrap();
        let world = write(dir.path(), "world.jsonl", "");
        let err = resolve_includes(vec![include("chunk.jsonl")], Some(&world), None).unwrap_err();
        assert!(err.contains("'chunk.jsonl' is relative"), "{err}");
    }

    // A world held in memory has no file, and still resolves under a root.
    #[test]
    fn a_world_without_a_file_resolves_under_its_root() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "chunk.jsonl", "[\"Window\",{\"$id\":\"x\"}]\n");
        let out = resolve_includes(vec![include("chunk.jsonl")], None, Some(dir.path())).unwrap();
        assert_eq!(ids(&out), ["x"]);
    }

    #[test]
    fn a_cycle_is_an_error_naming_the_chain() {
        let dir = tempfile::tempdir().unwrap();
        let world = write(dir.path(), "world.jsonl", "");
        write(
            dir.path(),
            "a.jsonl",
            "[\"Include\",{\"path\":\"b.jsonl\"}]\n",
        );
        write(
            dir.path(),
            "b.jsonl",
            "[\"Include\",{\"path\":\"a.jsonl\"}]\n",
        );
        let err = resolve(vec![include("a.jsonl")], dir.path(), &world).unwrap_err();
        assert!(err.starts_with("Include cycle: "), "{err}");
        assert!(
            err.contains("a.jsonl -> ") && err.contains("b.jsonl -> "),
            "{err}"
        );
    }

    #[test]
    fn a_world_including_itself_is_a_cycle() {
        let dir = tempfile::tempdir().unwrap();
        let world = write(
            dir.path(),
            "world.jsonl",
            "[\"Include\",{\"path\":\"world.jsonl\"}]\n",
        );
        let err = resolve(vec![include("world.jsonl")], dir.path(), &world).unwrap_err();
        assert!(err.starts_with("Include cycle: "), "{err}");
    }

    // Including the same file twice is not a cycle: the stack only holds the
    // files currently being read.
    #[test]
    fn a_file_included_twice_side_by_side_is_not_a_cycle() {
        let dir = tempfile::tempdir().unwrap();
        let world = write(dir.path(), "world.jsonl", "");
        write(dir.path(), "chunk.jsonl", "[\"Scene\"]\n");
        let out = resolve(
            vec![include("chunk.jsonl"), include("chunk.jsonl")],
            dir.path(),
            &world,
        )
        .unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn a_missing_file_names_the_path_and_the_including_file() {
        let dir = tempfile::tempdir().unwrap();
        let world = write(dir.path(), "world.jsonl", "");
        let err = resolve(vec![include("gone.jsonl")], dir.path(), &world).unwrap_err();
        assert!(err.contains("gone.jsonl"), "{err}");
        assert!(err.contains("world.jsonl"), "{err}");
    }

    #[test]
    fn an_included_file_is_tuple_format_and_its_errors_name_it() {
        let dir = tempfile::tempdir().unwrap();
        let world = write(dir.path(), "world.jsonl", "");
        write(
            dir.path(),
            "old.jsonl",
            "[\"Scene\"]\n{\"type\":\"Window\"}\n",
        );
        let err = resolve(vec![include("old.jsonl")], dir.path(), &world).unwrap_err();
        assert!(err.contains("old.jsonl: line 2: "), "{err}");
        assert!(err.contains("not an object"), "{err}");
    }

    #[test]
    fn malformed_include_args_are_errors() {
        let bad = [
            (
                json!({"type": "Include", "args": {}}),
                "missing field `path`",
            ),
            (json!({"type": "Include", "args": {"path": 3}}), "`path`"),
            (
                json!({"type": "Include", "args": {"path": ""}}),
                "`path` is empty",
            ),
            (
                json!({"type": "Include", "args": {"$id": "i", "path": "a.jsonl"}}),
                "takes no `$id`",
            ),
        ];
        for (entry, needle) in bad {
            let err = resolve_includes(vec![entry], None, None).unwrap_err();
            assert!(err.starts_with("Include") && err.contains(needle), "{err}");
        }
    }

    // The editor's view keeps each line ahead of what it inlined, and tags
    // every entry with the file holding it.
    #[test]
    fn with_includes_keeps_the_lines_and_tags_each_entry() {
        let dir = tempfile::tempdir().unwrap();
        let world = write(dir.path(), "world.jsonl", "");
        let outer = write(
            dir.path(),
            "outer.jsonl",
            "[\"Window\",{\"$id\":\"o\"}]\n[\"Include\",{\"path\":\"inner.jsonl\"}]\n",
        );
        let inner = write(dir.path(), "inner.jsonl", "[\"Window\",{\"$id\":\"i\"}]\n");
        let out = with_includes(
            vec![window("a"), include("outer.jsonl"), window("z")],
            Some(&world),
            Some(dir.path()),
        )
        .unwrap();
        let shape: Vec<(bool, Option<PathBuf>)> = out
            .iter()
            .map(|s| (is_include(&s.entry), s.file.clone()))
            .collect();
        assert_eq!(
            shape,
            [
                (false, None),
                (true, None),
                (false, Some(outer.clone())),
                (true, Some(outer)),
                (false, Some(inner)),
                (false, None),
            ]
        );
    }

    // A project holding a library at `assets/`, marked as its own root.
    fn project_with_library() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let library = dir.path().join("assets");
        std::fs::create_dir_all(library.join(source::LIBRARY_MARKER)).unwrap();
        write(
            &library,
            "kits/look.jsonl",
            "[\"Include\",{\"path\":\"kits/shader.jsonl\"}]\n[\"Window\",{\"$id\":\"look\"}]\n",
        );
        write(
            &library,
            "kits/shader.jsonl",
            "[\"Shader\",{\"$id\":\"toon\",\"fragment\":\"shaders/toon.hlsl\"}]\n",
        );
        (dir, library)
    }

    fn fragment(entries: &[Value]) -> &str {
        entries
            .iter()
            .find_map(|e| e["args"]["fragment"].as_str())
            .unwrap()
    }

    // A library included from a project reads its own include lines from its
    // root, and the build sees its file paths from the project's.
    #[test]
    fn a_library_names_its_files_from_its_own_root() {
        let (dir, _) = project_with_library();
        let world = write(dir.path(), "world.jsonl", "");
        let out = resolve(vec![include("assets/kits/look.jsonl")], dir.path(), &world).unwrap();
        assert_eq!(ids(&out), ["toon", "look"]);
        assert_eq!(fragment(&out), "assets/shaders/toon.hlsl");
    }

    // A world inside the library is read from the library's root, whatever root
    // the build is given.
    #[test]
    fn a_world_inside_a_library_reads_from_the_library_root() {
        let (dir, library) = project_with_library();
        let world = write(&library, "worlds/gallery.jsonl", "");
        let entries = vec![
            include("kits/shader.jsonl"),
            json!({"type": "Shader", "args": {"$id": "own", "fragment": "shaders/own.hlsl"}}),
        ];
        let from_project = resolve(entries.clone(), dir.path(), &world).unwrap();
        assert_eq!(fragment(&from_project), "assets/shaders/toon.hlsl");
        assert_eq!(
            from_project[1]["args"]["fragment"],
            "assets/shaders/own.hlsl"
        );
        let from_library = resolve(entries, &library, &world).unwrap();
        assert_eq!(fragment(&from_library), "shaders/toon.hlsl");
    }

    // The authored view keeps every path as its file spells it.
    #[test]
    fn with_includes_does_not_rebase() {
        let (dir, _) = project_with_library();
        let world = write(dir.path(), "world.jsonl", "");
        let out = with_includes(
            vec![include("assets/kits/look.jsonl")],
            Some(&world),
            Some(dir.path()),
        )
        .unwrap();
        let entries: Vec<Value> = out.into_iter().map(|s| s.entry).collect();
        assert_eq!(fragment(&entries), "shaders/toon.hlsl");
    }

    // Each entry carries the root its paths resolve against: an unmarked file
    // takes the root of the file that included it.
    #[test]
    fn with_includes_records_each_entrys_root() {
        let (dir, library) = project_with_library();
        write(
            &library,
            "kits/plain.jsonl",
            "[\"Window\",{\"$id\":\"plain\"}]\n",
        );
        write(
            dir.path(),
            "shared/loose.jsonl",
            "[\"Window\",{\"$id\":\"loose\"}]\n",
        );
        write(
            &library,
            "kits/reach.jsonl",
            "[\"Include\",{\"path\":\"../shared/loose.jsonl\"}]\n",
        );
        let world = write(dir.path(), "world.jsonl", "");
        let out = with_includes(
            vec![
                window("own"),
                include("assets/kits/plain.jsonl"),
                include("assets/kits/reach.jsonl"),
            ],
            Some(&world),
            Some(dir.path()),
        )
        .unwrap();
        let root_of = |id: &str| {
            out.iter()
                .find(|s| s.entry["args"]["$id"] == id)
                .and_then(|s| s.root.as_deref())
                .map(source::normalized)
        };
        let library = source::normalized(&library);
        assert_eq!(root_of("own"), Some(source::normalized(dir.path())));
        assert_eq!(root_of("plain"), Some(library.clone()));
        assert_eq!(
            root_of("loose"),
            Some(library),
            "an unmarked file takes its includer's root"
        );
    }

    #[test]
    fn rebase_files_rewrites_only_the_file_fields() {
        let mut entry = json!({"type": "Shader", "args": {"$id": "s", "fragment": "a.hlsl", "vertex": "/abs/v.hlsl"}});
        rebase_files(&mut entry, Path::new("/p/lib"), Path::new("/p"));
        assert_eq!(entry["args"]["fragment"], "lib/a.hlsl");
        assert_eq!(entry["args"]["vertex"], "/abs/v.hlsl");
        assert_eq!(entry["args"]["$id"], "s");
    }
}
