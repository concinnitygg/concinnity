//! Resolving `Include` lines: each is replaced by the entries of the file it
//! names, recursively, with every relative path taken from the file the line is
//! in.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::schema::Include;
use crate::authoring::registry::RegisteredType;
use crate::authoring::world::{ID_KEY, parse_world_jsonl};

/// One entry of a world whose includes are resolved, with the file it was
/// read from.
#[derive(Debug, Clone, PartialEq)]
pub struct SourcedEntry {
    /// The entry, as a `{"type", "args"}` object.
    pub entry: Value,
    /// The included file the entry was read from; `None` for an entry of the
    /// world file itself.
    pub file: Option<PathBuf>,
}

/// Whether `entry` is an `Include` line.
pub fn is_include(entry: &Value) -> bool {
    entry.get("type").and_then(Value::as_str) == Some(RegisteredType::Include.as_str())
}

/// Replace every `Include` in `entries` with the entries of the file it names,
/// recursively: the world the build sees.
///
/// `world_file` is the file `entries` were read from, which a relative path
/// resolves against. A world held only in memory has none, so only an absolute
/// path resolves in it.
pub fn resolve_includes(
    entries: Vec<Value>,
    world_file: Option<&Path>,
) -> Result<Vec<Value>, String> {
    let sourced = Walk::new(world_file, false).run(entries, world_file)?;
    Ok(sourced.into_iter().map(|s| s.entry).collect())
}

/// `entries` with every include resolved as [`resolve_includes`] does, but
/// each `Include` line kept in place ahead of the entries it inlines, and
/// every entry tagged with the file it came from.
///
/// An `Include` line only counts toward the labels of its own type, so the
/// `<Type>#<n>` labels over this list are the ones the build gives.
pub fn with_includes(
    entries: Vec<Value>,
    world_file: Option<&Path>,
) -> Result<Vec<SourcedEntry>, String> {
    Walk::new(world_file, true).run(entries, world_file)
}

struct Walk {
    keep_lines: bool,
    // The files being read, outermost first, as canonical paths beside the
    // spelling an error shows.
    stack: Vec<(PathBuf, PathBuf)>,
}

impl Walk {
    fn new(world_file: Option<&Path>, keep_lines: bool) -> Self {
        let stack = world_file
            .and_then(|f| Some((std::fs::canonicalize(f).ok()?, f.to_path_buf())))
            .into_iter()
            .collect();
        Self { keep_lines, stack }
    }

    fn run(
        mut self,
        entries: Vec<Value>,
        world_file: Option<&Path>,
    ) -> Result<Vec<SourcedEntry>, String> {
        let mut out = Vec::with_capacity(entries.len());
        self.walk(entries, world_file, None, &mut out)?;
        Ok(out)
    }

    // `file` is where `entries` were read from; `included` is the same file
    // when it was reached through an `Include`, which is what the entries are
    // tagged with.
    fn walk(
        &mut self,
        entries: Vec<Value>,
        file: Option<&Path>,
        included: Option<&Path>,
        out: &mut Vec<SourcedEntry>,
    ) -> Result<(), String> {
        let tag = || included.map(Path::to_path_buf);
        for entry in entries {
            if !is_include(&entry) {
                out.push(SourcedEntry { entry, file: tag() });
                continue;
            }
            let target = target_path(&entry, file)?;
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
            if self.keep_lines {
                out.push(SourcedEntry { entry, file: tag() });
            }
            self.stack.push((canonical, target.clone()));
            self.walk(nested, Some(&target), Some(&target), out)?;
            self.stack.pop();
        }
        Ok(())
    }
}

// The file an `Include` line names, joined onto the directory of the file the
// line is in.
fn target_path(entry: &Value, file: Option<&Path>) -> Result<PathBuf, String> {
    let at = |msg: String| format!("Include{}: {msg}", in_file(file));
    let mut args = entry.get("args").cloned().unwrap_or(Value::Null);
    if args.is_null() {
        args = Value::Object(Default::default());
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
    let path = Path::new(&include.path);
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    match file {
        Some(file) => Ok(file.parent().unwrap_or(Path::new("")).join(path)),
        None => Err(at(format!(
            "'{}' is relative, and a world held in memory has no file to resolve it against",
            include.path
        ))),
    }
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

    #[test]
    fn entries_without_includes_pass_through() {
        let out = resolve_includes(vec![window("a")], None).unwrap();
        assert_eq!(out, [window("a")]);
    }

    #[test]
    fn an_include_is_replaced_in_place_by_its_file() {
        let dir = tempfile::tempdir().unwrap();
        let root = write(dir.path(), "world.jsonl", "");
        write(
            dir.path(),
            "chunk.jsonl",
            "[\"Window\",{\"$id\":\"b\"}]\n[\"Window\",{\"$id\":\"c\"}]\n",
        );
        let out = resolve_includes(
            vec![window("a"), include("chunk.jsonl"), window("d")],
            Some(&root),
        )
        .unwrap();
        assert_eq!(ids(&out), ["a", "b", "c", "d"]);
    }

    // A path is taken from the file the line is in, at every depth, never from
    // the working directory.
    #[test]
    fn nested_includes_resolve_relative_to_their_own_file() {
        let dir = tempfile::tempdir().unwrap();
        let root = write(dir.path(), "worlds/world.jsonl", "");
        write(
            dir.path(),
            "worlds/parts/outer.jsonl",
            "[\"Window\",{\"$id\":\"outer\"}]\n[\"Include\",{\"path\":\"inner/leaf.jsonl\"}]\n",
        );
        write(
            dir.path(),
            "worlds/parts/inner/leaf.jsonl",
            "[\"Window\",{\"$id\":\"leaf\"}]\n",
        );
        let out = resolve_includes(vec![include("parts/outer.jsonl")], Some(&root)).unwrap();
        assert_eq!(ids(&out), ["outer", "leaf"]);
    }

    #[test]
    fn an_absolute_path_resolves_without_a_world_file() {
        let dir = tempfile::tempdir().unwrap();
        let chunk = write(dir.path(), "chunk.jsonl", "[\"Window\",{\"$id\":\"x\"}]\n");
        let out = resolve_includes(vec![include(chunk.to_str().unwrap())], None).unwrap();
        assert_eq!(ids(&out), ["x"]);
    }

    #[test]
    fn a_relative_path_needs_a_world_file() {
        let err = resolve_includes(vec![include("chunk.jsonl")], None).unwrap_err();
        assert!(err.contains("'chunk.jsonl' is relative"), "{err}");
    }

    #[test]
    fn a_cycle_is_an_error_naming_the_chain() {
        let dir = tempfile::tempdir().unwrap();
        let root = write(dir.path(), "world.jsonl", "");
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
        let err = resolve_includes(vec![include("a.jsonl")], Some(&root)).unwrap_err();
        assert!(err.starts_with("Include cycle: "), "{err}");
        assert!(
            err.contains("a.jsonl -> ") && err.contains("b.jsonl -> "),
            "{err}"
        );
    }

    #[test]
    fn a_world_including_itself_is_a_cycle() {
        let dir = tempfile::tempdir().unwrap();
        let root = write(
            dir.path(),
            "world.jsonl",
            "[\"Include\",{\"path\":\"world.jsonl\"}]\n",
        );
        let err = resolve_includes(vec![include("world.jsonl")], Some(&root)).unwrap_err();
        assert!(err.starts_with("Include cycle: "), "{err}");
    }

    // Including the same file twice is not a cycle: the stack only holds the
    // files currently being read.
    #[test]
    fn a_file_included_twice_side_by_side_is_not_a_cycle() {
        let dir = tempfile::tempdir().unwrap();
        let root = write(dir.path(), "world.jsonl", "");
        write(dir.path(), "chunk.jsonl", "[\"Scene\"]\n");
        let out = resolve_includes(
            vec![include("chunk.jsonl"), include("chunk.jsonl")],
            Some(&root),
        )
        .unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn a_missing_file_names_the_path_and_the_including_file() {
        let dir = tempfile::tempdir().unwrap();
        let root = write(dir.path(), "world.jsonl", "");
        let err = resolve_includes(vec![include("gone.jsonl")], Some(&root)).unwrap_err();
        assert!(err.contains("gone.jsonl"), "{err}");
        assert!(err.contains("world.jsonl"), "{err}");
    }

    #[test]
    fn an_included_file_is_tuple_format_and_its_errors_name_it() {
        let dir = tempfile::tempdir().unwrap();
        let root = write(dir.path(), "world.jsonl", "");
        write(
            dir.path(),
            "old.jsonl",
            "[\"Scene\"]\n{\"type\":\"Window\"}\n",
        );
        let err = resolve_includes(vec![include("old.jsonl")], Some(&root)).unwrap_err();
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
            let err = resolve_includes(vec![entry], None).unwrap_err();
            assert!(err.starts_with("Include") && err.contains(needle), "{err}");
        }
    }

    // The editor's view keeps each line ahead of what it inlined, and tags
    // every entry with the file holding it.
    #[test]
    fn with_includes_keeps_the_lines_and_tags_each_entry() {
        let dir = tempfile::tempdir().unwrap();
        let root = write(dir.path(), "world.jsonl", "");
        let outer = write(
            dir.path(),
            "outer.jsonl",
            "[\"Window\",{\"$id\":\"o\"}]\n[\"Include\",{\"path\":\"inner.jsonl\"}]\n",
        );
        let inner = write(dir.path(), "inner.jsonl", "[\"Window\",{\"$id\":\"i\"}]\n");
        let out = with_includes(
            vec![window("a"), include("outer.jsonl"), window("z")],
            Some(&root),
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
}
