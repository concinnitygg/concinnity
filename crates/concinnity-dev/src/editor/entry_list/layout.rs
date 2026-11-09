//! The include structure of an entry list: which `Include` line opens which
//! file, and where each file's run of entries ends. A list read from disk is
//! always well nested (every included file's entries sit right after the line
//! that names it, nested files inside them); an edit that moves an entry out of
//! its file's run would save that entry somewhere the build no longer reads it
//! from, so the walk reports the first entry out of place.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use concinnity_cook::build_only::include::is_include;

use super::paths::include_target;

// One `Include` line and the run of entries it brings in.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Opening {
    // The `Include` line's position.
    pub line: usize,
    // The file it names, spelled as the build tags that file's entries.
    pub target: PathBuf,
    // One past the last entry of its run, nested runs included.
    pub end: usize,
}

#[derive(Debug, Default)]
pub(super) struct Layout {
    pub openings: Vec<Opening>,
    // The first entry sitting outside the run of the file it belongs to.
    pub misplaced: Option<usize>,
}

impl Layout {
    // Walk `values` (tagged by `files`, `None` for the world file at `world`)
    // as the build reads them.
    pub(super) fn of(
        values: &[serde_json::Value],
        files: &[Option<Arc<Path>>],
        world: &Path,
    ) -> Self {
        let mut layout = Self::default();
        // The files being read, outermost first, each beside its opening.
        let mut stack: Vec<(Option<PathBuf>, Option<usize>)> = vec![(None, None)];
        for (i, (value, file)) in values.iter().zip(files).enumerate() {
            let file = file.as_deref();
            let Some(depth) = stack.iter().rposition(|(f, _)| f.as_deref() == file) else {
                layout.misplaced = Some(i);
                break;
            };
            for (_, opening) in stack.drain(depth + 1..) {
                if let Some(o) = opening {
                    layout.openings[o].end = i;
                }
            }
            let Some(path) = include_path(value) else {
                continue;
            };
            let target = include_target(file.unwrap_or(world), path);
            stack.push((Some(target.clone()), Some(layout.openings.len())));
            layout.openings.push(Opening {
                line: i,
                target,
                end: values.len(),
            });
        }
        layout
    }

    // How many lines open `target`.
    pub(super) fn times_opened(&self, target: &Path) -> usize {
        self.openings.iter().filter(|o| o.target == target).count()
    }
}

// The `path` an `Include` line names.
fn include_path(value: &serde_json::Value) -> Option<&str> {
    if !is_include(value) {
        return None;
    }
    value.get("args")?.get("path")?.as_str()
}
