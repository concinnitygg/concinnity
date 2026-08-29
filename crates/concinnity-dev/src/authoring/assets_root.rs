// The asset search root every dev-session build and reload resolves bare
// source filenames against.
//
// The cook takes that root as a parameter rather than reading one for itself,
// so a host names it. This host is the dev session: it builds and reloads the
// project the CLI anchored at startup, so the root is that project's
// `assets/`. A session with no state root installed has no tree to search and
// resolves nothing, which is what `None` means downstream.

use std::path::PathBuf;

pub(crate) fn assets_dir() -> Option<PathBuf> {
    concinnity_cook::paths::assets_dir()
}
