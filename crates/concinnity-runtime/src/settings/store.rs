//! The settings store on disk: core's `StoredSettings` at the state root's
//! `settings`, the mutable sibling of the build-regenerated `data`. It is stored
//! as CBOR, binary like the data blobs but self-describing; bincode, which the
//! data blobs use, is positional and safe only for files regenerated each build.

use concinnity_core::settings::StoredSettings;
use concinnity_host::store::paths::StateTree;
use std::path::Path;

// Load from `tree`'s `settings` file. Returns defaults when nothing is stored or
// the file is unreadable.
pub(crate) fn load(tree: Option<&StateTree>) -> StoredSettings {
    tree.map(|tree| load_from(&tree.settings_path()))
        .unwrap_or_default()
}

// Persist to `tree`'s `settings` file, creating the directory as needed. An app
// with no state tree has nowhere to persist to, which is not an error: the
// choices apply for the rest of the run.
pub(crate) fn save(settings: &StoredSettings, tree: Option<&StateTree>) -> std::io::Result<()> {
    match tree {
        Some(tree) => concinnity_host::store::cbor::write(&tree.settings_path(), settings),
        None => Ok(()),
    }
}

// Read settings from `path`. Split from `load` so the read path can be tested
// against a sandbox file, never the developer's real one. No file yet, or a
// truncated / incompatible one, starts from defaults.
fn load_from(path: &Path) -> StoredSettings {
    concinnity_host::store::cbor::read(path, "settings store").unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::UpscaleQuality;
    use concinnity_core::settings::UserGraphicsSettings;

    // The on-disk path must stay sandboxable so a test can never clobber the
    // developer's real `settings` file. A non-default `render_scale` mirrors a
    // real persisted choice and proves a populated field survives the trip.
    #[test]
    fn settings_save_load_roundtrip_is_sandboxed() {
        let s = StoredSettings {
            graphics: UserGraphicsSettings {
                render_scale: Some(UpscaleQuality::Performance),
                vsync: Some(true),
                exposure_ev: Some(-1.5),
                ..Default::default()
            },
            ..Default::default()
        };

        let dir = concinnity_testing::TempTree::new();
        let tree = StateTree::at(dir.path());

        save(&s, Some(&tree)).unwrap();
        assert!(tree.settings_path().exists());
        assert_eq!(load_from(&tree.settings_path()), s);
    }

    // Settings load and save against the tree they are handed, and an app with
    // no tree loads defaults and persists nothing rather than guessing a file.
    #[test]
    fn settings_follow_the_tree_they_are_given_and_no_tree_writes_nothing() {
        let dir = concinnity_testing::TempTree::new();
        let tree = StateTree::at(dir.path());

        let stored = StoredSettings {
            graphics: UserGraphicsSettings {
                vsync: Some(true),
                ..Default::default()
            },
            ..Default::default()
        };
        save(&stored, Some(&tree)).unwrap();
        assert!(tree.settings_path().exists());
        assert_eq!(load(Some(&tree)), stored);

        assert_eq!(load(None), StoredSettings::default());
        assert!(save(&stored, None).is_ok());
    }
}
