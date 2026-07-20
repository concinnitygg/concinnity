// Persistence for the declarative logic state: the shared variables plus
// which `once` reactions have fired, written by the `save` action and
// restored at world start. One file per install, beside the story saves.
// Variables are keyed by their authored names, stable across world edits.
// Fired flags are keyed by (asset id, content hash) so a save from an edited
// world degrades safely: a rule whose id or content changed just loses its
// flag (and may fire once more), never inherits another rule's.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use crate::assets::Reaction;

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub(super) struct LogicSave {
    #[serde(default)]
    pub(super) vars: BTreeMap<String, i32>,
    // (reaction asset id, content hash) of every fired `once` rule.
    #[serde(default)]
    pub(super) fired: Vec<(u32, u64)>,
}

pub(super) fn state_file(dir: &Path) -> PathBuf {
    dir.join("state")
}

// Content hash of a rule definition (asset identity excluded via its serde
// skip), so a loaded fired flag applies only to the rule it was saved for.
pub(super) fn def_hash(def: &Reaction) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    postcard::to_allocvec(def)
        .unwrap_or_default()
        .hash(&mut hasher);
    hasher.finish()
}

pub(super) fn read_save(path: &Path) -> Option<LogicSave> {
    let bytes = std::fs::read(path).ok()?;
    match ciborium::from_reader(&bytes[..]) {
        Ok(save) => Some(save),
        Err(e) => {
            tracing::warn!("ReactionSystem: saved state unreadable, starting fresh: {e}");
            None
        }
    }
}

pub(super) fn write_save(path: &Path, save: &LogicSave) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut bytes = Vec::new();
    ciborium::into_writer(save, &mut bytes).map_err(std::io::Error::other)?;
    std::fs::write(path, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_round_trips_through_cbor() {
        let dir = std::env::temp_dir().join(format!("cn-logic-save-{}", std::process::id()));
        let path = state_file(&dir);
        let mut save = LogicSave::default();
        save.vars.insert("score".into(), 12);
        save.fired.push((3, 0xfeed));
        write_save(&path, &save).unwrap();

        let back = read_save(&path).expect("state readable");
        assert_eq!(back.vars.get("score"), Some(&12));
        assert_eq!(back.fired, vec![(3, 0xfeed)]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn def_hash_tracks_content_not_identity() {
        use crate::assets::{Reaction, ReactionSource};
        use crate::ecs::asset_id::AssetId;

        let a = Reaction {
            asset_id: AssetId(1),
            on: ReactionSource::Start,
            ..Default::default()
        };
        let same_content = Reaction {
            asset_id: AssetId(9),
            ..a.clone()
        };
        assert_eq!(def_hash(&a), def_hash(&same_content));

        let edited = Reaction {
            on: ReactionSource::Variable("v".into()),
            ..a.clone()
        };
        assert_ne!(def_hash(&a), def_hash(&edited));
    }
}
