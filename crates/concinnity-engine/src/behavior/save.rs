// Persistence for behavior state: the world variables plus which `once`
// behaviors have fired, written by the `save` node and restored at world start.
// Variables are keyed by their authored names, stable across world edits and
// across a re-cook that reassigns slots. Fired flags are keyed by (asset id,
// content hash) so a save from an edited world degrades safely: a behavior
// whose id or content changed just loses its flag (and may fire once more),
// never inherits another's.
//
// Per-entity locals are never saved: a spawned entity has no identity that
// survives a re-cook, so there is nothing stable to key them by.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use crate::assets::{Behavior, Literal};

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub(super) struct BehaviorSave {
    #[serde(default)]
    pub(super) vars: BTreeMap<String, Literal>,
    #[serde(default)]
    pub(super) fired: Vec<(u32, u64)>,
}

pub(super) fn state_file(dir: &Path) -> PathBuf {
    dir.join("state")
}

// Content hash of a behavior definition (asset identity excluded via its serde
// skip), so a loaded fired flag applies only to the behavior it was saved for.
pub(super) fn def_hash(def: &Behavior) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    postcard::to_allocvec(def)
        .unwrap_or_default()
        .hash(&mut hasher);
    hasher.finish()
}

pub(super) fn read_save(path: &Path) -> Option<BehaviorSave> {
    let bytes = std::fs::read(path).ok()?;
    match ciborium::from_reader(&bytes[..]) {
        Ok(save) => Some(save),
        Err(e) => {
            tracing::warn!("BehaviorSystem: saved state unreadable, starting fresh: {e}");
            None
        }
    }
}

pub(super) fn write_save(path: &Path, save: &BehaviorSave) -> std::io::Result<()> {
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
    fn def_hash_tracks_content_not_identity() {
        use crate::assets::BehaviorSource;
        use crate::ecs::asset_id::AssetId;

        let a = Behavior {
            asset_id: AssetId(1),
            on: BehaviorSource::Tick,
            ..Default::default()
        };
        let same_content = Behavior {
            asset_id: AssetId(9),
            ..a.clone()
        };
        assert_eq!(def_hash(&a), def_hash(&same_content));

        let edited = Behavior {
            on: BehaviorSource::Variable("v".into()),
            ..a.clone()
        };
        assert_ne!(def_hash(&a), def_hash(&edited));
    }

    #[test]
    fn save_round_trips_through_cbor() {
        let dir = std::env::temp_dir().join(format!("cn-behavior-save-{}", std::process::id()));
        let path = state_file(&dir);
        let mut save = BehaviorSave::default();
        save.vars.insert("score".into(), Literal::Int(12));
        save.fired.push((3, 0xfeed));
        write_save(&path, &save).unwrap();

        let back = read_save(&path).expect("state readable");
        assert_eq!(back.vars.get("score"), Some(&Literal::Int(12)));
        assert_eq!(back.fired, vec![(3, 0xfeed)]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
