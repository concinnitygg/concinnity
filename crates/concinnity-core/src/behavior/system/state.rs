// Persisted behavior state: the world variables plus which `once` behaviors
// have fired, written by the `save` node and restored at world start.
// Variables are keyed by their authored names, stable across world edits and
// across a re-cook that reassigns slots. Fired flags are keyed by (asset id,
// content hash) so a save from an edited world degrades safely: a behavior
// whose id or content changed just loses its flag (and may fire once more),
// never inherits another's.
//
// Per-entity locals are never saved: a spawned entity has no identity that
// survives a re-cook, so there is nothing stable to key them by.
//
// Where the state is kept is the host's: a file, a preferences blob, nothing at
// all. This module owns the shape and the keying; a `BehaviorStore` owns the
// medium.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::components::{Behavior, BehaviorLiteral};

/// The behavior state a world carries between runs.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct BehaviorState {
    /// World variables by authored name.
    #[serde(default)]
    pub vars: BTreeMap<String, BehaviorLiteral>,
    /// `(asset id, content hash)` of every `once` behavior that has fired.
    #[serde(default)]
    pub fired: Vec<(u32, u64)>,
}

/// Where a host keeps persisted behavior state.
///
/// Read once when the world starts and written after any tick a `save` node ran
/// in. A world whose host installs no store runs its behaviors and persists
/// nothing.
pub trait BehaviorStore: core::fmt::Debug + Send {
    /// The stored state, or `None` when nothing was stored or it could not be
    /// read.
    fn read(&self) -> Option<BehaviorState>;

    /// Store `state`, replacing whatever was there. Only the implementor knows
    /// what the write was to, so reporting a failure is its job.
    fn write(&self, state: &BehaviorState);
}

/// Content hash of a behavior definition. Asset identity is excluded (its serde
/// skip), so a restored fired flag applies to the behavior it was saved for and
/// to no other.
pub fn def_hash(def: &Behavior) -> u64 {
    let bytes = postcard::to_allocvec(def).unwrap_or_default();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::BehaviorSource;
    use crate::ecs::asset_id::AssetId;
    use alloc::string::ToString;

    #[test]
    fn def_hash_tracks_content_not_identity() {
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
            on: BehaviorSource::Variable("v".to_string()),
            ..a.clone()
        };
        assert_ne!(def_hash(&a), def_hash(&edited));
    }
}
