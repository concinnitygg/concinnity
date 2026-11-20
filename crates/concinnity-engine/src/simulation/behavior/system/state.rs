// Persisted behavior state: world variables by authored name and fired `once`
// flags by (asset id, content hash), so an edited behavior loses its flag
// rather than inheriting another's. Per-entity locals are never saved. A
// `BehaviorStore` owns the medium.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use concinnity_core::fnv::Fnv1a;

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

/// Content hash of a behavior definition. A save pairs it with the behavior's
/// asset id, so a restored fired flag applies to the behavior it was saved for,
/// unedited, and to no other.
pub fn def_hash(def: &Behavior) -> u64 {
    Fnv1a::of(&postcard::to_allocvec(def).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::BehaviorSource;
    use alloc::string::ToString;

    #[test]
    fn def_hash_tracks_content() {
        let a = Behavior {
            on: BehaviorSource::Tick,
            ..Default::default()
        };
        assert_eq!(def_hash(&a), def_hash(&a.clone()));

        let edited = Behavior {
            on: BehaviorSource::Variable("v".to_string()),
            ..a.clone()
        };
        assert_ne!(def_hash(&a), def_hash(&edited));
    }
}
