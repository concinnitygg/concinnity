//! How the editor addresses one asset of the world it is editing: the currency
//! of the selection, the open form, the gizmo, and every panel that follows the
//! active member.
//!
//! Neither of the obvious candidates works. A live-preview rebuild resets the
//! name interner and re-interns in declaration order, so a stored `AssetId`
//! drifts onto whatever asset now sits at that index. A name is authored
//! content: the user renames it, and an entry need not declare one.
//!
//! So an authored entry is addressed by the session key its list minted for it,
//! and an asset the build generated -- which has no authored line -- by the
//! identity the expansion gave it. Both are functions of the entry list, so
//! both re-resolve after a rebuild instead of drifting.

use crate::editor::entry_list::EntryId;

// One addressable asset of the edited world.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum AssetHandle {
    // An authored world.jsonl entry, by its session key.
    Entry(EntryId),
    // An asset a build-time expansion produced (a `SceneImport`'s props, a
    // `MainMenu`'s buttons), by the identity that expansion gave it. The
    // expansion mints it from the entry list, so it survives a rebuild; it
    // stops resolving when the entry that generated it stops generating it,
    // which is the same moment the asset itself goes away.
    Generated(String),
}

impl AssetHandle {
    // The authored entry's key, or `None` for a generated asset.
    pub(crate) fn entry(&self) -> Option<EntryId> {
        match self {
            Self::Entry(key) => Some(*key),
            Self::Generated(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::entry_list::EntryList;
    use serde_json::json;

    fn one_key() -> EntryId {
        EntryList::new(vec![json!({"type": "Prop"})])
            .key_at(0)
            .unwrap()
    }

    // Only an authored handle names an entry to write back to; that is how the
    // gizmo, duplicate and delete skip a generated asset.
    #[test]
    fn only_an_authored_handle_reports_an_entry() {
        let key = one_key();
        assert_eq!(AssetHandle::Entry(key).entry(), Some(key));
        assert_eq!(AssetHandle::Generated("bistro_prop_7".into()).entry(), None);
    }

    #[test]
    fn an_entry_handle_never_equals_a_generated_one() {
        let key = one_key();
        assert_ne!(
            AssetHandle::Entry(key),
            AssetHandle::Generated(key.to_string()),
            "the two address spaces must not collide"
        );
    }
}
