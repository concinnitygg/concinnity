//! The viewport selection: an ordered set of `AssetHandle`s. The last member is
//! the "active" one: the edit form follows it and its highlight ring is drawn
//! brighter.
//!
//! Members are re-resolved to the live world every frame (`hook/handles.rs`),
//! so a member the current build does not produce simply goes undrawn instead
//! of dropping out of the set.

use std::collections::BTreeSet;

use crate::editor::asset_handle::AssetHandle;

#[derive(Default)]
pub(crate) struct Selection {
    members: Vec<AssetHandle>,
}

impl Selection {
    pub(crate) fn contains(&self, handle: &AssetHandle) -> bool {
        self.members.contains(handle)
    }

    // The active member: the most recently added one.
    pub(crate) fn active(&self) -> Option<&AssetHandle> {
        self.members.last()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &AssetHandle> {
        self.members.iter()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.members.clear();
    }

    // Plain click: the selection becomes exactly this member.
    pub(crate) fn replace(&mut self, handle: AssetHandle) {
        self.members.clear();
        self.members.push(handle);
    }

    // Shift-click: remove a present member, else add it (making it active).
    // Returns whether the handle is selected afterwards.
    pub(crate) fn toggle(&mut self, handle: AssetHandle) -> bool {
        if let Some(i) = self.members.iter().position(|m| *m == handle) {
            self.members.remove(i);
            false
        } else {
            self.members.push(handle);
            true
        }
    }

    // Marquee release: the selection becomes `handles` (first duplicate wins).
    pub(crate) fn set(&mut self, handles: Vec<AssetHandle>) {
        self.members.clear();
        self.extend(handles);
    }

    // Shift-marquee release: append the members not already selected, keeping
    // the existing order (and the existing active member if nothing is new).
    pub(crate) fn extend(&mut self, handles: Vec<AssetHandle>) {
        for handle in handles {
            if !self.contains(&handle) {
                self.members.push(handle);
            }
        }
    }
}

/// The selection resolved to the names the world knows its members by this
/// frame, for the surfaces that draw rows and icons by name. Identity stays in
/// `AssetHandle`; this is only the label a tint is looked up by, so a member
/// the current build does not produce is simply absent.
#[derive(Debug, Default)]
pub(crate) struct SelectedNames {
    names: BTreeSet<String>,
    active: Option<String>,
}

impl SelectedNames {
    pub(crate) fn new(names: BTreeSet<String>, active: Option<String>) -> Self {
        Self { names, active }
    }

    pub(crate) fn contains(&self, name: &str) -> bool {
        self.names.contains(name)
    }

    pub(crate) fn is_active(&self, name: &str) -> bool {
        self.active.as_deref() == Some(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::entry_list::{EntryId, EntryList};

    // Distinct session keys, standing in for authored entries.
    fn keys(n: usize) -> Vec<EntryId> {
        let list = EntryList::new((0..n).map(|_| serde_json::json!({})).collect());
        (0..n).map(|i| list.key_at(i).unwrap()).collect()
    }

    fn sel(handles: &[AssetHandle]) -> Selection {
        let mut s = Selection::default();
        s.extend(handles.to_vec());
        s
    }

    fn generated(name: &str) -> AssetHandle {
        AssetHandle::Generated(name.to_string())
    }

    #[test]
    fn replace_makes_a_single_member_selection() {
        let k = keys(3);
        let mut s = sel(&[AssetHandle::Entry(k[0]), AssetHandle::Entry(k[1])]);
        s.replace(AssetHandle::Entry(k[2]));
        assert_eq!(s.iter().count(), 1);
        assert_eq!(s.active(), Some(&AssetHandle::Entry(k[2])));
    }

    #[test]
    fn toggle_adds_then_removes_and_moves_active() {
        let k = keys(2);
        let mut s = Selection::default();
        assert!(s.toggle(AssetHandle::Entry(k[0])));
        assert!(s.toggle(AssetHandle::Entry(k[1])));
        assert_eq!(
            s.active(),
            Some(&AssetHandle::Entry(k[1])),
            "the newest member is active"
        );

        assert!(
            !s.toggle(AssetHandle::Entry(k[0])),
            "a second toggle removes"
        );
        assert_eq!(s.iter().collect::<Vec<_>>(), [&AssetHandle::Entry(k[1])]);
        assert!(!s.toggle(AssetHandle::Entry(k[1])));
        assert_eq!(s.active(), None, "an emptied selection has no active");
        assert!(s.is_empty());
    }

    #[test]
    fn set_replaces_and_dedups_preserving_first_occurrence() {
        let k = keys(3);
        let mut s = sel(&[AssetHandle::Entry(k[2])]);
        s.set(vec![
            AssetHandle::Entry(k[0]),
            AssetHandle::Entry(k[1]),
            AssetHandle::Entry(k[0]),
        ]);
        assert_eq!(
            s.iter().collect::<Vec<_>>(),
            [&AssetHandle::Entry(k[0]), &AssetHandle::Entry(k[1])]
        );
        assert!(!s.contains(&AssetHandle::Entry(k[2])));
    }

    #[test]
    fn extend_appends_only_new_members() {
        let k = keys(2);
        let mut s = sel(&[AssetHandle::Entry(k[0]), generated("imported_a")]);
        s.extend(vec![generated("imported_a"), AssetHandle::Entry(k[1])]);
        assert_eq!(
            s.iter().collect::<Vec<_>>(),
            [
                &AssetHandle::Entry(k[0]),
                &generated("imported_a"),
                &AssetHandle::Entry(k[1])
            ]
        );
        assert_eq!(s.active(), Some(&AssetHandle::Entry(k[1])));
    }

    #[test]
    fn an_authored_entry_and_a_generated_asset_are_separate_members() {
        let k = keys(1);
        let mut s = Selection::default();
        s.toggle(AssetHandle::Entry(k[0]));
        s.toggle(generated("scene_prop_3"));
        assert_eq!(s.iter().count(), 2);
        // Toggling one off leaves the other alone.
        s.toggle(generated("scene_prop_3"));
        assert_eq!(s.iter().collect::<Vec<_>>(), [&AssetHandle::Entry(k[0])]);
    }

    #[test]
    fn selected_names_answers_membership_and_the_active_label() {
        let names = SelectedNames::new(
            ["lamp".to_string(), "floor".to_string()]
                .into_iter()
                .collect(),
            Some("lamp".to_string()),
        );
        assert!(names.contains("lamp") && names.contains("floor"));
        assert!(!names.contains("wall"));
        assert!(names.is_active("lamp"));
        assert!(!names.is_active("floor"), "only one member is active");
    }
}
