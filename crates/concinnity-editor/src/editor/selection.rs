// src/editor/selection.rs
//
// The viewport selection: an ordered set of asset names. Names, not AssetIds,
// because every live-preview rebuild resets the interner and re-interns, so a
// stored id could silently drift to a different asset; members are re-resolved
// by name each frame. The last member is the "active" one: the edit form
// follows it and its highlight ring is drawn brighter.

#[derive(Default)]
pub(crate) struct Selection {
    names: Vec<String>,
}

impl Selection {
    pub(crate) fn contains(&self, name: &str) -> bool {
        self.names.iter().any(|n| n == name)
    }

    // The active member: the most recently added one.
    pub(crate) fn active(&self) -> Option<&str> {
        self.names.last().map(String::as_str)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &str> {
        self.names.iter().map(String::as_str)
    }

    pub(crate) fn clear(&mut self) {
        self.names.clear();
    }

    // Plain click: the selection becomes exactly this member.
    pub(crate) fn replace(&mut self, name: String) {
        self.names.clear();
        self.names.push(name);
    }

    // Shift-click: remove a present member, else add it (making it active).
    // Returns whether the name is selected afterwards.
    pub(crate) fn toggle(&mut self, name: String) -> bool {
        if let Some(i) = self.names.iter().position(|n| *n == name) {
            self.names.remove(i);
            false
        } else {
            self.names.push(name);
            true
        }
    }

    // Marquee release: the selection becomes `names` (first duplicate wins).
    pub(crate) fn set(&mut self, names: Vec<String>) {
        self.names.clear();
        self.extend(names);
    }

    // Shift-marquee release: append the members not already selected, keeping
    // the existing order (and the existing active member if nothing is new).
    pub(crate) fn extend(&mut self, names: Vec<String>) {
        for name in names {
            if !self.contains(&name) {
                self.names.push(name);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sel(names: &[&str]) -> Selection {
        let mut s = Selection::default();
        s.extend(names.iter().map(|n| n.to_string()).collect());
        s
    }

    #[test]
    fn replace_makes_a_single_member_selection() {
        let mut s = sel(&["a", "b"]);
        s.replace("c".into());
        assert_eq!(s.iter().collect::<Vec<_>>(), ["c"]);
        assert_eq!(s.active(), Some("c"));
    }

    #[test]
    fn toggle_adds_then_removes_and_moves_active() {
        let mut s = Selection::default();
        assert!(s.toggle("a".into()));
        assert!(s.toggle("b".into()));
        assert_eq!(s.active(), Some("b"), "the newest member is active");

        assert!(!s.toggle("a".into()), "a second toggle removes");
        assert_eq!(s.iter().collect::<Vec<_>>(), ["b"]);
        assert!(!s.toggle("b".into()));
        assert_eq!(s.active(), None, "an emptied selection has no active");
    }

    #[test]
    fn set_replaces_and_dedups_preserving_first_occurrence() {
        let mut s = sel(&["x"]);
        s.set(vec!["a".into(), "b".into(), "a".into()]);
        assert_eq!(s.iter().collect::<Vec<_>>(), ["a", "b"]);
        assert!(!s.contains("x"));
    }

    #[test]
    fn extend_appends_only_new_members() {
        let mut s = sel(&["a", "b"]);
        s.extend(vec!["b".into(), "c".into()]);
        assert_eq!(s.iter().collect::<Vec<_>>(), ["a", "b", "c"]);
        assert_eq!(s.active(), Some("c"));
    }
}
