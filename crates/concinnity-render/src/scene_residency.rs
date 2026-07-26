// src/scene_residency.rs
//
// Scene residency bookkeeping: which scenes are pinned (their streamed
// content wanted on the GPU), which of each scene's members are currently
// resident, and the derived per-scene load state and progress. Pure
// bookkeeping against `core` + `alloc` only, like the streaming policy core:
// the driver applies pin changes to the stream planners as blocked flags and
// reports residency transitions back here.

use crate::ecs::asset_id::AssetId;

// A streamed item a scene exclusively owns: the driver's channel tag (which
// pool the item lives in) plus the pool's item id.
pub type Member = (u8, u32);

pub const CHANNEL_TEXTURE: u8 = 0;
pub const CHANNEL_MESH: u8 = 1;
// A shader bucket's render pipeline, built on pin and released on unload. The
// item id is the bucket (the material's `ShaderHandle` value).
pub const CHANNEL_SHADER: u8 = 2;

// Load state of one scene's streamed content, derived from pins + residency.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SceneLoadState {
    // Unpinned, nothing resident.
    Unloaded,
    // Pinned, some members still loading.
    Loading,
    // Pinned, every member resident (trivially true with no members).
    Resident,
    // Unpinned, members still draining off the GPU.
    Unloading,
}

// Pin changes produced by one `sync_pins` call, as planner blocked-flag
// updates the driver must apply.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct PinChanges {
    // Members whose owning scene became unpinned: block them.
    pub blocked: Vec<Member>,
    // Members whose owning scene became pinned: unblock them.
    pub unblocked: Vec<Member>,
}

struct SceneSet {
    scene: AssetId,
    pinned: bool,
    members: Vec<Member>,
    // Parallel to `members`.
    resident: Vec<bool>,
}

pub struct SceneResidency {
    sets: Vec<SceneSet>,
    // (member, set index), sorted by member for binary search.
    owner: Vec<(Member, usize)>,
}

impl SceneResidency {
    // Build from each scene's exclusively-owned members. Every scene starts
    // unpinned with nothing resident; the driver blocks every owned member
    // ([`all_members`](Self::all_members)) at setup, then the first
    // `sync_pins` unblocks the pinned scenes' members.
    pub fn new(scenes: Vec<(AssetId, Vec<Member>)>) -> Self {
        let mut owner: Vec<(Member, usize)> = scenes
            .iter()
            .enumerate()
            .flat_map(|(set_idx, (_, members))| members.iter().map(move |&m| (m, set_idx)))
            .collect();
        owner.sort_unstable();
        let sets = scenes
            .into_iter()
            .map(|(scene, members)| SceneSet {
                scene,
                pinned: false,
                resident: vec![false; members.len()],
                members,
            })
            .collect();
        Self { sets, owner }
    }

    pub fn is_empty(&self) -> bool {
        self.sets.is_empty()
    }

    // Every scene-owned member, for the driver's setup pass: all owned members
    // start blocked, matching every scene starting unpinned.
    pub fn all_members(&self) -> impl Iterator<Item = Member> + '_ {
        self.owner.iter().map(|&(m, _)| m)
    }

    // Establish the pin set: exactly the scenes in `pinned` are pinned after
    // this call. Returns the members whose blocked state must change on the
    // stream planners (a no-op sync returns empty changes).
    pub fn sync_pins(&mut self, pinned: &[AssetId]) -> PinChanges {
        let mut changes = PinChanges::default();
        for set in &mut self.sets {
            let want = pinned.contains(&set.scene);
            if want == set.pinned {
                continue;
            }
            set.pinned = want;
            let out = if want {
                &mut changes.unblocked
            } else {
                &mut changes.blocked
            };
            out.extend_from_slice(&set.members);
        }
        changes
    }

    // Record a member's residency transition (a completed upload or an applied
    // eviction). Members owned by no scene are ignored.
    pub fn note_resident(&mut self, member: Member, resident: bool) {
        let Ok(pos) = self.owner.binary_search_by_key(&member, |&(m, _)| m) else {
            return;
        };
        let set = &mut self.sets[self.owner[pos].1];
        if let Some(i) = set.members.iter().position(|&m| m == member) {
            set.resident[i] = resident;
        }
    }

    // The scene's derived load state, or `None` for an unknown scene.
    pub fn state(&self, scene: AssetId) -> Option<SceneLoadState> {
        self.sets
            .iter()
            .find(|s| s.scene == scene)
            .map(derive_state)
    }

    // Fraction of the scene's members resident (1.0 with no members), or
    // `None` for an unknown scene.
    pub fn progress(&self, scene: AssetId) -> Option<f32> {
        self.sets
            .iter()
            .find(|s| s.scene == scene)
            .map(derive_progress)
    }

    // Whether any pinned scene still has members loading.
    pub fn any_loading(&self) -> bool {
        self.sets
            .iter()
            .any(|s| derive_state(s) == SceneLoadState::Loading)
    }

    // Every scene's `(id, state, progress)`, in declaration order.
    pub fn status(&self) -> Vec<(AssetId, SceneLoadState, f32)> {
        self.sets
            .iter()
            .map(|s| (s.scene, derive_state(s), derive_progress(s)))
            .collect()
    }
}

fn derive_state(set: &SceneSet) -> SceneLoadState {
    let all_resident = set.resident.iter().all(|&r| r);
    let none_resident = set.resident.iter().all(|&r| !r);
    match (set.pinned, all_resident, none_resident) {
        (true, true, _) => SceneLoadState::Resident,
        (true, false, _) => SceneLoadState::Loading,
        (false, _, true) => SceneLoadState::Unloaded,
        (false, _, false) => SceneLoadState::Unloading,
    }
}

fn derive_progress(set: &SceneSet) -> f32 {
    if set.members.is_empty() {
        return 1.0;
    }
    let resident = set.resident.iter().filter(|&&r| r).count();
    resident as f32 / set.members.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn residency() -> SceneResidency {
        SceneResidency::new(vec![
            (
                AssetId(1),
                vec![
                    (CHANNEL_TEXTURE, 0),
                    (CHANNEL_TEXTURE, 1),
                    (CHANNEL_MESH, 5),
                ],
            ),
            (AssetId(2), vec![(CHANNEL_TEXTURE, 2)]),
            (AssetId(3), vec![]),
        ])
    }

    #[test]
    fn scenes_start_unpinned_and_unloaded() {
        let r = residency();
        assert_eq!(r.state(AssetId(1)), Some(SceneLoadState::Unloaded));
        assert_eq!(r.state(AssetId(9)), None);
        assert_eq!(r.progress(AssetId(1)), Some(0.0));
    }

    #[test]
    fn first_sync_unblocks_only_the_pinned_scene() {
        // Setup blocks every owned member; the first sync then unblocks the
        // pinned scene's members and leaves the rest blocked (no changes).
        let mut r = residency();
        let all: Vec<Member> = r.all_members().collect();
        assert_eq!(all.len(), 4);
        let changes = r.sync_pins(&[AssetId(1)]);
        assert_eq!(
            changes.unblocked,
            vec![
                (CHANNEL_TEXTURE, 0),
                (CHANNEL_TEXTURE, 1),
                (CHANNEL_MESH, 5)
            ]
        );
        assert!(changes.blocked.is_empty());
    }

    #[test]
    fn repeated_sync_is_a_no_op() {
        let mut r = residency();
        r.sync_pins(&[AssetId(1)]);
        assert_eq!(r.sync_pins(&[AssetId(1)]), PinChanges::default());
    }

    #[test]
    fn pin_switch_swaps_blocked_and_unblocked() {
        let mut r = residency();
        r.sync_pins(&[AssetId(1)]);
        let changes = r.sync_pins(&[AssetId(2)]);
        assert_eq!(
            changes.blocked,
            vec![
                (CHANNEL_TEXTURE, 0),
                (CHANNEL_TEXTURE, 1),
                (CHANNEL_MESH, 5)
            ]
        );
        assert_eq!(changes.unblocked, vec![(CHANNEL_TEXTURE, 2)]);
    }

    #[test]
    fn state_and_progress_follow_residency_notes() {
        let mut r = residency();
        r.sync_pins(&[AssetId(1)]);
        assert_eq!(r.state(AssetId(1)), Some(SceneLoadState::Loading));

        r.note_resident((CHANNEL_TEXTURE, 0), true);
        r.note_resident((CHANNEL_TEXTURE, 1), true);
        assert_eq!(r.state(AssetId(1)), Some(SceneLoadState::Loading));
        assert!((r.progress(AssetId(1)).unwrap() - 2.0 / 3.0).abs() < 1e-6);

        r.note_resident((CHANNEL_MESH, 5), true);
        assert_eq!(r.state(AssetId(1)), Some(SceneLoadState::Resident));
        assert_eq!(r.progress(AssetId(1)), Some(1.0));

        // Unpinning with content still resident drains through Unloading.
        r.sync_pins(&[AssetId(2)]);
        assert_eq!(r.state(AssetId(1)), Some(SceneLoadState::Unloading));
        r.note_resident((CHANNEL_TEXTURE, 0), false);
        r.note_resident((CHANNEL_TEXTURE, 1), false);
        r.note_resident((CHANNEL_MESH, 5), false);
        assert_eq!(r.state(AssetId(1)), Some(SceneLoadState::Unloaded));
    }

    #[test]
    fn memberless_scene_is_resident_when_pinned() {
        let mut r = residency();
        r.sync_pins(&[AssetId(3)]);
        assert_eq!(r.state(AssetId(3)), Some(SceneLoadState::Resident));
        assert_eq!(r.progress(AssetId(3)), Some(1.0));
    }

    #[test]
    fn unowned_member_notes_are_ignored() {
        let mut r = residency();
        r.note_resident((CHANNEL_TEXTURE, 99), true);
        assert_eq!(r.progress(AssetId(1)), Some(0.0));
    }

    #[test]
    fn any_loading_tracks_pinned_scenes_only() {
        let mut r = residency();
        assert!(!r.any_loading());
        r.sync_pins(&[AssetId(1)]);
        assert!(r.any_loading());
        r.note_resident((CHANNEL_TEXTURE, 0), true);
        r.note_resident((CHANNEL_TEXTURE, 1), true);
        r.note_resident((CHANNEL_MESH, 5), true);
        assert!(!r.any_loading());
        // Unpinning drains through Unloading, which is not a load in flight.
        r.sync_pins(&[]);
        assert!(!r.any_loading());
    }

    #[test]
    fn status_lists_scenes_in_declaration_order() {
        let mut r = residency();
        r.sync_pins(&[AssetId(2)]);
        r.note_resident((CHANNEL_TEXTURE, 2), true);
        let status = r.status();
        assert_eq!(status[0].0, AssetId(1));
        assert_eq!(status[1], (AssetId(2), SceneLoadState::Resident, 1.0));
        assert_eq!(status[2], (AssetId(3), SceneLoadState::Unloaded, 1.0));
    }
}
