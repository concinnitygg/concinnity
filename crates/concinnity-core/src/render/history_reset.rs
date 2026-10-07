//! When the temporal history every accumulating pass reprojects from stops
//! describing the frame about to render: a camera cut, a scene switch, or a
//! change to the render settings. The tracker compares each frame's view
//! against the last and folds in the causes reported from outside it, so every
//! temporal consumer (TAA, the upscalers, SSGI, the occlusion pyramid) drops
//! its history on the same frame. A resize is not among them: each backend
//! rebuilds its temporal targets, and with them their history, on the frame
//! the surface changes.

use core::sync::atomic::{AtomicU8, Ordering};

use crate::ecs::asset_id::AssetId;
use crate::transform::Mat4;

/// Why a frame's temporal history cannot be reprojected, as a set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HistoryResetCauses(u8);

impl HistoryResetCauses {
    /// The history is still valid.
    pub const NONE: Self = Self(0);
    /// The camera jumped: a teleport, a shot change, a snap turn.
    pub const CAMERA_CUT: Self = Self(1 << 0);
    /// Another scene became the active one.
    pub const SCENE_SWITCH: Self = Self(1 << 1);
    /// A render setting changed what the frame draws.
    pub const SETTINGS_CHANGE: Self = Self(1 << 2);

    /// Whether any cause is present.
    pub const fn any(self) -> bool {
        self.0 != 0
    }

    /// Whether every cause in `other` is present.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Both sets together.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// A frame's view as the tracker compares it with the previous one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistoryView {
    /// World-space camera position, before any render-origin rebase, so a
    /// rebase is not mistaken for a teleport.
    pub position: [f32; 3],
    /// The camera's view matrix; only its rotation is read.
    pub view: Mat4,
    /// Vertical field of view in radians.
    pub fov_y_radians: f32,
    /// The active scene, or `None` in a world without scenes.
    pub scene: Option<AssetId>,
}

/// The farthest the camera moves between two frames, in world units, before
/// the move counts as a cut.
pub const CUT_DISTANCE: f32 = 10.0;
/// The cosine of the largest turn between two frames that is not a cut (60
/// degrees).
pub const CUT_TURN_COS: f32 = 0.5;
/// The largest field-of-view ratio between two frames that is not a cut.
pub const CUT_FOV_RATIO: f32 = 1.25;

/// The previous frame's view and the causes reported since, turned into one
/// reset decision per frame.
#[derive(Clone, Debug, Default)]
pub struct HistoryResetTracker {
    prev: Option<HistoryView>,
    requested: HistoryResetCauses,
}

impl HistoryResetTracker {
    /// Report causes found outside the view, applied to the next observed
    /// frame.
    pub fn request(&mut self, causes: HistoryResetCauses) {
        self.requested = self.requested.union(causes);
    }

    /// Compare `view` with the previous frame's and return why this frame's
    /// history is invalid, including every cause requested since the last
    /// call. The first frame resets nothing: there is no history yet.
    pub fn observe(&mut self, view: HistoryView) -> HistoryResetCauses {
        let requested = core::mem::take(&mut self.requested);
        match self.prev.replace(view) {
            Some(prev) => requested.union(view_changes(&prev, &view)),
            None => HistoryResetCauses::NONE,
        }
    }
}

// What separates two consecutive views beyond what reprojection can bridge.
fn view_changes(prev: &HistoryView, cur: &HistoryView) -> HistoryResetCauses {
    let mut causes = HistoryResetCauses::NONE;
    if is_camera_cut(prev, cur) {
        causes = causes.union(HistoryResetCauses::CAMERA_CUT);
    }
    if prev.scene != cur.scene {
        causes = causes.union(HistoryResetCauses::SCENE_SWITCH);
    }
    causes
}

fn is_camera_cut(prev: &HistoryView, cur: &HistoryView) -> bool {
    let moved = sub(cur.position, prev.position);
    if dot(moved, moved) > CUT_DISTANCE * CUT_DISTANCE {
        return true;
    }
    if dot(forward(&prev.view), forward(&cur.view)) < CUT_TURN_COS {
        return true;
    }
    let (wide, narrow) = if cur.fov_y_radians > prev.fov_y_radians {
        (cur.fov_y_radians, prev.fov_y_radians)
    } else {
        (prev.fov_y_radians, cur.fov_y_radians)
    };
    wide > narrow * CUT_FOV_RATIO
}

// The world-space viewing direction of a view matrix, up to sign: the view
// rotation's third row.
fn forward(view: &Mat4) -> [f32; 3] {
    [view[0][2], view[1][2], view[2][2]]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Causes reported from outside the frame (a settings change) for the next
/// frame's [`HistoryResetTracker::request`]. Carried as a resource between
/// the system that applies the change and the one that observes the frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PendingHistoryReset(pub HistoryResetCauses);

/// Whether a temporal upscaler discards its history on its next dispatch:
/// pending from creation until the first dispatch, and again after each
/// [`request`](Self::request). A request made while the creation reset is
/// still pending folds into it. Atomic, so a backend may request on one thread
/// and dispatch on another.
#[derive(Debug)]
pub struct UpscalerResetLatch(AtomicU8);

const PENDING_NONE: u8 = 0;
const PENDING_CREATED: u8 = 1;
const PENDING_REQUESTED: u8 = 2;

/// What one dispatch found pending.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsumedReset {
    /// Nothing: the history carries on.
    None,
    /// The upscaler was just created or rebuilt.
    Created,
    /// A frame asked for its history to be dropped.
    Requested,
}

impl ConsumedReset {
    /// Whether this dispatch discards the history.
    pub const fn discards(self) -> bool {
        !matches!(self, ConsumedReset::None)
    }
}

impl Default for UpscalerResetLatch {
    fn default() -> Self {
        Self(AtomicU8::new(PENDING_CREATED))
    }
}

impl UpscalerResetLatch {
    /// Discard the history on the next dispatch.
    pub fn request(&self) {
        let _ = self.0.compare_exchange(
            PENDING_NONE,
            PENDING_REQUESTED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    /// The upscaler was rebuilt, so its next dispatch starts over as if new.
    pub fn rebuilt(&self) {
        self.0.store(PENDING_CREATED, Ordering::Release);
    }

    /// What this dispatch finds pending, clearing it.
    pub fn take(&self) -> ConsumedReset {
        match self.0.swap(PENDING_NONE, Ordering::AcqRel) {
            PENDING_CREATED => ConsumedReset::Created,
            PENDING_REQUESTED => ConsumedReset::Requested,
            _ => ConsumedReset::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::camera::view_matrix;

    fn view_at(position: [f32; 3], yaw: f32) -> HistoryView {
        looking(position, yaw, 0.0)
    }

    fn looking(position: [f32; 3], yaw: f32, pitch: f32) -> HistoryView {
        HistoryView {
            position,
            view: view_matrix(position, yaw, pitch),
            fov_y_radians: 1.0,
            scene: Some(AssetId(1)),
        }
    }

    fn primed(view: HistoryView) -> HistoryResetTracker {
        let mut tracker = HistoryResetTracker::default();
        assert_eq!(tracker.observe(view), HistoryResetCauses::NONE);
        tracker
    }

    #[test]
    fn the_first_frame_resets_nothing_even_when_asked() {
        let mut tracker = HistoryResetTracker::default();
        tracker.request(HistoryResetCauses::SETTINGS_CHANGE);
        assert!(!tracker.observe(view_at([0.0; 3], 0.0)).any());
    }

    #[test]
    fn a_steady_or_smoothly_moving_camera_keeps_its_history() {
        let mut tracker = primed(view_at([0.0; 3], 0.0));
        assert!(!tracker.observe(view_at([0.0; 3], 0.0)).any());
        assert!(!tracker.observe(view_at([0.5, 0.0, -0.5], 0.05)).any());
        assert!(!tracker.observe(view_at([1.0, 0.0, -1.0], 0.1)).any());
    }

    #[test]
    fn a_jump_beyond_the_cut_distance_is_a_cut() {
        let mut tracker = primed(view_at([0.0; 3], 0.0));
        let causes = tracker.observe(view_at([CUT_DISTANCE + 0.1, 0.0, 0.0], 0.0));
        assert_eq!(causes, HistoryResetCauses::CAMERA_CUT);
        // The cut frame becomes the new reference.
        assert!(
            !tracker
                .observe(view_at([CUT_DISTANCE + 0.2, 0.0, 0.0], 0.0))
                .any()
        );
    }

    #[test]
    fn a_turn_past_sixty_degrees_is_a_cut_and_a_smaller_one_is_not() {
        let mut tracker = primed(view_at([0.0; 3], 0.0));
        assert!(!tracker.observe(view_at([0.0; 3], 0.9)).any());
        let causes = tracker.observe(view_at([0.0; 3], 0.9 + 1.1));
        assert_eq!(causes, HistoryResetCauses::CAMERA_CUT);
    }

    #[test]
    fn a_pitch_past_sixty_degrees_is_a_cut_and_a_smaller_one_is_not() {
        let mut tracker = primed(looking([0.0; 3], 0.0, 0.0));
        assert!(!tracker.observe(looking([0.0; 3], 0.0, 0.9)).any());
        let causes = tracker.observe(looking([0.0; 3], 0.0, 0.9 - 1.1));
        assert_eq!(causes, HistoryResetCauses::CAMERA_CUT);
    }

    // A yaw and a pitch that are each under the cut angle on their own still
    // cut when together they turn the view past it.
    #[test]
    fn a_combined_turn_is_measured_as_one_rotation() {
        let level = looking([0.0; 3], 0.0, 0.0);
        let mut tracker = primed(level);
        assert!(!tracker.observe(looking([0.0; 3], 0.8, 0.0)).any());
        let mut tracker = primed(level);
        assert!(!tracker.observe(looking([0.0; 3], 0.0, 0.8)).any());
        let mut tracker = primed(level);
        assert_eq!(
            tracker.observe(looking([0.0; 3], 0.8, 0.8)),
            HistoryResetCauses::CAMERA_CUT
        );
    }

    #[test]
    fn a_field_of_view_snap_is_a_cut() {
        let base = view_at([0.0; 3], 0.0);
        let mut tracker = primed(base);
        let zoomed = HistoryView {
            fov_y_radians: base.fov_y_radians / (CUT_FOV_RATIO + 0.1),
            ..base
        };
        assert_eq!(tracker.observe(zoomed), HistoryResetCauses::CAMERA_CUT);
        let eased = HistoryView {
            fov_y_radians: zoomed.fov_y_radians * 1.05,
            ..zoomed
        };
        assert!(!tracker.observe(eased).any());
    }

    #[test]
    fn a_scene_switch_is_its_own_cause() {
        let base = view_at([0.0; 3], 0.0);
        let mut tracker = primed(base);
        let switched = HistoryView {
            scene: Some(AssetId(2)),
            ..base
        };
        assert_eq!(tracker.observe(switched), HistoryResetCauses::SCENE_SWITCH);
        assert!(!tracker.observe(switched).any());
    }

    #[test]
    fn a_requested_cause_lands_on_the_next_frame_only() {
        let base = view_at([0.0; 3], 0.0);
        let mut tracker = primed(base);
        tracker.request(HistoryResetCauses::SETTINGS_CHANGE);
        let jumped = view_at([50.0, 0.0, 0.0], 0.0);
        let causes = tracker.observe(jumped);
        assert!(causes.contains(HistoryResetCauses::SETTINGS_CHANGE));
        assert!(causes.contains(HistoryResetCauses::CAMERA_CUT));
        assert!(!tracker.observe(jumped).any());
    }

    #[test]
    fn the_latch_resets_on_creation_and_on_each_request() {
        let latch = UpscalerResetLatch::default();
        assert_eq!(latch.take(), ConsumedReset::Created);
        assert_eq!(latch.take(), ConsumedReset::None);
        latch.request();
        latch.request();
        assert_eq!(latch.take(), ConsumedReset::Requested);
        assert!(!latch.take().discards());
    }

    // A request while the creation reset is pending, or after a rebuild, folds
    // into it; a rebuild overrides a pending request.
    #[test]
    fn creation_absorbs_a_pending_request() {
        let latch = UpscalerResetLatch::default();
        latch.request();
        assert_eq!(latch.take(), ConsumedReset::Created);
        latch.request();
        latch.rebuilt();
        assert_eq!(latch.take(), ConsumedReset::Created);
        latch.rebuilt();
        latch.request();
        assert_eq!(latch.take(), ConsumedReset::Created);
    }

    #[test]
    fn every_consumed_reset_but_none_discards() {
        assert!(ConsumedReset::Requested.discards());
        assert!(ConsumedReset::Created.discards());
        assert!(!ConsumedReset::None.discards());
    }

    #[test]
    fn cause_sets_combine() {
        let both = HistoryResetCauses::SETTINGS_CHANGE.union(HistoryResetCauses::SCENE_SWITCH);
        assert!(both.contains(HistoryResetCauses::SETTINGS_CHANGE));
        assert!(both.contains(HistoryResetCauses::SCENE_SWITCH));
        assert!(!both.contains(HistoryResetCauses::CAMERA_CUT));
        assert!(!HistoryResetCauses::NONE.any());
    }
}
