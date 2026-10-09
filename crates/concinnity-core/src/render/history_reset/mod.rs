//! When the temporal history every accumulating pass reprojects from stops
//! describing the frame about to render: a camera cut, a scene switch, or a
//! change to the render settings. Whatever moves the camera discontinuously (a
//! [`CameraTrack`](crate::components::CameraTrack) cut, a behavior teleport, a
//! debug camera write) says so through [`PendingHistoryReset::raise`]. The
//! tracker folds those causes in with what it sees itself, a change of camera
//! or scene and a view that strays far from the camera's predicted motion, so
//! every temporal consumer (TAA, the upscalers, SSGI, the occlusion pyramid)
//! drops its history on the same frame. A resize is not among them: each
//! backend rebuilds its temporal targets, and with them their history, on the
//! frame the surface changes.

mod latch;
mod motion;
mod tracker;

pub use latch::{ConsumedReset, UpscalerResetLatch};
pub use tracker::{FrameClock, HistoryResetTracker, HistoryView};

use crate::ecs::PipelineContext;

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

    /// This set less every cause in `other`.
    #[must_use]
    pub const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
}

impl core::fmt::Display for HistoryResetCauses {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let names = [
            (Self::CAMERA_CUT, "camera cut"),
            (Self::SCENE_SWITCH, "scene switch"),
            (Self::SETTINGS_CHANGE, "settings change"),
        ];
        let mut first = true;
        for (cause, name) in names {
            if self.contains(cause) {
                if !first {
                    f.write_str(", ")?;
                }
                f.write_str(name)?;
                first = false;
            }
        }
        if first {
            f.write_str("none")?;
        }
        Ok(())
    }
}

/// Causes reported from outside the frame for the next frame's
/// [`HistoryResetTracker::request`]. Carried as a resource between whatever
/// cut the camera or changed a setting and the system that observes the frame.
/// Causes are only ever added, so no report can erase another made in the
/// same tick.
#[derive(Debug)]
pub struct PendingHistoryReset(HistoryResetCauses);

impl PendingHistoryReset {
    /// Add `causes` to whatever is already pending. Raise it on the tick the
    /// camera's new pose is written, or the one that moves what the camera
    /// follows; the cut lands on the first frame drawn from the moved camera.
    pub fn raise(ctx: &mut PipelineContext, causes: HistoryResetCauses) {
        match ctx.resource_mut::<PendingHistoryReset>() {
            Some(pending) => pending.0 = pending.0.union(causes),
            None => {
                ctx.insert_resource(PendingHistoryReset(causes));
            }
        }
    }

    /// Everything pending, leaving nothing.
    pub fn take(ctx: &mut PipelineContext) -> HistoryResetCauses {
        ctx.resource_mut::<PendingHistoryReset>()
            .map_or(HistoryResetCauses::NONE, |pending| {
                core::mem::take(&mut pending.0)
            })
    }

    /// What is pending, without taking it.
    pub fn causes(&self) -> HistoryResetCauses {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::World;

    #[test]
    fn cause_sets_combine() {
        let both = HistoryResetCauses::SETTINGS_CHANGE.union(HistoryResetCauses::SCENE_SWITCH);
        assert!(both.contains(HistoryResetCauses::SETTINGS_CHANGE));
        assert!(both.contains(HistoryResetCauses::SCENE_SWITCH));
        assert!(!both.contains(HistoryResetCauses::CAMERA_CUT));
        assert!(!HistoryResetCauses::NONE.any());
    }

    #[test]
    fn raised_causes_accumulate_until_taken() {
        let mut world = World::new();
        let mut ctx = world.context();
        assert_eq!(
            PendingHistoryReset::take(&mut ctx),
            HistoryResetCauses::NONE
        );
        PendingHistoryReset::raise(&mut ctx, HistoryResetCauses::CAMERA_CUT);
        PendingHistoryReset::raise(&mut ctx, HistoryResetCauses::SETTINGS_CHANGE);
        let causes = PendingHistoryReset::take(&mut ctx);
        assert!(causes.contains(HistoryResetCauses::CAMERA_CUT));
        assert!(causes.contains(HistoryResetCauses::SETTINGS_CHANGE));
        assert_eq!(
            PendingHistoryReset::take(&mut ctx),
            HistoryResetCauses::NONE
        );
        PendingHistoryReset::raise(&mut ctx, HistoryResetCauses::SCENE_SWITCH);
        assert_eq!(
            PendingHistoryReset::take(&mut ctx),
            HistoryResetCauses::SCENE_SWITCH
        );
    }

    #[test]
    fn a_cause_set_drops_causes_and_names_the_rest() {
        let both = HistoryResetCauses::CAMERA_CUT.union(HistoryResetCauses::SCENE_SWITCH);
        assert_eq!(
            both.without(HistoryResetCauses::CAMERA_CUT),
            HistoryResetCauses::SCENE_SWITCH
        );
        assert_eq!(alloc::format!("{both}"), "camera cut, scene switch");
        assert_eq!(alloc::format!("{}", HistoryResetCauses::NONE), "none");
    }
}
