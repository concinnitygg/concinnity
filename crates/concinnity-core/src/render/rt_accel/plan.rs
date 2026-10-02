//! When the scene acceleration structure is updated, and which update runs.
//!
//! One frame's update has two halves. The first is settled before anything is
//! recorded: whether the draw BLAS head is refreshed (the participating set
//! changed, or the `Rebuild` diagnostic asks for every BLAS) and whether skinned
//! geometry takes part. The second is settled after any refresh, since a refresh
//! changes which transforms the TLAS was built from: re-skin, rebuild the TLAS
//! from the current transforms, or keep the live one.

use crate::render::error::RenderResult;
use crate::render::rt_geom::RtDynamicMode;

/// How the draw BLAS head is refreshed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefreshMode {
    /// Keep every BLAS whose geometry is unchanged and build only the rest.
    Reuse,
    /// Build every draw BLAS anew.
    RebuildAll,
}

/// The first half of a frame's update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RtUpdatePlan {
    /// Refresh the draw BLAS head before anything else, when set.
    pub refresh: Option<RefreshMode>,
    /// Skinned geometry is visible, so the frame re-skins and builds the TLAS
    /// over the head plus the skinned BLAS.
    pub skinned: bool,
    /// The skinned BLAS are built from scratch rather than refit.
    pub full_skinned_build: bool,
}

/// The second half of a frame's update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RtStep {
    /// The live TLAS stays as it is.
    Keep,
    /// Rebuild the TLAS and geometry table over the head from current transforms.
    Tlas,
    /// Re-skin, update the skinned BLAS, and rebuild the TLAS over the head plus
    /// the skinned BLAS.
    Skinned,
}

/// Whether a renderer with ray tracing on but no acceleration structure yet
/// should build one: geometry has just appeared, or skinned geometry is present.
pub fn seed_wanted(mode: RtDynamicMode, topology_dirty: bool, skinned_present: bool) -> bool {
    mode.is_dynamic() && (topology_dirty || skinned_present)
}

/// The first half of the update, or `None` when `mode` never updates the BVH.
pub(crate) fn plan_update(
    mode: RtDynamicMode,
    topology_dirty: bool,
    skinned_visible: bool,
) -> Option<RtUpdatePlan> {
    if !mode.is_dynamic() {
        return None;
    }
    let rebuild_all = mode == RtDynamicMode::Rebuild;
    let refresh = if rebuild_all {
        Some(RefreshMode::RebuildAll)
    } else if topology_dirty {
        Some(RefreshMode::Reuse)
    } else {
        None
    };
    Some(RtUpdatePlan {
        refresh,
        skinned: skinned_visible,
        full_skinned_build: rebuild_all,
    })
}

/// What the live TLAS reflects, for settling the second half of the update.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LiveState {
    /// Every participating draw object is still resident in its slot, so the
    /// head's transforms can be re-read in build order.
    pub(crate) models_current: bool,
    /// A participating transform differs from the one the TLAS was built with.
    pub(crate) moved: bool,
    /// The TLAS references skinned BLAS.
    pub(crate) has_skinned: bool,
}

/// The second half of the update. A refresh already built a TLAS over the
/// current head, so a frame without skinned geometry has nothing left to do
/// after one. A head whose objects changed shape without a refresh is left for
/// the refresh the change will flag.
pub(crate) fn next_step(mode: RtDynamicMode, plan: &RtUpdatePlan, live: LiveState) -> RtStep {
    if !live.models_current {
        return RtStep::Keep;
    }
    if plan.skinned {
        return RtStep::Skinned;
    }
    if plan.refresh.is_some() {
        return RtStep::Keep;
    }
    let rebuild = match mode {
        // A skinned tail left in the TLAS (the last skinned object just went
        // away) has to be dropped even when nothing static moved.
        RtDynamicMode::Auto => live.has_skinned || live.moved,
        RtDynamicMode::Rebuild | RtDynamicMode::Tlas => true,
        RtDynamicMode::Off => false,
    };
    if rebuild { RtStep::Tlas } else { RtStep::Keep }
}

/// What one frame's update did when it did not fail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RtUpdate {
    /// Every planned step ran.
    Done,
    /// A step waited on frames still in flight, so the live TLAS stayed.
    Skipped,
}

/// What a BVH does once its draw and cluster geometry are all gone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmptyHead {
    /// The skinned TLAS this frame builds replaces the live one: commit the
    /// empty head and hold its orphans until that TLAS publishes.
    AwaitSkinned,
    /// Skinned geometry can still rejoin: build the TLAS over what is visible,
    /// which may be nothing.
    Build,
    /// Nothing can rejoin without a topology change: drop the BVH.
    Drop,
}

/// How a BVH with no draw or cluster geometry left carries on. `skinned_follows`
/// is this frame's skinned step; `skinned_present` is whether any skinned
/// object exists to be traced, visible or not.
pub fn empty_head(skinned_follows: bool, skinned_present: bool) -> EmptyHead {
    if skinned_follows {
        EmptyHead::AwaitSkinned
    } else if skinned_present {
        EmptyHead::Build
    } else {
        EmptyHead::Drop
    }
}

/// Whether the update is failing, so a failure is reported once when a streak
/// begins and once when it ends rather than on every frame.
#[derive(Debug, Default)]
pub struct FailureStreak {
    failing: bool,
}

/// A change in whether the update is failing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreakChange {
    /// The first failure after a success.
    Began,
    /// The first success after a failure.
    Ended,
}

impl FailureStreak {
    /// Record one update's outcome, returning the change it made, if any. A
    /// skipped update neither starts nor ends a streak.
    pub fn record(&mut self, outcome: &RenderResult<RtUpdate>) -> Option<StreakChange> {
        let failed = match outcome {
            Ok(RtUpdate::Skipped) => return None,
            Ok(RtUpdate::Done) => false,
            Err(_) => true,
        };
        let change = match (self.failing, failed) {
            (false, true) => Some(StreakChange::Began),
            (true, false) => Some(StreakChange::Ended),
            _ => None,
        };
        self.failing = failed;
        change
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODES: [RtDynamicMode; 4] = [
        RtDynamicMode::Off,
        RtDynamicMode::Auto,
        RtDynamicMode::Rebuild,
        RtDynamicMode::Tlas,
    ];

    fn live(models_current: bool, moved: bool, has_skinned: bool) -> LiveState {
        LiveState {
            models_current,
            moved,
            has_skinned,
        }
    }

    #[test]
    fn off_never_updates_or_seeds() {
        assert_eq!(plan_update(RtDynamicMode::Off, true, true), None);
        assert!(!seed_wanted(RtDynamicMode::Off, true, true));
    }

    #[test]
    fn a_seed_waits_for_geometry() {
        assert!(!seed_wanted(RtDynamicMode::Auto, false, false));
        assert!(seed_wanted(RtDynamicMode::Auto, true, false));
        assert!(seed_wanted(RtDynamicMode::Tlas, false, true));
    }

    #[test]
    fn a_topology_change_refreshes_the_head_reusing_what_it_can() {
        for mode in [RtDynamicMode::Auto, RtDynamicMode::Tlas] {
            let plan = plan_update(mode, true, false).expect("dynamic");
            assert_eq!(plan.refresh, Some(RefreshMode::Reuse));
            assert!(!plan.full_skinned_build);
            assert_eq!(
                plan_update(mode, false, false).expect("dynamic").refresh,
                None
            );
        }
    }

    #[test]
    fn rebuild_rebuilds_every_blas_on_every_frame() {
        for dirty in [false, true] {
            let plan = plan_update(RtDynamicMode::Rebuild, dirty, true).expect("dynamic");
            assert_eq!(plan.refresh, Some(RefreshMode::RebuildAll));
            assert!(plan.full_skinned_build);
        }
    }

    #[test]
    fn visible_skinned_geometry_always_takes_the_skinned_step() {
        for mode in MODES.into_iter().filter(|m| m.is_dynamic()) {
            for dirty in [false, true] {
                let plan = plan_update(mode, dirty, true).expect("dynamic");
                assert_eq!(
                    next_step(mode, &plan, live(true, false, false)),
                    RtStep::Skinned
                );
            }
        }
    }

    #[test]
    fn a_head_that_changed_shape_waits_for_its_refresh() {
        for mode in MODES.into_iter().filter(|m| m.is_dynamic()) {
            for skinned in [false, true] {
                let plan = plan_update(mode, false, skinned).expect("dynamic");
                assert_eq!(
                    next_step(mode, &plan, live(false, true, true)),
                    RtStep::Keep
                );
            }
        }
    }

    #[test]
    fn a_refresh_is_the_whole_static_update() {
        let plan = plan_update(RtDynamicMode::Auto, true, false).expect("dynamic");
        assert_eq!(
            next_step(RtDynamicMode::Auto, &plan, live(true, true, true)),
            RtStep::Keep
        );
    }

    #[test]
    fn auto_rebuilds_the_tlas_only_when_needed() {
        let mode = RtDynamicMode::Auto;
        let plan = plan_update(mode, false, false).expect("dynamic");
        assert_eq!(
            next_step(mode, &plan, live(true, false, false)),
            RtStep::Keep
        );
        assert_eq!(
            next_step(mode, &plan, live(true, true, false)),
            RtStep::Tlas
        );
        // The last skinned object went away: drop its tail.
        assert_eq!(
            next_step(mode, &plan, live(true, false, true)),
            RtStep::Tlas
        );
    }

    #[test]
    fn tlas_rebuilds_on_every_static_frame() {
        let mode = RtDynamicMode::Tlas;
        let plan = plan_update(mode, false, false).expect("dynamic");
        assert_eq!(
            next_step(mode, &plan, live(true, false, false)),
            RtStep::Tlas
        );
    }

    fn failed() -> RenderResult<RtUpdate> {
        Err(crate::render::error::RenderError::Other(
            "build failed".into(),
        ))
    }

    #[test]
    fn a_failure_is_reported_once_per_streak() {
        let mut streak = FailureStreak::default();
        assert_eq!(streak.record(&Ok(RtUpdate::Done)), None);
        assert_eq!(streak.record(&failed()), Some(StreakChange::Began));
        assert_eq!(streak.record(&failed()), None);
        assert_eq!(
            streak.record(&Ok(RtUpdate::Done)),
            Some(StreakChange::Ended)
        );
        assert_eq!(streak.record(&Ok(RtUpdate::Done)), None);
    }

    #[test]
    fn a_skipped_update_neither_starts_nor_ends_a_streak() {
        let mut streak = FailureStreak::default();
        assert_eq!(streak.record(&Ok(RtUpdate::Skipped)), None);
        assert_eq!(streak.record(&failed()), Some(StreakChange::Began));
        assert_eq!(streak.record(&Ok(RtUpdate::Skipped)), None);
        assert_eq!(
            streak.record(&Ok(RtUpdate::Done)),
            Some(StreakChange::Ended)
        );
    }

    #[test]
    fn an_empty_bvh_is_dropped_only_when_nothing_can_rejoin_it() {
        assert_eq!(empty_head(true, true), EmptyHead::AwaitSkinned);
        assert_eq!(empty_head(false, true), EmptyHead::Build);
        assert_eq!(empty_head(false, false), EmptyHead::Drop);
    }
}
