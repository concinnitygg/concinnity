//! One frame's update of a live scene acceleration structure, in the order
//! every backend runs it: plan, refresh the draw BLAS head if the plan says so,
//! then keep, rebuild the TLAS, or re-skin. The backend records each step
//! through [`RtAccelSteps`]; the order and the bookkeeping between steps live
//! here.

use crate::gfx::render_types::{DrawObject, SkinnedDrawObject};
use crate::render::error::RenderResult;
use crate::render::rt_accel::{AccelBook, EmptyHead, RefreshMode, RtStep, RtUpdate, empty_head};
use crate::render::rt_geom::RtDynamicMode;

/// The refresh step's inputs: how the head is refreshed, and what the frame's
/// skinned step does with an emptied head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefreshStep {
    /// How the draw BLAS head is refreshed.
    pub mode: RefreshMode,
    /// This frame's update re-skins after the refresh.
    pub skinned_follows: bool,
    /// Skinned geometry exists to be traced, visible or not.
    pub skinned_present: bool,
}

impl RefreshStep {
    /// What a refresh that leaves no draw or cluster geometry does.
    pub fn if_empty(&self) -> EmptyHead {
        empty_head(self.skinned_follows, self.skinned_present)
    }
}

/// The GPU half of a live acceleration structure's update: one backend's
/// recording of each step [`drive_update`] schedules for the frame. An
/// implementer borrows the frame's command target and geometry for the call.
pub trait RtAccelSteps {
    /// The backend's BLAS handle.
    type Blas;
    /// The backend's TLAS instance record.
    type Instance;

    /// The bookkeeping the update plans from and commits to.
    fn book(&mut self) -> &mut AccelBook<Self::Blas, Self::Instance>;

    /// Bring the draw BLAS head in line with the current draw set.
    fn refresh_topology(&mut self, refresh: RefreshStep) -> RenderResult<()>;

    /// Rebuild the TLAS and geometry table over the head from current
    /// transforms.
    fn rebuild_tlas(&mut self) -> RenderResult<()>;

    /// Stop publishing the skinned tail: nothing static is left and no skinned
    /// geometry can rejoin.
    fn release_skinned(&mut self);

    /// Re-skin, update the skinned BLAS, and rebuild the TLAS over the head
    /// plus the skinned BLAS, from scratch when `full_build`.
    fn rebuild_skinned(&mut self, full_build: bool) -> RenderResult<RtUpdate>;
}

/// One frame's inputs to [`drive_update`].
#[derive(Clone, Copy)]
pub struct RtDriveInputs<'a> {
    /// The update mode.
    pub mode: RtDynamicMode,
    /// The participating draw set changed since the last update.
    pub topology_dirty: bool,
    /// The frame's draw list.
    pub draw_objects: &'a [DrawObject],
    /// The frame's skinned objects, when skinned geometry takes part.
    pub skinned: Option<&'a [SkinnedDrawObject]>,
}

/// Run one frame's update of `device`'s live acceleration structure. A failed
/// refresh keeps the last head, is planned again next frame, and still lets
/// the step after it run. A refresh on the skinned path leaves the TLAS to the
/// skinned step, so a skinned step that did not finish keeps the refresh owed.
/// When both fail, the refresh's error is the one reported.
pub fn drive_update<D: RtAccelSteps>(
    device: &mut D,
    inputs: RtDriveInputs<'_>,
) -> RenderResult<RtUpdate> {
    let RtDriveInputs {
        mode,
        topology_dirty,
        draw_objects,
        skinned,
    } = inputs;
    let Some(plan) = device.book().plan(mode, topology_dirty, skinned) else {
        return Ok(RtUpdate::Done);
    };

    let mut refreshed = Ok(());
    if let Some(refresh_mode) = plan.refresh {
        refreshed = device.refresh_topology(RefreshStep {
            mode: refresh_mode,
            skinned_follows: plan.skinned,
            skinned_present: skinned.is_some(),
        });
        if refreshed.is_err() {
            device.book().owe_refresh();
        }
    }

    let stepped = match device.book().next_step(mode, &plan, draw_objects) {
        RtStep::Keep => Ok(RtUpdate::Done),
        RtStep::Tlas if device.book().is_empty() && skinned.is_none() => {
            device.release_skinned();
            Ok(RtUpdate::Done)
        }
        // An empty head still gets a (zero-instance) TLAS, so the trace stops
        // reaching what left.
        RtStep::Tlas => device.rebuild_tlas().map(|()| RtUpdate::Done),
        RtStep::Skinned => device.rebuild_skinned(plan.full_skinned_build),
    };
    if plan.refresh.is_some() && plan.skinned && !matches!(stepped, Ok(RtUpdate::Done)) {
        device.book().owe_refresh();
    }
    refreshed.and(stepped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::SaturatingFrom;
    use crate::render::error::RenderError;
    use crate::render::rt_accel::SeedSet;
    use crate::test_support::{draw_object, skinned_draw_object};
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;

    #[derive(Debug, Clone, PartialEq)]
    enum Call {
        Refresh(RefreshStep),
        Tlas,
        Release,
        Skinned(bool),
    }

    struct Fake {
        book: AccelBook<u32, u32>,
        calls: Vec<Call>,
        fail_refresh: bool,
        fail_step: bool,
        skinned_result: RtUpdate,
    }

    impl Fake {
        fn seeded(draw: &[DrawObject]) -> Self {
            let seed = SeedSet::new(draw, &[], false);
            let blas = (0..u32::saturating_from(seed.objects.len())).collect();
            Self {
                book: AccelBook::new(&seed, blas, draw, 1).expect("one BLAS per object"),
                calls: Vec::new(),
                fail_refresh: false,
                fail_step: false,
                skinned_result: RtUpdate::Done,
            }
        }

        fn step(&self, label: &str) -> RenderResult<()> {
            if self.fail_step {
                Err(RenderError::Other(String::from(label)))
            } else {
                Ok(())
            }
        }
    }

    impl RtAccelSteps for Fake {
        type Blas = u32;
        type Instance = u32;

        fn book(&mut self) -> &mut AccelBook<u32, u32> {
            &mut self.book
        }

        fn refresh_topology(&mut self, refresh: RefreshStep) -> RenderResult<()> {
            self.calls.push(Call::Refresh(refresh));
            if self.fail_refresh {
                Err(RenderError::Other(String::from("refresh")))
            } else {
                Ok(())
            }
        }

        fn rebuild_tlas(&mut self) -> RenderResult<()> {
            self.calls.push(Call::Tlas);
            self.book.commit_static();
            self.step("tlas")
        }

        fn release_skinned(&mut self) {
            self.calls.push(Call::Release);
        }

        fn rebuild_skinned(&mut self, full_build: bool) -> RenderResult<RtUpdate> {
            self.calls.push(Call::Skinned(full_build));
            self.step("skinned").map(|()| self.skinned_result)
        }
    }

    fn inputs<'a>(
        mode: RtDynamicMode,
        dirty: bool,
        draw: &'a [DrawObject],
        skinned: Option<&'a [SkinnedDrawObject]>,
    ) -> RtDriveInputs<'a> {
        RtDriveInputs {
            mode,
            topology_dirty: dirty,
            draw_objects: draw,
            skinned,
        }
    }

    fn error_text(r: RenderResult<RtUpdate>) -> String {
        match r {
            Err(RenderError::Other(s)) => s,
            other => panic!("expected an error, got {other:?}"),
        }
    }

    #[test]
    fn a_mode_that_never_updates_records_nothing() {
        let draw = vec![draw_object()];
        let mut d = Fake::seeded(&draw);
        let r = drive_update(&mut d, inputs(RtDynamicMode::Off, true, &draw, None));
        assert_eq!(r, Ok(RtUpdate::Done));
        assert!(d.calls.is_empty());
    }

    #[test]
    fn a_still_scene_keeps_and_a_tlas_mode_rebuilds() {
        let draw = vec![draw_object()];
        let mut d = Fake::seeded(&draw);
        drive_update(&mut d, inputs(RtDynamicMode::Auto, false, &draw, None)).unwrap();
        assert!(d.calls.is_empty(), "nothing moved");
        drive_update(&mut d, inputs(RtDynamicMode::Tlas, false, &draw, None)).unwrap();
        assert_eq!(d.calls, vec![Call::Tlas]);
    }

    // The refresh rebuilds the static TLAS itself, so nothing follows it.
    #[test]
    fn a_dirty_topology_refreshes_and_stops() {
        let draw = vec![draw_object()];
        let mut d = Fake::seeded(&draw);
        drive_update(&mut d, inputs(RtDynamicMode::Tlas, true, &draw, None)).unwrap();
        assert_eq!(
            d.calls,
            vec![Call::Refresh(RefreshStep {
                mode: RefreshMode::Reuse,
                skinned_follows: false,
                skinned_present: false,
            })]
        );
    }

    #[test]
    fn skinned_geometry_takes_the_skinned_step() {
        let draw = vec![draw_object()];
        let skinned = [skinned_draw_object()];
        let mut d = Fake::seeded(&draw);
        drive_update(
            &mut d,
            inputs(RtDynamicMode::Auto, true, &draw, Some(&skinned)),
        )
        .unwrap();
        assert_eq!(d.calls.len(), 2);
        let Call::Refresh(refresh) = d.calls[0].clone() else {
            panic!("refresh first: {:?}", d.calls);
        };
        assert!(refresh.skinned_follows && refresh.skinned_present);
        assert_eq!(refresh.if_empty(), EmptyHead::AwaitSkinned);
        assert!(matches!(d.calls[1], Call::Skinned(_)));
    }

    #[test]
    fn a_failed_refresh_still_steps_and_is_owed() {
        let draw = vec![draw_object()];
        let skinned = [skinned_draw_object()];
        let mut d = Fake::seeded(&draw);
        d.fail_refresh = true;
        let r = drive_update(
            &mut d,
            inputs(RtDynamicMode::Auto, true, &draw, Some(&skinned)),
        );
        assert_eq!(error_text(r), "refresh");
        assert!(
            matches!(d.calls.last(), Some(Call::Skinned(_))),
            "{:?}",
            d.calls
        );
        d.fail_refresh = false;
        d.calls.clear();
        drive_update(&mut d, inputs(RtDynamicMode::Tlas, false, &draw, None)).unwrap();
        assert!(
            matches!(d.calls[0], Call::Refresh(_)),
            "owed: {:?}",
            d.calls
        );
    }

    #[test]
    fn a_double_failure_reports_the_refresh() {
        let draw = vec![draw_object()];
        let skinned = [skinned_draw_object()];
        let mut d = Fake::seeded(&draw);
        d.fail_refresh = true;
        d.fail_step = true;
        let i = inputs(RtDynamicMode::Auto, true, &draw, Some(&skinned));
        assert_eq!(error_text(drive_update(&mut d, i)), "refresh");
    }

    #[test]
    fn an_unfinished_skinned_step_keeps_the_refresh_owed() {
        let draw = vec![draw_object()];
        let skinned = [skinned_draw_object()];
        let mut d = Fake::seeded(&draw);
        d.skinned_result = RtUpdate::Skipped;
        let i = inputs(RtDynamicMode::Auto, true, &draw, Some(&skinned));
        assert_eq!(drive_update(&mut d, i), Ok(RtUpdate::Skipped));
        d.calls.clear();
        drive_update(
            &mut d,
            inputs(RtDynamicMode::Auto, false, &draw, Some(&skinned)),
        )
        .unwrap();
        assert!(
            matches!(d.calls.first(), Some(Call::Refresh(_))),
            "{:?}",
            d.calls
        );
    }

    #[test]
    fn an_emptied_book_with_nothing_to_rejoin_releases_the_skinned_tail() {
        let mut draw = vec![draw_object()];
        let mut d = Fake::seeded(&draw);
        // The object leaves; the refresh (a no-op here) leaves the head as is,
        // so empty it by hand the way a backend's refresh commit would.
        draw[0].resident = false;
        let refresh = d.book.plan_refresh(&draw, false, RefreshMode::Reuse);
        let fresh: Vec<Option<u32>> = vec![None; refresh.indices().len()];
        d.book.commit_refresh(refresh, fresh, &draw);
        assert!(d.book.is_empty());
        drive_update(&mut d, inputs(RtDynamicMode::Tlas, false, &draw, None)).unwrap();
        assert_eq!(d.calls, vec![Call::Release]);
    }
}
