// concinnity-physics/src/sim/ccd/toi.rs
//
// Where along its own step a mover first met something, and which regions it
// went through on the way.
//
// One walk answers both. The traversal is the query path's: the broad phase
// hands back the window of proxies the swept box could reach, the window is
// filtered by layer, and only what survives is swept exactly. Doing the two
// questions together is what keeps a fast body's extra cost to one window
// walk rather than two.
//
// The sweep is run in the target's frame: the mover advances by the motion of
// the pair relative to each other, against the target where the step began.
// Two bodies that ran at each other would each stop at the other's starting
// place otherwise, which is to say they would swap sides.
//
// The proxies still hold the bounds the step began with, since nothing has
// refreshed them since the sweep. That is the frame this pass wants, but it
// means a target that has moved is bounded where it was, so the swept box is
// widened by the furthest anything travelled this step before it is used to
// pick candidates.

use crate::{ColliderShape, LayerMask};

use crate::sim::aabb::shape_bounds;
use crate::sim::body::Body;
use crate::sim::collide::Pose;
use crate::sim::math::{Quat, Vec3};
use crate::sim::query::field;
use crate::sim::query::gjk::Support;
use crate::sim::query::sweep::sweep;
use crate::sim::scene::Scene;
use crate::sim::sensor::swept;

/// One mover's path through the step, as the scan is asked about it.
pub(crate) struct Probe<'a> {
    /// The mover's own slot, left out of its own scan.
    pub(crate) slot: u32,
    pub(crate) shape: &'a ColliderShape,
    /// The orientation the step left the mover with, held for the whole
    /// sweep. Stopping the mover on this path leaves it clear at the pose it
    /// actually ends the step in.
    pub(crate) rotation: Quat,
    pub(crate) start: Vec3,
    pub(crate) motion: Vec3,
    pub(crate) mask: LayerMask,
    /// The furthest any body travelled this step, added to the swept box so a
    /// pair that meets only because both moved is still a candidate.
    pub(crate) expand: f32,
}

/// What a mover ran into first.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Blocked {
    /// Slot of the body that was met.
    pub(crate) target: u32,
    /// Fraction of the mover's own motion covered before the contact.
    pub(crate) toi: f32,
    /// Unit normal on the body that was met, pointing back at the mover.
    pub(crate) normal: Vec3,
    /// Motion of the pair relative to each other, which is what the time of
    /// impact was measured along.
    pub(crate) relative_motion: Vec3,
}

/// The first body the mover met, reporting every region it passed clean
/// through to `on_region` along the way.
///
/// `motion_of` says how far a candidate travelled during the step, so the
/// sweep can be run in that candidate's frame.
pub(crate) fn scan(
    scene: Scene<'_>,
    probe: &Probe<'_>,
    motion_of: impl Fn(u32, &Body) -> Vec3,
    mut on_region: impl FnMut(u32),
) -> Option<Blocked> {
    let pose = Pose {
        position: probe.start,
        rotation: probe.rotation,
    };
    let moving = Support::new(probe.shape, pose);
    let swept_bounds = shape_bounds(probe.shape, probe.start, probe.rotation)
        .union(shape_bounds(
            probe.shape,
            probe.start + probe.motion,
            probe.rotation,
        ))
        .expanded(probe.expand);

    let axis = scene.broadphase.axis();
    let mut best: Option<Blocked> = None;
    for &slot in scene
        .broadphase
        .slab_window(swept_bounds.min.get(axis), swept_bounds.max.get(axis))
    {
        if slot == probe.slot {
            continue;
        }
        let proxy = scene.broadphase.proxy(slot);
        if !probe.mask.interacts_with(proxy.mask) || !swept_bounds.overlaps(proxy.bounds) {
            continue;
        }
        let Some(body) = scene.bodies.get_at(slot as usize) else {
            continue;
        };
        if body.is_sensor() {
            if let Some(shape) = body.convex()
                && swept::passed_through(&moving, probe.motion, &Support::new(shape, pose_of(body)))
            {
                on_region(slot);
            }
            continue;
        }

        let target_motion = motion_of(slot, body);
        let relative = probe.motion - target_motion;
        let found = match body.terrain_index() {
            // Terrain never moves, so the frame the sweep runs in is the
            // world's and the swept box already names the right cells.
            Some(index) => field::sweep(
                scene.fields,
                index,
                probe.shape,
                pose,
                relative,
                swept_bounds,
            ),
            None => body.convex().and_then(|shape| {
                let began = Pose {
                    position: body.position - target_motion,
                    rotation: body.orientation,
                };
                sweep(&moving, relative, &Support::new(shape, began))
            }),
        };
        let Some(impact) = found else {
            continue;
        };
        // A pair already touching had a manifold this step and the solver
        // owns it; stopping the mover where it began would freeze a body
        // that is resting on something.
        if impact.started_touching || impact.toi >= 1.0 {
            continue;
        }
        if nearer(&best, slot, impact.toi) {
            best = Some(Blocked {
                target: slot,
                toi: impact.toi,
                normal: impact.normal,
                relative_motion: relative,
            });
        }
    }
    best
}

/// Whether the mover still crosses clean through `region` once its motion has
/// been cut back to where the sweep stopped it.
///
/// A mover the scan reported a region for was measured over its whole step;
/// one that was also stopped part way along never reached the far side of
/// everything it passed over, and a region it ends up inside belongs to the
/// boundary test again.
pub(crate) fn still_crossed(
    scene: Scene<'_>,
    probe: &Probe<'_>,
    motion: Vec3,
    region: u32,
) -> bool {
    let Some(body) = scene.bodies.get_at(region as usize) else {
        return false;
    };
    let Some(shape) = body.convex() else {
        return false;
    };
    let moving = Support::new(
        probe.shape,
        Pose {
            position: probe.start,
            rotation: probe.rotation,
        },
    );
    swept::passed_through(&moving, motion, &Support::new(shape, pose_of(body)))
}

/// Whether a fresh impact beats the one held. Time decides; the body slot
/// breaks a tie, so the answer does not depend on the order the window
/// happened to arrive in.
fn nearer(best: &Option<Blocked>, slot: u32, toi: f32) -> bool {
    match best {
        None => true,
        Some(held) => toi < held.toi || (toi == held.toi && slot < held.target),
    }
}

fn pose_of(body: &Body) -> Pose {
    Pose {
        position: body.position,
        rotation: body.orientation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::math::vec3;

    fn blocked(target: u32, toi: f32) -> Option<Blocked> {
        Some(Blocked {
            target,
            toi,
            normal: vec3(0.0, 1.0, 0.0),
            relative_motion: Vec3::ZERO,
        })
    }

    #[test]
    fn the_earliest_impact_wins_and_the_slot_breaks_a_tie() {
        assert!(nearer(&None, 9, 0.9));
        assert!(nearer(&blocked(3, 0.5), 7, 0.4));
        assert!(!nearer(&blocked(3, 0.5), 7, 0.6));
        assert!(nearer(&blocked(7, 0.5), 3, 0.5), "the lower slot holds it");
        assert!(!nearer(&blocked(3, 0.5), 7, 0.5));
    }
}
