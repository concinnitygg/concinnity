// concinnity-physics/src/sim/character/resolve.rs
//
// One character move, resolved by sweeping and deflecting rather than by
// simulating. A character is not a rigid body: it has to stop dead against a
// wall, climb a kerb it would otherwise trip on, and stay stuck to the ground
// over a lip, none of which a solver produces from mass and impulses.
//
// The whole of it is sweeps against the same storage the step reads, so this
// mutates nothing and can be asked between steps or during one. What it costs
// is a handful of casts per move: one to see whether the mover starts on the
// ground, up to a few for the deflections, three more if it tries a step, and
// one to decide where it ended up. Every one of them is bounded, and nothing
// here allocates -- the state of a move is this function's locals.

use crate::{BodyHandle, CharacterMove, CharacterMoveInput, ColliderShape, LayerMask};

use crate::sim::math::{Vec3, vec3};
use crate::sim::query::{self, RayQuery, ShapeCast, ShapeCastHit};
use crate::sim::scene::Scene;

use super::capsule::CharacterCapsule;
use super::config::CharacterConfig;
use super::slide;

/// Deflections one move may take. A wedge cannot be resolved by sliding, so
/// this is what stops a mover from circling between two walls forever; a move
/// that runs out keeps the ground it has covered and drops the rest.
const MAX_DEFLECTIONS: usize = 5;

/// How far clear of a surface a mover is held after stopping against it. Above
/// the gap a sweep counts as a touch, so the next sweep starts apart from the
/// surface rather than flush against it and stopping at once.
const CONTACT_OFFSET: f32 = 5.0e-4;

/// How far below a mover ground is looked for. Wide enough to find the
/// surface a mover was just set down on, tight enough that a mover in the air
/// is in the air.
const GROUND_PROBE: f32 = 0.02;

/// Motion shorter than this is nothing left to resolve.
const MIN_MOTION: f32 = 1.0e-6;

/// The least a step has to gain, forward or upward, to count as one. A step
/// that gains no ground has its obstacle still in the way at the raised
/// height, which is what a wall looks like; one that gains no height is the
/// far side of something the mover would have been hopping over.
const STEP_PROGRESS: f32 = 4.0 * CONTACT_OFFSET;

/// Resolve `input` against the bodies, without moving any of them.
pub(crate) fn resolve(
    scene: Scene<'_>,
    config: &CharacterConfig,
    capsule: &CharacterCapsule,
    input: &CharacterMoveInput,
) -> CharacterMove {
    let center = Vec3::from_array(input.center);
    let desired = Vec3::from_array(input.desired);
    if !is_finite(center) || !is_finite(desired) {
        return CharacterMove {
            translation: [0.0; 3],
            grounded: false,
        };
    }
    Mover {
        scene,
        config: *config,
        shape: capsule.shape(),
        radius: libm::fabsf(capsule.radius()),
        foot: libm::fabsf(capsule.half_height()) + libm::fabsf(capsule.radius()),
        exclude: Some(input.exclude),
        mask: input.mask,
    }
    .run(center, desired)
}

/// One mover and the scene it is resolved against, so the sweeps a move takes
/// carry their filter rather than passing it along at every call.
struct Mover<'a> {
    scene: Scene<'a>,
    config: CharacterConfig,
    shape: ColliderShape,
    radius: f32,
    /// Distance from the capsule's centre to its lowest point.
    foot: f32,
    exclude: Option<BodyHandle>,
    mask: LayerMask,
}

/// Where climbing an obstacle left the mover, and what is left of the move.
struct Step {
    position: Vec3,
    remaining: Vec3,
}

/// Ground found under the mover: where it comes to rest on it, and how far
/// down that is.
#[derive(Clone, Copy)]
struct Ground {
    at: Vec3,
    drop: f32,
}

impl Mover<'_> {
    fn run(&self, center: Vec3, desired: Vec3) -> CharacterMove {
        // Whether the mover began the move on the ground, or close enough to
        // it to still be attached. Auto-step and ground-snap are both gated on
        // it, so neither fires on a character in mid-air, and both are gated
        // on gravity governing the mover at all, which is what makes a
        // free-flying camera do neither.
        let footed = self.config.grounded && self.ground_below(center).is_some();

        let mut position = center;
        let mut remaining = desired;
        let mut met: Option<Vec3> = None;
        let mut stepped = false;

        for _ in 0..MAX_DEFLECTIONS {
            if remaining.length_squared() <= MIN_MOTION * MIN_MOTION {
                break;
            }
            let Some(hit) = self.cast(position, remaining) else {
                position += remaining;
                break;
            };
            let normal = Vec3::from_array(hit.normal);
            position += remaining * hit.toi + normal * clearance(hit.gap);
            remaining = remaining * (1.0 - hit.toi);

            let walkable = self.config.is_walkable(normal);
            if !walkable
                && footed
                && !stepped
                && let Some(step) = self.step_over(position, remaining)
            {
                stepped = true;
                position = step.position;
                remaining = step.remaining;
                met = None;
                continue;
            }

            let slid = slide::deflect(remaining, normal, walkable);
            remaining = match met {
                Some(previous) if slide::re_entrant(slid, previous) => {
                    slide::crease(remaining, previous, normal)
                }
                _ => slid,
            };
            met = Some(normal);
        }

        // One probe answers both questions: ground within reach of the
        // capsule's feet is what it is standing on, and ground further down
        // than that is what a mover that walked off a lip stays attached to
        // rather than launching off. A mover on its way up is jumping, and a
        // jump is left alone.
        let ground = self.ground_below(position);
        let mut grounded = ground.is_some_and(|found| found.drop <= GROUND_PROBE);
        if !grounded
            && footed
            && desired.y <= 0.0
            && let Some(found) = ground
        {
            position = found.at;
            grounded = true;
        }

        CharacterMove {
            translation: (position - center).to_array(),
            grounded,
        }
    }

    /// Climb onto whatever blocked the move, if it is low enough to climb,
    /// there is room above it, and there is somewhere to come down on.
    fn step_over(&self, position: Vec3, remaining: Vec3) -> Option<Step> {
        let height = self.config.step_height;
        let ahead = slide::horizontal(remaining);
        let asked = ahead.length();
        if height <= 0.0 || asked <= MIN_MOTION {
            return None;
        }

        // Rise a hair past the limit, so a lip exactly `step_height` tall is
        // cleared rather than caught on.
        let room = height + CONTACT_OFFSET;
        let lift = match self.cast(position, vec3(0.0, room, 0.0)) {
            Some(hit) => room * hit.toi - CONTACT_OFFSET,
            None => room,
        };
        if lift <= CONTACT_OFFSET {
            return None;
        }
        let lifted = position + vec3(0.0, lift, 0.0);

        // The capsule stopped with its surface against the obstacle, so
        // standing on top of it means carrying its axis a radius further than
        // the move itself asked for.
        let probe = ahead * ((asked + self.radius + CONTACT_OFFSET) / asked);
        let advance = match self.cast(lifted, probe) {
            Some(hit) => probe * hit.toi,
            None => probe,
        };
        if advance.length_squared() <= STEP_PROGRESS * STEP_PROGRESS {
            return None;
        }

        // Nothing to come down on means the mover stepped over a hole rather
        // than onto a lip, and a landing too steep to stand on is not a step
        // either.
        let landing = lifted + advance;
        let stepped = self.set_down(landing, lift + CONTACT_OFFSET)?.at;
        if stepped.y - position.y <= STEP_PROGRESS {
            return None;
        }
        Some(Step {
            position: stepped,
            remaining: leftover(ahead, advance),
        })
    }

    /// The nearest body a sweep of the mover's capsule along `motion` runs
    /// into.
    fn cast(&self, from: Vec3, motion: Vec3) -> Option<ShapeCastHit> {
        query::shape_cast(
            self.scene,
            &ShapeCast {
                shape: self.shape,
                origin: from.to_array(),
                euler_deg: [0.0; 3],
                motion: motion.to_array(),
                exclude: self.exclude,
                mask: self.mask,
            },
        )
    }

    /// The ground under the mover, out to the furthest it stays attached: what
    /// it steps onto is what it stays attached to stepping off.
    fn ground_below(&self, from: Vec3) -> Option<Ground> {
        self.set_down(from, self.config.step_height.max(GROUND_PROBE))
    }

    /// Where the mover comes to rest set down from `from`, and `None` when
    /// there is no ground within `reach` under it.
    fn set_down(&self, from: Vec3, reach: f32) -> Option<Ground> {
        let hit = self.cast(from, vec3(0.0, -reach, 0.0))?;
        let standing = self.config.is_walkable(Vec3::from_array(hit.normal))
            || self.ground_under_foot(from, reach);
        standing.then(|| Ground {
            at: settled(from, reach, &hit),
            drop: reach * hit.toi,
        })
    }

    /// Whether there is walkable ground within `reach` of the mover's lowest
    /// point, straight below its axis.
    ///
    /// A capsule rounding a ledge is stopped by the ledge itself, at a normal
    /// too steep to stand on, while the ground it is stepping down onto is
    /// right below it. The two look the same to a sweep and differ under the
    /// axis, which is what tells stepping down from falling.
    fn ground_under_foot(&self, from: Vec3, reach: f32) -> bool {
        query::raycast(
            self.scene,
            &RayQuery {
                origin: from.to_array(),
                dir: [0.0, -1.0, 0.0],
                max_dist: self.foot + reach,
                exclude: self.exclude,
                mask: self.mask,
            },
        )
        .is_some_and(|hit| self.config.is_walkable(Vec3::from_array(hit.normal)))
    }
}

/// How far along a contact normal a mover has to move to be clear of the
/// surface it stopped against: out of whatever it overlaps, plus the offset
/// that keeps the next sweep from starting flush against it.
///
/// A sweep that stopped short of a body reports a gap of zero or a hair more,
/// and gets only the offset. A move that began touching, or a little inside,
/// reports a negative gap and is separated by it rather than having the
/// contact ignored -- which is what keeps a capsule spawned exactly on the
/// floor from sinking a little every tick.
fn clearance(gap: f32) -> f32 {
    if gap < 0.0 {
        CONTACT_OFFSET - gap
    } else {
        CONTACT_OFFSET
    }
}

/// Where a downward probe of `reach` leaves the mover: on the surface it
/// found, held clear of it.
fn settled(from: Vec3, reach: f32, hit: &ShapeCastHit) -> Vec3 {
    let normal = Vec3::from_array(hit.normal);
    from + vec3(0.0, -reach * hit.toi, 0.0) + normal * clearance(hit.gap)
}

/// What is left of a horizontal move once a step has carried part of it. A
/// step that reached further than the move asked for leaves nothing: the
/// remainder must not turn round and walk the mover back off the lip.
fn leftover(ahead: Vec3, advance: Vec3) -> Vec3 {
    let left = ahead.length() - advance.length();
    if left <= MIN_MOTION {
        return Vec3::ZERO;
    }
    ahead.normalize_or_zero() * left
}

fn is_finite(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sweep_that_stopped_short_is_only_held_off_the_surface() {
        assert_eq!(clearance(0.0), CONTACT_OFFSET);
        assert_eq!(clearance(1.0e-5), CONTACT_OFFSET);
    }

    #[test]
    fn a_move_that_began_inside_is_separated_by_what_it_overlaps() {
        assert!((clearance(-0.1) - (0.1 + CONTACT_OFFSET)).abs() < 1.0e-6);
    }

    #[test]
    fn a_step_that_reached_past_the_move_leaves_nothing_of_it() {
        let ahead = vec3(0.0, 0.0, 0.05);
        assert_eq!(leftover(ahead, vec3(0.0, 0.0, 0.35)), Vec3::ZERO);
        assert_eq!(leftover(ahead, ahead), Vec3::ZERO);
        let left = leftover(ahead, vec3(0.0, 0.0, 0.02));
        assert!((left - vec3(0.0, 0.0, 0.03)).length() < 1.0e-6, "{left:?}");
    }

    #[test]
    fn a_probe_leaves_the_mover_on_what_it_found() {
        let hit = ShapeCastHit {
            body: BodyHandle::from_parts(0, 0),
            toi: 0.5,
            point: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            gap: 0.0,
            started_touching: false,
        };
        let landed = settled(vec3(0.0, 2.0, 0.0), 1.0, &hit);
        assert!(
            (landed.y - (1.5 + CONTACT_OFFSET)).abs() < 1.0e-6,
            "{landed:?}"
        );
    }
}
