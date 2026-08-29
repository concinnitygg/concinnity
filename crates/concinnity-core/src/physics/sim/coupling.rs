// A manifold's normal constraints solved together instead of one after
// another.
//
// The points of one manifold are not independent. An impulse at one corner of
// a box spins the box about that corner, so it changes the approach speed at
// the other three, and a sweep that visits the points in turn leaves that for
// a later iteration to notice. Under a resting load the later iterations
// arrive in time. Under an impact they do not: the first corner alone is
// given the whole approach speed to answer, and the impulse that takes is
// large enough to throw the body outright, so the remaining corners spend the
// rest of the step failing to take back what it did.
//
// So the coupling is written down. `Coupling` holds what a unit impulse at
// each point does to the approach speed at every point, which turns the
// manifold into one small system with a non-negativity condition on each
// impulse -- a contact pushes and never pulls. Projected passes run over that
// system alone, touching no body until the manifold agrees with itself, and
// only the total each point settled on is applied.
//
// A face-on-face contact's matrix is well conditioned, so each pass takes
// roughly an order off what is left and a handful of them answer it. What
// decides how many run is how far the manifold is from agreeing rather than
// how hard it was hit: a contact already holding a resting load leaves on the
// first pass, and an impact stays for as long as it takes.

use super::contact::MAX_MANIFOLD_POINTS;
use super::math::Vec3;
use super::solver::SolverBody;

/// Projected passes one solve may run over a manifold's own system.
///
/// A ceiling rather than a target: the loop leaves as soon as a pass stops
/// moving anything, so this only bounds the cost of the contacts that need
/// the work. Fixed rather than left to a tolerance alone so that two runs of
/// the same scene take the same path.
const MAX_PASSES: usize = 8;

/// Approach speed below which another pass is not worth running, in world
/// units per second.
///
/// A speed, rather than an impulse or a fraction of one. What a leftover
/// impulse costs is the motion it fails to cancel, and a fraction of a large
/// impulse is still a large error, so a manifold is done once another pass
/// would change what its points are doing by less than this. That is what
/// lets a resting stack leave after a pass or two while a violent impact
/// stays for as long as it needs.
///
/// Set far under [`SimConfig::sleep_linear_velocity`], the speed at which the
/// world stops calling a body moving at all.
///
/// [`SimConfig::sleep_linear_velocity`]: crate::physics::SimConfig::sleep_linear_velocity
const SETTLED: f32 = 1.0e-4;

/// One normal impulse per point of a manifold, in the order it holds them.
pub(crate) type Impulses = [f32; MAX_MANIFOLD_POINTS];

/// What each of those points is being asked to answer, `None` where nothing
/// is being asked of it.
pub(crate) type Errors = [Option<f32>; MAX_MANIFOLD_POINTS];

fn dot(row: &Impulses, impulses: &Impulses) -> f32 {
    row.iter()
        .zip(impulses)
        .map(|(entry, impulse)| entry * impulse)
        .sum()
}

/// One body's part of a point's lever: the moment a unit normal impulse
/// applies, and the spin that moment produces.
#[derive(Debug, Clone, Copy, Default)]
struct Lever {
    moment: Vec3,
    spin: Vec3,
}

/// How much a unit normal impulse at each of a manifold's points changes the
/// approach speed at each of them, including itself.
///
/// Symmetric by construction, and stored whole so a row is read without index
/// arithmetic. The diagonal is the mass a point on its own would be solved
/// through, which is why the per-point normal mass lives here too.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Coupling {
    rows: [[f32; MAX_MANIFOLD_POINTS]; MAX_MANIFOLD_POINTS],
    /// Reciprocal of each row's diagonal, or zero where nothing can move.
    mass: [f32; MAX_MANIFOLD_POINTS],
}

impl Coupling {
    /// Build from each point's offsets to the two body centres, in the order
    /// the manifold holds them.
    pub(crate) fn build(
        a: &SolverBody,
        b: &SolverBody,
        normal: Vec3,
        anchors: &[(Vec3, Vec3)],
    ) -> Self {
        let count = anchors.len().min(MAX_MANIFOLD_POINTS);
        let mut levers = [(Lever::default(), Lever::default()); MAX_MANIFOLD_POINTS];
        for (lever, &(ra, rb)) in levers.iter_mut().zip(anchors).take(count) {
            let (ma, mb) = (ra.cross(normal), rb.cross(normal));
            *lever = (
                Lever {
                    moment: ma,
                    spin: a.inv_inertia.mul_vec3(ma),
                },
                Lever {
                    moment: mb,
                    spin: b.inv_inertia.mul_vec3(mb),
                },
            );
        }

        let linear = a.inv_mass + b.inv_mass;
        let mut coupling = Coupling {
            rows: [[0.0; MAX_MANIFOLD_POINTS]; MAX_MANIFOLD_POINTS],
            mass: [0.0; MAX_MANIFOLD_POINTS],
        };
        for i in 0..count {
            for j in 0..count {
                coupling.rows[i][j] = linear
                    + levers[i].0.moment.dot(levers[j].0.spin)
                    + levers[i].1.moment.dot(levers[j].1.spin);
            }
            let diagonal = coupling.rows[i][i];
            coupling.mass[i] = if diagonal > 0.0 { 1.0 / diagonal } else { 0.0 };
        }
        coupling
    }

    /// The approach speed `impulses` would produce at each point, which is
    /// what a softened pass measures its remembered impulse against.
    pub(crate) fn approach_from(&self, impulses: &Impulses) -> Impulses {
        let mut out = [0.0; MAX_MANIFOLD_POINTS];
        for (row, slot) in self.rows.iter().zip(&mut out) {
            *slot = dot(row, impulses);
        }
        out
    }

    /// Solve the manifold's normal impulses together.
    ///
    /// `held[i]` is what point `i` enters the solve holding, and `error[i]`
    /// the velocity error it is being asked to answer -- `None` for a point
    /// this solve leaves alone, which is also what the slots past the
    /// manifold's own points hold. The result is the change each point's
    /// impulse wants, with every total kept non-negative.
    pub(crate) fn solve(&self, held: &Impulses, error: &Errors) -> Impulses {
        // A point this solve leaves alone is given no mass rather than a
        // branch: it then answers every pass with no change of its own, and
        // still feels what the others do.
        let mut approach = [0.0; MAX_MANIFOLD_POINTS];
        let mut mass = [0.0; MAX_MANIFOLD_POINTS];
        for i in 0..MAX_MANIFOLD_POINTS {
            if let Some(error) = error[i] {
                approach[i] = error;
                mass[i] = self.mass[i];
            }
        }

        // The first pass is the answer a point-at-a-time sweep would have
        // reached; what the rest are for is the coupling it left behind.
        let mut delta = [0.0; MAX_MANIFOLD_POINTS];
        if self.sweep(&mut delta, approach, held, &mass) <= SETTLED {
            return delta;
        }
        for _ in 1..MAX_PASSES {
            // Re-derived from the impulses rather than carried on from what
            // the last pass left behind: an impact's impulses are large and
            // the speeds they cancel into are near zero, so starting each
            // pass from the impulses is what keeps that cancellation from
            // compounding.
            let mut asking = approach;
            for (asking, row) in asking.iter_mut().zip(&self.rows) {
                *asking += dot(row, &delta);
            }
            if self.sweep(&mut delta, asking, held, &mass) <= SETTLED {
                break;
            }
        }
        delta
    }

    /// One projected pass over the points, returning the largest approach
    /// speed any of them changed by.
    ///
    /// `approach` is what each point is still asking for as the pass enters
    /// it. A point that moves feeds its own column into every point's figure
    /// there and then, which is the same arithmetic as re-reading the row for
    /// each point but leaves the next one nothing to wait on. The arrays are
    /// the manifold's full width so the pass unrolls, since a slot past the
    /// points in play has no mass and a row of zeroes.
    fn sweep(
        &self,
        delta: &mut Impulses,
        mut approach: Impulses,
        held: &Impulses,
        mass: &Impulses,
    ) -> f32 {
        let mut moved = 0.0f32;
        for i in 0..MAX_MANIFOLD_POINTS {
            let carried = held[i] + delta[i];
            let total = (carried - approach[i] * mass[i]).max(0.0);
            let change = total - carried;
            delta[i] = total - held[i];
            // Symmetric, so the row is also the column this point acts down.
            for (asking, entry) in approach.iter_mut().zip(&self.rows[i]) {
                *asking += entry * change;
            }
            // What the change is worth as a speed, which is what says
            // whether another pass would be felt.
            moved = moved.max(change.abs() * self.rows[i][i]);
        }
        moved
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::sim::body::Body;
    use crate::physics::sim::math::{Quat, vec3};
    use crate::physics::{ColliderShape, DynamicParams, LayerMask};

    const HALF: f32 = 0.1;

    fn small_box() -> SolverBody {
        SolverBody::from_body(&Body::dynamic(
            ColliderShape::Cuboid {
                half_extents: [HALF, HALF, HALF],
            },
            vec3(0.0, HALF, 0.0),
            Quat::IDENTITY,
            DynamicParams {
                mass: 1.0,
                friction: 0.0,
                restitution: 0.0,
                gravity_scale: 1.0,
                linear_damping: 0.0,
            },
            LayerMask::ALL,
        ))
    }

    fn floor() -> SolverBody {
        SolverBody::from_body(&Body::fixed(
            ColliderShape::Cuboid {
                half_extents: [20.0, 5.0, 20.0],
            },
            vec3(0.0, -5.0, 0.0),
            Quat::IDENTITY,
            0.0,
            LayerMask::ALL,
        ))
    }

    const FLOOR_CENTRE: Vec3 = vec3(0.0, -5.0, 0.0);
    const BOX_CENTRE: Vec3 = vec3(0.0, HALF, 0.0);

    /// The four bottom corners of `small_box`, against an immovable floor, as
    /// the offsets to each body's centre the coupling is built from.
    fn corner_anchors() -> [(Vec3, Vec3); 4] {
        [(-HALF, -HALF), (HALF, -HALF), (HALF, HALF), (-HALF, HALF)].map(|(x, z)| {
            let point = vec3(x, 0.0, z);
            (point - FLOOR_CENTRE, point - BOX_CENTRE)
        })
    }

    fn corner_coupling() -> Coupling {
        Coupling::build(&floor(), &small_box(), Vec3::Y, &corner_anchors())
    }

    #[test]
    fn a_corner_patch_couples_symmetrically() {
        let coupling = corner_coupling();
        for i in 0..4 {
            for j in 0..4 {
                let (ij, ji) = (coupling.rows[i][j], coupling.rows[j][i]);
                assert!((ij - ji).abs() < 1.0e-4, "[{i}][{j}] {ij} against {ji}");
            }
        }
        // Opposite corners of a face push against each other: one corner's
        // impulse lifts the far corner rather than pressing it down.
        assert!(
            coupling.rows[0][2] < 0.0,
            "opposite corners must oppose: {}",
            coupling.rows[0][2]
        );
        assert!(
            coupling.rows[0][0] > 0.0 && coupling.mass[0] > 0.0,
            "a movable point has mass"
        );
    }

    // The whole reason the matrix exists: a face-on impact is answered by the
    // patch as a whole, so it stops the body without spinning it. The
    // point-at-a-time sweep hands the first corner the entire approach speed,
    // and the lever arm under it turns that into thousands of radians a
    // second no later point can take back.
    #[test]
    fn a_face_on_impact_stops_a_body_without_spinning_it() {
        let coupling = corner_coupling();
        let delta = coupling.solve(&[0.0; 4], &[Some(-2000.0); 4]);

        let total: f32 = delta.iter().sum();
        assert!((total - 2000.0).abs() < 1.0, "the patch delivered {total}");

        // Four corners of a face can hold the same load several ways -- an
        // impulse split across one diagonal is worth exactly what the same
        // impulse split across the other is -- so what has to be checked is
        // the torque the split leaves behind, not the split itself.
        let torque = corner_anchors()
            .iter()
            .zip(&delta)
            .fold(Vec3::ZERO, |sum, (&(_, rb), &impulse)| {
                sum + rb.cross(Vec3::Y * impulse)
            });
        assert!(
            torque.length() < 1.0,
            "the patch left {torque:?} of spin behind"
        );
    }

    #[test]
    fn a_solved_patch_leaves_no_approach_behind() {
        let coupling = corner_coupling();
        let held = [0.0; 4];
        let error = [Some(-2000.0), Some(-2000.0), Some(-1800.0), Some(-1800.0)];
        let delta = coupling.solve(&held, &error);
        let produced = coupling.approach_from(&delta);
        for i in 0..4 {
            let left = error[i].expect("every point is asked") + produced[i];
            assert!(left.abs() < 1.0, "point {i} still approaches at {left}");
        }
    }

    // A contact pushes and never pulls: a point being pulled apart takes no
    // impulse, and the rest of the manifold answers without it.
    #[test]
    fn a_separating_point_takes_nothing() {
        let coupling = corner_coupling();
        let held = [0.0; 4];
        let error = [Some(-10.0), Some(-10.0), Some(50.0), Some(50.0)];
        let delta = coupling.solve(&held, &error);
        assert_eq!((delta[2], delta[3]), (0.0, 0.0), "{delta:?}");
        assert!(delta[0] > 0.0 && delta[1] > 0.0, "{delta:?}");
    }

    #[test]
    fn a_point_the_pass_leaves_alone_does_not_move() {
        let coupling = corner_coupling();
        let held = [0.5; 4];
        let error = [Some(-10.0), None, Some(-10.0), None];
        let delta = coupling.solve(&held, &error);
        assert_eq!((delta[1], delta[3]), (0.0, 0.0), "{delta:?}");
    }

    // A single point has nothing to couple to, so the block answer has to be
    // the one the plain per-point rule would have given.
    #[test]
    fn a_lone_point_matches_the_per_point_answer() {
        let coupling = Coupling::build(&floor(), &small_box(), Vec3::Y, &corner_anchors()[..1]);
        let delta = coupling.solve(&[0.0; 4], &[Some(-5.0), None, None, None]);
        assert!(
            (delta[0] - 5.0 * coupling.mass[0]).abs() < 1.0e-4,
            "{delta:?}"
        );
    }

    #[test]
    fn an_immovable_pair_couples_nothing() {
        let coupling = Coupling::build(&floor(), &floor(), Vec3::Y, &corner_anchors());
        assert_eq!(coupling.mass[0], 0.0);
        assert_eq!(coupling.solve(&[0.0; 4], &[Some(-100.0); 4]), [0.0; 4]);
    }
}
