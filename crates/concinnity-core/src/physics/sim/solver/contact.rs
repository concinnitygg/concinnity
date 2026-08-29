// The contact rows a step solves, and the passes that move them.
//
// Each pass walks a run of constraints and answers them one after another, so
// a body's velocity carries what the constraints before it decided. That is
// the sequential in sequential impulses, and it is preserved exactly: a run is
// one or more whole islands, and constraints keep the order the manifolds gave
// them within each.
//
// What a constraint holds beyond its rows is the two figures a step reports
// with. The impulse a point ends holding is one substep's worth, which is what
// the next step warm starts from; the impulse it delivered is the whole step's,
// accumulated as it is applied, which is what an impact is measured by.

use super::bodies::Bodies;
use crate::math::sqrt;
use crate::physics::sim::config::{SimConfig, Softness};
use crate::physics::sim::contact::{MAX_MANIFOLD_POINTS, Manifold};
use crate::physics::sim::coupling::Coupling;
use crate::physics::sim::math::Vec3;

use super::bodies::SolverBody;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ConstraintPoint {
    /// Offset from each body's centre to the contact, as of the top of the
    /// step. Rotated by the body's delta rotation as the substeps advance.
    anchor_a: Vec3,
    anchor_b: Vec3,
    /// The separation the anchors imply at zero displacement, so the current
    /// separation is this plus how far the bodies have moved apart.
    base_separation: f32,
    tangent_mass: [f32; 2],
    normal_impulse: f32,
    tangent_impulse: [f32; 2],
    max_normal_impulse: f32,
    /// Normal impulse delivered over the whole step, accumulated as it is
    /// applied. The per-substep figure above is what the next step starts
    /// from; this is what the step actually did.
    step_impulse: f32,
    /// Approach speed before the step, which is what restitution bounces off.
    approach_velocity: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ContactConstraint {
    a: u32,
    b: u32,
    manifold: u32,
    normal: Vec3,
    tangent: [Vec3; 2],
    friction: f32,
    restitution: f32,
    /// What one point's normal impulse does to the approach speed at every
    /// other, so the manifold is answered as one system rather than a point
    /// at a time.
    coupling: Coupling,
    points: [ConstraintPoint; MAX_MANIFOLD_POINTS],
    count: u8,
}

impl ContactConstraint {
    /// Whether this slot holds rows to solve. A manifold that produced no
    /// points still gets a slot, so the grouping can be counted before the
    /// contacts are built, and every pass then steps over it.
    pub(crate) fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Index into the manifold list the step solved.
    pub(crate) fn manifold(&self) -> u32 {
        self.manifold
    }

    /// Total normal impulse this contact delivered over the step.
    pub(crate) fn delivered(&self) -> f32 {
        self.points[..self.count as usize]
            .iter()
            .map(|point| point.step_impulse)
            .sum()
    }

    /// Hand the impulses back to the manifold they came from, so next step's
    /// warm start begins where this one left off.
    pub(crate) fn store(&self, manifolds: &mut [Manifold]) {
        let manifold = &mut manifolds[self.manifold as usize];
        for (point, stored) in manifold
            .points_mut()
            .iter_mut()
            .zip(&self.points[..self.count as usize])
        {
            point.normal_impulse = stored.normal_impulse;
            point.tangent_impulse = stored.tangent_impulse;
        }
    }
}

/// Build one run of constraints from the manifolds the partition gave it.
pub(crate) fn prepare(
    constraints: &mut [ContactConstraint],
    source: &[u32],
    manifolds: &[Manifold],
    bodies: &Bodies<'_>,
) {
    for (constraint, &index) in constraints.iter_mut().zip(source) {
        let manifold = &manifolds[index as usize];
        let (a, b) = (manifold.a, manifold.b);
        let (body_a, body_b) = (*bodies.get(a), *bodies.get(b));
        let normal = manifold.normal;
        let tangent0 = normal.any_perpendicular();
        let tangent1 = normal.cross(tangent0);

        let mut anchors = [(Vec3::ZERO, Vec3::ZERO); MAX_MANIFOLD_POINTS];
        let mut count = 0usize;
        for (anchor, point) in anchors.iter_mut().zip(manifold.points()) {
            *anchor = (point.point - body_a.position, point.point - body_b.position);
            count += 1;
        }
        if count == 0 {
            *constraint = ContactConstraint {
                manifold: index,
                ..ContactConstraint::default()
            };
            continue;
        }

        *constraint = ContactConstraint {
            a,
            b,
            manifold: index,
            normal,
            tangent: [tangent0, tangent1],
            // The two materials are already combined on the manifold, so the
            // solve never reaches back into body storage while it holds the
            // dense arrays.
            friction: manifold.friction,
            restitution: manifold.restitution,
            coupling: Coupling::build(&body_a, &body_b, normal, &anchors[..count]),
            points: [ConstraintPoint::default(); MAX_MANIFOLD_POINTS],
            count: count as u8,
        };

        for ((slot, point), &(ra, rb)) in constraint
            .points
            .iter_mut()
            .zip(manifold.points())
            .zip(&anchors)
        {
            *slot = ConstraintPoint {
                anchor_a: ra,
                anchor_b: rb,
                base_separation: point.separation - (rb - ra).dot(normal),
                tangent_mass: [
                    effective_mass(&body_a, &body_b, ra, rb, tangent0),
                    effective_mass(&body_a, &body_b, ra, rb, tangent1),
                ],
                normal_impulse: point.normal_impulse,
                tangent_impulse: point.tangent_impulse,
                max_normal_impulse: 0.0,
                step_impulse: 0.0,
                approach_velocity: (body_b.velocity_at(rb) - body_a.velocity_at(ra)).dot(normal),
            };
        }
    }
}

/// Re-apply what every point was holding, so a resting stack begins each
/// substep already carrying its own weight.
pub(crate) fn warm_start(constraints: &mut [ContactConstraint], bodies: &mut Bodies<'_>) {
    for constraint in constraints.iter_mut() {
        let normal = constraint.normal;
        let [t0, t1] = constraint.tangent;
        let (a, b) = (constraint.a, constraint.b);
        for point in &mut constraint.points[..constraint.count as usize] {
            let impulse = normal * point.normal_impulse
                + t0 * point.tangent_impulse[0]
                + t1 * point.tangent_impulse[1];
            let ra = bodies.get(a).delta_rotation.rotate(point.anchor_a);
            let rb = bodies.get(b).delta_rotation.rotate(point.anchor_b);
            point.step_impulse += point.normal_impulse;
            bodies.get_mut(a).apply_impulse(-impulse, ra);
            bodies.get_mut(b).apply_impulse(impulse, rb);
        }
    }
}

/// The normal impulses of one manifold, then its friction.
///
/// The normals go together. Each point's velocity error is measured first and
/// the manifold's own system is solved for all of them at once, so what reaches
/// a body is the total the points agreed on rather than the first point's guess
/// with the rest correcting after it.
pub(crate) fn solve(
    constraints: &mut [ContactConstraint],
    bodies: &mut Bodies<'_>,
    soft: &Softness,
    config: &SimConfig,
    inv_h: f32,
    use_bias: bool,
) {
    for constraint in constraints.iter_mut() {
        let normal = constraint.normal;
        let [t0, t1] = constraint.tangent;
        let (a, b) = (constraint.a, constraint.b);
        let count = constraint.count as usize;

        let mut arms = [(Vec3::ZERO, Vec3::ZERO); MAX_MANIFOLD_POINTS];
        let mut held = [0.0; MAX_MANIFOLD_POINTS];
        for (index, point) in constraint.points[..count].iter().enumerate() {
            arms[index] = (
                bodies.get(a).delta_rotation.rotate(point.anchor_a),
                bodies.get(b).delta_rotation.rotate(point.anchor_b),
            );
            held[index] = point.normal_impulse;
        }
        // What the impulses already held are worth in approach speed, which is
        // what the softened pass forgets its share of. Only that pass remembers
        // anything, so the relax pass after it does not pay for this.
        let carried = if use_bias {
            constraint.coupling.approach_from(&held)
        } else {
            [0.0; MAX_MANIFOLD_POINTS]
        };

        let mut error = [None; MAX_MANIFOLD_POINTS];
        for (index, point) in constraint.points[..count].iter().enumerate() {
            let (ra, rb) = arms[index];
            let travel = (bodies.get(b).delta_position + rb) - (bodies.get(a).delta_position + ra);
            let separation = travel.dot(normal) + point.base_separation;

            let (bias, mass_scale, impulse_scale) = if separation > 0.0 {
                // A gap the bodies are still closing: allow exactly enough
                // approach to touch this substep and no more.
                (separation * inv_h, 1.0, 0.0)
            } else if use_bias {
                (
                    (soft.bias_rate * separation).max(-config.max_push_velocity),
                    soft.mass_scale,
                    soft.impulse_scale,
                )
            } else {
                (0.0, 1.0, 0.0)
            };

            let approach =
                (bodies.get(b).velocity_at(rb) - bodies.get(a).velocity_at(ra)).dot(normal);
            error[index] = Some(mass_scale * (approach + bias) + impulse_scale * carried[index]);
        }

        let solved = constraint.coupling.solve(&held, &error);
        for (index, point) in constraint.points[..count].iter_mut().enumerate() {
            let applied = solved[index];
            point.normal_impulse += applied;
            point.max_normal_impulse = point.max_normal_impulse.max(point.normal_impulse);
            point.step_impulse += applied;

            let (ra, rb) = arms[index];
            let impulse = normal * applied;
            bodies.get_mut(a).apply_impulse(-impulse, ra);
            bodies.get_mut(b).apply_impulse(impulse, rb);
        }

        for point in &mut constraint.points[..count] {
            let ra = bodies.get(a).delta_rotation.rotate(point.anchor_a);
            let rb = bodies.get(b).delta_rotation.rotate(point.anchor_b);
            let relative = bodies.get(b).velocity_at(rb) - bodies.get(a).velocity_at(ra);

            let mut wanted = [
                point.tangent_impulse[0] - point.tangent_mass[0] * relative.dot(t0),
                point.tangent_impulse[1] - point.tangent_mass[1] * relative.dot(t1),
            ];
            // Friction is a disc, not a square: clamp the two tangents together
            // so a diagonal slide is no easier than a straight one.
            let limit = constraint.friction * point.normal_impulse;
            let magnitude = sqrt(wanted[0] * wanted[0] + wanted[1] * wanted[1]);
            if magnitude > limit {
                let scale = if magnitude > f32::MIN_POSITIVE {
                    limit / magnitude
                } else {
                    0.0
                };
                wanted[0] *= scale;
                wanted[1] *= scale;
            }
            let impulse = t0 * (wanted[0] - point.tangent_impulse[0])
                + t1 * (wanted[1] - point.tangent_impulse[1]);
            point.tangent_impulse = wanted;

            bodies.get_mut(a).apply_impulse(-impulse, ra);
            bodies.get_mut(b).apply_impulse(impulse, rb);
        }
    }
}

/// Bounce, from the speed measured before the step. Doing it last means the
/// bounce is not fighting the penetration correction.
pub(crate) fn apply_restitution(
    constraints: &mut [ContactConstraint],
    bodies: &mut Bodies<'_>,
    config: &SimConfig,
) {
    let threshold = config.restitution_threshold;
    for constraint in constraints.iter_mut() {
        if constraint.restitution <= 0.0 {
            continue;
        }
        let normal = constraint.normal;
        let (a, b) = (constraint.a, constraint.b);
        let count = constraint.count as usize;

        let mut arms = [(Vec3::ZERO, Vec3::ZERO); MAX_MANIFOLD_POINTS];
        let mut held = [0.0; MAX_MANIFOLD_POINTS];
        let mut error = [None; MAX_MANIFOLD_POINTS];
        for (index, point) in constraint.points[..count].iter().enumerate() {
            let (ra, rb) = (
                bodies.get(a).delta_rotation.rotate(point.anchor_a),
                bodies.get(b).delta_rotation.rotate(point.anchor_b),
            );
            arms[index] = (ra, rb);
            held[index] = point.normal_impulse;
            // A slow approach does not bounce, or a settling body never stops;
            // a point that carried no load has nothing to bounce.
            if point.approach_velocity > -threshold || point.max_normal_impulse == 0.0 {
                continue;
            }
            let approach =
                (bodies.get(b).velocity_at(rb) - bodies.get(a).velocity_at(ra)).dot(normal);
            error[index] = Some(approach + constraint.restitution * point.approach_velocity);
        }
        if error[..count].iter().all(Option::is_none) {
            continue;
        }

        let solved = constraint.coupling.solve(&held, &error);
        for (index, point) in constraint.points[..count].iter_mut().enumerate() {
            let applied = solved[index];
            point.normal_impulse += applied;
            point.step_impulse += applied;
            let (ra, rb) = arms[index];
            let impulse = normal * applied;
            bodies.get_mut(a).apply_impulse(-impulse, ra);
            bodies.get_mut(b).apply_impulse(impulse, rb);
        }
    }
}

/// Mass seen along `direction` at the contact: the linear part plus what the
/// lever arms make of the angular part.
pub(crate) fn effective_mass(
    a: &SolverBody,
    b: &SolverBody,
    ra: Vec3,
    rb: Vec3,
    direction: Vec3,
) -> f32 {
    let angular_a = ra.cross(direction);
    let angular_b = rb.cross(direction);
    let k = a.inv_mass
        + b.inv_mass
        + angular_a.dot(a.inv_inertia.mul_vec3(angular_a))
        + angular_b.dot(b.inv_inertia.mul_vec3(angular_b));
    if k > 0.0 { 1.0 / k } else { 0.0 }
}
