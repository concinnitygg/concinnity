// Asking the world a question without stepping it: where does this ray land,
// and how far does this shape get.
//
// Both questions are answered the same way. The broad phase hands back the
// window of proxies whose bounds the query's own bounds could reach, the
// window is filtered by layer and by the excluded body, and only what survives
// is measured exactly. The window is a slice of the sorted sweep order, so the
// traversal is a walk over an array with no hashing and no set anywhere in it,
// and two runs of the same query visit the same bodies in the same order.
//
// Nothing here allocates. A query keeps one hit, not a list, and every
// intermediate is a fixed-size array on the stack; the nearest hit is chosen
// by distance with the body slot breaking ties, so the answer does not even
// depend on the order the window happened to arrive in.
//
// Sensors are left out here, in the one filter every query runs through, so a
// region that records overlap never stops a ray, a sweep, or a character.
//
// The window is only as tight as one sorted axis can make it, which is a real
// limit rather than a tuning problem. A ray longer than the scene is wide, or
// a scene holding one proxy that spans it, leaves the window covering nearly
// every body, and each of those costs a bounds test. A hierarchy over the
// static set is what makes that case cheap, and it is a structure of its own
// rather than something this sorted array can be talked into.

pub(super) mod field;
pub(super) mod gjk;
mod ray;
mod simplex;
pub(super) mod sweep;

use crate::memory::Pool;

use crate::physics::{BodyHandle, ColliderShape, LayerMask, RayHit};

use super::aabb::shape_bounds;
use super::body::Body;
use super::collide::Pose;
use super::math::{Quat, Vec3};
use super::scene::Scene;

use gjk::Support;
use ray::{BoundsProbe, Ray};

/// A shape swept through the world along a straight line.
///
/// The sweep is a translation: the shape keeps the orientation it starts with
/// for the whole of `motion`. That is what a character move is, and it is what
/// makes the time of impact exact rather than bounded.
#[derive(Debug, Clone, Copy)]
pub struct ShapeCast {
    /// What to sweep.
    pub shape: ColliderShape,
    /// Where the shape's centre starts, in world space.
    pub origin: [f32; 3],
    /// The shape's orientation, held for the whole sweep.
    pub euler_deg: [f32; 3],
    /// The whole displacement to sweep along. A zero motion asks only whether
    /// the shape is already touching something.
    pub motion: [f32; 3],
    /// A body to leave out, usually the sweeping character's own.
    pub exclude: Option<BodyHandle>,
    /// Layers the sweep interacts with.
    pub mask: LayerMask,
}

impl ShapeCast {
    /// A sweep of `shape` from `origin` along `motion`, unrotated, hitting
    /// everything.
    pub fn new(shape: ColliderShape, origin: [f32; 3], motion: [f32; 3]) -> Self {
        ShapeCast {
            shape,
            origin,
            euler_deg: [0.0; 3],
            motion,
            exclude: None,
            mask: LayerMask::ALL,
        }
    }
}

/// What a [`ShapeCast`] ran into.
#[derive(Debug, Clone, Copy)]
pub struct ShapeCastHit {
    /// The body that was hit.
    pub body: BodyHandle,
    /// Fraction of the cast's `motion` covered before the contact, in
    /// `[0, 1]`. Multiply the motion by it to get the safe displacement.
    pub toi: f32,
    /// World-space contact point on the body that was hit.
    pub point: [f32; 3],
    /// Unit-length normal on the body that was hit, pointing back toward the
    /// swept shape. This is the direction to slide along.
    pub normal: [f32; 3],
    /// Distance between the two surfaces where the sweep stopped: zero or a
    /// hair positive for a shape that stopped short of the body, negative for
    /// one that began inside it. Separating along `normal` by `-gap` is what
    /// clears the overlap.
    pub gap: f32,
    /// Whether the shape was already touching this body before it moved. A
    /// caller that slides along `normal` has to separate first, or it will be
    /// handed the same zero-length move again.
    pub started_touching: bool,
}

/// What a raycast is asked. One struct rather than five arguments because the
/// filtering half of it is shared with [`ShapeCast`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct RayQuery {
    pub(crate) origin: [f32; 3],
    pub(crate) dir: [f32; 3],
    pub(crate) max_dist: f32,
    pub(crate) exclude: Option<BodyHandle>,
    pub(crate) mask: LayerMask,
}

/// The nearest ray hit, or `None`.
///
/// `dir` need not be unit length; a zero direction, a non-finite one, or a
/// non-positive `max_dist` all miss.
pub(crate) fn raycast(scene: Scene<'_>, ray_query: &RayQuery) -> Option<RayHit> {
    let max_dist = ray_query.max_dist;
    if !(max_dist.is_finite() && max_dist > 0.0) {
        return None;
    }
    let direction = Vec3::from_array(ray_query.dir);
    let length = direction.length();
    if !(length.is_finite() && length > 0.0) {
        return None;
    }
    let origin = Vec3::from_array(ray_query.origin);
    if !origin.is_finite() {
        return None;
    }
    let ray = Ray {
        origin,
        direction: direction * (1.0 / length),
    };
    let far = origin + ray.direction * max_dist;

    let axis = scene.broadphase.axis();
    let (low, high) = (
        origin.get(axis).min(far.get(axis)),
        origin.get(axis).max(far.get(axis)),
    );

    let probe = BoundsProbe::new(ray);
    // Shrinks to the nearest hit found so far: everything past it is out of
    // the running, and most of a long ray's window is past it.
    let mut reach = max_dist;
    let mut best: Option<(u32, RayHit)> = None;
    for &slot in scene.broadphase.slab_window(low, high) {
        let proxy = scene.broadphase.proxy(slot);
        if !ray_query.mask.interacts_with(proxy.mask) || !probe.reaches(proxy.bounds, reach) {
            continue;
        }
        let Some((_, body)) = candidate(scene.bodies, slot, ray_query.exclude) else {
            continue;
        };
        let found = match body.terrain_index() {
            Some(index) => field::raycast(scene.fields, index, ray, reach),
            None => body
                .convex()
                .and_then(|shape| ray::cast(ray, shape, pose_of(body), reach)),
        };
        let Some(impact) = found else {
            continue;
        };
        if nearer(
            best.map(|(kept, hit)| (kept, hit.distance)),
            slot,
            impact.distance,
        ) {
            reach = impact.distance;
            best = Some((
                slot,
                RayHit {
                    point: (origin + ray.direction * impact.distance).to_array(),
                    normal: impact.normal.to_array(),
                    distance: impact.distance,
                },
            ));
        }
    }
    best.map(|(_, hit)| hit)
}

/// The nearest body a swept shape runs into, or `None`.
pub(crate) fn shape_cast(scene: Scene<'_>, cast: &ShapeCast) -> Option<ShapeCastHit> {
    let origin = Vec3::from_array(cast.origin);
    let motion = Vec3::from_array(cast.motion);
    if !origin.is_finite() || !motion.is_finite() {
        return None;
    }
    let rotation = Quat::from_euler_deg(cast.euler_deg);
    let start = Pose {
        position: origin,
        rotation,
    };
    let moving = Support::new(&cast.shape, start);
    let start_bounds = shape_bounds(&cast.shape, origin, rotation);
    let travel =
        |toi: f32| start_bounds.union(shape_bounds(&cast.shape, origin + motion * toi, rotation));
    let swept = travel(1.0);

    let axis = scene.broadphase.axis();
    // Shrinks to the swept box the nearest contact leaves reachable, so a
    // sweep that stops early does not measure what it has already passed.
    let mut reach = swept;
    let mut best: Option<(u32, ShapeCastHit)> = None;
    for &slot in scene
        .broadphase
        .slab_window(swept.min.get(axis), swept.max.get(axis))
    {
        let proxy = scene.broadphase.proxy(slot);
        if !cast.mask.interacts_with(proxy.mask) || !reach.overlaps(proxy.bounds) {
            continue;
        }
        let Some((handle, body)) = candidate(scene.bodies, slot, cast.exclude) else {
            continue;
        };
        let found = match body.terrain_index() {
            Some(index) => field::sweep(scene.fields, index, &cast.shape, start, motion, reach),
            None => body.convex().and_then(|shape| {
                sweep::sweep(&moving, motion, &Support::new(shape, pose_of(body)))
            }),
        };
        let Some(impact) = found else {
            continue;
        };
        if nearer(best.map(|(kept, hit)| (kept, hit.toi)), slot, impact.toi) {
            reach = travel(impact.toi);
            best = Some((
                slot,
                ShapeCastHit {
                    body: handle,
                    toi: impact.toi,
                    point: impact.point.to_array(),
                    normal: impact.normal.to_array(),
                    gap: impact.gap,
                    started_touching: impact.started_touching,
                },
            ));
        }
    }
    best.map(|(_, hit)| hit)
}

/// The body at a slot, with its handle, once the cheap filters have let it
/// through.
fn candidate(
    bodies: &Pool<Body>,
    slot: u32,
    exclude: Option<BodyHandle>,
) -> Option<(BodyHandle, &Body)> {
    let handle = super::world::handle_at(bodies, slot)?;
    if exclude == Some(handle) {
        return None;
    }
    let body = bodies.get_at(slot as usize)?;
    // A region records what overlaps it rather than resisting it, so every
    // query passes straight through one.
    if body.is_sensor() {
        return None;
    }
    Some((handle, body))
}

/// Whether a fresh hit beats the one held. Distance decides; the body slot
/// breaks a tie, so the answer does not depend on traversal order.
fn nearer(best: Option<(u32, f32)>, slot: u32, measure: f32) -> bool {
    match best {
        None => true,
        Some((kept, held)) => measure < held || (measure == held && slot < kept),
    }
}

fn pose_of(body: &Body) -> Pose {
    Pose {
        position: body.position,
        rotation: body.orientation,
    }
}
