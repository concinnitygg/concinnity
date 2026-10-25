// Where a ray meets a planet's ground: a march along the ray against the
// height function the tiles are built from, each step as long as the ground
// below can be near, then a bisection onto the surface.

use super::PlanetFrame;
use super::geo::GeoPosition;
use super::shape::PlanetShape;
use concinnity_core::math::dvec3;
use concinnity_core::math::dvec3::DVec3;

/// Where a ray met a planet's ground, in authored coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundHit {
    /// The ground point.
    pub point: DVec3,
    /// The ground's outward unit normal there.
    pub normal: DVec3,
    /// How far along the ray the point lies, in meters.
    pub distance: f64,
}

/// Where a ray from the simulated frame met a planet's ground.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundPick {
    /// Where the ground point is on the planet.
    pub geo: GeoPosition,
    /// The ground point in the simulated frame.
    pub point: [f32; 3],
    /// The ground's outward unit normal in the simulated frame.
    pub normal: [f32; 3],
    /// How far along the ray the point lies, in meters.
    pub distance: f64,
}

// Steps the march takes before calling the ray a miss.
const MAX_STEPS: u32 = 2048;
// Halvings the narrowing takes at most.
const MAX_BISECTIONS: u32 = 64;
// How close the narrowing gets to the surface along the ray, in meters.
const HIT_TOLERANCE: f64 = 1e-4;
// The shortest step, in meters and as a share of the distance travelled.
const MIN_STEP: f64 = 1e-3;
const MIN_STEP_SHARE: f64 = 1e-4;
// Room above the highest ground the march starts in, in meters.
const SHELL_MARGIN: f64 = 1.0;

impl PlanetShape {
    /// Where the ray from authored point `origin` along unit direction `dir`
    /// first meets the ground within `max_distance` meters. `None` when it
    /// misses, or when `origin` lies below the ground.
    pub fn ray_hit(&self, origin: DVec3, dir: DVec3, max_distance: f64) -> Option<GroundHit> {
        let rel = dvec3::sub(origin, self.center);
        let top = self.radius + self.highest_height() + SHELL_MARGIN;
        let (enter, exit) = sphere_span(rel, dir, top)?;
        let end = exit.min(max_distance);
        let mut t = enter.max(0.0);
        if t > end {
            return None;
        }
        let at = |t: f64| dvec3::add(rel, dvec3::scale(dir, t));
        let mut above = self.clearance(at(t));
        if above <= 0.0 {
            return None;
        }
        // How fast the clearance can shrink per meter along any ray.
        let lipschitz = 1.0 + self.max_slope() * self.radius / self.lowest_radius().max(1.0);
        for _ in 0..MAX_STEPS {
            if t >= end {
                return None;
            }
            let step = (above / lipschitz).max(MIN_STEP).max(t * MIN_STEP_SHARE);
            let next = (t + step).min(end);
            let next_above = self.clearance(at(next));
            if next_above <= 0.0 {
                let t = self.narrow(rel, dir, t, next);
                let (_, normal) = self.surface_and_normal(dvec3::normalize_or(at(t), dir));
                return Some(GroundHit {
                    point: dvec3::add(self.center, at(t)),
                    normal,
                    distance: t,
                });
            }
            t = next;
            above = next_above;
        }
        None
    }

    // How far the point `rel` from the center stands above the ground under
    // it, in meters, negative below it.
    fn clearance(&self, rel: DVec3) -> f64 {
        let r = dvec3::length(rel);
        r - (self.radius + self.height(dvec3::normalize_or(rel, [0.0, 1.0, 0.0])))
    }

    // The distance along the ray where it crosses the ground, between `lo`
    // above it and `hi` at or below it.
    fn narrow(&self, rel: DVec3, dir: DVec3, mut lo: f64, mut hi: f64) -> f64 {
        for _ in 0..MAX_BISECTIONS {
            if hi - lo <= HIT_TOLERANCE {
                break;
            }
            let mid = 0.5 * (lo + hi);
            if self.clearance(dvec3::add(rel, dvec3::scale(dir, mid))) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        hi
    }
}

impl PlanetFrame {
    /// Where the ray from local point `origin` along local direction `dir`
    /// first meets the ground within `max_distance` meters: the march runs in
    /// authored coordinates in double precision, so it holds anywhere on the
    /// planet. `None` when it misses or starts below the ground.
    pub fn pick_ground(
        &self,
        origin: [f32; 3],
        dir: [f32; 3],
        max_distance: f64,
    ) -> Option<GroundPick> {
        let dir = dvec3::normalize_or(self.frame.dir_to_world(dir), [0.0; 3]);
        if dir == [0.0; 3] {
            return None;
        }
        let hit = self
            .shape
            .ray_hit(self.frame.to_world(origin), dir, max_distance)?;
        Some(GroundPick {
            geo: self.shape.geo_at(hit.point),
            point: self.frame.to_local(hit.point),
            normal: self.frame.dir_to_local(hit.normal),
            distance: hit.distance,
        })
    }
}

// Where the ray from `rel` (relative to a sphere's center) along unit `dir`
// is inside the sphere of `radius`: the distances it enters and leaves at,
// the entry negative when it starts inside.
fn sphere_span(rel: DVec3, dir: DVec3, radius: f64) -> Option<(f64, f64)> {
    let b = dvec3::dot(rel, dir);
    // The squared distance from the center to the ray's closest point, taken
    // from the perpendicular part so it holds far from the sphere.
    let perp = dvec3::sub(rel, dvec3::scale(dir, b));
    let gap = radius * radius - dvec3::dot(perp, perp);
    if gap < 0.0 {
        return None;
    }
    let half = libm::sqrt(gap);
    let exit = -b + half;
    (exit >= 0.0).then_some((-b - half, exit))
}

#[cfg(test)]
mod tests;
