//! The brush's outline on the ground: a closed ring of points a set distance
//! along the ground from the brush's center, each set on the planet's height
//! function and lifted a little off it, in the simulated frame.

use concinnity_core::math::dvec3;
use concinnity_engine::planet::{DVec3, PlanetFrame, geo_axes};

/// The points one ring is drawn through.
pub(crate) const RING_POINTS: usize = 64;

/// Lay into `out` the ring `radius` meters along the ground around the
/// ground under unit direction `center`, each point `lift` meters above the
/// ground.
pub(crate) fn lay(
    planet: &PlanetFrame,
    center: DVec3,
    radius: f64,
    lift: f64,
    out: &mut [[f32; 3]; RING_POINTS],
) {
    let shape = &planet.shape;
    let [east, north, up] = geo_axes(center);
    let angle = radius / (shape.radius + shape.height(up)).max(1.0);
    let (sin_a, cos_a) = angle.sin_cos();
    for (k, point) in out.iter_mut().enumerate() {
        let turn = std::f64::consts::TAU * k as f64 / RING_POINTS as f64;
        let (sin_t, cos_t) = turn.sin_cos();
        let side = dvec3::add(dvec3::scale(east, cos_t), dvec3::scale(north, sin_t));
        let dir = dvec3::add(dvec3::scale(up, cos_a), dvec3::scale(side, sin_a));
        let r = shape.radius + shape.height(dir) + lift;
        *point = planet
            .frame
            .to_local(dvec3::add(shape.center, dvec3::scale(dir, r)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::planet::{Continents, LocalFrame, PlanetShape};

    fn planet(frame: LocalFrame) -> PlanetFrame {
        PlanetFrame {
            shape: PlanetShape {
                center: [0.0, -50_000.0, 0.0],
                radius: 50_000.0,
                amplitude: 40.0,
                feature_size: 2_000.0,
                octaves: 6,
                seed: 9,
                continents: Continents::NONE,
            },
            frame,
        }
    }

    // Every point sits `lift` over the ground, `radius` along it from the
    // center, and the ring goes once around.
    #[test]
    fn the_ring_follows_the_ground_at_the_brush_radius() {
        let pf = planet(LocalFrame::AUTHORED);
        let shape = pf.shape;
        let center = dvec3::normalize_or([0.1, 1.0, -0.05], [0.0; 3]);
        let mut out = [[0.0; 3]; RING_POINTS];
        lay(&pf, center, 12.0, 0.25, &mut out);
        for p in out {
            let world = pf.frame.to_world(p);
            let dir = shape.up_at(world);
            let over =
                dvec3::length(dvec3::sub(world, shape.center)) - (shape.radius + shape.height(dir));
            assert!((over - 0.25).abs() < 0.01, "{over}");
            let along = dvec3::length(dvec3::cross(center, dir)).atan2(dvec3::dot(center, dir))
                * (shape.radius + shape.height(center));
            assert!((along - 12.0).abs() < 0.01, "{along}");
        }
        let gap = |a: [f32; 3], b: [f32; 3]| {
            dvec3::length(dvec3::sub(dvec3::from_f32(a), dvec3::from_f32(b)))
        };
        let step = std::f64::consts::TAU * 12.0 / RING_POINTS as f64;
        for k in 0..RING_POINTS {
            let d = gap(out[k], out[(k + 1) % RING_POINTS]);
            assert!((d - step).abs() < 0.05 * step + 0.05, "{d} {step}");
        }
    }

    // In a frame moved far around the planet the ring still lands about the
    // local origin, where the brush is.
    #[test]
    fn the_ring_lands_in_a_far_frame() {
        let mut frame = LocalFrame::AUTHORED;
        let base = planet(frame);
        for _ in 0..30 {
            let at = [800.0, 0.0, 500.0];
            frame = frame.recentered(at, base.shape.up_at(frame.to_world(at))).0;
        }
        let pf = planet(frame);
        let center = pf.shape.up_at(frame.to_world([0.0; 3]));
        let mut out = [[0.0; 3]; RING_POINTS];
        lay(&pf, center, 5.0, 0.0, &mut out);
        for p in out {
            let flat = (p[0] * p[0] + p[2] * p[2]).sqrt();
            assert!((flat - 5.0).abs() < 0.2, "{p:?}");
        }
    }
}
