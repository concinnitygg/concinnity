//! The sphere a planet's ground stays under: about the planet's center, out to
//! its highest ground. A ray that rises out of it meets no more ground.

use crate::environment::planet::PlanetFrame;
use crate::gfx::render_types::GroundBoundGpu;
use concinnity_core::math::{dvec3, narrow_f32, vec3};

/// The GPU record of no ground at all.
pub const NO_GROUND_BOUND: GroundBoundGpu = GroundBoundGpu {
    sphere: [0.0; 4],
    camera: [0.0; 4],
};

/// The bound of `planet`'s ground for a camera at simulated `camera`, moved by
/// `offset` into the space the frame draws in, with the camera's height over
/// it formed in double precision; [`NO_GROUND_BOUND`] without a planet.
pub fn ground_bound(
    planet: Option<&PlanetFrame>,
    camera: [f32; 3],
    offset: [f32; 3],
) -> GroundBoundGpu {
    let Some(planet) = planet else {
        return NO_GROUND_BOUND;
    };
    let shape = &planet.shape;
    let radius = shape.radius + shape.highest_height();
    let [x, y, z] = vec3::add(planet.center(), offset);
    let reach = dvec3::length(dvec3::sub(planet.frame.to_world(camera), shape.center));
    GroundBoundGpu {
        sphere: [x, y, z, narrow_f32(radius)],
        camera: [narrow_f32(reach - radius), 0.0, 0.0, 0.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::planet::LocalFrame;

    fn planet() -> PlanetFrame {
        PlanetFrame {
            shape: crate::components::Planet::default().shape(),
            frame: LocalFrame::AUTHORED,
        }
    }

    #[test]
    fn the_bound_reaches_the_highest_ground() {
        let planet = planet();
        let shape = &planet.shape;
        let top = shape.radius + shape.highest_height();
        assert!(shape.highest_height() > 0.0);
        let bound = ground_bound(Some(&planet), [0.0; 3], [0.0; 3]);
        assert_eq!(bound.sphere[3], narrow_f32(top));
        assert_eq!(bound.sphere[..3], planet.center());
    }

    // On an Earth-sized planet single precision steps by half a meter, so a
    // camera a few meters over the ground keeps its height only in double.
    #[test]
    fn the_camera_height_is_formed_in_double_precision() {
        let planet = PlanetFrame {
            shape: crate::components::Planet {
                center: [0.0, -6.371e6, 0.0],
                radius: 6.371e6,
                ..crate::components::Planet::default()
            }
            .shape(),
            frame: LocalFrame::AUTHORED,
        };
        let shape = &planet.shape;
        let radius = shape.radius + shape.highest_height();
        let top = shape.center[1] + radius;
        let camera = [0.0, narrow_f32(top) + 12.3, 0.0];
        let bound = ground_bound(Some(&planet), camera, [0.0; 3]);
        let expected = f64::from(camera[1]) - top;
        assert!((f64::from(bound.camera[0]) - expected).abs() < 0.01);
        let single = (camera[1] - narrow_f32(shape.center[1])) - narrow_f32(radius);
        assert!((f64::from(single) - expected).abs() > 0.01);
    }

    #[test]
    fn the_offset_moves_the_center_into_the_drawn_space() {
        let planet = planet();
        let still = ground_bound(Some(&planet), [0.0; 3], [0.0; 3]);
        let moved = ground_bound(Some(&planet), [0.0; 3], [1.0, -2.0, 3.0]);
        assert_eq!(
            moved.sphere[..3],
            vec3::add(planet.center(), [1.0, -2.0, 3.0])
        );
        assert_eq!(moved.sphere[3], still.sphere[3]);
        assert_eq!(moved.camera, still.camera);
    }

    #[test]
    fn a_world_without_a_planet_has_no_ground_bound() {
        assert_eq!(ground_bound(None, [0.0; 3], [0.0; 3]), NO_GROUND_BOUND);
    }
}
