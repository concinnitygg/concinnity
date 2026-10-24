//! A planet's per-point sky as the view blocks carry it: the daylight curves
//! and the lights that ride them.

use crate::render::environment::atmosphere::{DAYLIGHT_KNOTS, knot_coord};
use concinnity_core::math::SaturatingFrom;

/// The directional lights the curves light: the first (the shadowed one) and
/// the second. `DAYLIGHT_BODIES` in `daylight_types.hlsl`.
pub const DAYLIGHT_BODIES: usize = 2;

/// The largest view block every backend binds with its draws.
pub(super) const VIEW_BLOCK_LIMIT: usize = 4096;

/// A body lighting the planet as one of the first directional lights.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DaylightBody {
    /// Unit direction toward it.
    pub direction: [f32; 3],
    /// Its light above the air: color times intensity.
    pub light: [f32; 3],
}

/// Where this frame's view stands toward the planet, and what lights it.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DaylightView {
    /// The planet's center in the simulated frame, in meters.
    pub center: [f32; 3],
    /// Unit up where the camera is.
    pub camera_up: [f32; 3],
    /// How much each point's own sky stands in for the camera's wherever it
    /// is, from the camera's height: 0 on the ground, 1 high above it.
    pub camera_reach: f32,
    /// The cosines between a point's up and the camera's at which the
    /// point's own sky starts to take over and has.
    pub reach: [f32; 2],
    /// The ground's radius, in meters.
    pub ground_radius: f32,
    /// The height over the ground, in meters, below which a point is lit as
    /// the ground, fading out over its upper half.
    pub ground_height: f32,
    /// The bodies lighting as the first two directional lights, in order.
    pub bodies: [Option<DaylightBody>; DAYLIGHT_BODIES],
}

/// The per-point sky of a planet with an atmosphere. Matches `DaylightBlock`
/// in `daylight_types.hlsl`; [`OFF`](Self::OFF) for any other world, whose
/// passes then shade exactly as without it.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct DaylightBlock {
    /// xyz: the planet's center; w: 1 when the curves light the view.
    pub center: [f32; 4],
    /// xyz: up where the camera is; w: [`DaylightView::camera_reach`].
    pub camera_up: [f32; 4],
    /// xy: [`DaylightView::reach`]; z, w: [`DaylightView::ground_radius`] and
    /// [`DaylightView::ground_height`].
    pub reach: [f32; 4],
    /// xyz: toward body i; w: 1 when there is one.
    pub body_dir: [[f32; 4]; DAYLIGHT_BODIES],
    /// rgb: body i's light above the air.
    pub body_light: [[f32; 4]; DAYLIGHT_BODIES],
    /// Pads the header to a whole number of rows.
    pub _pad: [f32; 4],
    /// The packed curves (`DaylightCurves::packed`).
    pub knots: [[f32; 4]; DAYLIGHT_KNOTS * 3],
}

impl Default for DaylightBlock {
    fn default() -> Self {
        Self::OFF
    }
}

impl DaylightBlock {
    /// No per-point sky.
    pub const OFF: Self = Self {
        center: [0.0; 4],
        camera_up: [0.0; 4],
        reach: [0.0; 4],
        body_dir: [[0.0; 4]; DAYLIGHT_BODIES],
        body_light: [[0.0; 4]; DAYLIGHT_BODIES],
        _pad: [0.0; 4],
        knots: [[0.0; 4]; DAYLIGHT_KNOTS * 3],
    };

    /// The block for `view` over the packed curves `knots`.
    pub fn new(knots: [[f32; 4]; DAYLIGHT_KNOTS * 3], view: &DaylightView) -> Self {
        let [cx, cy, cz] = view.center;
        let [ux, uy, uz] = view.camera_up;
        let mut block = Self {
            center: [cx, cy, cz, 1.0],
            camera_up: [ux, uy, uz, view.camera_reach],
            reach: [
                view.reach[0],
                view.reach[1],
                view.ground_radius,
                view.ground_height,
            ],
            knots,
            ..Self::OFF
        };
        for (i, body) in view.bodies.iter().enumerate() {
            if let Some(b) = body {
                let [x, y, z] = b.direction;
                let [r, g, bl] = b.light;
                block.body_dir[i] = [x, y, z, 1.0];
                block.body_light[i] = [r, g, bl, 0.0];
            }
        }
        block
    }

    /// Whether the curves light the view.
    pub fn is_on(&self) -> bool {
        self.center[3] > 0.5
    }

    /// The sky's light on a level surface where the camera is, from every
    /// body. Mirrors `daylight_level_irradiance` in `daylight.hlsl`.
    pub fn camera_irradiance(&self) -> [f32; 3] {
        let up = [self.camera_up[0], self.camera_up[1], self.camera_up[2]];
        let mut e = [0.0; 3];
        for (dir, light) in self.body_dir.iter().zip(&self.body_light) {
            if dir[3] <= 0.5 {
                continue;
            }
            let mu = up[0] * dir[0] + up[1] * dir[1] + up[2] * dir[2];
            let irradiance = self.knot_row(mu, 1);
            for (e, (l, i)) in e.iter_mut().zip(light.iter().zip(irradiance)) {
                *e += l * i;
            }
        }
        e
    }

    // Row `lane` of the knots at cosine `mu`, linear between knots. Mirrors
    // `daylight_at` in `daylight.hlsl`.
    fn knot_row(&self, mu: f32, lane: usize) -> [f32; 3] {
        let x = knot_coord(mu);
        let i = usize::saturating_from(x).min(DAYLIGHT_KNOTS - 2);
        let t = x - i as f32;
        let a = self.knots[i * 3 + lane];
        let b = self.knots[i * 3 + 3 + lane];
        core::array::from_fn(|c| a[c] + (b[c] - a[c]) * t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    #[test]
    fn daylight_block_layout_matches_the_shader() {
        assert_eq!(size_of::<DaylightBlock>(), 1664);
        assert_eq!(offset_of!(DaylightBlock, center), 0);
        assert_eq!(offset_of!(DaylightBlock, camera_up), 16);
        assert_eq!(offset_of!(DaylightBlock, reach), 32);
        assert_eq!(offset_of!(DaylightBlock, body_dir), 48);
        assert_eq!(offset_of!(DaylightBlock, body_light), 80);
        assert_eq!(offset_of!(DaylightBlock, _pad), 112);
        assert_eq!(offset_of!(DaylightBlock, knots), 128);
    }

    #[test]
    fn the_constants_match_the_shader() {
        use crate::render::shader::consts::uint;
        let src = crate::render::shader::embedded::DAYLIGHT_TYPES;
        assert_eq!(uint(src, "DAYLIGHT_BODIES"), DAYLIGHT_BODIES);
    }

    // Off is all zero, so a world without a planet uploads what it always
    // did beside it, and the shader's switch reads it as off.
    #[test]
    fn off_lights_nothing() {
        assert!(!DaylightBlock::OFF.is_on());
        assert!(
            bytemuck::bytes_of(&DaylightBlock::OFF)
                .iter()
                .all(|&b| b == 0)
        );
    }

    // The camera's sky reads each lit body's irradiance at its height over
    // the camera's horizon, between knots, scaled by its light.
    #[test]
    fn the_camera_irradiance_reads_each_body_between_knots() {
        let knots = core::array::from_fn(|row| {
            let v = (row / 3) as f32;
            if row % 3 == 1 {
                [v, 2.0 * v, 0.0, 0.0]
            } else {
                [0.0; 4]
            }
        });
        let body = |direction| {
            Some(DaylightBody {
                direction,
                light: [1.0, 1.0, 3.0],
            })
        };
        let view = |bodies| DaylightView {
            center: [0.0; 3],
            camera_up: [0.0, 1.0, 0.0],
            camera_reach: 0.0,
            reach: [0.9999, 0.998],
            ground_radius: 1.0,
            ground_height: 1.0,
            bodies,
        };
        let top = (DAYLIGHT_KNOTS - 1) as f32;
        let overhead = DaylightBlock::new(knots, &view([body([0.0, 1.0, 0.0]), None]));
        assert_eq!(overhead.camera_irradiance(), [top, 2.0 * top, 0.0]);
        // On the horizon, halfway along the knots; two bodies add.
        let both = DaylightBlock::new(knots, &view([body([1.0, 0.0, 0.0]), body([0.0, 1.0, 0.0])]));
        let e = both.camera_irradiance();
        assert!((e[0] - (0.5 * top + top)).abs() < 1e-4, "{e:?}");
        assert_eq!(DaylightBlock::OFF.camera_irradiance(), [0.0; 3]);
    }

    #[test]
    fn a_view_fills_its_bodies_in_order() {
        let block = DaylightBlock::new(
            [[0.5; 4]; DAYLIGHT_KNOTS * 3],
            &DaylightView {
                center: [1.0, -50_000.0, 2.0],
                camera_up: [0.0, 1.0, 0.0],
                camera_reach: 0.25,
                reach: [0.9999, 0.998],
                ground_radius: 50_000.0,
                ground_height: 5_000.0,
                bodies: [
                    None,
                    Some(DaylightBody {
                        direction: [0.0, 0.0, 1.0],
                        light: [3.0, 2.0, 1.0],
                    }),
                ],
            },
        );
        assert!(block.is_on());
        assert_eq!(block.center, [1.0, -50_000.0, 2.0, 1.0]);
        assert_eq!(block.camera_up[3], 0.25);
        assert_eq!(block.reach, [0.9999, 0.998, 50_000.0, 5_000.0]);
        assert_eq!(block.body_dir[0][3], 0.0);
        assert_eq!(block.body_dir[1], [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(block.body_light[1], [3.0, 2.0, 1.0, 0.0]);
        assert_eq!(block.knots[5], [0.5; 4]);
    }
}
