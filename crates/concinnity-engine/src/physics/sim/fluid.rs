// Water pressing on a body from outside the solver: the body's pieces and
// the water's level over each at the start of a step, which every substep
// reads the body's own motion against. Re-reading the submerged volume at
// each substep's pose is what keeps a stiff float from gaining energy: the
// buoyancy is integrated like gravity, not held fixed through the step.

use crate::physics::sim::math::{Quat, Vec3};

/// The most pieces one body's water load is split into.
pub const MAX_FLUID_PIECES: usize = 8;

/// The shape of one piece the water acts on, in the body's frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FluidShape {
    /// A box of these half-extents along the body's axes.
    Cell {
        /// Half-extents along the body's axes.
        half_extents: [f32; 3],
    },
    /// A ball.
    Ball {
        /// Its radius.
        radius: f32,
    },
    /// Half a ball, its flat face on the piece's center and its dome
    /// toward `axis`.
    HalfBall {
        /// The ball's radius.
        radius: f32,
        /// Unit direction from the flat face through the dome, in the body's
        /// frame.
        axis: [f32; 3],
    },
}

/// One piece of a body the water acts on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FluidPiece {
    /// Its center, in the body's frame: a half ball's flat face.
    pub offset: [f32; 3],
    /// Its volume, in cubic meters.
    pub volume: f32,
    /// Its shape.
    pub shape: FluidShape,
}

/// The water around one body for one step: its pieces, a point of the
/// water's surface over each, which way is up out of the water, and how
/// dense the water is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FluidLoad {
    /// The pieces; the first `count` are used.
    pub pieces: [FluidPiece; MAX_FLUID_PIECES],
    /// How many pieces there are.
    pub count: usize,
    /// A point of the water's surface over each piece, in world units.
    pub levels: [[f32; 3]; MAX_FLUID_PIECES],
    /// Unit direction up out of the water.
    pub up: [f32; 3],
    /// The water's density, in kilograms per cubic meter.
    pub density: f32,
}

/// A piece's submerged part.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Submerged {
    /// Its volume, in cubic meters.
    pub volume: f32,
    /// How far below the piece's center its middle lies, along up.
    pub drop: f32,
    /// How fast the volume grows as the water rises, in square meters: the
    /// area the waterline cuts through the piece.
    pub waterline: f32,
}

/// How much of `piece`, its body turned by `rotation` (`[x, y, z, w]`), lies
/// under a surface `depth` meters above the piece's center along unit `up`.
/// It grows smoothly from nothing as the surface reaches the piece's lowest
/// point to all of it at its highest, so a body at the waterline feels no
/// step.
pub fn submerged(piece: &FluidPiece, rotation: [f32; 4], up: [f32; 3], depth: f32) -> Submerged {
    submerged_in(
        piece,
        Quat::from_xyzw(rotation),
        Vec3::from_array(up),
        depth,
    )
}

fn submerged_in(piece: &FluidPiece, rotation: Quat, up: Vec3, depth: f32) -> Submerged {
    match piece.shape {
        FluidShape::Cell { half_extents: h } => {
            // How far the box reaches along up either way: exact for a box
            // standing upright, close for one turned.
            let reach = (0..3)
                .map(|i| {
                    let mut axis = Vec3::ZERO;
                    axis.set(i, 1.0);
                    rotation.rotate(axis).dot(up).abs() * h[i]
                })
                .sum::<f32>()
                .max(1e-6);
            let share = ((depth + reach) / (2.0 * reach)).clamp(0.0, 1.0);
            let cut = share > 0.0 && share < 1.0;
            Submerged {
                volume: piece.volume * share,
                drop: reach * (1.0 - share),
                waterline: if cut {
                    piece.volume / (2.0 * reach)
                } else {
                    0.0
                },
            }
        }
        FluidShape::Ball { radius } => {
            let (share, drop) = ball(radius, depth);
            Submerged {
                volume: piece.volume * share,
                drop,
                waterline: piece.volume * ball_waterline(radius, depth),
            }
        }
        FluidShape::HalfBall { radius, axis } => {
            // Exact with the dome straight up, straight down and level (where
            // it is the ball's own curve), blended between by the tilt.
            let r = radius.max(1e-6);
            let tilt = rotation.rotate(Vec3::from_array(axis)).dot(up);
            let (level, drop) = ball(r, depth);
            let upright = if tilt >= 0.0 {
                let d = depth.clamp(0.0, r);
                1.0 - (r - d) * (r - d) * (2.0 * r + d) / (2.0 * r * r * r)
            } else {
                let h = (depth + r).clamp(0.0, r);
                h * h * (3.0 * r - h) / (2.0 * r * r * r)
            };
            let t = tilt.abs().min(1.0);
            // The level half ball's cut is half the ball's; upright, it is
            // the ball's on the dome's side of the flat face, none past it.
            let flat = ball_waterline(r, depth);
            let dome = if (tilt >= 0.0) == (depth >= 0.0) {
                2.0 * flat
            } else {
                0.0
            };
            Submerged {
                volume: piece.volume * (level + (upright - level) * t),
                drop,
                waterline: piece.volume * (flat + (dome - flat) * t),
            }
        }
    }
}

// The submerged share of a ball of radius `r` whose center lies `depth`
// under the surface, and how far below its center the submerged cap's
// middle lies.
fn ball(r: f32, depth: f32) -> (f32, f32) {
    let r = r.max(1e-6);
    let h = (depth + r).clamp(0.0, 2.0 * r);
    let share = h * h * (3.0 * r - h) / (4.0 * r * r * r);
    let drop = 3.0 * (2.0 * r - h) * (2.0 * r - h) / (4.0 * (3.0 * r - h)).max(1e-6);
    (share, drop)
}

// The waterline's cut through a ball of radius `r` whose center lies
// `depth` under the surface, per unit of the ball's volume.
fn ball_waterline(r: f32, depth: f32) -> f32 {
    let r = r.max(1e-6);
    let area = core::f32::consts::PI * (r * r - depth * depth).max(0.0);
    area / (4.0 / 3.0 * core::f32::consts::PI * r * r * r)
}

/// What the water does to a body at one pose: the force and the torque about
/// its center of mass, and how stiffly it resists the body sinking further
/// (newtons per meter) and turning (newton meters per radian), which the
/// solver integrates implicitly so no stiff float can gain energy.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FluidPush {
    pub(crate) force: Vec3,
    pub(crate) torque: Vec3,
    pub(crate) stiffness: f32,
    pub(crate) turn_stiffness: f32,
}

impl FluidLoad {
    /// The force and the torque about the center of mass the water puts on
    /// a body at `position` turned by `rotation`, the pull of gravity on it
    /// `pull` (meters per second squared, its gravity scale included).
    pub(crate) fn push_on(&self, position: Vec3, rotation: Quat, pull: Vec3) -> FluidPush {
        let up = Vec3::from_array(self.up);
        let lift = pull * -self.density;
        let weight = self.density * pull.length();
        let mut force = Vec3::ZERO;
        let mut torque = Vec3::ZERO;
        let mut stiffness = 0.0;
        let mut turn_stiffness = 0.0;
        for (piece, level) in self.pieces[..self.count.min(MAX_FLUID_PIECES)]
            .iter()
            .zip(&self.levels)
        {
            let at = position + rotation.rotate(Vec3::from_array(piece.offset));
            let depth = (Vec3::from_array(*level) - at).dot(up);
            let s = submerged_in(piece, rotation, up, depth);
            if s.volume <= 0.0 {
                continue;
            }
            let f = lift * s.volume;
            force += f;
            let arm = at - position;
            torque += (arm - up * s.drop).cross(f);
            let k = weight * s.waterline;
            stiffness += k;
            turn_stiffness += k * (arm - up * arm.dot(up)).length_squared();
        }
        FluidPush {
            force,
            torque,
            stiffness,
            turn_stiffness,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UP: [f32; 3] = [0.0, 1.0, 0.0];
    const LEVEL: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

    fn cell(h: f32) -> FluidPiece {
        FluidPiece {
            offset: [0.0; 3],
            volume: 8.0 * h * h * h,
            shape: FluidShape::Cell {
                half_extents: [h; 3],
            },
        }
    }

    fn half_ball(axis: [f32; 3]) -> FluidPiece {
        FluidPiece {
            offset: [0.0; 3],
            volume: 2.0 / 3.0 * core::f32::consts::PI,
            shape: FluidShape::HalfBall { radius: 1.0, axis },
        }
    }

    // An upright box submerges in proportion to its depth, and its middle
    // rises with the water.
    #[test]
    fn a_level_cell_submerges_linearly() {
        let c = cell(0.5);
        assert_eq!(submerged(&c, LEVEL, UP, -0.5).volume, 0.0);
        assert_eq!(submerged(&c, LEVEL, UP, 0.5).volume, c.volume);
        let half = submerged(&c, LEVEL, UP, 0.0);
        assert!((half.volume - 0.5 * c.volume).abs() < 1e-6);
        assert!((half.drop - 0.25).abs() < 1e-6);
        let mut last = -1.0;
        for i in 0..=20 {
            let v = submerged(&c, LEVEL, UP, -0.6 + 0.06 * i as f32).volume;
            assert!(v >= last);
            last = v;
        }
    }

    // A ball's cap is exact: half at its center, all once covered.
    #[test]
    fn a_ball_submerges_by_its_cap() {
        let v = 4.0 / 3.0 * core::f32::consts::PI * 8.0;
        let b = FluidPiece {
            offset: [0.0; 3],
            volume: v,
            shape: FluidShape::Ball { radius: 2.0 },
        };
        assert!((submerged(&b, LEVEL, UP, 0.0).volume - 0.5 * v).abs() < 1e-4);
        assert_eq!(submerged(&b, LEVEL, UP, 2.5).volume, v);
        assert_eq!(submerged(&b, LEVEL, UP, -2.5).volume, 0.0);
        // A cap one radius deep: pi h^2 (3r - h) / 3 at h = 1.
        let cap = submerged(&b, LEVEL, UP, -1.0).volume;
        assert!(
            (cap - core::f32::consts::PI * 5.0 / 3.0).abs() < 1e-4,
            "{cap}"
        );
    }

    // A half ball with its dome down is all under once the water reaches
    // its flat face, and with its dome up holds nothing until then; lying
    // level it fills like the ball it is half of.
    #[test]
    fn a_half_ball_fills_by_its_dome() {
        let down = half_ball([0.0, -1.0, 0.0]);
        let up = half_ball([0.0, 1.0, 0.0]);
        let level = half_ball([1.0, 0.0, 0.0]);
        let v = down.volume;
        assert!((submerged(&down, LEVEL, UP, 0.0).volume - v).abs() < 1e-5);
        assert_eq!(submerged(&down, LEVEL, UP, -1.0).volume, 0.0);
        assert_eq!(submerged(&up, LEVEL, UP, 0.0).volume, 0.0);
        assert!((submerged(&up, LEVEL, UP, 1.0).volume - v).abs() < 1e-5);
        assert!((submerged(&level, LEVEL, UP, 0.0).volume - 0.5 * v).abs() < 1e-5);
        // The dome's lower half, a cap half a radius tall: pi h^2 (3r - h) / 3.
        let cap = submerged(&down, LEVEL, UP, -0.5).volume;
        let want = core::f32::consts::PI * 0.25 * 2.5 / 3.0;
        assert!((cap - want).abs() < 1e-5, "{cap} vs {want}");
        for i in 0..=40 {
            let d = -1.2 + 0.06 * i as f32;
            for p in [down, up, level] {
                let s = submerged(&p, LEVEL, UP, d).volume;
                assert!((0.0..=v + 1e-6).contains(&s));
            }
        }
    }

    // A box on its edge reaches farther along up, so it takes longer to
    // cover.
    #[test]
    fn a_turned_cell_reaches_farther() {
        let c = cell(0.5);
        let half_angle = 0.5 * core::f32::consts::FRAC_PI_4;
        let tilted = [0.0, 0.0, half_angle.sin(), half_angle.cos()];
        let reach = 0.5 * core::f32::consts::SQRT_2;
        let at = submerged(&c, tilted, UP, reach - 1e-3);
        assert!(at.volume < c.volume && at.volume > 0.99 * c.volume);
    }

    // The push lifts against gravity by the water displaced, and turns a
    // body whose pieces sit unevenly in it.
    #[test]
    fn the_load_lifts_and_turns() {
        let mut pieces = [cell(0.5); MAX_FLUID_PIECES];
        pieces[0].offset = [-0.5, 0.0, 0.0];
        pieces[1].offset = [0.5, 0.0, 0.0];
        let load = FluidLoad {
            pieces,
            count: 2,
            // The water halfway up the left piece, over the top of the right.
            levels: [
                [0.0; 3],
                [0.0, 1.0, 0.0],
                [0.0; 3],
                [0.0; 3],
                [0.0; 3],
                [0.0; 3],
                [0.0; 3],
                [0.0; 3],
            ],
            up: UP,
            density: 1000.0,
        };
        let push = load.push_on(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::from_array([0.0, -10.0, 0.0]),
        );
        let (f, t) = (push.force, push.torque);
        // Only the left piece is cut by the waterline: one square meter.
        assert!(
            (push.stiffness - 1000.0 * 10.0).abs() < 1e-2,
            "{}",
            push.stiffness
        );
        assert!(
            (f.to_array()[1] - 1000.0 * 10.0 * 1.5).abs() < 1e-2,
            "{f:?}"
        );
        // More lift on the right turns the body toward -X down, about +Z.
        assert!(t.to_array()[2] > 0.0, "{t:?}");
    }
}
