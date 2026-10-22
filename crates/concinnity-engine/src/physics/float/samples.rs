// Where water is felt on a floating body: its collider split into a few
// pieces whose volumes sum to the collider's. A box is eight cells, a ball is
// one ball whose submerged cap is exact, and a capsule is its two end caps as
// half balls and its cylinder as two cells of the same cross-section.

use crate::physics::{ColliderShape, FluidPiece, FluidShape, MAX_FLUID_PIECES};
use core::f32::consts::PI;

/// A body's pieces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Samples {
    pieces: [FluidPiece; MAX_FLUID_PIECES],
    count: usize,
}

impl Samples {
    /// `shape` split into its pieces.
    pub(super) fn of(shape: &ColliderShape) -> Self {
        let empty = FluidPiece {
            offset: [0.0; 3],
            volume: 0.0,
            shape: FluidShape::Ball { radius: 0.0 },
        };
        let mut pieces = [empty; MAX_FLUID_PIECES];
        let count = match *shape {
            ColliderShape::Cuboid { half_extents } => {
                let h = half_extents.map(|e| 0.5 * e.max(0.0));
                let volume = 8.0 * h[0] * h[1] * h[2];
                for (i, piece) in pieces.iter_mut().enumerate() {
                    let side = |bit: usize, a: usize| if i & bit == 0 { -h[a] } else { h[a] };
                    *piece = FluidPiece {
                        offset: [side(1, 0), side(2, 1), side(4, 2)],
                        volume,
                        shape: FluidShape::Cell { half_extents: h },
                    };
                }
                MAX_FLUID_PIECES
            }
            ColliderShape::Ball { radius } => {
                let r = radius.max(0.0);
                pieces[0] = FluidPiece {
                    offset: [0.0; 3],
                    volume: ball_volume(r),
                    shape: FluidShape::Ball { radius: r },
                };
                1
            }
            ColliderShape::Capsule {
                half_height,
                radius,
            } => {
                let (r, hh) = (radius.max(0.0), half_height.max(0.0));
                // A square of the circle's area, half the cylinder tall.
                let side = 0.5 * r * concinnity_core::math::sqrt(PI);
                for (i, (y, out)) in [(-hh, -1.0), (hh, 1.0)].into_iter().enumerate() {
                    pieces[i] = FluidPiece {
                        offset: [0.0, y, 0.0],
                        volume: 0.5 * ball_volume(r),
                        shape: FluidShape::HalfBall {
                            radius: r,
                            axis: [0.0, out, 0.0],
                        },
                    };
                    pieces[2 + i] = FluidPiece {
                        offset: [0.0, 0.5 * y, 0.0],
                        volume: PI * r * r * hh,
                        shape: FluidShape::Cell {
                            half_extents: [side, 0.5 * hh, side],
                        },
                    };
                }
                4
            }
        };
        Self { pieces, count }
    }

    /// The pieces.
    pub(super) fn pieces(&self) -> &[FluidPiece] {
        &self.pieces[..self.count]
    }

    /// Every piece slot, the unused ones past [`Samples::pieces`].
    pub(super) fn slots(&self) -> &[FluidPiece; MAX_FLUID_PIECES] {
        &self.pieces
    }

    /// The volume of every piece together, in cubic meters.
    pub(super) fn volume(&self) -> f32 {
        self.pieces().iter().map(|p| p.volume).sum()
    }
}

/// A ball's volume.
pub(super) fn ball_volume(radius: f32) -> f32 {
    4.0 / 3.0 * PI * radius * radius * radius
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collider_volume(shape: &ColliderShape) -> f32 {
        match *shape {
            ColliderShape::Cuboid { half_extents: h } => 8.0 * h[0] * h[1] * h[2],
            ColliderShape::Ball { radius } => ball_volume(radius),
            ColliderShape::Capsule {
                half_height,
                radius,
            } => ball_volume(radius) + PI * radius * radius * 2.0 * half_height,
        }
    }

    // The pieces stand for the whole collider, balanced about its center.
    #[test]
    fn the_pieces_fill_the_collider() {
        for shape in [
            ColliderShape::Cuboid {
                half_extents: [1.0, 0.4, 2.5],
            },
            ColliderShape::Ball { radius: 0.7 },
            ColliderShape::Capsule {
                half_height: 0.9,
                radius: 0.3,
            },
        ] {
            let s = Samples::of(&shape);
            let want = collider_volume(&shape);
            assert!((s.volume() - want).abs() < 1e-4 * want, "{shape:?}");
            let mut moment = [0.0f32; 3];
            for p in s.pieces() {
                for (m, o) in moment.iter_mut().zip(p.offset) {
                    *m += o * p.volume;
                }
            }
            assert!(moment.iter().all(|m| m.abs() < 1e-5), "{shape:?}");
        }
        let ball = Samples::of(&ColliderShape::Ball { radius: 1.0 });
        assert_eq!(ball.pieces().len(), 1);
        let cells = Samples::of(&ColliderShape::Cuboid {
            half_extents: [1.0; 3],
        });
        assert_eq!(cells.pieces().len(), MAX_FLUID_PIECES);
    }
}
