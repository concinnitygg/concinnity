//! What floating bodies push aside: each buoyant body's pieces as they sit in
//! the water at the last tick, which the renderer's wake turns into waves.
//! A body stirs the water by the change in what it displaces, so the physics
//! only reports where each piece is and how much water it holds back.

use crate::ecs::Entity;
use crate::physics::MAX_FLUID_PIECES;
use alloc::vec::Vec;
use concinnity_core::math::exp;

/// One piece of a floating body as it stirs the water.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StirPiece {
    /// The piece's center, in the simulated frame.
    pub at: [f32; 3],
    /// The water's surface over the piece's center, in the simulated frame.
    pub level: [f32; 3],
    /// The radius of a ball of the piece's whole volume, in meters: how wide
    /// a patch of water it pushes on.
    pub radius: f32,
    /// The water it holds back, in cubic meters: its submerged volume,
    /// weighed down the deeper it lies, so a piece far under the surface
    /// barely moves it.
    pub volume: f32,
}

impl StirPiece {
    /// A piece of `radius` at `at` whose center lies `depth` meters under
    /// the surface at `level` (negative above it), with `submerged` cubic
    /// meters of it in the water.
    pub fn new(at: [f32; 3], level: [f32; 3], radius: f32, depth: f32, submerged: f32) -> Self {
        let r = radius.max(1e-3);
        Self {
            at,
            level,
            radius: r,
            volume: submerged * exp(-(depth - r).max(0.0) / r),
        }
    }
}

/// One floating body's pieces at the last tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaterStir {
    /// The body's entity.
    pub entity: Entity,
    /// Up out of the water.
    pub up: [f32; 3],
    /// The body's velocity, in meters per second.
    pub velocity: [f32; 3],
    /// Its pieces; the first `count` are used.
    pub pieces: [StirPiece; MAX_FLUID_PIECES],
    /// How many pieces it has.
    pub count: usize,
}

impl WaterStir {
    /// The pieces in use.
    pub fn pieces(&self) -> &[StirPiece] {
        &self.pieces[..self.count.min(MAX_FLUID_PIECES)]
    }
}

/// Every floating body's stir after the frame's ticks, which the physics
/// refills each frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WaterStirs(pub Vec<WaterStir>);

#[cfg(test)]
mod tests {
    use super::*;

    // A piece holds back all of its submerged volume until its top is under,
    // then less and less the deeper it sinks.
    #[test]
    fn deep_pieces_hold_back_less() {
        let at_surface = StirPiece::new([0.0; 3], [0.0; 3], 0.5, 0.0, 0.2);
        assert_eq!(at_surface.volume, 0.2);
        let just_under = StirPiece::new([0.0; 3], [0.0; 3], 0.5, 0.5, 0.5);
        assert_eq!(just_under.volume, 0.5);
        let deep = StirPiece::new([0.0; 3], [0.0; 3], 0.5, 1.0, 0.5);
        assert!((deep.volume - 0.5 * exp(-1.0)).abs() < 1e-6);
        assert_eq!(
            StirPiece::new([0.0; 3], [0.0; 3], 0.0, 0.0, 0.0).radius,
            1e-3
        );
    }
}
