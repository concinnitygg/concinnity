//! A planet world: the surface a [`Planet`](crate::components::Planet)
//! describes, the quadtree of tiles it is drawn in, and the moving frame the
//! world is simulated in so that positions anywhere on it stay precise.
//!
//! The pieces are pure and independent of any thread or device:
//!
//! - [`PlanetShape`]: the ground as one function of direction, the single
//!   source every tile mesh and collider is built from.
//! - [`TileId`], [`tile_mesh`] and [`TileLod`]: the cube-sphere quadtree, one
//!   tile's skirted mesh, and which tiles a camera wants and can draw.
//! - [`LocalFrame`], [`Rebase`] and [`FrameRebases`]: where the simulated
//!   frame sits in double precision, the rigid move from one frame to the
//!   next, and the record of those moves every system holding positions
//!   catches up from.
//! - [`ground_patch`]: the ground around one point as a height grid in the
//!   simulated frame, for bodies to collide with.

mod dvec;
mod frame;
mod ground;
mod lod;
mod rebase;
mod shape;
mod tile;

pub use dvec::{DMat3, DVec3};
pub use frame::LocalFrame;
pub use ground::{GroundPatch, ground_patch};
pub use lod::{RETAIN_WIDTHS, SPLIT_WIDTHS, TileLod, TileSelection};
pub use rebase::{FrameRebases, Rebase, RebaseCursor, needs_rebase};
pub use shape::{CubeFace, MAX_OCTAVES, MIN_OCTAVES, PlanetShape};
pub use tile::{MAX_TILE_LEVEL, TILE_CELLS, TileBounds, TileId, TileMesh, tile_bounds, tile_mesh};

/// Smallest radius a planet keeps, in meters.
pub const MIN_PLANET_RADIUS: f32 = 100.0;

/// Most tiles a planet keeps resident at once, wherever the camera is: the
/// bound the GPU pools for its meshes are sized from.
pub const MAX_RESIDENT_TILES: usize = 960;

/// How far the camera strays from the simulated frame's origin before the
/// frame moves to it, in meters.
pub const REBASE_DISTANCE: f32 = 1_000.0;

/// An f32 point widened to double precision.
pub fn dvec_from_f32(v: [f32; 3]) -> DVec3 {
    dvec::from_f32(v)
}

/// Where a planet world's simulated frame is, and the planet it sits on:
/// published by the planet system for every system that places things on the
/// planet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanetFrame {
    /// The planet's surface.
    pub shape: PlanetShape,
    /// The frame the world is simulated in.
    pub frame: LocalFrame,
}

impl PlanetFrame {
    /// The planet's center in the simulated frame.
    pub fn center(&self) -> [f32; 3] {
        self.frame.to_local(self.shape.center)
    }

    /// Up at local point `p`: away from the planet's center.
    pub fn up_at(&self, p: [f32; 3]) -> [f32; 3] {
        let world = self.frame.to_world(p);
        self.frame.dir_to_local(self.shape.up_at(world))
    }
}

/// A planet's pull in the simulated frame: every body accelerates toward
/// `center` at `strength` meters per second squared.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanetGravity {
    /// The planet's center, in the simulated frame.
    pub center: [f32; 3],
    /// The acceleration's magnitude.
    pub strength: f32,
}

/// The ground bodies near the camera collide with: published by the planet
/// system each time it builds a new patch, `revision` counting them.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanetGround {
    /// Bumped with every new patch.
    pub revision: u64,
    /// The patch, in the simulated frame.
    pub patch: GroundPatch,
}
