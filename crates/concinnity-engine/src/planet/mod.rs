//! A planet world: the surface a [`Planet`] describes, the quadtree of tiles it
//! is drawn in, and the moving frame the world is simulated in so that
//! positions anywhere on it stay precise.
//!
//! The pieces are pure and independent of any thread or device:
//!
//! - [`PlanetShape`]: the ground as one function of direction, the single
//!   source every tile mesh and collider is built from, its land and seas
//!   laid out by [`Continents`].
//! - [`TileId`], [`tile_mesh`] and [`TileLod`]: the cube-sphere quadtree, one
//!   tile's mesh, and which tiles a camera wants and can draw, each
//!   level reaching as far as its cells look a set size on screen.
//! - [`LocalFrame`], [`Rebase`] and [`FrameRebases`]: where the simulated
//!   frame sits in double precision, the rigid move from one frame to the
//!   next, and the record of those moves every system holding positions
//!   catches up from.
//! - [`ground_patch`]: the ground around one point as a height grid in the
//!   simulated frame, for bodies to collide with.
//! - [`GeoPosition`] and [`GeoPose`]: latitude, longitude and height on the
//!   planet, and the conversions to and from the simulated frame.
//! - [`Horizon`]: the planet's lowest ground as an occluder.
//! - [`PlanetSystem`]: keeps the simulated frame, gravity and ground around
//!   the camera, building ground on a host's [`GroundBuilder`].

mod carry;
mod coast;
mod cover;
mod follow;
mod frame;
mod geo;
mod ground;
mod horizon;
mod lod;
mod morph;
mod rebase;
mod seams;
mod shape;
mod system;
mod tile;

use crate::components::TerrainLayer;
use crate::ecs::MaterialHandle;
use alloc::vec::Vec;
pub use carry::carry_world;
pub use coast::{BEACH_SLOPE, CONTINENT_OCTAVES, Continents, HILL_RISE, SEABED_RELIEF};
pub use concinnity_core::math::dmat3::DMat3;
pub use concinnity_core::math::dvec3::DVec3;
pub use cover::{GroundCover, MAX_GROUND_COVERS};
pub use follow::{PATCH_CELLS, PATCH_HALF_WIDTH, PatchPlan, PatchPurpose, build_patch, next_patch};
pub use frame::LocalFrame;
pub use geo::{CameraGeo, GeoPose, GeoPosition, geo_axes, geo_direction, geo_lat_lon};
pub use ground::{GroundPatch, ground_patch};
pub use horizon::{Horizon, HorizonOccluder};
pub use lod::{
    MAX_CELL_SAG, MAX_RANGE_WIDTHS, MIN_RANGE_WIDTHS, MORPH_START, RANGE_CELL_POINTS, REPLAN_DRIFT,
    RETAIN_RANGE, TileBoundsCache, TileLod, TileSelection, TileView, points_per_radian,
};
pub use morph::{TileMorphFrame, morph_factor, morph_point, packed_seams, parent_span};
pub use rebase::{FrameRebases, Rebase, RebaseCursor, needs_rebase};
pub use seams::{DrawnTile, TileSeams};
pub use shape::{CubeFace, MAX_OCTAVES, MIN_OCTAVES, PlanetShape};
pub(crate) use shape::{OCTAVE_TURN, octave_weight_sum};
pub use system::{
    GroundBuilder, InlineGround, PatchRequest, PatchResult, PlanetSystem, StartBuilder,
    inline_ground,
};
pub use tile::{
    MAX_TILE_LEVEL, TILE_CELLS, TileBounds, TileId, TileMesh, TileSide, tile_bounds, tile_mesh,
};

/// Smallest radius a planet keeps, in meters.
pub const MIN_PLANET_RADIUS: f32 = 100.0;

/// Most tiles a planet keeps resident at once, wherever the camera is: the
/// bound the GPU pools for its meshes are sized from.
pub const MAX_RESIDENT_TILES: usize = 4096;

/// The finest a tile mesh's cells get under the camera, in meters: the
/// deepest any planet's quadtree goes, at the finest the planet detail
/// setting allows.
pub const FINEST_CELL_SIZE: f64 = 1.0;

/// How far the camera strays from the simulated frame's origin before the
/// frame moves to it, in meters.
const REBASE_DISTANCE: f32 = 1_000.0;

/// How near the camera something stays drawn although the planet's horizon
/// hides it, in meters: twice the longest shadow distance a camera can be set
/// to, so a caster just below the horizon still shadows the ground in view.
pub const HORIZON_SHADOW_REACH: f64 = 640.0;

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

/// A whole world to stand on: a sphere of ground around `center`, raised
/// into hills and continents and sunk into ocean basins, rendered with
/// `material`, and pulling every body toward its center.
///
/// The ground is generated from `seed`: hills up to `amplitude` meters above
/// the radius, the broadest about `feature_size` meters across, with
/// `octaves` layers of finer detail on them, each half the width and half the
/// height of the one before. The same seed gives the same planet on every run.
/// Nothing about the ground is stored: it is built around the camera as it
/// moves, finer near it and coarser toward the horizon, so a planet costs the
/// same to load whatever its size.
///
/// With a `land_share` under 1, the planet divides into land and sea around
/// its `radius`, which becomes sea level: continents about `continent_size`
/// meters across rise above it over `land_share` of the surface, and ocean
/// basins up to `ocean_depth` meters deep sink below it over the rest. The
/// coasts are gentle: from the waterline a beach `beach_width` meters wide
/// rises a few meters, the hills grow in behind it over half their width,
/// and out to sea the floor falls away at the beach's slope across a shallow
/// shelf before it levels off toward the ocean depth. The ground then spans
/// from `ocean_depth` below the radius to `amplitude` above it, or to the top
/// of the beach where the hills are lower than that. The water
/// itself is a [`WaterSurface`](#watersurface) whose `shape` is `planet_sea`,
/// with a `sea_level` of 0 to fill the basins to the coasts.
///
/// Gravity points at the center from wherever a body is, `gravity` meters per
/// second squared, so a first-person camera with a [`RigidBody`](#rigidbody)
/// walks all the way around. Up is always away from the center: looking
/// around turns about it, and the horizon stays level.
///
/// Far from where the world started, positions stay as precise as they were
/// near it: the world is simulated around the camera, wherever on the planet
/// it is. A world has at most one planet.
///
/// The ground takes the look of `material`, overlaid in order by each of
/// `covers` where its rule holds, at most 4 of them: sand along the coasts,
/// rock on cliffs. The covers are drawn on a planet that grows grass in its
/// `layers`; one growing none shows its `material` alone. Each layer's own
/// rule keeps its blades off the covers they should not grow on.
///
/// Places on the planet are named by latitude and longitude, in degrees.
/// Latitude 0, longitude 0 is straight up from the center, where a world
/// whose center is below the origin starts; the north pole lies toward `-Z`
/// from the center, so a camera at the start facing `-Z` faces north, and
/// longitude grows eastward, toward `+X`. Altitude is measured above
/// `radius`, height above the ground above the hills.
///
/// ```rust
/// # use concinnity_engine::components::Planet;
/// Planet {
///     radius: 20_000.0,
///     center: [0.0, -20_000.0, 0.0],
///     amplitude: 35.0,
///     seed: 4,
///     land_share: 0.4,
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct Planet {
    /// World-space center. The default puts the top of a planet of the
    /// default radius at the origin.
    #[asset(default = [0.0, -50_000.0, 0.0])]
    pub center: [f32; 3],
    /// Radius of sea level, or of the lowest ground on a planet that is all
    /// land, in meters. At least 100.
    #[asset(default = 50_000.0)]
    pub radius: f32,
    /// Highest hilltop above `radius`, in meters. An `amplitude` of 0 gives a
    /// smooth sphere.
    #[asset(default = 40.0)]
    pub amplitude: f32,
    /// Width of the broadest hills, in meters.
    #[asset(default = 2_000.0)]
    pub feature_size: f32,
    /// Layers of ever finer detail on the hills. Clamped to [1, 16].
    #[asset(default = 8)]
    pub octaves: u32,
    /// Shapes the ground: each seed gives a different planet.
    pub seed: u32,
    /// Share of the surface that is land, in [0, 1]; the rest lies under sea
    /// level. 1 makes the whole planet land.
    #[asset(default = 1.0)]
    pub land_share: f32,
    /// Width of the broadest continents and oceans, in meters. At least 100.
    #[asset(default = 20_000.0)]
    pub continent_size: f32,
    /// Deepest the sea floor lies below sea level, in meters.
    #[asset(default = 60.0)]
    pub ocean_depth: f32,
    /// Width of the gentle rise from the waterline to where the hills begin,
    /// in meters. At least 1.
    #[asset(default = 60.0)]
    pub beach_width: f32,
    /// The [Material](#material) the ground renders with.
    pub material: Option<MaterialHandle>,
    /// Looks laid over `material` in order, each where its rule holds. At
    /// most 4; later ones are dropped.
    pub covers: Vec<GroundCover>,
    /// Pull toward the center, in meters per second squared.
    #[asset(default = crate::physics::GRAVITY)]
    pub gravity: f32,
    /// The grass that grows on the planet, one entry per look. Each grows
    /// over the whole planet as a meadow: soft-edged bare patches a couple of
    /// dozen meters across are laid over the ground, different for every
    /// layer, and the same every run. A layer's `density_mask` is not
    /// supported on a planet: a world that sets one fails validation.
    pub layers: Vec<TerrainLayer>,
}

impl Planet {
    /// The surface this planet's args describe.
    pub fn shape(&self) -> PlanetShape {
        let radius = f64::from(self.radius);
        PlanetShape {
            center: concinnity_core::math::dvec3::from_f32(self.center),
            radius,
            amplitude: f64::from(self.amplitude),
            feature_size: f64::from(self.feature_size),
            octaves: self.octaves,
            seed: self.seed,
            continents: Continents::new(
                radius,
                self.seed,
                f64::from(self.continent_size),
                f64::from(self.land_share),
                f64::from(self.ocean_depth),
                f64::from(self.beach_width),
            ),
        }
    }
}
