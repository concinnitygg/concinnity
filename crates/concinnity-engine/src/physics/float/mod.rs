// Bodies floating in the world's water: each tick, every buoyant body's
// pieces are read against the surface the water draws, and the water's level
// over each goes to the simulation, whose substeps buoy the body by the water
// it displaces at each pose, as they pull it by gravity. The body is dragged
// toward the water's motion.

mod forces;
mod samples;

#[cfg(test)]
mod tests;

use crate::components::Buoyancy;
use crate::physics::{
    BodyHandle, ColliderShape, FluidLoad, MAX_FLUID_PIECES, Simulation, submerged,
};
use crate::planet::LocalFrame;
use crate::water::query::{WaterQuery, WaterSpot};
use crate::water::stir::StirPiece;
use concinnity_core::math::quat_rotate;
use forces::drag;
use samples::Samples;

// How far a body moves from where its shore spot was taken before it is
// taken again, in meters, and how long a spot is kept at most, in seconds.
const SPOT_REACH: f64 = 2.0;
const SPOT_SECONDS: f32 = 0.5;

// Water moving slower than this, in meters per second, lets a settled body
// sleep.
const STILL_WATER: f32 = 0.02;

// The depth a piece off the water reads, in meters: under nothing.
const DRY: f32 = -1.0e6;

/// One buoyant body's floating state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Floater {
    samples: Samples,
    volume: f32,
    buoyancy: Buoyancy,
    spot: WaterSpot,
    // Seconds left before the spot is taken again.
    spot_left: f32,
    // The share of a spot's life the first one is kept for less.
    stagger: f32,
    // The share of the body under water at its last tick.
    share: f32,
    // Each piece as it stirred the water at its last tick, up out of the
    // water there and the body's velocity.
    stir: [StirPiece; MAX_FLUID_PIECES],
    up: [f32; 3],
    velocity: [f32; 3],
}

impl Floater {
    /// A body of `shape` floating as `buoyancy` says, its first spot taken
    /// `stagger` of the way into the spot's life, so bodies added together
    /// refresh apart.
    pub(super) fn new(shape: &ColliderShape, buoyancy: Buoyancy, stagger: f32) -> Self {
        let samples = Samples::of(shape);
        Self {
            volume: samples.volume(),
            samples,
            buoyancy: buoyancy.validated(),
            spot: WaterSpot::OPEN,
            spot_left: 0.0,
            stagger: stagger.clamp(0.0, 1.0),
            share: 0.0,
            stir: [StirPiece::default(); MAX_FLUID_PIECES],
            up: [0.0, 1.0, 0.0],
            velocity: [0.0; 3],
        }
    }

    /// Whether any of the body was in water at its last tick.
    pub(super) fn afloat(&self) -> bool {
        self.share > 0.0
    }

    /// The share of the body under water at its last tick, from 0 to 1.
    pub(super) fn submerged_share(&self) -> f32 {
        self.share
    }

    /// Each piece as it stirred the water at its last tick.
    pub(super) fn stir(&self) -> &[StirPiece] {
        &self.stir[..self.samples.pieces().len()]
    }

    /// Up out of the water and the body's velocity at its last tick.
    pub(super) fn motion(&self) -> ([f32; 3], [f32; 3]) {
        (self.up, self.velocity)
    }

    // Off the water: every piece holds back nothing where it is.
    fn dry_out(&mut self, position: [f32; 3]) {
        self.share = 0.0;
        for s in &mut self.stir {
            s.at = position;
            s.volume = 0.0;
        }
    }

    /// One tick of the water's push on body `handle`, `dt` seconds starting
    /// `time` seconds into the world, the simulation laid out in `frame`:
    /// the water's level over each piece, handed to the simulation for its
    /// substeps to buoy the body by, and the drag toward the water's motion.
    /// A held body is only read, so its `afloat` stays current; a settled one
    /// stays asleep on still water.
    pub(super) fn push(
        &mut self,
        handle: BodyHandle,
        world: &mut Simulation,
        water: &WaterQuery,
        frame: &LocalFrame,
        time: f64,
        dt: f32,
    ) {
        let Some((position, rotation)) = world.body_pose_quat(handle) else {
            self.dry_out([0.0; 3]);
            return;
        };
        self.spot_left -= dt;
        let here = frame.to_world(position);
        if self.spot_left <= 0.0 || self.spot.distance(water.planet(), here) > SPOT_REACH {
            self.spot = water.spot(frame, position);
            self.spot_left = SPOT_SECONDS * (1.0 - self.stagger);
            self.stagger = 0.0;
        }
        let Some(patch) = water.patch(&self.spot, frame, position, time) else {
            self.dry_out(position);
            return;
        };
        let w = patch.center;
        // A piece off the water (past a rectangle's edge or under its floor)
        // reads a level far below it.
        let mut levels = [[0.0f32; 3]; MAX_FLUID_PIECES];
        let mut wet = 0.0;
        let pieces = self.samples.pieces().iter().zip(&mut levels);
        for ((piece, level), stir) in pieces.zip(&mut self.stir) {
            let o = quat_rotate(rotation, piece.offset);
            let at: [f32; 3] = core::array::from_fn(|i| position[i] + o[i]);
            let depth = water
                .depth(&self.spot, &patch, frame, at, time)
                .unwrap_or(DRY);
            *level = core::array::from_fn(|i| at[i] + w.up[i] * depth);
            let under = submerged(piece, rotation, w.up, depth).volume;
            *stir = StirPiece::new(at, *level, piece_radius(piece.volume), depth, under);
            wet += under;
        }
        self.share = (wet / self.volume.max(1e-9)).clamp(0.0, 1.0);
        self.up = w.up;
        let Some(state) = world.body_state(handle) else {
            return;
        };
        self.velocity = state.linear_velocity;
        // The load goes over dry as well as wet: a body falling in meets the
        // water in the substep it reaches it, not a tick later.
        let moving = w.velocity.iter().any(|v| v.abs() > STILL_WATER);
        if state.sleeping && !moving {
            return;
        }
        let load = FluidLoad {
            pieces: *self.samples.slots(),
            count: self.samples.pieces().len(),
            levels,
            up: w.up,
            density: w.density,
        };
        world.set_fluid_load(handle, &load);
        let (linear, angular) = drag(
            state.linear_velocity,
            state.angular_velocity,
            w.velocity,
            self.share,
            self.buoyancy,
            dt,
        );
        world.set_body_velocity(handle, linear, angular);
    }
}

// The radius of a ball of `volume`.
fn piece_radius(volume: f32) -> f32 {
    libm::cbrtf(3.0 * volume.max(0.0) / (4.0 * core::f32::consts::PI))
}
