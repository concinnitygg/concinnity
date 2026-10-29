//! Spray thrown up where floating bodies hit the water hard: a burst on the
//! water's spray emitter as a body plunges in or leaps out, and a steady
//! throw from the bow of one moving fast through it. Read off the bodies'
//! stirs (see [`super::stir`]) after the physics steps, and asked of the
//! emitter through [`ParticleBursts`].

use crate::ecs::asset_id::AssetId;
use crate::ecs::{Entity, PipelineContext, SimTiming, StepResult, System, World};
use crate::render::particles::{ParticleBurst, ParticleBursts};
use crate::water::WaterSurface;
use crate::water::stir::{WaterStir, WaterStirs};
use alloc::vec::Vec;
use concinnity_core::math::vec3::{cross, dot, length, normalize_or, scale, sub};
use concinnity_core::math::{SaturatingFrom, floor};

/// How fast a body must gain or lose submerged volume to throw a splash, in
/// shares of its whole volume per second: a body bobbing on the swell stays
/// under it, one plunging in or leaping out passes it.
const SPRAY_PLUNGE: f32 = 2.5;

/// Particles per share of its volume per second a body plunges at, and the
/// most one splash throws.
const SPRAY_PER_PLUNGE: f32 = 24.0;
const SPRAY_MAX_SPLASH: u32 = 240;

/// The speed through the water past which a body throws spray from its bow,
/// in meters per second, and the particles it throws per second for each
/// meter per second past it.
const SPRAY_BOW_SPEED: f32 = 4.0;
const SPRAY_BOW_RATE: f32 = 80.0;

/// How far a bow's spray leans up and out to each side, as shares of up,
/// and the share of the body's own speed it keeps: it peels off the bow in a
/// sheet to either side and rides along with it.
const SPRAY_BOW_FLARE: f32 = 1.0;
const SPRAY_BOW_CARRY: f32 = 0.9;

// What one world's spray has thrown since it started.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct SprayThrown {
    // Splashes thrown by plunging or leaping bodies.
    splashes: u32,
    // Bursts thrown from fast bows.
    bow_bursts: u32,
    // Particles asked for in all.
    particles: u64,
}

/// What one body's spray remembers from frame to frame.
#[derive(Debug, Clone, Copy, PartialEq)]
struct BodySpray {
    entity: Entity,
    // The water it held under last frame, in cubic meters.
    volume: f32,
    // Bow particles owed but not yet thrown.
    owed: f32,
    // Whether its bow throws spray.
    bowing: bool,
}

/// Throws the spray of every floating body on the world's spray emitter.
#[derive(Debug)]
pub struct WaterSpraySystem {
    emitter: AssetId,
    // Last frame's bodies, by entity.
    bodies: Vec<BodySpray>,
    // This frame's, built beside them.
    next: Vec<BodySpray>,
    // The frame's stirs, copied out of the world.
    stirs: Vec<WaterStir>,
    thrown: SprayThrown,
}

impl WaterSpraySystem {
    /// The system for a world whose water names a spray emitter.
    pub fn for_world(world: &World) -> Option<Self> {
        let emitter = world.query::<WaterSurface>().find_map(|s| s.spray)?.id();
        Some(Self::new(emitter))
    }

    fn new(emitter: AssetId) -> Self {
        Self {
            emitter,
            bodies: Vec::new(),
            next: Vec::new(),
            stirs: Vec::new(),
            thrown: SprayThrown::default(),
        }
    }

    #[cfg(test)]
    fn thrown(&self) -> SprayThrown {
        self.thrown
    }

    /// Push the bursts `stirs` throw over the `dt` seconds since the last
    /// frame onto `out`.
    fn throw(&mut self, dt: f32, out: &mut ParticleBursts) {
        self.next.clear();
        for stir in &self.stirs {
            let last = self
                .bodies
                .binary_search_by_key(&stir.entity, |b| b.entity)
                .ok()
                .map(|i| self.bodies[i]);
            let body = spray_of(stir, last, dt, self.emitter, out, &mut self.thrown);
            self.next.push(body);
        }
        self.next.sort_unstable_by_key(|b| b.entity);
        core::mem::swap(&mut self.bodies, &mut self.next);
    }
}

impl System for WaterSpraySystem {
    fn step(&mut self, ctx: &mut PipelineContext<'_>) -> StepResult {
        let dt = ctx
            .resource::<SimTiming>()
            .map_or(0.0, |t| t.ticks as f32 * t.tick_dt);
        if dt <= 0.0 {
            return StepResult::Continue;
        }
        if ctx.resource::<ParticleBursts>().is_none() {
            ctx.insert_resource(ParticleBursts::default());
        }
        self.stirs.clear();
        if let Some(stirs) = ctx.resource::<WaterStirs>() {
            self.stirs.extend_from_slice(&stirs.0);
        }
        if let Some(bursts) = ctx.resource_mut::<ParticleBursts>() {
            self.throw(dt, bursts);
        }
        StepResult::Continue
    }
}

// One body's spray this frame, from `stir` and what it held `last` frame:
// pushed onto `out` as bursts on `emitter` and counted in `thrown`. Returns
// what it remembers.
fn spray_of(
    stir: &WaterStir,
    last: Option<BodySpray>,
    dt: f32,
    emitter: AssetId,
    out: &mut ParticleBursts,
    thrown: &mut SprayThrown,
) -> BodySpray {
    let pieces = stir.pieces();
    let volume: f32 = pieces.iter().map(|p| p.volume).sum();
    let radius = pieces.iter().map(|p| p.radius).fold(0.0, f32::max);
    let whole: f32 = pieces
        .iter()
        .map(|p| 4.0 / 3.0 * core::f32::consts::PI * p.radius * p.radius * p.radius)
        .sum::<f32>()
        .max(1e-6);
    let mut body = BodySpray {
        entity: stir.entity,
        volume,
        owed: last.map_or(0.0, |l| l.owed),
        bowing: false,
    };
    if volume <= 0.0 && last.is_none_or(|l| l.volume <= 0.0) {
        body.owed = 0.0;
        return body;
    }
    // Where it meets the surface: the water over its pieces, weighed by how
    // much each holds under, or over all of them once it has left.
    let weights = |p: &crate::water::stir::StirPiece| if volume > 0.0 { p.volume } else { 1.0 };
    let total: f32 = pieces.iter().map(weights).sum::<f32>().max(1e-6);
    let at: [f32; 3] = core::array::from_fn(|i| {
        pieces.iter().map(|p| p.level[i] * weights(p)).sum::<f32>() / total
    });
    let up = stir.up;
    if let Some(last) = last {
        let plunge = (volume - last.volume) / (dt * whole);
        if plunge.abs() > SPRAY_PLUNGE {
            let count = u32::saturating_from(floor(plunge.abs() * SPRAY_PER_PLUNGE))
                .clamp(4, SPRAY_MAX_SPLASH);
            let taken = out.push(ParticleBurst {
                emitter,
                position: at,
                direction: up,
                speed: (plunge.abs() / (3.0 * SPRAY_PLUNGE)).clamp(0.6, 1.5),
                carry: [0.0; 3],
                floor: Some(up),
                count,
            });
            if taken {
                thrown.splashes += 1;
                thrown.particles += u64::from(count);
                tracing::debug!(
                    "spray: splash of {count} at {plunge:.1} body volumes a second \
                     (splash {}, {} particles thrown)",
                    thrown.splashes,
                    thrown.particles
                );
            }
        }
    }
    // A bow throws only while it is in the water.
    let along = sub(stir.velocity, scale(up, dot(stir.velocity, up)));
    let speed = length(along);
    if volume <= 0.0 || speed <= SPRAY_BOW_SPEED {
        body.owed = 0.0;
        return body;
    }
    body.bowing = true;
    if !last.is_some_and(|l| l.bowing) {
        tracing::debug!(
            "spray: a bow throws spray at {speed:.1} m/s ({} bow bursts so far)",
            thrown.bow_bursts
        );
    }
    body.owed += (speed - SPRAY_BOW_SPEED) * SPRAY_BOW_RATE * dt;
    // Thrown in pairs, a sheet to each side.
    let pairs = floor(0.5 * body.owed);
    if pairs >= 1.0 {
        body.owed -= 2.0 * pairs;
        let ahead = normalize_or(along, 1e-6, [0.0; 3]);
        let side = normalize_or(cross(up, ahead), 1e-6, [0.0; 3]);
        let count = u32::saturating_from(pairs);
        for flare in [SPRAY_BOW_FLARE, -SPRAY_BOW_FLARE] {
            let throw: [f32; 3] = core::array::from_fn(|i| up[i] + flare * side[i]);
            let out_side = flare.signum() * 0.5;
            let off: [f32; 3] =
                core::array::from_fn(|i| radius * (0.7 * ahead[i] + out_side * side[i]));
            if out.push(ParticleBurst {
                emitter,
                position: core::array::from_fn(|i| at[i] + off[i]),
                direction: normalize_or(throw, 1e-6, up),
                speed: (speed / (2.0 * SPRAY_BOW_SPEED)).clamp(0.6, 1.4),
                carry: scale(ahead, SPRAY_BOW_CARRY * speed),
                floor: Some(up),
                count,
            }) {
                thrown.bow_bursts += 1;
                thrown.particles += u64::from(count);
            }
        }
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::MAX_FLUID_PIECES;
    use crate::water::stir::StirPiece;

    fn ball(volume: f32, velocity: [f32; 3]) -> WaterStir {
        let mut pieces = [StirPiece::default(); MAX_FLUID_PIECES];
        pieces[0] = StirPiece {
            at: [0.0, -0.1, 0.0],
            level: [0.0, 0.2, 0.0],
            radius: 0.6,
            volume,
        };
        WaterStir {
            entity: Entity::dangling(),
            up: [0.0, 1.0, 0.0],
            velocity,
            pieces,
            count: 1,
        }
    }

    fn run(system: &mut WaterSpraySystem, stir: &WaterStir, dt: f32) -> Vec<ParticleBurst> {
        let mut out = ParticleBursts::default();
        system.stirs.clear();
        system.stirs.push(*stir);
        system.throw(dt, &mut out);
        out.as_slice().to_vec()
    }

    // A body seen for the first time throws nothing; a hard plunge throws a
    // splash at the surface over it, and a gentle settle does not.
    #[test]
    fn a_plunge_splashes_and_a_settle_does_not() {
        let mut s = WaterSpraySystem::new(AssetId(3));
        assert!(run(&mut s, &ball(0.0, [0.0; 3]), 0.02).is_empty());
        let splash = run(&mut s, &ball(0.1, [0.0, -5.0, 0.0]), 0.02);
        assert_eq!(splash.len(), 1);
        assert_eq!(s.thrown().splashes, 1);
        assert_eq!(s.thrown().particles, u64::from(splash[0].count));
        let b = splash[0];
        assert_eq!(b.emitter, AssetId(3));
        assert!((b.position[1] - 0.2).abs() < 1e-6, "{:?}", b.position);
        assert_eq!(b.direction, [0.0, 1.0, 0.0]);
        // A tenth of a cubic meter in a fiftieth of a second, of a 0.9 cubic
        // meter ball.
        let whole = 4.0 / 3.0 * core::f32::consts::PI * 0.6f32.powi(3);
        let want = floor(0.1 / (0.02 * whole) * SPRAY_PER_PLUNGE).min(SPRAY_MAX_SPLASH as f32);
        assert_eq!(b.count as f32, want);
        assert!(run(&mut s, &ball(0.105, [0.0; 3]), 0.02).is_empty());
        // Bobbing a tenth of its volume over a second throws nothing.
        assert!(run(&mut s, &ball(0.13, [0.0; 3]), 0.1).is_empty());
        // Leaping out splashes too.
        assert_eq!(run(&mut s, &ball(0.0, [0.0, 5.0, 0.0]), 0.02).len(), 1);
    }

    // A body plowing through the water faster than the bow speed throws a
    // steady spray from its front in a sheet to each side, carried along
    // with it; a slow one throws none.
    #[test]
    fn a_fast_body_throws_bow_spray() {
        let mut s = WaterSpraySystem::new(AssetId(3));
        run(&mut s, &ball(0.3, [2.0, 0.0, 0.0]), 0.02);
        let mut slow = 0;
        for _ in 0..50 {
            slow += run(&mut s, &ball(0.3, [2.0, 0.0, 0.0]), 0.02).len();
        }
        assert_eq!(slow, 0);
        let mut thrown = 0;
        let mut sides = [0; 2];
        for _ in 0..50 {
            for b in run(&mut s, &ball(0.3, [10.0, 0.0, 0.0]), 0.02) {
                assert!(b.position[0] > 0.3 && (b.position[1] - 0.2).abs() < 1e-6);
                assert!(b.direction[1] > 0.0 && b.direction[2].abs() > 0.5);
                assert!(
                    (b.carry[0] - SPRAY_BOW_CARRY * 10.0).abs() < 1e-4,
                    "{:?}",
                    b.carry
                );
                sides[usize::from(b.direction[2] > 0.0)] += b.count;
                thrown += b.count;
            }
        }
        // A second at 6 m/s past the bow speed, `SPRAY_BOW_RATE` particles a
        // second each, split evenly between the sides.
        let want = (10.0 - SPRAY_BOW_SPEED) * SPRAY_BOW_RATE;
        assert!((thrown as f32 - want).abs() <= 2.0, "{thrown}");
        assert_eq!(sides[0], sides[1]);
        assert_eq!(u64::from(thrown), s.thrown().particles);
        // Out of the water the bow throws nothing, and owes nothing after.
        assert!(run(&mut s, &ball(0.0, [10.0, 0.0, 0.0]), 0.02).len() <= 1);
        assert!(run(&mut s, &ball(0.0, [10.0, 0.0, 0.0]), 0.02).is_empty());
    }
}
