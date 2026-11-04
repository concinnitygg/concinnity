// The rolling drives: each frame takes every driven body's steering from its
// `RollingControl`, each tick pushes the body by it, and the frame ends by
// telling the control how the body moves, whether it is on the ground and
// afloat.

use alloc::vec::Vec;

use crate::components::{Collider, DriveScript, RollingControl, Transform};
use crate::ecs::{Entity, PipelineContext};
use crate::physics::drive_script::compass;
use crate::physics::layers::LayerMask;
use crate::physics::rolling_drive::{
    RollingBody, TOW_RAMP, flight_ceiling, fly, jump_impulse, push, steer_flight, stop_climb, tow,
    tow_hold,
};
use crate::physics::{BodyHandle, ColliderShape, Simulation};
use crate::planet::PlanetFrame;

use super::super::convert::collider_shape;
use super::super::props::PropBodies;
use super::up_at;
use concinnity_core::math::vec3::{dot, length, scale};
use concinnity_core::math::{SaturatingFrom, ceil};

// How long a jump asked for in the air is kept for the body to land, in
// seconds.
const JUMP_WINDOW: f32 = 0.15;

// How long a jump is held before the body flies, in seconds, so a tapped jump
// stays a jump.
const CLIMB_HOLD: f32 = 0.25;

// How far below a flying body the ground is looked for, in meters. Past it,
// only a planet's surface is.
const GROUND_REACH: f32 = 1_000.0;

// The steepest ground the body counts as standing on: about 50 degrees.
const GROUND_MIN_COS: f32 = 0.64;

#[derive(Debug, Clone, Copy)]
struct Towed {
    pace: f32,
    draft: f32,
    last: f32,
}

// How a held jump or descend is flying the body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flight {
    // Rolling, or falling under gravity alone.
    Off,
    Up,
    Down,
    // A climb let go of, winding down.
    Easing,
}

#[derive(Debug, Clone, Copy)]
struct Driven {
    entity: Entity,
    handle: BodyHandle,
    control: RollingControl,
    radius: f32,
    // Ticks left in which a jump asked for is taken.
    jump_ticks: u32,
    // The script key that steered the last tick, so a key's jump is asked for
    // once, as it takes over.
    script_key: Option<usize>,
    // While a scripted tow pulls the body: the pace it has reached, and the
    // share of the body under water it floated at and had last tick.
    towed: Option<Towed>,
    // Seconds the player has held a direction, and the jump.
    steered_seconds: f32,
    climb_seconds: f32,
    flight: Flight,
    grounded: bool,
    afloat: f32,
}

/// What a flying body is measured against: the planet's surface and sea, and
/// whatever lies under it within reach.
#[derive(Debug, Clone, Copy)]
pub(super) struct Altimeter<'a> {
    pub(super) planet: Option<PlanetSurface<'a>>,
    /// What the body's downward ray meets.
    pub(super) mask: LayerMask,
}

/// A planet's ground as a flying body reads it.
#[derive(Debug, Clone, Copy)]
pub(super) struct PlanetSurface<'a> {
    pub(super) planet: &'a PlanetFrame,
    /// The sea's level over the planet's radius, where it has one.
    pub(super) sea_level: Option<f32>,
}

impl Altimeter<'_> {
    /// How far the body `handle` at local point `at` is above what it would
    /// land on, along `up`: the nearer of the ground under it and the
    /// planet's surface or sea, or the reach where neither is found.
    pub(super) fn height(
        &self,
        world: &Simulation,
        handle: BodyHandle,
        at: [f32; 3],
        up: [f32; 3],
    ) -> f32 {
        let below = world
            .raycast(at, scale(up, -1.0), GROUND_REACH, Some(handle), self.mask)
            .map(|hit| hit.distance);
        let over_planet = self.planet.map(|ground| ground.height(at));
        match (below, over_planet) {
            (Some(a), Some(b)) => a.min(b),
            (Some(h), None) | (None, Some(h)) => h,
            (None, None) => GROUND_REACH,
        }
    }
}

impl PlanetSurface<'_> {
    // How far local point `at` lies above the planet's ground, or its sea
    // where that is higher.
    fn height(&self, at: [f32; 3]) -> f32 {
        let ground = self.planet.height_over_ground(at);
        self.sea_level
            .map_or(ground, |level| ground.min(self.planet.altitude(at) - level))
    }
}

/// What every drive's tick reads besides its own body: north before it is
/// laid along the ground, and the ground a flying body climbs from.
#[derive(Debug, Clone, Copy)]
pub(super) struct Surroundings<'a> {
    pub(super) north: [f32; 3],
    pub(super) altimeter: Altimeter<'a>,
}

/// Every driven body and its steering for the frame.
#[derive(Debug, Default)]
pub(super) struct Drives {
    driven: Vec<Driven>,
    // Ticks a jump asked for is kept for.
    jump_window: u32,
}

impl Drives {
    pub(super) fn with_capacity(count: usize) -> Self {
        Self {
            driven: Vec::with_capacity(count),
            jump_window: 0,
        }
    }

    /// Take this frame's steering from every control whose body is simulated
    /// freely, keeping a jump asked for in an earlier frame that has not yet
    /// run out. A jump is taken from the control as it is read; one asked of
    /// a body something else poses is dropped.
    pub(super) fn gather(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        props: &PropBodies,
        tick_dt: f32,
    ) {
        let free = |entity: Entity| {
            props
                .prop_of(entity)
                .filter(|p| p.dynamic && !p.is_posed())
                .map(|p| p.handle)
        };
        self.driven.retain(|d| free(d.entity) == Some(d.handle));
        let window = u32::saturating_from(ceil(JUMP_WINDOW / tick_dt.max(1e-6)));
        self.jump_window = window;
        for (entity, control) in ctx.query_mut_with_entity::<RollingControl>() {
            let Some(handle) = free(entity) else {
                control.jump = false;
                continue;
            };
            let at = if let Some(at) = self.driven.iter().position(|d| d.entity == entity) {
                at
            } else {
                self.driven.push(Driven {
                    entity,
                    handle,
                    control: *control,
                    radius: 0.0,
                    jump_ticks: 0,
                    script_key: None,
                    towed: None,
                    steered_seconds: 0.0,
                    climb_seconds: 0.0,
                    flight: Flight::Off,
                    grounded: false,
                    afloat: 0.0,
                });
                self.driven.len() - 1
            };
            let driven = &mut self.driven[at];
            if control.jump {
                driven.jump_ticks = window;
                control.jump = false;
            }
            driven.control = *control;
        }
        for driven in &mut self.driven {
            driven.radius = rolling_radius(ctx, driven.entity);
        }
    }

    /// One `dt`-second tick of every drive starting `time` seconds into the
    /// world: the steering a script has for that moment, a jump the body can
    /// take, the flight a held jump gives it, then the roll and the paddle
    /// for this tick's step, or the tow a script's pace asks for in their
    /// place. A script's steering neither ramps nor flies.
    #[expect(clippy::float_cmp, reason = "zero push is no force to apply")]
    pub(super) fn tick(
        &mut self,
        world: &mut Simulation,
        ctx: &PipelineContext<'_>,
        props: &PropBodies,
        time: f64,
        dt: f32,
        around: &Surroundings<'_>,
    ) {
        let north = around.north;
        let gravity = world.config().gravity;
        for driven in &mut self.driven {
            driven.afloat = props.submerged_share(driven.entity);
            let Some(state) = world.body_state(driven.handle) else {
                continue;
            };
            let up = up_at(world, state.position);
            let mut pace = 0.0;
            let mut heading_deg = 0.0;
            let mut scripted = false;
            if let Some((index, key)) = ctx
                .get::<DriveScript>(driven.entity)
                .and_then(|script| script.key_at(time))
            {
                DriveScript::steer(key, &mut driven.control, up, north);
                if key.jump && driven.script_key != Some(index) {
                    driven.jump_ticks = self.jump_window;
                }
                driven.script_key = Some(index);
                pace = key.pace;
                heading_deg = key.heading_deg;
                scripted = true;
            }
            driven.grounded = world.is_supported(driven.handle, up, GROUND_MIN_COS);
            let body = RollingBody {
                up,
                linear_velocity: state.linear_velocity,
                angular_velocity: state.angular_velocity,
                mass: state.mass,
                inertia: (state.inertia[0] + state.inertia[1] + state.inertia[2]) / 3.0,
                radius: driven.radius,
                afloat: driven.afloat,
                grounded: driven.grounded,
            };
            if pace > 0.0 {
                let toward = compass(heading_deg, up, north);
                let share = driven.afloat;
                let towed = driven.towed.unwrap_or_else(|| Towed {
                    pace: dot(state.linear_velocity, toward).max(0.0),
                    draft: share,
                    last: share,
                });
                let step = TOW_RAMP * dt;
                let pace = towed.pace + (pace - towed.pace).clamp(-step, step);
                // A share of a ball's volume per meter it sinks, near its middle.
                let per_meter = 0.75 / driven.radius.max(1e-3);
                let deeper = (share - towed.draft) / per_meter;
                let sinking = (share - towed.last) / (dt.max(1e-6) * per_meter);
                let mut force = tow(pace, toward, &body);
                if towed.draft > 0.0 {
                    let hold = tow_hold(deeper, sinking, &body);
                    force = core::array::from_fn(|i| force[i] + hold[i]);
                }
                driven.towed = Some(Towed {
                    pace,
                    draft: towed.draft,
                    last: share,
                });
                driven.jump_ticks = 0;
                world.add_force(driven.handle, force, [0.0; 3]);
                continue;
            }
            driven.towed = None;
            let was = driven.flight;
            driven.steer_time(scripted, dt);
            if driven.flight == Flight::Off
                && !scripted
                && matches!(was, Flight::Up | Flight::Easing)
                && dot(state.linear_velocity, up) > 0.0
            {
                driven.flight = Flight::Easing;
            }
            if driven.flight == Flight::Easing {
                world.add_force(driven.handle, stop_climb(&body), [0.0; 3]);
            }
            if matches!(driven.flight, Flight::Up | Flight::Down) {
                let height = around
                    .altimeter
                    .height(world, driven.handle, state.position, up)
                    - driven.radius;
                let speed = driven.control.climb_speed(height);
                let mut ahead = speed.max(driven.control.ramped(driven.steered_seconds).max_speed);
                if let Some(center) = world.config().gravity_center {
                    let distance = length(core::array::from_fn(|i| state.position[i] - center[i]));
                    ahead = ahead.min(flight_ceiling(gravity, distance));
                }
                let rising = if driven.flight == Flight::Down {
                    -speed
                } else {
                    speed
                };
                let lift = fly(rising, gravity, &body);
                let steer = steer_flight(&driven.control, ahead, &body);
                world.add_force(
                    driven.handle,
                    core::array::from_fn(|i| lift[i] + steer[i]),
                    [0.0; 3],
                );
            }
            if driven.jump_ticks > 0 {
                driven.jump_ticks -= 1;
                if driven.grounded && driven.control.jump_speed > 0.0 {
                    world.apply_impulse(
                        driven.handle,
                        jump_impulse(&driven.control, &body),
                        [0.0; 3],
                    );
                    driven.jump_ticks = 0;
                }
            }
            let push = push(&driven.control.ramped(driven.steered_seconds), &body);
            if push.force != [0.0; 3] || push.torque != [0.0; 3] {
                world.add_force(driven.handle, push.force, push.torque);
            }
        }
    }

    /// Hand each control what its body's last tick found.
    pub(super) fn publish(
        &self,
        world: &Simulation,
        ctx: &mut PipelineContext<'_>,
        props: &PropBodies,
    ) {
        for driven in &self.driven {
            if let Some(control) = ctx.get_mut::<RollingControl>(driven.entity) {
                control.velocity = world
                    .body_state(driven.handle)
                    .map_or([0.0; 3], |state| state.linear_velocity);
                control.grounded = driven.grounded;
                control.afloat = props.submerged_share(driven.entity);
            }
        }
    }
}

impl Driven {
    // Count how long the player has held a direction and the jump this
    // `dt`-second tick, and whether the body flies: up while a jump is held
    // past a moment, down while descend is held off the ground and the water.
    // Holding both flies neither way. A script holds none of them.
    fn steer_time(&mut self, scripted: bool, dt: f32) {
        let steering = !scripted && length(self.control.heading) > 1e-4;
        self.steered_seconds = if steering {
            self.steered_seconds + dt
        } else {
            0.0
        };
        let can_fly = !scripted && self.control.climb_rate > 0.0;
        let climbing = can_fly && self.control.climb && !self.control.descend;
        self.climb_seconds = if climbing {
            self.climb_seconds + dt
        } else {
            0.0
        };
        let diving = can_fly
            && self.control.descend
            && !self.control.climb
            && !self.grounded
            && self.afloat <= 0.0;
        self.flight = if self.climb_seconds >= CLIMB_HOLD {
            Flight::Up
        } else if diving {
            Flight::Down
        } else {
            Flight::Off
        };
    }
}

// The radius `entity`'s collider rolls on: a ball's or a capsule's, or a
// box's narrowest half width.
fn rolling_radius(ctx: &PipelineContext<'_>, entity: Entity) -> f32 {
    let Some(collider) = ctx.get::<Collider>(entity) else {
        return 0.0;
    };
    let scale = ctx.get::<Transform>(entity).map_or([1.0; 3], |t| t.scale);
    match collider_shape(&collider.0, scale) {
        ColliderShape::Ball { radius } | ColliderShape::Capsule { radius, .. } => radius,
        ColliderShape::Cuboid { half_extents } => {
            half_extents[0].min(half_extents[1]).min(half_extents[2])
        }
    }
}
