// The rolling drives: each frame takes every driven body's steering from its
// `RollingControl`, each tick pushes the body by it, and the frame ends by
// telling the control whether the body is on the ground and afloat.

use alloc::vec::Vec;

use crate::components::{Collider, DriveScript, RollingControl, Transform};
use crate::ecs::{Entity, PipelineContext};
use crate::physics::drive_script::compass;
use crate::physics::rolling_drive::{RollingBody, TOW_RAMP, jump_impulse, push, tow, tow_hold};
use crate::physics::{BodyHandle, ColliderShape, Simulation};

use super::super::convert::collider_shape;
use super::super::props::PropBodies;
use super::up_at;
use concinnity_core::math::vec3::dot;
use concinnity_core::math::{SaturatingFrom, ceil};

// How long a jump asked for in the air is kept for the body to land, in
// seconds.
const JUMP_WINDOW: f32 = 0.15;

// The steepest ground the body counts as standing on: about 50 degrees.
const GROUND_MIN_COS: f32 = 0.64;

#[derive(Debug, Clone, Copy)]
struct Towed {
    pace: f32,
    draft: f32,
    last: f32,
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
    grounded: bool,
    afloat: f32,
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
    /// take, then the roll and the paddle for this tick's step, or the tow a
    /// script's pace asks for in their place. North lies along `north` before
    /// it is laid along the ground.
    pub(super) fn tick(
        &mut self,
        world: &mut Simulation,
        ctx: &PipelineContext<'_>,
        props: &PropBodies,
        (time, dt): (f64, f32),
        north: [f32; 3],
    ) {
        for driven in &mut self.driven {
            driven.afloat = props.submerged_share(driven.entity);
            let Some(state) = world.body_state(driven.handle) else {
                continue;
            };
            let up = up_at(world, state.position);
            let mut pace = 0.0;
            let mut heading_deg = 0.0;
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
            let push = push(&driven.control, &body);
            if push.force != [0.0; 3] || push.torque != [0.0; 3] {
                world.add_force(driven.handle, push.force, push.torque);
            }
        }
    }

    /// Hand each control what its body's last tick found.
    pub(super) fn publish(&self, ctx: &mut PipelineContext<'_>, props: &PropBodies) {
        for driven in &self.driven {
            if let Some(control) = ctx.get_mut::<RollingControl>(driven.entity) {
                control.grounded = driven.grounded;
                control.afloat = props.submerged_share(driven.entity);
            }
        }
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
