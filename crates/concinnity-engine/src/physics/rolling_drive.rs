// A steered rolling body: the authored drive, the runtime control it resolves
// into on its prop's entity, and the push one tick of it gives the body.

use crate::components::Prop;
use crate::ecs::Ref;
use concinnity_core::math::vec3::{cross, dot, length, normalize_or, scale};

/// Rolls a companion [Prop](#prop)'s body the way it is steered.
///
/// The prop needs a ball or capsule `collider` and a
/// [`PropBody`](#propbody). Steering turns the body about the axis square to
/// the ground's up and the heading, so it rolls toward the heading by the
/// grip its contact has, under whatever gravity pulls it: down `-Y`, or
/// toward a [`Planet`](#planet)'s center. In water the roll gives way to a
/// paddle by the share of the body that floats, since a spinning ball does
/// not swim. A third-person `Camera3D` that follows the prop steers it from
/// the movement keys or the left stick, relative to where it looks.
///
/// ```json
/// ["RollingDrive", { "prop_name": "ball", "acceleration": 8.0, "max_speed": 12.0 }]
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
pub struct RollingDrive {
    /// The [Prop](#prop) this drive rolls. Must match a Prop with a
    /// [`PropBody`](#propbody) declared in the same world.
    pub prop_name: Option<Ref<Prop>>,
    /// How hard the drive rolls the body on level ground, in meters per
    /// second squared, whatever it weighs.
    #[asset(default = 8.0)]
    pub acceleration: f32,
    /// Speed along the heading, in meters per second, past which the drive
    /// stops adding to it. A slope or a push can still carry the body faster.
    #[asset(default = 10.0)]
    pub max_speed: f32,
    /// What sprinting multiplies `acceleration` and `max_speed` by.
    #[asset(default = 1.8)]
    pub sprint_multiplier: f32,
    /// Speed a jump gives the body away from the ground, in meters per
    /// second. 0 never jumps. A jump only leaves the ground.
    #[asset(default = 5.0)]
    pub jump_speed: f32,
    /// Share of the drive that pushes a floating body through the water:
    /// fully afloat it paddles at `paddle` times `acceleration`.
    #[asset(default = 0.3)]
    pub paddle: f32,
}

/// The live steering of a [`RollingDrive`], on its prop's entity.
///
/// Runtime-only. Whatever steers the body writes `heading`, `sprint` and
/// `jump`; the physics reads them every tick and writes back whether the body
/// is on the ground and how much of it floats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RollingControl {
    /// Rolling acceleration on level ground, in meters per second squared.
    pub acceleration: f32,
    /// Speed along the heading the drive stops at, in meters per second.
    pub max_speed: f32,
    /// What sprinting multiplies the acceleration and top speed by.
    pub sprint_multiplier: f32,
    /// Jump speed away from the ground, in meters per second.
    pub jump_speed: f32,
    /// Share of the drive left to paddle with when fully afloat.
    pub paddle: f32,
    /// Where to roll, in the simulated frame: a direction along the ground
    /// whose length, at most 1, is the throttle. Zero coasts.
    pub heading: [f32; 3],
    /// Whether to sprint.
    pub sprint: bool,
    /// A jump asked for. The physics takes it, jumping on the first tick the
    /// body is on the ground within a moment of the ask, or dropping it.
    pub jump: bool,
    /// Whether the body was on the ground at its last tick.
    pub grounded: bool,
    /// The share of the body under water at its last tick, from 0 to 1.
    pub afloat: f32,
}

impl RollingControl {
    /// The control a [`RollingDrive`] starts with: its tuning, steered nowhere.
    pub fn new(drive: &RollingDrive) -> Self {
        Self {
            acceleration: drive.acceleration.max(0.0),
            max_speed: drive.max_speed.max(0.0),
            sprint_multiplier: drive.sprint_multiplier.max(0.0),
            jump_speed: drive.jump_speed.max(0.0),
            paddle: drive.paddle.clamp(0.0, 1.0),
            heading: [0.0; 3],
            sprint: false,
            jump: false,
            grounded: false,
            afloat: 0.0,
        }
    }
}

/// What a rolling body is like at one tick, as the drive reads it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RollingBody {
    /// Away from the pull, unit length.
    pub up: [f32; 3],
    pub linear_velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub mass: f32,
    /// Moment of inertia about the rolling axis.
    pub inertia: f32,
    /// The radius it rolls on.
    pub radius: f32,
    /// Share of it under water, 0 to 1.
    pub afloat: f32,
    /// Whether it stands on something.
    pub grounded: bool,
}

/// The force and torque one tick of a drive puts on its body, in newtons and
/// newton meters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RollingPush {
    pub force: [f32; 3],
    pub torque: [f32; 3],
}

/// The push `control` gives `body`: on the ground, torque about
/// `up x heading` that rolls it at the drive's acceleration on level ground,
/// giving way to a paddle along the heading by the share afloat. Both fade to
/// nothing as the body nears the top speed along the heading, read as the
/// faster of how it moves and how its spin would carry it, so spin wound up
/// off the ground cannot land it past the top speed.
pub(crate) fn push(control: &RollingControl, body: &RollingBody) -> RollingPush {
    const NONE: RollingPush = RollingPush {
        force: [0.0; 3],
        torque: [0.0; 3],
    };
    // Only the part of the heading along the ground steers.
    let along = dot(control.heading, body.up);
    let flat = core::array::from_fn(|i| control.heading[i] - body.up[i] * along);
    let throttle = length(flat).min(1.0);
    if throttle <= 1e-4 || body.radius <= 0.0 {
        return NONE;
    }
    let toward = normalize_or(flat, 1e-6, [0.0; 3]);
    let boost = if control.sprint {
        control.sprint_multiplier
    } else {
        1.0
    };
    let top = control.max_speed * boost;
    if top <= 0.0 {
        return NONE;
    }
    let axis = normalize_or(cross(body.up, toward), 1e-6, [0.0; 3]);
    let moving = dot(body.linear_velocity, toward);
    let spinning = body.radius * dot(body.angular_velocity, axis);
    let ease = (1.0 - moving.max(spinning) / top).clamp(0.0, 1.0);
    let accel = control.acceleration * boost * throttle * ease;
    let afloat = body.afloat.clamp(0.0, 1.0);
    // Rolling without slipping at `accel` takes `accel (I / r + m r)`.
    let roll = if body.grounded {
        accel * (body.inertia / body.radius + body.mass * body.radius) * (1.0 - afloat)
    } else {
        0.0
    };
    let paddle = accel * body.mass * control.paddle * afloat;
    RollingPush {
        force: scale(toward, paddle),
        torque: scale(axis, roll),
    }
}

/// The impulse a jump gives `body`, in newton seconds: what brings its speed
/// away from the ground up to the jump speed, and nothing for a body already
/// rising faster.
pub(crate) fn jump_impulse(control: &RollingControl, body: &RollingBody) -> [f32; 3] {
    let rising = dot(body.linear_velocity, body.up);
    scale(body.up, body.mass * (control.jump_speed - rising).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn control() -> RollingControl {
        RollingControl::new(&RollingDrive {
            acceleration: 8.0,
            max_speed: 10.0,
            sprint_multiplier: 2.0,
            paddle: 0.5,
            ..RollingDrive::default()
        })
    }

    // A 2 kg half-meter solid ball on a planet's side, where up is +X.
    fn ball(velocity: [f32; 3], afloat: f32) -> RollingBody {
        RollingBody {
            up: [1.0, 0.0, 0.0],
            linear_velocity: velocity,
            angular_velocity: [0.0; 3],
            mass: 2.0,
            inertia: 0.4 * 2.0 * 0.25,
            radius: 0.5,
            afloat,
            grounded: true,
        }
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-4)
    }

    // Heading along the ground turns the ball about up x heading by exactly
    // the torque that rolls it at the acceleration; the part of the heading
    // along up is ignored.
    #[test]
    fn the_drive_rolls_about_up_cross_heading() {
        let mut c = control();
        c.heading = [0.7, 0.0, 1.0];
        let p = push(&c, &ball([0.0; 3], 0.0));
        // a (I / r + m r) = 8 (0.4 + 1.0) = 11.2, about +X x +Z = -Y.
        assert!(close(p.torque, [0.0, -11.2, 0.0]), "{:?}", p.torque);
        assert_eq!(p.force, [0.0; 3]);
    }

    // The drive fades as the speed along the heading nears the top, sprinting
    // raises the top, and a ball already past it gets nothing.
    #[test]
    fn the_drive_fades_to_the_top_speed() {
        let mut c = control();
        c.heading = [0.0, 0.0, 1.0];
        let half = push(&c, &ball([0.0, 0.0, 5.0], 0.0)).torque[1];
        assert!((half + 5.6).abs() < 1e-4, "{half}");
        assert_eq!(push(&c, &ball([0.0, 0.0, 12.0], 0.0)).torque, [0.0; 3]);
        c.sprint = true;
        let sprinting = push(&c, &ball([0.0, 0.0, 10.0], 0.0)).torque[1];
        assert!((sprinting + 11.2).abs() < 1e-4, "{sprinting}");
        c.heading = [0.0, 0.0, 0.25];
        c.sprint = false;
        let partial = push(&c, &ball([0.0; 3], 0.0)).torque[1];
        assert!(
            (partial + 2.8).abs() < 1e-4,
            "the throttle scales it: {partial}"
        );
    }

    // Afloat, the roll gives way to a paddle along the heading.
    #[test]
    fn afloat_the_drive_paddles() {
        let mut c = control();
        c.heading = [0.0, 1.0, 0.0];
        let half = push(&c, &ball([0.0; 3], 0.5));
        assert!(close(half.torque, [0.0, 0.0, 5.6]), "{:?}", half.torque);
        // m a paddle afloat = 2 x 8 x 0.5 x 0.5.
        assert!(close(half.force, [0.0, 4.0, 0.0]), "{:?}", half.force);
        let floating = push(
            &c,
            &RollingBody {
                grounded: false,
                ..ball([0.0; 3], 1.0)
            },
        );
        assert!(close(floating.torque, [0.0; 3]));
        assert!(close(floating.force, [0.0, 8.0, 0.0]));
    }

    // No heading or a heading straight up coasts; a jump leaves along up.
    #[test]
    fn coasting_pushes_nothing_and_a_jump_leaves_the_ground() {
        let mut c = control();
        let none = RollingPush {
            force: [0.0; 3],
            torque: [0.0; 3],
        };
        assert_eq!(push(&c, &ball([0.0; 3], 0.0)), none);
        c.heading = [1.0, 0.0, 0.0];
        assert_eq!(push(&c, &ball([0.0; 3], 0.0)), none);
        assert!(close(
            jump_impulse(&c, &ball([0.0; 3], 0.0)),
            [10.0, 0.0, 0.0]
        ));
        // A body already rising takes only what brings it to the jump speed,
        // so a second press on the way up adds nothing.
        assert!(close(
            jump_impulse(&c, &ball([2.0, 0.0, 0.0], 0.0)),
            [6.0, 0.0, 0.0]
        ));
        assert_eq!(jump_impulse(&c, &ball([7.0, 0.0, 0.0], 0.0)), [0.0; 3]);
        // Falling, it still leaves at the jump speed.
        assert!(close(
            jump_impulse(&c, &ball([-3.0, 0.0, 0.0], 0.0)),
            [16.0, 0.0, 0.0]
        ));
    }

    // Off the ground the drive does not wind up spin, and spin already wound
    // up counts toward the top speed, so landing cannot carry the body past
    // it.
    #[test]
    fn spin_counts_toward_the_top_speed_and_only_the_ground_rolls() {
        let mut c = control();
        c.heading = [0.0, 0.0, 1.0];
        let airborne = RollingBody {
            grounded: false,
            ..ball([0.0; 3], 0.0)
        };
        assert_eq!(push(&c, &airborne).torque, [0.0; 3]);
        // Rolling toward +Z with up +X spins about -Y: r w = 0.5 x 20 = 10.
        let spun = RollingBody {
            angular_velocity: [0.0, -20.0, 0.0],
            ..ball([0.0; 3], 0.0)
        };
        assert_eq!(push(&c, &spun).torque, [0.0; 3]);
        let half = RollingBody {
            angular_velocity: [0.0, -10.0, 0.0],
            ..ball([0.0, 0.0, 1.0], 0.0)
        };
        assert!((push(&c, &half).torque[1] + 5.6).abs() < 1e-4);
    }
}
