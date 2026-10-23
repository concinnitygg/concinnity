// A rolling drive steered by a script on the simulation clock rather than by
// a player: the keys a benchmark authors, and the runtime script physics
// reads every tick.

use alloc::vec::Vec;

use crate::components::RollingControl;
use concinnity_core::math::vec3::{cross, dot, normalize_or, scale, sub};
use concinnity_core::math::{cos, sin};

/// One stretch of a scripted [`RollingDrive`](#rollingdrive): how the drive
/// is steered from `seconds` until the next key.
///
/// The heading is a compass bearing, taken at the body every tick: degrees
/// clockwise from north along the ground, north being toward the
/// [`Planet`](#planet)'s north pole, or toward `-Z` without a planet. It
/// stays true however far the body rolls.
///
/// ```rust
/// # use concinnity_engine::components::DriveKey;
/// DriveKey { seconds: 3.0, heading_deg: 90.0, throttle: 1.0, sprint: true, jump: false };
/// ```
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct DriveKey {
    /// When the key takes over, in seconds after a benchmark's warm-up.
    pub seconds: f32,
    /// Where to roll, in degrees clockwise from north.
    pub heading_deg: f32,
    /// How hard to drive, from 0 (coast) to 1.
    #[asset(default = 1.0)]
    pub throttle: f32,
    /// Whether to sprint.
    pub sprint: bool,
    /// Jump as the key takes over, if the body is on the ground.
    pub jump: bool,
}

/// A rolling drive's steering, scripted on the simulation clock.
///
/// Runtime-only. Put on a [`RollingControl`]'s entity, it steers the drive
/// in place of whatever writes the control: the physics reads the key due at
/// each tick's own moment, so the steering is the same however frames group
/// their ticks.
#[derive(Debug, Clone, PartialEq)]
pub struct DriveScript {
    // In time order, each key's `seconds` on the simulation clock.
    keys: Vec<DriveKey>,
}

impl DriveScript {
    /// The script `keys` make, each key's moment moved `offset_seconds` later
    /// onto the simulation clock.
    pub fn new(keys: &[DriveKey], offset_seconds: f32) -> Self {
        let mut keys: Vec<DriveKey> = keys
            .iter()
            .map(|key| DriveKey {
                seconds: key.seconds + offset_seconds,
                ..*key
            })
            .collect();
        keys.sort_by(|a, b| a.seconds.total_cmp(&b.seconds));
        Self { keys }
    }

    /// The key steering at `seconds` on the simulation clock and its index,
    /// or `None` before the first.
    pub(crate) fn key_at(&self, seconds: f64) -> Option<(usize, &DriveKey)> {
        let next = self
            .keys
            .iter()
            .position(|key| f64::from(key.seconds) > seconds)
            .unwrap_or(self.keys.len());
        let at = next.checked_sub(1)?;
        Some((at, &self.keys[at]))
    }

    /// Steer `control` by `key` at a body whose up is `up`, north lying along
    /// `north` before it is laid along the ground.
    pub(crate) fn steer(
        key: &DriveKey,
        control: &mut RollingControl,
        up: [f32; 3],
        north: [f32; 3],
    ) {
        control.heading = scale(
            compass(key.heading_deg, up, north),
            key.throttle.clamp(0.0, 1.0),
        );
        control.sprint = key.sprint;
    }
}

/// The unit direction along the ground `heading_deg` clockwise from north,
/// at a body whose up is `up`, with north along `north` before it is laid on
/// the ground. At a pole north is the world axis least parallel to `up`.
pub(crate) fn compass(heading_deg: f32, up: [f32; 3], north: [f32; 3]) -> [f32; 3] {
    let along_ground = |v: [f32; 3]| sub(v, scale(up, dot(v, up)));
    let flat = along_ground(north);
    let north = if dot(flat, flat) > 1e-12 {
        normalize_or(flat, 1e-6, flat)
    } else {
        const AXES: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let axis = AXES
            .into_iter()
            .min_by(|a, b| dot(*a, up).abs().total_cmp(&dot(*b, up).abs()))
            .unwrap_or(AXES[0]);
        normalize_or(along_ground(axis), 1e-6, axis)
    };
    let east = cross(north, up);
    let h = heading_deg.to_radians();
    normalize_or(
        core::array::from_fn(|i| north[i] * cos(h) + east[i] * sin(h)),
        1e-6,
        north,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-5)
    }

    // On level ground north is -Z and east +X; on a planet's side the same
    // bearing lies along that ground.
    #[test]
    fn a_bearing_is_clockwise_from_north_along_the_ground() {
        let up = [0.0, 1.0, 0.0];
        let north = [0.0, 0.0, -1.0];
        assert!(close(compass(0.0, up, north), [0.0, 0.0, -1.0]));
        assert!(close(compass(90.0, up, north), [1.0, 0.0, 0.0]));
        assert!(close(compass(180.0, up, north), [0.0, 0.0, 1.0]));
        let tilted = normalize_or([0.3, 1.0, 0.0], 1e-6, up);
        let east = compass(90.0, tilted, north);
        assert!(dot(east, tilted).abs() < 1e-6, "along the ground");
        assert!(east[0] > 0.9, "{east:?}");
    }

    // At a pole north still lies along the ground, so every bearing does.
    #[test]
    fn a_bearing_at_a_pole_lies_along_the_ground() {
        let up = [0.0, 0.0, -1.0];
        for heading in [0.0, 90.0, 225.0] {
            let along = compass(heading, up, [0.0, 0.0, -1.0]);
            assert!(dot(along, up).abs() < 1e-6, "{along:?}");
            assert!((dot(along, along) - 1.0).abs() < 1e-5, "{along:?}");
        }
    }

    // Keys take over in time order from their own moment, shifted onto the
    // simulation clock; before the first nothing steers.
    #[test]
    fn the_key_due_is_the_last_one_started() {
        let key = |seconds, heading_deg| DriveKey {
            seconds,
            heading_deg,
            ..DriveKey::default()
        };
        let script = DriveScript::new(&[key(5.0, 90.0), key(0.0, 10.0)], 2.0);
        assert!(script.key_at(1.9).is_none());
        assert_eq!(
            script.key_at(2.0).map(|(i, k)| (i, k.heading_deg)),
            Some((0, 10.0))
        );
        assert_eq!(script.key_at(6.99).map(|(i, _)| i), Some(0));
        assert_eq!(
            script.key_at(7.0).map(|(i, k)| (i, k.heading_deg)),
            Some((1, 90.0))
        );
        assert_eq!(script.key_at(1e9).map(|(i, _)| i), Some(1));
    }

    // A key's throttle scales the heading it steers by and its sprint
    // carries over.
    #[test]
    fn a_key_steers_the_control() {
        let mut control = RollingControl::new(&crate::components::RollingDrive::default());
        let key = DriveKey {
            heading_deg: 90.0,
            throttle: 0.5,
            sprint: true,
            ..DriveKey::default()
        };
        DriveScript::steer(&key, &mut control, [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(close(control.heading, [0.5, 0.0, 0.0]));
        assert!(control.sprint);
    }
}
