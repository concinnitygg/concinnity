// The knobs a step is tuned by, in one place so a caller can reason about them
// together and so every default is stated once.
//
// The contact stiffness is expressed as a frequency and a damping ratio rather
// than as a bias factor: a frequency keeps its meaning when the tick rate or
// the substep count changes, where a raw bias factor does not.

use crate::physics::GRAVITY;

/// Tuning for one simulation.
///
/// [`SimConfig::default`] is what the engine runs with; the fields are public
/// so a caller reproducing a recorded step, or trading stability for speed,
/// can set them explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimConfig {
    /// Downward acceleration in world units per second squared.
    pub gravity: f32,
    /// Velocity/position passes per step. More substeps buy stiffer stacks at
    /// a near-linear cost; below `1` the step does nothing.
    pub substeps: u32,
    /// Contact stiffness as a frequency in hertz. Clamped against the substep
    /// rate, since a contact cannot be stiffer than the rate it is solved at.
    pub contact_hertz: f32,
    /// Contact damping ratio. Above `1` the contact is overdamped, which is
    /// what stops a resting stack from breathing.
    pub contact_damping_ratio: f32,
    /// Joint stiffness as a frequency in hertz, clamped against the substep
    /// rate the same way the contact one is. Joints are held stiffer than
    /// contacts because a contact that sinks a millimetre is still right and
    /// a joint that gives a millimetre reads as broken.
    pub joint_hertz: f32,
    /// Joint damping ratio. Far lower than the contact one, and damped at all
    /// only so a joint built out of place settles rather than ringing as it
    /// closes: the correction's damping is what bleeds a pendulum's swing, so
    /// every part of it that is not needed is energy a joint gives back.
    pub joint_damping_ratio: f32,
    /// Ceiling on the speed penetration is pushed out at, so a deep overlap
    /// resolves over several steps instead of launching.
    pub max_push_velocity: f32,
    /// Penetration the solver leaves alone. Resting contacts settle just
    /// inside the surface, which keeps them from being lost and remade.
    pub linear_slop: f32,
    /// Gap within which a contact is still created, so an approaching body is
    /// slowed before it overlaps rather than after.
    pub speculative_margin: f32,
    /// Approach speed below which a contact does not bounce, whatever the
    /// restitution. Without it a bouncy body never comes to rest.
    pub restitution_threshold: f32,
    /// Whether a settled island stops being simulated.
    pub allow_sleep: bool,
    /// Speed below which a body counts as still, in world units per second.
    pub sleep_linear_velocity: f32,
    /// Spin below which a body counts as still, in radians per second.
    pub sleep_angular_velocity: f32,
    /// How long a whole island must be still before it sleeps, in seconds.
    pub time_to_sleep: f32,
    /// Slack added to a body's bounds, so small motion does not invalidate
    /// them.
    pub bounds_margin: f32,
    /// Whether a body that outruns the step's own contact test is caught by
    /// a sweep. On for every freely simulated and position-driven body, since
    /// without it one passes straight through thin geometry.
    pub ccd_enabled: bool,
    /// Fraction of a body's thinnest dimension its motion over one step has
    /// to exceed before that sweep runs.
    ///
    /// At or below `1` nothing can outrun the gate: passing through anything
    /// takes more motion than the mover's own width, so the sweep is already
    /// armed by the time tunnelling is possible. The margin below `1` covers
    /// the rotation the sweep does not model.
    pub ccd_motion_ratio: f32,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            gravity: GRAVITY,
            substeps: 4,
            contact_hertz: 30.0,
            contact_damping_ratio: 10.0,
            joint_hertz: 60.0,
            joint_damping_ratio: 0.5,
            max_push_velocity: 3.0,
            linear_slop: 0.005,
            speculative_margin: 0.02,
            restitution_threshold: 1.0,
            allow_sleep: true,
            sleep_linear_velocity: 0.05,
            sleep_angular_velocity: 0.1,
            time_to_sleep: 0.5,
            bounds_margin: 0.05,
            ccd_enabled: true,
            ccd_motion_ratio: 0.5,
        }
    }
}

impl SimConfig {
    /// Substeps as a positive count, so a caller cannot configure a step that
    /// integrates nothing.
    pub(crate) fn substep_count(&self) -> u32 {
        self.substeps.max(1)
    }
}

/// The three coefficients a soft constraint is solved with, derived once per
/// step from a frequency, a damping ratio, and the substep timestep.
///
/// This is the implicit-spring formulation: `bias_rate` converts a position
/// error into a velocity, and the two scales split each impulse between
/// correcting the error now and remembering it for the next iteration, which
/// is what keeps the correction from injecting energy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Softness {
    pub(crate) bias_rate: f32,
    pub(crate) mass_scale: f32,
    pub(crate) impulse_scale: f32,
}

impl Softness {
    pub(crate) fn new(hertz: f32, damping_ratio: f32, h: f32) -> Self {
        if hertz <= 0.0 || h <= 0.0 {
            return Softness {
                bias_rate: 0.0,
                mass_scale: 1.0,
                impulse_scale: 0.0,
            };
        }
        let omega = 2.0 * core::f32::consts::PI * hertz;
        let a1 = 2.0 * damping_ratio + h * omega;
        let a2 = h * omega * a1;
        let a3 = 1.0 / (1.0 + a2);
        Softness {
            bias_rate: omega / a1,
            mass_scale: a2 * a3,
            impulse_scale: a3,
        }
    }

    /// A constraint solved with no bias at all: the relax pass, which removes
    /// the velocity the biased pass added without reopening the penetration.
    pub(crate) const RIGID: Softness = Softness {
        bias_rate: 0.0,
        mass_scale: 1.0,
        impulse_scale: 0.0,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_matches_the_engine_gravity_and_simulates_something() {
        let c = SimConfig::default();
        assert_eq!(c.gravity, GRAVITY);
        assert!(c.substep_count() >= 1);
        assert!(c.speculative_margin > c.linear_slop);
        assert!(
            c.joint_hertz > c.contact_hertz,
            "a joint has to be held stiffer than a contact"
        );
        assert!(c.ccd_enabled, "a fast body must not pass through geometry");
        assert!(
            c.ccd_motion_ratio > 0.0 && c.ccd_motion_ratio <= 1.0,
            "past one width of motion a body can tunnel, so the gate has to \
             be armed before then"
        );
    }

    #[test]
    fn a_zero_substep_config_still_takes_one_pass() {
        let c = SimConfig {
            substeps: 0,
            ..SimConfig::default()
        };
        assert_eq!(c.substep_count(), 1);
    }

    // The coefficients must stay in the range the solver assumes: a positive
    // bias rate, a mass scale in [0, 1], and an impulse scale in [0, 1].
    #[test]
    fn softness_coefficients_stay_in_the_solvers_range() {
        for hertz in [1.0, 30.0, 240.0] {
            for zeta in [0.5, 1.0, 10.0] {
                let s = Softness::new(hertz, zeta, 1.0 / 240.0);
                assert!(s.bias_rate > 0.0, "{hertz} {zeta}: {s:?}");
                assert!((0.0..=1.0).contains(&s.mass_scale), "{hertz} {zeta}: {s:?}");
                assert!(
                    (0.0..=1.0).contains(&s.impulse_scale),
                    "{hertz} {zeta}: {s:?}"
                );
                assert!(
                    (s.mass_scale + s.impulse_scale - 1.0).abs() < 1.0e-5,
                    "the two scales partition one impulse: {s:?}"
                );
            }
        }
    }

    // A stiffer contact pushes harder for the same error.
    #[test]
    fn a_higher_frequency_raises_the_bias_rate() {
        let h = 1.0 / 240.0;
        let soft = Softness::new(10.0, 10.0, h);
        let stiff = Softness::new(60.0, 10.0, h);
        assert!(stiff.bias_rate > soft.bias_rate, "{soft:?} {stiff:?}");
    }

    #[test]
    fn a_disabled_frequency_degrades_to_the_rigid_solve() {
        assert_eq!(Softness::new(0.0, 1.0, 1.0 / 240.0), Softness::RIGID);
        assert_eq!(Softness::new(30.0, 1.0, 0.0), Softness::RIGID);
    }
}
