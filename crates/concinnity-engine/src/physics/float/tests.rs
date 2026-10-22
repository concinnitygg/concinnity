use super::*;
use crate::components::{WaterSurface, WaterWave};
use crate::physics::{DynamicParams, LayerMask, SimConfig};
use crate::render::environment::wind::WindField;
use alloc::vec::Vec;
use concinnity_core::math::SaturatingFrom;

const TICK: f32 = 1.0 / 60.0;

// Fresh water at rest at `y = 0`, with `waves` of swell.
fn pond(waves: Vec<WaterWave>) -> WaterQuery {
    let surface = WaterSurface {
        extent: [200.0, 200.0],
        density: 1000.0,
        waves,
        subdivisions: 255,
        ..Default::default()
    };
    WaterQuery::new([&surface], &WindField::STILL, None)
}

struct Float {
    sim: Simulation,
    body: BodyHandle,
    floater: Floater,
    water: WaterQuery,
    time: f64,
}

// How a test body is built: its density, its drag, how gravity pulls it.
#[derive(Clone, Copy)]
struct Make {
    density: f32,
    drag: f32,
    gravity_scale: f32,
}

impl Make {
    fn dense(density: f32) -> Self {
        Self {
            density,
            drag: 6.0,
            gravity_scale: 1.0,
        }
    }
}

impl Float {
    fn new(shape: ColliderShape, density: f32, at: [f32; 3], water: WaterQuery) -> Self {
        Self::made(shape, Make::dense(density), at, water)
    }

    fn made(shape: ColliderShape, make: Make, at: [f32; 3], water: WaterQuery) -> Self {
        let mut sim = Simulation::new(SimConfig::default(), 4);
        sim.reserve_fluid_loads(1);
        let volume = Samples::of(&shape).volume();
        let body = sim
            .add_dynamic(
                &shape,
                at,
                [0.0; 3],
                DynamicParams {
                    mass: make.density * volume,
                    linear_damping: 0.0,
                    gravity_scale: make.gravity_scale,
                    ..DynamicParams::default()
                },
                LayerMask::ALL,
            )
            .unwrap();
        let floater = Floater::new(
            &shape,
            Buoyancy {
                linear_drag: make.drag,
                angular_drag: make.drag,
            },
            0.0,
        );
        Self {
            sim,
            body,
            floater,
            water,
            time: 0.0,
        }
    }

    fn run(&mut self, seconds: f32, dt: f32) {
        let steps = usize::saturating_from((seconds / dt).round());
        for _ in 0..steps {
            self.water.prepare(self.time);
            self.floater.push(
                self.body,
                &mut self.sim,
                &self.water,
                &LocalFrame::AUTHORED,
                self.time,
                dt,
            );
            self.sim.step(dt);
            self.time += f64::from(dt);
        }
    }

    fn state(&self) -> crate::physics::BodyState {
        self.sim.body_state(self.body).unwrap()
    }
}

// A box half as dense as the water settles half submerged, its top as far
// above the water as its bottom is below.
#[test]
fn a_box_floats_at_its_archimedes_draft() {
    let shape = ColliderShape::Cuboid {
        half_extents: [0.5, 0.25, 0.5],
    };
    for (density, center) in [(500.0, 0.0), (250.0, 0.125), (800.0, -0.15)] {
        let mut f = Float::new(shape, density, [0.0, center + 0.3, 0.0], pond(Vec::new()));
        f.run(12.0, TICK);
        let y = f.state().position[1];
        assert!((y - center).abs() < 0.01, "{density}: {y} vs {center}");
        assert!(f.floater.afloat());
    }
}

// A ball's draft follows its cap exactly: a quarter as dense as the water
// floats with a quarter of its volume under.
#[test]
fn a_ball_floats_on_its_cap() {
    let r = 0.5;
    let mut f = Float::new(
        ColliderShape::Ball { radius: r },
        250.0,
        [0.0, 0.5, 0.0],
        pond(Vec::new()),
    );
    f.run(12.0, TICK);
    let h = r - f.state().position[1];
    let share = h * h * (3.0 * r - h) / (4.0 * r * r * r);
    assert!((share - 0.25).abs() < 0.01, "{share}");
}

// Denser than the water, a body sinks.
#[test]
fn a_dense_body_sinks() {
    let shape = ColliderShape::Cuboid {
        half_extents: [0.3; 3],
    };
    let mut f = Float::new(shape, 2500.0, [0.0, 0.5, 0.0], pond(Vec::new()));
    f.run(6.0, TICK);
    assert!(f.state().position[1] < -5.0, "{}", f.state().position[1]);
}

// The water slows a body moving through it, and stops it spinning.
#[test]
fn drag_damps_motion() {
    let shape = ColliderShape::Cuboid {
        half_extents: [0.5, 0.25, 0.5],
    };
    let mut f = Float::new(shape, 500.0, [0.0, 0.0, 0.0], pond(Vec::new()));
    f.run(2.0, TICK);
    f.sim
        .set_body_velocity(f.body, [3.0, 0.0, 0.0], [0.0, 2.0, 0.0]);
    f.run(4.0, TICK);
    let s = f.state();
    assert!(s.linear_velocity[0].abs() < 0.1, "{:?}", s.linear_velocity);
    assert!(
        s.angular_velocity[1].abs() < 0.1,
        "{:?}",
        s.angular_velocity
    );
}

// On a sea of one long swell a short plank lies along the water: its pitch
// follows the wave's slope under it.
#[test]
fn pitch_follows_the_swell() {
    let swell = WaterWave {
        amplitude: 0.3,
        wavelength: 24.0,
        direction: [1.0, 0.0],
        steepness: 0.0,
    };
    let water = pond(alloc::vec![swell]);
    let shape = ColliderShape::Cuboid {
        half_extents: [1.0, 0.1, 0.3],
    };
    let mut f = Float::new(shape, 400.0, [0.0, 0.0, 0.0], water);
    f.run(4.0, TICK);
    let mut pitch = Vec::new();
    let mut slope = Vec::new();
    for _ in 0..240 {
        f.run(TICK, TICK);
        let s = f.state();
        // The plank's long axis, and how far it rises per meter.
        let along = quat_rotate(s.rotation, [1.0, 0.0, 0.0]);
        pitch.push(along[1] / along[0].hypot(along[2]));
        let spot = WaterSpot::OPEN;
        let w = f
            .water
            .sample(&spot, &LocalFrame::AUTHORED, s.position, f.time)
            .unwrap();
        slope.push(-w.normal[0] / w.normal[1]);
    }
    let n = pitch.len() as f32;
    let mean = |v: &[f32]| v.iter().sum::<f32>() / n;
    let (mp, ms) = (mean(&pitch), mean(&slope));
    let cov: f32 = pitch
        .iter()
        .zip(&slope)
        .map(|(p, s)| (p - mp) * (s - ms))
        .sum();
    let var = |v: &[f32], m: f32| v.iter().map(|x| (x - m) * (x - m)).sum::<f32>();
    let corr = cov / (var(&pitch, mp) * var(&slope, ms)).sqrt();
    let gain = (var(&pitch, mp) / var(&slope, ms)).sqrt();
    assert!(corr > 0.9, "correlation {corr}");
    assert!((0.7..1.3).contains(&gain), "gain {gain}");
}

// The same drop floats the same way every time.
#[test]
fn floating_is_deterministic() {
    let swell = WaterWave {
        amplitude: 0.2,
        wavelength: 9.0,
        direction: [0.6, 0.8],
        steepness: 0.5,
    };
    let run = || {
        let shape = ColliderShape::Capsule {
            half_height: 0.6,
            radius: 0.25,
        };
        let mut f = Float::new(shape, 600.0, [0.3, 2.0, -0.2], pond(alloc::vec![swell]));
        f.run(5.0, TICK);
        let s = f.state();
        (s.position, s.rotation, s.linear_velocity)
    };
    assert_eq!(run(), run());
}

// Long steps add no energy: a box dropped onto still water at a few frames a
// step bobs no higher than it fell from and settles to the same draft.
#[test]
fn long_steps_gain_no_energy() {
    let shape = ColliderShape::Cuboid {
        half_extents: [0.5, 0.25, 0.5],
    };
    for dt in [1.0 / 30.0, 1.0 / 15.0] {
        let mut f = Float::new(shape, 500.0, [0.0, 1.0, 0.0], pond(Vec::new()));
        let mut highest_after_entry = f32::MIN;
        let mut entered = false;
        for _ in 0..usize::saturating_from(12.0 / dt) {
            f.run(dt, dt);
            let y = f.state().position[1];
            entered |= y < 0.0;
            if entered {
                highest_after_entry = highest_after_entry.max(y);
            }
        }
        assert!(highest_after_entry < 1.0, "{dt}: {highest_after_entry}");
        assert!(
            f.state().position[1].abs() < 0.02,
            "{dt}: {}",
            f.state().position[1]
        );
    }
}

// On still water a floating body settles and falls asleep, and stays at its
// draft asleep.
#[test]
fn still_water_lets_a_body_sleep() {
    let shape = ColliderShape::Cuboid {
        half_extents: [0.5, 0.25, 0.5],
    };
    let mut f = Float::new(shape, 500.0, [0.0, 0.3, 0.0], pond(Vec::new()));
    f.run(20.0, TICK);
    let s = f.state();
    assert!(s.sleeping);
    f.run(2.0, TICK);
    assert!(f.state().sleeping);
    assert!(f.state().position[1].abs() < 0.01);
}

// With no drag at all nothing takes energy out, and nothing may put any in:
// a box bobbing on still water, and a stiff light one (a thousandth of the
// water's density, as a body left at mass 0 is) bobbing hundreds of times a
// second's worth, keep their swing over many periods at either step.
#[test]
fn undamped_bobbing_does_not_grow() {
    let shape = ColliderShape::Cuboid {
        half_extents: [0.5, 0.5, 0.5],
    };
    for density in [500.0, 1.0] {
        for dt in [1.0 / 60.0, 1.0 / 30.0] {
            let make = Make {
                density,
                drag: 0.0,
                gravity_scale: 1.0,
            };
            let rest = 0.5 - density / 1000.0;
            let mut f = Float::made(shape, make, [0.0, rest + 0.05, 0.0], pond(Vec::new()));
            let swing = |f: &mut Float, seconds: f32| {
                let mut hi = f32::MIN;
                for _ in 0..usize::saturating_from(seconds / dt) {
                    f.run(dt, dt);
                    hi = hi.max(f.state().position[1]);
                }
                hi - rest
            };
            let early = swing(&mut f, 2.0);
            swing(&mut f, 16.0);
            let late = swing(&mut f, 2.0);
            assert!(
                early < 0.06 && late < early + 2e-3,
                "{density} {dt}: {early} -> {late}"
            );
            assert!(f.floater.afloat(), "{density} {dt}: left the water");
        }
    }
}

// The water lifts as gravity pulls, gravity scale and all: at half gravity
// a body floats at the same draft, and with none it stays where it lies.
#[test]
fn the_lift_follows_the_gravity_scale() {
    let shape = ColliderShape::Cuboid {
        half_extents: [0.5, 0.25, 0.5],
    };
    let half = Make {
        gravity_scale: 0.5,
        ..Make::dense(500.0)
    };
    let mut f = Float::made(shape, half, [0.0, 0.2, 0.0], pond(Vec::new()));
    f.run(20.0, TICK);
    assert!(
        f.state().position[1].abs() < 0.01,
        "{}",
        f.state().position[1]
    );
    let none = Make {
        gravity_scale: 0.0,
        ..Make::dense(500.0)
    };
    let mut f = Float::made(shape, none, [0.0, -0.1, 0.0], pond(Vec::new()));
    f.run(5.0, TICK);
    assert!(
        (f.state().position[1] + 0.1).abs() < 1e-4,
        "{}",
        f.state().position[1]
    );
}

// An upright capsule's draft is the one its shape gives: the water's level
// at which the pieces displace a given share of it, against the exact
// volume of a capsule below that level, from within its lower cap to its
// cylinder.
#[test]
fn an_upright_capsule_floats_at_its_draft() {
    use core::f32::consts::PI;
    let (r, hh) = (0.3f32, 0.6f32);
    let samples = Samples::of(&ColliderShape::Capsule {
        half_height: hh,
        radius: r,
    });
    let total = samples.volume();
    let level = [0.0, 0.0, 0.0, 1.0];
    // Volume under the water `depth` above the capsule's center.
    let pieces = |depth: f32| -> f32 {
        samples
            .pieces()
            .iter()
            .map(|p| submerged(p, level, [0.0, 1.0, 0.0], depth - p.offset[1]).volume)
            .sum()
    };
    let exact = |depth: f32| -> f32 {
        let h = depth + hh + r;
        let cap = |h: f32| PI * h * h * (3.0 * r - h) / 3.0;
        if h <= r {
            cap(h.max(0.0))
        } else if h <= r + 2.0 * hh {
            cap(r) + PI * r * r * (h - r)
        } else {
            total - cap((2.0 * (r + hh) - h).max(0.0))
        }
    };
    let solve = |f: &dyn Fn(f32) -> f32, share: f32| {
        let (mut lo, mut hi) = (-hh - r, hh + r);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if f(mid) < share * total {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    };
    for share in [0.05, 0.1, 0.25, 0.5, 0.75, 0.95] {
        let got = solve(&pieces, share);
        let want = solve(&exact, share);
        assert!((got - want).abs() < 0.01, "{share}: {got} vs {want}");
    }
}

// A held body is only read: it is never pushed, but whether it is in the
// water stays current, so a body held over the sea can be let go.
#[test]
fn a_held_body_over_water_reads_afloat() {
    let shape = ColliderShape::Cuboid {
        half_extents: [0.5, 0.25, 0.5],
    };
    let mut f = Float::new(shape, 500.0, [0.0, 3.0, 0.0], pond(Vec::new()));
    f.sim.make_kinematic(f.body);
    f.run(TICK, TICK);
    assert!(!f.floater.afloat());
    f.sim.teleport_body(f.body, [0.0, 0.0, 0.0], [0.0; 3]);
    f.run(TICK, TICK);
    assert!(f.floater.afloat());
    assert_eq!(f.sim.body_pose_quat(f.body).unwrap().0, [0.0; 3]);
}
