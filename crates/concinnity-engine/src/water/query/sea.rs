//! A planet sea read on the CPU the way `water.hlsl`'s `WATER_SEA` path
//! draws it: at a point's place on the planet, each cube face's copy of the
//! simulated sea blended by its share, damped over the shallows, and the
//! shore's trains on top.

use super::{PatchShift, ShoreSpot, WaterPatch, WaterSample, WaterSpot, unit};
use crate::planet::{DVec3, LocalFrame, PlanetShape};
use crate::water::WaterBody;
use crate::water::ocean::band::OceanBand;
use crate::water::sea::SeaFace;
use crate::water::sea::shore::{FOAM_RIDE, ShoreNoise, ShorePoint, ShoreWaves};
use concinnity_core::math::{dvec3, narrow_f32};

// How many times the rest point under a place is refined against the
// horizontal displacement.
const REST_STEPS: usize = 1;

// The step the shore's surface velocity is differenced over, in seconds.
const SHORE_RATE_STEP: f64 = 1.0e-2;

/// The waves over one rest point of the sea, in the planet's axes.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct SeaWaves {
    pub(super) height: f64,
    pub(super) shift: DVec3,
    pub(super) slope: DVec3,
    pub(super) velocity: DVec3,
}

// The spot at `world`, on a body that is a planet sea.
pub(super) fn spot(body: &WaterBody, planet: &PlanetShape, world: DVec3) -> Option<WaterSpot> {
    let sea = body.sea.as_ref()?;
    let dir = dvec3::normalize_or(dvec3::sub(world, sea.sphere.center), [0.0, 1.0, 0.0]);
    let shore = sea.shore.as_ref().and_then(|shore| {
        let (inland, gradient) = planet.inland(dir)?;
        Some(ShoreSpot {
            inland,
            gradient,
            noise: shore.noise_at(dir),
        })
    });
    Some(WaterSpot { dir, shore })
}

// The sea at local point `at` of `frame`, and the horizontal displacement
// of the water that moved there.
pub(super) fn sample(
    body: &WaterBody,
    band: Option<&OceanBand>,
    spot: &WaterSpot,
    frame: &LocalFrame,
    at: [f32; 3],
    time: f64,
) -> Option<(WaterSample, DVec3, Option<ShorePoint>)> {
    let reader = Reader::of(body, band, spot)?;
    let (dir, altitude) = reader.place(frame, at);
    let mut rest = dir;
    for _ in 0..REST_STEPS {
        let shift = reader.waves(rest, time, false).shift;
        rest = reader.rest_under(dir, shift);
    }
    let w = reader.waves(rest, time, true);
    let up = frame.dir_to_local(rest);
    let slope = frame.dir_to_local(w.slope);
    let sample = WaterSample {
        depth: narrow_f32(w.height - altitude),
        up,
        normal: unit([up[0] - slope[0], up[1] - slope[1], up[2] - slope[2]], up),
        velocity: frame.dir_to_local(w.velocity),
        density: body.surface.density,
    };
    let shore = reader.shore(rest, None).map(|(_, point, _)| point);
    Some((sample, w.shift, shore))
}

// How far the sea's surface lies above local point `at` of `frame`, the
// water there moved by `patch`'s displacement.
pub(super) fn depth(
    body: &WaterBody,
    band: Option<&OceanBand>,
    spot: &WaterSpot,
    patch: &WaterPatch,
    frame: &LocalFrame,
    at: [f32; 3],
    time: f64,
) -> Option<f32> {
    let reader = Reader::of(body, band, spot)?;
    let PatchShift::Sea(shift) = patch.shift else {
        return None;
    };
    let (dir, altitude) = reader.place(frame, at);
    let rest = reader.rest_under(dir, shift);
    Some(narrow_f32(
        reader.height(rest, patch.shore.as_ref(), time) - altitude,
    ))
}

// What one sample reads the sea from.
struct Reader<'a> {
    body: &'a WaterBody,
    band: Option<&'a OceanBand>,
    spot: &'a WaterSpot,
    radius: f64,
    blend: f64,
}

impl<'a> Reader<'a> {
    fn of(body: &'a WaterBody, band: Option<&'a OceanBand>, spot: &'a WaterSpot) -> Option<Self> {
        let sea = body.sea.as_ref()?;
        Some(Self {
            body,
            band,
            spot,
            radius: sea.sphere.radius,
            blend: sea.sphere.blend_rate(),
        })
    }

    // Local point `at`'s direction from the planet's center and its height
    // above the sea's rest.
    fn place(&self, frame: &LocalFrame, at: [f32; 3]) -> (DVec3, f64) {
        let center = self.body.sea.as_ref().map_or([0.0; 3], |s| s.sphere.center);
        let rel = dvec3::sub(frame.to_world(at), center);
        (
            dvec3::normalize_or(rel, [0.0, 1.0, 0.0]),
            dvec3::length(rel) - self.radius,
        )
    }

    // The rest direction whose water `shift` moved to `dir`.
    fn rest_under(&self, dir: DVec3, shift: DVec3) -> DVec3 {
        dvec3::normalize_or(dvec3::sub(dir, dvec3::scale(shift, 1.0 / self.radius)), dir)
    }

    // Only the height of the waves over rest direction `rest`, its shore's
    // shoaling taken from `near` when given.
    fn height(&self, rest: DVec3, near: Option<&ShorePoint>, time: f64) -> f64 {
        let shore = self.shore(rest, near);
        let damping = shore.as_ref().map_or(1.0, |(waves, point, _)| {
            waves.cascade_damping(point.depth)[0]
        });
        let mut h = 0.0;
        if let Some(band) = self.band
            && damping > 0.0
        {
            for (face, w) in SeaFace::shares(rest, self.blend) {
                if w > 0.0 {
                    h +=
                        damping * w * f64::from(band.height(time, face.project(self.radius, rest)));
                }
            }
        }
        if let Some((waves, point, noise)) = shore {
            h += waves.height(&point, &noise, waves.phases_at(time));
        }
        h
    }

    // The waves over rest direction `rest`; `full` adds the slope and the
    // velocity to the height and the shift.
    fn waves(&self, rest: DVec3, time: f64, full: bool) -> SeaWaves {
        let shore = self.shore(rest, None);
        let damping = shore.as_ref().map_or(1.0, |(waves, point, _)| {
            waves.cascade_damping(point.depth)[0]
        });
        let mut out = match self.band {
            Some(band) if damping > 0.0 => self.faces(band, rest, time, full, damping),
            _ => SeaWaves::default(),
        };
        if let Some((waves, point, noise)) = shore {
            let phases = waves.phases_at(time);
            let s = waves.surface(&point, &noise, phases);
            out.height += s.height;
            out.shift = dvec3::add(out.shift, s.shift);
            if full {
                out.slope = dvec3::add(out.slope, s.slope);
                let later = waves.surface(&point, &noise, waves.phases_at(time + SHORE_RATE_STEP));
                let k = 1.0 / SHORE_RATE_STEP;
                let rise = (later.height - s.height) * k;
                let sway = dvec3::scale(dvec3::sub(later.shift, s.shift), k);
                // A breaking crest's bore carries the water at its front in
                // toward the beach, as fast as its fresh foam rides.
                let bore = dvec3::scale(s.celerity, FOAM_RIDE * s.foam);
                out.velocity = dvec3::add(
                    out.velocity,
                    dvec3::add(dvec3::add(dvec3::scale(rest, rise), sway), bore),
                );
            }
        }
        out
    }

    // The shore's trains and where `rest` lies against the shore, offset from
    // the body's spot along the sea; its shoaling from `near` when given.
    fn shore(
        &self,
        rest: DVec3,
        near: Option<&ShorePoint>,
    ) -> Option<(&ShoreWaves, ShorePoint, ShoreNoise)> {
        let waves = self.body.sea.as_ref()?.shore.as_ref()?;
        let spot = self.spot.shore.as_ref()?;
        let offset = dvec3::scale(dvec3::sub(rest, self.spot.dir), self.radius);
        let inland = spot.inland + dvec3::dot(spot.gradient, offset);
        let noise = ShoreNoise {
            phase: spot.noise.phase + dvec3::dot(spot.noise.gradient, offset),
            ..spot.noise
        };
        let point = match near {
            Some(near) => waves.point_beside(near, inland, spot.gradient),
            None => waves.point(inland, spot.gradient),
        };
        Some((waves, point, noise))
    }

    // The simulated sea over `rest`: each face around its cube corner, by
    // its share, `damping` of it.
    fn faces(
        &self,
        band: &OceanBand,
        rest: DVec3,
        time: f64,
        full: bool,
        damping: f64,
    ) -> SeaWaves {
        let mut out = SeaWaves::default();
        for (face, w) in SeaFace::shares(rest, self.blend) {
            if w <= 0.0 {
                continue;
            }
            let share = damping * w;
            let s = face.project(self.radius, rest);
            let stretch = face.stretch(rest);
            let [tu, tv] = face.tangents(rest);
            let along = |a: [f32; 2], k: f64| {
                dvec3::scale(
                    dvec3::add(
                        dvec3::scale(tu, f64::from(a[0])),
                        dvec3::scale(tv, f64::from(a[1])),
                    ),
                    share * k,
                )
            };
            if !full {
                out.shift = dvec3::add(out.shift, along(band.shift(time, s), 1.0 / stretch));
                continue;
            }
            let b = band.sample(time, s);
            out.height += share * f64::from(b.height);
            out.shift = dvec3::add(out.shift, along(b.shift, 1.0 / stretch));
            out.slope = dvec3::add(out.slope, along(b.slope, stretch));
            out.velocity = dvec3::add(
                out.velocity,
                dvec3::add(
                    along(b.shift_rate, 1.0 / stretch),
                    dvec3::scale(rest, share * f64::from(b.rise_rate)),
                ),
            );
        }
        out
    }
}
