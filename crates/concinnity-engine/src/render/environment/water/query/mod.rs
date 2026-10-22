//! Where the water is, on the CPU: the surface every `WaterSurface` draws,
//! read at a point and an instant for whatever floats in it.
//!
//! The query resolves the world's water exactly as the renderer does and
//! reads the same waves: a summed surface's displacing waves, a simulated
//! sea's longest waves through its [`OceanBand`], and on a planet sea the
//! faces its waves are laid out on, their damping over the shallows and the
//! shore's trains. A planet sea is read at a point's place on the planet, so
//! a move of the simulated frame changes nothing it reports. What it leaves
//! out is the chop: the simulated sea's shorter cascades and every ripple,
//! a tenth or so of the height's variance.

mod rect;
mod sea;

#[cfg(test)]
mod tests;

use crate::components::WaterSurface;
use crate::planet::{DVec3, LocalFrame, PlanetShape};
use crate::render::environment::water::WaterBody;
use crate::render::environment::water::ocean::OceanQuality;
use crate::render::environment::water::ocean::band::OceanBand;
use crate::render::environment::water::sea::shore::{ShoreNoise, ShorePoint};
use crate::render::environment::wind::WindField;
use alloc::vec::Vec;

/// The water at one point and instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WaterSample {
    /// How far the surface lies above the point along `up`, in meters:
    /// positive under water, negative above it.
    pub depth: f32,
    /// Away from the water's rest: the sea's radial up on a planet, +Y on a
    /// rectangle.
    pub up: [f32; 3],
    /// The surface's normal over the point.
    pub normal: [f32; 3],
    /// How fast the water at the surface moves, in meters per second.
    pub velocity: [f32; 3],
    /// The water's density, in kilograms per cubic meter.
    pub density: f32,
}

/// The water around one body for one tick, read once at its center and
/// shared by its pieces: which surface it floats in, how that surface's water
/// has moved, and the water's up, motion and density there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WaterPatch {
    // Which of the query's surfaces it is.
    body: usize,
    shift: PatchShift,
    // The shore where the center's water rests, for its pieces' shoaling.
    shore: Option<ShorePoint>,
    /// The water at the center.
    pub center: WaterSample,
}

// How far the water around a body moved from its rest.
#[derive(Debug, Clone, Copy, PartialEq)]
enum PatchShift {
    // Along a rectangle's X and Z.
    Flat([f64; 2]),
    // Along a planet sea, in the planet's axes.
    Sea(DVec3),
}

/// The slow part of a planet sea's shore at one place: what a body's
/// samples, all within a few meters of it, share. Taken once per body and
/// reused while the body stays near it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WaterSpot {
    // The place on the planet it was taken at.
    dir: DVec3,
    shore: Option<ShoreSpot>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ShoreSpot {
    inland: f64,
    gradient: DVec3,
    noise: ShoreNoise,
}

impl WaterSpot {
    /// A spot that holds no shore: the open sea, or no planet sea at all.
    pub(crate) const OPEN: Self = Self {
        dir: [0.0, 1.0, 0.0],
        shore: None,
    };

    /// How far, in meters along the sea, `world` lies from where the spot
    /// was taken on a planet of `radius`; 0 off a planet.
    pub(crate) fn distance(&self, planet: Option<&PlanetShape>, world: DVec3) -> f64 {
        let Some(planet) = planet else {
            return 0.0;
        };
        let rel = concinnity_core::math::dvec3::sub(world, planet.center);
        let dir = concinnity_core::math::dvec3::normalize_or(rel, [0.0, 1.0, 0.0]);
        concinnity_core::math::dvec3::length(concinnity_core::math::dvec3::sub(dir, self.dir))
            * planet.radius
    }
}

// One surface as the query reads it.
#[derive(Debug, Clone)]
struct QueryBody {
    body: WaterBody,
    band: Option<usize>,
}

/// The world's water surfaces, readable at any point and instant.
#[derive(Debug, Clone)]
pub(crate) struct WaterQuery {
    bodies: Vec<QueryBody>,
    // One per simulated sea, `None` where its wind raises none.
    bands: Vec<Option<OceanBand>>,
    planet: Option<PlanetShape>,
}

impl WaterQuery {
    /// The water of `surfaces` moved by `wind`, a planet sea lying on
    /// `planet`. The simulated seas are resolved at the full quality: the
    /// waves the query reads are the same at every preset.
    pub(crate) fn new<'a>(
        surfaces: impl IntoIterator<Item = &'a WaterSurface>,
        wind: &WindField,
        planet: Option<&PlanetShape>,
    ) -> Self {
        let (bodies, plan) = WaterBody::resolve_all(surfaces, wind, OceanQuality::FULL, planet);
        let bands = plan
            .simulations
            .iter()
            .map(|s| OceanBand::new(&s.sea, &s.cascades))
            .collect::<Vec<_>>();
        let bodies = bodies
            .into_iter()
            .filter(|b| b.surface.visible)
            .map(|body| {
                let band = body.ocean.as_ref().map(|o| o.simulation);
                QueryBody { body, band }
            })
            .collect();
        Self {
            bodies,
            bands,
            planet: planet.copied(),
        }
    }

    /// Whether the world has no water.
    pub(crate) fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    /// The planet a planet sea lies on.
    pub(crate) fn planet(&self) -> Option<&PlanetShape> {
        self.planet.as_ref()
    }

    /// Make every simulated sea ready to be read `time` seconds into the
    /// world.
    pub(crate) fn prepare(&mut self, time: f64) {
        for band in self.bands.iter_mut().flatten() {
            band.prepare(time);
        }
    }

    /// The shore's slow part at local point `at` of `frame`, for
    /// [`WaterQuery::sample`] to read points near it with.
    pub(crate) fn spot(&self, frame: &LocalFrame, at: [f32; 3]) -> WaterSpot {
        self.bodies
            .iter()
            .find_map(|b| sea::spot(&b.body, self.planet.as_ref()?, frame.to_world(at)))
            .unwrap_or(WaterSpot::OPEN)
    }

    /// The water at local point `at` of `frame`, `time` seconds into the
    /// world, its shore read from `spot`; `None` where no surface lies over
    /// or under it. Where surfaces overlap, the highest is the one read.
    #[cfg(test)]
    pub(crate) fn sample(
        &self,
        spot: &WaterSpot,
        frame: &LocalFrame,
        at: [f32; 3],
        time: f64,
    ) -> Option<WaterSample> {
        self.patch(spot, frame, at, time).map(|p| p.center)
    }

    /// The water around a body centered at local point `at`, for
    /// [`WaterQuery::depth`] to read its pieces against; `None` where no
    /// surface lies over or under it.
    pub(crate) fn patch(
        &self,
        spot: &WaterSpot,
        frame: &LocalFrame,
        at: [f32; 3],
        time: f64,
    ) -> Option<WaterPatch> {
        let mut best: Option<WaterPatch> = None;
        for (i, b) in self.bodies.iter().enumerate() {
            let band = self.band(b);
            let patch = match (&b.body.sea, self.planet.as_ref()) {
                (Some(_), Some(_)) => sea::sample(&b.body, band, spot, frame, at, time)
                    .map(|(s, shift, shore)| (s, PatchShift::Sea(shift), shore)),
                (Some(_), None) => None,
                (None, _) => rect::sample(&b.body, band, at, time)
                    .map(|(s, shift)| (s, PatchShift::Flat(shift), None)),
            };
            if let Some((center, shift, shore)) = patch
                && best.is_none_or(|b| center.depth > b.center.depth)
            {
                best = Some(WaterPatch {
                    body: i,
                    shift,
                    shore,
                    center,
                });
            }
        }
        best
    }

    /// How far the surface of `patch`'s water lies above local point `at`
    /// near its center, in meters along its up: positive under water. Only
    /// the height is read, the water's displacement taken from the center;
    /// `None` off the surface.
    pub(crate) fn depth(
        &self,
        spot: &WaterSpot,
        patch: &WaterPatch,
        frame: &LocalFrame,
        at: [f32; 3],
        time: f64,
    ) -> Option<f32> {
        let b = self.bodies.get(patch.body)?;
        let band = self.band(b);
        match b.body.sea {
            Some(_) => sea::depth(&b.body, band, spot, patch, frame, at, time),
            None => rect::depth(&b.body, band, patch, at, time),
        }
    }

    fn band(&self, b: &QueryBody) -> Option<&OceanBand> {
        b.band.and_then(|i| self.bands.get(i)?.as_ref())
    }
}

// `v` scaled to unit length, or `fallback` when it has none.
fn unit(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let l = concinnity_core::math::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if l > 1e-12 && l.is_finite() {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        fallback
    }
}
