//! The subjects a view is measured against, in the space a frame draws in:
//! where each lies and which of them a camera ray meets first.

use super::facts::{Sighting, Subject, SubjectKind, VIEW_SUBJECTS};
use crate::gfx::frustum::{Aabb, Frustum};
use crate::gfx::render_types::{GroundBoundGpu, LevelLayerGpu, LevelSurfacesGpu, MAX_LEVEL_LAYERS};
use crate::render::shadow::far_anchor::GroundSphere;
use concinnity_core::math::sqrt;
use concinnity_core::math::vec3::{add, dot, length, scale, sub};

/// The planet's slot among [`ViewFacts::subjects`](super::ViewFacts::subjects).
pub(crate) const PLANET_SLOT: usize = 0;
// The first water surface's slot; each level layer keeps its own index after it.
const WATER_SLOT: usize = 1;
/// The prop's slot.
pub(crate) const PROP_SLOT: usize = WATER_SLOT + MAX_LEVEL_LAYERS;
const _: () = assert!(PROP_SLOT + 1 == VIEW_SUBJECTS);

// How far past its box a surface still counts as the prop's: the depth's
// precision, and the box's own rounding.
const PROP_MARGIN: f32 = 0.02;
const PROP_MARGIN_FLOOR: f32 = 0.01;

/// A planet as a view measures it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PlanetSubject {
    /// The ground under the camera.
    pub ground: GroundSphere,
    /// The sphere the ground stays under, with the camera's height over it.
    pub bound: GroundBoundGpu,
}

/// What a camera ray meets first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Hit {
    /// Nothing: the ray leaves the scene.
    Sky,
    /// The subject in this slot, at this distance.
    Subject(usize, f32),
    /// A surface no subject holds, at this distance.
    Other(f32),
}

/// The subjects of one frame, in the space it draws in.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ViewSubjects {
    /// The planet, if the world has one.
    pub planet: Option<PlanetSubject>,
    /// The water surfaces, as reflections meet them; the unused ones off.
    pub water: [LevelLayerGpu; MAX_LEVEL_LAYERS],
    /// The prop a reflection probe rides, as a world box.
    pub prop: Option<Aabb>,
}

impl ViewSubjects {
    /// The subjects of a frame drawn over `ground` under the bound `bound`,
    /// with the water surfaces `level` and the probed prop's box `prop`.
    pub(crate) fn new(
        ground: Option<GroundSphere>,
        bound: &GroundBoundGpu,
        level: &LevelSurfacesGpu,
        prop: Option<Aabb>,
    ) -> Self {
        Self {
            planet: ground
                .filter(|_| bound.sphere[3] > 0.0)
                .map(|ground| PlanetSubject {
                    ground,
                    bound: *bound,
                }),
            water: level.layers,
            prop,
        }
    }

    /// What the camera at `eye` meets first along unit `dir`: a water surface
    /// in front of any opaque surface, the opaque surface `opaque` lying at
    /// `offset` from the camera, or the sky.
    pub(crate) fn hit(&self, eye: [f32; 3], dir: [f32; 3], opaque: Option<[f32; 3]>) -> Hit {
        let water = self.water_hit(eye, dir);
        match opaque.map(|offset| (offset, length(offset))) {
            Some((offset, distance)) if water.is_none_or(|(_, t)| t >= distance) => {
                match self.holder(eye, offset) {
                    Some(slot) => Hit::Subject(slot, distance),
                    None => Hit::Other(distance),
                }
            }
            _ => water.map_or(Hit::Sky, |(slot, t)| Hit::Subject(slot, t)),
        }
    }

    /// Each subject as a camera at `eye` seeing through `frustum` sees it,
    /// with nothing measured.
    pub(crate) fn subjects(
        &self,
        eye: [f32; 3],
        frustum: &Frustum,
    ) -> [Option<Subject>; VIEW_SUBJECTS] {
        let mut out = [None; VIEW_SUBJECTS];
        out[PLANET_SLOT] = self.planet.map(|p| {
            let center_distance = length(sub(eye, p.ground.center));
            let [cx, cy, cz, radius] = p.bound.sphere;
            Subject {
                kind: SubjectKind::Planet,
                size: 2.0 * p.ground.radius,
                distance: (center_distance - p.ground.radius).max(0.0),
                center_distance,
                in_view: frustum.intersects_box([cx, cy, cz], [radius; 3]),
                seen: Sighting::default(),
            }
        });
        for (i, layer) in self.water.iter().enumerate() {
            out[WATER_SLOT + i] = water_subject(layer, eye, frustum);
        }
        out[PROP_SLOT] = self.prop.map(|(lo, hi)| {
            let center = scale(add(lo, hi), 0.5);
            Subject {
                kind: SubjectKind::Prop,
                size: length(sub(hi, lo)),
                distance: box_distance((lo, hi), eye),
                center_distance: length(sub(eye, center)),
                in_view: frustum.intersects_aabb(lo, hi),
                seen: Sighting::default(),
            }
        });
        out
    }

    /// The camera's height over the planet's sea, if any of the water
    /// surfaces is one.
    pub(crate) fn sea_height(&self) -> Option<f32> {
        self.water
            .iter()
            .find(|l| in_use(l) && l.shape[3] > 0.0)
            .map(|l| l.extent[2])
    }

    // The nearest water surface a ray from the camera at `eye` along `dir`
    // meets from above, with its slot.
    fn water_hit(&self, eye: [f32; 3], dir: [f32; 3]) -> Option<(usize, f32)> {
        self.water
            .iter()
            .enumerate()
            .filter_map(|(i, layer)| layer_distance(layer, eye, dir).map(|t| (WATER_SLOT + i, t)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }

    // The subject an opaque surface at `offset` from the camera at `eye`
    // belongs to: the prop when inside its box, else the planet when under
    // the height its ground stays under.
    fn holder(&self, eye: [f32; 3], offset: [f32; 3]) -> Option<usize> {
        let point = add(eye, offset);
        if let Some((lo, hi)) = self.prop {
            let margin = (length(sub(hi, lo)) * PROP_MARGIN).max(PROP_MARGIN_FLOOR);
            if (0..3).all(|k| point[k] >= lo[k] - margin && point[k] <= hi[k] + margin) {
                return Some(PROP_SLOT);
            }
        }
        let planet = self.planet?;
        let [cx, cy, cz, radius] = planet.bound.sphere;
        let height = planet.bound.camera[0];
        // |point - center|^2 - radius^2 about the camera, so a planet-sized
        // sphere still resolves near it.
        let from_center = sub(eye, [cx, cy, cz]);
        let inside =
            height * (2.0 * radius + height) + 2.0 * dot(from_center, offset) + dot(offset, offset);
        (inside <= 0.0).then_some(PLANET_SLOT)
    }
}

fn in_use(layer: &LevelLayerGpu) -> bool {
    layer.extent[3] >= 0.5
}

// How far a ray from the camera at `eye` along unit `dir` travels before it
// meets `layer` from above; `level_layer_distance` in `level_layer.hlsl` with
// the ray starting at the camera.
fn layer_distance(layer: &LevelLayerGpu, eye: [f32; 3], dir: [f32; 3]) -> Option<f32> {
    if !in_use(layer) {
        return None;
    }
    let height = layer.extent[2];
    let radius = layer.shape[3];
    let center = [layer.shape[0], layer.shape[1], layer.shape[2]];
    if radius <= 0.0 {
        if height <= 0.0 || dir[1] >= 0.0 {
            return None;
        }
        let t = height / -dir[1];
        let x = eye[0] + dir[0] * t - center[0];
        let z = eye[2] + dir[2] * t - center[2];
        return (x.abs() <= layer.extent[0] && z.abs() <= layer.extent[1]).then_some(t);
    }
    // |eye - center|^2 - radius^2, from the camera's height over the shell,
    // which the host formed in double precision.
    let c = height * (2.0 * radius + height);
    let b = dot(sub(eye, center), dir);
    if c <= 0.0 || b >= 0.0 {
        return None;
    }
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    Some(c / (sqrt(disc) - b))
}

fn water_subject(layer: &LevelLayerGpu, eye: [f32; 3], frustum: &Frustum) -> Option<Subject> {
    if !in_use(layer) {
        return None;
    }
    let center = [layer.shape[0], layer.shape[1], layer.shape[2]];
    let radius = layer.shape[3];
    let center_distance = length(sub(eye, center));
    let (size, distance, in_view) = if radius > 0.0 {
        (
            2.0 * radius,
            layer.extent[2].abs(),
            frustum.intersects_box(center, [radius; 3]),
        )
    } else {
        let [hx, hz] = [layer.extent[0], layer.extent[1]];
        let dx = ((eye[0] - center[0]).abs() - hx).max(0.0);
        let dz = ((eye[2] - center[2]).abs() - hz).max(0.0);
        let dy = layer.extent[2];
        // The surface lies `extent[2]` below the camera.
        let level = eye[1] - dy;
        (
            2.0 * sqrt(hx * hx + hz * hz),
            sqrt(dx * dx + dy * dy + dz * dz),
            frustum.intersects_box([center[0], level, center[2]], [hx, 0.0, hz]),
        )
    };
    Some(Subject {
        kind: SubjectKind::Water,
        size,
        distance,
        center_distance,
        in_view,
        seen: Sighting::default(),
    })
}

fn box_distance((lo, hi): Aabb, p: [f32; 3]) -> f32 {
    let d: [f32; 3] = core::array::from_fn(|k| (lo[k] - p[k]).max(p[k] - hi[k]).max(0.0));
    length(d)
}
