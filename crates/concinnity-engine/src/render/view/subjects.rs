//! The subjects a view is measured against, in the space a frame draws in:
//! where each lies and which of them a camera ray meets first.

use super::facts::{Sighting, Subject, SubjectKind, VIEW_SUBJECTS};
use super::screen::ScreenRect;
use crate::gfx::frustum::{Aabb, Frustum};
use crate::gfx::render_types::{GroundBoundGpu, LevelLayerGpu, LevelSurfacesGpu, MAX_LEVEL_LAYERS};
use crate::render::shadow::far_anchor::GroundSphere;
use crate::transform::Mat4;
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

    /// Each subject as a camera at `eye` seeing through `frustum`, by the
    /// view-projection `vp`, sees it, with nothing measured.
    pub(crate) fn subjects(
        &self,
        eye: [f32; 3],
        frustum: &Frustum,
        vp: &Mat4,
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
                clearance: 0.0,
                in_view: frustum.intersects_box([cx, cy, cz], [radius; 3]),
                rect: ScreenRect::FULL,
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
                clearance: self.clearance((lo, hi), eye),
                in_view: frustum.intersects_aabb(lo, hi),
                rect: ScreenRect::of_box(vp, (lo, hi)),
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

    // How far `bounds` stands clear of the planet's ground and the water
    // surfaces, for a camera at `eye`: over the highest the ground reaches,
    // so a prop anywhere within the ground's relief counts as touching it; 0
    // where it meets one, or with neither.
    fn clearance(&self, bounds: Aabb, eye: [f32; 3]) -> f32 {
        let ground = self.planet.map(|p| {
            let [cx, cy, cz, radius] = p.bound.sphere;
            let sphere = Shell {
                center: [cx, cy, cz],
                radius,
                camera_height: p.bound.camera[0],
            };
            sphere.heights(bounds, eye)[0].max(0.0)
        });
        let water = self
            .water
            .iter()
            .filter(|l| in_use(l))
            .map(|l| layer_gap(l, bounds, eye));
        ground
            .into_iter()
            .chain(water)
            .reduce(f32::min)
            .unwrap_or(0.0)
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
        let bound = Shell {
            center: [cx, cy, cz],
            radius,
            camera_height: planet.bound.camera[0],
        };
        (bound.excess(eye, offset) <= 0.0).then_some(PLANET_SLOT)
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
        clearance: 0.0,
        in_view,
        rect: ScreenRect::FULL,
        seen: Sighting::default(),
    })
}

// A sphere a camera stands over, by the camera's height over it, which the
// host forms in double precision.
struct Shell {
    center: [f32; 3],
    radius: f32,
    camera_height: f32,
}

impl Shell {
    // |point - center|^2 - radius^2 for the point `offset` from the camera at
    // `eye`, formed about the camera so a planet-sized sphere still resolves
    // near it.
    fn excess(&self, eye: [f32; 3], offset: [f32; 3]) -> f32 {
        let h = self.camera_height;
        let from_center = sub(eye, self.center);
        h * (2.0 * self.radius + h) + 2.0 * dot(from_center, offset) + dot(offset, offset)
    }

    // The least and greatest height of the corners of `bounds` over the
    // sphere, for a camera at `eye`; negative under it.
    fn heights(&self, (lo, hi): Aabb, eye: [f32; 3]) -> [f32; 2] {
        let from_center = sub(eye, self.center);
        (0..8u8).fold([f32::INFINITY, f32::NEG_INFINITY], |[low, high], corner| {
            let point: [f32; 3] =
                core::array::from_fn(|k| if corner >> k & 1 == 0 { lo[k] } else { hi[k] });
            let offset = sub(point, eye);
            let reach = length(add(from_center, offset)) + self.radius;
            let height = self.excess(eye, offset) / reach.max(f32::MIN_POSITIVE);
            [low.min(height), high.max(height)]
        })
    }
}

fn layer_gap(layer: &LevelLayerGpu, (lo, hi): Aabb, eye: [f32; 3]) -> f32 {
    let center = [layer.shape[0], layer.shape[1], layer.shape[2]];
    if layer.shape[3] > 0.0 {
        let shell = Shell {
            center,
            radius: layer.shape[3],
            camera_height: layer.extent[2],
        };
        let [low, high] = shell.heights((lo, hi), eye);
        return low.max(-high).max(0.0);
    }
    let level = eye[1] - layer.extent[2];
    let outside = |lo: f32, hi: f32, from: f32, to: f32| (from - hi).max(lo - to).max(0.0);
    let gap = [
        outside(
            lo[0],
            hi[0],
            center[0] - layer.extent[0],
            center[0] + layer.extent[0],
        ),
        outside(lo[1], hi[1], level, level),
        outside(
            lo[2],
            hi[2],
            center[2] - layer.extent[1],
            center[2] + layer.extent[1],
        ),
    ];
    length(gap)
}

fn box_distance((lo, hi): Aabb, p: [f32; 3]) -> f32 {
    let d: [f32; 3] = core::array::from_fn(|k| (lo[k] - p[k]).max(p[k] - hi[k]).max(0.0));
    length(d)
}
