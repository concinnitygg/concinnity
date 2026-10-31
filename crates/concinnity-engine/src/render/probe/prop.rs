//! The prop reflection probe every backend runs the same way: which prop it
//! rides on and which cull records that prop draws through, where it is this
//! frame, the record the shaders read it by, and the cycle that spreads its
//! six faces over several frames.

use crate::components::{GlobalTransform, PropReflectionProbe, RenderHandle};
use crate::ecs::{Entity, PipelineContext};
use crate::gfx::frustum::transform_aabb;
use crate::planet::PlanetFrame;
use crate::render::draw_list::DrawListData;
use crate::render::probe::bake::CAPTURE_FACES;
use crate::render::probe::capture::PrefilterPlan;
use crate::render::probe::prop_reflection_probe::MAX_PROP_PROBE_RESOLUTION;
use crate::render::uniforms::{ProbeUniforms, StandIns};
use concinnity_core::math::vec3;

/// Fraction of the prop's smallest half-extent its probe's influence fades
/// over past its bounds, as a declared probe's does by default.
const PROP_BLEND_FRACTION: f32 = 0.2;

/// Floor of the prop probe's blend margin and half-extents, so a flat prop
/// still has a box to fade across.
const MIN_PROP_EXTENT: f32 = 1e-2;

/// Up where a world has no planet.
const WORLD_UP: [f32; 3] = [0.0, 1.0, 0.0];

/// The world's prop probe as the frame extraction tracks it: the prop's
/// entity, its bounds in its own space, the cull records its draws occupy,
/// and how the probe captures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PropProbeTarget {
    entity: Entity,
    local_min: [f32; 3],
    local_max: [f32; 3],
    records: [u32; 2],
    resolution: u32,
    capture_distance: Option<f32>,
    faces_per_frame: u32,
}

impl PropProbeTarget {
    /// The world's `PropReflectionProbe` resolved against the draw list the
    /// props were just built into. `None` without one, or when its prop draws
    /// nothing bounded. A second probe is reported and ignored.
    pub fn resolve(ctx: &PipelineContext<'_>, data: &DrawListData) -> Option<Self> {
        let mut probes = ctx.query::<PropReflectionProbe>();
        let probe = probes.next()?;
        if probes.next().is_some() {
            tracing::warn!(
                "reflection probes: a world rides one PropReflectionProbe; the rest are ignored"
            );
        }
        let Some(prop) = probe.prop else {
            tracing::warn!("reflection probes: a PropReflectionProbe names no prop");
            return None;
        };
        let entity = ctx.entity_of(prop.id())?;
        let draws = &ctx.get::<RenderHandle>(entity)?.draws;
        let index = data
            .prop_draw_indices
            .iter()
            .position(|d| d.first() == draws.first())?;
        let (local_min, local_max) = data.prop_local_bounds[index];
        if !crate::gfx::lod::bounds_finite(local_min, local_max) {
            tracing::warn!("reflection probes: the probe's prop has no bounds to ride on");
            return None;
        }
        Some(Self {
            entity,
            local_min,
            local_max,
            records: contiguous_records(draws.iter().map(|d| d.0)),
            resolution: probe.resolution.min(MAX_PROP_PROBE_RESOLUTION),
            capture_distance: probe.capture_distance,
            faces_per_frame: probe.faces_per_frame,
        })
    }

    /// Where the probe is this frame: the prop's bounds carried by its
    /// current world transform, and the plane it rests on along the planet's
    /// up there (world up without a planet). `None` once the prop has no
    /// transform.
    pub fn frame(&self, ctx: &PipelineContext<'_>) -> Option<PropProbeFrame> {
        let model = ctx.get::<GlobalTransform>(self.entity)?.0;
        let (bounds_min, bounds_max) = transform_aabb(self.local_min, self.local_max, model);
        let center = local_center(self.local_min, self.local_max, model);
        let up = ctx
            .resource::<PlanetFrame>()
            .map_or(WORLD_UP, |planet| planet.up_at(center));
        Some(PropProbeFrame {
            bounds_min,
            bounds_max,
            ground: ground_plane(self.local_min, self.local_max, model, up),
            records: self.records,
            resolution: self.resolution,
            capture_distance: self.capture_distance,
            faces_per_frame: self.faces_per_frame,
            sea: false,
        })
    }
}

// The center of the local bounds `[local_min, local_max]` under `model`.
fn local_center(local_min: [f32; 3], local_max: [f32; 3], model: [[f32; 4]; 4]) -> [f32; 3] {
    let c: [f32; 3] = core::array::from_fn(|i| 0.5 * (local_min[i] + local_max[i]));
    core::array::from_fn(|r| {
        model[0][r] * c[0] + model[1][r] * c[1] + model[2][r] * c[2] + model[3][r]
    })
}

// The plane a prop with local bounds `[local_min, local_max]` under `model`
// rests on along `up` (unit): `up` and its offset, `dot(up, p) == w` on the
// plane. It passes through the bottom of the ellipsoid the bounds enclose,
// which stays put as a round prop turns where its box's corners would swing.
fn ground_plane(
    local_min: [f32; 3],
    local_max: [f32; 3],
    model: [[f32; 4]; 4],
    up: [f32; 3],
) -> [f32; 4] {
    let reach = vec3::length(core::array::from_fn(|i| {
        0.5 * (local_max[i] - local_min[i]) * vec3::dot([model[i][0], model[i][1], model[i][2]], up)
    }));
    let offset = vec3::dot(up, local_center(local_min, local_max, model)) - reach;
    [up[0], up[1], up[2], offset]
}

// The record range `[first, end)` that holds exactly `draws`, or an empty
// range when they are not one run: a range spanning other draws would leave
// those out of the capture too.
fn contiguous_records(draws: impl Iterator<Item = u32>) -> [u32; 2] {
    let mut range: Option<[u32; 2]> = None;
    for d in draws {
        range = match range {
            None => Some([d, d + 1]),
            Some([first, end]) if d == end => Some([first, end + 1]),
            Some(_) => {
                tracing::warn!(
                    "reflection probes: the probe's prop draws are not one run; it is captured in its own probe"
                );
                return [0, 0];
            }
        };
    }
    range.unwrap_or([0, 0])
}

/// The prop probe this frame: where the prop is, the cull records its own
/// draws occupy, and how the probe captures. Read by every backend from
/// `FrameParams`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PropProbeFrame {
    /// Lower corner of the prop's world bounds this frame.
    pub bounds_min: [f32; 3],
    /// Upper corner of the prop's world bounds this frame.
    pub bounds_max: [f32; 3],
    /// The plane the prop rests on: its up (xyz, unit) and offset (w), so
    /// `dot(up, p) == w` on it.
    pub ground: [f32; 4],
    /// The cull records `[first, end)` the prop draws through, which its own
    /// capture leaves out.
    pub records: [u32; 2],
    /// Cube-face edge in texels.
    pub resolution: u32,
    /// How far each face's capture reaches along its axis; `None` without
    /// limit.
    pub capture_distance: Option<f32>,
    /// How many faces a frame renders.
    pub faces_per_frame: u32,
    /// Whether the frame's daylight carries a planet sea for the faces to
    /// shade in place of its surface.
    pub sea: bool,
}

impl PropProbeFrame {
    /// This frame with the sea the faces stand in for, or none.
    #[must_use]
    pub fn with_sea(self, sea: bool) -> Self {
        Self { sea, ..self }
    }

    /// What a face shades its ground as in place of what it does not draw:
    /// the canopy for the blades, and the sea's surface where there is one.
    pub fn stand_ins(&self) -> StandIns {
        if self.sea {
            StandIns::CanopyAndSea
        } else {
            StandIns::Canopy
        }
    }

    /// Where the faces are captured from: the center of the prop's bounds.
    pub fn eye(&self) -> [f32; 3] {
        core::array::from_fn(|i| 0.5 * (self.bounds_min[i] + self.bounds_max[i]))
    }

    /// The convolution of a capture at this probe's face size into its cube
    /// of the probe array, whose chain starts at [`PrefilterPlan::RUNTIME`]'s
    /// face size.
    pub fn plan(&self) -> PrefilterPlan {
        prop_probe_plan(self.resolution)
    }

    /// The record the shaders read this probe by: an influence box on the
    /// prop's bounds, no box parallax (the scene it captures lies far beyond
    /// any box around a prop), a fade to the sky past it, its finest mip, and
    /// the mark of a capture that leaves its prop out, whose lookups aim at
    /// the header's [`ground`](crate::render::uniforms::ProbeSet::ground).
    pub fn record(&self) -> ProbeUniforms {
        let eye = self.eye();
        let half: [f32; 3] = core::array::from_fn(|i| {
            (0.5 * (self.bounds_max[i] - self.bounds_min[i])).max(MIN_PROP_EXTENT)
        });
        let margin = (PROP_BLEND_FRACTION * half[0].min(half[1]).min(half[2])).max(MIN_PROP_EXTENT);
        let [px, py, pz] = eye;
        ProbeUniforms {
            box_min: [px - half[0], py - half[1], pz - half[2], 0.0],
            box_max: [px + half[0], py + half[1], pz + half[2], 1.0],
            probe_pos: [px, py, pz, self.plan().first_mip() as f32],
            blend_margin: [margin, margin, margin, 1.0],
        }
    }
}

/// The faces of the prop probe one frame renders, and whether it then
/// convolves the capture into the probe's cube.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceSlice {
    /// The first face rendered.
    pub first: usize,
    /// How many faces are rendered, from `first` on.
    pub count: usize,
    /// The slice renders the cycle's last face, so the capture is whole and
    /// convolved after it.
    pub convolve: bool,
    /// An earlier frame convolved a whole capture, so the cube holds one the
    /// shaders can read.
    pub filled: bool,
}

impl FaceSlice {
    /// The faces rendered.
    pub fn faces(&self) -> core::ops::Range<usize> {
        self.first..self.first + self.count
    }
}

/// The prop probe's walk through its six faces, a slice a frame, which a
/// backend keeps from frame to frame. A new or rebuilt capture starts it over.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FaceCycle {
    next: usize,
    filled: bool,
}

impl FaceCycle {
    /// This frame's slice of `faces_per_frame` faces, moving the cycle past it.
    pub fn advance(&mut self, faces_per_frame: u32) -> FaceSlice {
        let per_frame = usize::try_from(faces_per_frame).unwrap_or(CAPTURE_FACES);
        let first = self.next;
        let count = per_frame.clamp(1, CAPTURE_FACES - first);
        self.next = (first + count) % CAPTURE_FACES;
        let slice = FaceSlice {
            first,
            count,
            convolve: self.next == 0,
            filled: self.filled,
        };
        self.filled |= slice.convolve;
        slice
    }

    /// Start over from the first face with nothing convolved.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// The convolution of a prop probe's capture of `resolution`-texel faces into
/// its cube of the probe array, whose chain starts at
/// [`PrefilterPlan::RUNTIME`]'s face size.
pub fn prop_probe_plan(resolution: u32) -> PrefilterPlan {
    PrefilterPlan::RUNTIME.from_mip(first_mip(resolution))
}

// The cube mip a face of `resolution` texels is written into.
fn first_mip(resolution: u32) -> u32 {
    let full = PrefilterPlan::RUNTIME.face_size();
    (full / resolution.clamp(1, full)).ilog2()
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::approx::close;

    const TOL: f32 = 1e-5;

    fn frame(resolution: u32) -> PropProbeFrame {
        PropProbeFrame {
            bounds_min: [1.0, 2.0, 3.0],
            bounds_max: [3.0, 4.0, 7.0],
            ground: [0.0, 1.0, 0.0, 2.0],
            records: [4, 6],
            resolution,
            capture_distance: None,
            faces_per_frame: 2,
            sea: false,
        }
    }

    fn slices(faces_per_frame: u32, frames: usize) -> alloc::vec::Vec<FaceSlice> {
        let mut cycle = FaceCycle::default();
        (0..frames)
            .map(|_| cycle.advance(faces_per_frame))
            .collect()
    }

    #[test]
    fn two_faces_a_frame_renew_the_cube_every_third_frame() {
        let s = slices(2, 4);
        let faces: alloc::vec::Vec<_> = s.iter().map(FaceSlice::faces).collect();
        assert_eq!(faces, [0..2, 2..4, 4..6, 0..2]);
        let convolved: alloc::vec::Vec<_> = s.iter().map(|s| s.convolve).collect();
        assert_eq!(convolved, [false, false, true, false]);
        // The cube is readable from the frame after the first convolution.
        let filled: alloc::vec::Vec<_> = s.iter().map(|s| s.filled).collect();
        assert_eq!(filled, [false, false, false, true]);
    }

    #[test]
    fn six_faces_a_frame_convolve_every_frame() {
        for s in slices(6, 3) {
            assert_eq!((s.faces(), s.convolve), (0..6, true));
        }
        assert!(!slices(6, 1)[0].filled);
        assert!(slices(6, 2)[1].filled);
    }

    #[test]
    fn every_face_renders_once_a_cycle_whatever_the_count() {
        for per_frame in [0, 1, 2, 3, 4, 5, 6, 7] {
            let mut rendered = [0; CAPTURE_FACES];
            let mut cycle = FaceCycle::default();
            loop {
                let s = cycle.advance(per_frame);
                s.faces().for_each(|f| rendered[f] += 1);
                if s.convolve {
                    break;
                }
            }
            assert_eq!(rendered, [1; CAPTURE_FACES], "{per_frame}");
        }
    }

    #[test]
    fn a_reset_starts_over_unfilled() {
        let mut cycle = FaceCycle::default();
        for _ in 0..4 {
            cycle.advance(3);
        }
        cycle.reset();
        assert_eq!(
            cycle.advance(3),
            FaceSlice {
                first: 0,
                count: 3,
                convolve: false,
                filled: false
            }
        );
    }

    #[test]
    fn the_faces_stand_in_for_the_sea_only_where_there_is_one() {
        assert_eq!(frame(128).stand_ins(), StandIns::Canopy);
        assert_eq!(
            frame(128).with_sea(true).stand_ins(),
            StandIns::CanopyAndSea
        );
    }

    #[test]
    fn the_record_rides_the_bounds_without_parallax() {
        let r = frame(128).record();
        assert_eq!(r.probe_pos, [2.0, 3.0, 5.0, 2.0]);
        assert_eq!(&r.box_min[..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&r.box_max[..3], &[3.0, 4.0, 7.0]);
        assert_eq!(r.box_min[3], 0.0, "no parallax");
        assert_eq!(r.box_max[3], 1.0, "fades to the sky");
        assert!((r.blend_margin[0] - 0.2).abs() < 1e-6);
        assert_eq!(r.blend_margin[3], 1.0, "leaves its prop out");
    }

    #[test]
    fn a_smaller_face_lands_further_down_the_chain() {
        assert_eq!(frame(512).plan().first_mip(), 0);
        assert_eq!(frame(256).plan().first_mip(), 1);
        assert_eq!(frame(128).plan().first_mip(), 2);
        assert_eq!(frame(128).plan().face_size(), 128);
        assert_eq!(frame(16).plan().first_mip(), 5);
        assert_eq!(frame(16).record().probe_pos[3], 5.0);
    }

    #[test]
    fn a_flat_prop_still_has_a_box() {
        let flat = PropProbeFrame {
            bounds_min: [0.0; 3],
            bounds_max: [2.0, 0.0, 2.0],
            ..frame(128)
        };
        let r = flat.record();
        assert!(r.box_max[1] > r.box_min[1]);
        assert!(r.blend_margin[0] > 0.0);
    }

    // A column-major model turning `angle` radians about z, scaled by
    // `scale`, then moved to `at`.
    fn model(angle: f32, scale: f32, at: [f32; 3]) -> [[f32; 4]; 4] {
        let (s, c) = angle.sin_cos();
        [
            [c * scale, s * scale, 0.0, 0.0],
            [-s * scale, c * scale, 0.0, 0.0],
            [0.0, 0.0, scale, 0.0],
            [at[0], at[1], at[2], 1.0],
        ]
    }

    #[test]
    fn a_ball_rests_one_radius_below_its_center_however_it_turns() {
        let (lo, hi) = ([-0.6; 3], [0.6; 3]);
        let up = [0.0, 1.0, 0.0];
        for angle in [0.0, 0.3, core::f32::consts::FRAC_PI_4, 2.0] {
            let plane = ground_plane(lo, hi, model(angle, 1.0, [6.0, 7.3, 0.6]), up);
            assert!(
                close(plane, [0.0, 1.0, 0.0, 6.7], TOL),
                "{angle}: {plane:?}"
            );
        }
        let scaled = ground_plane(lo, hi, model(0.0, 2.0, [0.0; 3]), up);
        assert!(close(scaled, [0.0, 1.0, 0.0, -1.2], TOL), "{scaled:?}");
    }

    #[test]
    fn the_plane_follows_a_tilted_up() {
        let up = vec3::normalize([1.0, 1.0, 0.0]);
        let at = [10.0, 10.0, 0.0];
        let ball = ground_plane([-0.5; 3], [0.5; 3], model(0.0, 1.0, at), up);
        assert!(
            close(ball, [up[0], up[1], up[2], vec3::dot(up, at) - 0.5], TOL),
            "{ball:?}"
        );

        // A slab half as tall as it is wide rests on its broad face when that
        // faces the tilted up.
        let slab = ground_plane(
            [-1.0, -0.25, -1.0],
            [1.0, 0.25, 1.0],
            model(-core::f32::consts::FRAC_PI_4, 1.0, at),
            up,
        );
        assert!(
            close(slab, [up[0], up[1], up[2], vec3::dot(up, at) - 0.25], TOL),
            "{slab:?}"
        );
    }

    #[test]
    fn a_prop_off_its_bounds_center_rests_under_that_center() {
        let plane = ground_plane(
            [0.0; 3],
            [2.0, 1.0, 2.0],
            model(0.0, 1.0, [0.0, 5.0, 0.0]),
            [0.0, 1.0, 0.0],
        );
        assert!(close(plane, [0.0, 1.0, 0.0, 5.0], TOL), "{plane:?}");
        assert_eq!(
            local_center([0.0; 3], [2.0, 1.0, 2.0], model(0.0, 1.0, [0.0, 5.0, 0.0])),
            [1.0, 5.5, 1.0]
        );
    }

    #[test]
    fn only_one_run_of_records_is_left_out() {
        assert_eq!(contiguous_records([7u32].into_iter()), [7, 8]);
        assert_eq!(contiguous_records([3u32, 4, 5].into_iter()), [3, 6]);
        assert_eq!(contiguous_records([3u32, 5].into_iter()), [0, 0]);
        assert_eq!(contiguous_records(core::iter::empty()), [0, 0]);
    }
}
