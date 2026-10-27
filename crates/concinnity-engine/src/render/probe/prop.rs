//! The prop reflection probe every backend runs the same way: which prop it
//! rides on and which cull records that prop draws through, where it is this
//! frame, and the record the shaders read it by. The backends render its six
//! faces and convolve them every frame.

use crate::components::{GlobalTransform, PropReflectionProbe, RenderHandle};
use crate::ecs::{Entity, PipelineContext};
use crate::gfx::frustum::transform_aabb;
use crate::render::draw_list::DrawListData;
use crate::render::probe::capture::PrefilterPlan;
use crate::render::probe::prop_reflection_probe::MAX_PROP_PROBE_RESOLUTION;
use crate::render::uniforms::{ProbeUniforms, StandIns};

/// Fraction of the prop's smallest half-extent its probe's influence fades
/// over past its bounds, as a declared probe's does by default.
const PROP_BLEND_FRACTION: f32 = 0.2;

/// Floor of the prop probe's blend margin and half-extents, so a flat prop
/// still has a box to fade across.
const MIN_PROP_EXTENT: f32 = 1e-2;

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
        })
    }

    /// Where the probe is this frame: the prop's bounds carried by its
    /// current world transform. `None` once the prop has no transform.
    pub fn frame(&self, ctx: &PipelineContext<'_>) -> Option<PropProbeFrame> {
        let model = ctx.get::<GlobalTransform>(self.entity)?.0;
        let (bounds_min, bounds_max) = transform_aabb(self.local_min, self.local_max, model);
        Some(PropProbeFrame {
            bounds_min,
            bounds_max,
            records: self.records,
            resolution: self.resolution,
            capture_distance: self.capture_distance,
            sea: false,
        })
    }
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
    /// The cull records `[first, end)` the prop draws through, which its own
    /// capture leaves out.
    pub records: [u32; 2],
    /// Cube-face edge in texels.
    pub resolution: u32,
    /// How far each face's capture reaches along its axis; `None` without
    /// limit.
    pub capture_distance: Option<f32>,
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
    /// prop's bounds, no parallax (the scene it captures lies far beyond any
    /// box around a prop), a fade to the sky past it, its finest mip, and the
    /// mark of a capture that leaves its prop out.
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

    fn frame(resolution: u32) -> PropProbeFrame {
        PropProbeFrame {
            bounds_min: [1.0, 2.0, 3.0],
            bounds_max: [3.0, 4.0, 7.0],
            records: [4, 6],
            resolution,
            capture_distance: None,
            sea: false,
        }
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

    #[test]
    fn only_one_run_of_records_is_left_out() {
        assert_eq!(contiguous_records([7u32].into_iter()), [7, 8]);
        assert_eq!(contiguous_records([3u32, 4, 5].into_iter()), [3, 6]);
        assert_eq!(contiguous_records([3u32, 5].into_iter()), [0, 0]);
        assert_eq!(contiguous_records(core::iter::empty()), [0, 0]);
    }
}
