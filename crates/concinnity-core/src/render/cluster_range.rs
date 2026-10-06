//! The depth range the cluster grid slices. The grid's exponential slices need a
//! finite far end, and the camera projection has none, so the far end follows
//! what the grid bins: it reaches exactly as far from the camera as the local
//! lights and the reflection probes' influence do.

use crate::gfx::render_types::GpuLight;
use crate::math::vec3::{length, sub};
use crate::render::uniforms::ProbeUniforms;
use alloc::vec::Vec;

/// The local lights' bounding spheres, kept from when the lights were packed so
/// every frame can place the far end of the cluster grid around its camera.
#[derive(Clone, Debug, Default)]
pub struct ClusterReach {
    spheres: Vec<([f32; 3], f32)>,
}

impl ClusterReach {
    /// The reach of the packed local lights.
    pub fn new(lights: &[GpuLight]) -> Self {
        Self {
            spheres: lights
                .iter()
                .map(|l| (l.position, l.range.max(0.0)))
                .collect(),
        }
    }

    /// The far end of the cluster grid for a camera at `cam_pos` whose near
    /// plane is `near`: the farthest any light sphere or probe influence (the
    /// box grown by its blend margin) reaches from the camera, capped by
    /// `view_distance` when the camera has one. Nothing the grid bins reaches
    /// past it, so slicing out there would only coarsen the slices that matter.
    /// Never below twice the near plane, which keeps the slicing ratio above 1
    /// when nothing is binned.
    pub fn range(
        &self,
        cam_pos: [f32; 3],
        near: f32,
        probes: &[ProbeUniforms],
        view_distance: Option<f32>,
    ) -> f32 {
        let lights = self
            .spheres
            .iter()
            .map(|&(center, radius)| length(sub(center, cam_pos)) + radius);
        let probes = probes.iter().map(|p| {
            let (lo, hi) = p.influence_bounds();
            farthest_corner(cam_pos, lo, hi)
        });
        let reach = lights.chain(probes).fold(0.0f32, f32::max);
        let reach = view_distance.map_or(reach, |d| reach.min(d));
        reach.max(2.0 * near)
    }
}

// Distance from `p` to the farthest point of the box `lo..hi`.
fn farthest_corner(p: [f32; 3], lo: [f32; 3], hi: [f32; 3]) -> f32 {
    let axis = |i: usize| (p[i] - lo[i]).abs().max((hi[i] - p[i]).abs());
    length([axis(0), axis(1), axis(2)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::uniforms::probe::PROBE_BLEND_MARGIN;

    fn light(position: [f32; 3], range: f32) -> GpuLight {
        GpuLight {
            position,
            range,
            ..GpuLight::ZERO
        }
    }

    fn probe(box_min: [f32; 3], box_max: [f32; 3]) -> ProbeUniforms {
        ProbeUniforms {
            box_min: [box_min[0], box_min[1], box_min[2], 1.0],
            box_max: [box_max[0], box_max[1], box_max[2], 0.0],
            probe_pos: [0.0; 4],
        }
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-4 * b.abs().max(1.0)
    }

    // The farthest light sphere sets the range, whichever side of the camera
    // it is on.
    #[test]
    fn the_range_reaches_the_far_side_of_the_farthest_light() {
        let reach = ClusterReach::new(&[
            light([0.0, 0.0, -30.0], 5.0),
            light([40.0, 0.0, 0.0], 2.0),
            light([0.0, 0.0, 10.0], 1.0),
        ]);
        assert!(close(reach.range([0.0; 3], 0.05, &[], None), 42.0));
        // From another position the same lights reach differently.
        assert!(close(reach.range([0.0, 0.0, 20.0], 0.05, &[], None), 55.0));
    }

    #[test]
    fn a_probe_reaches_its_farthest_grown_corner() {
        let reach = ClusterReach::default();
        let p = probe([10.0, -2.0, -6.0], [20.0, 4.0, 6.0]);
        let margin = PROBE_BLEND_MARGIN * 3.0;
        let corner = [20.0 + margin, 4.0 + margin, 6.0 + margin];
        let expected = length(corner);
        assert!(close(reach.range([0.0; 3], 0.05, &[p], None), expected));

        // Inside the box the range still covers the whole influence.
        let inside = [15.0, 0.0, 0.0];
        let expected = length(sub([10.0 - margin, 4.0 + margin, 6.0 + margin], inside));
        assert!(close(reach.range(inside, 0.05, &[p], None), expected));
    }

    #[test]
    fn lights_and_probes_share_one_range() {
        let reach = ClusterReach::new(&[light([0.0, 0.0, -10.0], 2.0)]);
        let far_probe = probe([-1.0, -1.0, -101.0], [1.0, 1.0, -99.0]);
        let range = reach.range([0.0; 3], 0.05, &[far_probe], None);
        assert!(range > 101.0, "{range}");
        let near_probe = probe([-1.0, -1.0, -3.0], [1.0, 1.0, -1.0]);
        assert!(close(
            reach.range([0.0; 3], 0.05, &[near_probe], None),
            12.0
        ));
    }

    #[test]
    fn a_view_distance_caps_the_range() {
        let reach = ClusterReach::new(&[light([0.0, 0.0, -900.0], 50.0)]);
        assert!(close(reach.range([0.0; 3], 0.1, &[], None), 950.0));
        assert_eq!(reach.range([0.0; 3], 0.1, &[], Some(300.0)), 300.0);
        assert!(close(reach.range([0.0; 3], 0.1, &[], Some(5000.0)), 950.0));
    }

    // With nothing to bin, or nothing past the near plane, the slicing ratio
    // still stays above 1.
    #[test]
    fn the_range_keeps_a_floor_past_the_near_plane() {
        let empty = ClusterReach::default();
        assert_eq!(empty.range([3.0, 4.0, 5.0], 0.05, &[], None), 0.1);
        assert_eq!(empty.range([0.0; 3], 0.5, &[], Some(0.2)), 1.0);
        let tiny = ClusterReach::new(&[light([0.0, 0.0, -0.01], 0.01)]);
        assert_eq!(tiny.range([0.0; 3], 0.05, &[], None), 0.1);
    }

    // A light whose range is negative reaches no farther than its center.
    #[test]
    fn a_negative_range_counts_as_none() {
        let reach = ClusterReach::new(&[light([0.0, 0.0, -20.0], -5.0)]);
        assert!(close(reach.range([0.0; 3], 0.05, &[], None), 20.0));
    }
}
