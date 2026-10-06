//! The world's flat reflectors: the plane each one mirrors across, the mirror
//! slot it samples, and the bound its surface is drawn within.

use super::MAX_PLANAR_PLANES;
use super::slots::{PlanarAssignment, assign_planar_slots};
use super::visibility::{PlanarFramePlan, PlanarReflector, ReflectorHull};
use crate::components::{GlassPanel, PassResolution, WaterSurface};
use crate::geometry::glass_quad::plane_basis;
use crate::math::vec3::normalize_or;
use crate::render::backend_init::PlanarBudget;
use crate::render::uniforms::{WATER_MAX_WAVES, WaterParams};
use crate::transform::Mat4;
use alloc::vec::Vec;
use core::f32::consts::TAU;

/// The rest plane of a water surface: horizontal through its center, normal +y.
pub fn water_plane(surface: &WaterSurface) -> [f32; 4] {
    [0.0, 1.0, 0.0, -surface.center[1]]
}

/// The plane of a glass pane with unit `normal` through `center`.
pub fn pane_plane(normal: [f32; 3], center: [f32; 3]) -> [f32; 4] {
    [
        normal[0],
        normal[1],
        normal[2],
        -(normal[0] * center[0] + normal[1] * center[1] + normal[2] * center[2]),
    ]
}

// The box a water surface's displaced grid stays inside, widened by how far
// its mirror lookup is pushed off the fragment by the wave normal. The Gerstner
// sum moves a vertex at most `amplitude` vertically and `steepness / (k * count)`
// horizontally per wave (see `gerstner_displace` in water.hlsl).
fn water_hull(surface: &WaterSurface) -> ReflectorHull {
    let waves = &surface.waves[..surface.waves.len().min(WATER_MAX_WAVES)];
    let count = waves.len().max(1) as f32;
    let (mut rise, mut sway) = (0.0f32, 0.0f32);
    for w in waves {
        rise += w.amplitude.abs();
        let k = TAU / w.wavelength.max(1e-3);
        sway += w.steepness.clamp(0.0, 1.0) / (k * count);
    }
    let lookup = WaterParams::planar_lane(surface.roughness, true)[1];
    ReflectorHull::aabb(
        surface.center,
        [surface.extent[0] + sway, rise, surface.extent[1] + sway],
        lookup,
    )
}

// The pane's own quad, exactly as `geometry::glass_quad` builds it. Glass
// samples its mirror at the fragment itself, so there is no lookup margin.
fn glass_hull(panel: &GlassPanel) -> ReflectorHull {
    let n = normalize_or(panel.normal, 1e-6, [0.0, 0.0, 1.0]);
    let (t, b) = plane_basis(n);
    let hw = panel.half_size[0].max(1e-3);
    let hh = panel.half_size[1].max(1e-3);
    let c = panel.center;
    let corner = |su: f32, sv: f32| -> [f32; 3] {
        core::array::from_fn(|i| c[i] + t[i] * su * hw + b[i] * sv * hh)
    };
    ReflectorHull::quad(
        [
            corner(-1.0, -1.0),
            corner(1.0, -1.0),
            corner(1.0, 1.0),
            corner(-1.0, 1.0),
        ],
        0.0,
    )
}

/// A world's planar reflection layout, fixed at init: which mirror slot every
/// water surface and glass pane samples, the drawn reflectors a frame plans its
/// mirror renders from, and the mirror targets' resolution.
#[derive(Clone, Debug)]
pub struct PlanarReflectors {
    /// One slot per reflector, water surfaces first, then glass panes, so
    /// `slots[..water.len()]` are the surfaces'. `representatives` holds one
    /// plane per mirror render.
    pub assignment: PlanarAssignment,
    /// The visible reflectors that hold a slot.
    pub reflectors: Vec<PlanarReflector>,
    /// Mirror target resolution relative to the render resolution.
    pub resolution: PassResolution,
}

impl PlanarReflectors {
    /// Group every reflector's plane into at most `budget.planes` mirror slots
    /// (capped at [`MAX_PLANAR_PLANES`]). Water is listed first so it keeps a
    /// slot when the budget is tight.
    pub fn plan(water: &[WaterSurface], glass: &[GlassPanel], budget: PlanarBudget) -> Self {
        let planes: Vec<[f32; 4]> = water
            .iter()
            .map(water_plane)
            .chain(glass.iter().map(|g| pane_plane(g.normal, g.center)))
            .collect();
        let assignment = assign_planar_slots(&planes, budget.planes.min(MAX_PLANAR_PLANES));
        let hulls = water
            .iter()
            .map(|w| (w.visible, water_hull(w)))
            .chain(glass.iter().map(|g| (g.visible, glass_hull(g))));
        let reflectors = hulls
            .zip(&assignment.slots)
            .filter_map(|((visible, hull), slot)| {
                slot.filter(|_| visible)
                    .map(|slot| PlanarReflector { slot, hull })
            })
            .collect();
        Self {
            assignment,
            reflectors,
            resolution: budget.resolution,
        }
    }

    /// The mirror planes, one render each.
    pub fn planes(&self) -> &[[f32; 4]] {
        &self.assignment.representatives
    }

    /// The mirror target size for a `width` x `height` render resolution.
    pub fn target_size(&self, width: u32, height: u32) -> (u32, u32) {
        let d = self.resolution.scale_divisor();
        ((width / d).max(1), (height / d).max(1))
    }

    /// Reflectors whose plane found no slot and keep the probe cube.
    pub fn overflow(&self) -> usize {
        self.assignment.slots.iter().filter(|s| s.is_none()).count()
    }

    /// This frame's mirror work under the (jittered) `view_proj` the
    /// reflectors are rasterized with.
    pub fn frame_plan(&self, view_proj: Mat4) -> PlanarFramePlan {
        PlanarFramePlan::build(&self.reflectors, view_proj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::WaterWave;
    use crate::render::depth::camera_projection;
    use crate::transform::mat4_mul;
    use alloc::vec;

    fn looking_down_z_from(eye: [f32; 3]) -> Mat4 {
        let view = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-eye[0], -eye[1], -eye[2], 1.0],
        ];
        mat4_mul(camera_projection(1.2, 1.5, 0.1), view)
    }

    fn budget(planes: usize) -> PlanarBudget {
        PlanarBudget {
            planes,
            resolution: PassResolution::Full,
        }
    }

    fn pool(center: [f32; 3]) -> WaterSurface {
        WaterSurface {
            center,
            extent: [4.0, 3.0],
            ..Default::default()
        }
    }

    fn pane(center: [f32; 3], normal: [f32; 3]) -> GlassPanel {
        GlassPanel {
            center,
            normal,
            half_size: [1.0, 1.5],
            ..Default::default()
        }
    }

    #[test]
    fn pane_plane_passes_through_center_with_unit_normal() {
        let p = pane_plane([0.0, 0.0, 1.0], [2.0, 1.0, -3.0]);
        assert_eq!(p, [0.0, 0.0, 1.0, 3.0]);
        let on = p[0] * 2.0 + p[1] * 1.0 + p[2] * -3.0 + p[3];
        assert!(on.abs() < 1e-6);
    }

    #[test]
    fn pane_plane_offset_is_negative_normal_dot_center() {
        let (n, c) = ([0.6, 0.0, 0.8], [2.0, 5.0, -1.0]);
        let p = pane_plane(n, c);
        assert!((p[3] + (n[0] * c[0] + n[1] * c[1] + n[2] * c[2])).abs() < 1e-5);
    }

    #[test]
    fn water_plane_is_horizontal_at_the_surface() {
        assert_eq!(water_plane(&pool([1.0, -2.0, 3.0])), [0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn water_takes_the_first_slot_and_hidden_reflectors_are_not_drawn() {
        let water = [pool([0.0, 0.0, 0.0])];
        let mut hidden = pane([0.0, 1.0, -5.0], [0.0, 0.0, 1.0]);
        hidden.visible = false;
        let glass = [pane([3.0, 1.0, -5.0], [1.0, 0.0, 0.0]), hidden];
        let r = PlanarReflectors::plan(&water, &glass, budget(4));
        assert_eq!(r.assignment.slots, vec![Some(0), Some(1), Some(2)]);
        let drawn: Vec<usize> = r.reflectors.iter().map(|p| p.slot).collect();
        assert_eq!(drawn, vec![0, 1], "the hidden pane renders no mirror");
        assert_eq!(r.overflow(), 0);
    }

    #[test]
    fn the_resolution_scales_the_mirror_target() {
        let half = PlanarBudget {
            planes: 1,
            resolution: PassResolution::Half,
        };
        let r = PlanarReflectors::plan(&[pool([0.0; 3])], &[], half);
        assert_eq!(r.target_size(1920, 1081), (960, 540));
        assert_eq!(r.target_size(1, 1), (1, 1));
        let full = PlanarReflectors::plan(&[pool([0.0; 3])], &[], budget(1));
        assert_eq!(full.target_size(1920, 1080), (1920, 1080));
        assert_eq!(full.planes().len(), 1);
    }

    #[test]
    fn the_budget_is_capped_at_capacity_and_overflow_is_counted() {
        let glass: Vec<GlassPanel> = (0..6)
            .map(|i| pane([i as f32 * 10.0, 1.0, 0.0], [1.0, 0.0, 0.0]))
            .collect();
        let r = PlanarReflectors::plan(&[], &glass, budget(99));
        assert_eq!(r.assignment.representatives.len(), MAX_PLANAR_PLANES);
        assert_eq!(r.overflow(), 6 - MAX_PLANAR_PLANES);
        assert_eq!(r.reflectors.len(), MAX_PLANAR_PLANES);
    }

    #[test]
    fn the_water_hull_covers_the_wave_crests_and_the_ripple_lookup() {
        let mut water = pool([0.0, 0.0, 0.0]);
        water.waves = vec![WaterWave {
            amplitude: 0.5,
            wavelength: 4.0,
            steepness: 1.0,
            ..Default::default()
        }];
        water.roughness = 1.0;
        let flat = {
            let mut w = water.clone();
            w.waves.clear();
            w.roughness = 0.0;
            w
        };
        // Seen from above and in front: the crests rise toward the camera, so
        // the waved surface reaches lower on screen than the flat one.
        let vp = looking_down_z_from([0.0, 6.0, 30.0]);
        let waved = water_hull(&water).uv_rect(vp).unwrap();
        let still = water_hull(&flat).uv_rect(vp).unwrap();
        let lookup = WaterParams::planar_lane(1.0, true)[1];
        assert!(lookup > 0.0);
        assert!(
            waved.max[1] > still.max[1] + lookup * 0.5,
            "{waved:?} vs {still:?}"
        );
        assert!(waved.min[0] < still.min[0], "{waved:?} vs {still:?}");
    }

    #[test]
    fn the_glass_hull_matches_the_drawn_quad() {
        let g = pane([0.0, 1.0, -5.0], [0.0, 0.0, 1.0]);
        let (verts, _) =
            crate::geometry::glass_quad::build_glass_quad(g.center, g.normal, g.half_size);
        let vp = looking_down_z_from([0.0, 1.0, 0.0]);
        let quad = ReflectorHull::quad(core::array::from_fn(|i| verts[i].0), 0.0);
        assert_eq!(glass_hull(&g).uv_rect(vp), quad.uv_rect(vp));
    }

    #[test]
    fn the_frame_plan_skips_a_pool_behind_the_camera() {
        let water = [pool([0.0, 0.0, -20.0]), pool([0.0, 5.0, 30.0])];
        let r = PlanarReflectors::plan(&water, &[], budget(4));
        let plan = r.frame_plan(looking_down_z_from([0.0, 2.0, 0.0]));
        assert!(plan.samples_mirror(r.assignment.slots[0]));
        assert!(!plan.samples_mirror(r.assignment.slots[1]));
    }
}
