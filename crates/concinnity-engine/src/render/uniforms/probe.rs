//! The reflection-probe set the forward, SSR, transparent and ray-traced
//! passes read: a `ProbeSet` header holding the live count, one `ProbeUniforms`
//! record per probe in a structured buffer, and one cube-map array holding every
//! probe's prefiltered radiance, a cube per probe. Matches `shaders/probe_types.hlsl`.
//! Nothing in a shader depends on how many probes the host allocated room for.

use concinnity_core::math::SaturatingFrom;

/// One reflection probe's parallax box. The specular IBL term box-projects the
/// reflection vector against [`box_min`, `box_max`] (the probe's influence volume)
/// and re-anchors the cube sample at the box hit relative to `probe_pos` (the
/// capture point), so a static captured cube tracks a moving first-person
/// camera. Four float4s keep every field 16-byte aligned. `box_min.w` is the
/// parallax flag: 0 samples the raw reflection vector, for a probe whose scene
/// lies far past its box (one riding on a prop, which aims at its prop's
/// [`ProbeSet::ground`] instead). `box_max.w` is the sky
/// fallback flag: 0 keeps the capture at any distance past the box.
/// `probe_pos.w` is the finest mip the probe's cube holds, which a lookup never
/// reads finer than. `blend_margin.w` marks a probe captured around a prop it
/// leaves out, whose reflection on that prop already holds what occludes it.
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct ProbeUniforms {
    /// xyz = influence-box min; w = parallax (1.0 = on, 0.0 = off).
    pub box_min: [f32; 4],
    /// xyz = influence-box max; w = sky fallback (1.0 = fade to the
    /// environment past the influence, 0.0 = keep the capture).
    pub box_max: [f32; 4],
    /// xyz = probe capture position; w = the cube's finest mip.
    pub probe_pos: [f32; 4],
    /// xyz = distance along each axis the blend weight ramps over across the
    /// box surface, so the probe's influence reaches that far outside the box;
    /// w = 1.0 for a probe captured around the prop of [`ProbeSet::subject`],
    /// resting on [`ProbeSet::ground`].
    pub blend_margin: [f32; 4],
}

impl ProbeUniforms {
    /// The distance along each axis the probe's blend weight ramps over across
    /// its influence box's surface, as the shaders read it.
    pub fn blend_margin(&self) -> [f32; 3] {
        let [x, y, z, _] = self.blend_margin;
        [x, y, z]
    }

    /// The corners of everywhere the probe's blend weight can be non-zero: the
    /// influence box grown by its blend margin, as the shaders grow it.
    pub fn influence_bounds(&self) -> ([f32; 3], [f32; 3]) {
        let margin = self.blend_margin();
        (
            core::array::from_fn(|i| self.box_min[i] - margin[i]),
            core::array::from_fn(|i| self.box_max[i] + margin[i]),
        )
    }

    /// How much of the reflection at `point` the environment takes over from
    /// this probe when it is the nearest capture of a point no probe's
    /// influence reaches, as `probe_sky_share` in `shaders/probe_common.hlsl`
    /// computes it: 0 inside [`Self::influence_bounds`], rising smoothly to 1
    /// one largest half-extent of the box beyond them. A capture stands in for
    /// the scenery about as far past its box as the box is wide. 0 for a record
    /// without the sky fallback (`box_max.w` off), which keeps the probe whole.
    #[cfg(test)]
    fn sky_share(&self, point: [f32; 3]) -> f32 {
        if self.box_max[3] <= 0.5 {
            return 0.0;
        }
        let (lo, hi) = self.influence_bounds();
        let outside: [f32; 3] =
            core::array::from_fn(|i| (lo[i] - point[i]).max(point[i] - hi[i]).max(0.0));
        let distance = concinnity_core::math::sqrt(outside.iter().map(|d| d * d).sum());
        let reach = (0..3)
            .map(|i| 0.5 * (self.box_max[i] - self.box_min[i]))
            .fold(1e-4f32, f32::max);
        let t = (distance / reach).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }
}

/// The header of the probe set: how many records of the probe record buffer are
/// live, and so how many slices of the probe cube array the shaders read. Slices
/// and records past `count` are never read, so they need no fallback content.
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct ProbeSet {
    /// Live records in the probe record buffer.
    pub count: u32,
    /// Mip levels of each probe cube, which map a surface's roughness onto the
    /// cube's prefilter chain independently of the environment map's.
    pub mip_count: u32,
    /// The object records `[first, end)` of the prop a live prop probe rides
    /// on; empty without one.
    pub subject: [u32; 2],
    /// The plane that prop rests on: its up (xyz, unit) and offset (w), so
    /// `dot(up, p) == w` on it. A lookup of the prop probe re-aims a ray
    /// falling toward the plane at where it meets it. Zero without one.
    pub ground: [f32; 4],
}

impl ProbeSet {
    /// An empty set: no probes, so the shaders fall back to the sky.
    pub const EMPTY: Self = Self {
        count: 0,
        mip_count: 0,
        subject: [0; 2],
        ground: [0.0; 4],
    };

    /// The header for `count` live records of `mip_count`-level cubes.
    pub fn new(count: usize, mip_count: u32) -> Self {
        Self {
            count: u32::saturating_from(count),
            mip_count,
            ..Self::EMPTY
        }
    }
}

/// Records a probe record buffer is allocated with before a world asks for more:
/// the automatic seeding budget, so a seeded world never grows it.
pub const DEFAULT_PROBE_RECORD_CAPACITY: usize = crate::render::probe::seed::AUTO_SEED_BUDGET;

/// The capacity a probe resource must be reallocated to so it holds `required`
/// entries, or `None` while `current` already does. A reallocation never goes
/// below `floor`, which is how a buffer that grows lazily per frame lands on
/// the capacity its siblings already have instead of growing once per probe.
pub fn grown_probe_capacity(current: usize, required: usize, floor: usize) -> Option<usize> {
    (required > current).then(|| required.max(floor))
}

/// Per-dispatch params for the runtime reflection-probe prefilter kernels.
/// Matches `ProbePrefilterParams` in `shaders/probe_prefilter.hlsl`. 32 bytes.
///
/// Built by [`crate::render::probe::capture::PrefilterPlan`], which is what
/// decides the sizes, the roughness per mip and the firefly clamp so the three
/// backends dispatch identical work.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct ProbePrefilterParams {
    /// Destination cube-face edge in texels.
    pub dst_size: u32,
    /// Source cube-face edge at mip 0, in texels.
    pub src_size: u32,
    /// GGX samples per output texel.
    pub sample_count: u32,
    /// Source mip the downsample kernel reduces; it writes `src_mip + 1`.
    pub src_mip: u32,
    /// GGX roughness of the destination mip.
    pub roughness: f32,
    /// Firefly clamp luminance; `<= 0` disables the cap.
    pub clamp_lum: f32,
    /// Mip levels the source pyramid has, bounding the solid-angle lod.
    pub src_mip_count: f32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    // `ProbeUniforms` in `shaders/probe_types.hlsl`: four float4s, the element
    // stride of the probe record buffer on every target.
    #[test]
    fn probe_uniforms_layout_matches_the_shader() {
        assert_eq!(size_of::<ProbeUniforms>(), 64);
        assert_eq!(offset_of!(ProbeUniforms, box_min), 0);
        assert_eq!(offset_of!(ProbeUniforms, box_max), 16);
        assert_eq!(offset_of!(ProbeUniforms, probe_pos), 32);
        assert_eq!(offset_of!(ProbeUniforms, blend_margin), 48);
    }

    #[test]
    fn influence_bounds_grow_each_axis_by_its_margin() {
        let probe = ProbeUniforms {
            box_min: [-4.0, 0.0, -10.0, 1.0],
            box_max: [4.0, 2.0, 10.0, 0.0],
            probe_pos: [0.0; 4],
            blend_margin: [0.5, 0.25, 2.0, 0.0],
        };
        assert_eq!(probe.blend_margin(), [0.5, 0.25, 2.0]);
        let (lo, hi) = probe.influence_bounds();
        assert_eq!((lo, hi), ([-4.5, -0.25, -12.0], [4.5, 2.25, 12.0]));
    }

    fn room() -> ProbeUniforms {
        ProbeUniforms {
            box_min: [-4.0, 0.0, -2.0, 1.0],
            box_max: [4.0, 3.0, 2.0, 1.0],
            probe_pos: [0.0, 1.5, 0.0, 0.0],
            blend_margin: [0.5, 0.5, 0.5, 0.0],
        }
    }

    #[test]
    fn the_sky_takes_no_share_within_the_probes_influence() {
        let probe = room();
        assert_eq!(probe.sky_share([0.0, 1.5, 0.0]), 0.0);
        assert_eq!(probe.sky_share([4.5, 3.5, 2.5]), 0.0);
        assert_eq!(probe.sky_share([-4.5, -0.5, 0.0]), 0.0);
    }

    #[test]
    fn the_sky_takes_over_one_largest_half_extent_past_the_influence() {
        let probe = room();
        // The largest half-extent is 4 (x), whichever axis the point leaves by.
        assert_eq!(probe.sky_share([0.0, 1.5, 2.5 + 4.0]), 1.0);
        assert_eq!(probe.sky_share([0.0, 3.5 + 4.0, 0.0]), 1.0);
        assert_eq!(probe.sky_share([100.0, 1.5, 0.0]), 1.0);
        let half = probe.sky_share([0.0, 1.5, 2.5 + 2.0]);
        assert!((half - 0.5).abs() < 1e-6, "{half}");
    }

    #[test]
    fn the_sky_share_rises_with_distance_from_a_corner() {
        let probe = room();
        let mut last = 0.0;
        for step in 1..=14 {
            let d = step as f32 * 0.2;
            let share = probe.sky_share([4.5 + d, 3.5 + d, 2.5]);
            assert!(share > last && share <= 1.0, "{d}: {share} after {last}");
            last = share;
        }
        // A diagonal step of d out of the corner is sqrt(2) d away.
        let diagonal = probe.sky_share([4.5 + 2.0, 3.5 + 2.0, 0.0]);
        let straight = probe.sky_share([4.5 + 2.0 * core::f32::consts::SQRT_2, 1.5, 0.0]);
        assert!((diagonal - straight).abs() < 1e-5);
    }

    // A probe riding on a prop has no parallax, yet still gives way to the
    // sky past its reach like any other.
    #[test]
    fn a_probe_without_parallax_still_fades_to_the_sky() {
        let probe = ProbeUniforms {
            box_min: [-4.0, 0.0, -2.0, 0.0],
            ..room()
        };
        assert_eq!(probe.sky_share([100.0, 1.5, 0.0]), 1.0);
        assert_eq!(probe.sky_share([0.0, 1.5, 0.0]), 0.0);
    }

    #[test]
    fn a_probe_without_the_sky_fallback_keeps_its_whole_share_at_any_distance() {
        let probe = ProbeUniforms {
            box_max: [4.0, 3.0, 2.0, 0.0],
            ..room()
        };
        for d in [0.0, 0.5, 2.0, 4.0, 40.0, 1e6] {
            assert_eq!(probe.sky_share([4.5 + d, 3.5 + d, 2.5 + d]), 0.0, "{d}");
        }
    }

    #[test]
    fn the_sky_fallback_flag_does_not_change_the_fade_when_on() {
        let on = room();
        let off = ProbeUniforms {
            box_max: [4.0, 3.0, 2.0, 0.0],
            ..room()
        };
        assert_eq!(on.influence_bounds(), off.influence_bounds());
        assert_eq!(on.sky_share([0.0, 1.5, 2.5]), 0.0);
        let half = on.sky_share([0.0, 1.5, 2.5 + 2.0]);
        assert!((half - 0.5).abs() < 1e-6, "{half}");
        assert_eq!(on.sky_share([0.0, 1.5, 2.5 + 4.0]), 1.0);
        assert_eq!(off.sky_share([0.0, 1.5, 2.5 + 4.0]), 0.0);
    }

    // `ProbeSet` in `shaders/probe_types.hlsl`: the count, the cube mip count
    // and the prop's record range in one 16-byte constant register, its ground
    // plane in the next.
    #[test]
    fn probe_set_layout_matches_the_shader() {
        assert_eq!(size_of::<ProbeSet>(), 32);
        assert_eq!(offset_of!(ProbeSet, count), 0);
        assert_eq!(offset_of!(ProbeSet, mip_count), 4);
        assert_eq!(offset_of!(ProbeSet, subject), 8);
        assert_eq!(offset_of!(ProbeSet, ground), 16);
        let set = ProbeSet::new(12, 10);
        assert_eq!((set.count, set.mip_count), (12, 10));
    }

    #[test]
    fn a_probe_resource_grows_only_past_its_capacity() {
        assert_eq!(grown_probe_capacity(8, 8, 1), None);
        assert_eq!(grown_probe_capacity(8, 3, 1), None);
        assert_eq!(grown_probe_capacity(8, 12, 1), Some(12));
        // An empty resource grows to what is asked, not to the floor's default.
        assert_eq!(grown_probe_capacity(0, 2, 1), Some(2));
        assert_eq!(grown_probe_capacity(0, 0, 1), None);
    }

    #[test]
    fn a_growth_lands_on_the_floor_when_it_is_larger() {
        // A per-frame buffer catching up with a 12-slice array grows once to 12,
        // not to 9, then 10, then 11 as probes install one per bake.
        assert_eq!(grown_probe_capacity(8, 9, 12), Some(12));
        assert_eq!(grown_probe_capacity(12, 10, 12), None);
        assert_eq!(
            grown_probe_capacity(0, 1, DEFAULT_PROBE_RECORD_CAPACITY),
            Some(DEFAULT_PROBE_RECORD_CAPACITY)
        );
    }

    // Eight tightly-packed 4-byte scalars, the layout
    // `ProbePrefilterParams` in `shaders/probe_prefilter.hlsl` declares. Every
    // field is a scalar, so no target 16-aligns one and shifts the rest; a
    // vector added here would, and would silently feed each kernel garbage.
    #[test]
    fn probe_prefilter_params_layout_matches_the_shader() {
        assert_eq!(size_of::<ProbePrefilterParams>(), 32);
        assert_eq!(offset_of!(ProbePrefilterParams, dst_size), 0);
        assert_eq!(offset_of!(ProbePrefilterParams, src_size), 4);
        assert_eq!(offset_of!(ProbePrefilterParams, sample_count), 8);
        assert_eq!(offset_of!(ProbePrefilterParams, src_mip), 12);
        assert_eq!(offset_of!(ProbePrefilterParams, roughness), 16);
        assert_eq!(offset_of!(ProbePrefilterParams, clamp_lum), 20);
        assert_eq!(offset_of!(ProbePrefilterParams, src_mip_count), 24);
        assert_eq!(offset_of!(ProbePrefilterParams, _pad), 28);
    }
}
