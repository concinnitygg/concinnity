//! The reflection-probe set the forward, SSR, transparent and ray-traced
//! passes read: a `ProbeSet` header holding the live count, one `ProbeUniforms`
//! record per probe in a structured buffer, and one cube-map array holding every
//! probe's prefiltered radiance, a cube per probe. Matches `shaders/probe_types.hlsl`.
//! Nothing in a shader depends on how many probes the host allocated room for.

/// One reflection probe's parallax box. The specular IBL term box-projects the
/// reflection vector against [box_min, box_max] (the probe's influence volume)
/// and re-anchors the cube sample at the box hit relative to `probe_pos` (the
/// capture point), so a static captured cube tracks a moving first-person
/// camera. Three float4s keep every field 16-byte aligned. `box_min.w` is the
/// enabled flag: 0 disables parallax (and signals no baked probe), so the shader
/// samples the raw reflection vector.
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct ProbeUniforms {
    /// xyz = influence-box min; w = enabled (1.0 = parallax on, 0.0 = off).
    pub box_min: [f32; 4],
    /// xyz = influence-box max; w unused.
    pub box_max: [f32; 4],
    /// xyz = probe capture position; w unused.
    pub probe_pos: [f32; 4],
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
    /// Padding to one 16-byte constant register.
    pub _pad: [u32; 2],
}

impl ProbeSet {
    /// An empty set: no probes, so the shaders fall back to the sky.
    pub const EMPTY: ProbeSet = ProbeSet {
        count: 0,
        mip_count: 0,
        _pad: [0; 2],
    };

    /// The header for `count` live records of `mip_count`-level cubes.
    pub fn new(count: usize, mip_count: u32) -> ProbeSet {
        ProbeSet {
            count: count as u32,
            mip_count,
            ..ProbeSet::EMPTY
        }
    }
}

/// Records a probe record buffer is allocated with before a world asks for more:
/// the automatic seeding budget, so a seeded world never grows it.
pub const DEFAULT_PROBE_RECORD_CAPACITY: usize = crate::render::reflection_probe::AUTO_SEED_BUDGET;

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
/// Built by [`crate::render::reflection_probe::PrefilterPlan`], which is what
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

    // `ProbeUniforms` in `shaders/probe_types.hlsl`: three float4s, the element
    // stride of the probe record buffer on every target.
    #[test]
    fn probe_uniforms_layout_matches_the_shader() {
        assert_eq!(size_of::<ProbeUniforms>(), 48);
        assert_eq!(offset_of!(ProbeUniforms, box_min), 0);
        assert_eq!(offset_of!(ProbeUniforms, box_max), 16);
        assert_eq!(offset_of!(ProbeUniforms, probe_pos), 32);
    }

    // `ProbeSet` in `shaders/probe_types.hlsl`: the count and the cube mip
    // count padded with two scalars, one 16-byte constant register.
    #[test]
    fn probe_set_layout_matches_the_shader() {
        assert_eq!(size_of::<ProbeSet>(), 16);
        assert_eq!(offset_of!(ProbeSet, count), 0);
        assert_eq!(offset_of!(ProbeSet, mip_count), 4);
        assert_eq!(offset_of!(ProbeSet, _pad), 8);
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
