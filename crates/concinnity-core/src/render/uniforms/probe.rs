//! The reflection-probe set the forward, SSR and ray-traced resolves all read.
//! Matches `ProbeUniforms` / `ProbeSet` in `shaders/probe_types.slang`, whose
//! `MAX_PROBES` is baked in from the constant below.

/// Maximum reflection probes a frame can bind. The shader's `MAX_PROBES` define
/// is injected from this value, so the two cannot drift.
pub const MAX_PROBES: usize = 8;

// Every automatically seeded probe has to fit in the bound set.
const _: () = assert!(crate::render::reflection_probe::AUTO_SEED_BUDGET <= MAX_PROBES);

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

impl ProbeUniforms {
    /// The "no probe" value: parallax disabled, so the shader samples the raw
    /// reflection vector (which, with the probe cube slot aliasing the sky until
    /// a bake, reproduces the pre-probe reflection exactly).
    pub const DISABLED: ProbeUniforms = ProbeUniforms {
        box_min: [0.0; 4],
        box_max: [0.0; 4],
        probe_pos: [0.0; 4],
    };
}

/// The full set of reflection probes. `count` is how many of `probes` are live;
/// the fragment shader blends every probe whose influence box covers the surface
/// (a partition-of-unity weight by signed box distance), falling back to the
/// nearest when the surface is outside all boxes, and samples those slices of
/// the probe cube array. Slices beyond `count` hold the sky fallback cube and a
/// `DISABLED` box.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct ProbeSet {
    /// Live entries in `probes`.
    pub count: u32,
    /// Padding so the field layout matches the shader-side struct.
    /// Padding so the field layout matches the shader-side struct.
    pub _pad: [u32; 3],
    /// Probe entries; the first `count` are live.
    pub probes: [ProbeUniforms; MAX_PROBES],
}

impl ProbeSet {
    /// An empty set: no probes, so the shader falls back to the sky.
    pub const EMPTY: ProbeSet = ProbeSet {
        count: 0,
        _pad: [0; 3],
        probes: [ProbeUniforms::DISABLED; MAX_PROBES],
    };
}

/// Per-dispatch params for the runtime reflection-probe prefilter kernels.
/// Matches `ProbePrefilterParams` in `shaders/probe_prefilter.slang`. 32 bytes.
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

    // Eight tightly-packed 4-byte scalars, the layout
    // `ProbePrefilterParams` in `shaders/probe_prefilter.slang` declares. Every
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
