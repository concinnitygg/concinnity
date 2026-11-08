//! The composite pass's bindings: each source texture with its own sampler.

/// The DirectX registers.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Sampler, Srv};

    /// The composite's tunables, as root constants.
    pub const PARAMS: Register<Cbv> = Register::new("post", 0);
    /// The scene: the HDR resolve, or the TAA output.
    pub const HDR: Register<Srv> = Register::new("hdr_tex", 0);
    /// The bloom top octave.
    pub const BLOOM: Register<Srv> = Register::new("bloom_tex", 1);
    /// The 3D color-grading LUT.
    pub const LUT: Register<Srv> = Register::new("lut_tex", 2);
    /// The G-buffer normal and depth.
    pub const GBUFFER_NORMAL_DEPTH: Register<Srv> = Register::new("gbuf_nd_tex", 3);
    /// The G-buffer roughness.
    pub const GBUFFER_ROUGHNESS: Register<Srv> = Register::new("gbuf_rough_tex", 4);
    /// The blurred SSAO occlusion.
    pub const AO: Register<Srv> = Register::new("ao_tex", 5);
    /// The G-buffer motion.
    pub const GBUFFER_MOTION: Register<Srv> = Register::new("gbuf_motion_tex", 6);
    /// The reactive mask.
    pub const REACTIVE: Register<Srv> = Register::new("reactive_tex", 7);
    /// The sampler half of `hdr_tex`.
    pub const HDR_SAMPLER: Register<Sampler> = Register::new("hdr_tex_samp", 0);
    /// The sampler half of `bloom_tex`.
    pub const BLOOM_SAMPLER: Register<Sampler> = Register::new("bloom_tex_samp", 1);
    /// The sampler half of `lut_tex`.
    pub const LUT_SAMPLER: Register<Sampler> = Register::new("lut_tex_samp", 2);
    /// The sampler half of `gbuf_nd_tex`.
    pub const GBUFFER_NORMAL_DEPTH_SAMPLER: Register<Sampler> =
        Register::new("gbuf_nd_tex_samp", 3);
    /// The sampler half of `gbuf_rough_tex`.
    pub const GBUFFER_ROUGHNESS_SAMPLER: Register<Sampler> =
        Register::new("gbuf_rough_tex_samp", 4);
    /// The sampler half of `ao_tex`.
    pub const AO_SAMPLER: Register<Sampler> = Register::new("ao_tex_samp", 5);
    /// The sampler half of `gbuf_motion_tex`.
    pub const GBUFFER_MOTION_SAMPLER: Register<Sampler> = Register::new("gbuf_motion_tex_samp", 6);
    /// The sampler half of `reactive_tex`.
    pub const REACTIVE_SAMPLER: Register<Sampler> = Register::new("reactive_tex_samp", 7);
}
