//! The SDF volumes' bindings, which every volume's program shares. A volume's
//! program is spliced from its world, so no table row checks these.

/// The DirectX registers.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Sampler, Srv};

    /// The pass's view block.
    pub const VIEW: Register<Cbv> = Register::new("view_cb", 0);
    /// The volume's block.
    pub const VOLUME: Register<Cbv> = Register::new("vol_cb", 1);
    /// The light block.
    pub const LIGHTS: Register<Cbv> = Register::new("lights_cb", 2);
    /// The cascaded shadow block.
    pub const SHADOW: Register<Cbv> = Register::new("shadow_cb", 3);
    /// The cascade a shadow draw renders, as root constants.
    pub const CASCADE: Register<Cbv> = Register::new("cascade_cb", 4);
    /// The cascaded shadow map array, with the irradiance, prefilter and scene
    /// color in the registers after.
    pub const SHADOW_MAP: Register<Srv> = Register::new("shadow_map", 0);
    /// The sky irradiance cube.
    pub const IRRADIANCE_CUBE: Register<Srv> = Register::new("irradiance_cube", 1);
    /// The sky prefilter cube.
    pub const PREFILTER_CUBE: Register<Srv> = Register::new("prefilter_cube", 2);
    /// The scene color behind the volume.
    pub const SCENE_COLOR: Register<Srv> = Register::new("scene_color", 3);
    /// The shadow comparison sampler, with the cube and scene samplers in the registers after.
    pub const SHADOW_SAMPLER: Register<Sampler> = Register::new("shadow_samp", 0);
    /// The cubes' sampler.
    pub const CUBE_SAMPLER: Register<Sampler> = Register::new("cube_samp", 1);
    /// The scene color's sampler.
    pub const SCENE_SAMPLER: Register<Sampler> = Register::new("scene_samp", 2);
}
