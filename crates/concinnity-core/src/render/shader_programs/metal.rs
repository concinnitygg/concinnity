//! Single-source engine shader programs for the Metal backend.
//!
//! Each program compiles a file under `src/render/shaders/` to its own metallib
//! through dxc's SPIR-V, spirv-cross and the Metal toolchain: at build time
//! where the host has them, and at renderer init otherwise, cached in the
//! content-addressed shader cache. Each library holds one entry point, so a
//! variant binds only the resources it reads, and its artifact name is the key
//! the build script files it under and the renderer looks it up by.
//!
//! The sources' `CN_BACKEND_METAL` branches reproduce the engine's Metal
//! binding layout; `assert_metal_abi` in the device build script locks the
//! emitted slot assignment.
//!
//! Every depth-reading program reads the resolved single-sample depth here,
//! where Vulkan and DirectX read the multisampled original, so each compiles
//! once, under `USE_MSAA 0`.

use super::{ShaderProgram, Table, shared};

/// Where each member of the bindless main pass's texture argument buffer
/// (`buffer(7)` in `main_bindless.hlsl`) sits, by `[[id(n)]]`. The fixed
/// members come first and the unsized texture pool last, so no fixed id depends
/// on the pool's length. The ids are the source's register numbers; the device
/// build script reads each one back from the emitted MSL.
pub mod bindless_textures {
    /// The cascaded shadow map array.
    pub const SHADOW_MAP: usize = 0;
    /// The sky irradiance cube.
    pub const IRRADIANCE_CUBE: usize = 1;
    /// The sky prefilter cube.
    pub const PREFILTER_CUBE: usize = 2;
    /// The blurred SSAO occlusion.
    pub const SSAO: usize = 3;
    /// The reflection-probe cube array.
    pub const PROBE_CUBES: usize = 4;
    /// The spot shadow map array.
    pub const SPOT_SHADOW_MAP: usize = 5;
    /// The LTC matrix table.
    pub const LTC_MATRIX: usize = 6;
    /// The LTC magnitude table.
    pub const LTC_MAGNITUDE: usize = 7;
    /// Members ahead of the pool. A pass that reads only the pool binds the
    /// buffer at this many members' offset.
    pub const FIXED: usize = 8;

    /// The id of pool slot `slot`.
    pub const fn pool(slot: usize) -> usize {
        FIXED + slot
    }
}

// Metal builds the Hi-Z pyramid a mip at a time: an init kernel, chosen by the
// main pass's sample count, then one downsample dispatch per level.
/// `hiz_init_msaa` from `hiz_build.hlsl`.
pub static HIZ_INIT_MSAA: ShaderProgram = ShaderProgram {
    file: "hiz_build.hlsl",
    entry: "hiz_init_msaa",
    label: "hiz_init_msaa.hlsl",
    gates: &["HIZ_INIT_MSAA"],
    msaa: false,
};
/// `hiz_init_single` from `hiz_build.hlsl`.
pub static HIZ_INIT_SINGLE: ShaderProgram = ShaderProgram {
    file: "hiz_build.hlsl",
    entry: "hiz_init_single",
    label: "hiz_init_single.hlsl",
    gates: &["HIZ_INIT_SINGLE"],
    msaa: false,
};
/// `hiz_downsample` from `hiz_build.hlsl`.
pub static HIZ_DOWNSAMPLE: ShaderProgram = ShaderProgram {
    file: "hiz_build.hlsl",
    entry: "hiz_downsample",
    label: "hiz_downsample.hlsl",
    gates: &["HIZ_DOWNSAMPLE"],
    msaa: false,
};

/// Every program in this module.
pub static ALL: &[&ShaderProgram] = &[&HIZ_INIT_MSAA, &HIZ_INIT_SINGLE, &HIZ_DOWNSAMPLE];

/// Everything the Metal backend compiles.
pub static TABLE: Table = Table {
    programs: &[shared::ALL, ALL],
    msaa: &[false],
};

#[cfg(test)]
mod tests {
    use super::*;

    // Every fixed member has its own id below the pool, so the pool's length
    // never moves one, and pool slot 0 is the first id past them.
    #[test]
    fn the_pool_follows_every_fixed_bindless_member() {
        use bindless_textures as b;
        let mut fixed = [
            b::SHADOW_MAP,
            b::IRRADIANCE_CUBE,
            b::PREFILTER_CUBE,
            b::SSAO,
            b::PROBE_CUBES,
            b::SPOT_SHADOW_MAP,
            b::LTC_MATRIX,
            b::LTC_MAGNITUDE,
        ];
        fixed.sort_unstable();
        assert_eq!(
            fixed,
            core::array::from_fn(|i| i),
            "fixed ids are 0..FIXED, gap-free"
        );
        assert_eq!(fixed.len(), b::FIXED);
        assert_eq!(b::pool(0), b::FIXED);
        assert_eq!(b::pool(1023), 1031);
    }

    #[test]
    fn every_declared_program_is_in_the_table() {
        super::super::declared::assert_table_is_complete(include_str!("metal.rs"));
    }
}
