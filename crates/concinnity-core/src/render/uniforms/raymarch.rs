//! The blocks the raymarched SDF volume pass binds.
//!
//! One declaration each, for all three backends: the three hosts bind the same
//! bytes at different slots, and the slots are the only thing they disagree
//! about. Before the pass was single-sourced these were three `#[repr(C)]`
//! copies apiece, kept in step by hand.

use crate::components::sdf_volume::SDF_PARAMS_LEN;

/// Per-frame view inputs, 208 bytes. Every field after `inv_vp` is a `float4`
/// lane or smaller, which is what keeps the three targets agreeing: a `float3`
/// occupies 16 bytes in a Metal constant buffer and 12 on SPIR-V and DXIL.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct RaymarchView {
    /// View-projection matrix, column-major.
    pub vp: [[f32; 4]; 4],
    /// Inverse view-projection matrix, column-major.
    pub inv_vp: [[f32; 4]; 4],
    /// World-space camera position in `xyz`; `w` is padding.
    pub cam_pos: [f32; 4],
    /// Render-target size in pixels.
    pub viewport: [f32; 2],
    /// Seconds since the world started, available to the authored field.
    pub time: f32,
    /// Mip count of the bound IBL prefilter cube. 0 means no `EnvironmentMap`
    /// is bound and the ambient helper takes its hemispheric fallback.
    pub prefilter_mip_count: f32,
    /// Rows of the rotation taking a world direction into the environment
    /// cubemap's baked frame, so a volume's ambient turns with the sky.
    pub sky_rot: [[f32; 4]; 3],
}

/// Per-volume uniforms, 176 bytes. `centre` and `extent` each pair with the pad
/// that completes their `float4` lane, for the reason above.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct RaymarchVolumeUniforms {
    /// World-space centre of the bounding box.
    pub centre: [f32; 3],
    /// Padding completing the lane `centre` opens.
    pub _pad0: f32,
    /// Half-widths of the bounding box.
    pub extent: [f32; 3],
    /// Padding completing the lane `extent` opens.
    pub _pad1: f32,
    /// `1 / max_gradient`; the cone-step scale factor.
    pub cone_ratio: f32,
    /// Per-volume march far clip, in metres.
    pub max_distance: f32,
    /// Per-volume step cap, clamped at load.
    pub max_steps: i32,
    /// Non-zero when the volume samples the shadow maps.
    pub receive_shadows: i32,
    /// The authored parameter block the field interprets.
    pub params: [f32; SDF_PARAMS_LEN],
}

/// Which cascade a shadow-caster draw targets, 16 bytes. A root constant on
/// DirectX, a push constant on Vulkan, and a bound buffer on Metal.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct RaymarchShadowCascade {
    /// Index into the cascade light view-projections.
    pub cascade_idx: u32,
    /// Padding to the 16-byte block the hosts allocate.
    pub _pad: [u32; 3],
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    // The sizes the three hosts allocate for these blocks. The per-field
    // offsets are checked against slangc's reflection in
    // `concinnity-device/src/shader_layout/`, which is what catches a shader-side
    // spelling that lays out differently on one target than on the others.
    #[test]
    fn the_blocks_are_the_sizes_the_hosts_bind() {
        assert_eq!(size_of::<RaymarchView>(), 208);
        assert_eq!(size_of::<RaymarchVolumeUniforms>(), 176);
        assert_eq!(size_of::<RaymarchShadowCascade>(), 16);
    }
}
