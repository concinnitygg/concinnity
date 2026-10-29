//! A ground rule and a planet's ground covers as the shaders read them.
//! Matches `GroundRule` / `GroundCover` in `shaders/ground_rule_types.hlsl`.

/// A ground rule as ramps: each bound's ramp is `x * k + c` over its measure
/// (altitude in meters, or the ground's steepness: `sqrt(1 - cos)` of the
/// angle between its normal and up),
/// rising from 0 outside the bound to 1 inside it. An open bound has `k` 0 and
/// `c` [`GROUND_RULE_OPEN`](crate::terrain::GROUND_RULE_OPEN).
/// The default holds everywhere. 32 bytes.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GroundRuleGpu {
    /// `k` and `c` of the lowest altitude's ramp, then of the highest's.
    pub altitude: [f32; 4],
    /// `k` and `c` of the gentlest slope's ramp, then of the steepest's.
    pub slope: [f32; 4],
}

/// One ground cover: where it lies and the look it gives the ground. 64
/// bytes.
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GroundCoverGpu {
    /// Where the cover lies.
    pub rule: GroundRuleGpu,
    /// Linear RGB multiplier on the albedo sample.
    pub tint: [f32; 3],
    /// Index of the albedo texture in the bindless pool.
    pub albedo_index: u32,
    /// Perceptual roughness.
    pub roughness: f32,
    /// Metallic factor.
    pub metallic: f32,
    /// Keeps the record a whole number of rows.
    pub _pad: [f32; 2],
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    // Mirrors `GroundRule` in shaders/ground_rule_types.hlsl.
    #[test]
    fn ground_rule_matches_hlsl() {
        assert_eq!(size_of::<GroundRuleGpu>(), 32);
        assert_eq!(offset_of!(GroundRuleGpu, altitude), 0);
        assert_eq!(offset_of!(GroundRuleGpu, slope), 16);
    }

    // Mirrors `GroundCover` in shaders/ground_rule_types.hlsl.
    #[test]
    fn ground_cover_matches_hlsl() {
        assert_eq!(size_of::<GroundCoverGpu>(), 64);
        assert_eq!(offset_of!(GroundCoverGpu, rule), 0);
        assert_eq!(offset_of!(GroundCoverGpu, tint), 32);
        assert_eq!(offset_of!(GroundCoverGpu, albedo_index), 44);
        assert_eq!(offset_of!(GroundCoverGpu, roughness), 48);
        assert_eq!(offset_of!(GroundCoverGpu, metallic), 52);
        assert_eq!(offset_of!(GroundCoverGpu, _pad), 56);
    }
}
