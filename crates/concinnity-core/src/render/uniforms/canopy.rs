//! The canopy block: what the main pass shades a grass field's far field
//! with, past where the blades end. Matches `CanopyParams` / `CanopyLayer` in
//! `shaders/canopy.hlsl`.

/// Most layers the canopy block describes: every far field in the world
/// together.
pub const MAX_CANOPY_LAYERS: usize = 16;

/// One grass layer as its far field reads it: the blade look, how much leaf
/// it grows, and how its blades lean. 64 bytes, every member in its own
/// 16-byte row.
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct CanopyLayerGpu {
    /// Linear RGB at the blade root.
    pub root_color: [f32; 3],
    /// Hue and brightness variation between clumps, in [0, 1].
    pub color_variation: f32,
    /// Linear RGB at the blade tip.
    pub tip_color: [f32; 3],
    /// How far the field's color drifts across hundreds of meters, in [0, 1].
    pub macro_variation: f32,
    /// Clump diameter, in meters.
    pub clump_size: f32,
    /// Blade height variation as a fraction of the mean height.
    pub height_variance: f32,
    /// One-sided blade area per square meter of ground where the layer grows
    /// fully.
    pub leaf_area: f32,
    /// A blade's mean bend, its lean and the wind's push: the tip's sideways
    /// offset per unit of length.
    pub bend: f32,
    /// The layer's placement seed, the kernel's, so its clumps line up with
    /// the blades'.
    pub seed: u32,
    /// The blade area seen per unit of leaf area along the mean direction the
    /// sky reaches the ground from.
    pub sky_projection: f32,
    /// Keeps the record a whole number of rows.
    pub _pad: [f32; 2],
}

/// The canopy block: every far field's layers, and the frame a planet's ground
/// is laid out in. 1088 bytes.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct CanopyParams {
    /// Rows over a simulated point `(x, y, z, 1)` giving it relative to the
    /// planet's center, along the planet's own axes; zero off a planet.
    pub planet_rows: [[f32; 4]; 3],
    /// The planet's radius, in meters.
    pub planet_radius: f32,
    /// The meadow noise's lattice cells per unit of direction.
    pub meadow_scale: f32,
    /// The mean of the meadow's density over the planet, which the far field
    /// fades toward where a pixel covers several of its patches.
    pub meadow_mean: f32,
    /// Layers in `layers`.
    pub layer_count: u32,
    /// The layers, each far field's a contiguous run.
    pub layers: [CanopyLayerGpu; MAX_CANOPY_LAYERS],
}

impl Default for CanopyParams {
    fn default() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    // Mirrors `CanopyLayer` in shaders/canopy.hlsl.
    #[test]
    fn canopy_layer_matches_hlsl() {
        assert_eq!(size_of::<CanopyLayerGpu>(), 64);
        assert_eq!(offset_of!(CanopyLayerGpu, root_color), 0);
        assert_eq!(offset_of!(CanopyLayerGpu, color_variation), 12);
        assert_eq!(offset_of!(CanopyLayerGpu, tip_color), 16);
        assert_eq!(offset_of!(CanopyLayerGpu, macro_variation), 28);
        assert_eq!(offset_of!(CanopyLayerGpu, clump_size), 32);
        assert_eq!(offset_of!(CanopyLayerGpu, height_variance), 36);
        assert_eq!(offset_of!(CanopyLayerGpu, leaf_area), 40);
        assert_eq!(offset_of!(CanopyLayerGpu, bend), 44);
        assert_eq!(offset_of!(CanopyLayerGpu, seed), 48);
        assert_eq!(offset_of!(CanopyLayerGpu, sky_projection), 52);
        assert_eq!(offset_of!(CanopyLayerGpu, _pad), 56);
    }

    // Mirrors `CanopyParams` in shaders/canopy.hlsl.
    #[test]
    fn canopy_params_matches_hlsl() {
        assert_eq!(offset_of!(CanopyParams, planet_rows), 0);
        assert_eq!(offset_of!(CanopyParams, planet_radius), 48);
        assert_eq!(offset_of!(CanopyParams, meadow_scale), 52);
        assert_eq!(offset_of!(CanopyParams, meadow_mean), 56);
        assert_eq!(offset_of!(CanopyParams, layer_count), 60);
        assert_eq!(offset_of!(CanopyParams, layers), 64);
        assert_eq!(size_of::<CanopyParams>(), 1088);
        // Metal sets the block inline, which caps it at 4 KiB.
        const { assert!(size_of::<CanopyParams>() <= 4096) };
        assert_eq!(CanopyParams::default().layer_count, 0);
    }
}
