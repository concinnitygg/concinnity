//! The canopy block: what the main pass shades a grass field's far field
//! with, past where the blades end. Matches `CanopyParams` / `CanopyLayer` in
//! `shaders/canopy.hlsl`.

/// Most layers the canopy block describes: every far field in the world
/// together.
pub const MAX_CANOPY_LAYERS: usize = 16;

/// One grass layer as its ground and far field read it: the blade look, how
/// much leaf it grows, how its blades lean and crowd, and the litter they
/// shed. 80 bytes, every member in its own 16-byte row.
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
    /// Linear RGB of the dead grass on the ground.
    pub litter_color: [f32; 3],
    /// How strongly blades crowd toward their clump's center, in [0, 1].
    pub clumping: f32,
}

/// The canopy block: every far field's layers, the frame a planet's ground is
/// laid out in, the noise its height is summed from, its coasts, and where the
/// blades hand the field to it. 1568 bytes.
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
    /// Rows of the turn from one height octave's lattice to the next, in the
    /// planet's axes.
    pub ground_turn: [[f32; 4]; 3],
    /// Each height octave's hash salt, four to a row.
    pub ground_salts: [[u32; 4]; 4],
    /// Lattice cells of the broadest height octave per unit of direction.
    pub ground_frequency: f32,
    /// Meters of height per unit of the octaves' weighted sum.
    pub ground_height_scale: f32,
    /// Width of the broadest height octave's lattice cells, in meters.
    pub ground_cell: f32,
    /// Height octaves summed; 0 off a planet.
    pub ground_octaves: u32,
    /// Each continent octave's hash salt, four to a row.
    pub continent_salts: [[u32; 4]; 2],
    /// Lattice cells of the broadest continent octave per unit of direction.
    pub continent_frequency: f32,
    /// Meters inland per unit of the continent octaves' weighted sum.
    pub continent_scale: f32,
    /// Meters inland at a weighted sum of 0, negated: the coast lies where the
    /// sum times `continent_scale` reaches it.
    pub continent_offset: f32,
    /// Continent octaves summed; 0 on a planet without seas.
    pub continent_octaves: u32,
    /// Width of the rise from the waterline before the hills, in meters.
    pub beach_width: f32,
    /// How high that rise climbs, in meters.
    pub beach_rise: f32,
    /// How far past the beach the hills take to reach their full height, in
    /// meters.
    pub hill_rise: f32,
    /// Deepest the sea floor lies below sea level, in meters.
    pub ocean_depth: f32,
    /// How far past the waterline the sea floor takes to fall most of the way
    /// to the ocean depth, in meters.
    pub shelf_length: f32,
    /// The sea floor's lift per meter of hill under it.
    pub seabed_lift: f32,
    /// Keeps the record a whole number of rows.
    pub _pad: [f32; 2],
    /// This frame's hand-off from the blades
    /// ([`GrassLodView`](crate::render::environment::grass::GrassLodView)): where it
    /// starts, where the blades end, and its phase
    /// ([`GrassHistory::crossfade_phase`](crate::render::environment::grass::GrassHistory::crossfade_phase));
    /// `w` unused.
    pub band: [f32; 4],
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
        assert_eq!(size_of::<CanopyLayerGpu>(), 80);
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
        assert_eq!(offset_of!(CanopyLayerGpu, litter_color), 64);
        assert_eq!(offset_of!(CanopyLayerGpu, clumping), 76);
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
        assert_eq!(offset_of!(CanopyParams, ground_turn), 1344);
        assert_eq!(offset_of!(CanopyParams, ground_salts), 1392);
        assert_eq!(offset_of!(CanopyParams, ground_frequency), 1456);
        assert_eq!(offset_of!(CanopyParams, ground_height_scale), 1460);
        assert_eq!(offset_of!(CanopyParams, ground_cell), 1464);
        assert_eq!(offset_of!(CanopyParams, ground_octaves), 1468);
        assert_eq!(offset_of!(CanopyParams, continent_salts), 1472);
        assert_eq!(offset_of!(CanopyParams, continent_frequency), 1504);
        assert_eq!(offset_of!(CanopyParams, continent_scale), 1508);
        assert_eq!(offset_of!(CanopyParams, continent_offset), 1512);
        assert_eq!(offset_of!(CanopyParams, continent_octaves), 1516);
        assert_eq!(offset_of!(CanopyParams, beach_width), 1520);
        assert_eq!(offset_of!(CanopyParams, beach_rise), 1524);
        assert_eq!(offset_of!(CanopyParams, hill_rise), 1528);
        assert_eq!(offset_of!(CanopyParams, ocean_depth), 1532);
        assert_eq!(offset_of!(CanopyParams, shelf_length), 1536);
        assert_eq!(offset_of!(CanopyParams, seabed_lift), 1540);
        assert_eq!(offset_of!(CanopyParams, _pad), 1544);
        assert_eq!(offset_of!(CanopyParams, band), 1552);
        assert_eq!(size_of::<CanopyParams>(), 1568);
        // Metal sets the block inline, which caps it at 4 KiB.
        const { assert!(size_of::<CanopyParams>() <= 4096) };
        assert_eq!(CanopyParams::default().layer_count, 0);
    }
}
