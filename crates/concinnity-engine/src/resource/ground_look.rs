//! The looks a [`Texture`](super::texture::Texture) can draw ground from: what
//! the cook synthesizes a sand or dirt surface's maps out of.

use crate::components::Vocabulary;

/// Which of a ground surface's maps a [Texture](#texture) holds.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize, Vocabulary,
)]
#[serde(rename_all = "lowercase")]
pub enum GroundMap {
    /// Base color, with the surface's relief height in alpha (0 lowest, 1
    /// highest), which height-blended ground covers read.
    #[default]
    #[vocab("albedo")]
    Albedo,
    /// Tangent-space normal.
    #[vocab("normal")]
    Normal,
}

/// The look of a patch of sand, one meter across and repeating seamlessly:
/// wind ripples over loose grains, dark mineral grains, and scattered shell
/// fragments.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct SandLook {
    /// Mean linear albedo of the dry sand.
    #[asset(default = [0.45, 0.38, 0.27])]
    pub color: [f32; 3],
    /// Perceptual roughness of the dry sand.
    #[asset(default = 0.9)]
    pub roughness: f32,
    /// Crest-to-crest spacing of the wind ripples, in meters.
    #[asset(default = 0.09)]
    pub ripple_spacing: f32,
    /// Trough-to-crest height of the ripples, in meters. 0 leaves the sand
    /// unrippled.
    #[asset(default = 0.006)]
    pub ripple_height: f32,
    /// Direction the wind blew the ripples along, in degrees from the
    /// texture's u axis. The crests run across it.
    pub ripple_direction_deg: f32,
    /// How coarse and loose the grains look, in [0, 1].
    #[asset(default = 0.5)]
    pub grain: f32,
    /// Share of dark mineral grains speckling the sand and gathering in the
    /// ripples' troughs, in [0, 1].
    #[asset(default = 0.25)]
    pub dark_grains: f32,
    /// How thickly shell fragments are strewn, in [0, 1].
    #[asset(default = 0.1)]
    pub shells: f32,
    /// Linear albedo of the shell fragments.
    #[asset(default = [0.78, 0.72, 0.63])]
    pub shell_color: [f32; 3],
    /// Seed of the pattern; two sands with different seeds never repeat each
    /// other.
    pub seed: u32,
}

/// The look of a patch of bare earth, one meter across and repeating
/// seamlessly: crumbs and clods of soil, stones lying in it, darker humus,
/// and dried cracks.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct DirtLook {
    /// Mean linear albedo of the soil.
    #[asset(default = [0.1, 0.075, 0.052])]
    pub color: [f32; 3],
    /// Perceptual roughness of the soil.
    #[asset(default = 0.95)]
    pub roughness: f32,
    /// Typical width of a clod, in meters.
    #[asset(default = 0.04)]
    pub clod_size: f32,
    /// How lumpy the clods stand, in [0, 1].
    #[asset(default = 0.6)]
    pub clods: f32,
    /// How thickly stones lie in the soil, in [0, 1].
    #[asset(default = 0.25)]
    pub stones: f32,
    /// Typical width of a stone, in meters.
    #[asset(default = 0.035)]
    pub stone_size: f32,
    /// Linear albedo of the stones.
    #[asset(default = [0.17, 0.16, 0.14])]
    pub stone_color: [f32; 3],
    /// How dark and patchy the humus mottling is, in [0, 1].
    #[asset(default = 0.5)]
    pub humus: f32,
    /// How deeply the soil has dried and cracked, in [0, 1]. 0 leaves it
    /// uncracked.
    pub cracks: f32,
    /// Seed of the pattern; two dirts with different seeds never repeat each
    /// other.
    pub seed: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_sand_takes_its_defaults() {
        let sand: SandLook = serde_json::from_str("{}").unwrap();
        assert_eq!(sand, SandLook::default());
        assert_eq!(sand.ripple_spacing, 0.09);
        assert_eq!(sand.roughness, 0.9);
    }

    #[test]
    fn an_empty_dirt_takes_its_defaults() {
        let dirt: DirtLook = serde_json::from_str("{}").unwrap();
        assert_eq!(dirt, DirtLook::default());
        assert_eq!(dirt.cracks, 0.0);
        assert_eq!(dirt.clod_size, 0.04);
    }

    #[test]
    fn a_ground_map_reads_its_authored_name() {
        let map: GroundMap = serde_json::from_str("\"normal\"").unwrap();
        assert_eq!(map, GroundMap::Normal);
        assert_eq!(GroundMap::default(), GroundMap::Albedo);
    }
}
