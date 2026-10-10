//! The far field: past where the blades end, a terrain's ground blends toward
//! the color its grass shows from afar, over the same band the blades fade
//! out across, so the field's edge cannot be seen.
//!
//! Every corner of the terrain's grid carries its far-field color
//! premultiplied by how much grass covers it there. The surface shader recovers
//! the coverage from that color's luminance against the terrain's reference
//! [`FarField::luma`], so bare ground keeps its material's own albedo. With one
//! grass look, or several of one brightness, the recovery is exact.

use alloc::vec::Vec;

use super::{GrassBladeLook, GrassTerrain};
use crate::components::validate;
use crate::render::grass::lod;
use crate::render::grass::tiles::GRASS_DRAW_DISTANCE;

/// Where along a blade, root to tip, the color a distant field shows sits:
/// from afar the tips cover the roots.
pub const FAR_FIELD_TIP_MIX: f32 = 0.6;

/// The share of light a flat surface of a blade's albedo returns that a field
/// of blades does: from afar the blades shade each other and their roots, so
/// the field reads darker than its blades' own color lit in the open.
pub const FAR_FIELD_SHADE: f32 = 0.5;

/// Blades per square meter at which a layer hides the ground entirely from
/// afar; sparser layers cover it in proportion.
pub const FAR_FIELD_FULL_COVER_DENSITY: f32 = 60.0;

/// How the drawn blade colors shift on average: a dry clump's tint, weighted
/// by the mean of the squared clump hash the draws apply it with.
const DRY_TINT: [f32; 3] = [1.35, 1.1, 0.55];
const DRY_MEAN_WEIGHT: f32 = 1.0 / 3.0;

/// Relative luminance of a linear RGB color (Rec. 709 weights), the measure
/// the surface shader recovers coverage with.
pub fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// The albedo a field of `look` shows from afar: its root-to-tip ramp at
/// [`FAR_FIELD_TIP_MIX`], with the dry clumps' shift averaged in (mirroring the
/// draws' `grass_albedo`), darkened by [`FAR_FIELD_SHADE`].
pub fn far_field_color(look: &GrassBladeLook) -> [f32; 3] {
    let t = FAR_FIELD_TIP_MIX;
    let ramp = t * t * (3.0 - 2.0 * t);
    let v = look.color_variation * DRY_MEAN_WEIGHT;
    core::array::from_fn(|i| {
        let c = look.root_color[i] + (look.tip_color[i] - look.root_color[i]) * ramp;
        c * (1.0 + v * (DRY_TINT[i] - 1.0)) * FAR_FIELD_SHADE
    })
}

/// A terrain's far field: the premultiplied color at each grid corner and the
/// reference luminance coverage is measured against.
#[derive(Debug, Clone, PartialEq)]
pub struct FarField {
    /// Per grid corner, row-major like the grid's heights: the far-field color
    /// times the share of ground the grass covers there.
    pub colors: Vec<[f32; 3]>,
    /// The luminance a fully covered corner's color has.
    pub luma: f32,
}

/// The distances the far field blends over: where the blades start to fade
/// out, and where they end.
pub fn far_field_band() -> [f32; 2] {
    [lod::fade_start(GRASS_DRAW_DISTANCE), GRASS_DRAW_DISTANCE]
}

impl FarField {
    /// The far field `terrain`'s visible layers make, or `None` when no layer
    /// grows anything.
    pub fn of(terrain: &GrassTerrain) -> Option<Self> {
        let layers: Vec<_> = terrain
            .layers
            .iter()
            .filter(|(g, _)| g.visible)
            .map(|(g, mask)| {
                let g = validate::grass(g.clone());
                let cover = (g.density / FAR_FIELD_FULL_COVER_DENSITY).min(1.0);
                (
                    far_field_color(&GrassBladeLook::of(&g)),
                    cover,
                    mask.as_ref(),
                )
            })
            .filter(|(_, cover, _)| *cover > 0.0)
            .collect();
        if layers.is_empty() {
            return None;
        }
        let side = terrain.grid.side();
        let n = (side - 1).max(1) as f32;
        let mut colors = Vec::with_capacity(side * side);
        let (mut weighted_luma, mut weight) = (0.0f32, 0.0f32);
        for row in 0..side {
            for col in 0..side {
                let (s, t) = (col as f32 / n, row as f32 / n);
                let mut bare = 1.0f32;
                let mut sum = [0.0f32; 3];
                let mut total = 0.0f32;
                for (color, cover, mask) in &layers {
                    let c = cover * mask.map_or(1.0, |m| m.sample(s, t));
                    bare *= 1.0 - c;
                    total += c;
                    for (s, x) in sum.iter_mut().zip(color) {
                        *s += c * x;
                    }
                    weighted_luma += c * luminance(*color);
                    weight += c;
                }
                let covered = 1.0 - bare;
                colors.push(if total > 0.0 {
                    sum.map(|x| x / total * covered)
                } else {
                    [0.0; 3]
                });
            }
        }
        if weight <= 0.0 {
            return None;
        }
        Some(Self {
            colors,
            luma: weighted_luma / weight,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Grass;
    use crate::terrain::{DensityMask, TerrainGrid};
    use alloc::vec;

    fn terrain(layers: Vec<(Grass, Option<DensityMask>)>) -> GrassTerrain {
        GrassTerrain {
            center: [0.0; 3],
            grid: TerrainGrid::new(4, [10.0, 10.0], vec![0.0; 25]).unwrap(),
            layers,
        }
    }

    fn grass(root: [f32; 3], tip: [f32; 3], density: f32) -> Grass {
        Grass {
            root_color: root,
            tip_color: tip,
            color_variation: 0.0,
            density,
            ..Grass::default()
        }
    }

    // The coverage the shader recovers from a corner's luminance.
    fn recovered(f: &FarField, i: usize) -> f32 {
        (luminance(f.colors[i]) / f.luma).clamp(0.0, 1.0)
    }

    #[test]
    fn the_far_color_sits_toward_the_tip() {
        let look = GrassBladeLook::of(&grass([0.0; 3], [1.0, 0.5, 0.25], 100.0));
        let c = far_field_color(&look);
        let ramp = 0.6 * 0.6 * (3.0 - 1.2) * FAR_FIELD_SHADE;
        assert!((c[0] - ramp).abs() < 1e-6);
        assert!((c[1] - 0.5 * ramp).abs() < 1e-6);
        assert!((c[2] - 0.25 * ramp).abs() < 1e-6);
    }

    #[test]
    fn color_variation_shifts_toward_the_dry_tint_on_average() {
        let mut g = grass([0.2; 3], [0.2; 3], 100.0);
        g.color_variation = 0.6;
        let c = far_field_color(&GrassBladeLook::of(&g));
        let plain = 0.2 * FAR_FIELD_SHADE;
        assert!(c[0] > plain && c[1] > plain && c[2] < plain);
        assert!((c[0] - plain * (1.0 + 0.2 * 0.35)).abs() < 1e-6);
    }

    #[test]
    fn a_dense_unmasked_layer_covers_every_corner_fully() {
        let t = terrain(vec![(
            grass([0.02, 0.05, 0.01], [0.2, 0.3, 0.06], 120.0),
            None,
        )]);
        let f = FarField::of(&t).unwrap();
        assert_eq!(f.colors.len(), 25);
        let full = far_field_color(&GrassBladeLook::of(&t.layers[0].0));
        assert!((f.luma - luminance(full)).abs() < 1e-6);
        for (i, c) in f.colors.iter().enumerate() {
            assert_eq!(*c, full);
            assert!((recovered(&f, i) - 1.0).abs() < 1e-5);
        }
    }

    // A mask's bare texels leave the corner uncovered, so the shader shows the
    // material there; a sparse layer covers in proportion to its density.
    #[test]
    fn masks_and_sparse_layers_cover_in_proportion() {
        let mask = DensityMask::new(2, 1, vec![0, 255]).unwrap();
        let sparse = grass([0.1; 3], [0.1; 3], 0.5 * FAR_FIELD_FULL_COVER_DENSITY);
        let t = terrain(vec![(sparse, Some(mask))]);
        let f = FarField::of(&t).unwrap();
        // Row 0: s = 0 (bare) through s = 1 (full mask).
        assert_eq!(f.colors[0], [0.0; 3]);
        assert!((recovered(&f, 0)).abs() < 1e-6);
        assert!((recovered(&f, 4) - 0.5).abs() < 1e-5);
        assert!((recovered(&f, 2) - 0.25).abs() < 1e-5);
    }

    // Two layers of one brightness overlapping: coverage compounds, and the
    // color is their coverage-weighted mix.
    #[test]
    fn overlapping_layers_compound_their_coverage() {
        let half = 0.5 * FAR_FIELD_FULL_COVER_DENSITY;
        let a = grass([0.1, 0.2, 0.1], [0.1, 0.2, 0.1], half);
        let mut b = a.clone();
        b.root_color = [0.2, 0.1703, 0.1];
        b.tip_color = b.root_color;
        let ca = far_field_color(&GrassBladeLook::of(&a));
        let cb = far_field_color(&GrassBladeLook::of(&b));
        assert!((luminance(ca) - luminance(cb)).abs() < 1e-3);
        let f = FarField::of(&terrain(vec![(a, None), (b, None)])).unwrap();
        assert!((recovered(&f, 12) - 0.75).abs() < 1e-3);
        let mix: [f32; 3] = core::array::from_fn(|k| 0.75 * 0.5 * (ca[k] + cb[k]));
        for (got, want) in f.colors[12].iter().zip(mix) {
            assert!((got - want).abs() < 1e-6);
        }
    }

    #[test]
    fn a_terrain_growing_nothing_has_no_far_field() {
        assert_eq!(FarField::of(&terrain(vec![])), None);
        let hidden = Grass {
            visible: false,
            ..Grass::default()
        };
        let none = grass([0.1; 3], [0.1; 3], 0.0);
        assert_eq!(
            FarField::of(&terrain(vec![(hidden, None), (none, None)])),
            None
        );
        let bare = DensityMask::new(1, 1, vec![0]).unwrap();
        let masked_out = grass([0.1; 3], [0.1; 3], 100.0);
        assert_eq!(FarField::of(&terrain(vec![(masked_out, Some(bare))])), None);
    }

    // The tint takes over exactly as the blades fade out.
    #[test]
    fn the_band_is_the_blade_fade() {
        let [start, end] = far_field_band();
        assert_eq!(lod::fade(start, GRASS_DRAW_DISTANCE), 1.0);
        assert_eq!(lod::fade(end, GRASS_DRAW_DISTANCE), 0.0);
        assert!(start < end);
    }
}
