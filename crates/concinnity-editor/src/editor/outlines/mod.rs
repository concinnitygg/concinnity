// src/editor/outlines/mod.rs
//
// Extent outlines for assets with spatial reach but no rendered geometry:
// trigger volumes, light ranges and cones, camera frusta, probe bounds, and
// prop colliders. This module is the pure policy half: the display categories
// behind the Display menu's toggles and the per-type stroke styling. The
// generators live in `shapes.rs`; the per-frame drive that reads live
// components and publishes lines is `hook/outline_drive.rs`.

pub(crate) mod shapes;

use super::billboards;
use shapes::Stroke;

// Selected entities always outline at full strength; category-toggled
// unselected ones draw thinner and dimmer so an authoring pass reads as a
// layer over the scene rather than competing with it.
const SELECTED_WIDTH_PX: f32 = 2.0;
const TOGGLED_WIDTH_PX: f32 = 1.5;
const TOGGLED_ALPHA: f32 = 0.55;

// Outline colours reuse the billboard icon's hashed hue, re-saturated for the
// HDR scene target (overlay-tuned tints wash out to pastel after tonemapping,
// the same reason the origin axes carry more saturation than the gizmo).
const OUTLINE_SAT: f32 = 0.8;
const OUTLINE_VAL: f32 = 0.9;

// Outlined entities per frame, selection first, bounding the per-frame line
// generation the way MAX_BILLBOARDS bounds the icon pool.
pub(crate) const MAX_OUTLINED: usize = 64;

// One display category behind a Display-menu toggle. Selection always
// outlines the first four; Colliders is toggle-only (selected props already
// carry the highlight rect, and their collider is a separate audit concern).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Category {
    Volumes,
    Lights,
    Cameras,
    Probes,
    Colliders,
}

impl Category {
    // Every category with its Display-menu label, in UI order.
    pub(crate) const LABELED: [(Category, &'static str); 5] = [
        (Category::Volumes, "Volumes"),
        (Category::Lights, "Lights"),
        (Category::Cameras, "Cameras"),
        (Category::Probes, "Probes"),
        (Category::Colliders, "Colliders"),
    ];

    const fn bit(self) -> u32 {
        1 << self as u32
    }
}

// The per-session set of always-on categories (all off by default; selection
// still outlines regardless).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct CategorySet(pub u32);

impl CategorySet {
    pub(crate) const fn contains(self, c: Category) -> bool {
        self.0 & c.bit() != 0
    }

    #[must_use]
    pub(crate) const fn toggled(self, c: Category) -> CategorySet {
        CategorySet(self.0 ^ c.bit())
    }

    pub(crate) const fn any(self) -> bool {
        self.0 != 0
    }
}

// The stroke for a type's outline: the icon's hue, full strength when
// selected, the dimmer layer treatment when shown by a category toggle.
pub(crate) fn stroke(ty: &str, selected: bool) -> Stroke {
    let [r, g, b] = billboards::hsv_to_rgb(billboards::hue_deg(ty), OUTLINE_SAT, OUTLINE_VAL);
    if selected {
        Stroke {
            color: [r, g, b, 1.0],
            width_px: SELECTED_WIDTH_PX,
        }
    } else {
        Stroke {
            color: [r, g, b, TOGGLED_ALPHA],
            width_px: TOGGLED_WIDTH_PX,
        }
    }
}

// A secondary stroke for a shape's inner detail (the spot cone's inner-angle
// circle): the same colour at half strength.
pub(crate) fn inner_stroke(s: Stroke) -> Stroke {
    Stroke {
        color: [s.color[0], s.color[1], s.color[2], s.color[3] * 0.5],
        width_px: TOGGLED_WIDTH_PX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_set_toggles_independent_bits() {
        let set = CategorySet::default();
        assert!(!set.any());
        let set = set.toggled(Category::Lights).toggled(Category::Probes);
        assert!(set.contains(Category::Lights));
        assert!(set.contains(Category::Probes));
        assert!(!set.contains(Category::Volumes));
        assert!(set.any());
        assert!(!set.toggled(Category::Lights).contains(Category::Lights));
    }

    #[test]
    fn labeled_covers_every_category_once() {
        let mut bits = 0u32;
        for (c, label) in Category::LABELED {
            assert!(!label.is_empty());
            assert_eq!(bits & c.bit(), 0, "{label} repeats");
            bits |= c.bit();
        }
        assert_eq!(bits.count_ones() as usize, Category::LABELED.len());
    }

    #[test]
    fn selection_outdraws_the_category_layer() {
        let sel = stroke("TriggerVolume", true);
        let layer = stroke("TriggerVolume", false);
        assert_eq!(sel.color[..3], layer.color[..3], "same hue either way");
        assert!(sel.color[3] > layer.color[3]);
        assert!(sel.width_px > layer.width_px);
        // The outline hue matches the billboard icon's hue for the same type.
        let icon = billboards::tint("TriggerVolume");
        let dominant =
            |c: [f32; 4]| -> usize { (0..3).max_by(|&a, &b| c[a].total_cmp(&c[b])).unwrap() };
        assert_eq!(dominant(sel.color), dominant(icon));
    }

    #[test]
    fn inner_stroke_halves_the_alpha_and_keeps_the_hue() {
        let s = stroke("SpotLight", true);
        let inner = inner_stroke(s);
        assert_eq!(inner.color[..3], s.color[..3]);
        assert!((inner.color[3] - s.color[3] * 0.5).abs() < 1e-6);
    }

    #[test]
    fn light_types_stay_distinct() {
        let hues: Vec<f32> = ["PointLight", "SpotLight", "RectAreaLight"]
            .iter()
            .map(|t| billboards::hue_deg(t))
            .collect();
        assert!(hues[0] != hues[1] && hues[1] != hues[2] && hues[0] != hues[2]);
    }
}
