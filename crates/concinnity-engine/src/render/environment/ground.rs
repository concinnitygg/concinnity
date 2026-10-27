//! Ground rules on a planet: where a cover lies or a grass layer grows, by
//! altitude above sea level and slope, packed for the shaders.
//!
//! Each bound of a rule is a ramp over its measure, from 0 outside it to 1
//! inside, eased and multiplied with the others. One planet-fixed edge noise
//! ([`edge_noise`]) moves every bound the same way along its measure, so a
//! boundary wanders along the ground instead of following a contour, and two
//! rules meeting at one bound with one edge width stay complementary: the
//! ramp rising there gains what the one falling there loses. The weight and
//! the noise here are the shaders' `ground_rule_weight` and
//! `ground_rule_noise` line for line.

use crate::components::GroundRule;
use crate::gfx::render_types::{MaterialUniforms, NO_ALBEDO_SLOT, albedo_pool_index};
use crate::render::material_entry::MaterialEntry;
use crate::render::uniforms::ground::{GroundCoverGpu, GroundRuleGpu};
use crate::terrain::GROUND_RULE_OPEN;
#[cfg(test)]
use crate::terrain::{GROUND_RULE_WANDER, steepness};

/// How much shorter than its look grass grows where its rule fades to
/// nothing: its height scales by `1 - (1 - weight) * GROUND_RULE_SHORTENING`.
/// Matches `GROUND_RULE_SHORTENING` in the shaders.
#[cfg(test)]
const GROUND_RULE_SHORTENING: f32 = 0.65;

/// Width of the edge noise's broad and fine patches, in meters. Matches
/// `GROUND_RULE_NOISE_SCALES` in the shaders.
#[cfg(test)]
const NOISE_SCALES: [f32; 2] = [16.0, 4.7];

// Salts of the edge noise's two scales.
#[cfg(test)]
const NOISE_SALTS: [u32; 2] = [0x510e_527f, 0x9b05_688c];

/// A cover's `albedo_index` when its material has no albedo texture: the
/// shaders take its tint alone and sample nothing. Matches
/// `GROUND_COVER_UNTEXTURED` in the shaders.
pub const GROUND_COVER_UNTEXTURED: u32 = u32::MAX;

impl GroundRuleGpu {
    /// The rule that holds everywhere.
    pub const EVERYWHERE: Self = Self {
        altitude: [0.0, GROUND_RULE_OPEN, 0.0, GROUND_RULE_OPEN],
        slope: [0.0, GROUND_RULE_OPEN, 0.0, GROUND_RULE_OPEN],
    };

    /// `rule` as ramps, validated first.
    pub fn of(rule: &GroundRule) -> Self {
        let [altitude, slope] = rule.ramps();
        Self { altitude, slope }
    }

    // Every bound's ramp at `altitude` meters above sea level, on ground of
    // `steepness` (see [`steepness`]), before the edge noise.
    fn ramps(&self, altitude: f32, steep: f32) -> [f32; 4] {
        let [a, b, c, d] = self.altitude;
        let [e, f, g, h] = self.slope;
        [
            altitude * a + b,
            altitude * c + d,
            steep * e + f,
            steep * g + h,
        ]
    }

    // Whether the edge noise can move the rule's weight at `altitude` and
    // `steep`: some ramp lies within the noise's reach of its edge.
    #[cfg(test)]
    fn near_edge(&self, altitude: f32, steep: f32) -> bool {
        self.ramps(altitude, steep)
            .iter()
            .any(|&r| r > -GROUND_RULE_WANDER && r < 1.0 + GROUND_RULE_WANDER)
    }

    /// How fully the rule holds at `altitude` meters above sea level on
    /// ground of steepness `steep` (see [`steepness`](crate::terrain::steepness)),
    /// the edge noise there being `noise`, in [0, 1]. The noise moves each
    /// ramp along its measure's rising direction; an open ramp stays put.
    pub fn weight(&self, altitude: f32, steep: f32, noise: f32) -> f32 {
        let k = [
            self.altitude[0],
            self.altitude[2],
            self.slope[0],
            self.slope[2],
        ];
        self.ramps(altitude, steep)
            .iter()
            .zip(k)
            .map(|(&r, k)| {
                let shift = if k > 0.0 {
                    noise
                } else if k < 0.0 {
                    -noise
                } else {
                    0.0
                };
                let t = (r + shift).clamp(0.0, 1.0);
                t * t * (3.0 - 2.0 * t)
            })
            .product()
    }
}

impl Default for GroundRuleGpu {
    fn default() -> Self {
        Self::EVERYWHERE
    }
}

/// The edge noise at fixed ground point `p` (meters, in the planet's own
/// axes) under a pixel covering `footprint` meters: a ramp shift of at most
/// [`GROUND_RULE_WANDER`] either way, fading to 0 where the pixel cannot hold
/// the patches.
#[cfg(test)]
pub fn edge_noise(p: [f32; 3], footprint: f32) -> f32 {
    use super::grass::litter::faded_patch_noise;
    let broad = faded_patch_noise(p, NOISE_SCALES[0], NOISE_SALTS[0], footprint);
    let fine = faded_patch_noise(p, NOISE_SCALES[1], NOISE_SALTS[1], footprint);
    (broad * 0.65 + fine * 0.35 - 0.5) * (2.0 * GROUND_RULE_WANDER)
}

/// Meters above sea level of a point `v` from a reference point: the
/// reference lies `reference_altitude` above sea level and `reference_radius`
/// from the planet's center, `up_v` is `v` along the up there, and `distance`
/// is the point's own distance from the center. Every term is small but
/// `distance` and `reference_radius`, which only divide, so the result keeps
/// its precision at any planet radius where `distance - radius` would not.
/// The shaders' `ground_altitude`.
pub fn ground_altitude(
    reference_altitude: f32,
    reference_radius: f32,
    up_v: f32,
    v_v: f32,
    distance: f32,
) -> f32 {
    reference_altitude + (2.0 * reference_radius * up_v + v_v) / (reference_radius + distance)
}

impl GroundCoverGpu {
    /// A cover lying where `rule` holds, looking like the material baked to
    /// `entry` (the default material for none) against a texture pool of
    /// `texture_count` entries.
    pub fn of(rule: &GroundRule, entry: Option<MaterialEntry>, texture_count: u32) -> Self {
        let entry = entry.unwrap_or(MaterialEntry::UNTEXTURED);
        let MaterialUniforms {
            tint,
            roughness,
            metallic,
            ..
        } = entry.uniforms;
        Self {
            rule: GroundRuleGpu::of(rule),
            tint,
            albedo_index: if entry.albedo_slot == NO_ALBEDO_SLOT {
                GROUND_COVER_UNTEXTURED
            } else {
                albedo_pool_index(entry.albedo_slot, texture_count)
            },
            roughness,
            metallic,
            _pad: [0.0; 2],
        }
    }
}

/// A planet's covers as the canopy block carries them, in order, at most
/// [`MAX_GROUND_COVERS`](crate::planet::MAX_GROUND_COVERS).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GroundCovers {
    /// The covers; the first `count` are laid.
    pub covers: [GroundCoverGpu; crate::planet::MAX_GROUND_COVERS],
    /// Covers laid.
    pub count: u32,
}

impl GroundCovers {
    /// The covers `planet` lays, each material looked up by `material` and
    /// its albedo resolved against a texture pool of `texture_count` entries.
    pub fn of(
        planet: &crate::components::Planet,
        material: impl Fn(crate::ecs::MaterialHandle) -> Option<MaterialEntry>,
        texture_count: u32,
    ) -> Self {
        let mut out = Self::default();
        for (slot, cover) in out.covers.iter_mut().zip(&planet.covers) {
            *slot = GroundCoverGpu::of(
                &cover.rule,
                cover.material.and_then(&material),
                texture_count,
            );
            out.count += 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{GroundCover, Planet};
    use crate::ecs::MaterialHandle;
    use concinnity_core::math::cos;

    fn band(lo: Option<f32>, hi: Option<f32>, edge: f32) -> GroundRuleGpu {
        GroundRuleGpu::of(&GroundRule {
            altitude: [lo, hi],
            altitude_edge: edge,
            ..GroundRule::default()
        })
    }

    #[test]
    fn an_open_rule_holds_everywhere_at_any_noise() {
        let r = GroundRuleGpu::of(&GroundRule::default());
        assert_eq!(r, GroundRuleGpu::EVERYWHERE);
        for alt in [-500.0, -1.0, 0.0, 3.0, 9_000.0] {
            for noise in [-GROUND_RULE_WANDER, 0.0, GROUND_RULE_WANDER] {
                assert_eq!(r.weight(alt, 0.3, noise), 1.0);
            }
            assert!(!r.near_edge(alt, 0.3));
        }
    }

    // Inside the band the rule holds, outside it does not, and each bound
    // is crossed at half weight over the edge's width.
    #[test]
    fn a_band_holds_between_its_bounds() {
        let r = band(Some(-4.0), Some(2.0), 1.0);
        assert_eq!(r.weight(-1.0, 1.0, 0.0), 1.0);
        assert_eq!(r.weight(-5.0, 1.0, 0.0), 0.0);
        assert_eq!(r.weight(3.0, 1.0, 0.0), 0.0);
        assert!((r.weight(-4.0, 1.0, 0.0) - 0.5).abs() < 1e-6);
        assert!((r.weight(2.0, 1.0, 0.0) - 0.5).abs() < 1e-6);
        assert_eq!(r.weight(2.5, 1.0, 0.0), 0.0);
        assert_eq!(r.weight(1.5, 1.0, 0.0), 1.0);
        // Monotone across the upper edge.
        let mut last = 1.0;
        for i in 0..=20 {
            let w = r.weight(1.5 + i as f32 * 0.05, 1.0, 0.0);
            assert!(w <= last);
            last = w;
        }
    }

    // Bounds past the ends of their measure close the rule.
    #[test]
    fn unreachable_bounds_hold_nowhere() {
        let above_all = band(Some(f32::INFINITY), None, 1.0);
        let cliff_only = GroundRuleGpu::of(&GroundRule {
            slope: [Some(90.0), None],
            ..GroundRule::default()
        });
        let level_only = GroundRuleGpu::of(&GroundRule {
            slope: [None, Some(0.0)],
            ..GroundRule::default()
        });
        for alt in [-1.0e6, 0.0, 9_000.0] {
            for steep in [0.0, 0.5, 1.0] {
                for noise in [-GROUND_RULE_WANDER, GROUND_RULE_WANDER] {
                    assert_eq!(above_all.weight(alt, steep, noise), 0.0);
                    assert_eq!(cliff_only.weight(alt, steep, noise), 0.0);
                    assert_eq!(level_only.weight(alt, steep, noise), 0.0);
                }
            }
        }
        assert!(above_all.altitude.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn an_unset_end_is_open() {
        let below = band(None, Some(0.0), 2.0);
        assert_eq!(below.weight(-10_000.0, 1.0, 0.0), 1.0);
        assert_eq!(below.weight(5.0, 1.0, 0.0), 0.0);
        let above = band(Some(0.0), None, 2.0);
        assert_eq!(above.weight(10_000.0, 1.0, 0.0), 1.0);
        assert_eq!(above.weight(-5.0, 1.0, 0.0), 0.0);
    }

    // The noise moves a boundary by up to its wander times the edge, and a
    // rule ending where another begins (same bound, same edge) splits the
    // ground between them under the one noise both read there.
    #[test]
    fn noise_moves_edges_and_meeting_rules_stay_complementary() {
        let sand = band(None, Some(2.0), 1.0);
        let grass = band(Some(2.0), None, 1.0);
        assert!(sand.weight(2.0, 1.0, -1.0) > 0.99);
        assert!(sand.weight(2.0, 1.0, 1.0) < 0.01);
        assert!(grass.weight(2.0, 1.0, 1.0) > 0.99);
        for i in 0..=40 {
            let alt = i as f32 * 0.1;
            for noise in [-1.5, -0.4, 0.0, 0.7, 1.5] {
                let total = sand.weight(alt, 1.0, noise) + grass.weight(alt, 1.0, noise);
                assert!((total - 1.0).abs() < 1e-5, "{alt} {noise}: {total}");
            }
        }
        // The same along the real noise over a strip of ground, and on a
        // shared slope bound.
        let gentle = GroundRuleGpu::of(&GroundRule {
            slope: [None, Some(25.0)],
            ..GroundRule::default()
        });
        let steep = GroundRuleGpu::of(&GroundRule {
            slope: [Some(25.0), None],
            ..GroundRule::default()
        });
        for i in 0..2000 {
            let p = [0.37 * i as f32, 50_000.0, 11.0 - 0.21 * i as f32];
            let noise = edge_noise(p, 0.0);
            let alt = 1.0 + (i % 40) as f32 * 0.05;
            let total = sand.weight(alt, 1.0, noise) + grass.weight(alt, 1.0, noise);
            assert!((total - 1.0).abs() < 1e-5, "{p:?}: {total}");
            let level = steepness(20.0 + (i % 50) as f32 * 0.2);
            let total = gentle.weight(0.0, level, noise) + steep.weight(0.0, level, noise);
            assert!((total - 1.0).abs() < 1e-4, "{p:?}: {total}");
        }
    }

    // On an Earth-sized planet a point's altitude from a nearby reference
    // holds to a millimeter where the distance from the center less the
    // radius, in single precision, is off by decimeters.
    #[test]
    fn altitude_from_a_reference_keeps_its_precision_at_any_radius() {
        use concinnity_core::math::dvec3;
        let radius = 6_371_000.0f64;
        let center = [0.0, -radius, 0.0];
        let at = |lat: f64, lon: f64, h: f64| -> [f64; 3] {
            let dir = [lat.cos() * lon.sin(), lat.cos() * lon.cos(), lat.sin()];
            dvec3::add(center, dvec3::scale(dir, radius + h))
        };
        // The reference, a tile's origin, 3.25 m up.
        let reference = at(0.0001, 0.0002, 3.25);
        let up = dvec3::normalize_or(dvec3::sub(reference, center), [0.0, 1.0, 0.0]);
        let up32 = dvec3::to_f32(up);
        let mut naive_worst = 0.0f64;
        for i in 0..200 {
            let h = -6.0 + 0.0613 * f64::from(i);
            let p = at(
                0.0001 + 1.0e-5 * f64::from(i % 17),
                0.0002 - 2.0e-5 * f64::from(i % 11),
                h,
            );
            let v = dvec3::to_f32(dvec3::sub(p, reference));
            let rel = dvec3::to_f32(dvec3::sub(p, center));
            let distance = concinnity_core::math::vec3::length(rel);
            let got = ground_altitude(
                3.25,
                concinnity_core::math::narrow_f32(radius + 3.25),
                concinnity_core::math::vec3::dot(up32, v),
                concinnity_core::math::vec3::dot(v, v),
                distance,
            );
            assert!((f64::from(got) - h).abs() < 1e-3, "{h}: {got}");
            naive_worst = naive_worst
                .max((f64::from(distance - concinnity_core::math::narrow_f32(radius)) - h).abs());
        }
        assert!(naive_worst > 0.1, "{naive_worst}");
    }

    #[test]
    fn the_default_record_holds_everywhere() {
        assert_eq!(GroundRuleGpu::default(), GroundRuleGpu::EVERYWHERE);
    }

    #[test]
    fn slopes_band_by_degrees() {
        let gentle = GroundRuleGpu::of(&GroundRule {
            slope: [None, Some(20.0)],
            slope_edge: 4.0,
            ..GroundRule::default()
        });
        assert_eq!(gentle.weight(0.0, steepness(5.0), 0.0), 1.0);
        assert_eq!(gentle.weight(0.0, steepness(35.0), 0.0), 0.0);
        assert!((gentle.weight(0.0, steepness(20.0), 0.0) - 0.5).abs() < 0.02);
        let steep = GroundRuleGpu::of(&GroundRule {
            slope: [Some(20.0), None],
            slope_edge: 4.0,
            ..GroundRule::default()
        });
        assert_eq!(steep.weight(0.0, steepness(35.0), 0.0), 1.0);
        assert_eq!(steep.weight(0.0, steepness(5.0), 0.0), 0.0);
        let total =
            gentle.weight(0.0, steepness(20.5), 0.0) + steep.weight(0.0, steepness(20.5), 0.0);
        assert!((total - 1.0).abs() < 0.02);
        // A bound closer to level ground than half its edge still holds half
        // at the bound.
        let flat = GroundRuleGpu::of(&GroundRule {
            slope: [Some(1.0), None],
            slope_edge: 10.0,
            ..GroundRule::default()
        });
        assert!((flat.weight(0.0, steepness(1.0), 0.0) - 0.5).abs() < 1e-3);
        assert_eq!(flat.weight(0.0, 0.0, 0.0), 0.0);
        // Level ground and cliffs leave the band open.
        let all = GroundRuleGpu::of(&GroundRule {
            slope: [Some(0.0), Some(90.0)],
            ..GroundRule::default()
        });
        assert_eq!(all, GroundRuleGpu::EVERYWHERE);
    }

    #[test]
    fn near_edge_marks_where_the_noise_can_matter() {
        let r = band(Some(0.0), Some(10.0), 1.0);
        assert!(r.near_edge(0.0, 1.0));
        assert!(r.near_edge(10.8, 1.0));
        assert!(!r.near_edge(5.0, 1.0));
        assert!(!r.near_edge(-3.0, 1.0));
        // Away from every edge, no noise moves the weight.
        for noise in [-GROUND_RULE_WANDER, GROUND_RULE_WANDER] {
            assert_eq!(r.weight(5.0, 1.0, noise), 1.0);
            assert_eq!(r.weight(-3.0, 1.0, noise), 0.0);
            assert_eq!(r.weight(13.0, 1.0, noise), 0.0);
        }
    }

    #[test]
    fn edge_noise_is_fixed_bounded_and_fades_with_footprint() {
        let mut sum = 0.0;
        let mut spread = 0.0f32;
        let (mut fine_sum, mut coarse_sum) = (0.0, 0.0);
        let n = 4000;
        for i in 0..n {
            let a = i as f32 * 0.618_034;
            let p = [
                3_000.0 + 0.71 * i as f32 * cos(a),
                50_000.0,
                -800.0 + 0.53 * i as f32,
            ];
            let v = edge_noise(p, 0.0);
            assert_eq!(v, edge_noise(p, 0.0));
            assert!(v.abs() <= GROUND_RULE_WANDER);
            assert_eq!(edge_noise(p, 64.0), 0.0);
            fine_sum += v.abs();
            coarse_sum += edge_noise(p, 20.0).abs();
            sum += v;
            spread = spread.max(v.abs());
        }
        assert!((sum / n as f32).abs() < 0.1, "mean {}", sum / n as f32);
        assert!(spread > 0.6, "spread {spread}");
        assert!(coarse_sum < fine_sum * 0.8, "{coarse_sum} {fine_sum}");
    }

    #[test]
    fn the_constants_are_the_shaders() {
        use crate::render::shader::consts::{float, literal};
        let src = crate::render::shader::embedded::GRASS_FIELD;
        assert_eq!(float(src, "GROUND_RULE_OPEN"), GROUND_RULE_OPEN);
        assert_eq!(float(src, "GROUND_RULE_WANDER"), GROUND_RULE_WANDER);
        assert_eq!(float(src, "GROUND_RULE_SHORTENING"), GROUND_RULE_SHORTENING);
        let scales = literal(src, "GROUND_RULE_NOISE_SCALES");
        assert_eq!(
            scales,
            alloc::format!("float2({:?}, {:?})", NOISE_SCALES[0], NOISE_SCALES[1])
        );
        let types = crate::render::shader::embedded::GROUND_RULE_TYPES;
        assert_eq!(
            literal(types, "GROUND_COVER_UNTEXTURED"),
            alloc::format!("0x{GROUND_COVER_UNTEXTURED:08x}")
        );
        let salts = [
            literal(src, "GROUND_RULE_SALT_BROAD"),
            literal(src, "GROUND_RULE_SALT_FINE"),
        ];
        assert_eq!(
            salts,
            NOISE_SALTS
                .map(|s| alloc::format!("0x{s:08x}"))
                .each_ref()
                .map(alloc::string::String::as_str)
        );
    }

    #[test]
    fn covers_take_their_material_and_stop_at_the_cap() {
        let rule = GroundRule {
            altitude: [None, Some(2.0)],
            ..GroundRule::default()
        };
        let mut sand = MaterialEntry::UNTEXTURED;
        sand.uniforms.tint = [0.7, 0.6, 0.4];
        sand.uniforms.roughness = 0.9;
        sand.albedo_slot = 2;
        let planet = Planet {
            covers: alloc::vec![
                GroundCover {
                    material: Some(MaterialHandle::new(3)),
                    rule,
                };
                crate::planet::MAX_GROUND_COVERS + 1
            ],
            ..Planet::default()
        };
        let covers = GroundCovers::of(&planet, |h| (h.index() == 3).then_some(sand), 5);
        assert_eq!(covers.count as usize, crate::planet::MAX_GROUND_COVERS);
        let c = covers.covers[0];
        assert_eq!(c.tint, [0.7, 0.6, 0.4]);
        assert_eq!(c.roughness, 0.9);
        assert_eq!(c.albedo_index, 2);
        assert_eq!(c.rule, GroundRuleGpu::of(&rule));
        // No material: the default look, sampling no texture.
        let bare = GroundCovers::of(
            &Planet {
                covers: alloc::vec![GroundCover::default()],
                ..Planet::default()
            },
            |_| None,
            5,
        );
        assert_eq!(bare.count, 1);
        assert_eq!(bare.covers[0].albedo_index, GROUND_COVER_UNTEXTURED);
        assert_eq!(bare.covers[0].tint, [1.0; 3]);
    }
}
