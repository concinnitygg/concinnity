//! The CPU mirror of the cloud density's shaping (`clouds_common.hlsl`): where
//! a view ray crosses the cloud shell, how a layer's clouds rise inside it,
//! and how the weather and the coverage carve them out of the noise.

use concinnity_core::math::vec3::dot;
use concinnity_core::math::{lerp, smoothstep, sqrt};

/// The spheres a view ray is marched between, as radii from the planet's
/// center in kilometers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudShell {
    /// The ground, which ends every ray that reaches it.
    pub ground: f32,
    /// The bottom of the lowest layer.
    pub lower: f32,
    /// The top of the highest layer.
    pub upper: f32,
}

// Distances along a unit ray from `origin` at which it crosses the sphere of
// `radius` about the center, nearest first; `None` when it misses it.
fn sphere_hits(origin: [f32; 3], dir: [f32; 3], radius: f32) -> Option<[f32; 2]> {
    let b = dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let s = sqrt(disc);
    Some([-b - s, -b + s])
}

/// The stretches of the unit ray from `origin` (kilometers from the planet's
/// center) along `dir` that lie inside `shell` before the ray reaches the
/// ground, as distances along it, nearer first; both `None` when the ray
/// never enters the shell. A ray that dips under the shell's bottom and
/// misses the ground rises back into the shell beyond, its second stretch.
/// `cloud_shell_spans`.
pub fn shell_spans(origin: [f32; 3], dir: [f32; 3], shell: &CloudShell) -> [Option<[f32; 2]>; 2] {
    let span = |start: f32, end: f32| (end > start).then_some([start.max(0.0), end]);
    let r = sqrt(dot(origin, origin));
    let Some(upper) = sphere_hits(origin, dir, shell.upper) else {
        return [None, None];
    };
    let lower = sphere_hits(origin, dir, shell.lower);
    let hits_ground = sphere_hits(origin, dir, shell.ground).is_some_and(|g| g[0] > 0.0);
    if r < shell.lower {
        // Below the clouds: from leaving the bottom sphere to leaving the top.
        if hits_ground {
            return [None, None];
        }
        return [span(lower.map_or(0.0, |l| l[1]), upper[1]), None];
    }
    let start = if r <= shell.upper {
        0.0
    } else if upper[0] < 0.0 {
        return [None, None];
    } else {
        upper[0]
    };
    match lower.filter(|l| l[0] > 0.0) {
        // Down to the bottom, then up out of it to the top past the ground.
        Some(l) => [
            span(start, l[0]),
            if hits_ground {
                None
            } else {
                span(l[1], upper[1])
            },
        ],
        None => [span(start, upper[1]), None],
    }
}

/// How dense a layer's clouds are at height fraction `h` through the layer
/// (0 at its bottom, 1 at its top), for clouds growing `towering` tall where
/// the weather's field reads `weather`: a flat base, rounding off toward a top
/// that the towering and the weather set. `cloud_height_profile`.
pub fn height_profile(h: f32, towering: f32, weather: f32) -> f32 {
    if !(0.0..=1.0).contains(&h) {
        return 0.0;
    }
    let top = (lerp(0.25, 1.0, towering) * (0.55 + 0.45 * weather)).clamp(0.1, 1.0);
    let base = (h / 0.06).clamp(0.0, 1.0);
    base * (1.0 - smoothstep(top * 0.45, top, h))
}

/// The share of a cell that is cloud where the weather reads `weather`, for a
/// layer covering `coverage` of the sky: none at no coverage, all of it at
/// full. `cloud_weather_cover`.
pub fn weather_cover(weather: f32, coverage: f32) -> f32 {
    let threshold = 1.0 - coverage;
    let cover = smoothstep(threshold - 0.2, threshold + 0.2, weather);
    cover * (coverage * 8.0).clamp(0.0, 1.0)
}

// How sharply the regions' smooth field splits into cloudy and clear.
// `CLOUD_REGION_CONTRAST`.
const REGION_CONTRAST: f32 = 3.5;

/// The coverage of a layer covering `coverage` of the sky where its regions'
/// field reads `region` (centered on 0.5), gathered by `clustering`: cloudier
/// regions toward overcast, clearer ones toward clear, as far as the coverage
/// leaves room. `cloud_regional_coverage`.
pub fn regional_coverage(coverage: f32, region: f32, clustering: f32) -> f32 {
    let lean = ((region - 0.5) * REGION_CONTRAST + 0.5).clamp(0.0, 1.0) * 2.0 - 1.0;
    let reach = clustering * coverage.min(1.0 - coverage);
    (coverage + lean * reach).clamp(0.0, 1.0)
}

/// The density, in `[0, 1]`, of base-shape noise `shape` under height profile
/// `profile` in a cell covered `cover`: the noise's peaks become the clouds
/// and the rest clear air. `cloud_coverage_density`.
pub fn coverage_density(shape: f32, profile: f32, cover: f32) -> f32 {
    if cover <= 0.0 {
        return 0.0;
    }
    let v = shape * profile;
    let lo = 1.0 - cover;
    (((v - lo) / (1.0 - lo).max(1e-4)).clamp(0.0, 1.0)) * cover
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::approx::close;

    const SHELL: CloudShell = CloudShell {
        ground: 50.0,
        lower: 50.6,
        upper: 51.8,
    };

    fn at(height: f32) -> [f32; 3] {
        [0.0, 50.0 + height, 0.0]
    }

    #[test]
    fn looking_up_from_the_ground_crosses_the_layer() {
        let [Some([a, b]), None] = shell_spans(at(0.002), [0.0, 1.0, 0.0], &SHELL) else {
            panic!("one stretch");
        };
        assert!(close(a, 0.598, 1e-3));
        assert!(close(b, 1.798, 1e-3));
    }

    #[test]
    fn looking_down_from_the_ground_sees_none() {
        assert_eq!(
            shell_spans(at(0.002), [0.0, -1.0, 0.0], &SHELL),
            [None, None]
        );
    }

    // A level ray from the ground leaves the bottom sphere far off and the
    // top sphere farther still, both short of the horizon of the shell.
    #[test]
    fn a_level_ray_from_the_ground_reaches_the_layer_at_its_horizon() {
        let [Some([a, b]), None] = shell_spans(at(0.002), [1.0, 0.0, 0.0], &SHELL) else {
            panic!("one stretch");
        };
        let reach = |r: f32| sqrt(r * r - 50.002 * 50.002);
        assert!(close(a, reach(50.6), 1e-2));
        assert!(close(b, reach(51.8), 1e-2));
    }

    #[test]
    fn inside_the_layer_the_march_starts_at_the_camera() {
        let [Some([a, b]), None] = shell_spans(at(1.0), [0.0, 1.0, 0.0], &SHELL) else {
            panic!("one stretch");
        };
        assert!(close(a, 0.0, 0.0));
        assert!(close(b, 0.8, 1e-3));
        let [Some([a, b]), None] = shell_spans(at(1.0), [0.0, -1.0, 0.0], &SHELL) else {
            panic!("one stretch");
        };
        assert!(close(a, 0.0, 0.0));
        assert!(close(b, 0.4, 1e-3));
    }

    // From just inside the bottom of the layer, a ray dipping under it and
    // missing the ground rises back into the layer far off, so the stretches
    // it is marched over run on into those of a ray just clearing the bottom.
    #[test]
    fn a_ray_dipping_under_the_layer_rises_back_into_it() {
        let o = at(0.65);
        let r = 50.65_f32;
        // A level ray's closest approach to the center is `r`; one that
        // dips by `angle` passes `r * cos(angle)` from it.
        let past = |closest: f32| {
            let c = closest / r;
            let s = sqrt(1.0 - c * c);
            shell_spans(o, [c, -s, 0.0], &SHELL)
        };
        let [Some([a, b]), Some([c, d])] = past(SHELL.lower - 1e-4) else {
            panic!("two stretches");
        };
        let [Some([e, f]), None] = past(SHELL.lower + 1e-4) else {
            panic!("one stretch");
        };
        let tangent = sqrt(r * r - SHELL.lower * SHELL.lower);
        assert!(close(a, 0.0, 0.0) && close(e, 0.0, 0.0));
        assert!(b < tangent && c > tangent && c - b < 0.25, "{b} {c}");
        assert!(close(d, f, 0.1), "{d} {f}");
    }

    #[test]
    fn a_ray_dipping_into_the_ground_does_not_rise_back() {
        let [Some(_), None] = shell_spans(at(0.65), [1.0, -0.5, 0.0], &SHELL) else {
            panic!("one stretch");
        };
    }

    #[test]
    fn from_orbit_the_march_runs_from_the_top_down_to_the_bottom() {
        let [Some([a, b]), None] = shell_spans(at(150.0), [0.0, -1.0, 0.0], &SHELL) else {
            panic!("one stretch");
        };
        assert!(close(a, 148.2, 1e-2));
        assert!(close(b, 149.4, 1e-2));
        assert_eq!(
            shell_spans(at(150.0), [0.0, 1.0, 0.0], &SHELL),
            [None, None]
        );
    }

    // A ray grazing the limb from orbit passes over the bottom of the layer
    // and out of its top again; one grazing lower passes under the bottom
    // and through the layer again on its way out.
    #[test]
    fn a_limb_ray_from_orbit_crosses_the_layer_once_or_twice() {
        let o = [0.0, 0.0, -200.0];
        let p = |dir: [f32; 3], t: f32| sqrt(dot(o, o) + 2.0 * t * dot(o, dir) + t * t);
        let unit = |dir: [f32; 3]| {
            let n = sqrt(dot(dir, dir));
            dir.map(|v| v / n)
        };
        // Its closest approach to the center is 51.2 km, inside the layer.
        let over = unit([0.0, 0.2648, 1.0]);
        let [Some([a, b]), None] = shell_spans(o, over, &SHELL) else {
            panic!("one stretch");
        };
        assert!(a > 0.0 && b > a);
        assert!(close(p(over, a), SHELL.upper, 1e-2));
        assert!(close(p(over, b), SHELL.upper, 1e-2));
        // Its closest approach is 50.3 km, between the ground and the layer.
        let under = unit([0.0, 0.2600, 1.0]);
        let [Some([a, b]), Some([c, d])] = shell_spans(o, under, &SHELL) else {
            panic!("two stretches");
        };
        assert!(a < b && b < c && c < d);
        for (t, radius) in [
            (a, SHELL.upper),
            (b, SHELL.lower),
            (c, SHELL.lower),
            (d, SHELL.upper),
        ] {
            assert!(close(p(under, t), radius, 1e-2), "{t}");
        }
    }

    #[test]
    fn the_profile_has_a_flat_base_and_a_rounded_top() {
        assert!(close(height_profile(0.0, 0.5, 0.5), 0.0, 1e-6));
        assert!(height_profile(0.1, 0.5, 0.5) > 0.9);
        assert!(close(height_profile(1.0, 0.5, 0.5), 0.0, 1e-6));
        assert!(close(height_profile(-0.1, 0.5, 0.5), 0.0, 0.0));
        // Towers reach higher than heaps, and heaps than sheets.
        let reach = |towering: f32| {
            (0..100)
                .map(|i| i as f32 / 100.0)
                .rfind(|&h| height_profile(h, towering, 0.5) > 0.05)
                .unwrap()
        };
        assert!(reach(1.0) > reach(0.5) && reach(0.5) > reach(0.0));
    }

    #[test]
    fn coverage_runs_from_clear_to_overcast() {
        for w in [0.2, 0.5, 0.8] {
            assert!(close(weather_cover(w, 0.0), 0.0, 1e-6));
            assert!(close(weather_cover(w, 1.0), 1.0, 1e-6));
        }
        assert!(weather_cover(0.7, 0.45) > weather_cover(0.4, 0.45));
    }

    #[test]
    fn no_clustering_keeps_the_coverage_everywhere() {
        for region in [0.0, 0.3, 0.5, 0.9] {
            assert!(close(regional_coverage(0.45, region, 0.0), 0.45, 1e-6));
        }
    }

    // Full clustering clears the clearest regions and overcasts the cloudiest,
    // and the field's middle keeps the layer's own coverage.
    #[test]
    fn clustering_spreads_the_regions_from_clear_to_overcast() {
        assert!(close(regional_coverage(0.5, 0.0, 1.0), 0.0, 1e-6));
        assert!(close(regional_coverage(0.5, 1.0, 1.0), 1.0, 1e-6));
        assert!(close(regional_coverage(0.45, 0.5, 1.0), 0.45, 1e-6));
        assert!(regional_coverage(0.45, 0.6, 0.5) > 0.45);
        assert!(regional_coverage(0.45, 0.4, 0.5) < 0.45);
    }

    #[test]
    fn the_region_contrast_matches_the_shader() {
        let src = crate::render::shader::embedded::CLOUDS_COMMON;
        assert_eq!(
            crate::render::shader::consts::float(src, "CLOUD_REGION_CONTRAST"),
            REGION_CONTRAST
        );
    }

    // A clear or an overcast layer has no room to gather.
    #[test]
    fn clear_and_overcast_layers_stay_as_they_are() {
        for region in [0.0, 1.0] {
            assert!(close(regional_coverage(0.0, region, 1.0), 0.0, 1e-6));
            assert!(close(regional_coverage(1.0, region, 1.0), 1.0, 1e-6));
        }
    }

    #[test]
    fn the_shape_is_carved_by_the_cover() {
        assert!(close(coverage_density(0.9, 1.0, 0.0), 0.0, 0.0));
        assert!(close(coverage_density(0.2, 1.0, 0.5), 0.0, 1e-6));
        let full = coverage_density(0.9, 1.0, 1.0);
        assert!(close(full, 0.9, 1e-5));
        assert!(coverage_density(0.9, 1.0, 0.6) < full);
    }
}
