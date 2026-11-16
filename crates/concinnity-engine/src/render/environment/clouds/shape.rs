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

/// The stretch of the line from `q` (kilometers from the planet's center)
/// along the unit `light` inside `shell`'s layers, toward the light, as
/// distances along it: the one holding `q` or the first past it, short of the
/// bottom sphere; `None` when the line toward the light meets no layer.
/// `cloud_light_span`.
pub fn light_span(q: [f32; 3], light: [f32; 3], shell: &CloudShell) -> Option<[f32; 2]> {
    let upper = sphere_hits(q, light, shell.upper).filter(|u| u[1] > 0.0)?;
    let lower = sphere_hits(q, light, shell.lower);
    let [start, end] = match lower {
        Some(l) if l[0] > 0.0 => [upper[0], l[0]],
        Some(l) => [upper[0].max(l[1]), upper[1]],
        None => upper,
    };
    let start = start.max(0.0);
    (end > start).then_some([start, end])
}

/// The stretch of the whole line through `q` (kilometers from the planet's
/// center) along the unit `light` where it climbs through `shell` toward the
/// light, as distances along it from `q`, behind `q` too; `None` when the
/// line misses the shell. `cloud_layer_chord`.
pub fn layer_chord(q: [f32; 3], light: [f32; 3], shell: &CloudShell) -> Option<[f32; 2]> {
    let top = sphere_hits(q, light, shell.upper)?;
    let [start, end] = match sphere_hits(q, light, shell.lower) {
        Some(bottom) => [bottom[1].max(top[0]), top[1]],
        None => top,
    };
    (end > start).then_some([start, end])
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
/// full. `soft` widens the edge between cloud and clear, so a cell too small
/// for a pixel reads as partial cover rather than all or nothing.
/// `cloud_weather_cover`.
pub fn weather_cover(weather: f32, coverage: f32, soft: f32) -> f32 {
    let threshold = 1.0 - coverage;
    let band = 0.2 + soft;
    let cover = smoothstep(threshold - band, threshold + band, weather);
    cover * (coverage * 8.0).clamp(0.0, 1.0)
}

// How far the cover's edge widens once a pixel spans a whole cell, and the
// pixel's share of a cell at which it starts and stops widening.
// `CLOUD_SOFT_COVER`, `CLOUD_SOFT_COVER_START` and `CLOUD_SOFT_COVER_END`.
#[cfg(test)]
const SOFT_COVER: f32 = 0.5;
#[cfg(test)]
const SOFT_COVER_START: f32 = 0.25;
#[cfg(test)]
const SOFT_COVER_END: f32 = 1.0;

/// How far the cover's edge widens (the `soft` of [`weather_cover`]) for a
/// pixel `footprint_km` across over cells `cell_km` apart: not at all while
/// a cell spans four pixels, fully once one spans a pixel.
/// `cloud_soft_cover`.
#[cfg(test)]
fn soft_cover(footprint_km: f32, cell_km: f32) -> f32 {
    SOFT_COVER * smoothstep(SOFT_COVER_START, SOFT_COVER_END, footprint_km / cell_km)
}

/// The coverage of a layer covering `coverage` of the sky on average where
/// the weather leans `lean` (-1 clear to 1 overcast): toward overcast or
/// clear as far as the lean goes. `cloud_lean_coverage`.
#[cfg(test)]
pub(super) fn lean_coverage(coverage: f32, lean: f32) -> f32 {
    let room = if lean > 0.0 { 1.0 - coverage } else { coverage };
    (coverage + lean * room).clamp(0.0, 1.0)
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
    fn the_light_from_the_ground_crosses_the_whole_layer() {
        let [a, b] = light_span(at(0.0), [0.0, 1.0, 0.0], &SHELL).expect("a stretch");
        assert!(close(a, 0.6, 1e-3));
        assert!(close(b, 1.8, 1e-3));
    }

    // A point among the layer whose line toward a low light dips into the
    // bottom sphere ahead is lit through the stretch before it, not past it.
    #[test]
    fn a_low_light_from_inside_the_layer_stops_at_the_bottom() {
        let r = 51.0_f32;
        let c = (SHELL.lower - 0.1) / r;
        let light = [c, -sqrt(1.0 - c * c), 0.0];
        let [a, b] = light_span(at(1.0), light, &SHELL).expect("a stretch");
        let below = sphere_hits(at(1.0), light, SHELL.lower).expect("dips under");
        assert!(close(a, 0.0, 0.0));
        assert!(close(b, below[0], 1e-4));
    }

    #[test]
    fn above_the_layers_the_light_meets_none() {
        assert_eq!(light_span(at(2.0), [0.0, 1.0, 0.0], &SHELL), None);
    }

    // The chord through a point in the layer keeps the layer behind it, and
    // through a point under or over it the stretch the line climbs through.
    #[test]
    fn the_chord_is_where_the_line_climbs_through_the_layer() {
        let up = [0.0, 1.0, 0.0];
        let [a, b] = layer_chord(at(1.2), up, &SHELL).expect("a chord");
        assert!(close([a, b], [-0.6, 0.6], 1e-3));
        let [a, b] = layer_chord(at(0.0), up, &SHELL).expect("a chord");
        assert!(close([a, b], [0.6, 1.8], 1e-3));
        let [a, b] = layer_chord(at(2.0), up, &SHELL).expect("a chord");
        assert!(close([a, b], [-1.4, -0.2], 1e-3));
    }

    // A low light's line dips under the layer and climbs out again: the chord
    // is the stretch climbing toward the light, not the one coming down.
    #[test]
    fn a_low_light_chord_is_the_climb_toward_the_light() {
        let light = [1.0, 0.0, 0.0];
        let [a, b] = layer_chord(at(0.0), light, &SHELL).expect("a chord");
        let top = sphere_hits(at(0.0), light, SHELL.upper).expect("crosses the top");
        let bottom = sphere_hits(at(0.0), light, SHELL.lower).expect("crosses the bottom");
        assert!(close([a, b], [bottom[1], top[1]], 1e-4));
        assert!(a > 0.0);
    }

    // A line passing over the bottom sphere crosses the layer in one stretch.
    #[test]
    fn a_grazing_chord_crosses_the_layer_once() {
        let middle = at(1.2);
        let light = [1.0, 0.0, 0.0];
        let top = sphere_hits(middle, light, SHELL.upper).expect("crosses the top");
        assert_eq!(layer_chord(middle, light, &SHELL), Some(top));
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
            assert!(close(weather_cover(w, 0.0, 0.0), 0.0, 1e-6));
            assert!(close(weather_cover(w, 1.0, 0.0), 1.0, 1e-6));
        }
        assert!(weather_cover(0.7, 0.45, 0.0) > weather_cover(0.4, 0.45, 0.0));
    }

    // A soft edge lifts the clear side and dims the cloudy one toward the
    // coverage, as cells too small to see blend into partial cover.
    #[test]
    fn a_soft_edge_blends_cells_into_partial_cover() {
        let (clear, cloud) = (0.35, 0.75);
        assert!(weather_cover(clear, 0.45, 0.6) > weather_cover(clear, 0.45, 0.0));
        assert!(weather_cover(cloud, 0.45, 0.6) < weather_cover(cloud, 0.45, 0.0));
    }

    #[test]
    fn the_soft_edge_follows_the_pixel_and_matches_the_shader() {
        assert_eq!(soft_cover(0.1, 1.0), 0.0);
        assert!(close(soft_cover(1.0, 1.0), SOFT_COVER, 1e-6));
        assert!(soft_cover(0.5, 1.0) > 0.0 && soft_cover(0.5, 1.0) < SOFT_COVER);
        let src = crate::render::shader::embedded::CLOUDS_COMMON;
        let float = |name| crate::render::shader::consts::float(src, name);
        assert_eq!(float("CLOUD_SOFT_COVER"), SOFT_COVER);
        assert_eq!(float("CLOUD_SOFT_COVER_START"), SOFT_COVER_START);
        assert_eq!(float("CLOUD_SOFT_COVER_END"), SOFT_COVER_END);
    }

    // No lean keeps the layer's own coverage; a full lean clears or
    // overcasts it.
    #[test]
    fn the_lean_moves_coverage_toward_clear_or_overcast() {
        for c in [0.0, 0.3, 0.45, 1.0] {
            assert!(close(lean_coverage(c, 0.0), c, 1e-6));
        }
        assert!(close(lean_coverage(0.45, 1.0), 1.0, 1e-6));
        assert!(close(lean_coverage(0.45, -1.0), 0.0, 1e-6));
        assert!(close(lean_coverage(0.4, 0.5), 0.7, 1e-6));
        assert!(close(lean_coverage(0.4, -0.5), 0.2, 1e-6));
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
