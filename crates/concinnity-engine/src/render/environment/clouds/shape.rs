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

/// The stretch of the unit ray from `origin` (kilometers from the planet's
/// center) along `dir` that lies inside `shell` before the ray reaches the
/// ground or the shell's bottom, as distances along it; `None` when the ray
/// never enters the shell. A ray that dips through the space under the shell
/// is marched only until it reaches it. `cloud_shell_span`.
pub fn shell_span(origin: [f32; 3], dir: [f32; 3], shell: &CloudShell) -> Option<[f32; 2]> {
    let r = sqrt(dot(origin, origin));
    let upper = sphere_hits(origin, dir, shell.upper)?;
    let lower = sphere_hits(origin, dir, shell.lower);
    let ground = sphere_hits(origin, dir, shell.ground).filter(|g| g[0] > 0.0);
    let (start, end) = if r < shell.lower {
        // Below the clouds: from leaving the bottom sphere to leaving the top.
        if ground.is_some() {
            return None;
        }
        (lower.map_or(0.0, |l| l[1]), upper[1])
    } else if r <= shell.upper {
        // Inside: to the top, or down to the bottom.
        let down = lower.map(|l| l[0]).filter(|&t| t > 0.0);
        (0.0, down.unwrap_or(upper[1]))
    } else {
        // Above: from entering the top, to the bottom or out of the top again.
        if upper[0] < 0.0 {
            return None;
        }
        let down = lower.map(|l| l[0]).filter(|&t| t > 0.0);
        (upper[0], down.unwrap_or(upper[1]))
    };
    (end > start).then_some([start.max(0.0), end])
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
        let [a, b] = shell_span(at(0.002), [0.0, 1.0, 0.0], &SHELL).unwrap();
        assert!(close(a, 0.598, 1e-3));
        assert!(close(b, 1.798, 1e-3));
    }

    #[test]
    fn looking_down_from_the_ground_sees_none() {
        assert!(shell_span(at(0.002), [0.0, -1.0, 0.0], &SHELL).is_none());
    }

    // A level ray from the ground leaves the bottom sphere far off and the
    // top sphere farther still, both short of the horizon of the shell.
    #[test]
    fn a_level_ray_from_the_ground_reaches_the_layer_at_its_horizon() {
        let [a, b] = shell_span(at(0.002), [1.0, 0.0, 0.0], &SHELL).unwrap();
        let reach = |r: f32| sqrt(r * r - 50.002 * 50.002);
        assert!(close(a, reach(50.6), 1e-2));
        assert!(close(b, reach(51.8), 1e-2));
    }

    #[test]
    fn inside_the_layer_the_march_starts_at_the_camera() {
        let [a, b] = shell_span(at(1.0), [0.0, 1.0, 0.0], &SHELL).unwrap();
        assert!(close(a, 0.0, 0.0));
        assert!(close(b, 0.8, 1e-3));
        let [a, b] = shell_span(at(1.0), [0.0, -1.0, 0.0], &SHELL).unwrap();
        assert!(close(a, 0.0, 0.0));
        assert!(close(b, 0.4, 1e-3));
    }

    #[test]
    fn from_orbit_the_march_runs_from_the_top_down_to_the_bottom() {
        let [a, b] = shell_span(at(150.0), [0.0, -1.0, 0.0], &SHELL).unwrap();
        assert!(close(a, 148.2, 1e-2));
        assert!(close(b, 149.4, 1e-2));
        assert!(shell_span(at(150.0), [0.0, 1.0, 0.0], &SHELL).is_none());
    }

    // A ray grazing the limb from orbit passes over the bottom of the layer
    // and out of its top again.
    #[test]
    fn a_limb_ray_from_orbit_crosses_the_top_twice() {
        let o = [0.0, 0.0, -200.0];
        // Its closest approach to the center is 51.2 km, inside the layer.
        let dir = [0.0, 0.2648, 1.0];
        let n = sqrt(dot(dir, dir));
        let dir = dir.map(|v| v / n);
        let [a, b] = shell_span(o, dir, &SHELL).unwrap();
        assert!(a > 0.0 && b > a);
        let p = |t: f32| sqrt(dot(o, o) + 2.0 * t * dot(o, dir) + t * t);
        assert!(close(p(a), SHELL.upper, 1e-2));
        assert!(close(p(b), SHELL.upper, 1e-2));
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
    fn the_shape_is_carved_by_the_cover() {
        assert!(close(coverage_density(0.9, 1.0, 0.0), 0.0, 0.0));
        assert!(close(coverage_density(0.2, 1.0, 0.5), 0.0, 1e-6));
        let full = coverage_density(0.9, 1.0, 1.0);
        assert!(close(full, 0.9, 1e-5));
        assert!(coverage_density(0.9, 1.0, 0.6) < full);
    }
}
