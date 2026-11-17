//! The CPU mirror of how the clouds scatter light (`clouds_common.hlsl`):
//! once, thrown forward, and on through the cloud as the two-stream diffuse
//! light that brightens a thick cloud. The tests send light into a level
//! slab and integrate what leaves it, up and down, in every direction,
//! against the two-stream approximation, and solve the diffuse light's table
//! the shader reads.

use concinnity_core::math::{SaturatingFrom, exp, exp2, log2, narrow_f32, powf};

// How strongly the first scattering throws light forward. `CLOUD_FORWARD_G`.
const FORWARD_G: f32 = 0.6;
// The two-stream approximation's asymmetry for cloud droplets, and its factor
// on the share not scattered forward. `CLOUD_TWO_STREAM_G` and
// `CLOUD_TWO_STREAM_SCALE`.
const TWO_STREAM_G: f32 = 0.85;
const TWO_STREAM_SCALE: f32 = 0.75;
// The diffuse light's table: entries, and the doublings of the column's
// optical depth it starts at and steps by. `CLOUD_DIFFUSE_COUNT`,
// `CLOUD_DIFFUSE_LOW` and `CLOUD_DIFFUSE_STEPS`.
const DIFFUSE_COUNT: usize = 41;
const DIFFUSE_LOW: f32 = -4.0;
const DIFFUSE_STEPS: f32 = 4.0;
// The column optical depths over which the table turns from the profile lit
// from its bottom to the straight line.
const DIFFUSE_LINE_FROM: f32 = 3.0;
const DIFFUSE_LINE_TO: f32 = 5.0;
// The share of the light above a deep cloud its first scattering sends back
// up out of it. `CLOUD_FIRST_BACK`.
const FIRST_BACK: f32 = 0.023_994;
// How much the thin lit skin's powder effect darkens it. `CLOUD_POWDER`.
const POWDER: f32 = 0.6;
// The least sine of the light's height the cloud behind a point is reckoned
// down to its layer's bottom at. `CLOUD_BEHIND_MIN_SINE`.
const BEHIND_MIN_SINE: f32 = 0.2;

const PI: f32 = core::f32::consts::PI;

/// The Henyey-Greenstein phase at cosine `nu` with asymmetry `g`.
/// `cloud_hg`.
fn hg(nu: f32, g: f32) -> f32 {
    let g2 = g * g;
    (1.0 - g2) / (4.0 * PI * powf((1.0 + g2 - 2.0 * g * nu).max(1e-4), 1.5))
}

fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How fast a cloud's optical depth turns light back. `cloud_two_stream_rate`.
fn two_stream_rate() -> f32 {
    TWO_STREAM_SCALE * (1.0 - TWO_STREAM_G)
}

/// The share of a light through `optical_depth` of cloud, straight or
/// scattered. `cloud_sunlit_share`.
fn sunlit_share(optical_depth: f32) -> f32 {
    1.0 / (1.0 + two_stream_rate() * optical_depth)
}

/// The share of a light that keeps on straight in its beam through
/// `optical_depth` of cloud, which the ground under it takes as sunlight.
/// `cloud_beam_share`.
fn beam_share(optical_depth: f32) -> f32 {
    exp(-optical_depth)
}

/// One layer's Beer shadow map texel, `[front, back, optical depth]` along
/// the light: the optical depth from the cloud's start toward the light down
/// to `along`. `cloud_beer_depth`.
fn beer_depth(layer: [f32; 3], along: f32) -> f32 {
    let [front, back, depth] = layer;
    let span = (front - back).max(1e-4);
    depth / span * (front - along).clamp(0.0, span)
}

/// The optical depth of the cloud behind the point at `along`, down to its
/// layer's bottom `height_km` below it, the light at sine `sine` over the
/// horizon. `cloud_beer_behind`.
fn beer_behind(layer: [f32; 3], along: f32, height_km: f32, sine: f32) -> f32 {
    let reach_km = height_km.max(0.0) / sine.max(BEHIND_MIN_SINE);
    beer_depth(layer, along - reach_km) - beer_depth(layer, along)
}

/// The two-stream reflectance of a level cloud `optical_depth` deep.
fn two_stream_reflectance(optical_depth: f32) -> f32 {
    1.0 - sunlit_share(optical_depth)
}

/// The light per unit extinction a point `depth` deep toward a light sends
/// along a ray at cosine `nu` to the light, per unit of that light, by its
/// first scattering under the powder effect. `cloud_sample_light`'s first
/// term.
fn first_light(nu: f32, depth: f32) -> f32 {
    let toward = (nu * 0.5 + 0.5).clamp(0.0, 1.0);
    let skin = 1.0 - POWDER * exp(-2.0 * depth);
    let powder = skin + (1.0 - skin) * toward;
    hg(nu, FORWARD_G) * exp(-depth) * powder
}

/// The column optical depth the diffuse table's entry `i` is solved at.
fn table_depth(i: usize) -> f32 {
    exp2(i as f32 / DIFFUSE_STEPS + DIFFUSE_LOW)
}

/// The profile `u + v (T - d) + w e^-(T - d)` at `below = T - d`.
fn profile([u, v, w]: [f32; 3], below: f32) -> f32 {
    (u + v * below + w * exp(-below)).max(0.0)
}

/// The diffuse light a point `depth` deep in a column `column` deep scatters
/// per unit extinction, per unit of the light above, from the shader's
/// `table`. `cloud_diffuse_share`.
fn diffuse_share(table: &[[f32; 3]], column: f32, depth: f32) -> f32 {
    let t = column.max(0.0);
    let below = t - depth.clamp(0.0, t);
    let top = table_depth(DIFFUSE_COUNT - 1);
    if t > top {
        let through = sunlit_share(t);
        let back = 1.0 - through - FIRST_BACK;
        let v = (back - through) / (PI * (t - 4.0 / 3.0));
        return profile([through / PI - 2.0 / 3.0 * v, v, 0.0], below);
    }
    let last = (DIFFUSE_COUNT - 1) as f32;
    let x = ((log2(t.max(1e-6)) - DIFFUSE_LOW) * DIFFUSE_STEPS).clamp(0.0, last);
    let i = usize::saturating_from(x).min(DIFFUSE_COUNT - 2);
    let f = x - i as f32;
    let c: [f32; 3] = core::array::from_fn(|j| table[i][j] + (table[i + 1][j] - table[i][j]) * f);
    profile(c, below)
}

/// What a level cloud `optical_depth` deep, lit from straight above, sends
/// up out of its top and down out of its bottom over every direction, per
/// unit of the light above, from a source `light(nu, depth)` per unit
/// extinction: `[up, down]`. The light leaving along a ray dims with the
/// cloud it crosses on the way out.
fn fluxes(optical_depth: f32, light: impl Fn(f32, f32) -> f32) -> [f64; 2] {
    const DIRECTIONS: usize = 128;
    const STEPS: usize = 2000;
    let t = f64::from(optical_depth);
    let dd = t / STEPS as f64;
    let mut out = [0.0f64; 2];
    for j in 0..DIRECTIONS {
        let mu = (j as f64 + 0.5) / DIRECTIONS as f64;
        let mut seen = [0.0f64; 2];
        for i in 0..STEPS {
            let d = (i as f64 + 0.5) * dd;
            let (nu, depth) = (narrow_f32(mu), narrow_f32(d));
            seen[0] += f64::from(light(-nu, depth)) * (-d / mu).exp();
            seen[1] += f64::from(light(nu, depth)) * (-(t - d) / mu).exp();
        }
        for (o, s) in out.iter_mut().zip(seen) {
            *o += 2.0 * core::f64::consts::PI * s * dd / DIRECTIONS as f64;
        }
    }
    out
}

/// The diffuse profile's terms for a column `optical_depth` deep: with the
/// first scattering the column sends up and down the two-stream shares,
/// through a profile lit from its bottom up to `DIFFUSE_LINE_TO` and a
/// straight line from `DIFFUSE_LINE_FROM`. Where the bottom profile cannot
/// keep both shares it keeps their sum.
fn solve_profile(optical_depth: f32) -> [f32; 3] {
    let t = optical_depth;
    let [first_up, first_down] = fluxes(t, first_light);
    let k = f64::from(two_stream_rate());
    let td = f64::from(t);
    let up = k * td / (1.0 + k * td) - first_up;
    let down = 1.0 / (1.0 + k * td) - (-td).exp() - first_down;
    let [u1, d1] = fluxes(t, |_, _| 1.0);
    let [u2, d2] = fluxes(t, |_, d| t - d);
    let [u3, d3] = fluxes(t, |_, d| exp(-(t - d)));
    let pair = |ua: f64, da: f64, ub: f64, db: f64| {
        let det = ua * db - ub * da;
        [(up * db - ub * down) / det, (ua * down - da * up) / det]
    };
    let [a, b] = pair(u1, d1, u2, d2);
    let line = [narrow_f32(a), narrow_f32(b), 0.0];
    let [c, w] = pair(u1, d1, u3, d3);
    let bottom = if c < 0.0 {
        [0.0, 0.0, narrow_f32((up + down) / (u3 + d3))]
    } else {
        [narrow_f32(c), 0.0, narrow_f32(w)]
    };
    if t < DIFFUSE_LINE_FROM {
        return bottom;
    }
    let s = smoothstep(DIFFUSE_LINE_FROM, DIFFUSE_LINE_TO, t);
    core::array::from_fn(|j| bottom[j] + (line[j] - bottom[j]) * s)
}

/// The shader's `CLOUD_DIFFUSE_TABLE`.
fn shader_table() -> alloc::vec::Vec<[f32; 3]> {
    let src = crate::render::shader::embedded::CLOUDS_COMMON;
    let start = src.find("CLOUD_DIFFUSE_TABLE[").expect("the table");
    let rest = &src[start..];
    let body = &rest[rest.find('{').expect("its entries") + 1..rest.find("};").expect("its end")];
    body.split("float3(")
        .skip(1)
        .map(|entry| {
            let inner = &entry[..entry.find(')').expect("an entry")];
            let mut terms = inner
                .split(',')
                .map(|v| v.trim().parse::<f32>().expect("a term"));
            core::array::from_fn(|_| terms.next().expect("three terms"))
        })
        .collect()
}

/// What leaves a level cloud `optical_depth` deep in every direction, by the
/// shader's light: `[up, down]`, the straight beam counted down.
fn slab_fluxes(table: &[[f32; 3]], optical_depth: f32) -> [f64; 2] {
    let t = optical_depth;
    let [up, down] = fluxes(t, |nu, d| first_light(nu, d) + diffuse_share(table, t, d));
    [up, down + f64::from(beam_share(t))]
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::approx::close;

    fn sweep() -> impl Iterator<Item = f32> {
        (0..=50).map(|i| exp2(i as f32 * 0.25 - 4.3))
    }

    // The shader's table is the profile solved at each of its depths.
    #[test]
    fn the_diffuse_table_is_the_flux_solve() {
        let table = shader_table();
        assert_eq!(table.len(), DIFFUSE_COUNT);
        for (i, entry) in table.iter().enumerate() {
            let solved = solve_profile(table_depth(i));
            let scale = solved.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
            for j in 0..3 {
                assert!(
                    (entry[j] - solved[j]).abs() <= 0.02 * scale,
                    "{i}: {entry:?} against {solved:?}"
                );
            }
        }
    }

    // A level cloud of any depth sends out, up, down and straight through,
    // nearly all the light it is given and never more.
    #[test]
    fn the_light_keeps_the_light_it_is_given() {
        let table = shader_table();
        for t in sweep() {
            let [up, down] = slab_fluxes(&table, t);
            let total = up + down;
            assert!((0.95..=1.01).contains(&total), "{t}: {total}");
        }
    }

    // From an optical depth of three a level cloud reflects and lets
    // through what the two-stream approximation does.
    #[test]
    fn a_cloud_reflects_and_lets_through_the_two_stream_shares() {
        let table = shader_table();
        for t in sweep().filter(|t| *t >= DIFFUSE_LINE_FROM) {
            let [up, down] = slab_fluxes(&table, t);
            let reflected = f64::from(two_stream_reflectance(t));
            let through = f64::from(sunlit_share(t));
            assert!((up / reflected - 1.0).abs() < 0.02, "{t}: {up} {reflected}");
            assert!((down / through - 1.0).abs() < 0.02, "{t}: {down} {through}");
        }
    }

    // The ground under a cloud takes its beam as sunlight and the rest
    // through the sky, which never pass what the two-stream approximation
    // lets through.
    #[test]
    fn the_ground_takes_no_more_than_the_cloud_lets_through() {
        let table = shader_table();
        for t in sweep() {
            let [_, down] = slab_fluxes(&table, t);
            assert!(down <= f64::from(sunlit_share(t)) + 5e-3, "{t}: {down}");
        }
    }

    // A deep cloud's first scattering sends back up the share the profile
    // past the table leaves out.
    #[test]
    fn the_first_back_share_is_a_deep_clouds() {
        let [up, _] = fluxes(table_depth(DIFFUSE_COUNT - 1), first_light);
        assert!(close(narrow_f32(up), FIRST_BACK, 1e-4), "{up}");
    }

    // Under a high sun a deep deck's whole column stands behind and before
    // a point in it, so the deck reflects as deep as it is.
    #[test]
    fn a_deep_deck_keeps_its_whole_column() {
        let deck = [0.6, -0.6, 50.0];
        let along = 0.4;
        let column = beer_depth(deck, along) + beer_behind(deck, along, 1.0, 1.0);
        assert!(close(column, 50.0, 1e-3), "{column}");
    }

    // Under a low sun the line through a point runs far through its layer,
    // over clouds well behind it: only the stretch down to the layer's bottom
    // counts.
    #[test]
    fn a_low_sun_counts_the_cloud_behind_down_to_the_bottom() {
        let line = [5.0, -30.0, 35.0];
        let behind = beer_behind(line, 0.0, 1.0, 0.05);
        assert!(close(behind, 5.0, 1e-3), "{behind}");
        assert!(close(beer_behind(line, 0.0, 0.0, 0.05), 0.0, 0.0));
    }

    #[test]
    fn the_constants_match_the_shader() {
        let src = crate::render::shader::embedded::CLOUDS_COMMON;
        let float = |name| crate::render::shader::consts::float(src, name);
        let uint = |name| crate::render::shader::consts::uint(src, name);
        assert_eq!(float("CLOUD_FORWARD_G"), FORWARD_G);
        assert_eq!(float("CLOUD_TWO_STREAM_G"), TWO_STREAM_G);
        assert_eq!(float("CLOUD_TWO_STREAM_SCALE"), TWO_STREAM_SCALE);
        assert_eq!(uint("CLOUD_DIFFUSE_COUNT"), DIFFUSE_COUNT);
        assert_eq!(float("CLOUD_DIFFUSE_LOW"), DIFFUSE_LOW);
        assert_eq!(float("CLOUD_DIFFUSE_STEPS"), DIFFUSE_STEPS);
        assert_eq!(float("CLOUD_FIRST_BACK"), FIRST_BACK);
        assert_eq!(float("CLOUD_POWDER"), POWDER);
        assert_eq!(float("CLOUD_BEHIND_MIN_SINE"), BEHIND_MIN_SINE);
    }
}
