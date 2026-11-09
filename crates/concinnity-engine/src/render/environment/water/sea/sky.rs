// Which of a planet sea's rings shade under the planet's own sky.

use core::ops::Range;

use concinnity_core::math::acos;
use concinnity_core::math::narrow_f32;

use crate::environment::water::sea::SeaBody;
use crate::gfx::render_types::DAYLIGHT_COS_MARGIN;
use crate::render::uniforms::DaylightBlock;

/// The rings of `sea`'s grid out from its anchor that a planet's own sky,
/// under `daylight`, reaches no point of: all of them where no sky lights the
/// view, none while the camera is high enough for it to reach everywhere. A
/// ring's points lie no farther along the sea from the camera than across the
/// grid, since a face's map only ever stretches the sea, the anchor lies within
/// half a cell of the camera, and a wave moves a vertex at most
/// [`SeaBody::sway`].
pub fn rings_without_sky(sea: &SeaBody, daylight: &DaylightBlock) -> u32 {
    let rings = sea.grid.rings();
    if daylight.center[3] <= 0.5 {
        return rings;
    }
    if daylight.camera_up[3] > 0.0 {
        return 0;
    }
    let reach =
        acos((daylight.reach[0] + DAYLIGHT_COS_MARGIN).min(1.0)) * narrow_f32(sea.sphere.radius);
    let slack = core::f32::consts::FRAC_1_SQRT_2 * sea.grid.inner_cell + sea.sway;
    (0..=rings)
        .rev()
        .find(|&k| {
            core::f32::consts::SQRT_2 * sea.grid.line(i32::try_from(k).unwrap_or(i32::MAX)) + slack
                <= reach
        })
        .unwrap_or(0)
}

/// The index ranges of `sea`'s ring mesh under `daylight`: the inner rings,
/// which shade without the planet's own sky, then the rest, which shade with
/// it.
pub fn sky_split(sea: &SeaBody, daylight: &DaylightBlock) -> [Range<u32>; 2] {
    let inner = sea.grid.ring_indices(rings_without_sky(sea, daylight));
    let all = sea.grid.ring_indices(sea.grid.rings());
    [0..inner, inner..all]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::water::sea::SeaSphere;
    use concinnity_core::math::cos;

    const RADIUS: f64 = 50_000.0;

    // A sea whose waves carry a vertex up to 6 m sideways.
    fn sea() -> SeaBody {
        let mut sea = SeaBody::new(
            SeaSphere {
                center: [0.0, -RADIUS, 0.0],
                radius: RADIUS,
            },
            56.0,
        );
        sea.sway = 6.0;
        sea
    }

    fn sky(camera_reach: f32) -> DaylightBlock {
        DaylightBlock {
            center: [0.0, -50_000.0, 0.0, 1.0],
            camera_up: [0.0, 1.0, 0.0, camera_reach],
            reach: [
                cos(0.5_f32.to_radians()),
                cos(3.0_f32.to_radians()),
                50_000.0,
                5_000.0,
            ],
            ..DaylightBlock::OFF
        }
    }

    // From the ground, the inner rings out to where the sky starts, less the
    // slack, shade without it; the sky reaches every point past them only.
    #[test]
    fn the_rings_the_sky_misses_end_short_of_where_it_starts() {
        let s = sea();
        let k = rings_without_sky(&s, &sky(0.0));
        assert!(k > 0 && k < s.grid.rings(), "{k}");
        // 0.5 deg round a 50 km sea is 436 m.
        let start = 0.5_f32.to_radians() * 50_000.0;
        let corner = core::f32::consts::SQRT_2 * s.grid.line(i32::try_from(k).unwrap());
        let next = core::f32::consts::SQRT_2 * s.grid.line(i32::try_from(k + 1).unwrap());
        assert!(corner < start, "{corner} m reaches the sky at {start} m");
        let slack = core::f32::consts::FRAC_1_SQRT_2 * s.grid.inner_cell + s.sway;
        assert!(
            corner + slack <= start,
            "{corner} m plus {slack} m of slack"
        );
        assert!(next + slack > start, "{next} m left short");
        let [inner, outer] = sky_split(&s, &sky(0.0));
        assert_eq!(inner, 0..s.grid.ring_indices(k));
        assert_eq!(outer.start, inner.end);
        assert_eq!(outer.end, s.grid.ring_indices(s.grid.rings()));
    }

    #[test]
    fn a_high_camera_draws_the_whole_sea_with_its_sky_and_none_without() {
        let s = sea();
        assert_eq!(rings_without_sky(&s, &sky(0.2)), 0);
        assert_eq!(sky_split(&s, &sky(0.2))[0], 0..0);
        assert_eq!(rings_without_sky(&s, &DaylightBlock::OFF), s.grid.rings());
        assert!(sky_split(&s, &DaylightBlock::OFF)[1].is_empty());
    }
}
