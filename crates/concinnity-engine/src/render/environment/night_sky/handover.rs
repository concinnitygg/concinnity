//! Which light casts the shadows: the sun by day, the moon by night.
//!
//! Only the first directional light is shadowed. The sun keeps that place
//! while any of its disc is above the horizon. Once it has fully set its
//! light is exactly zero, so the moon can take the place without a visible
//! change; the moon's light then fades in over the next
//! [`MOON_FADE_DEG`] of the sun's descent (twilight, when the sky still far
//! outshines it). Before sunrise the same happens in reverse. Neither light
//! ever jumps, and the shadow direction only switches while both are dark.

use crate::components::DirectionalLight;
use crate::render::light::lights::DirectionalLightSet;

/// Degrees the sun sinks past setting over which the moon's light fades in.
pub(crate) const MOON_FADE_DEG: f32 = 6.0;

/// The share of the moon's light that is on, for a sun `set_depth` radians
/// past fully set (negative while any of it is up): 0 until it has set, 1
/// once it is [`MOON_FADE_DEG`] further down.
pub(crate) fn moon_share(set_depth: f32) -> f32 {
    let t = (set_depth / MOON_FADE_DEG.to_radians()).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The directional set for a frame: `sun` (already lit through the air) in
/// the shadowed place while the moon's share is 0, else `moon` (its full
/// light through the air) scaled by `share` in that place, the set sun left
/// out. `rest` follows either way.
pub(crate) fn shadowed_first(
    sun: DirectionalLight,
    moon: Option<DirectionalLight>,
    share: f32,
    rest: &[DirectionalLight],
) -> DirectionalLightSet {
    let first = match moon {
        Some(moon) if share > 0.0 => DirectionalLight {
            intensity: moon.intensity * share,
            ..moon
        },
        _ => sun,
    };
    DirectionalLightSet::collect(core::iter::once(first).chain(rest.iter().copied()))
}

/// [`shadowed_first`] for a planet whose ground is lit point by point: both
/// bodies, the shadowed one first and the other second, unshadowed, so a
/// point lit by the one the camera has lost still gets it from the daylight
/// curves. The moon is scaled by `share` in either place, so the camera's
/// light from it never steps.
pub(crate) fn shadowed_pair(
    sun: DirectionalLight,
    moon: Option<DirectionalLight>,
    share: f32,
    rest: &[DirectionalLight],
) -> DirectionalLightSet {
    let moon = moon.map(|moon| DirectionalLight {
        intensity: moon.intensity * share,
        ..moon
    });
    let (first, second) = match moon {
        Some(moon) if share > 0.0 => (moon, Some(sun)),
        _ => (sun, moon),
    };
    DirectionalLightSet::collect(
        core::iter::once(first)
            .chain(second)
            .chain(rest.iter().copied()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Atmosphere;
    use crate::render::environment::atmosphere::{AtmosphereFrame, AtmosphereSettings, sun_light};
    use concinnity_core::math::{cos, sin};

    fn light(direction: [f32; 3], color: [f32; 3], intensity: f32) -> DirectionalLight {
        DirectionalLight {
            direction,
            color,
            intensity,
            angular_radius_deg: 0.0,
        }
    }

    // Through a sunset the shadowed light goes from the sun to the moon with
    // no step in the light reaching the scene: at the swap both are dark,
    // and afterwards the moon only brightens.
    #[test]
    fn the_shadowed_light_hands_over_without_a_step() {
        let s = AtmosphereSettings::resolve(&Atmosphere::default(), None);
        let camera = [0.0, s.bottom_radius + 0.002, 0.0];
        let moon = light([-0.3, 0.8, 0.2], [0.9, 0.85, 0.8], 0.001);
        let total = |set: &DirectionalLightSet| {
            set.as_slice()
                .iter()
                .map(|l| l.color.iter().sum::<f32>() * l.intensity)
                .sum::<f32>()
        };
        let mut before = f32::MAX;
        let mut swapped = false;
        for step in 0..400 {
            // The sun from 3 degrees up to 13 below.
            let elevation = (3.0 - step as f32 * 0.04).to_radians();
            let sun = light([cos(elevation), sin(elevation), 0.0], [1.0; 3], 3.0);
            let frame = AtmosphereFrame::new(&s, camera, &sun);
            let share = moon_share(frame.sun_set_depth);
            let set = shadowed_first(sun_light(&frame, &sun), Some(moon), share, &[]);
            let now = total(&set);
            if set.as_slice()[0].direction == moon.direction {
                if swapped {
                    assert!(now >= before, "{step}: {before} -> {now}");
                } else {
                    assert_eq!(frame.sun_transmittance, [0.0; 3], "swap at {elevation}");
                    assert!(now < 1e-4 && before < 0.01, "{before} -> {now}");
                }
                swapped = true;
            } else {
                assert!(!swapped, "the sun took the shadows back while setting");
            }
            before = now;
        }
        assert!(swapped);
        assert!(
            (before - 2.55 * 0.001).abs() < 1e-6,
            "the moon fully on: {before}"
        );
        assert_eq!(moon_share((MOON_FADE_DEG + 1.0).to_radians()), 1.0);
        assert_eq!(moon_share(-0.01), 0.0);
    }

    // A planet lit point by point keeps both bodies: the shadowed one first,
    // the other next, then the rest.
    #[test]
    fn a_planet_keeps_both_bodies() {
        let sun = light([0.0, 1.0, 0.0], [1.0; 3], 3.0);
        let moon = light([0.0, -1.0, 0.0], [0.9; 3], 0.01);
        let fill = light([1.0, 0.0, 0.0], [0.2; 3], 1.0);
        let day = shadowed_pair(sun, Some(moon), 0.0, &[fill]);
        let dark_moon = DirectionalLight {
            intensity: 0.0,
            ..moon
        };
        assert_eq!(day.as_slice(), &[sun, dark_moon, fill]);
        let night = shadowed_pair(sun, Some(moon), 0.5, &[fill]);
        assert_eq!(night.as_slice()[0].direction, moon.direction);
        assert_eq!(night.as_slice()[0].intensity, 0.005);
        assert_eq!(night.as_slice()[1], sun);
        assert_eq!(night.as_slice()[2], fill);
        assert_eq!(shadowed_pair(sun, None, 1.0, &[]).as_slice(), &[sun]);
    }

    // Through a sunset the moon moves from the second place to the first
    // without its light stepping.
    #[test]
    fn a_planet_moon_changes_place_without_a_step() {
        let s = AtmosphereSettings::resolve(&Atmosphere::default(), None);
        let camera = [0.0, s.bottom_radius + 0.002, 0.0];
        let moon = light([-0.3, 0.8, 0.2], [0.9, 0.85, 0.8], 0.001);
        let mut last: Option<f32> = None;
        let mut swapped = false;
        for step in 0..400 {
            let elevation = (3.0 - step as f32 * 0.04).to_radians();
            let sun = light([cos(elevation), sin(elevation), 0.0], [1.0; 3], 3.0);
            let frame = AtmosphereFrame::new(&s, camera, &sun);
            let share = moon_share(frame.sun_set_depth);
            let set = shadowed_pair(sun_light(&frame, &sun), Some(moon), share, &[]);
            let place = set
                .as_slice()
                .iter()
                .position(|l| l.direction == moon.direction)
                .expect("the moon is in the set");
            swapped |= place == 0;
            let now = set.as_slice()[place].intensity;
            if let Some(before) = last {
                assert!(
                    (now - before).abs() < 0.001 * 0.05,
                    "{step}: {before} -> {now}"
                );
            }
            last = Some(now);
        }
        assert!(swapped);
        assert_eq!(last, Some(0.001));
    }

    // Without a moon the sun keeps the place all night, and the other
    // lights always follow the shadowed one.
    #[test]
    fn the_sun_stays_first_without_a_moon() {
        let sun = light([0.0, -1.0, 0.0], [0.0; 3], 3.0);
        let fill = light([0.0, 1.0, 0.0], [0.2; 3], 1.0);
        let set = shadowed_first(sun, None, 1.0, &[fill]);
        assert_eq!(set.as_slice(), &[sun, fill]);
        let moon = light([1.0, 0.0, 0.0], [1.0; 3], 0.5);
        let night = shadowed_first(sun, Some(moon), 0.5, &[fill]);
        assert_eq!(night.as_slice()[0].intensity, 0.25);
        assert_eq!(night.as_slice()[1], fill);
    }
}
