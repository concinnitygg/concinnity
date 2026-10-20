// The atmosphere's per-frame inputs, and the lights it drives: the sun, lit
// through the air between it and the camera, and at night the moon, which
// takes over the shadowed first place once the sun has set.

use crate::components::{DirectionalLight, SkyRotation};
use crate::ecs::PipelineContext;
use crate::gfx::render_types::LightUniforms;
use crate::planet::PlanetFrame;
use crate::render::environment::atmosphere::{AtmosphereFrame, AtmosphereSettings, sun_light};
use crate::render::environment::night_sky::{NightFrame, shadowed_first};
use crate::render::light::lights::DirectionalLightSet;
use crate::sky::SkyOrientation;

// How far the sun's transmittance, or the moon's light as a share of its
// full brightness, may drift from what the lights were last pushed with
// before they are pushed again.
const LIGHT_TOLERANCE: f32 = 1e-3;

// What the lights were last pushed for: the sun's transmittance and the
// moon's share of its full light, per color.
type PushedLights = ([f32; 3], [f32; 3]);

// The world's atmosphere and the lights the backend last received.
pub(super) struct AtmosphereDrive {
    settings: AtmosphereSettings,
    pushed: Option<PushedLights>,
}

impl AtmosphereDrive {
    pub(super) fn new(settings: AtmosphereSettings) -> Self {
        Self {
            settings,
            pushed: None,
        }
    }

    // This frame's inputs for a camera at `cam_pos` in the simulated frame,
    // and the directional set to install when the sun's or the moon's light
    // has changed since it was last pushed (or `turned` already carries a new
    // set, which this one then replaces).
    pub(super) fn frame(
        &mut self,
        ctx: &PipelineContext<'_>,
        cam_pos: [f32; 3],
        turned: Option<DirectionalLightSet>,
    ) -> (AtmosphereFrame, Option<DirectionalLightSet>) {
        let sky = ctx
            .resource::<SkyOrientation>()
            .copied()
            .unwrap_or_default();
        let set = authored_lights(ctx);
        let lights = set.as_slice();
        let sun = DirectionalLight {
            direction: sky.rotate(lights[0].direction),
            ..lights[0]
        };
        let camera = cam_pos.map(f64::from);
        let center = ctx
            .resource::<PlanetFrame>()
            .map(|f| f.frame.to_local_f64(f.shape.center))
            .or(self.settings.planet_center);
        let mut frame = AtmosphereFrame::new(
            &self.settings,
            self.settings.camera_km(camera, center),
            &sun,
        );
        if let Some(night) = &self.settings.night {
            // The celestial pole is the sky's axis, or straight up for a sky
            // that does not turn.
            let pole = ctx
                .query::<SkyRotation>()
                .next()
                .map_or([0.0, 1.0, 0.0], |r| r.axis);
            let night = NightFrame::new(night, pole, lights[0].direction, &sky);
            frame = frame.with_night(&self.settings, night);
        }
        let moon = moon_light(&self.settings, &frame, &lights[0]);
        let key = (frame.sun_transmittance, moon.map_or([0.0; 3], |m| m.1));
        let changed = self.pushed.is_none_or(|(sun, moon)| {
            (0..3).any(|i| {
                (sun[i] - key.0[i]).abs() > LIGHT_TOLERANCE
                    || (moon[i] - key.1[i]).abs() > LIGHT_TOLERANCE
            })
        });
        let set = (changed || turned.is_some()).then(|| {
            self.pushed = Some(key);
            let under_sky = super::lights::lights_under_sky(lights.iter(), &sky);
            let turned = under_sky.as_slice();
            let share = frame.night.and_then(|n| n.moon).map_or(0.0, |m| m.share);
            shadowed_first(
                sun_light(&frame, &turned[0]),
                moon.map(|m| m.0),
                share,
                &turned[1..],
            )
        });
        (frame, set)
    }
}

// The moon's full light through the air as a directional light (the sun's
// light above the air scaled by the moon's brightness and phase, tinted by
// the air toward the moon), and that light per color as a share of the full
// moon's above the air, switched by the handover share.
fn moon_light(
    settings: &AtmosphereSettings,
    frame: &AtmosphereFrame,
    sun: &DirectionalLight,
) -> Option<(DirectionalLight, [f32; 3])> {
    let moon = frame.night?.moon?;
    let brightness = settings.night?.moon?.brightness;
    let light = DirectionalLight {
        direction: moon.direction,
        color: core::array::from_fn(|i| sun.color[i] * moon.transmittance[i]),
        intensity: sun.intensity * brightness * moon.phase,
    };
    let seen = moon.transmittance.map(|t| t * moon.phase * moon.share);
    Some((light, seen))
}

// The world's directional lights as authored, or the neutral sun a world with
// none is lit by. Inline, so the per-frame drive allocates nothing.
fn authored_lights(ctx: &PipelineContext<'_>) -> DirectionalLightSet {
    let lights = DirectionalLightSet::collect(ctx.query::<DirectionalLight>().copied());
    if !lights.as_slice().is_empty() {
        return lights;
    }
    let d = &LightUniforms::DEFAULT.directional[0];
    DirectionalLightSet::collect(core::iter::once(DirectionalLight {
        direction: d.direction,
        color: d.color,
        intensity: d.intensity,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Atmosphere, NightSky};
    use crate::ecs::World;

    fn drive() -> AtmosphereDrive {
        AtmosphereDrive::new(AtmosphereSettings::resolve(&Atmosphere::default(), None))
    }

    fn world_with_sun(direction: [f32; 3]) -> World {
        let mut world = World::new();
        world.push(DirectionalLight {
            direction,
            color: [1.0; 3],
            intensity: 2.0,
        });
        world
    }

    // The sun is pushed once, tinted; a still frame pushes nothing more.
    #[test]
    fn the_sun_is_pushed_tinted_and_only_when_it_changes() {
        let mut world = world_with_sun([1.0, 0.05, 0.0]);
        let mut drive = drive();
        let (frame, set) = drive.frame(&world.context(), [0.0, 2.0, 0.0], None);
        let set = set.expect("the first frame installs the sun");
        let sun = set.as_slice()[0];
        assert!(
            sun.color[0] > 2.0 * sun.color[2],
            "a low sun is red: {:?}",
            sun.color
        );
        assert_eq!(sun.intensity, 2.0);
        assert!(frame.camera_height(&drive.settings) > 0.0);
        let (_, again) = drive.frame(&world.context(), [0.0, 2.0, 0.0], None);
        assert!(again.is_none());
        // Climbing through the haze changes the light, so it is pushed again.
        let (_, high) = drive.frame(&world.context(), [0.0, 8_000.0, 0.0], None);
        assert!(high.is_some());
    }

    // The sky's rotation carries the sun below the horizon, which puts it out,
    // and a second light is left as authored.
    #[test]
    fn a_turned_sky_sets_the_sun() {
        let mut world = world_with_sun([0.0, 0.0, 1.0]);
        world.push(DirectionalLight {
            direction: [0.0, 1.0, 0.0],
            color: [0.2, 0.3, 0.4],
            intensity: 1.0,
        });
        world.insert_resource(SkyOrientation::new([1.0, 0.0, 0.0], -60.0));
        let (_, set) = drive().frame(&world.context(), [0.0, 2.0, 0.0], None);
        let set = set.expect("installed");
        assert_eq!(set.as_slice()[0].color, [0.0; 3]);
        assert_eq!(set.as_slice()[1].color, [0.2, 0.3, 0.4]);
    }

    // At night under a full moon the moon holds the shadowed place, lit
    // through the air; by day the sun keeps it and the moon casts nothing.
    #[test]
    fn the_moon_takes_the_shadows_at_night() {
        use crate::render::environment::night_sky::NightSkySettings;
        let settings = AtmosphereSettings::resolve(&Atmosphere::default(), None)
            .with_night(Some(NightSkySettings::resolve(&NightSky::default())));
        let mut night = world_with_sun([0.1, -1.0, 0.0]);
        let mut drive = AtmosphereDrive::new(settings);
        let (frame, set) = drive.frame(&night.context(), [0.0, 2.0, 0.0], None);
        let first = set.expect("installed").as_slice()[0];
        let moon = frame.night.and_then(|n| n.moon).expect("a moon");
        assert_eq!(first.direction, moon.direction);
        assert!(moon.direction[1] > 0.9, "full: opposite the sun");
        let full = 2.0 * NightSky::default().moon_brightness * moon.phase;
        assert!((first.intensity / full - 1.0).abs() < 1e-4);
        assert!(first.color.iter().all(|&c| c > 0.7), "{:?}", first.color);
        let mut day = world_with_sun([0.1, 1.0, 0.0]);
        let (_, set) = AtmosphereDrive::new(settings).frame(&day.context(), [0.0, 2.0, 0.0], None);
        let set = set.expect("installed");
        assert_eq!(set.as_slice().len(), 1);
        assert_eq!(set.as_slice()[0].intensity, 2.0);
    }

    // A world with no directional light is lit by the neutral sun, through the
    // air.
    #[test]
    fn the_neutral_sun_stands_in() {
        let mut world = World::new();
        let (_, set) = drive().frame(&world.context(), [0.0, 2.0, 0.0], None);
        let set = set.expect("installed");
        assert_eq!(set.as_slice().len(), 1);
        let d = LightUniforms::DEFAULT.directional[0].direction;
        assert_eq!(set.as_slice()[0].direction, d);
    }
}
