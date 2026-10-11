// The atmosphere's per-frame inputs, and the sun light it drives: the first
// directional light, lit through the air between the sun and the camera.

use concinnity_core::components::DirectionalLight;
use concinnity_core::ecs::PipelineContext;
use concinnity_core::gfx::render_types::LightUniforms;
use concinnity_core::planet::PlanetFrame;
use concinnity_core::render::atmosphere::{AtmosphereFrame, AtmosphereSettings, sun_light};
use concinnity_core::render::lights::DirectionalLightSet;
use concinnity_core::sky::SkyOrientation;

// How far the sun's transmittance may drift from what the lights were last
// pushed with before they are pushed again.
const SUN_TOLERANCE: f32 = 1e-3;

// The world's atmosphere and the sun color the backend last received.
pub(super) struct AtmosphereDrive {
    settings: AtmosphereSettings,
    pushed: Option<[f32; 3]>,
}

impl AtmosphereDrive {
    pub(super) fn new(settings: AtmosphereSettings) -> Self {
        Self {
            settings,
            pushed: None,
        }
    }

    // This frame's inputs for a camera at `cam_pos` in the simulated frame,
    // and the directional set to install when the sun's light has changed
    // since it was last pushed (or `turned` already carries a new set, which
    // this one then replaces).
    pub(super) fn frame(
        &mut self,
        ctx: &PipelineContext,
        cam_pos: [f32; 3],
        turned: Option<DirectionalLightSet>,
    ) -> (AtmosphereFrame, Option<DirectionalLightSet>) {
        let sky = ctx
            .resource::<SkyOrientation>()
            .copied()
            .unwrap_or_default();
        let lights = authored_lights(ctx);
        let sun = DirectionalLight {
            direction: sky.rotate(lights[0].direction),
            ..lights[0]
        };
        let camera = cam_pos.map(f64::from);
        let center = ctx
            .resource::<PlanetFrame>()
            .map(|f| f.frame.to_local_f64(f.shape.center))
            .or(self.settings.planet_center);
        let frame = AtmosphereFrame::new(
            &self.settings,
            self.settings.camera_km(camera, center),
            &sun,
        );
        let changed = self.pushed.is_none_or(|p| {
            (0..3).any(|i| (p[i] - frame.sun_transmittance[i]).abs() > SUN_TOLERANCE)
        });
        let set = (changed || turned.is_some()).then(|| {
            self.pushed = Some(frame.sun_transmittance);
            sun_driven_set(&frame, &lights, &sky)
        });
        (frame, set)
    }
}

// The world's directional lights as authored, or the neutral sun a world with
// none is lit by.
fn authored_lights(ctx: &PipelineContext) -> Vec<DirectionalLight> {
    let lights: Vec<DirectionalLight> = ctx.query::<DirectionalLight>().copied().collect();
    if !lights.is_empty() {
        return lights;
    }
    let d = &LightUniforms::DEFAULT.directional[0];
    vec![DirectionalLight {
        direction: d.direction,
        color: d.color,
        intensity: d.intensity,
    }]
}

// The directional set under the sky's rotation, its first light (the sun)
// tinted by the air it reaches the camera through.
fn sun_driven_set(
    frame: &AtmosphereFrame,
    lights: &[DirectionalLight],
    sky: &SkyOrientation,
) -> DirectionalLightSet {
    let turned = super::scene_lights::lights_under_sky(lights.iter(), sky);
    DirectionalLightSet::collect(turned.as_slice().iter().enumerate().map(|(i, light)| {
        if i == 0 {
            sun_light(frame, light)
        } else {
            *light
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::Atmosphere;
    use concinnity_core::ecs::World;

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
