// The atmosphere's per-frame inputs, and the lights it drives: the sun, lit
// through the air between it and the camera, and at night the moon, which
// takes over the shadowed first place once the sun has set. Around a planet
// both bodies stay lit, and its ground and sea are lit point by point from
// the daylight curves.

use crate::components::{DirectionalLight, SkyRotation};
use crate::ecs::PipelineContext;
use crate::gfx::render_types::LightUniforms;
use crate::planet::PlanetFrame;
use crate::render::environment::atmosphere::{
    AtmosphereFrame, AtmosphereSettings, DAYLIGHT_KNOTS, DaylightCurves, SeaLook, SurfaceBelow,
    SurfaceView, sun_light,
};
use crate::render::environment::night_sky::{NightFrame, shadowed_first, shadowed_pair};
use crate::render::light::lights::DirectionalLightSet;
use crate::render::uniforms::{DaylightBlock, DaylightBody, DaylightView};
use crate::sky::SkyOrientation;
use concinnity_core::fanout::FanOut;
use concinnity_core::math::vec3::{length, normalize_or};
use concinnity_core::math::{cos, smoothstep};

// How far the sun's transmittance, or the moon's light as a share of its
// full brightness, may drift from what the lights were last pushed with
// before they are pushed again.
const LIGHT_TOLERANCE: f32 = 1e-3;

// What the lights were last pushed for: the sun's transmittance, the moon's
// light per color as a share of the full moon's above the air, and whether
// the moon held the first place.
type PushedLights = ([f32; 3], [f32; 3], bool);

// The heights over the ground, in kilometers, across which every point's own
// sky takes over from the camera's: from there the camera's sky is no longer
// the one over the ground it sees.
const OWN_SKY_HEIGHT_KM: [f32; 2] = [0.25, 2.0];

// The angles about the planet's center, in degrees, from the camera at which a
// point's own sky starts to take over from the camera's, and has.
const OWN_SKY_REACH_DEG: [f32; 2] = [0.5, 3.0];

// The height over the ground, in meters, below which a surface is lit as the
// ground under its own sky; above it, as high as the air, the light the camera
// gets stands for it.
const GROUND_HEIGHT_M: f32 = 5_000.0;

// The world's atmosphere and the lights the backend last received.
pub(super) struct AtmosphereDrive {
    settings: AtmosphereSettings,
    pushed: Option<PushedLights>,
    // The packed daylight curves, around a planet.
    daylight: Option<[[f32; 4]; DAYLIGHT_KNOTS * 3]>,
    // The ground around the camera last sampled, around a planet with a
    // surface palette.
    surface: Option<SurfaceBelow>,
}

impl AtmosphereDrive {
    #[expect(
        clippy::large_types_passed_by_value,
        reason = "stored, so a borrow would only add a copy"
    )]
    pub(super) fn new<S: FanOut>(settings: AtmosphereSettings, rows: &S) -> Self {
        let daylight = settings
            .planet_center
            .map(|_| DaylightCurves::bake(&settings, rows).packed());
        Self {
            settings,
            pushed: None,
            daylight,
            surface: None,
        }
    }

    // This frame's inputs for a camera at `cam_pos` in the simulated frame,
    // the directional set to install when the sun's or the moon's light has
    // changed since it was last pushed (or `turned` already carries a new set,
    // which this one then replaces), and the planet's per-point sky.
    pub(super) fn frame(
        &mut self,
        ctx: &PipelineContext<'_>,
        cam_pos: [f32; 3],
        turned: Option<DirectionalLightSet>,
    ) -> (AtmosphereFrame, Option<DirectionalLightSet>, DaylightBlock) {
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
        )
        .with_surface(self.surface_below(ctx, cam_pos));
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
        let share = frame.night.and_then(|n| n.moon).map_or(0.0, |m| m.share);
        let moon_first = moon.is_some() && share > 0.0;
        let key = (
            frame.sun_transmittance,
            moon.map_or([0.0; 3], |m| m.1),
            moon_first,
        );
        let changed = self.pushed.is_none_or(|(sun, moon, first)| {
            first != key.2
                || (0..3).any(|i| {
                    (sun[i] - key.0[i]).abs() > LIGHT_TOLERANCE
                        || (moon[i] - key.1[i]).abs() > LIGHT_TOLERANCE
                })
        });
        let set = (changed || turned.is_some()).then(|| {
            self.pushed = Some(key);
            let under_sky = super::lights::lights_under_sky(lights.iter(), &sky);
            let turned = under_sky.as_slice();
            let place = if self.daylight.is_some() {
                shadowed_pair
            } else {
                shadowed_first
            };
            place(
                sun_light(&frame, &turned[0]),
                moon.map(|m| m.0),
                share,
                &turned[1..],
            )
        });
        let daylight = self.daylight.map_or(DaylightBlock::OFF, |knots| {
            let sun_body = DaylightBody {
                direction: sun.direction,
                light: sun.color.map(|c| c * sun.intensity),
            };
            let moon_body = moon.map(|(m, _)| DaylightBody {
                direction: m.direction,
                light: moon_above_air(&self.settings, &frame, &lights[0]),
            });
            let bodies = if moon_first {
                [moon_body, Some(sun_body)]
            } else {
                [Some(sun_body), moon_body]
            };
            let block = DaylightBlock::new(
                knots,
                &daylight_view(&self.settings, &frame, center, bodies),
            );
            match self.sea() {
                Some(sea) => block.with_sea(sea.deep_color, sea.level),
                None => block,
            }
        });
        (frame, set, daylight)
    }
}

impl AtmosphereDrive {
    // The planet's sea, when its ground has a palette with one.
    pub(super) fn sea(&self) -> Option<SeaLook> {
        self.settings.surface.as_ref()?.sea
    }

    // The ground around a camera at `cam_pos`, sampled again once the camera
    // has moved far enough from where it last was; `None` without a planet
    // or a palette for its ground.
    fn surface_below(
        &mut self,
        ctx: &PipelineContext<'_>,
        cam_pos: [f32; 3],
    ) -> Option<SurfaceBelow> {
        let palette = self.settings.surface.as_ref()?;
        let planet = ctx.resource::<PlanetFrame>()?;
        let view = SurfaceView::of(planet, cam_pos);
        if self.surface.is_none_or(|s| s.stale(&view)) {
            self.surface = Some(SurfaceBelow::sample(palette, &planet.shape, &view));
        }
        self.surface
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

// The moon's light above the air: the sun's scaled by the moon's brightness
// and phase, black without a moon.
fn moon_above_air(
    settings: &AtmosphereSettings,
    frame: &AtmosphereFrame,
    sun: &DirectionalLight,
) -> [f32; 3] {
    let phase = frame.night.and_then(|n| n.moon).map_or(0.0, |m| m.phase);
    let brightness = settings
        .night
        .and_then(|n| n.moon)
        .map_or(0.0, |m| m.brightness);
    sun.color.map(|c| c * sun.intensity * brightness * phase)
}

// Where the camera of `frame` stands toward the planet centered at `center`
// in the simulated frame, lit by `bodies`.
fn daylight_view(
    settings: &AtmosphereSettings,
    frame: &AtmosphereFrame,
    center: Option<[f64; 3]>,
    bodies: [Option<DaylightBody>; 2],
) -> DaylightView {
    let camera = frame.camera_km;
    let height = length(camera) - settings.bottom_radius;
    DaylightView {
        center: center
            .unwrap_or_default()
            .map(concinnity_core::math::narrow_f32),
        camera_up: normalize_or(camera, 1e-6, [0.0, 1.0, 0.0]),
        camera_reach: smoothstep(OWN_SKY_HEIGHT_KM[0], OWN_SKY_HEIGHT_KM[1], height),
        reach: OWN_SKY_REACH_DEG.map(|deg| cos(deg.to_radians())),
        ground_radius: settings.bottom_radius * 1_000.0,
        ground_height: GROUND_HEIGHT_M,
        bodies,
    }
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
    use crate::render::environment::night_sky::NightSkySettings;
    use concinnity_core::fanout::Inline;
    use concinnity_core::math::sin;
    use std::sync::OnceLock;

    // An Earth-sized planet below the origin, its ground at the origin.
    const RADIUS_M: f32 = 6_360_000.0;
    const CENTER: [f64; 3] = [0.0, -6_360_000.0, 0.0];

    // Earth's air around the planet, under a night sky whose moon is a waxing
    // crescent: up in the day sky, and still up after sunset.
    fn planet_settings() -> AtmosphereSettings {
        AtmosphereSettings::resolve(&Atmosphere::default(), Some((CENTER, RADIUS_M))).with_night(
            Some(NightSkySettings::resolve(&NightSky {
                moon_start_deg: 60.0,
                ..NightSky::default()
            })),
        )
    }

    // A drive around the planet, its curves baked once for every test.
    fn planet_drive() -> AtmosphereDrive {
        static KNOTS: OnceLock<[[f32; 4]; DAYLIGHT_KNOTS * 3]> = OnceLock::new();
        let settings = planet_settings();
        let knots = KNOTS.get_or_init(|| DaylightCurves::bake(&settings, &Inline).packed());
        AtmosphereDrive {
            settings,
            pushed: None,
            daylight: Some(*knots),
            surface: None,
        }
    }

    // The camera 2 m above the ground, `deg` degrees round the planet from
    // the top.
    fn on_ground(deg: f32) -> [f32; 3] {
        let (r, a) = (RADIUS_M + 2.0, deg.to_radians());
        [r * sin(a), r * cos(a) - RADIUS_M, 0.0]
    }

    fn drive() -> AtmosphereDrive {
        AtmosphereDrive::new(
            AtmosphereSettings::resolve(&Atmosphere::default(), None),
            &Inline,
        )
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
        let (frame, set, _) = drive.frame(&world.context(), [0.0, 2.0, 0.0], None);
        let set = set.expect("the first frame installs the sun");
        let sun = set.as_slice()[0];
        assert!(
            sun.color[0] > 2.0 * sun.color[2],
            "a low sun is red: {:?}",
            sun.color
        );
        assert_eq!(sun.intensity, 2.0);
        assert!(frame.camera_height(&drive.settings) > 0.0);
        let (_, again, _) = drive.frame(&world.context(), [0.0, 2.0, 0.0], None);
        assert!(again.is_none());
        // Climbing through the haze changes the light, so it is pushed again.
        let (_, high, _) = drive.frame(&world.context(), [0.0, 8_000.0, 0.0], None);
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
        let (_, set, _) = drive().frame(&world.context(), [0.0, 2.0, 0.0], None);
        let set = set.expect("installed");
        assert_eq!(set.as_slice()[0].color, [0.0; 3]);
        assert_eq!(set.as_slice()[1].color, [0.2, 0.3, 0.4]);
    }

    // At night under a full moon the moon holds the shadowed place, lit
    // through the air; by day the sun keeps it and the moon casts nothing.
    #[test]
    fn the_moon_takes_the_shadows_at_night() {
        let settings = AtmosphereSettings::resolve(&Atmosphere::default(), None)
            .with_night(Some(NightSkySettings::resolve(&NightSky::default())));
        let mut night = world_with_sun([0.1, -1.0, 0.0]);
        let mut drive = AtmosphereDrive::new(settings, &Inline);
        let (frame, set, _) = drive.frame(&night.context(), [0.0, 2.0, 0.0], None);
        let first = set.expect("installed").as_slice()[0];
        let moon = frame.night.and_then(|n| n.moon).expect("a moon");
        assert_eq!(first.direction, moon.direction);
        assert!(moon.direction[1] > 0.9, "full: opposite the sun");
        let full = 2.0 * NightSky::default().moon_brightness * moon.phase;
        assert!((first.intensity / full - 1.0).abs() < 1e-4);
        assert!(first.color.iter().all(|&c| c > 0.7), "{:?}", first.color);
        let mut day = world_with_sun([0.1, 1.0, 0.0]);
        let (_, set, _) =
            AtmosphereDrive::new(settings, &Inline).frame(&day.context(), [0.0, 2.0, 0.0], None);
        let set = set.expect("installed");
        assert_eq!(set.as_slice().len(), 1);
        assert_eq!(set.as_slice()[0].intensity, 2.0);
    }

    // A world with no directional light is lit by the neutral sun, through the
    // air.
    #[test]
    fn the_neutral_sun_stands_in() {
        let mut world = World::new();
        let (_, set, _) = drive().frame(&world.context(), [0.0, 2.0, 0.0], None);
        let set = set.expect("installed");
        assert_eq!(set.as_slice().len(), 1);
        let d = LightUniforms::DEFAULT.directional[0].direction;
        assert_eq!(set.as_slice()[0].direction, d);
    }

    // Around a planet whose ground has a palette, the frame carries the
    // ground around the camera, sampled again only once the camera moves.
    #[test]
    fn a_planet_frame_carries_the_ground_around_the_camera() {
        use crate::render::environment::atmosphere::SurfacePalette;
        use crate::render::environment::ground::GroundCovers;
        let mut world = world_with_sun([0.0, 1.0, 0.0]);
        let mut drive = planet_drive();
        let (frame, _, _) = drive.frame(&world.context(), [0.0, 2.0, 0.0], None);
        assert!(frame.surface.is_none(), "no palette, no table");

        let planet = crate::components::Planet::default();
        world.insert_resource(PlanetFrame {
            shape: planet.shape(),
            frame: crate::planet::LocalFrame::AUTHORED,
        });
        let palette = SurfacePalette::new([0.3, 0.2, 0.1], GroundCovers::default(), None, None);
        drive.settings = drive.settings.with_surface(Some(palette));
        let cam = [0.0, -9_960.0, 0.0];
        let (frame, _, _) = drive.frame(&world.context(), cam, None);
        let first = frame.surface.expect("a table");
        assert_eq!(first.land, [0.3, 0.2, 0.1]);
        let (frame, _, _) = drive.frame(&world.context(), cam, None);
        assert_eq!(frame.surface, Some(first));
    }

    // A world without a planet keeps the per-point sky off.
    #[test]
    fn only_a_planet_gets_the_per_point_sky() {
        let mut world = world_with_sun([0.0, 1.0, 0.0]);
        let (_, _, block) = drive().frame(&world.context(), [0.0, 2.0, 0.0], None);
        assert_eq!(block, DaylightBlock::OFF);
        let (_, _, block) = planet_drive().frame(&world.context(), on_ground(0.0), None);
        assert!(block.is_on());
    }

    // Walking round the planet from day into a moonlit night, the lights the
    // drive last pushed are always the ones it would push now, and the curves
    // light the first two in their order, whichever body holds the first
    // place.
    #[test]
    fn a_planet_pushes_its_lights_whenever_they_change() {
        let mut world = world_with_sun([0.0, 1.0, 0.0]);
        let mut drive = planet_drive();
        let brightness = NightSky::default().moon_brightness;
        let mut pushed = None;
        let (mut day, mut night) = (false, false);
        for step in 0..2_400 {
            let at = on_ground(-20.0 + step as f32 * 0.05);
            let (frame, set, block) = drive.frame(&world.context(), at, None);
            if let Some(set) = set {
                pushed = Some(set);
            }
            let pushed = pushed.as_ref().expect("the first frame pushes");
            let (_, now, _) = planet_drive().frame(&world.context(), at, None);
            let now = now.expect("a fresh drive pushes");
            let moon = frame.night.and_then(|n| n.moon).expect("a moon");
            let (a, b) = (pushed.as_slice(), now.as_slice());
            assert_eq!(a.len(), b.len(), "{step}");
            for (a, b) in a.iter().zip(b) {
                assert_eq!(a.direction, b.direction, "{step}");
                let full = if b.direction == moon.direction {
                    2.0 * brightness
                } else {
                    2.0
                };
                for c in 0..3 {
                    let (x, y) = (a.color[c] * a.intensity, b.color[c] * b.intensity);
                    assert!((x - y).abs() <= 1.5e-3 * full, "{step}: {x} {y}");
                }
            }
            for (body, light) in block.body_dir.iter().zip(b) {
                assert_eq!(body[3], 1.0, "{step}");
                for (x, y) in body.iter().zip(light.direction) {
                    assert!((x - y).abs() < 1e-5, "{step}");
                }
            }
            if b[0].direction == moon.direction {
                night = true;
            } else {
                assert!(!night, "{step}: the sun took the first place back");
                day = true;
            }
        }
        assert!(day && night);
    }

    // The per-point sky stands in for the camera's away from it, and
    // everywhere once the camera is high above the ground.
    #[test]
    fn the_view_reaches_from_the_camera() {
        let s = planet_settings();
        let sun = DirectionalLight {
            direction: [0.0, 1.0, 0.0],
            color: [1.0; 3],
            intensity: 2.0,
        };
        let view = |height_m: f64| {
            let camera = [0.0, height_m, 0.0];
            let frame = AtmosphereFrame::new(&s, s.camera_km(camera, Some(CENTER)), &sun);
            daylight_view(&s, &frame, Some(CENTER), [None; 2])
        };
        let ground = view(2.0);
        assert_eq!(ground.camera_reach, 0.0);
        assert!((ground.camera_up[1] - 1.0).abs() < 1e-6);
        assert!(ground.reach[0] > ground.reach[1] && ground.reach[1] < 1.0);
        assert_eq!(ground.center, [0.0, -RADIUS_M, 0.0]);
        assert!((ground.ground_radius - RADIUS_M).abs() < 1.0);
        assert_eq!(ground.ground_height, GROUND_HEIGHT_M);
        let between = view(1_000.0).camera_reach;
        assert!(0.0 < between && between < 1.0, "{between}");
        assert_eq!(view(10_000.0).camera_reach, 1.0);
    }

    // The moon's light above the air is the sun's scaled by its brightness
    // and phase, and black without a night sky.
    #[test]
    fn the_moon_above_the_air_is_the_sun_scaled() {
        let sun = DirectionalLight {
            direction: [0.0, -1.0, 0.0],
            color: [1.0, 0.9, 0.8],
            intensity: 2.0,
        };
        let mut world = world_with_sun(sun.direction);
        let mut drive = planet_drive();
        let (frame, _, _) = drive.frame(&world.context(), on_ground(0.0), None);
        let phase = frame.night.and_then(|n| n.moon).expect("a moon").phase;
        let lit = moon_above_air(&drive.settings, &frame, &sun);
        let brightness = NightSky::default().moon_brightness;
        for (lit, color) in lit.iter().zip(sun.color) {
            let want = color * 2.0 * brightness * phase;
            assert!((lit - want).abs() <= want * 1e-6, "{color}");
        }
        let bare = AtmosphereSettings::resolve(&Atmosphere::default(), None);
        let frame = AtmosphereFrame::new(&bare, [0.0, bare.bottom_radius + 0.002, 0.0], &sun);
        assert_eq!(moon_above_air(&bare, &frame, &sun), [0.0; 3]);
    }
}
