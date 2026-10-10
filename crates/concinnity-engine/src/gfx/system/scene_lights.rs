// The world's authored lights, packed once at init into the backend's static
// light data.

use concinnity_core::components::{DirectionalLight, PointLight, RectAreaLight, SpotLight};
use concinnity_core::ecs::PipelineContext;
use concinnity_core::gfx::render_types::LightUniforms;
use concinnity_core::planet::Rebase;
use concinnity_core::render::lights::{self, DirectionalLightSet, LightData};
use concinnity_core::sky::SkyOrientation;

// Pack every light into the local-light buffers and the shared uniforms, whose
// `ambient_intensity` scales each backend's IBL / flat-fallback ambient. Lights
// are read, not drained, so editor tooling can still address them by name.
pub(super) fn gather_lights(
    ctx: &PipelineContext,
    ambient_intensity: f32,
) -> (LightData, LightUniforms) {
    let dir_lights: Vec<DirectionalLight> = ctx.query::<DirectionalLight>().cloned().collect();
    let pt_lights: Vec<PointLight> = ctx.query::<PointLight>().cloned().collect();
    let spot_lights: Vec<SpotLight> = ctx.query::<SpotLight>().cloned().collect();
    let rect_lights: Vec<RectAreaLight> = ctx.query::<RectAreaLight>().cloned().collect();
    let data = lights::build_light_data(&pt_lights, &spot_lights, &rect_lights);
    let uniforms =
        lights::build_light_uniforms(dir_lights, pt_lights, &data.lights, ambient_intensity);
    (data, uniforms)
}

// Carry every local light into the frame `rebase` moves the world to, and pack
// the moved lights for the backend to rewrite in place. `None` for a world with
// no local light. The ambient scale rides `LightUniforms` but is not part of
// what moves, so it is left at zero here.
pub(super) fn carry_local_lights(
    ctx: &mut PipelineContext,
    rebase: &Rebase,
) -> Option<(LightData, LightUniforms)> {
    let mut any = false;
    for light in ctx.query_mut::<PointLight>() {
        light.position = rebase.apply_point(light.position);
        any = true;
    }
    for light in ctx.query_mut::<SpotLight>() {
        light.position = rebase.apply_point(light.position);
        light.direction = rebase.apply_vector(light.direction);
        any = true;
    }
    for light in ctx.query_mut::<RectAreaLight>() {
        light.center = rebase.apply_point(light.center);
        light.normal = rebase.apply_vector(light.normal);
        any = true;
    }
    any.then(|| gather_lights(ctx, 0.0))
}

// Every authored direction carried by the sky's current rotation. A
// directional light is at infinity by definition, so it rides the celestial
// sphere and turns with it. Returned inline so the per-frame extraction that
// calls it under a turning sky allocates nothing.
pub(super) fn lights_under_sky<'a>(
    lights: impl Iterator<Item = &'a DirectionalLight>,
    sky: &SkyOrientation,
) -> DirectionalLightSet {
    DirectionalLightSet::collect(lights.map(|light| DirectionalLight {
        direction: sky.rotate(light.direction),
        ..*light
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::ecs::World;

    // A frame move carries each kind of local light to where the move says,
    // and the packed lights the backend rewrites are the carried ones.
    #[test]
    fn a_frame_move_carries_every_local_light() {
        let rebase = Rebase {
            rotation: concinnity_core::math::quat_from_axis_angle([1.0, 0.0, 0.2], 0.02),
            translation: [-1_000.0, 12.0, 4.0],
        };
        let mut world = World::new();
        world.push(PointLight {
            position: [1_001.0, 2.0, 3.0],
            ..Default::default()
        });
        world.push(SpotLight {
            position: [998.0, 5.0, -2.0],
            direction: [0.0, -1.0, 0.0],
            ..Default::default()
        });
        world.push(RectAreaLight {
            center: [1_004.0, 3.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            ..Default::default()
        });
        let (data, uniforms) =
            carry_local_lights(&mut world.context(), &rebase).expect("the world has lights");
        let ctx = world.context();
        let close = |a: [f32; 3], b: [f32; 3]| (0..3).all(|i| (a[i] - b[i]).abs() < 1e-3);
        let point = ctx.query::<PointLight>().next().unwrap();
        assert!(close(
            point.position,
            rebase.apply_point([1_001.0, 2.0, 3.0])
        ));
        let spot = ctx.query::<SpotLight>().next().unwrap();
        assert!(close(spot.position, rebase.apply_point([998.0, 5.0, -2.0])));
        assert!(close(spot.direction, rebase.apply_vector([0.0, -1.0, 0.0])));
        let rect = ctx.query::<RectAreaLight>().next().unwrap();
        assert!(close(rect.center, rebase.apply_point([1_004.0, 3.0, 0.0])));
        assert!(close(rect.normal, rebase.apply_vector([0.0, 0.0, 1.0])));
        assert_eq!(data.lights.len(), 3);
        assert!(close(data.lights[0].position, point.position));
        assert!(close(uniforms.point[0].position, point.position));
        assert_eq!(uniforms.num_point, 1);
        assert!(carry_local_lights(&mut World::new().context(), &rebase).is_none());
    }

    // The lights stay resident after packing, and the ambient reaches the uniforms.
    #[test]
    fn lights_are_packed_without_being_drained() {
        let mut world = World::new();
        world.push(PointLight::default());
        world.push(DirectionalLight::default());
        let ctx = world.context();

        let (data, uniforms) = gather_lights(&ctx, 0.25);
        assert_eq!(data.lights.len(), 1);
        assert_eq!(uniforms.ambient_intensity, 0.25);
        assert_eq!(ctx.query::<PointLight>().count(), 1);
        assert_eq!(ctx.query::<DirectionalLight>().count(), 1);
    }

    // A turned sky carries every directional light with it, so the light and
    // whatever body is hung on the same rotation keep agreeing.
    #[test]
    fn the_sky_carries_the_directional_lights_round() {
        let authored = DirectionalLight {
            direction: [0.0, 0.0, 1.0],
            ..Default::default()
        };
        let sky = SkyOrientation::new([1.0, 0.0, 0.0], 90.0);
        let turned = lights_under_sky(std::iter::once(&authored), &sky);
        let turned = turned.as_slice();
        assert_eq!(turned.len(), 1);
        let d = turned[0].direction;
        assert!(d[1] > 0.99, "a quarter turn puts it overhead: {d:?}");
        assert_eq!(turned[0].color, authored.color);
        assert_eq!(turned[0].intensity, authored.intensity);
    }

    // A still sky leaves the authored set exactly as it was.
    #[test]
    fn an_unturned_sky_leaves_the_lights_alone() {
        let authored = DirectionalLight {
            direction: [0.35, 0.55, 1.0],
            ..Default::default()
        };
        let same = lights_under_sky(std::iter::once(&authored), &SkyOrientation::default());
        for k in 0..3 {
            assert!((same.as_slice()[0].direction[k] - authored.direction[k]).abs() < 1e-6);
        }
    }
}
