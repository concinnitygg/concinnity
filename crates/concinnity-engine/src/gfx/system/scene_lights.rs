// The world's authored lights, packed once at init into the backend's static
// light data.

use concinnity_core::components::{DirectionalLight, PointLight, RectAreaLight, SpotLight};
use concinnity_core::ecs::PipelineContext;
use concinnity_core::gfx::render_types::LightUniforms;
use concinnity_core::render::lights::{self, LightData};

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

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::ecs::{Arena, ComponentStorage, FrameContext, Resources};
    use concinnity_core::profile::FrameProfile;
    use concinnity_host::store::blob::BlobData;

    // The lights stay resident after packing, and the ambient reaches the uniforms.
    #[test]
    fn lights_are_packed_without_being_drained() {
        let mut components = ComponentStorage::default();
        components.push_typed(PointLight::default());
        components.push_typed(DirectionalLight::default());
        let (mut blob, mut profile, mut resources) =
            (BlobData::empty(), FrameProfile::default(), Resources::new());
        let scratch = Arena::with_capacity(1024);
        let ctx = PipelineContext {
            components: &mut components,
            blob: &mut blob,
            profile: &mut profile,
            resources: &mut resources,
            frame: FrameContext::new(&scratch),
        };

        let (data, uniforms) = gather_lights(&ctx, 0.25);
        assert_eq!(data.lights.len(), 1);
        assert_eq!(uniforms.ambient_intensity, 0.25);
        assert_eq!(ctx.query::<PointLight>().count(), 1);
        assert_eq!(ctx.query::<DirectionalLight>().count(), 1);
    }
}
