// src/metal/lights.rs
//
// Runtime replacement of the directional-light set. The lights are packed into
// `LightUniforms` at init and pushed to the fragment shader every frame, so a
// new sun is a field rewrite rather than a buffer rebuild. What init derived
// from the first light and cached -- the cascade shadow direction -- is
// re-derived here, since nothing else refreshes it.

use crate::components::DirectionalLight;

use super::context::MtlContext;

impl MtlContext {
    // Replace the live directional lights. The main pass, fog, raymarch, and RT
    // reflection params all read `light_uniforms` afresh each draw, so they need
    // nothing beyond the rewrite; `shadow.light_dir` is the one init-time cache.
    pub(crate) fn update_directional_lights(&mut self, lights: &[DirectionalLight]) {
        let (directional, num_directional) = crate::gfx::lights::directional_light_data(lights);
        if self.light_uniforms.directional == directional
            && self.light_uniforms.num_directional == num_directional
        {
            return;
        }
        self.light_uniforms.directional = directional;
        self.light_uniforms.num_directional = num_directional;
        self.shadow.light_dir = crate::gfx::lights::sun_direction(&self.light_uniforms);
    }
}
