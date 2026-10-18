//! The textures `MtlContext::draw_frame` hands the post-decoration stack as the
//! scene.

use concinnity_core::render::error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLTexture;

use crate::metal::context::MtlContext;

// The textures the post-decoration stack reads as the scene.
pub(super) struct SceneTextures {
    pub(super) scene_input: Retained<ProtocolObject<dyn MTLTexture>>,
    pub(super) scene_color: Retained<ProtocolObject<dyn MTLTexture>>,
}

impl MtlContext {
    pub(super) fn scene_textures(&self) -> error::RenderResult<SceneTextures> {
        // `scene_input` is the engine-owned texture the post-decoration stack
        // treats as the pre-TAA scene: the reflection composite's output when a
        // reflection path is live, else the raw `hdr_resolve`.
        //
        // That output is the *composited* scene, not the reflection. Both the
        // SSR and the RT resolve write radiance into `ssr.reflection`, then call
        // the shared `encode_reflection_composite`, which blends that over
        // `hdr_resolve` into the output. Worth stating precisely: the DirectX
        // equivalent split the two apart and left its upscaler reading the
        // radiance buffer as if it were the scene.
        //
        // `scene_color` is what Bloom + Composite read:
        //   - the upscaler's output (drawable-res) when MetalFX is on,
        //   - the TAA resolve target when TAA is on,
        //   - otherwise just the pre-TAA scene (no temporal stage).
        let scene_input = if self.reflection_path().resolves() {
            self.ssr
                .composite
                .as_ref()
                .ok_or_else(|| {
                    error::RenderError::Other(
                        "reflections enabled but the reflection composite is missing".into(),
                    )
                })?
                .output()
                .texture()
                .clone()
        } else {
            self.targets.hdr.hdr_resolve.clone()
        };
        let scene_color = if let Some(u) = &self.upscale.scaler {
            u.output.clone()
        } else if let Some(out) = self.taa.output() {
            out.clone()
        } else {
            scene_input.clone()
        };

        Ok(SceneTextures {
            scene_input,
            scene_color,
        })
    }
}
