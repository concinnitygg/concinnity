//! The textures `MtlContext::draw_frame` hands the post-decoration stack as the
//! scene.

use concinnity_engine::render::backend::error;
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
        // `scene_input` is the pre-TAA scene: the reflection composite's output
        // (the composited scene, not the reflection) when a resolve runs, else
        // `hdr_resolve`. `scene_color` is what bloom and composite read: the
        // upscaler output, the TAA target, or `scene_input`.
        let scene_input = if self.state.planner.reads().reflections {
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
