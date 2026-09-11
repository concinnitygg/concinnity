// src/metal/resources/textures.rs
//
// Texture-pool slot updates + IBL / color-grading hot-swap. Driven both by
// the streaming subsystem (per-slot upload + eviction placeholders) and by
// asset hot-reload (`cn debug` only) for envmaps + LUTs.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::bake;

use crate::metal::context::MtlContext;
use crate::metal::texture::{upload_texture, upload_texture_image};

impl MtlContext {
    // Replace albedo texture-pool `slot` with freshly decoded RGBA8 pixels.
    //
    // The asset-streaming subsystem calls this to bring a texture resident
    // after init. Bumping the texture epoch re-encodes the bindless argument
    // buffer into every ring slot, so the swapped texture is picked up from the
    // next `draw_frame` with no pipeline rebuild.
    pub(crate) fn update_texture_slot(
        &mut self,
        slot: usize,
        image: &concinnity_core::bake::texture::TextureImage,
    ) -> Result<(), String> {
        if slot >= self.textures.len() {
            return Err(format!(
                "update_texture_slot: slot {} out of range (pool size {})",
                slot,
                self.textures.len()
            ));
        }
        self.textures[slot] = upload_texture_image(&self.allocator, image)?;
        self.texture_epoch += 1;
        Ok(())
    }

    // Reset albedo texture-pool `slot` to a 1x1 mid-gray placeholder.
    //
    // Used by the asset-streaming subsystem to mark a slot whose texture is
    // not yet resident; a later `update_texture_slot` brings the real texture
    // back. The gray is distinct from the white no-texture fallback so a
    // not-yet-streamed slot reads differently under inspection.
    pub(crate) fn evict_texture_slot(&mut self, slot: usize) -> Result<(), String> {
        if slot >= self.textures.len() {
            return Err(format!(
                "evict_texture_slot: slot {} out of range (pool size {})",
                slot,
                self.textures.len()
            ));
        }
        self.textures[slot] = upload_texture(&self.allocator, 1, 1, &[128, 128, 128, 255])?;
        self.texture_epoch += 1;
        Ok(())
    }

    // Swap the live 3D color-grading LUT for a fresh payload. Driven by
    // asset hot-reload (`cn debug` only). The composite pass binds
    // `self.color_lut` every frame, so the new texture is sampled on the
    // next `draw_frame` with no pipeline rebuild.
    pub(crate) fn update_color_lut(&mut self, size: u32, data: &[u8]) -> Result<(), String> {
        let tex = crate::metal::texture::upload_color_lut(&self.allocator, size, data)?;
        self.color_lut = tex;
        Ok(())
    }

    // Swap the live IBL cubemap pair for a freshly precomputed envmap payload.
    // Driven by asset hot-reload (`cn debug` only). The fragment shader binds
    // `self.env_map.irradiance` and `self.env_map.prefilter` every frame, so
    // the new cubes are sampled on the next `draw_frame` with no pipeline
    // rebuild. The new payload may declare different mip / face sizes than
    // the original -- `EnvironmentMapTextures` is replaced wholesale.
    pub(crate) fn update_environment_map(&mut self, payload: &[u8]) -> Result<(), String> {
        let view = bake::environment_map::deserialize(payload)
            .map_err(|e| format!("envmap hot-reload payload malformed: {}", e))?;
        let new_env = crate::metal::texture::upload_environment_map(
            &self.allocator,
            view.irradiance_face,
            view.irradiance_bytes,
            view.prefilter_face,
            &view.prefilter_mip_bytes,
        )?;
        self.env_map = new_env;
        self.texture_epoch += 1;
        Ok(())
    }
}
