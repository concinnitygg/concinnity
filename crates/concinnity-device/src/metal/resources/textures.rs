// src/metal/resources/textures.rs
//
// Texture-pool slot updates + IBL / colour-grading hot-swap. Driven both by
// the streaming subsystem (per-slot upload + eviction placeholders) and by
// asset hot-reload (`cn debug` only) for envmaps + LUTs.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::runtime::ProtocolObject;

use crate::metal::context::MtlContext;
use crate::metal::texture::{upload_texture, upload_texture_image};

impl MtlContext {
    // The texture a `normal_map_slot` samples for the legacy per-draw normal
    // binding: a real normal map is a texture in the shared pool at its own slot;
    // `NO_NORMAL_MAP_SLOT` selects the flat-normal fallback (the first entry of
    // `fallback_textures`).
    pub(in crate::metal) fn normal_pool_texture(
        &self,
        normal_map_slot: usize,
    ) -> &ProtocolObject<dyn objc2_metal::MTLTexture> {
        if normal_map_slot == crate::gfx::render_types::NO_NORMAL_MAP_SLOT {
            self.fallback_textures[0].as_ref()
        } else {
            let last = self.textures.len().saturating_sub(1);
            self.textures[normal_map_slot.min(last)].as_ref()
        }
    }

    // The same for an albedo `texture_slot`: `NO_ALBEDO_SLOT` selects the white
    // fallback (the second entry), so an untextured material shows its tint
    // rather than whichever texture holds slot 0.
    pub(in crate::metal) fn albedo_pool_texture(
        &self,
        texture_slot: usize,
    ) -> &ProtocolObject<dyn objc2_metal::MTLTexture> {
        if texture_slot == crate::gfx::render_types::NO_ALBEDO_SLOT {
            self.fallback_textures[1].as_ref()
        } else {
            let last = self.textures.len().saturating_sub(1);
            self.textures[texture_slot.min(last)].as_ref()
        }
    }

    // Replace albedo texture-pool `slot` with freshly decoded RGBA8 pixels.
    //
    // The asset-streaming subsystem calls this to bring a texture resident
    // after init. Both the bindless pool bind and the per-draw fallback bind
    // read `self.textures` fresh each frame, so the swapped texture is picked
    // up on the next `draw_frame` with no pipeline rebuild.
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
        Ok(())
    }

    // Reset albedo texture-pool `slot` to a 1x1 mid-grey placeholder.
    //
    // Used by the asset-streaming subsystem to mark a slot whose texture is
    // not yet resident; a later `update_texture_slot` brings the real texture
    // back. The grey is distinct from the white no-texture fallback so a
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
        Ok(())
    }

    // Swap the live 3D colour-grading LUT for a fresh payload. Driven by
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
        let view = crate::bake::environment_map::deserialise(payload)
            .map_err(|e| format!("envmap hot-reload payload malformed: {}", e))?;
        let new_env = crate::metal::texture::upload_environment_map(
            &self.allocator,
            view.irradiance_face,
            view.irradiance_bytes,
            view.prefilter_face,
            &view.prefilter_mip_bytes,
        )?;
        self.env_map = new_env;
        Ok(())
    }
}
