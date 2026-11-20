//! Texture-pool slot updates + IBL / color-grading hot-swap. Driven both by
//! the streaming subsystem (per-slot upload + eviction placeholders) and by
//! asset hot-reload (`cn run -f` only) for envmaps + LUTs.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_engine::bake;
use concinnity_engine::gfx::render_types::TextureSlot;
use concinnity_engine::render::backend::TextureUploads;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};

use crate::metal::context::MtlContext;
use crate::metal::texture::{upload_texture, upload_texture_image};

fn texture_slot_in_range(op: &str, slot: usize, pool_size: usize) -> RenderResult<()> {
    if slot < pool_size {
        Ok(())
    } else {
        Err(RenderError::Other(format!(
            "{op}: slot {slot} out of range (pool size {pool_size})"
        )))
    }
}

impl TextureUploads for MtlContext {
    // Reset albedo pool `slot` to a 1x1 mid-gray placeholder until it streams
    // back, distinct from the white no-texture fallback.
    fn evict_texture_slot(&mut self, slot: TextureSlot) -> RenderResult<()> {
        let slot = slot.index();
        texture_slot_in_range("evict_texture_slot", slot, self.scene.textures.len())?;
        self.scene.textures[slot] =
            upload_texture(&self.hw.allocator, 1, 1, &[128, 128, 128, 255])?;
        self.arg_buffers.texture_epoch += 1;
        Ok(())
    }

    // Replace albedo pool `slot` with fresh pixels; the texture epoch bump
    // rewrites every argument-buffer ring slot.
    fn update_texture_slot(
        &mut self,
        slot: TextureSlot,
        image: &bake::texture::TextureImage,
    ) -> RenderResult<()> {
        let slot = slot.index();
        texture_slot_in_range("update_texture_slot", slot, self.scene.textures.len())?;
        self.scene.textures[slot] = upload_texture_image(&self.hw.allocator, image)?;
        self.arg_buffers.texture_epoch += 1;
        Ok(())
    }

    // Swap the live 3D color-grading LUT for a fresh payload. Driven by
    // asset hot-reload (`cn run -f` only). The composite pass binds
    // `self.color_lut` every frame, so the new texture is sampled on the
    // next `draw_frame` with no pipeline rebuild.
    fn update_color_lut(&mut self, size: u32, data: &[u8]) -> RenderResult<()> {
        let tex = crate::metal::texture::upload_color_lut(&self.hw.allocator, size, data)?;
        self.scene.color_lut = tex;
        Ok(())
    }

    // Swap the IBL cubes on hot reload; sizes may differ from the original.
    fn update_environment_map(&mut self, payload: &[u8]) -> RenderResult<()> {
        let view = bake::environment_map::deserialize(payload)
            .map_err(|e| RenderError::Other(format!("envmap hot-reload payload malformed: {e}")))?;
        let new_env = crate::metal::texture::upload_environment_map(
            &self.hw.allocator,
            view.irradiance_face,
            view.irradiance_bytes,
            view.prefilter_face,
            &view.prefilter_mip_bytes,
        )?;
        let old = std::mem::replace(&mut self.scene.env_map, new_env);
        // An atmosphere writes both cubes; the map is held for a world that
        // drops it.
        if self.atmosphere.is_some() {
            self.scene.env_map = old;
        }

        self.arg_buffers.texture_epoch += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_inside_the_pool_is_accepted() {
        assert_eq!(texture_slot_in_range("op", 0, 1), Ok(()));
        assert_eq!(texture_slot_in_range("op", 3, 4), Ok(()));
    }

    #[test]
    fn slot_at_or_past_the_pool_size_is_other() {
        assert_eq!(
            texture_slot_in_range("evict_texture_slot", 4, 4),
            Err(RenderError::Other(
                "evict_texture_slot: slot 4 out of range (pool size 4)".to_string()
            ))
        );
        assert!(texture_slot_in_range("op", 0, 0).is_err());
    }
}
