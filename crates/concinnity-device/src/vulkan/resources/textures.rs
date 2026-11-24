//! Texture-pool slot management for `VkContext`: bindless-pool, decal and
//! particle descriptor rewires when an albedo or normal-map slot is streamed in
//! or evicted. Vulkan bakes texture *views* into descriptor sets, so a slot swap
//! must walk every set that samples this slot.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::bake;
use concinnity_engine::gfx::render_types::TextureSlot;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::backend::{RenderBackend, TextureUploads};

use super::super::context::VkContext;
use super::super::descriptor_layout::{IRRADIANCE_CUBE_BINDING, PREFILTER_CUBE_BINDING};
use super::super::set_writes::SetWrites;
use super::super::texture::{
    GpuUploadContext, StreamedUploadRetire, upload_texture_image, upload_texture_image_deferred,
};

impl VkContext {
    // Re-point bindless texture-pool element `index` of `set` to `view`.
    // Keeps the bindless texture pool in sync with a streamed texture swap; the
    // pool is one handle-indexed image set (albedo + normal maps share it).
    fn write_pool_image(&self, set: vk::DescriptorSet, index: u32, view: vk::ImageView) {
        SetWrites::new(set)
            .sampled_image_at(1, index, view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .apply(&self.hw.device);
    }

    // Re-point every descriptor that samples texture-pool `slot` and may be
    // referenced by in-flight command buffers: the per-frame bindless pool
    // copies plus the decal / clone / particle sets. Only legal under a device
    // drain (the fallback streaming path and the `cn run -f` hot-reload paths).
    fn rewrite_bound_texture_sets(&self, slot: usize) {
        let view = self.scene.textures[slot].view;
        // The bindless pool addresses each texture at pool index == its handle,
        // shared by albedo and normal sampling, so one re-point covers both.
        for &set in &self.cull.bindless_sets {
            self.write_pool_image(set, u32::saturating_from(slot), view);
        }
        // Per-decal albedo descriptors. Walk the decal-side slot tracker;
        // a world with no decals pays nothing here.
        self.rewrite_decal_albedo_slot(slot);
        // Particle emitters sample an albedo from the texture pool into their
        // own render set, so re-point any whose source slot is this one.
        self.rewrite_particle_albedo_slot(slot);
    }

    // Re-point every descriptor that samples texture-pool `slot`. Only legal
    // under a device drain. An in-flight reflection-probe bake needs no arm
    // here: each bake face snapshots the live pool into its own set right
    // before recording.
    fn rewrite_texture_slot(&self, slot: usize) {
        self.rewrite_bound_texture_sets(slot);
    }

    // Whether replacing pool `slot` must drain the device first: true when a
    // single-copy set (decal, particle) that may be pending samples it, or the
    // world GPU-drives nothing.
    fn streamed_slot_needs_drain(&self, slot: usize) -> bool {
        let bindless_active =
            self.cull.bindless_pipeline.is_some() && self.state.draw.record_count() > 0;
        if !bindless_active {
            return true;
        }
        self.decal_samples_slot(slot) || self.particle_samples_slot(slot)
    }

    // After this slot's fence wait: re-point its bindless pool copy at swapped
    // slots and free retires whose fence has signaled.
    pub(in crate::vulkan) fn apply_streamed_texture_rewrites(&mut self, frame: usize) {
        self.stream.frame += 1;
        if !self.stream.pool_rewrites.is_empty() {
            let last = self.scene.textures.len().saturating_sub(1);
            for slot in self.stream.pool_rewrites.begin_frame() {
                let view = self.scene.textures[slot.min(last)].view;
                if let Some(&set) = self.cull.bindless_sets.get(frame) {
                    self.write_pool_image(set, u32::saturating_from(slot), view);
                }
            }
        }
        let now = self.stream.frame;
        let depth = self.stream.retire_depth;
        while let Some(retire) = self.stream.retires.pop_due(now, depth) {
            retire.destroy(&self.hw.device, self.commands.command_pool);
        }
    }

    // Free every parked streamed-texture retire immediately. Only legal after
    // a device drain; the world-reload and drop paths call this before
    // tearing the pool down.
    pub(in crate::vulkan) fn drain_stream_retires(&mut self) {
        for retire in self.stream.retires.drain() {
            retire.destroy(&self.hw.device, self.commands.command_pool);
        }
    }
}

impl TextureUploads for VkContext {
    // Reset texture-pool `slot` to a 1x1 mid-gray placeholder.
    fn evict_texture_slot(&mut self, slot: TextureSlot) -> RenderResult<()> {
        let gray = bake::texture::TextureImage::rgba8(1, 1, vec![128, 128, 128, 255]);
        self.update_texture_slot(slot, &gray)
    }

    // Replace albedo pool `slot` with a fresh texture without stalling: submit the
    // upload, re-point each frame's pool copy as its fence retires, and park the
    // old image until nothing uses it. A pending single-copy set that samples the
    // slot forces a drain-and-rewrite instead.
    fn update_texture_slot(
        &mut self,
        slot: TextureSlot,
        image: &bake::texture::TextureImage,
    ) -> RenderResult<()> {
        let slot = slot.index();
        if slot >= self.scene.textures.len() {
            return Err(RenderError::Other(format!(
                "update_texture_slot: slot {} out of range (pool size {})",
                slot,
                self.scene.textures.len()
            )));
        }
        let ctx = GpuUploadContext {
            alloc: &self.hw.alloc,
            device: &self.hw.device,
            command_pool: self.commands.command_pool,
            queue: self.hw.graphics_queue,
        };
        if self.streamed_slot_needs_drain(slot) {
            self.wait_idle();
            let img = upload_texture_image(&ctx, image)?;
            // Rewrite every descriptor before destroying the old view, so no set ever names
            // a destroyed view.
            let old = std::mem::replace(&mut self.scene.textures[slot], img);
            self.rewrite_texture_slot(slot);
            // The full rewrite covered every per-frame pool copy, so any
            // propagation queued for this slot is already satisfied.
            self.stream.pool_rewrites.remove(slot);
            drop(old);
            return Ok(());
        }
        let (img, in_flight) = upload_texture_image_deferred(&ctx, image)?;
        let old = std::mem::replace(&mut self.scene.textures[slot], img);
        self.stream.pool_rewrites.queue(slot);
        self.stream.retires.push(
            self.stream.frame,
            StreamedUploadRetire {
                _image: old,
                _staging: in_flight.staging,
                cmd: in_flight.cmd,
            },
        );
        Ok(())
    }

    // Replace the color-grading LUT on hot reload: wait idle, upload, re-point every
    // composite set's binding 2, then drop the old image.
    fn update_color_lut(&mut self, size: u32, data: &[u8]) -> RenderResult<()> {
        self.wait_idle();
        let new_lut = super::super::texture::upload_color_lut(
            &GpuUploadContext {
                alloc: &self.hw.alloc,
                device: &self.hw.device,
                command_pool: self.commands.command_pool,
                queue: self.hw.graphics_queue,
            },
            size,
            data,
        )?;
        // Rewrite composite descriptors before destroying the old image; see
        // the texture-pool rewires above for the rationale.
        let new_view = new_lut.view;
        let old = std::mem::replace(&mut self.scene.color_lut, new_lut);
        for &set in &self.composite.sets {
            SetWrites::new(set)
                .sampled_image(2, new_view)
                .apply(&self.hw.device);
        }
        drop(old);
        Ok(())
    }

    // Swap the IBL cubes on hot reload and re-point every set that captured the old
    // views (global, planar, probe capture faces, raymarch), and refresh
    // `prefilter_mip_count`. Vulkan uploads mint new views, so every set is
    // rewritten.
    fn update_environment_map(&mut self, payload: &[u8]) -> RenderResult<()> {
        let view = bake::environment_map::deserialize(payload)
            .map_err(|e| RenderError::Other(format!("envmap hot-reload payload malformed: {e}")))?;
        self.wait_idle();
        let new_env = super::super::texture::upload_environment_map(
            &GpuUploadContext {
                alloc: &self.hw.alloc,
                device: &self.hw.device,
                command_pool: self.commands.command_pool,
                queue: self.hw.graphics_queue,
            },
            view.irradiance_face,
            view.irradiance_bytes,
            view.prefilter_face,
            &view.prefilter_mip_bytes,
        )?;
        let new_mip_count = new_env.prefilter_mip_count;
        // Rewrite every set before destroying the previous cubes; see the
        // texture-pool rewires above for the rationale.
        let mut old = std::mem::replace(&mut self.scene.env_map, new_env);
        // An atmosphere writes both cubes; the map is held for a world that
        // drops it.
        if self.atmosphere.is_some() {
            std::mem::swap(&mut self.scene.env_map, &mut old);
        } else {
            self.scene.prefilter_mip_count = new_mip_count;
        }
        let new_irradiance_view = self.scene.env_map.irradiance.view;
        let new_prefilter_view = self.scene.env_map.prefilter.view;
        self.rewrite_global_binding(IRRADIANCE_CUBE_BINDING);
        self.rewrite_global_binding(PREFILTER_CUBE_BINDING);
        self.rewrite_probe_capture_views();
        // The raymarch view sets sample both cubes outside the global set. The
        // SSR resolve reads the prefilter cube per frame, so it needs no
        // re-point.
        if let Some(rm) = self.raymarch.as_ref() {
            rm.rewire_ibl_cubes(&self.hw.device, new_irradiance_view, new_prefilter_view);
        }
        drop(old);
        Ok(())
    }
}
