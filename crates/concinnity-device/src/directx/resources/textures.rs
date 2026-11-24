//! Texture-pool slot updates for `DxContext`: streamed and hot-reloaded albedo /
//! normal-map slots, the IBL environment map and the color-grading LUT, plus
//! the runtime clone of a static draw object (which reuses the source's
//! descriptors).

use concinnity_engine::bake;
use concinnity_engine::gfx::render_types::TextureSlot;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::backend::{RenderBackend, TextureUploads};
use windows::Win32::Graphics::Direct3D12::*;

use super::super::context::DxContext;
use super::super::texture::{
    EnvironmentMapDescriptors, EnvironmentMapPayload, StreamedUploadRetire, upload_color_lut,
    upload_environment_map, upload_texture_image, upload_texture_image_deferred, write_texture_srv,
};

impl DxContext {
    // CPU descriptor handle for CBV/SRV/UAV heap `slot`.
    fn srv_slot_cpu(&self, slot: usize) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        // SAFETY: a property query on a live descriptor heap; it only reads.
        let base = unsafe {
            self.descriptors
                .srv_heap
                .GetCPUDescriptorHandleForHeapStart()
        };
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: base.ptr + slot * self.descriptors.srv_descriptor_size,
        }
    }

    // Re-point every per-frame flat-pool copy that samples texture-pool `slot`.
    // The swapped resource has exactly one descriptor per frame copy (index ==
    // its handle), shared by albedo + normal sampling and by the RT hit shader,
    // so one re-point per copy refreshes every consumer at once.
    fn rewrite_bound_texture_srvs(&self, slot: usize) {
        let resource = &self.scene.textures[slot];
        for f in 0..self.hw.frames() {
            write_texture_srv(
                &self.hw.device,
                resource,
                self.srv_slot_cpu(self.flat_pool_slot(f, slot)),
            );
        }
    }

    // Heap slot of pool index `slot` in frame `frame`'s flat-pool copy.
    fn flat_pool_slot(&self, frame: usize, slot: usize) -> usize {
        self.descriptors.layout.flat_pool_base_slot + frame * self.descriptors.flat_pool_len + slot
    }

    // Re-point every SRV that samples texture-pool `slot`. Only legal under a
    // device drain.
    fn rewrite_texture_slot(&self, slot: usize) {
        self.rewrite_bound_texture_srvs(slot);
    }

    // Whether replacing a pool slot must drain the device first: only when nothing
    // is GPU-driven, since the flat-pool copies propagate per frame.
    fn streamed_slot_needs_drain(&self) -> bool {
        !(self.cull.main_bindless_pso.is_some() && self.state.draw.record_count() > 0)
    }

    // After this slot's fence wait: re-point its flat-pool copy at swapped slots
    // and release retires whose fence has signaled.
    pub(in crate::directx) fn apply_streamed_texture_rewrites(&mut self, frame: usize) {
        self.stream.frame += 1;
        if !self.stream.pool_rewrites.is_empty() {
            let last = self.scene.textures.len().saturating_sub(1);
            for slot in self.stream.pool_rewrites.begin_frame() {
                let resource = &self.scene.textures[slot.min(last)];
                write_texture_srv(
                    &self.hw.device,
                    resource,
                    self.srv_slot_cpu(self.flat_pool_slot(frame, slot)),
                );
            }
        }
        self.stream
            .retires
            .collect(self.stream.frame, self.stream.retire_depth);
    }
}

impl TextureUploads for DxContext {
    // Reset pool `slot` to a 1x1 mid-gray placeholder until it streams back,
    // distinct from the white no-texture fallback.
    fn evict_texture_slot(&mut self, slot: TextureSlot) -> RenderResult<()> {
        let gray = bake::texture::TextureImage::rgba8(1, 1, vec![128, 128, 128, 255]);
        self.update_texture_slot(slot, &gray)
    }

    // Replace pool `slot` with a fresh texture without stalling: submit the upload,
    // re-point the build-time SRV pairs now and each frame's flat-pool copy as its
    // fence retires, and park the old resource until nothing uses it. A drain is
    // needed only per `streamed_slot_needs_drain`.
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
        if self.streamed_slot_needs_drain() {
            self.wait_idle();
            let texture = upload_texture_image(&self.hw.alloc, image)?;
            self.scene.textures[slot] = texture;
            self.rewrite_texture_slot(slot);
            // The full rewrite covered every flat-pool copy, so any propagation
            // queued for this slot is already satisfied.
            self.stream.pool_rewrites.remove(slot);
            return Ok(());
        }
        let (texture, in_flight) = upload_texture_image_deferred(&self.hw.alloc, image)?;
        let old = std::mem::replace(&mut self.scene.textures[slot], texture);
        self.stream.pool_rewrites.queue(slot);
        self.stream.retires.push(
            self.stream.frame,
            StreamedUploadRetire {
                texture: old,
                upload: in_flight.upload,
                allocator: in_flight.allocator,
                cmd: in_flight.cmd,
            },
        );
        Ok(())
    }

    // Replace the color-grading LUT on hot reload in the SRV slot the composite
    // already binds, after waiting idle.
    fn update_color_lut(&mut self, size: u32, data: &[u8]) -> RenderResult<()> {
        self.wait_idle();
        let srv_cpu = self.scene.color_lut.srv_cpu;
        let srv_gpu = self.scene.color_lut.srv_gpu;
        let new_lut = upload_color_lut(&self.hw.alloc, size, data, srv_cpu, srv_gpu)?;
        self.scene.color_lut = new_lut;
        Ok(())
    }

    // Swap the IBL cubes on hot reload into the same SRV slots [1] and [2], after
    // waiting idle; the new mip count reaches the next frame's view uniforms.
    fn update_environment_map(&mut self, payload: &[u8]) -> RenderResult<()> {
        let view = bake::environment_map::deserialize(payload)
            .map_err(|e| RenderError::Other(format!("envmap hot-reload payload malformed: {e}")))?;
        self.wait_idle();
        let irr_srv_cpu = self.scene.env_map.irradiance.srv_cpu;
        let irr_srv_gpu = self.scene.env_map.irradiance.srv_gpu;
        let pre_srv_cpu = self.scene.env_map.prefilter.srv_cpu;
        let pre_srv_gpu = self.scene.env_map.prefilter.srv_gpu;
        let new_env = upload_environment_map(
            &self.hw.alloc,
            EnvironmentMapPayload {
                irradiance_face: view.irradiance_face,
                irradiance_bytes: view.irradiance_bytes,
                prefilter_face: view.prefilter_face,
                mip_bytes: &view.prefilter_mip_bytes,
            },
            EnvironmentMapDescriptors {
                irr_srv_cpu,
                irr_srv_gpu,
                pre_srv_cpu,
                pre_srv_gpu,
            },
        )?;
        self.scene.env_map = new_env;
        // An atmosphere writes the environment's cubes; the map is held for a
        // world that drops it.
        if let Some(atmosphere) = self.atmosphere.as_ref() {
            atmosphere.reclaim_environment_slots(&self.hw.device);
            if let Some((_, mips)) = atmosphere.prefilter() {
                self.scene.env_map.prefilter_mip_count = mips;
            }
        }

        Ok(())
    }
}
