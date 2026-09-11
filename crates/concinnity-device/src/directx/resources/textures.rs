// src/directx/resources/textures.rs
//
// Texture-pool slot updates for DxContext: streamed and hot-reloaded albedo /
// normal-map slots, the IBL environment map and the color-grading LUT, plus
// the runtime clone of a static draw object (which reuses the source's
// descriptors).

use windows::Win32::Graphics::Direct3D12::*;

use crate::gfx::render_types::*;

use super::super::context::*;
use super::super::texture::*;

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
        let resource = &self.descriptors.textures[slot];
        for f in 0..FRAMES {
            write_texture_srv(
                &self.device,
                resource,
                self.srv_slot_cpu(self.flat_pool_slot(f, slot)),
            );
        }
    }

    // Heap slot of pool index `slot` in frame `frame`'s flat-pool copy.
    fn flat_pool_slot(&self, frame: usize, slot: usize) -> usize {
        self.descriptors.flat_pool_base_slot + frame * self.descriptors.flat_pool_len + slot
    }

    // Re-point every SRV that samples texture-pool `slot`. Only legal under a
    // device drain.
    fn rewrite_texture_slot(&self, slot: usize) {
        self.rewrite_bound_texture_srvs(slot);
    }

    // Whether replacing pool `slot` must drain the device first: true when an
    // SRV that samples the slot may be dereferenced by pending command lists
    // AND cannot wait for the per-frame propagation. The flat-pool copies
    // propagate per frame, so only a world with nothing to GPU-drive (which
    // draws nothing) answers yes.
    fn streamed_slot_needs_drain(&self) -> bool {
        !(self.cull.main_bindless_pso.is_some() && self.cull_count() > 0)
    }

    // Replace texture-pool `slot` with a freshly decoded texture.
    //
    // The asset-streaming subsystem calls this to bring a texture resident
    // after init. Like Vulkan -- and unlike Metal, whose bind paths re-read
    // the texture pool every frame -- the D3D12 per-object / per-cluster SRVs
    // are baked into the descriptor heap at init, so a streamed swap must
    // rewrite every heap slot that samples this pool index (as an albedo or a
    // normal map). The streaming fast path never stalls the device: the
    // upload is submitted without waiting (the in-order queue executes it
    // before any later frame's lists), the build-time pairs are re-pointed
    // immediately (undereferenced while the bindless pass drives every draw),
    // the per-frame flat-pool copies re-point one per frame as their fences
    // retire, and the old resource plus upload transients are parked on
    // `stream.retires` until every consumer provably moved off them. When a
    // pending-referenced SRV samples the slot (see `streamed_slot_needs_drain`)
    // the swap instead drains the device and rewrites everything in place,
    // matching the hot-reload paths below.
    pub(crate) fn update_texture_slot(
        &mut self,
        slot: usize,
        image: &crate::bake::texture::TextureImage,
    ) -> Result<(), String> {
        if slot >= self.descriptors.textures.len() {
            return Err(format!(
                "update_texture_slot: slot {} out of range (pool size {})",
                slot,
                self.descriptors.textures.len()
            ));
        }
        if self.streamed_slot_needs_drain() {
            self.wait_idle();
            let texture = upload_texture_image(&self.alloc, image)?;
            self.descriptors.textures[slot] = texture;
            self.rewrite_texture_slot(slot);
            // The full rewrite covered every flat-pool copy, so any propagation
            // queued for this slot is already satisfied.
            self.stream.pool_rewrites.remove(slot);
            return Ok(());
        }
        let (texture, in_flight) = upload_texture_image_deferred(&self.alloc, image)?;
        let old = std::mem::replace(&mut self.descriptors.textures[slot], texture);
        self.stream.pool_rewrites.queue(slot);
        // `+ 1`: the swap lands between frames, after the previous frame's
        // submit, so the first frame fence that covers the upload submission
        // is the one signaled by the NEXT draw -- waited FRAMES ticks after
        // that draw's own tick.
        self.stream
            .retires
            .push(super::super::texture::StreamedUploadRetire {
                texture: old,
                upload: in_flight.upload,
                allocator: in_flight.allocator,
                cmd: in_flight.cmd,
                retire_at: self.stream.frame + FRAMES as u64 + 1,
            });
        Ok(())
    }

    // Per-frame streamed-texture upkeep, called at the top of `draw_frame`
    // right after frame slot `frame`'s fence wait: re-point this frame's
    // flat-pool copy at any swapped slots (legal now -- the wait retired every
    // list that dereferences this copy), then release retires whose covering
    // fence has signaled (dropping the entry releases the COM references).
    pub(in crate::directx) fn apply_streamed_texture_rewrites(&mut self, frame: usize) {
        self.stream.frame += 1;
        if !self.stream.pool_rewrites.is_empty() {
            let last = self.descriptors.textures.len().saturating_sub(1);
            for slot in self.stream.pool_rewrites.begin_frame() {
                let resource = &self.descriptors.textures[slot.min(last)];
                write_texture_srv(
                    &self.device,
                    resource,
                    self.srv_slot_cpu(self.flat_pool_slot(frame, slot)),
                );
            }
        }
        let now = self.stream.frame;
        self.stream.retires.retain(|r| r.retire_at > now);
    }

    // Reset texture-pool `slot` to a 1x1 mid-gray placeholder.
    //
    // Used by the asset-streaming subsystem to mark a slot whose texture is
    // not yet resident; a later `update_texture_slot` brings the real texture
    // back. The gray is distinct from the white no-texture fallback so a
    // not-yet-streamed slot reads differently under inspection.
    pub(crate) fn evict_texture_slot(&mut self, slot: usize) -> Result<(), String> {
        let gray = crate::bake::texture::TextureImage::rgba8(1, 1, vec![128, 128, 128, 255]);
        self.update_texture_slot(slot, &gray)
    }

    // Replace the live color-grading LUT with a fresh `size³` RGBA8 payload.
    // Driven by asset hot-reload (`cn debug` only) when the file-backed
    // `ColorLut` source is saved. Reuses the SRV heap slot the composite pass
    // already binds, so the new texture is picked up on the next `draw_frame`
    // with no pipeline or descriptor-table change. `wait_idle` first
    // guarantees no in-flight command list still references the old texture
    // (or the now-stale SRV) before it is overwritten and dropped. Mirrors
    // `MtlContext::update_color_lut`.
    pub(crate) fn update_color_lut(&mut self, size: u32, data: &[u8]) -> Result<(), String> {
        self.wait_idle();
        let srv_cpu = self.color_lut.srv_cpu;
        let srv_gpu = self.color_lut.srv_gpu;
        let new_lut = upload_color_lut(&self.alloc, size, data, srv_cpu, srv_gpu)?;
        self.color_lut = new_lut;
        Ok(())
    }

    // Swap the live IBL cubemap pair for a freshly precomputed envmap payload.
    // Driven by asset hot-reload (`cn debug` only). Re-uploads into the same
    // SRV heap slots [1] (irradiance) + [2] (prefilter) the init path wrote,
    // so every pipeline that references those slots keeps working without a
    // descriptor-table rebind. The new payload may declare different mip /
    // face sizes than the original; `EnvironmentMapTextures` is replaced
    // wholesale and the next frame's `ViewUniforms` picks up the new
    // `prefilter_mip_count` from `self.env_map`. `wait_idle` first guarantees
    // no in-flight command list still references the old cubes (or the
    // now-stale SRVs) before they are overwritten and dropped. Mirrors
    // `MtlContext::update_environment_map`.
    pub(crate) fn update_environment_map(&mut self, payload: &[u8]) -> Result<(), String> {
        let view = crate::bake::environment_map::deserialize(payload)
            .map_err(|e| format!("envmap hot-reload payload malformed: {e}"))?;
        self.wait_idle();
        let irr_srv_cpu = self.env_map.irradiance.srv_cpu;
        let irr_srv_gpu = self.env_map.irradiance.srv_gpu;
        let pre_srv_cpu = self.env_map.prefilter.srv_cpu;
        let pre_srv_gpu = self.env_map.prefilter.srv_gpu;
        let new_env = upload_environment_map(
            &self.alloc,
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
        self.env_map = new_env;
        Ok(())
    }

    // Append a new draw object that re-uses an existing slot's geometry
    // region (vertex / index offsets, base_vertex, LOD alternates) with a
    // fresh model matrix, texture / normal-map slots, material, and cull
    // distance. Driven by `world.jsonl` hot-reload (`cn debug` only) when a
    // newly authored Prop references a Mesh / Model already present in the
    // init world. The clone is non-cullable (sentinel AABB) and joins
    // `draw.always` since the init-time BVH cannot refit; the dynamically added
    // prop is drawn every frame, like a streamed `VoxelWorld` chunk -- and
    // through the same runtime reserve in the cull records, so it needs no
    // descriptors of its own. Mirrors `MtlContext::clone_static_draw_object`.
    pub(crate) fn clone_static_draw_object(
        &mut self,
        src_draw_idx: usize,
        model: [[f32; 4]; 4],
        dst: crate::gfx::draw_slot::SlotAlloc,
    ) -> Result<(), String> {
        if runtime_reserve_full(&self.draw.objects, self.draw.n_objects, self.draw.n_runtime) {
            return Err(format!(
                "clone_static_draw_object: the runtime draw reserve ({}) is full",
                self.draw.n_runtime
            ));
        }
        let src = self.draw.objects.get(src_draw_idx).ok_or_else(|| {
            format!(
                "clone_static_draw_object: src draw {} out of range",
                src_draw_idx
            )
        })?;
        // A runtime spawn duplicates the template, swapping only the transform:
        // copy the source's material, pool slots, and cull distance.
        let texture_slot = src.texture_slot;
        let normal_map_slot = src.normal_map_slot;
        let material = src.material;
        let cull_distance = src.cull_distance;
        let obj = DrawObject {
            vertex_offset: src.vertex_offset,
            vertex_count: src.vertex_count,
            index_offset: src.index_offset,
            index_count: src.index_count,
            base_vertex: src.base_vertex,
            geometry_generation: src.geometry_generation,
            model,
            texture_slot,
            normal_map_slot,
            material,
            visible: true,
            resident: true,
            // Sentinel AABB so the init-time BVH cull skips the new draw:
            // it joins `draw.always` and is drawn every frame regardless of
            // camera position. Matches the runtime-streamed chunk pattern.
            bb_min: [f32::NAN; 3],
            bb_max: [f32::NAN; 3],
            cull_distance,
            lod_alternates: src.lod_alternates.clone(),
            shader_bucket: src.shader_bucket,
        };

        // Write at the engine-allocated destination slot.
        match dst {
            crate::gfx::draw_slot::SlotAlloc::Reuse(slot) => {
                self.draw.objects[slot] = obj;
                // The slot's model-history entry belongs to the prior occupant,
                // so the clone reprojects through its own transform for one
                // frame rather than ghosting from that occupant's.
                self.model_history.borrow_mut().reoccupy_draw(slot);
            }
            crate::gfx::draw_slot::SlotAlloc::Append(slot) => {
                debug_assert_eq!(
                    slot,
                    self.draw.objects.len(),
                    "appended draw slot must match the draw-object count"
                );
                self.draw.objects.push(obj);
                self.model_history.borrow_mut().reoccupy_draw(slot);
            }
        }
        // The cloned prop joins the RT-relevant draw set; the next RT update folds
        // it into the BVH (it reuses the source mesh's geometry slice, so only
        // this clone's BLAS is built).
        self.rt_topology_dirty = true;
        Ok(())
    }
}
