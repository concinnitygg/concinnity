// src/vulkan/resources/textures.rs
//
// Texture-pool slot management for VkContext: per-object / bindless / cluster /
// skinned / chunk descriptor rewires when an albedo or normal-map slot is
// streamed in or evicted. Mirrors the Metal pattern of "the texture pool gets
// re-read every frame," except Vulkan bakes texture *views* into descriptor
// sets at init, so a slot swap must walk every set that samples this slot.

use ash::vk;

use crate::gfx::render_types::NO_NORMAL_MAP_SLOT;

use super::super::context::*;
use super::super::texture::{
    GpuUploadContext, StreamedUploadRetire, upload_texture_image, upload_texture_image_deferred,
};

impl VkContext {
    pub(in crate::vulkan) fn write_object_image(
        &self,
        set: vk::DescriptorSet,
        binding: u32,
        view: vk::ImageView,
    ) {
        let info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)
            .sampler(self.linear_sampler.handle());
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(binding)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&info));
        // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every
        // set and resource it names belongs to this device.
        unsafe {
            self.device
                .update_descriptor_sets(std::slice::from_ref(&write), &[])
        };
    }

    // The image view a `normal_map_slot` samples for a baked per-object binding:
    // a real normal map is a texture in the shared pool at its own slot;
    // `NO_NORMAL_MAP_SLOT` selects the flat-normal fallback (the sole entry of
    // `normal_map_textures`).
    pub(in crate::vulkan) fn normal_pool_view(&self, normal_map_slot: usize) -> vk::ImageView {
        if normal_map_slot == NO_NORMAL_MAP_SLOT {
            self.normal_map_textures[0].view
        } else {
            let last = self.textures.len().saturating_sub(1);
            self.textures[normal_map_slot.min(last)].view
        }
    }

    // Re-point bindless texture-pool element `index` of `set` to `view`.
    // Keeps the bindless texture pool in sync with a streamed texture swap; the
    // pool is one handle-indexed image set (albedo + normal maps share it).
    fn write_pool_image(&self, set: vk::DescriptorSet, index: u32, view: vk::ImageView) {
        let info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)
            .sampler(self.linear_sampler.handle());
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&info));
        // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every
        // set and resource it names belongs to this device.
        unsafe {
            self.device
                .update_descriptor_sets(std::slice::from_ref(&write), &[])
        };
    }

    // Whether a draw samples texture `slot` as its normal map, given the pool
    // clamp. A real normal is a texture at its own handle; `NO_NORMAL_MAP_SLOT`
    // samples the never-streamed flat-normal fallback and matches no streamed
    // slot.
    fn normal_is_slot(&self, nms: usize, slot: usize) -> bool {
        let last = self.textures.len().saturating_sub(1);
        nms != NO_NORMAL_MAP_SLOT && nms.min(last) == slot
    }

    // Re-point the per-object / per-cluster / per-skinned / chunk descriptor
    // sets that sample texture-pool `slot` at the (just-swapped)
    // `self.textures[slot]` view. Albedo and normal maps share one pool, so a
    // streamed texture may be an albedo for one draw (binding 0) and a normal
    // map for another (binding 1); this re-points both wherever they resolve
    // to `slot`. These sets are only ever bound by the legacy (non-bindless)
    // draw loops, so while the bindless pass drives every draw they are not
    // referenced by pending command buffers and rewriting them needs no drain.
    fn rewrite_legacy_object_sets(&self, slot: usize) {
        let last = self.textures.len().saturating_sub(1);
        let view = self.textures[slot].view;
        for (set, obj) in self
            .descriptors
            .object_sets
            .iter()
            .zip(self.draw.objects.iter())
        {
            if obj.texture_slot.min(last) == slot {
                self.write_object_image(*set, 0, view);
            }
            if self.normal_is_slot(obj.normal_map_slot, slot) {
                self.write_object_image(*set, 1, view);
            }
        }
        for (set, cluster) in self
            .instanced
            .object_sets
            .iter()
            .zip(self.instanced.clusters.iter())
        {
            if cluster.texture_slot.min(last) == slot {
                self.write_object_image(*set, 0, view);
            }
            if self.normal_is_slot(cluster.normal_map_slot, slot) {
                self.write_object_image(*set, 1, view);
            }
        }
        for (set, obj) in self
            .skinned
            .object_sets
            .iter()
            .zip(self.skinned.draw_objects.iter())
        {
            if obj.texture_slot.min(last) == slot {
                self.write_object_image(*set, 0, view);
            }
            if self.normal_is_slot(obj.normal_map_slot, slot) {
                self.write_object_image(*set, 1, view);
            }
        }
        // The shared VoxelWorld chunk set bakes its (albedo, normal) views at
        // init too, and is not part of object/cluster/skinned, so re-point it.
        if let Some(set) = self.chunk_stream.object_set {
            if self.chunk_stream.texture_slot == Some(slot) {
                self.write_object_image(set, 0, view);
            }
            if let Some(chunk_nms) = self.chunk_stream.normal_map_slot
                && self.normal_is_slot(chunk_nms, slot)
            {
                self.write_object_image(set, 1, view);
            }
        }
    }

    // Re-point every descriptor that samples texture-pool `slot` and may be
    // referenced by in-flight command buffers: the per-frame bindless pool
    // copies plus the decal / clone / particle sets. Only legal under a device
    // drain (the fallback streaming path and the `cn debug` hot-reload paths).
    fn rewrite_bound_texture_sets(&self, slot: usize) {
        let view = self.textures[slot].view;
        // The bindless pool addresses each texture at pool index == its handle,
        // shared by albedo and normal sampling, so one re-point covers both.
        for &set in &self.cull.bindless_sets {
            self.write_pool_image(set, slot as u32, view);
        }
        // Per-decal albedo descriptors. Walk the decal-side slot tracker;
        // a world with no decals pays nothing here.
        self.rewrite_decal_albedo_slot(slot);
        // Runtime clones (from `clone_static_draw_object`) carry their own
        // (albedo, normal) descriptor sets, so re-point those that sample this
        // slot as either.
        let last = self.textures.len().saturating_sub(1);
        for (offset, &clone_tex) in self.clone.texture_slots.iter().enumerate() {
            let Some(&set) = self.clone.object_sets.get(offset) else {
                continue;
            };
            if clone_tex.min(last) == slot {
                self.write_object_image(set, 0, view);
            }
            if let Some(&clone_nms) = self.clone.normal_map_slots.get(offset)
                && self.normal_is_slot(clone_nms, slot)
            {
                self.write_object_image(set, 1, view);
            }
        }
        // Particle emitters sample an albedo from the texture pool into their
        // own render set, so re-point any whose source slot is this one.
        self.rewrite_particle_albedo_slot(slot);
    }

    // Re-point every descriptor that samples texture-pool `slot`. Only legal
    // under a device drain. An in-flight reflection-probe bake needs no arm
    // here: each bake face snapshots the live pool into its own set right
    // before recording.
    fn rewrite_texture_slot(&self, slot: usize) {
        self.rewrite_legacy_object_sets(slot);
        self.rewrite_bound_texture_sets(slot);
    }

    // Whether replacing pool `slot` must drain the device first: true when a
    // descriptor that samples the slot may be referenced by pending command
    // buffers AND cannot wait for the per-frame propagation. The bindless pool
    // copies propagate per frame slot; everything the legacy loops bind is
    // rewritten immediately while those loops are inert. What remains are the
    // single-copy sets of passes that run regardless of the bindless path
    // (decal / particle), the runtime-clone sets, and whole worlds where the
    // bindless pass is off (custom-shader; every legacy set is then live).
    fn streamed_slot_needs_drain(&self, slot: usize) -> bool {
        let bindless_active = self.cull.bindless_pipeline.is_some() && self.cull_count() > 0;
        if !bindless_active {
            return true;
        }
        let last = self.textures.len().saturating_sub(1);
        let clone_samples_slot = self.clone.slot_by_draw_idx.values().any(|&offset| {
            self.clone
                .texture_slots
                .get(offset)
                .is_some_and(|&t| t.min(last) == slot)
                || self
                    .clone
                    .normal_map_slots
                    .get(offset)
                    .is_some_and(|&n| self.normal_is_slot(n, slot))
        });
        clone_samples_slot || self.decal_samples_slot(slot) || self.particle_samples_slot(slot)
    }

    // Replace albedo texture-pool `slot` with a freshly decoded texture.
    //
    // The streaming fast path never stalls the device: the upload is
    // submitted without waiting (later submissions on the queue order after
    // its final barrier, so any frame recorded from here on samples it
    // safely), the legacy object sets are re-pointed immediately (unbound
    // while the bindless pass drives every draw), the per-frame bindless pool
    // copies re-point one per frame as their fences retire, and the old image
    // plus upload transients are parked on `stream.retires` until every
    // consumer provably moved off them. When a pending-referenced single-copy
    // set samples the slot (see `streamed_slot_needs_drain`) the swap instead
    // drains the device and rewrites everything in place, matching the
    // hot-reload paths below.
    pub(crate) fn update_texture_slot(
        &mut self,
        slot: usize,
        image: &crate::build::texture::TextureImage,
    ) -> crate::gfx::error::RenderResult<()> {
        if slot >= self.textures.len() {
            return Err(format!(
                "update_texture_slot: slot {} out of range (pool size {})",
                slot,
                self.textures.len()
            )
            .into());
        }
        let ctx = GpuUploadContext {
            alloc: &self.alloc,
            device: &self.device,
            command_pool: self.commands.command_pool,
            queue: self.graphics_queue,
        };
        if self.streamed_slot_needs_drain(slot) {
            self.wait_idle();
            let img = upload_texture_image(&ctx, image)?;
            // Swap in the new image, then rewrite every descriptor that
            // samples this slot BEFORE destroying the old view. The reverse
            // order left a brief window where descriptor sets referenced an
            // already-destroyed VkImageView, which validation layers and some
            // drivers flag even though vkUpdateDescriptorSets is write-only.
            let old = std::mem::replace(&mut self.textures[slot], img);
            self.rewrite_texture_slot(slot);
            // The full rewrite covered every per-frame pool copy, so any
            // propagation queued for this slot is already satisfied.
            self.stream.pool_rewrites.remove(slot);
            drop(old);
            return Ok(());
        }
        let (img, in_flight) = upload_texture_image_deferred(&ctx, image)?;
        let old = std::mem::replace(&mut self.textures[slot], img);
        self.rewrite_legacy_object_sets(slot);
        self.stream.pool_rewrites.queue(slot);
        // `+ 1`: the swap lands between frames, after the previous frame's
        // submit, so the first frame fence that covers the upload submission
        // is the one signalled by the NEXT draw -- waited `frames_in_flight`
        // ticks after that draw's own tick.
        self.stream.retires.push(StreamedUploadRetire {
            _image: old,
            _staging: in_flight.staging,
            cmd: in_flight.cmd,
            retire_at: self.stream.frame + self.frames_in_flight as u64 + 1,
        });
        Ok(())
    }

    // Reset texture-pool `slot` to a 1x1 mid-grey placeholder.
    pub(crate) fn evict_texture_slot(&mut self, slot: usize) -> Result<(), String> {
        let grey = crate::build::texture::TextureImage::rgba8(1, 1, vec![128, 128, 128, 255]);
        Ok(self.update_texture_slot(slot, &grey)?)
    }

    // Per-frame streamed-texture upkeep, called at the top of `draw_frame`
    // right after frame slot `frame`'s fence wait: re-point this slot's
    // bindless pool copy at any swapped slots (legal now -- the wait retired
    // every command buffer that binds this copy), then free retires whose
    // covering fence has signalled.
    pub(in crate::vulkan) fn apply_streamed_texture_rewrites(&mut self, frame: usize) {
        self.stream.frame += 1;
        if !self.stream.pool_rewrites.is_empty() {
            let last = self.textures.len().saturating_sub(1);
            for slot in self.stream.pool_rewrites.begin_frame() {
                let view = self.textures[slot.min(last)].view;
                if let Some(&set) = self.cull.bindless_sets.get(frame) {
                    self.write_pool_image(set, slot as u32, view);
                }
            }
        }
        if !self.stream.retires.is_empty() {
            let now = self.stream.frame;
            let device = self.device.clone();
            let pool = self.commands.command_pool;
            let mut i = 0;
            while i < self.stream.retires.len() {
                if self.stream.retires[i].retire_at <= now {
                    self.stream.retires.swap_remove(i).destroy(&device, pool);
                } else {
                    i += 1;
                }
            }
        }
    }

    // Free every parked streamed-texture retire immediately. Only legal after
    // a device drain; the world-reload and drop paths call this before
    // tearing the pool down.
    pub(in crate::vulkan) fn drain_stream_retires(&mut self) {
        let device = self.device.clone();
        let pool = self.commands.command_pool;
        for retire in self.stream.retires.drain(..) {
            retire.destroy(&device, pool);
        }
    }

    // Replace the live colour-grading LUT with a fresh `size³` RGBA8 payload.
    // Driven by asset hot-reload (`cn debug` only) when the file-backed
    // `ColorLut` source is saved. `wait_idle` first guarantees no in-flight
    // command buffer still references the old image. Builds the replacement
    // via the same `upload_color_lut` the init path uses, rewrites every
    // composite descriptor set's binding 2 to point at the new view, then
    // drops the previous image; same write-then-destroy order as the
    // texture-pool rewires above to keep validation layers happy. Mirrors
    // `DxContext::update_color_lut` / `MtlContext::update_color_lut`. Reached
    // only through the bin's `cn debug` runtime-mutation path (dead in the FFI
    // lib, live in the bin).
    #[allow(dead_code)]
    pub(crate) fn update_color_lut(&mut self, size: u32, data: &[u8]) -> Result<(), String> {
        self.wait_idle();
        let new_lut = super::super::texture::upload_color_lut(
            &GpuUploadContext {
                alloc: &self.alloc,
                device: &self.device,
                command_pool: self.commands.command_pool,
                queue: self.graphics_queue,
            },
            size,
            data,
        )?;
        // Rewrite composite descriptors before destroying the old image; see
        // the texture-pool rewires above for the rationale.
        let new_view = new_lut.view;
        let old = std::mem::replace(&mut self.color_lut, new_lut);
        for &set in &self.composite.sets {
            let info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(new_view)
                .sampler(self.composite.sampler.handle());
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&info));
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe {
                self.device
                    .update_descriptor_sets(std::slice::from_ref(&write), &[])
            };
        }
        drop(old);
        Ok(())
    }

    // Swap the live IBL cubemap pair for a freshly precomputed envmap payload.
    // Driven by asset hot-reload (`cn debug` only). Decodes the byte stream
    // emitted by `gfx::build::environment_map::serialise`, then re-uploads
    // the irradiance + prefilter cubes via the same `upload_environment_map`
    // the init path uses. Every consumer that captured the old image views is
    // re-pointed at the new ones: each `global_sets` entry (irradiance +
    // prefilter), the SSR resolve sets (prefilter), and the raymarch view sets
    // (both cubes). `prefilter_mip_count` is refreshed on `self` so the next
    // frame's `ViewUniforms` upload picks up the new mip count. Unlike DirectX,
    // which re-uploads into the same SRV heap slots so its consumers need no
    // re-wire, every Vulkan `upload_environment_map` mints fresh `vk::ImageView`
    // handles, so each descriptor set must be re-written. Mirrors
    // `DxContext::update_environment_map`. Reached
    // only through the bin's `cn debug` runtime-mutation path (dead in the FFI
    // lib, live in the bin).
    #[allow(dead_code)]
    pub(crate) fn update_environment_map(&mut self, payload: &[u8]) -> Result<(), String> {
        let view = crate::build::environment_map::deserialise(payload)
            .map_err(|e| format!("envmap hot-reload payload malformed: {e}"))?;
        self.wait_idle();
        let new_env = super::super::texture::upload_environment_map(
            &GpuUploadContext {
                alloc: &self.alloc,
                device: &self.device,
                command_pool: self.commands.command_pool,
                queue: self.graphics_queue,
            },
            view.irradiance_face,
            view.irradiance_bytes,
            view.prefilter_face,
            &view.prefilter_mip_bytes,
        )?;
        let new_irradiance_view = new_env.irradiance.view;
        let new_prefilter_view = new_env.prefilter.view;
        let new_mip_count = new_env.prefilter_mip_count;
        // Rewrite global sets before destroying the previous cubes; see the
        // texture-pool rewires above for the rationale.
        let old = std::mem::replace(&mut self.env_map, new_env);
        self.prefilter_mip_count = new_mip_count;
        for &set in &self.descriptors.global_sets {
            let irr_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(new_irradiance_view)
                .sampler(self.cube_sampler.handle());
            let pre_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(new_prefilter_view)
                .sampler(self.cube_sampler.handle());
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(IRRADIANCE_CUBE_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&irr_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(PREFILTER_CUBE_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&pre_info)),
            ];
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        }
        // The IBL cubes are also bound outside `global_sets`: the SSR resolve
        // sets sample the prefilter cube (set 0 binding 3) and the raymarch
        // view sets sample both cubes (bindings 4 + 5). They captured the old
        // image views too, so re-point them at the new cubes before the old
        // ones are destroyed below; otherwise the next SSR resolve / raymarch
        // draw reads a destroyed view and loses the device. Done here, not in
        // `global_sets`, because the resize path re-wires these the same way.
        if let Some(ssr) = self.ssr.as_ref() {
            let hdr_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|img| img.view).collect();
            // Keep the SSR resolve's G-buffer / roughness bindings on the unified
            // pre-pass per-frame views when present (empty falls back to SSR's own
            // pre-pass targets); only binding 3 (the prefilter cube) actually
            // moved, but `wire_resolve_sets` rewrites all four bindings.
            let (nd_views, rough_views) = match self.gbuffer.as_ref() {
                Some(gb) => (gb.normal_depth_views(), gb.roughness_views()),
                None => (Vec::new(), Vec::new()),
            };
            ssr.wire_resolve_sets(
                &self.device,
                &hdr_views,
                &nd_views,
                &rough_views,
                new_prefilter_view,
                self.cube_sampler.handle(),
            );
        }
        if let Some(rm) = self.raymarch.as_ref() {
            rm.rewire_ibl_cubes(
                &self.device,
                new_irradiance_view,
                new_prefilter_view,
                self.cube_sampler.handle(),
            );
        }
        // The RT-reflection sets sample the prefilter cube at binding 8 (the miss
        // fallback + the metallic/roughness IBL hit shading); re-point them too.
        if let Some(rt) = self.rt_reflections.as_ref() {
            rt.rewire_prefilter(&self.device, new_prefilter_view, self.cube_sampler.handle());
        }
        drop(old);
        Ok(())
    }
}

// Set-0 binding indices for the IBL cubemaps. Must match the bindings the
// init path writes in `vulkan/init.rs` when wiring `global_sets`. Kept as
// documentation of that layout; `init.rs` writes the literals directly.
#[allow(dead_code)]
const IRRADIANCE_CUBE_BINDING: u32 = 4;
#[allow(dead_code)]
const PREFILTER_CUBE_BINDING: u32 = 5;

impl VkContext {
    // Append a new draw object that re-uses an existing slot's geometry
    // region with a fresh model matrix, texture / normal-map slots,
    // material, and cull distance. Driven by `world.jsonl` hot-reload
    // (`cn debug` only) when a newly authored Prop references a Mesh /
    // Model already present in the init world. The clone is non-cullable
    // (sentinel AABB) and joins `draw.always` since the init-time BVH
    // cannot refit; the dynamically added prop is drawn every frame,
    // like a streamed `VoxelWorld` chunk. Allocates the clone's (albedo,
    // normal) descriptor set from a `MAX_CLONE_DRAWS`-deep pool reserved
    // at init, and records `draw_idx → clone_offset` in
    // `clone.slot_by_draw_idx` so the legacy main pass + the texture-pool
    // rewires above can find it. Mirrors
    // `DxContext::clone_static_draw_object`. Reached only through the bin's
    // `cn debug` runtime-mutation path (dead in the FFI lib, live in the bin).
    #[allow(dead_code)]
    pub(crate) fn clone_static_draw_object(
        &mut self,
        src_draw_idx: usize,
        model: [[f32; 4]; 4],
        dst: crate::gfx::draw_slot::SlotAlloc,
    ) -> Result<(), String> {
        use super::super::context::MAX_CLONE_DRAWS;

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
        let obj = crate::gfx::render_types::DrawObject {
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
            // Sentinel AABB so the init-time BVH cull skips this draw; it
            // joins `draw.always` and is drawn every frame. Matches the
            // chunk pattern.
            bb_min: [f32::NAN; 3],
            bb_max: [f32::NAN; 3],
            cull_distance,
            lod_alternates: src.lod_alternates.clone(),
            shader_bucket: src.shader_bucket,
        };

        let last_tex = self.textures.len().saturating_sub(1);
        let albedo_view = self.textures[texture_slot.min(last_tex)].view;
        let normal_view = self.normal_pool_view(normal_map_slot);

        // Reuse a vacated clone descriptor set, else allocate a fresh one up to
        // the pool cap. A reused set may still be referenced by an in-flight frame
        // (its prior occupant was drawn before being retired), so drain the GPU
        // before re-pointing it, then always overwrite its (albedo, normal)
        // bindings with this clone's views.
        let clone_offset = if let Some(offset) = self.clone.free_offsets.pop() {
            self.wait_idle();
            let set = self.clone.object_sets[offset];
            self.write_object_image(set, 0, albedo_view);
            self.write_object_image(set, 1, normal_view);
            self.clone.texture_slots[offset] = texture_slot;
            self.clone.normal_map_slots[offset] = normal_map_slot;
            offset
        } else {
            if self.clone.object_sets.len() >= MAX_CLONE_DRAWS {
                return Err(format!(
                    "clone_static_draw_object: MAX_CLONE_DRAWS ({MAX_CLONE_DRAWS}) exceeded"
                ));
            }
            // Lazily build the clone descriptor pool on first call. Sized for
            // MAX_CLONE_DRAWS (albedo, normal) sets: two samplers each.
            if self.clone.descriptor_pool.is_none() {
                let pool_sizes = [vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count((MAX_CLONE_DRAWS * 2) as u32)];
                let pool = self
                    .device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .pool_sizes(&pool_sizes)
                            .max_sets(MAX_CLONE_DRAWS as u32),
                    )
                    .map_err(|e| format!("clone descriptor pool: {e}"))?;
                self.clone.descriptor_pool = Some(pool);
            }
            let pool = self
                .clone
                .descriptor_pool
                .as_ref()
                .expect("clone descriptor pool built above")
                .handle();
            let object_set_layout = self.descriptors.object_set_layout.handle();

            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(std::slice::from_ref(&object_set_layout));
            // SAFETY: the create-info and every slice it borrows are live for the call, and each
            // handle it names belongs to this device.
            let set = unsafe { self.device.allocate_descriptor_sets(&alloc_info) }
                .map_err(|e| format!("allocate clone descriptor set: {e}"))?[0];
            self.write_object_image(set, 0, albedo_view);
            self.write_object_image(set, 1, normal_view);

            let offset = self.clone.object_sets.len();
            self.clone.object_sets.push(set);
            self.clone.texture_slots.push(texture_slot);
            self.clone.normal_map_slots.push(normal_map_slot);
            offset
        };

        // Write at the engine-allocated destination slot.
        let new_idx = match dst {
            crate::gfx::draw_slot::SlotAlloc::Reuse(slot) => {
                self.draw.objects[slot] = obj;
                // Seed the velocity prepass's previous-model snapshot so a
                // recycled slot does not ghost from the prior occupant's
                // transform for one frame. A slot past the snapshot's end (one
                // appended beyond the build-time object count) falls back to its
                // own current model in the prepass, so the guard is enough.
                if let Some(gb) = &mut self.gbuffer
                    && slot < gb.prev_models.len()
                {
                    gb.prev_models[slot] = model;
                }
                slot
            }
            crate::gfx::draw_slot::SlotAlloc::Append(slot) => {
                debug_assert_eq!(
                    slot,
                    self.draw.objects.len(),
                    "appended draw slot must match the draw-object count"
                );
                self.draw.objects.push(obj);
                self.draw.always_member.push(false);
                slot
            }
        };
        self.ensure_always_draw(new_idx);
        self.clone.slot_by_draw_idx.insert(new_idx, clone_offset);
        // The cloned prop joins the RT-relevant draw set; the next RT update folds
        // it into the BVH (it reuses the source mesh's geometry slice, so only
        // this clone's BLAS is built).
        self.rt_topology_dirty = true;
        Ok(())
    }
}
