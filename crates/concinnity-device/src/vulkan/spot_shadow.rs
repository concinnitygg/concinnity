// src/vulkan/spot_shadow.rs
//
// Spot shadow pass: one depth-only render per shadow-casting spot light into
// its layer of the spot shadow array. Structurally the cascade pass with a
// different projection source -- each slice reuses the same depth-only shadow
// render pass, pipeline, and caster sub-encoders, driven by a per-slice
// descriptor set whose `ShadowUniforms` holds that spot's light-space matrix in
// slot 0 rather than the CSM cascade set.
//
// Local lights are static, so the matrices are built once here and only the
// depth contents refresh. `spot_shadow.render_mask` (from `SpotShadowScheduler`)
// picks which slices redraw; a skipped slice keeps the depth it last rendered,
// which stays correct until a caster moves.

use ash::vk;

use crate::vulkan::owned::VkDevice;

use crate::gfx::render_types::{ShadowUniforms, SpotShadowData};
use crate::vulkan::allocator::{DeviceAllocator, PooledBuffer};
use crate::vulkan::context::{VkContext, VkSpotShadow};
use crate::vulkan::resources::alloc_descriptor_sets;
use crate::vulkan::texture::GpuImage;

// Everything `build_spot_shadow` needs from init. Grouped so the builder takes
// one parameter instead of a nine-argument list.
pub(super) struct SpotShadowBuild<'a> {
    pub alloc: &'a DeviceAllocator,
    pub instance: &'a ash::Instance,
    pub device: &'a VkDevice,
    pub physical_device: vk::PhysicalDevice,
    // The depth array, already created with one layer per shadowed spot (or the
    // 1x1 fallback when there are none).
    pub map: GpuImage,
    // The cascade pass's depth-only render pass, reused verbatim.
    pub render_pass: vk::RenderPass,
    // The one-UBO layout the shadow vertex shader binds at set 0.
    pub set_layout: vk::DescriptorSetLayout,
    pub slice_size: u32,
    pub spot_shadows: &'a [SpotShadowData],
}

// Build the spot shadow resources: per-slice framebuffers, the `SpotShadowData`
// storage buffer the forward pass indexes, and one `ShadowUniforms` slot per
// slice with a descriptor set pointing at it. All static for the world's
// lifetime; only the depth contents change per frame.
pub(super) fn build_spot_shadow(b: SpotShadowBuild<'_>) -> Result<VkSpotShadow, String> {
    let SpotShadowBuild {
        alloc,
        instance,
        device,
        physical_device,
        map,
        render_pass,
        set_layout,
        slice_size,
        spot_shadows,
    } = b;

    let framebuffers = if spot_shadows.is_empty() {
        Vec::new()
    } else {
        crate::vulkan::swapchain::create_shadow_framebuffers(device, render_pass, &map, slice_size)?
    };

    // The per-slice projections the forward pass reads. A world with no shadowed
    // spot still gets a one-element buffer: the shader never indexes it (every
    // `shadow_index` is -1) but the descriptor must still be valid.
    let data: Vec<SpotShadowData> = if spot_shadows.is_empty() {
        vec![SpotShadowData::ZERO]
    } else {
        spot_shadows.to_vec()
    };
    let data_size = std::mem::size_of_val(data.as_slice()) as u64;
    let data_buffer = alloc.create_buffer(
        data_size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    upload_records(&data_buffer, &data);

    // One `ShadowUniforms` per slice, each with the spot's matrix in
    // `light_vps[0]`, so the shared shadow vertex shader renders a spot slice by
    // pushing cascade_idx = 0. Slots are padded to the device's minimum uniform
    // buffer offset alignment so each slice's descriptor can point at its own.
    // SAFETY: a property query on a live handle; it only reads.
    let align = unsafe { instance.get_physical_device_properties(physical_device) }
        .limits
        .min_uniform_buffer_offset_alignment
        .max(1);
    let stride = (size_of::<ShadowUniforms>() as u64).div_ceil(align) * align;
    let slots = framebuffers.len().max(1) as u64;
    let ubo = alloc.create_buffer(
        stride * slots,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    if !spot_shadows.is_empty() {
        let uniforms: Vec<ShadowUniforms> = spot_shadows
            .iter()
            .map(|sd| {
                let mut u = crate::gfx::csm::empty_shadow_uniforms();
                u.light_vps[0] = sd.light_vp;
                u.active_cascades = 1;
                u
            })
            .collect();
        upload_strided(&ubo, &uniforms, stride);
    }

    // The pass's own descriptor pool: one single-UBO set per slice. Kept
    // separate from the shared pool so the slice count does not have to be
    // threaded into the main pool sizing.
    let set_count = framebuffers.len().max(1) as u32;
    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(set_count)];
    let descriptor_pool = device
        .create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(set_count),
        )
        .map_err(|e| format!("spot shadow descriptor pool: {e}"))?;

    let layouts: Vec<_> = (0..set_count).map(|_| set_layout).collect();
    let sets = alloc_descriptor_sets(device, descriptor_pool.handle(), &layouts)?;
    for (i, &set) in sets.iter().enumerate() {
        let info = vk::DescriptorBufferInfo::default()
            .buffer(ubo.buffer())
            .offset(i as u64 * stride)
            .range(size_of::<ShadowUniforms>() as u64);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&info));
        // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every
        // set and resource it names belongs to this device.
        unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
    }

    Ok(VkSpotShadow {
        map,
        framebuffers,
        slice_size,
        data_buffer,
        ubo,
        sets,
        _descriptor_pool: descriptor_pool,
        scheduler: Default::default(),
        render_mask: 0,
    })
}

// One-shot tightly packed upload of a record slice into a host-visible pooled
// buffer.
fn upload_records<T: Copy>(buffer: &PooledBuffer, records: &[T]) {
    buffer.write_slice(0, records);
}

// As `upload_records`, but places record `i` at `i * stride` so each slot can
// back its own uniform-buffer descriptor.
fn upload_strided<T: Copy>(buffer: &PooledBuffer, records: &[T], stride: u64) {
    for (i, r) in records.iter().enumerate() {
        buffer.write_val(i * stride as usize, r);
    }
}

// Push constants for the spot caster draws (80 bytes): model matrix + the
// `light_vps` index the shadow vertex shader projects through.
#[derive(Copy, Clone)]
#[repr(C)]
struct ShadowPush {
    model: [[f32; 4]; 4],
    cascade_idx: u32,
    _pad: [u32; 3],
}

// Every spot slice carries its own matrix in `light_vps[0]`.
const SPOT_SLICE_IDX: u32 = 0;

// One spot slice's draw state: the depth-only pipeline and its layout, plus the
// descriptor set holding that slice's `ShadowUniforms`.
#[derive(Clone, Copy)]
struct SpotSliceBinding {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set: vk::DescriptorSet,
}

impl VkContext {
    // One depth-only render per scheduled spot slice, into that slice's layer of
    // the array. pub(in crate::vulkan) so the render-graph executor can dispatch
    // it.
    pub(in crate::vulkan) fn encode_spot_shadow_pass(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        cam_pos: [f32; 3],
    ) {
        let (Some(pipeline), Some(layout)) = (
            self.shadow.pipeline.as_ref(),
            self.shadow.pipeline_layout.as_ref(),
        ) else {
            return;
        };
        let count = self.spot_shadow.count();
        if count == 0 {
            return;
        }

        let all = if count >= 32 {
            u32::MAX
        } else {
            (1u32 << count) - 1
        };
        // Defensive fallback to every slice if no mask was set this frame.
        let mask = if self.spot_shadow.render_mask == 0 {
            all
        } else {
            self.spot_shadow.render_mask
        };

        let sz = self.spot_shadow.slice_size;
        let extent = vk::Extent2D {
            width: sz,
            height: sz,
        };
        let clear = [vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        }];

        for slice in 0..count {
            if mask & (1u32 << slice) == 0 {
                continue;
            }
            let begin = vk::RenderPassBeginInfo::default()
                .render_pass(self.shadow.render_pass.handle())
                .framebuffer(self.spot_shadow.framebuffers[slice as usize].handle())
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .clear_values(&clear);
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe {
                self.device
                    .cmd_begin_render_pass(cmd, &begin, vk::SubpassContents::INLINE);
                // Negative height flips NDC y, matching the cascade pass and the
                // `-ndc.y` the forward sampler applies.
                let viewport = vk::Viewport {
                    x: 0.0,
                    y: sz as f32,
                    width: sz as f32,
                    height: -(sz as f32),
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                self.device.cmd_set_viewport(cmd, 0, &[viewport]);
                self.device.cmd_set_scissor(
                    cmd,
                    0,
                    &[vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent,
                    }],
                );
            }

            // Spot casters are walked on the CPU: the indirect buffer the
            // bindless cull fills is laid out per CSM cascade, so it has no
            // slots for these slices.
            self.encode_spot_casters(
                cmd,
                SpotSliceBinding {
                    pipeline: pipeline.handle(),
                    layout: layout.handle(),
                    set: self.spot_shadow.sets[slice as usize],
                },
                frame_idx,
                cam_pos,
            );

            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe { self.device.cmd_end_render_pass(cmd) };
        }
    }

    // Per-object depth-only casters for one spot slice, inside the render pass
    // the caller opened: `cmd_draw_indexed` for static + instanced (iterated per
    // instance) + skinned casters. The cascades draw indirectly off the cull
    // records instead, whose per-cascade layout has no slot for a spot slice.
    fn encode_spot_casters(
        &self,
        cmd: vk::CommandBuffer,
        bind: SpotSliceBinding,
        frame_idx: usize,
        cam_pos: [f32; 3],
    ) {
        // See-through glass (Layer 2) casts no shadow: it is rerouted out of every
        // opaque rasterisation while RT is live, and the GPU-driven cascade takes
        // the same decision through the cull kernel's ENABLED bit.
        let skip_seethrough = self.mesh_glass_active();
        let device = &self.device;
        let SpotSliceBinding {
            pipeline: shadow_pipeline,
            layout: shadow_pl,
            set: shadow_set,
        } = bind;
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, shadow_pipeline);

            // Global shadow descriptor: ShadowUniforms UBO.
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                shadow_pl,
                0,
                std::slice::from_ref(&shadow_set),
                &[],
            );

            device.cmd_bind_vertex_buffers(cmd, 0, &[self.geometry.vertex_buffer.buffer()], &[0]);
            device.cmd_bind_index_buffer(
                cmd,
                self.geometry.index_buffer.buffer(),
                0,
                vk::IndexType::UINT32,
            );

            for obj in &self.draw.objects {
                // A non-resident streamed mesh has no geometry in the
                // shared buffers yet -- skip it everywhere.
                if !obj.visible || !obj.resident {
                    continue;
                }
                if skip_seethrough && obj.material.see_through != 0 {
                    continue; // see-through glass casts no shadow (Layer 2)
                }
                // Pick the LOD by camera distance: the shadow pass uses
                // the same slice the main pass will, so silhouettes track
                // when the runtime swaps to a coarser LOD.
                let d = crate::gfx::lod::camera_distance(obj, cam_pos);
                let (index_offset, index_count) = obj.active_lod(d);
                let push = ShadowPush {
                    model: obj.model,
                    cascade_idx: SPOT_SLICE_IDX,
                    _pad: [0; 3],
                };
                device.cmd_push_constants(
                    cmd,
                    shadow_pl,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    std::slice::from_raw_parts(
                        &push as *const ShadowPush as *const u8,
                        std::mem::size_of::<ShadowPush>(),
                    ),
                );
                device.cmd_draw_indexed(
                    cmd,
                    index_count as u32,
                    1,
                    index_offset as u32,
                    obj.base_vertex,
                    0,
                );
                self.inc_draw_calls(1);
            }

            // Instanced clusters in the shadow pass: iterate instances
            // individually using the regular shadow pipeline. Cheap to
            // ship; visually identical to an instanced shadow shader. Walk
            // the same per-LOD buckets the Main pass uses (computed by
            // `prepare_instanced_clusters`) so shadow silhouettes track the
            // per-instance LOD the camera picked.
            for cluster_idx in 0..self.instanced.clusters.len() {
                let Some(buckets) = self.instanced.lod_buckets.get(cluster_idx) else {
                    continue;
                };
                for bucket in buckets {
                    for &model in &bucket.instances {
                        let push = ShadowPush {
                            model,
                            cascade_idx: SPOT_SLICE_IDX,
                            _pad: [0; 3],
                        };
                        device.cmd_push_constants(
                            cmd,
                            shadow_pl,
                            vk::ShaderStageFlags::VERTEX,
                            0,
                            std::slice::from_raw_parts(
                                &push as *const ShadowPush as *const u8,
                                std::mem::size_of::<ShadowPush>(),
                            ),
                        );
                        device.cmd_draw_indexed(
                            cmd,
                            bucket.index_count as u32,
                            1,
                            bucket.index_offset as u32,
                            0,
                            0,
                        );
                        self.inc_draw_calls(1);
                    }
                }
            }

            // Skinned meshes: deformed depth, drawn after the static
            // and instanced casters within the same cascade render
            // pass (no re-clear, so skinned depth appends).
            if let (Some(sk_pipeline), Some(sk_pl)) = (
                self.shadow.skinned_pipeline.as_ref(),
                self.shadow.skinned_pipeline_layout.as_ref(),
            ) && !self.skinned.draw_objects.is_empty()
            {
                let (sk_vbuf, sk_ibuf) = self.skinned_geometry();
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    sk_pipeline.handle(),
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    sk_pl.handle(),
                    0,
                    std::slice::from_ref(&shadow_set),
                    &[],
                );
                device.cmd_bind_vertex_buffers(cmd, 0, std::slice::from_ref(&sk_vbuf), &[0]);
                device.cmd_bind_index_buffer(cmd, sk_ibuf, 0, vk::IndexType::UINT32);
                for (i, obj) in self.skinned.draw_objects.iter().enumerate() {
                    if !obj.visible {
                        continue;
                    }
                    // Match the Main pass's per-object LOD pick so shadow
                    // silhouettes track the active skinned LOD.
                    let d = crate::gfx::lod::skinned_camera_distance(obj, cam_pos);
                    let (index_offset, index_count) = obj.active_lod(d);
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        sk_pl.handle(),
                        1,
                        std::slice::from_ref(&self.skinned.joint_sets[frame_idx][i]),
                        &[],
                    );
                    let push = ShadowPush {
                        model: obj.model,
                        cascade_idx: SPOT_SLICE_IDX,
                        _pad: [0; 3],
                    };
                    device.cmd_push_constants(
                        cmd,
                        sk_pl.handle(),
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        std::slice::from_raw_parts(
                            &push as *const ShadowPush as *const u8,
                            std::mem::size_of::<ShadowPush>(),
                        ),
                    );
                    device.cmd_draw_indexed(cmd, index_count as u32, 1, index_offset as u32, 0, 0);
                    self.inc_draw_calls(1);
                }
            }
        }
    }
}
