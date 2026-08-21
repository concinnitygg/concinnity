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

use ash::{Device, vk};

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
    pub device: &'a Device,
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
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    let descriptor_pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(set_count),
            None,
        )
    }
    .map_err(|e| format!("spot shadow descriptor pool: {e}"))?;

    let layouts: Vec<_> = (0..set_count).map(|_| set_layout).collect();
    let sets = alloc_descriptor_sets(device, descriptor_pool, &layouts)?;
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
        descriptor_pool,
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
        let (Some(pipeline), Some(layout)) = (self.shadow.pipeline, self.shadow.pipeline_layout)
        else {
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
                .render_pass(self.shadow.render_pass)
                .framebuffer(self.spot_shadow.framebuffers[slice as usize])
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

            // Spot casters go through the legacy CPU sub-encoders: the shadow
            // indirect buffer the bindless cull fills is laid out per CSM
            // cascade, so it has no slots for these slices. `slice_idx` 0 picks
            // `light_vps[0]`, which is this spot's matrix.
            self.encode_shadow_slice_legacy(
                cmd,
                super::shadow::ShadowSliceBinding {
                    pipeline,
                    layout,
                    set: self.spot_shadow.sets[slice as usize],
                    slice_idx: 0,
                },
                frame_idx,
                cam_pos,
            );

            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
            // these commands name is live for the call.
            unsafe { self.device.cmd_end_render_pass(cmd) };
        }
    }
}
