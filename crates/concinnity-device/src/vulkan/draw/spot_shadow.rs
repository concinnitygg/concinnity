//! Spot shadow pass: one depth-only render per shadow-casting spot light into
//! its layer of the spot shadow array. Structurally the cascade pass with a
//! different projection source -- each slice is a `ShadowView` drawn by the
//! same depth-only render pass and GPU-driven pipeline, with a per-slice
//! descriptor set whose `ShadowUniforms` holds that spot's light-space matrix in
//! slot 0 rather than the CSM cascade set.
//!
//! Local lights are static, so the matrices are built once here and only the
//! depth contents refresh. `spot_shadow.render_mask` (from `SpotShadowScheduler`)
//! picks which slices redraw; a skipped slice keeps the depth it last rendered,
//! which stays correct until a caster moves.

use ash::vk;
use concinnity_core::gfx::render_types::{ShadowUniforms, SpotShadowData};
use concinnity_core::render::csm;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::spot_shadow;

use super::shadow::ShadowView;
use crate::vulkan::allocator::{DeviceAllocator, PooledBuffer};
use crate::vulkan::context::{VkContext, VkSpotShadow};
use crate::vulkan::descriptor_layout::{PoolSizes, shadow_global_set};
use crate::vulkan::owned::VkDevice;
use crate::vulkan::resources::alloc_descriptor_sets;
use crate::vulkan::set_writes::SetWrites;
use crate::vulkan::texture::GpuImage;

// Everything `build_spot_shadow` needs from init. Grouped so the builder takes
// one parameter instead of a nine-argument list.
pub(in crate::vulkan) struct SpotShadowBuild<'a> {
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
pub(in crate::vulkan) fn build_spot_shadow(b: SpotShadowBuild<'_>) -> RenderResult<VkSpotShadow> {
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
                let mut u = csm::empty_shadow_uniforms();
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
    let pool_sizes = PoolSizes::default()
        .sets(&shadow_global_set(), set_count)
        .build();
    let descriptor_pool = device
        .create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(set_count),
        )
        .map_err(|e| crate::vulkan::error::map_vk_result(e, "spot shadow descriptor pool"))?;

    let layouts: Vec<_> = (0..set_count).map(|_| set_layout).collect();
    let sets = alloc_descriptor_sets(device, descriptor_pool.handle(), &layouts)?;
    for (i, &set) in sets.iter().enumerate() {
        SetWrites::new(set)
            .buffer(
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                ubo.buffer(),
                i as u64 * stride,
                size_of::<ShadowUniforms>() as u64,
            )
            .apply(device);
    }

    Ok(VkSpotShadow {
        map,
        framebuffers,
        slice_size,
        data_buffer,
        ubo,
        sets,
        _descriptor_pool: descriptor_pool,
        frusta: spot_shadows
            .iter()
            .map(spot_shadow::slice_frustum)
            .collect(),
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
    // One depth-only render pass per scheduled spot slice, GPU-driven like the
    // cascades: a per-slice cull against the spot's light frustum writes the
    // slice's own indirect buffer, and the slice draws it through the shared
    // bindless shadow pipeline with that slice's uniforms at set 0. Slices with
    // no records still clear, so the main pass samples valid depth.
    // pub(in crate::vulkan) so the render-graph executor can dispatch it.
    pub(in crate::vulkan) fn encode_spot_shadow_pass(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        cam_pos: [f32; 3],
    ) {
        if self.spot_shadow.count() == 0 {
            return;
        }
        let device = &self.hw.device;
        let gpu_driven = self.shadow_views_drawable()
            && self.cull.spot_indirect_buffers.get(frame_idx).is_some();

        // Cull every refreshed slice before opening any render pass (Vulkan
        // disallows compute inside one).
        if gpu_driven {
            self.encode_spot_culls(cmd, frame_idx, cam_pos);
        }

        let sz = self.spot_shadow.slice_size;
        for slice in self.spot_shadow.refreshed_slices() {
            let framebuffer = &self.spot_shadow.framebuffers[slice as usize];
            self.begin_shadow_slice(cmd, framebuffer.handle(), sz);
            if gpu_driven {
                self.draw_shadow_view(
                    device,
                    cmd,
                    frame_idx,
                    ShadowView {
                        uniforms_set: self.spot_shadow.sets[slice as usize],
                        // Every spot slice carries its own matrix in `light_vps[0]`.
                        vp_index: 0,
                        indirect: self.cull.spot_indirect_buffers[frame_idx][slice as usize]
                            .buffer(),
                    },
                );
            }
            // SAFETY: `cmd` is a command buffer in the recording state inside the render pass
            // `begin_shadow_slice` opened.
            unsafe { device.cmd_end_render_pass(cmd) };
        }
    }
}
