// src/vulkan/light_cull.rs
//
// Clustered light-binning compute pass. Once per frame, before the Main pass,
// bins the scene's local lights (the `GpuLight` SSBO at global set 0 binding 9)
// into per-cluster index lists over a screen-tiled, exponential-depth froxel
// grid. The forward pass then shades each fragment from only its cluster's
// lights instead of iterating every light. Mirrors src/metal/light_cull.rs.

use ash::Device;
use ash::vk;

use crate::gfx::render_types::{CLUSTER_COUNT, CLUSTER_LIGHT_LIST_STRIDE, ClusterParams};

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::context::VkContext;
use super::pipeline::spv_module;

// Byte size of the per-cluster light-index buffer: CLUSTER_COUNT blocks of
// CLUSTER_LIGHT_LIST_STRIDE u32 (slot 0 = count, slots 1.. = light indices).
pub(in crate::vulkan) fn cluster_list_size() -> vk::DeviceSize {
    (CLUSTER_COUNT * CLUSTER_LIGHT_LIST_STRIDE) as vk::DeviceSize
        * std::mem::size_of::<u32>() as vk::DeviceSize
}

// Clustered-lighting GPU state: the binning compute pipeline, the per-cluster
// light-index buffer it writes / the forward pass reads, and the `ClusterParams`
// uniform buffers. The buffers are always allocated (the forward shaders
// reference bindings 10 + 11 unconditionally, guarded by `use_clusters`); the
// pipeline and its descriptor set exist only when the world has local lights.
pub(in crate::vulkan) struct VkLightCull {
    pub pipeline: Option<vk::Pipeline>,
    pub pipeline_layout: Option<vk::PipelineLayout>,
    pub set_layout: Option<vk::DescriptorSetLayout>,
    pub descriptor_pool: Option<vk::DescriptorPool>,
    // One compute set per frame in flight (each pointing at that frame's
    // `ClusterParams` UBO).
    pub sets: Vec<vk::DescriptorSet>,
    // Per-cluster light-index lists. Device-local; written by the kernel and
    // read by the forward pass at global set 0 binding 11.
    pub cluster_buffer: PooledBuffer,
    // Per-frame `ClusterParams` UBOs (host-visible, persistently mapped), bound
    // at global set 0 binding 10 for the main camera.
    pub params_buffers: Vec<PooledBuffer>,
    // A single `use_clusters = 0` copy, written once at init. The planar +
    // probe re-renders bind this at binding 10 so they fall back to iterating
    // every local light (their viewpoint differs from the main camera's grid).
    pub unclustered_buffer: PooledBuffer,
}

impl VkLightCull {
    // Destroy every owned GPU object. Called from `VkContext::drop` after
    // `wait_idle`.
    pub(in crate::vulkan) fn destroy(&mut self, device: &Device) {
        unsafe {
            if let Some(p) = self.pipeline {
                device.destroy_pipeline(p, None);
            }
            if let Some(l) = self.pipeline_layout {
                device.destroy_pipeline_layout(l, None);
            }
            if let Some(l) = self.set_layout {
                device.destroy_descriptor_set_layout(l, None);
            }
            if let Some(p) = self.descriptor_pool {
                device.destroy_descriptor_pool(p, None);
            }
        }
        self.cluster_buffer = PooledBuffer::null();
        self.params_buffers.clear();
        self.unclustered_buffer = PooledBuffer::null();
    }
}

// Descriptor set layout for the light-cull kernel: the `ClusterParams` UBO, the
// per-scene `GpuLight` SSBO, and the per-cluster list SSBO.
fn create_light_cull_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, String> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(|e| format!("light cull set layout: {e}"))
}

// Build the whole clustered-lighting state. `local_light_buffer` is the
// per-scene `GpuLight` SSBO the kernel bins; when the scene has no local lights
// the pipeline + descriptor set are skipped (the graph then omits `LightCull`)
// but the buffers are still allocated for the forward pass's unconditional binds.
pub(in crate::vulkan) fn build_light_cull(
    alloc: &DeviceAllocator,
    device: &Device,
    frames: usize,
    local_light_buffer: vk::Buffer,
    local_light_size: vk::DeviceSize,
    has_local_lights: bool,
    hot_reload: bool,
) -> Result<VkLightCull, String> {
    // Per-cluster light lists: device-local, written by compute, read by the
    // fragment stage.
    let cluster_buffer = alloc.create_buffer(
        cluster_list_size(),
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    // Per-frame `ClusterParams` UBOs, persistently mapped.
    let params_size = std::mem::size_of::<ClusterParams>() as vk::DeviceSize;
    let mut params_buffers = Vec::with_capacity(frames);
    for _ in 0..frames {
        params_buffers.push(alloc.create_buffer(
            params_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?);
    }

    // The static `use_clusters = 0` copy the planar / probe global sets bind.
    let unclustered_buffer = alloc.create_buffer(
        params_size,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    unsafe {
        let zero = ClusterParams::ZERO;
        std::ptr::copy_nonoverlapping(
            &zero as *const ClusterParams as *const u8,
            unclustered_buffer.mapped_ptr(),
            std::mem::size_of::<ClusterParams>(),
        );
    }

    if !has_local_lights {
        return Ok(VkLightCull {
            pipeline: None,
            pipeline_layout: None,
            set_layout: None,
            descriptor_pool: None,
            sets: Vec::new(),
            cluster_buffer,
            params_buffers,
            unclustered_buffer,
        });
    }

    let set_layout = create_light_cull_set_layout(device)?;
    let set_layouts = [set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
        .map_err(|e| format!("light cull pipeline layout: {e}"))?;

    let spirv = super::builtins::LIGHT_CULL.compile(&super::builtins::Ctx::plain(hot_reload))?;
    let module = spv_module(device, &spirv)?;
    let entry = std::ffi::CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&entry);
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout);
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, e)| format!("light cull pipeline: {e}"))?[0];
    unsafe { device.destroy_shader_module(module, None) };

    // One compute set per frame, each pointing at that frame's params UBO.
    let f = frames as u32;
    let sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: f,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 2 * f,
        },
    ];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(f)
        .pool_sizes(&sizes);
    let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
        .map_err(|e| format!("light cull descriptor pool: {e}"))?;
    let layouts: Vec<_> = (0..frames).map(|_| set_layout).collect();
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);
    let sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }
        .map_err(|e| format!("light cull descriptor sets: {e}"))?;

    for (i, &set) in sets.iter().enumerate() {
        let params_info = vk::DescriptorBufferInfo::default()
            .buffer(params_buffers[i].buffer())
            .offset(0)
            .range(params_size);
        let lights_info = vk::DescriptorBufferInfo::default()
            .buffer(local_light_buffer)
            .offset(0)
            .range(local_light_size);
        let list_info = vk::DescriptorBufferInfo::default()
            .buffer(cluster_buffer.buffer())
            .offset(0)
            .range(cluster_list_size());
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&params_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&lights_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&list_info)),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    Ok(VkLightCull {
        pipeline: Some(pipeline),
        pipeline_layout: Some(pipeline_layout),
        set_layout: Some(set_layout),
        descriptor_pool: Some(descriptor_pool),
        sets,
        cluster_buffer,
        params_buffers,
        unclustered_buffer,
    })
}

impl VkContext {
    // Write this frame's live `ClusterParams` into its UBO. The `use_clusters = 0`
    // copy the planar / probe passes bind was filled once at init.
    pub(in crate::vulkan) fn write_cluster_params(&self, frame_idx: usize, params: &ClusterParams) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                params as *const ClusterParams as *const u8,
                self.light_cull.params_buffers[frame_idx].mapped_ptr(),
                std::mem::size_of::<ClusterParams>(),
            );
        }
    }

    // Dispatch the clustered light-binning pass. One invocation per cluster; the
    // kernel builds the cluster's world-space AABB and tests each local light's
    // sphere against it, writing the surviving indices into `cluster_buffer`.
    // The trailing barrier orders the write before the forward pass's read.
    pub(in crate::vulkan) fn encode_light_cull(&self, cmd: vk::CommandBuffer, frame_idx: usize) {
        let (Some(pipeline), Some(layout)) =
            (self.light_cull.pipeline, self.light_cull.pipeline_layout)
        else {
            return;
        };
        let Some(&set) = self.light_cull.sets.get(frame_idx) else {
            return;
        };
        let device = &self.device;
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            );
            // One invocation per cluster, 64-wide workgroups.
            device.cmd_dispatch(cmd, CLUSTER_COUNT.div_ceil(64), 1, 1);
            // Order the compute write before the forward pass samples the lists.
            let barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.light_cull.cluster_buffer.buffer())
                .offset(0)
                .size(cluster_list_size());
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&barrier),
                &[],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::render_types::MAX_LIGHTS_PER_CLUSTER;

    // The GLSL kernel hardcodes the list stride + per-cluster cap as `const
    // uint`s (shaderc has no cross-module constants), so they must track the
    // Rust values the CPU sizes the buffer with.
    #[test]
    fn glsl_cluster_constants_match_render_types() {
        let src = crate::vulkan::builtins::LIGHT_CULL.embedded;
        assert!(src.contains(&format!(
            "CLUSTER_LIGHT_LIST_STRIDE = {CLUSTER_LIGHT_LIST_STRIDE}u"
        )));
        assert!(src.contains(&format!(
            "MAX_LIGHTS_PER_CLUSTER = {MAX_LIGHTS_PER_CLUSTER}u"
        )));
    }

    #[test]
    fn cluster_list_size_covers_every_cluster() {
        let expected = (CLUSTER_COUNT * CLUSTER_LIGHT_LIST_STRIDE) as vk::DeviceSize * 4;
        assert_eq!(cluster_list_size(), expected);
    }
}
