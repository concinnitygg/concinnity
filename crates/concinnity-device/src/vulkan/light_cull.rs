//! Clustered binning compute pass. Once per frame, before the Main pass, bins
//! the scene's local lights (the `GpuLight` SSBO at global set 0 binding 9) into
//! per-cluster light lists and the reflection probes' influence boxes into
//! per-cluster probe masks, over a screen-tiled, exponential-depth froxel grid.
//! The forward, SSR and transparent passes then shade from only a fragment's
//! cluster's lights and blend only its cluster's probes. Mirrors
//! src/metal/light_cull.rs.

use ash::vk;
use concinnity_core::gfx::render_types::{CLUSTER_COUNT, CLUSTER_LIST_LEN, ClusterParams};
use concinnity_core::render::error::RenderResult;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::context::VkContext;
use super::descriptor_layout::{Binding, PoolSizes};
use super::pipeline::{SHADER_ENTRY, spv_module};
use super::resources::create_descriptor_set_layout;
use crate::vulkan::builtin_shaders::CompileProgram;
use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedPipeline, OwnedPipelineLayout, OwnedSetLayout, VkDevice,
};
use crate::vulkan::record::Recorder;

// Byte size of the per-cluster list buffer: CLUSTER_LIST_LEN u32, every
// cluster's light list and then every cluster's probe mask.
pub(in crate::vulkan) fn cluster_list_size() -> vk::DeviceSize {
    CLUSTER_LIST_LEN as vk::DeviceSize * std::mem::size_of::<u32>() as vk::DeviceSize
}

// Binding of the probe records in the kernel's set.
const PROBE_RECORDS_BINDING: u32 = 3;

// Clustered-lighting GPU state: the binning compute pipeline, the per-cluster
// list buffer it writes / the forward pass reads, and the `ClusterParams`
// uniform buffers. All of it always exists (the forward shaders reference
// bindings 10 + 11 unconditionally, guarded by `use_clusters`); the kernel runs
// only on frames with a light or a probe to bin.
pub(in crate::vulkan) struct VkLightCull {
    pub pipeline: OwnedPipeline,
    pub pipeline_layout: OwnedPipelineLayout,
    pub _set_layout: OwnedSetLayout,
    pub _descriptor_pool: OwnedDescriptorPool,
    // One compute set per frame in flight (each pointing at that frame's
    // `ClusterParams` UBO and probe records).
    pub sets: Vec<vk::DescriptorSet>,
    // Per-cluster light lists and probe masks. Device-local; written by the kernel
    // and read at global set 0 binding 11.
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
    // Point frame `frame`'s kernel set at that frame's probe records. Called
    // when the set is first wired and whenever the frame's records buffer is
    // replaced; the caller guarantees no submission still reads the set.
    pub(in crate::vulkan) fn write_probe_records(
        &self,
        device: &VkDevice,
        frame: usize,
        records: vk::DescriptorBufferInfo,
    ) {
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.sets[frame])
            .dst_binding(PROBE_RECORDS_BINDING)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&records));
        // SAFETY: the write and the buffer info it borrows are live for the call,
        // the set and buffer belong to this device, and the caller guarantees no
        // submission still references the set.
        unsafe { device.update_descriptor_sets(&[write], &[]) };
    }

    // Destroy every owned GPU object. Called from `VkContext::drop` after
    // `wait_idle`.
    pub(in crate::vulkan) fn destroy(&mut self, _device: &VkDevice) {
        self.cluster_buffer = PooledBuffer::null();
        self.params_buffers.clear();
        self.unclustered_buffer = PooledBuffer::null();
    }
}

// Descriptor set layout for the light-cull kernel: the `ClusterParams` UBO, the
// per-scene `GpuLight` SSBO, the per-cluster list SSBO and the frame's probe
// records SSBO.
fn light_cull_set_bindings() -> [Binding; 4] {
    use vk::DescriptorType as T;
    let compute = vk::ShaderStageFlags::COMPUTE;
    [
        (0, T::UNIFORM_BUFFER, compute),
        (1, T::STORAGE_BUFFER, compute),
        (2, T::STORAGE_BUFFER, compute),
        (PROBE_RECORDS_BINDING, T::STORAGE_BUFFER, compute),
    ]
}

// Build the whole clustered-lighting state. `local_light_buffer` is the
// per-scene `GpuLight` SSBO the kernel bins. Every set's probe records are
// written by `write_probe_records` once the probe set exists.
pub(in crate::vulkan) fn build_light_cull(
    alloc: &DeviceAllocator,
    device: &VkDevice,
    frames: usize,
    local_light_buffer: vk::Buffer,
    local_light_size: vk::DeviceSize,
    hot_reload: bool,
) -> RenderResult<VkLightCull> {
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
    unclustered_buffer.write_val(0, &ClusterParams::ZERO);

    let set_layout = create_descriptor_set_layout(device, &light_cull_set_bindings())?;
    let set_layouts = [set_layout.handle()];
    let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    let pipeline_layout = device
        .create_pipeline_layout(&layout_info)
        .map_err(|e| super::error::map_vk_result(e, "light cull pipeline layout"))?;

    let spirv = super::builtin_shaders::LIGHT_CULL.compile(hot_reload)?;
    let module = spv_module(device, &spirv)?;
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module.handle())
        .name(SHADER_ENTRY);
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout.handle());
    let pipeline = crate::vulkan::pipeline_cache::create_compute_pipeline(device, &pipeline_info)
        .map_err(|e| super::error::map_vk_result(e, "light cull pipeline"))?;

    // One compute set per frame, each pointing at that frame's params UBO.
    let f = frames as u32;
    let sizes = PoolSizes::default()
        .sets(&light_cull_set_bindings(), f)
        .build();
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(f)
        .pool_sizes(&sizes);
    let descriptor_pool = device
        .create_descriptor_pool(&pool_info)
        .map_err(|e| super::error::map_vk_result(e, "light cull descriptor pool"))?;
    let layouts: Vec<_> = (0..frames).map(|_| set_layout.handle()).collect();
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool.handle())
        .set_layouts(&layouts);
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    let sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }
        .map_err(|e| super::error::map_vk_result(e, "light cull descriptor sets"))?;

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
        // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every
        // set and resource it names belongs to this device.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    Ok(VkLightCull {
        pipeline,
        pipeline_layout,
        _set_layout: set_layout,
        _descriptor_pool: descriptor_pool,
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
        self.light_cull.params_buffers[frame_idx].write_val(0, params);
    }

    // Dispatch the clustered binning pass. One invocation per cluster; the
    // kernel builds the cluster's world-space AABB and tests each local light's
    // sphere and each probe's influence box against it, writing the surviving
    // indices into `cluster_buffer`. The trailing barrier orders the write
    // before the forward pass's read.
    pub(in crate::vulkan) fn encode_light_cull(&self, rec: &Recorder<'_>, frame_idx: usize) {
        let Some(&set) = self.light_cull.sets.get(frame_idx) else {
            return;
        };
        let layout = &self.light_cull.pipeline_layout;
        rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &self.light_cull.pipeline);
        rec.bind_descriptor_sets(vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
        // One invocation per cluster, 64-wide workgroups.
        rec.dispatch(CLUSTER_COUNT.div_ceil(64), 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_list_size_covers_every_list() {
        assert_eq!(cluster_list_size(), CLUSTER_LIST_LEN as vk::DeviceSize * 4);
    }
}
