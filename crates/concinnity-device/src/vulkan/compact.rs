//! Packs a cull's kept draws into one list per stream (`cull_compact.hlsl`):
//! the scan and scatter kernels, the set each cull target binds them through,
//! and the indirect draw of one list, by its GPU-written count where the device
//! has `VK_KHR_draw_indirect_count`, else at full capacity over commands zeroed
//! before the scatter.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types::{
    COUNT_SLOT_BYTES, CULL_GROUP_RECORDS, GpuCompactParams, StreamSlots, count_slot_offset,
    group_count_words,
};
use concinnity_engine::render::backend::error::RenderResult;

use super::context::VkContext;
use super::cull::INDIRECT_COMMAND_STRIDE;
use super::descriptor_layout::Binding;
use super::owned::{OwnedPipeline, OwnedPipelineLayout, OwnedSetLayout};
use super::pipeline_desc::compute_pipeline;
use super::record::cmd_push_constants;
use super::resources::create_descriptor_set_layout;
use super::set_writes::SetWrites;

// Bytes of `GpuCompactParams`, the kernels' push constant.
const COMPACT_PUSH_BYTES: u32 =
    concinnity_core::math::saturating_u32(size_of::<GpuCompactParams>());

// Usage of an indirect buffer a cull packs: the scatter writes it, the
// fallback path zeroes it, and the draws read it.
pub(in crate::vulkan) const INDIRECT_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
);

// Zero freshly made indirect buffers, so a list no scan has packed draws
// nothing.
pub(in crate::vulkan) fn zero_indirect_buffers<'a>(
    device: &super::owned::VkDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    buffers: impl IntoIterator<Item = &'a super::allocator::PooledBuffer>,
) -> RenderResult<()> {
    super::texture::one_shot_submit(device, command_pool, queue, |cmd| {
        for buffer in buffers {
            // SAFETY: `cmd` is recording outside a render pass, and every indirect buffer is made
            // with TRANSFER_DST and is not yet in use.
            unsafe { device.cmd_fill_buffer(cmd, buffer.buffer(), 0, vk::WHOLE_SIZE, 0) };
        }
    })
}

// Set 0 of both kernels: the variant block, the draw args, the group counts,
// the cull status, the commands and the list lengths behind them.
pub(in crate::vulkan) fn compact_set() -> [Binding; 6] {
    super::init::cull_set_bindings()
}

// The scan and the scatter over one layout.
pub(in crate::vulkan) struct VkCompactKernels {
    scan: OwnedPipeline,
    scatter: OwnedPipeline,
    layout: OwnedPipelineLayout,
    pub(in crate::vulkan) set_layout: OwnedSetLayout,
}

pub(in crate::vulkan) fn build_compact_kernels(
    device: &super::owned::VkDevice,
    hot_reload: bool,
) -> RenderResult<VkCompactKernels> {
    use super::builtin_shaders::{CULL_SCAN, CULL_SCATTER, CompileProgram as _};
    let set_layout = create_descriptor_set_layout(device, &compact_set())?;
    let push = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .size(COMPACT_PUSH_BYTES);
    let layout = device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&set_layout.handle()))
                .push_constant_ranges(std::slice::from_ref(&push)),
        )
        .map_err(|e| super::error::map_vk_result(e, "cull compact layout"))?;
    let scan = compute_pipeline(
        device,
        layout.handle(),
        &CULL_SCAN.compile(hot_reload)?,
        "cull scan",
    )?;
    let scatter = compute_pipeline(
        device,
        layout.handle(),
        &CULL_SCATTER.compile(hot_reload)?,
        "cull scatter",
    )?;
    Ok(VkCompactKernels {
        scan,
        scatter,
        layout,
        set_layout,
    })
}

// Bytes the commands of an indirect buffer holding `commands` take.
pub(in crate::vulkan) fn command_bytes(commands: usize) -> u64 {
    commands as u64 * u64::from(INDIRECT_COMMAND_STRIDE)
}

// Bytes of an indirect buffer holding `commands`, its count slots behind them.
pub(in crate::vulkan) fn indirect_buffer_bytes(commands: usize) -> u64 {
    count_slot_offset(command_bytes(commands.max(1))) + COUNT_SLOT_BYTES
}

// Bytes of the group-count scratch a cull of `records` writes.
pub(in crate::vulkan) fn group_count_bytes(records: usize) -> u64 {
    group_count_words(u32::saturating_from(records)).max(1) as u64 * 4
}

// An indirect buffer a cull packs into, and how many commands it holds ahead
// of its count slots.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct DrawLists {
    pub(in crate::vulkan) buffer: vk::Buffer,
    pub(in crate::vulkan) commands: usize,
}

// What one cull target's set binds.
pub(in crate::vulkan) struct CompactBuffers {
    pub(in crate::vulkan) variants: (vk::Buffer, u64),
    pub(in crate::vulkan) draw_args: (vk::Buffer, u64),
    pub(in crate::vulkan) status: (vk::Buffer, u64),
    pub(in crate::vulkan) group_counts: (vk::Buffer, u64),
    pub(in crate::vulkan) lists: DrawLists,
}

pub(in crate::vulkan) fn write_compact_set(
    device: &super::owned::VkDevice,
    set: vk::DescriptorSet,
    b: &CompactBuffers,
) {
    let commands = command_bytes(b.lists.commands.max(1));
    SetWrites::new(set)
        .storage_buffer(0, b.variants.0, b.variants.1)
        .storage_buffer(1, b.draw_args.0, b.draw_args.1)
        .storage_buffer(2, b.group_counts.0, b.group_counts.1)
        .storage_buffer(3, b.status.0, b.status.1)
        .storage_buffer(4, b.lists.buffer, commands)
        .buffer(
            5,
            vk::DescriptorType::STORAGE_BUFFER,
            b.lists.buffer,
            count_slot_offset(commands),
            COUNT_SLOT_BYTES,
        )
        .apply(device);
}

// One cull's packing as a recording runs it: the target's set and lists, and
// what to pack.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct CompactPass {
    pub(in crate::vulkan) set: vk::DescriptorSet,
    pub(in crate::vulkan) lists: DrawLists,
    pub(in crate::vulkan) params: GpuCompactParams,
}

// Order a compute pass's writes before the next compute pass reads them.
fn compute_barrier(device: &ash::Device, cmd: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    // SAFETY: a global memory barrier on the recording command buffer; `barrier` is borrowed for
    // the call.
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&barrier),
            &[],
            &[],
        );
    }
}

impl VkContext {
    // Pack the cull just dispatched on `cmd`: scan its group counts, then
    // scatter its kept records' commands, each list in record order. The
    // caller orders the scatter's writes before the draws that read them.
    pub(in crate::vulkan) fn encode_compaction(&self, cmd: vk::CommandBuffer, pass: CompactPass) {
        let Some(kernels) = self.cull.compact.as_ref() else {
            return;
        };
        let device = &self.hw.device;
        compute_barrier(device, cmd);
        if self.hw.draw_count.is_none() {
            // The last draws that read these lists finish before they are zeroed.
            let reads = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            // SAFETY: a global memory barrier on the recording command buffer.
            unsafe {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::DRAW_INDIRECT,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&reads),
                    &[],
                    &[],
                );
            }
            // SAFETY: `cmd` is recording outside a render pass, and the lists' buffer is live and
            // was created with TRANSFER_DST.
            unsafe {
                device.cmd_fill_buffer(
                    cmd,
                    pass.lists.buffer,
                    0,
                    command_bytes(pass.lists.commands.max(1)),
                    0,
                );
            }
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE);
            // SAFETY: a global memory barrier on the recording command buffer.
            unsafe {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&barrier),
                    &[],
                    &[],
                );
            }
        }
        let layout = kernels.layout.handle();
        // SAFETY: `cmd` is recording, and the scan pipeline is owned by `self.cull`.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kernels.scan.handle());
        }
        // SAFETY: `pass.set` was written against `kernels.set_layout`, the layout's only set.
        unsafe {
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                std::slice::from_ref(&pass.set),
                &[],
            );
        }
        // SAFETY: `layout` is the bound kernels' layout, and the params are borrowed for the call.
        unsafe {
            cmd_push_constants(
                device,
                cmd,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                &pass.params,
            );
        }
        // SAFETY: the scan pipeline, its set and push constants are bound above.
        unsafe { device.cmd_dispatch(cmd, pass.params.bucket_count.clamp(1, 64) + 1, 1, 1) };
        compute_barrier(device, cmd);
        // SAFETY: the scatter pipeline is owned by `self.cull`; the set and push constants stay
        // bound across the switch, the two kernels sharing one layout.
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                kernels.scatter.handle(),
            );
        }
        let groups = pass.params.object_count.div_ceil(CULL_GROUP_RECORDS);
        // SAFETY: the scatter pipeline, its set and push constants are bound above.
        unsafe { device.cmd_dispatch(cmd, groups, 1, 1) };
    }

    // Draw stream `stream`'s list of `lists`, whose commands sit at `slots`.
    pub(in crate::vulkan) fn draw_stream(
        &self,
        cmd: vk::CommandBuffer,
        lists: DrawLists,
        stream: u32,
        slots: StreamSlots,
    ) {
        if slots.capacity == 0 {
            return;
        }
        let stride = INDIRECT_COMMAND_STRIDE;
        let offset = u64::from(slots.region + slots.first) * u64::from(stride);
        let count_offset =
            count_slot_offset(command_bytes(lists.commands.max(1))) + 4 * u64::from(stream);
        match &self.hw.draw_count {
            // SAFETY: `cmd` is recording inside a render pass with a graphics pipeline bound; the
            // list and its count slot lie inside `lists.buffer`, which outlives the frame.
            Some(draw_count) => unsafe {
                draw_count.cmd_draw_indexed_indirect_count(
                    cmd,
                    lists.buffer,
                    offset,
                    lists.buffer,
                    count_offset,
                    slots.capacity,
                    stride,
                );
            },
            // SAFETY: as above; the commands past the list's count were zeroed before the scatter,
            // so they draw nothing.
            None => unsafe {
                self.hw.device.cmd_draw_indexed_indirect(
                    cmd,
                    lists.buffer,
                    offset,
                    slots.capacity,
                    stride,
                );
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_slots_follow_the_commands_aligned() {
        let stride = u64::from(INDIRECT_COMMAND_STRIDE);
        assert_eq!(command_bytes(300), 300 * stride);
        let total = indirect_buffer_bytes(300);
        assert_eq!(total, count_slot_offset(300 * stride) + COUNT_SLOT_BYTES);
        assert_eq!(count_slot_offset(300 * stride) % 256, 0);
        assert!(count_slot_offset(300 * stride) >= 300 * stride);
    }

    #[test]
    fn a_buffer_holds_at_least_one_command() {
        assert_eq!(indirect_buffer_bytes(0), indirect_buffer_bytes(1));
        assert!(group_count_bytes(0) >= 4);
    }

    #[test]
    fn indirect_lists_can_be_zeroed_written_and_drawn() {
        for usage in [
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::BufferUsageFlags::INDIRECT_BUFFER,
            vk::BufferUsageFlags::TRANSFER_DST,
        ] {
            assert!(INDIRECT_USAGE.contains(usage), "{usage:?}");
        }
    }

    #[test]
    fn the_count_path_follows_the_extension() {
        let with: [&[u8]; 2] = [b"VK_KHR_swapchain", b"VK_KHR_draw_indirect_count"];
        let without: [&[u8]; 1] = [b"VK_KHR_swapchain"];
        assert!(crate::vulkan::device::draws_by_count(with.into_iter()));
        assert!(!crate::vulkan::device::draws_by_count(without.into_iter()));
    }
}
