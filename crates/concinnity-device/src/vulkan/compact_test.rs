// The packing kernels as Vulkan runs them, on the path a device without
// `VK_KHR_draw_indirect_count` takes: the commands zeroed, then the scan and
// the scatter over status words and group counts laid out as the cull leaves
// them. Every stream's list must be the one `render_types::compact` builds,
// in record order, its count in its slot, and every command past the count a
// zeroed no-op. A large cull scans more groups per stream than one scan pass
// holds, carrying each pass's total into the next. The shadow cull ahead of
// them must leave the status words and group counts the spec does.

use ash::vk;
use concinnity_engine::gfx::cull_status::CullStatus;
use concinnity_engine::gfx::render_types::{
    BucketSpans, DRAW_STREAMS, GpuCompactParams, GpuCullVariants, GpuDrawArgs, GpuObjectData,
    MAX_SHADER_BUCKETS, StreamLayout, bucket_regions, compact, count_slot_offset, draw_args_flags,
    group_count_words, group_counts, scan_group_counts,
};
use concinnity_engine::render::uniforms::vulkan::CullParams;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::builtin_shaders::{CULL_SCAN, CULL_SCATTER, CULL_SHADOW, CompileProgram as _};
use super::compact::{command_bytes, compact_set, indirect_buffer_bytes};
use super::test_gpu::{TestGpu, test_gpu};

const HOST: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
    vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
        | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
);
const BUCKETS: u32 = 4;

// The skinned tail is the last tenth of the records.
fn skinned_base(records: u32) -> u32 {
    records / 10 * 9
}

// Status words mixing outcomes and buckets, some routed outside their
// bucket's region.
fn status(records: u32) -> Vec<u32> {
    (0..records)
        .map(|r| {
            let outcome = if r % 5 == 1 {
                CullStatus::CULLED
            } else {
                CullStatus::DRAWN
            };
            outcome | (((r * 7 / 3) % BUCKETS) << CullStatus::BUCKET_SHIFT)
        })
        .collect()
}

fn draw_args(records: u32) -> Vec<GpuDrawArgs> {
    (0..records)
        .map(|r| GpuDrawArgs {
            index_count: 3 + r,
            index_offset: 2 * r,
            base_vertex: 0,
            flags: 0,
        })
        .collect()
}

// Overlapping buckets, two with runtime reserves and one with only a reserve.
fn variants(records: u32) -> GpuCullVariants {
    let n = records as usize;
    let spans = [
        BucketSpans {
            records: 0..n,
            runtime: 0..0,
        },
        BucketSpans {
            records: n / 30..n * 2 / 5,
            runtime: n * 2 / 3..n * 13 / 15,
        },
        BucketSpans {
            records: n / 6..n * 4 / 5,
            runtime: 0..0,
        },
        BucketSpans {
            records: 0..0,
            runtime: n / 3..n,
        },
    ];
    GpuCullVariants {
        regions: bucket_regions(&spans),
        ..GpuCullVariants::default()
    }
}

fn buffer(alloc: &DeviceAllocator, bytes: &[u8], usage: vk::BufferUsageFlags) -> PooledBuffer {
    let buffer = alloc
        .create_buffer(bytes.len().max(4) as u64, usage, HOST)
        .expect("buffer");
    buffer.write_bytes(0, bytes);
    buffer
}

fn module(gpu: &TestGpu, spirv: &[u8]) -> vk::ShaderModule {
    let words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    // SAFETY: `words` is a whole SPIR-V module, live for the call.
    unsafe {
        gpu.device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
    }
    .expect("module")
}

fn pipeline(gpu: &TestGpu, layout: vk::PipelineLayout, spirv: &[u8]) -> vk::Pipeline {
    let module = module(gpu, spirv);
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(c"main");
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    // SAFETY: the create-info and the module it names are live for the call.
    let pipeline = unsafe {
        gpu.device
            .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
    }
    .expect("pipeline")[0];
    // SAFETY: the pipeline no longer needs the module once created.
    unsafe { gpu.device.destroy_shader_module(module, None) };
    pipeline
}

#[test]
fn the_kernels_pack_every_list_as_the_spec_does() {
    pack_and_check(300);
}

#[test]
fn the_scan_carries_totals_across_its_passes() {
    // 130 groups a stream: two full scan passes and a partial third.
    pack_and_check(8300);
}

fn pack_and_check(records: u32) {
    concinnity_shader::require_dxc!();
    let Some(gpu) = test_gpu() else {
        return;
    };
    let device = &gpu.device;
    let alloc = DeviceAllocator::new(&gpu.instance, gpu.physical_device, device, 2);
    let variants = variants(records);
    let status = status(records);
    let layout = StreamLayout {
        regions: &variants.regions,
        bucket_count: BUCKETS,
        skinned_base: skinned_base(records),
        records,
    };
    let commands = variants
        .regions
        .iter()
        .take(BUCKETS as usize)
        .map(|r| r.len() as usize)
        .sum::<usize>();
    let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
    let variants_buf = buffer(&alloc, bytemuck::bytes_of(&variants), storage);
    let args_buf = buffer(&alloc, bytemuck::cast_slice(&draw_args(records)), storage);
    let counts = group_counts(&status, CullStatus::DRAWN, &layout);
    let counts_buf = buffer(&alloc, bytemuck::cast_slice(&counts), storage);
    let status_buf = buffer(&alloc, bytemuck::cast_slice(&status), storage);
    let lists_bytes = indirect_buffer_bytes(commands);
    let lists_buf = buffer(
        &alloc,
        &vec![0xAB; usize::try_from(lists_bytes).expect("bytes")],
        super::compact::INDIRECT_USAGE,
    );

    let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = compact_set()
        .iter()
        .map(|&(binding, ty, stages)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(stages)
        })
        .collect();
    // SAFETY: every create-info below and the slices it borrows are live for its call, and each
    // handle it names belongs to this device.
    let set_layout = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }
    .expect("set layout");
    let push = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .size(16);
    // SAFETY: as above.
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&set_layout))
                .push_constant_ranges(std::slice::from_ref(&push)),
            None,
        )
    }
    .expect("pipeline layout");
    let scan = pipeline(
        &gpu,
        pipeline_layout,
        &CULL_SCAN.compile(false).expect("scan"),
    );
    let scatter = pipeline(
        &gpu,
        pipeline_layout,
        &CULL_SCATTER.compile(false).expect("scatter"),
    );
    let sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 6,
    }];
    // SAFETY: as above.
    let pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&sizes)
                .max_sets(1),
            None,
        )
    }
    .expect("pool");
    // SAFETY: as above.
    let set = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(std::slice::from_ref(&set_layout)),
        )
    }
    .expect("set")[0];
    let command_span = command_bytes(commands);
    let infos = [
        (variants_buf.buffer(), 0, vk::WHOLE_SIZE),
        (args_buf.buffer(), 0, vk::WHOLE_SIZE),
        (counts_buf.buffer(), 0, vk::WHOLE_SIZE),
        (status_buf.buffer(), 0, vk::WHOLE_SIZE),
        (lists_buf.buffer(), 0, command_span),
        (
            lists_buf.buffer(),
            count_slot_offset(command_span),
            4 * DRAW_STREAMS as u64,
        ),
    ]
    .map(|(buffer, offset, range)| vk::DescriptorBufferInfo {
        buffer,
        offset,
        range,
    });
    let writes: Vec<vk::WriteDescriptorSet<'_>> = infos
        .iter()
        .enumerate()
        .map(|(b, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(u32::try_from(b).expect("binding"))
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        })
        .collect();
    // SAFETY: the writes name the live set and buffers above.
    unsafe { device.update_descriptor_sets(&writes, &[]) };

    // SAFETY: family 0 is the one queue family `test_gpu` created a queue in.
    let cmd_pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(0),
            None,
        )
    }
    .expect("command pool");
    // SAFETY: the pool is live.
    let cmd = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .command_buffer_count(1),
        )
    }
    .expect("command buffer")[0];
    let params = GpuCompactParams {
        object_count: records,
        bucket_count: BUCKETS,
        skinned_base: skinned_base(records),
        drawn: CullStatus::DRAWN,
    };
    let barrier = |src, dst, src_stage, dst_stage| {
        let b = vk::MemoryBarrier::default()
            .src_access_mask(src)
            .dst_access_mask(dst);
        // SAFETY: a global barrier on the recording command buffer.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[b],
                &[],
                &[],
            );
        }
    };
    // Every call below records into or submits `cmd`, whose pipelines, set and buffers stay live
    // until the queue idles.
    // SAFETY: as above; `cmd` is a fresh command buffer.
    unsafe { device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()) }
        .expect("begin");
    // SAFETY: as above; the lists' buffer was created with TRANSFER_DST.
    unsafe { device.cmd_fill_buffer(cmd, lists_buf.buffer(), 0, command_span, 0) };
    barrier(
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    );
    // SAFETY: as above.
    unsafe { device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, scan) };
    // SAFETY: as above; the set was written against the layout's only set.
    unsafe {
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &[set],
            &[],
        );
    }
    // SAFETY: as above; the push range covers the 16 bytes of params.
    unsafe {
        device.cmd_push_constants(
            cmd,
            pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&params),
        );
    }
    // SAFETY: as above.
    unsafe { device.cmd_dispatch(cmd, BUCKETS + 1, 1, 1) };
    barrier(
        vk::AccessFlags::SHADER_WRITE,
        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    );
    // SAFETY: as above.
    unsafe { device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, scatter) };
    // SAFETY: as above.
    unsafe { device.cmd_dispatch(cmd, records.div_ceil(64), 1, 1) };
    // SAFETY: as above.
    unsafe { device.end_command_buffer(cmd) }.expect("end");
    // SAFETY: family 0, index 0 is the one queue `test_gpu` created.
    let queue = unsafe { device.get_device_queue(0, 0) };
    let commands_to_run = [cmd];
    let submit = vk::SubmitInfo::default().command_buffers(&commands_to_run);
    // SAFETY: as above; the submit info borrows a live command buffer.
    unsafe { device.queue_submit(queue, &[submit], vk::Fence::null()) }.expect("submit");
    // SAFETY: the queue belongs to this device.
    unsafe { device.queue_wait_idle(queue) }.expect("wait");

    let read = |buffer: &PooledBuffer, offset: u64, len: usize| -> Vec<u32> {
        let offset = usize::try_from(offset).expect("offset");
        // SAFETY: the buffer is host-visible and coherent, and every read stays inside the bytes
        // it was created with.
        let start = unsafe { buffer.mapped_ptr().add(offset) };
        // SAFETY: the queue idled, so the GPU no longer writes the `4 * len` bytes from `start`.
        let bytes = unsafe { std::slice::from_raw_parts(start, 4 * len) };
        bytes
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect()
    };
    let words = |offset, len| read(&lists_buf, offset, len);
    let mut scanned = counts.clone();
    scan_group_counts(&mut scanned, &layout);
    assert_eq!(
        read(&counts_buf, 0, counts.len()),
        scanned,
        "the scan leaves each group's first place"
    );
    let lists = compact(&status, CullStatus::DRAWN, &layout);
    let slot_counts = words(count_slot_offset(command_span), DRAW_STREAMS);
    let mut drawn = 0;
    for stream in layout.streams() {
        let list = &lists[stream as usize];
        assert_eq!(
            slot_counts[stream as usize] as usize,
            list.len(),
            "stream {stream} count"
        );
        let slots = layout.slots(stream);
        let first = u64::from(slots.region + slots.first) * 20;
        let held = words(first, 5 * slots.capacity as usize);
        for (k, command) in held.chunks_exact(5).enumerate() {
            match list.get(k) {
                Some(&record) => assert_eq!(
                    command,
                    [3 + record, 1, 2 * record, 0, record],
                    "stream {stream} place {k}"
                ),
                None => assert_eq!(command, [0; 5], "stream {stream} past its count"),
            }
        }
        drawn += list.len();
    }
    assert!(
        drawn > records as usize / 3,
        "the lists hold most of the drawn records"
    );
    assert!(
        lists
            .iter()
            .take(MAX_SHADER_BUCKETS)
            .filter(|l| !l.is_empty())
            .count()
            == BUCKETS as usize,
        "every bucket's list holds draws"
    );

    // SAFETY: the queue idled, so nothing created above is in use; each handle is destroyed
    // once.
    unsafe { device.destroy_command_pool(cmd_pool, None) };
    // SAFETY: as above.
    unsafe { device.destroy_descriptor_pool(pool, None) };
    // SAFETY: as above.
    unsafe { device.destroy_pipeline(scan, None) };
    // SAFETY: as above.
    unsafe { device.destroy_pipeline(scatter, None) };
    // SAFETY: as above.
    unsafe { device.destroy_pipeline_layout(pipeline_layout, None) };
    // SAFETY: as above.
    unsafe { device.destroy_descriptor_set_layout(set_layout, None) };
    drop((variants_buf, args_buf, counts_buf, status_buf, lists_buf));
    alloc.destroy();
}

// Records along x, a unit box each from `r - 150`, some hidden and some never
// frustum-tested, against a light frustum keeping `-50 <= x <= 50`.
fn shadow_records(records: u32) -> (Vec<GpuObjectData>, Vec<GpuDrawArgs>, Vec<u32>) {
    let mut objects = Vec::new();
    let mut args = Vec::new();
    let mut expected = Vec::new();
    for r in 0..records {
        let x = r as f32 - 150.0;
        let visible = r % 7 != 3;
        let cullable = r % 11 != 5;
        let mut o: GpuObjectData = bytemuck::Zeroable::zeroed();
        o.bb_min = [x, 0.0, 0.0];
        o.bb_max = [x + 1.0, 1.0, 1.0];
        objects.push(o);
        args.push(GpuDrawArgs {
            flags: draw_args_flags(visible, true, cullable),
            ..draw_args(1)[0]
        });
        let inside = x + 1.0 >= -50.0 && x <= 50.0;
        expected.push(if visible && (inside || !cullable) {
            CullStatus::DRAWN
        } else {
            CullStatus::CULLED
        });
    }
    (objects, args, expected)
}

#[test]
fn the_shadow_cull_counts_what_it_keeps_as_the_spec_does() {
    concinnity_shader::require_dxc!();
    let Some(gpu) = test_gpu() else {
        return;
    };
    let device = &gpu.device;
    let alloc = DeviceAllocator::new(&gpu.instance, gpu.physical_device, device, 2);
    let records = 300;
    let (objects, args, expected) = shadow_records(records);
    let layout = StreamLayout {
        regions: &GpuCullVariants::default().regions,
        bucket_count: 1,
        skinned_base: skinned_base(records),
        records,
    };
    let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
    let words = group_count_words(records);
    let objects_buf = buffer(&alloc, bytemuck::cast_slice(&objects), storage);
    let args_buf = buffer(&alloc, bytemuck::cast_slice(&args), storage);
    let counts_buf = buffer(
        &alloc,
        bytemuck::cast_slice(&vec![u32::MAX; words]),
        storage,
    );
    let status_buf = buffer(
        &alloc,
        bytemuck::cast_slice(&vec![u32::MAX; records as usize]),
        storage,
    );

    let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = super::init::shadow_cull_set()
        .iter()
        .map(|&(binding, ty, stages)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(stages)
        })
        .collect();
    // SAFETY: every create-info below and the slices it borrows are live for its call, and each
    // handle it names belongs to this device.
    let set_layout = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }
    .expect("set layout");
    let push = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .size(u32::try_from(size_of::<CullParams>()).expect("push bytes"));
    // SAFETY: as above.
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&set_layout))
                .push_constant_ranges(std::slice::from_ref(&push)),
            None,
        )
    }
    .expect("pipeline layout");
    let cull = pipeline(
        &gpu,
        pipeline_layout,
        &CULL_SHADOW.compile(false).expect("shadow cull"),
    );
    let sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 4,
    }];
    // SAFETY: as above.
    let pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&sizes)
                .max_sets(1),
            None,
        )
    }
    .expect("pool");
    // SAFETY: as above.
    let set = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(std::slice::from_ref(&set_layout)),
        )
    }
    .expect("set")[0];
    let infos =
        [&objects_buf, &args_buf, &counts_buf, &status_buf].map(|b| vk::DescriptorBufferInfo {
            buffer: b.buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        });
    let writes: Vec<vk::WriteDescriptorSet<'_>> = infos
        .iter()
        .enumerate()
        .map(|(b, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(u32::try_from(b).expect("binding"))
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        })
        .collect();
    // SAFETY: the writes name the live set and buffers above.
    unsafe { device.update_descriptor_sets(&writes, &[]) };

    let wide = |n: [f32; 3]| [n[0], n[1], n[2], 1.0e6];
    let params = CullParams {
        planes: [
            [1.0, 0.0, 0.0, 50.0],
            [-1.0, 0.0, 0.0, 50.0],
            wide([0.0, 1.0, 0.0]),
            wide([0.0, -1.0, 0.0]),
            wide([0.0, 0.0, 1.0]),
            wide([0.0, 0.0, -1.0]),
        ],
        cam_pos: [0.0; 3],
        object_count: records,
        bucket_count: 1,
        skinned_base: skinned_base(records),
    };
    // SAFETY: family 0 is the one queue family `test_gpu` created a queue in.
    let cmd_pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(0),
            None,
        )
    }
    .expect("command pool");
    // SAFETY: the pool is live.
    let cmd = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .command_buffer_count(1),
        )
    }
    .expect("command buffer")[0];
    // Every call below records into or submits `cmd`, whose pipeline, set and buffers stay live
    // until the queue idles.
    // SAFETY: as above; `cmd` is a fresh command buffer.
    unsafe { device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()) }
        .expect("begin");
    // SAFETY: as above.
    unsafe { device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, cull) };
    // SAFETY: as above; the set was written against the layout's only set.
    unsafe {
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &[set],
            &[],
        );
    }
    // SAFETY: as above; the push range covers the params.
    unsafe {
        device.cmd_push_constants(
            cmd,
            pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&params),
        );
    }
    // SAFETY: as above.
    unsafe { device.cmd_dispatch(cmd, records.div_ceil(64), 1, 1) };
    // SAFETY: as above.
    unsafe { device.end_command_buffer(cmd) }.expect("end");
    // SAFETY: family 0, index 0 is the one queue `test_gpu` created.
    let queue = unsafe { device.get_device_queue(0, 0) };
    let commands_to_run = [cmd];
    let submit = vk::SubmitInfo::default().command_buffers(&commands_to_run);
    // SAFETY: as above; the submit info borrows a live command buffer.
    unsafe { device.queue_submit(queue, &[submit], vk::Fence::null()) }.expect("submit");
    // SAFETY: the queue belongs to this device.
    unsafe { device.queue_wait_idle(queue) }.expect("wait");

    let read = |buffer: &PooledBuffer, len: usize| -> Vec<u32> {
        // SAFETY: the queue idled, so the GPU no longer writes the host-visible, coherent buffer,
        // which holds at least `4 * len` bytes.
        let bytes = unsafe { std::slice::from_raw_parts(buffer.mapped_ptr(), 4 * len) };
        bytes
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect()
    };
    let status = read(&status_buf, records as usize);
    assert_eq!(status, expected, "every record's outcome, in bucket 0");
    let mut want = group_counts(&status, CullStatus::DRAWN, &layout);
    // A stream the scan does not walk is never written.
    let groups = want.len() / DRAW_STREAMS;
    for (word, count) in want.iter_mut().enumerate() {
        let stream = u32::try_from(word / groups).expect("stream");
        if !layout.streams().any(|s| s == stream) {
            *count = u32::MAX;
        }
    }
    assert_eq!(
        read(&counts_buf, words),
        want,
        "each group's kept records per stream"
    );
    assert!(expected.contains(&CullStatus::CULLED) && expected.contains(&CullStatus::DRAWN));

    // SAFETY: the queue idled, so nothing created above is in use; each handle is destroyed
    // once.
    unsafe { device.destroy_command_pool(cmd_pool, None) };
    // SAFETY: as above.
    unsafe { device.destroy_descriptor_pool(pool, None) };
    // SAFETY: as above.
    unsafe { device.destroy_pipeline(cull, None) };
    // SAFETY: as above.
    unsafe { device.destroy_pipeline_layout(pipeline_layout, None) };
    // SAFETY: as above.
    unsafe { device.destroy_descriptor_set_layout(set_layout, None) };
    drop((objects_buf, args_buf, counts_buf, status_buf));
    alloc.destroy();
}
