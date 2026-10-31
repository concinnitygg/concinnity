// The draw cull as the GPU runs it, against its CPU specs: the early kernel
// routes synthetic records under several views, and every record's bucket in
// its status word must be the one `GpuCullVariants::bucket` picks; the scan
// then lays out each stream's list, and the group counts it leaves and the
// execution ranges it writes must be the ones `streams` computes.

use concinnity_core::math::cos;
use concinnity_engine::gfx::cull_status::CullStatus;
use concinnity_engine::gfx::render_types::{
    BucketSpans, DRAW_STREAMS, GpuCompactParams, GpuCullVariants, GpuDrawArgs, GpuExecutionRange,
    GpuObjectData, GpuVariantRoute, StreamLayout, VARIANT_DAYLIGHT, VARIANT_FAR, bucket_regions,
    draw_args_bucket_bits, draw_args_flags, draw_args_route_bits, group_count_words, group_counts,
    scan_group_counts,
};
use concinnity_engine::render::uniforms::DaylightBlock;
use concinnity_engine::render::uniforms::metal::CullUniforms;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer as _, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCommandQueue as _,
    MTLComputeCommandEncoder as _, MTLDevice, MTLPixelFormat, MTLResourceOptions, MTLSize,
    MTLTextureDescriptor,
};

use super::builtin_shaders::{CULL_EARLY, CULL_SCAN, compute_pipeline};
use super::encode::ComputeEncode;

const BUCKETS: u32 = 8;
// The last records stand in for the skinned tail.
const SKINNED_BASE: u32 = 60;
const BASE: u32 = 3;
const RADIUS: f32 = 50_000.0;

struct Record {
    args: GpuDrawArgs,
    bb: ([f32; 3], [f32; 3]),
}

// Records on and around the ground near a 50 km planet's top, clear of the
// thresholds the views below draw so float rounding cannot split the two
// evaluations: boxes wholly near the eye, straddling 60 m, and either side of
// where the sky starts to reach (~430 m), each without a route, routed with a
// far band and routed without one; and records the cull always draws.
fn records() -> Vec<Record> {
    let mut out = Vec::new();
    for (center, half) in [
        (0.0, 5.0),
        (0.0, 40.0),
        (150.0, 5.0),
        (300.0, 40.0),
        (400.0, 5.0),
        (400.0, 40.0),
        (470.0, 5.0),
        (2_000.0, 40.0),
    ] {
        let bb = ([center - half, 0.0, -half], [center + half, 1.0, half]);
        for (route, far) in [(None, false), (Some(0), true), (Some(0), false)] {
            let flags = draw_args_flags(true, true, true)
                | draw_args_bucket_bits(BASE)
                | draw_args_route_bits(route, far);
            out.push(Record {
                args: args(flags),
                bb,
            });
        }
    }
    // Records the cull does not frustum-test, with no box and with a finite
    // one near the eye.
    for bb in [
        ([f32::NAN; 3], [f32::NAN; 3]),
        ([-5.0, 0.0, -5.0], [5.0, 1.0, 5.0]),
    ] {
        for far in [true, false] {
            let flags = draw_args_flags(true, true, false)
                | draw_args_bucket_bits(BASE)
                | draw_args_route_bits(Some(0), far);
            out.push(Record {
                args: args(flags),
                bb,
            });
        }
    }
    out
}

fn args(flags: u32) -> GpuDrawArgs {
    GpuDrawArgs {
        index_count: 3,
        index_offset: 0,
        base_vertex: 0,
        flags,
    }
}

fn sky(camera_reach: f32) -> DaylightBlock {
    DaylightBlock {
        center: [0.0, -RADIUS, 0.0, 1.0],
        camera_up: [0.0, 1.0, 0.0, camera_reach],
        reach: [
            cos(0.5_f32.to_radians()),
            cos(3.0_f32.to_radians()),
            RADIUS,
            5_000.0,
        ],
        ..DaylightBlock::OFF
    }
}

fn shared_buffer<T: bytemuck::NoUninit>(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[T],
) -> objc2::rc::Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>> {
    let bytes = bytemuck::cast_slice::<T, u8>(data);
    let ptr = std::ptr::NonNull::new(bytes.as_ptr().cast_mut().cast()).expect("data");
    // SAFETY: `ptr` points at `bytes.len()` initialized bytes for the call, which
    // Metal copies into the new buffer.
    unsafe {
        device.newBufferWithBytes_length_options(
            ptr,
            bytes.len().max(4),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("buffer")
}

// Read `len` words back from a shared buffer the GPU finished writing.
fn words(buffer: &ProtocolObject<dyn objc2_metal::MTLBuffer>, len: usize) -> Vec<u32> {
    // SAFETY: the command buffer that wrote `buffer` completed, so the GPU no longer writes the
    // shared buffer, which holds at least `len` words.
    unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u32>(), len) }.to_vec()
}

// What one cull left: each record's status word, the group counts as the scan
// left them, and every stream's execution range.
struct Culled {
    status: Vec<u32>,
    group_counts: Vec<u32>,
    ranges: Vec<u32>,
}

fn groups(enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>, width: usize) {
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
}

// Run the early kernel over `records` seen from `eye` under `variants`, then
// the scan, and read back what they wrote.
fn run(
    device: &ProtocolObject<dyn MTLDevice>,
    records: &[Record],
    variants: &GpuCullVariants,
    eye: [f32; 3],
) -> Culled {
    let pipeline = compute_pipeline(device, &CULL_EARLY, false).expect("cull kernel");
    let scan = compute_pipeline(device, &CULL_SCAN, false).expect("scan kernel");
    let objects: Vec<GpuObjectData> = records
        .iter()
        .map(|r| {
            let mut o: GpuObjectData = bytemuck::Zeroable::zeroed();
            o.bb_min = r.bb.0;
            o.bb_max = r.bb.1;
            o
        })
        .collect();
    let args: Vec<GpuDrawArgs> = records.iter().map(|r| r.args).collect();
    let objects = shared_buffer(device, &objects);
    let args = shared_buffer(device, &args);
    let status = shared_buffer(device, &vec![u32::MAX; records.len()]);
    let count_words = group_count_words(u32::try_from(records.len()).expect("records"));
    let counts = shared_buffer(device, &vec![u32::MAX; count_words]);
    let ranges = shared_buffer(device, &vec![u32::MAX; 2 * DRAW_STREAMS]);
    let desc = MTLTextureDescriptor::new();
    desc.setPixelFormat(MTLPixelFormat::R32Float);
    // SAFETY: a width of 1 is within every device's limits.
    unsafe { desc.setWidth(1) };
    // SAFETY: as for the width.
    unsafe { desc.setHeight(1) };
    let hiz = device.newTextureWithDescriptor(&desc).expect("texture");
    let uniforms = CullUniforms {
        planes: [[0.0; 4]; 6],
        cam_pos: [eye[0], eye[1], eye[2], 0.0],
        prev_view_proj: [[0.0; 4]; 4],
        hiz_size: [1.0, 1.0],
        hiz_mip_count: 1,
        hiz_enabled: 0,
        object_count: u32::try_from(records.len()).expect("records"),
        skinned_base: SKINNED_BASE,
        exclude_first: 0,
        bucket_count: BUCKETS,
        exclude_end: 0,
        _pad: [0; 3],
    };
    let queue = device.newCommandQueue().expect("queue");
    let cmd = queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("encoder");
    enc.set_pipeline(&pipeline);
    enc.set_buffer(&objects, 0, 0);
    enc.set_buffer(&args, 0, 1);
    enc.set_value(&uniforms, 2);
    enc.set_buffer(&status, 0, 5);
    enc.set_value(variants, 8);
    enc.set_buffer(&counts, 0, 9);
    enc.set_buffer(&ranges, 0, 10);
    enc.set_texture(&hiz, 0);
    groups(&enc, records.len().div_ceil(64));
    enc.set_pipeline(&scan);
    enc.set_value(
        &GpuCompactParams {
            object_count: uniforms.object_count,
            bucket_count: BUCKETS,
            skinned_base: SKINNED_BASE,
            drawn: CullStatus::DRAWN,
            ..Default::default()
        },
        11,
    );
    groups(&enc, BUCKETS as usize + 1);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    Culled {
        status: words(&status, records.len()),
        group_counts: words(&counts, count_words),
        ranges: words(&ranges, 2 * DRAW_STREAMS),
    }
}

#[test]
fn the_kernel_routes_every_record_as_the_spec_does() {
    concinnity_shader::require_dxc!();
    let Some(device) = objc2_metal::MTLCreateSystemDefaultDevice() else {
        return;
    };
    // Several 64-record groups, the last records standing in for the skinned
    // tail.
    let records: Vec<Record> = (0..5).flat_map(|_| records()).collect();
    assert!(records.len() > SKINNED_BASE as usize);
    let records_u32 = u32::try_from(records.len()).expect("records");
    let eye = [0.0, 2.0, 0.0];
    let mut picked = std::collections::BTreeSet::new();
    for (far_start, daylight) in [
        (60.0, sky(0.0)),
        (0.0, sky(0.0)),
        (500.0, sky(0.0)),
        (60.0, sky(0.3)),
        (60.0, DaylightBlock::OFF),
    ] {
        // Regions that leave some records outside some buckets, so the
        // packing drops them as the spec does.
        let spans: Vec<BucketSpans> = (0..BUCKETS as usize)
            .map(|b| BucketSpans {
                records: if b == 0 {
                    0..records.len()
                } else {
                    (b * 7)..records.len() - b * 3
                },
                runtime: 0..0,
            })
            .collect();
        let mut variants = GpuCullVariants {
            regions: bucket_regions(&spans),
            ..GpuCullVariants::default()
        };
        variants.routes[0] = GpuVariantRoute::new([4, 5, 6, 7], VARIANT_FAR | VARIANT_DAYLIGHT);
        variants.set_view(far_start, &daylight);
        let culled = run(&device, &records, &variants, eye);
        let layout = StreamLayout {
            regions: &variants.regions,
            bucket_count: BUCKETS,
            skinned_base: SKINNED_BASE,
            records: records_u32,
        };
        let mut counts = group_counts(&culled.status, CullStatus::DRAWN, &layout);
        let totals = scan_group_counts(&mut counts, &layout);
        // A stream the scan does not walk is never written.
        let groups = counts.len() / DRAW_STREAMS;
        for (word, count) in counts.iter_mut().enumerate() {
            let stream = u32::try_from(word / groups).expect("stream");
            if !layout.streams().any(|s| s == stream) {
                *count = u32::MAX;
            }
        }
        assert_eq!(
            culled.group_counts, counts,
            "group places at far start {far_start}"
        );
        let mut ranges = vec![u32::MAX; 2 * DRAW_STREAMS];
        for stream in layout.streams() {
            let s = stream as usize;
            let range = GpuExecutionRange {
                location: layout.slots(stream).first,
                length: totals[s],
            };
            ranges[2 * s] = range.location;
            ranges[2 * s + 1] = range.length;
        }
        assert_eq!(
            culled.ranges, ranges,
            "list ranges at far start {far_start}"
        );
        assert!(
            totals.iter().filter(|&&t| t > 0).count() > 2,
            "several lists hold draws"
        );
        for (i, (r, &word)) in records.iter().zip(&culled.status).enumerate() {
            let spec = variants.bucket(&r.args, r.bb, eye).min(BUCKETS - 1);
            assert_eq!(
                CullStatus::bucket(word),
                spec,
                "record {i} ({:?}) at far start {far_start}, sky {:?}",
                r.bb,
                daylight.camera_up
            );
            picked.insert(spec);
        }
    }
    assert_eq!(
        picked.into_iter().collect::<Vec<_>>(),
        [BASE, 4, 5, 6, 7],
        "every bucket is exercised"
    );
}
