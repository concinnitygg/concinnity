// The draw cull's variant rule as the GPU runs it, against its CPU spec
// (`GpuCullVariants::bucket`): the phase-1 kernel routes synthetic records
// under several views, and every record's bucket in its status word must be
// the one the spec picks.

use concinnity_core::math::cos;
use concinnity_engine::gfx::cull_status::CullStatus;
use concinnity_engine::gfx::render_types::{
    GpuCullVariants, GpuDrawArgs, GpuObjectData, GpuVariantRoute, VARIANT_DAYLIGHT, VARIANT_FAR,
    draw_args_bucket_bits, draw_args_flags, draw_args_route_bits,
};
use concinnity_engine::render::uniforms::DaylightBlock;
use concinnity_engine::render::uniforms::metal::CullUniforms;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer as _, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCommandQueue as _,
    MTLComputeCommandEncoder as _, MTLDevice, MTLPixelFormat, MTLResourceOptions, MTLSize,
    MTLTextureDescriptor,
};

use super::builtin_shaders::{CULL_PHASE1, compute_pipeline};
use super::encode::ComputeEncode;

const BUCKETS: u32 = 8;
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

// Run the phase-1 kernel over `records` seen from `eye` under `variants` and
// read back each record's status word.
fn run(
    device: &ProtocolObject<dyn MTLDevice>,
    records: &[Record],
    variants: &GpuCullVariants,
    eye: [f32; 3],
) -> Vec<u32> {
    let pipeline = compute_pipeline(device, &CULL_PHASE1, false).expect("cull kernel");
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
        skinned_base: u32::try_from(records.len()).expect("records"),
        cascade_base: 0,
        bucket_count: BUCKETS,
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
    enc.set_texture(&hiz, 0);
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: records.len(),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    // SAFETY: the command buffer completed, so the GPU no longer writes the shared buffer,
    // which holds one `u32` per record.
    let words = unsafe {
        std::slice::from_raw_parts(status.contents().as_ptr().cast::<u32>(), records.len())
    };
    words.to_vec()
}

#[test]
fn the_kernel_routes_every_record_as_the_spec_does() {
    concinnity_shader::require_dxc!();
    let Some(device) = objc2_metal::MTLCreateSystemDefaultDevice() else {
        return;
    };
    let records = records();
    let eye = [0.0, 2.0, 0.0];
    let mut picked = std::collections::BTreeSet::new();
    for (far_start, daylight) in [
        (60.0, sky(0.0)),
        (0.0, sky(0.0)),
        (500.0, sky(0.0)),
        (60.0, sky(0.3)),
        (60.0, DaylightBlock::OFF),
    ] {
        let mut variants = GpuCullVariants::default();
        variants.routes[0] = GpuVariantRoute::new([4, 5, 6, 7], VARIANT_FAR | VARIANT_DAYLIGHT);
        variants.set_view(far_start, &daylight);
        let words = run(&device, &records, &variants, eye);
        for (i, (r, &word)) in records.iter().zip(&words).enumerate() {
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
