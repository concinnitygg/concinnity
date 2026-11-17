// The receiver reduction as the GPU runs it, against the same arithmetic on
// the CPU: a synthetic depth over several tiles (one cut short on each axis,
// with cleared texels and every band populated) reduces to the bounds the CPU
// folds texel by texel, and the merge stamps the serial.

use concinnity_engine::gfx::render_types::{
    NUM_SHADOW_CASCADES, ReceiverBand, ReceiverBoundsParams, receiver_tiles,
};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCommandQueue as _, MTLDevice as _, MTLOrigin,
    MTLPixelFormat, MTLRegion, MTLSize, MTLStorageMode, MTLTexture as _, MTLTextureDescriptor,
    MTLTextureUsage,
};

use super::ShadowReceiversGpu;

const WIDTH: u32 = 150;
const HEIGHT: u32 = 90;

// Device depth for texel (x, y): cleared over the top rows, rising across the
// rest.
fn depth_at(x: u32, y: u32) -> f32 {
    if y < 12 || (x + y).is_multiple_of(37) {
        0.0
    } else {
        0.05 + 0.9 * (x as f32 / WIDTH as f32) * (y as f32 / HEIGHT as f32)
    }
}

// Light space taken as NDC x, y and device depth, and the view depth as the
// device depth, so each band holds the texels whose depth falls in it.
fn params() -> ReceiverBoundsParams {
    let mut to_light_space = [[0.0; 4]; 4];
    to_light_space[0][0] = 1.0;
    to_light_space[1][1] = 1.0;
    to_light_space[2][2] = 1.0;
    to_light_space[2][3] = 1.0;
    ReceiverBoundsParams {
        to_light_space,
        w_row: [0.0, 0.0, 0.0, 1.0],
        band_near: [0.0, 0.2, 0.5, 2.0],
        band_far: [0.3, 0.6, 1.0, 1.0],
        width: WIDTH,
        height: HEIGHT,
        groups_x: receiver_tiles(WIDTH, HEIGHT)[0],
        serial: 41,
    }
}

// The kernel's fold, on the CPU.
fn expected(p: &ReceiverBoundsParams) -> [ReceiverBand; NUM_SHADOW_CASCADES] {
    let mut bands = [ReceiverBand::EMPTY; NUM_SHADOW_CASCADES];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let d = depth_at(x, y);
            if d <= 0.0 {
                continue;
            }
            let u = (x as f32 + 0.5) / WIDTH as f32;
            let v = (y as f32 + 0.5) / HEIGHT as f32;
            let at = [u * 2.0 - 1.0, -(v * 2.0 - 1.0), d, 1.0];
            let w: f32 = (0..4).map(|k| p.w_row[k] * at[k]).sum();
            let q: [f32; 4] = core::array::from_fn(|r| {
                (0..4).map(|k| p.to_light_space[k][r] * at[k]).sum::<f32>() / w
            });
            for (b, band) in bands.iter_mut().enumerate() {
                if q[3] >= p.band_near[b] && q[3] <= p.band_far[b] {
                    for (k, v) in q.iter().enumerate() {
                        band.lo[k] = band.lo[k].min(*v);
                        band.hi[k] = band.hi[k].max(*v);
                    }
                }
            }
        }
    }
    bands
}

fn depth_texture(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
) -> objc2::rc::Retained<ProtocolObject<dyn objc2_metal::MTLTexture>> {
    let desc = MTLTextureDescriptor::new();
    desc.setPixelFormat(MTLPixelFormat::R32Float);
    // SAFETY: the extent is within every device's limits.
    unsafe { desc.setWidth(WIDTH as usize) };
    // SAFETY: as for the width.
    unsafe { desc.setHeight(HEIGHT as usize) };
    desc.setStorageMode(MTLStorageMode::Shared);
    desc.setUsage(MTLTextureUsage::ShaderRead);
    let texture = device.newTextureWithDescriptor(&desc).expect("texture");
    let texels: Vec<f32> = (0..HEIGHT)
        .flat_map(|y| (0..WIDTH).map(move |x| depth_at(x, y)))
        .collect();
    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: WIDTH as usize,
            height: HEIGHT as usize,
            depth: 1,
        },
    };
    // SAFETY: `region` covers the whole single-level texture, and `texels` holds one f32 per
    // texel at a row stride of `WIDTH` floats for the call.
    unsafe {
        texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
            region,
            0,
            std::ptr::NonNull::new(texels.as_ptr().cast_mut().cast()).expect("texels"),
            WIDTH as usize * 4,
        );
    }
    texture
}

#[test]
fn the_kernel_folds_every_band_as_the_cpu_does() {
    concinnity_shader::require_dxc!();
    let Some(device) = objc2_metal::MTLCreateSystemDefaultDevice() else {
        return;
    };
    let gpu = ShadowReceiversGpu::new(&device, false, 1, [WIDTH, HEIGHT]).expect("reduction");
    let p = params();
    assert!(gpu.holds(&p));
    let depth = depth_texture(&device);
    let queue = device.newCommandQueue().expect("queue");
    let cmd = queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("encoder");
    gpu.dispatch(&enc, &depth, &p, 0);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let got = gpu.read(0).expect("slot");
    assert_eq!(got.serial, 41);
    let want = expected(&p);
    assert!(want.iter().take(3).all(|b| !b.is_empty()));
    assert!(got.bands[3].is_empty(), "a band not in use stays empty");
    for (b, (g, w)) in got.bands.iter().zip(&want).enumerate() {
        for k in 0..4 {
            for (gv, wv) in [(g.lo[k], w.lo[k]), (g.hi[k], w.hi[k])] {
                assert!(
                    (gv - wv).abs() <= 1e-5 * wv.abs().max(1.0),
                    "band {b} lane {k}: {gv} against {wv}"
                );
            }
        }
    }
}
