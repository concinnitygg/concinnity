// The view sampling as the GPU runs it, against the same scan on the CPU: a
// synthetic depth over cells cut short on each axis, a row of cleared cells
// and a tie for the nearest texel samples to the cells the CPU reads off it,
// each stamped with the serial.

use concinnity_engine::gfx::render_types::{ViewSampleCell, ViewSampleParams};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCommandQueue as _, MTLDevice as _, MTLOrigin,
    MTLPixelFormat, MTLRegion, MTLSize, MTLStorageMode, MTLTexture as _, MTLTextureDescriptor,
    MTLTextureUsage,
};

use super::ViewSamplesGpu;

const WIDTH: u32 = 150;
const HEIGHT: u32 = 90;
const CELL: u32 = 20;
// The two texels tied for the nearest in the cell at column 2, row 2.
const TIED: [[u32; 2]; 2] = [[52, 41], [45, 47]];

// Device depth for texel (x, y): cleared over the first row of cells, rising
// across the rest, with two texels tied nearest in one cell.
fn depth_at(x: u32, y: u32) -> f32 {
    if y < CELL {
        0.0
    } else if TIED.contains(&[x, y]) {
        0.99
    } else {
        0.05 + 0.9 * (x as f32 / WIDTH as f32) * (y as f32 / HEIGHT as f32)
    }
}

fn params() -> ViewSampleParams {
    ViewSampleParams {
        width: WIDTH,
        height: HEIGHT,
        cell: CELL,
        columns: WIDTH.div_ceil(CELL),
        rows: HEIGHT.div_ceil(CELL),
        serial: 41,
        spot: [7, 13],
    }
}

fn pack([x, y]: [u32; 2]) -> u32 {
    (y << 16) | x
}

// The kernel's scan, on the CPU: the depth at each cell's spot, and its
// nearest texel, the greatest packed one among ties, or the spot where the
// cell is cleared.
fn expected(p: &ViewSampleParams) -> Vec<ViewSampleCell> {
    let mut cells = Vec::new();
    for row in 0..p.rows {
        for column in 0..p.columns {
            let origin = [column * p.cell, row * p.cell];
            let spot = [
                (origin[0] + p.spot[0]).min(p.width - 1),
                (origin[1] + p.spot[1]).min(p.height - 1),
            ];
            // The kernel's atomic max: the depth's bits, then the packed texel.
            let mut nearest = (0_u32, 0_u32);
            for y in origin[1]..(origin[1] + p.cell).min(p.height) {
                for x in origin[0]..(origin[0] + p.cell).min(p.width) {
                    nearest = nearest.max((depth_at(x, y).to_bits(), pack([x, y])));
                }
            }
            cells.push(ViewSampleCell {
                depth: depth_at(spot[0], spot[1]),
                nearest: f32::from_bits(nearest.0),
                texel: if nearest.0 > 0 { nearest.1 } else { pack(spot) },
                serial: p.serial,
            });
        }
    }
    cells
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
fn the_kernel_samples_every_cell_as_the_cpu_does() {
    concinnity_shader::require_dxc!();
    let Some(device) = objc2_metal::MTLCreateSystemDefaultDevice() else {
        return;
    };
    let gpu = ViewSamplesGpu::new(&device, false, 1).expect("sampling");
    let p = params();
    let depth = depth_texture(&device);
    let queue = device.newCommandQueue().expect("queue");
    let cmd = queue.commandBuffer().expect("command buffer");
    let enc = cmd.computeCommandEncoder().expect("encoder");
    gpu.dispatch(&enc, &depth, &p, 0);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let want = expected(&p);
    let got = &gpu.read(0).expect("slot")[..want.len()];
    // The row of cleared cells falls back to its spot, the tie to the
    // greater packed texel, and the last row and column are cut short.
    assert!(
        want.iter()
            .take(p.columns as usize)
            .all(|c| c.nearest == 0.0)
    );
    let tied = &want[(2 * p.columns + 2) as usize];
    assert_eq!(tied.texel, pack(TIED[1]));
    assert_eq!(got, &want[..]);
}
