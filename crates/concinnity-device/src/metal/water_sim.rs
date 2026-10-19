//! The simulated ocean on Metal (see `concinnity_core::render::water::ocean`):
//! the cascade arrays the water draw samples, and the `WaterSim` node that
//! fills them each frame from the shared `water_fft.hlsl` kernels. Every
//! resource here is device-created and hazard-tracked, and the kernels run
//! in one serial compute encoder, so each dispatch sees the last one's writes.
//! The foam buffer outlives frames; the next frame's node waits for this
//! one's through `graph_events::plan_frame`.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::render_graph::PassId;
use concinnity_core::render::uniforms::ocean::{
    OCEAN_FFT_THREADS, OCEAN_MIP_GROUP, OceanFftParams, ocean_mip_groups, ocean_mip_levels,
};
use concinnity_core::render::water::ocean::OceanPlan;
use concinnity_core::render::water::ocean::foam::FoamClock;
use concinnity_core::render::water::ocean::initial::initial_spectrum;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, ns_string};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLComputeCommandEncoder, MTLComputePassDescriptor,
    MTLComputePipelineState, MTLDevice, MTLPixelFormat, MTLResourceOptions, MTLSamplerAddressMode,
    MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerMipFilter, MTLSamplerState, MTLSize,
    MTLStorageMode, MTLTexture, MTLTextureType, MTLTextureUsage,
};
use std::cell::Cell;

use super::builtin_shaders::{
    WATER_FFT_COLUMNS, WATER_FFT_COLUMNS_FOAM, WATER_FFT_ROWS, WATER_MIP, compute_pipeline,
};
use super::context::MtlContext;
use super::descriptors::TextureDesc;
use super::encode::ComputeEncode;
use super::error::allocation_failed;
use super::scoped_encoder::ScopedEncoder;

type Texture = Retained<ProtocolObject<dyn MTLTexture>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Kernel = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

const CASCADE_FORMAT: MTLPixelFormat = MTLPixelFormat::RGBA16Float;

// Slots pinned by `water_fft.hlsl`'s registers.
const PARAMS_INDEX: usize = 0;
const SPECTRUM_INDEX: usize = 5;
const ROWS_INDEX: usize = 6;
const FOAM_INDEX: usize = 7;
const SRC_DISPLACEMENT_INDEX: usize = 0;
const SRC_SLOPES_INDEX: usize = 1;
const DST_DISPLACEMENT_INDEX: usize = 2;
const DST_SLOPES_INDEX: usize = 3;

// The water draw's ocean slots, pinned by `water.hlsl`'s registers in both
// stages.
pub(in crate::metal) const OCEAN_DISPLACEMENT_TEXTURE_INDEX: usize = 9;
pub(in crate::metal) const OCEAN_SLOPES_TEXTURE_INDEX: usize = 10;
pub(in crate::metal) const OCEAN_SAMPLER_INDEX: usize = 6;

// What the water draw samples: both cascade arrays and their sampler.
pub(in crate::metal) struct OceanArrays {
    pub(in crate::metal) displacement: Texture,
    pub(in crate::metal) slopes: Texture,
    pub(in crate::metal) sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
}

// The world's simulated oceans, or a stand-in pair of arrays for a world
// whose water is all summed, which the water draw binds and never samples.
pub(in crate::metal) struct OceanSim {
    pub(in crate::metal) arrays: OceanArrays,
    kernels: Option<OceanKernels>,
}

struct OceanKernels {
    plan: OceanPlan,
    rows: Kernel,
    columns: Kernel,
    // The columns kernel that also steps the foam, for a whitecapping sea.
    columns_foam: Kernel,
    mip: Kernel,
    // Every layer's initial spectrum, layer after layer.
    spectrum: Buffer,
    // Every layer's row transforms, two float4 per texel.
    row_data: Buffer,
    // Every layer's foam, one float per texel, kept across frames.
    foam: Buffer,
    foam_clock: Cell<FoamClock>,
    // One view per mip level of each array, every layer.
    displacement_mips: Vec<Texture>,
    slope_mips: Vec<Texture>,
}

impl OceanSim {
    pub(in crate::metal) fn build(
        device: &ProtocolObject<dyn MTLDevice>,
        plan: &OceanPlan,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let sampler = build_sampler(device)?;
        if plan.is_empty() {
            return Ok(Self {
                arrays: OceanArrays {
                    displacement: cascade_array(device, 1, 1, 1)?,
                    slopes: cascade_array(device, 1, 1, 1)?,
                    sampler,
                },
                kernels: None,
            });
        }
        let size = plan.quality.size;
        let layers = plan.layer_count();
        let mips = ocean_mip_levels(size);
        let displacement = cascade_array(device, size, layers, mips)?;
        let slopes = cascade_array(device, size, layers, mips)?;
        let displacement_mips = (0..mips)
            .map(|m| mip_view(&displacement, m, layers))
            .collect::<RenderResult<_>>()?;
        let slope_mips = (0..mips)
            .map(|m| mip_view(&slopes, m, layers))
            .collect::<RenderResult<_>>()?;
        let texels: Vec<[f32; 4]> = plan
            .simulations
            .iter()
            .flat_map(|sim| {
                (0..sim.cascades.count())
                    .flat_map(move |c| initial_spectrum(&sim.sea, &sim.cascades, c))
            })
            .collect();
        let spectrum = buffer_with(device, bytemuck::cast_slice(&texels), "ocean spectrum")?;
        let row_bytes = texels.len() * 2 * size_of::<[f32; 4]>();
        let row_data = device
            .newBufferWithLength_options(row_bytes, MTLResourceOptions::StorageModePrivate)
            .ok_or_else(|| allocation_failed("the ocean row transforms"))?;
        let foam = device
            .newBufferWithLength_options(
                texels.len() * size_of::<f32>(),
                MTLResourceOptions::StorageModePrivate,
            )
            .ok_or_else(|| allocation_failed("the ocean foam"))?;
        Ok(Self {
            arrays: OceanArrays {
                displacement,
                slopes,
                sampler,
            },
            kernels: Some(OceanKernels {
                plan: plan.clone(),
                rows: compute_pipeline(device, &WATER_FFT_ROWS, hot_reload)?,
                columns: compute_pipeline(device, &WATER_FFT_COLUMNS, hot_reload)?,
                columns_foam: compute_pipeline(device, &WATER_FFT_COLUMNS_FOAM, hot_reload)?,
                mip: compute_pipeline(device, &WATER_MIP, hot_reload)?,
                spectrum,
                row_data,
                foam,
                foam_clock: Cell::new(FoamClock::default()),
                displacement_mips,
                slope_mips,
            }),
        })
    }

    // Whether any body simulates an ocean, which puts `WaterSim` in the graph.
    pub(in crate::metal) fn simulates(&self) -> bool {
        self.kernels.is_some()
    }
}

// A `layers`-deep array of `size`-square cascades with `mips` levels.
fn cascade_array(
    device: &ProtocolObject<dyn MTLDevice>,
    size: u32,
    layers: u32,
    mips: u32,
) -> RenderResult<Texture> {
    let desc = TextureDesc {
        kind: MTLTextureType::Type2DArray,
        format: CASCADE_FORMAT,
        width: size as usize,
        height: size as usize,
        array_length: layers as usize,
        mip_count: mips as usize,
        usage: MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
        storage: MTLStorageMode::Private,
        ..Default::default()
    }
    .build();
    device
        .newTextureWithDescriptor(&desc)
        .ok_or_else(|| allocation_failed("an ocean cascade array"))
}

// Mip `level` of `array`, every layer, for a kernel to read or write.
fn mip_view(array: &Texture, level: u32, layers: u32) -> RenderResult<Texture> {
    // SAFETY: the view covers one existing level and every layer of the array
    // and keeps its format, so it reinterprets nothing.
    unsafe {
        array.newTextureViewWithPixelFormat_textureType_levels_slices(
            CASCADE_FORMAT,
            MTLTextureType::Type2DArray,
            NSRange::new(level as usize, 1),
            NSRange::new(0, layers as usize),
        )
    }
    .ok_or_else(|| RenderError::Other("failed to view an ocean cascade mip".into()))
}

fn buffer_with(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
    label: &str,
) -> RenderResult<Buffer> {
    let ptr = std::ptr::NonNull::new(bytes.as_ptr() as *mut std::ffi::c_void)
        .ok_or_else(|| RenderError::Other(format!("{label}: source pointer is null")))?;
    // SAFETY: the pointer and length describe the live `bytes`, which Metal
    // copies into the new buffer before the call returns.
    unsafe {
        device.newBufferWithBytes_length_options(
            ptr,
            bytes.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| allocation_failed(label))
}

// Trilinear, anisotropic and wrapping: a cascade tiles the sea.
fn build_sampler(
    device: &ProtocolObject<dyn MTLDevice>,
) -> RenderResult<Retained<ProtocolObject<dyn MTLSamplerState>>> {
    let desc = MTLSamplerDescriptor::new();
    desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
    desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
    desc.setMipFilter(MTLSamplerMipFilter::Linear);
    desc.setSAddressMode(MTLSamplerAddressMode::Repeat);
    desc.setTAddressMode(MTLSamplerAddressMode::Repeat);
    desc.setMaxAnisotropy(16);
    device
        .newSamplerStateWithDescriptor(&desc)
        .ok_or_else(|| RenderError::Other("failed to create the ocean sampler".into()))
}

fn dispatch(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    groups: [u32; 3],
    threads: [u32; 2],
) {
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: groups[0] as usize,
            height: groups[1] as usize,
            depth: groups[2] as usize,
        },
        MTLSize {
            width: threads[0] as usize,
            height: threads[1] as usize,
            depth: 1,
        },
    );
}

impl MtlContext {
    // Encode the `WaterSim` node: for every simulation, evolve and transform
    // its rows, then its columns into mip 0, then reduce down the mips.
    pub(in crate::metal) fn encode_water_sim(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        time: f64,
    ) -> RenderResult<()> {
        let Some(k) = self.water.ocean.as_ref().and_then(|o| o.kernels.as_ref()) else {
            return Ok(());
        };
        let desc = MTLComputePassDescriptor::new();
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_compute(&desc, PassId::WaterSim);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .computeCommandEncoderWithDescriptor(&desc)
                .ok_or_else(|| RenderError::Other("failed to get ocean compute encoder".into()))?,
            ns_string!("water sim"),
        );
        enc.set_buffer(&k.spectrum, 0, SPECTRUM_INDEX);
        enc.set_buffer(&k.row_data, 0, ROWS_INDEX);
        enc.set_buffer(&k.foam, 0, FOAM_INDEX);
        let mut clock = k.foam_clock.get();
        let keep = clock.advance(time);
        k.foam_clock.set(clock);
        let fft = [OCEAN_FFT_THREADS, 1];
        for sim in &k.plan.simulations {
            let params = OceanFftParams::simulate(sim, time, keep);
            let [size, _, _, cascades] = params.grid;
            enc.set_value(&params, PARAMS_INDEX);
            enc.set_pipeline(&k.rows);
            dispatch(&enc, [size, cascades, 1], fft);
            enc.set_pipeline(if sim.foam.is_some() {
                &k.columns_foam
            } else {
                &k.columns
            });
            enc.set_texture(&k.displacement_mips[0], DST_DISPLACEMENT_INDEX);
            enc.set_texture(&k.slope_mips[0], DST_SLOPES_INDEX);
            dispatch(&enc, [size, cascades, 1], fft);
            enc.set_pipeline(&k.mip);
            for level in 1..k.displacement_mips.len() {
                let reduce = params.reduce(level as u32);
                let groups = ocean_mip_groups(reduce.mip_size);
                enc.set_value(&reduce, PARAMS_INDEX);
                enc.set_texture(&k.displacement_mips[level - 1], SRC_DISPLACEMENT_INDEX);
                enc.set_texture(&k.slope_mips[level - 1], SRC_SLOPES_INDEX);
                enc.set_texture(&k.displacement_mips[level], DST_DISPLACEMENT_INDEX);
                enc.set_texture(&k.slope_mips[level], DST_SLOPES_INDEX);
                dispatch(
                    &enc,
                    [groups, groups, cascades],
                    [OCEAN_MIP_GROUP, OCEAN_MIP_GROUP],
                );
            }
        }
        Ok(())
    }
}
