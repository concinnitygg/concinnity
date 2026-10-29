//! The water's wake on Metal (see
//! `concinnity_engine::render::environment::water::wake`): the state the
//! shared `water_wake.hlsl` kernels run forward each frame, the field a planet
//! sea's draw samples, and their dispatch inside the `WaterSim` node, after
//! the ocean's cascades. Every resource is device-created and hazard-tracked,
//! and the kernels run in the node's one serial compute encoder, so each
//! dispatch sees the last one's writes; the state outlives frames like the
//! ocean's foam.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::environment::water::wake::{
    WAKE_COLUMN_GROUPS, WAKE_FFT_THREADS, WAKE_GROUP, WAKE_MIP_LEVELS, WAKE_RESOLUTION,
    WAKE_ROW_GROUPS, wake_groups,
};
use concinnity_engine::render::uniforms::wake::WaterWakeParams;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLPixelFormat,
    MTLResourceOptions, MTLSize, MTLStorageMode, MTLTexture, MTLTextureType, MTLTextureUsage,
};

use super::builtin_shaders::{
    WATER_WAKE_COLUMNS, WATER_WAKE_INVERSE, WATER_WAKE_MIP, WATER_WAKE_ROWS, WATER_WAKE_SHADE,
    compute_pipeline,
};
use super::descriptors::TextureDesc;
use super::encode::ComputeEncode;
use super::error::allocation_failed;

type Texture = Retained<ProtocolObject<dyn MTLTexture>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Kernel = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

const FIELD_FORMAT: MTLPixelFormat = MTLPixelFormat::RGBA16Float;

// Slots pinned by `water_wake.hlsl`'s registers.
const PARAMS_INDEX: usize = 0;
const STATE_INDEX: usize = 1;
const WORK_INDEX: usize = 2;
const DST_FIELD_INDEX: usize = 3;
const SRC_FIELD_INDEX: usize = 4;

// The water draw's wake slot, pinned by `water.hlsl`'s register in both
// stages.
pub(in crate::metal) const WAKE_FIELD_TEXTURE_INDEX: usize = 11;

// The wake's kernels, state and field.
pub(in crate::metal) struct WakeSim {
    rows: Kernel,
    columns: Kernel,
    inverse: Kernel,
    shade: Kernel,
    mip: Kernel,
    // Two halves of (height, rise, foam, _) per cell, kept across frames.
    state: Buffer,
    // The transform between the passes, one complex value per cell.
    work: Buffer,
    // The field the draw samples, and one view per mip level.
    field: Texture,
    field_mips: Vec<Texture>,
}

impl WakeSim {
    pub(in crate::metal) fn build(
        device: &ProtocolObject<dyn MTLDevice>,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let cells = (WAKE_RESOLUTION * WAKE_RESOLUTION) as usize;
        let private = |bytes: usize, label: &str| {
            device
                .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModePrivate)
                .ok_or_else(|| allocation_failed(label))
        };
        let field = field_texture(device, WAKE_RESOLUTION, WAKE_MIP_LEVELS)?;
        let field_mips = (0..WAKE_MIP_LEVELS)
            .map(|m| mip_view(&field, m))
            .collect::<RenderResult<_>>()?;
        Ok(Self {
            rows: compute_pipeline(device, &WATER_WAKE_ROWS, hot_reload)?,
            columns: compute_pipeline(device, &WATER_WAKE_COLUMNS, hot_reload)?,
            inverse: compute_pipeline(device, &WATER_WAKE_INVERSE, hot_reload)?,
            shade: compute_pipeline(device, &WATER_WAKE_SHADE, hot_reload)?,
            mip: compute_pipeline(device, &WATER_WAKE_MIP, hot_reload)?,
            state: private(2 * cells * size_of::<[f32; 4]>(), "the wake state")?,
            work: private(cells * size_of::<[f32; 2]>(), "the wake transform")?,
            field,
            field_mips,
        })
    }

    // The field the draw samples.
    pub(in crate::metal) fn field(&self) -> &Texture {
        &self.field
    }

    // Run the wake one frame forward under `params` and reduce its field.
    // Returns whether it encoded the run.
    pub(in crate::metal) fn encode(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        params: &WaterWakeParams,
    ) -> bool {
        let Some(top) = self.field_mips.first() else {
            return false;
        };
        let fft = [WAKE_FFT_THREADS, 1];
        let square = [WAKE_GROUP, WAKE_GROUP];
        enc.set_value(params, PARAMS_INDEX);
        enc.set_buffer(&self.state, 0, STATE_INDEX);
        enc.set_buffer(&self.work, 0, WORK_INDEX);
        enc.set_pipeline(&self.rows);
        dispatch(enc, [WAKE_ROW_GROUPS, 1], fft);
        enc.set_pipeline(&self.columns);
        dispatch(enc, [WAKE_COLUMN_GROUPS, 1], fft);
        enc.set_pipeline(&self.inverse);
        dispatch(enc, [WAKE_ROW_GROUPS, 1], fft);
        enc.set_pipeline(&self.shade);
        enc.set_texture(top, DST_FIELD_INDEX);
        let g = wake_groups(0);
        dispatch(enc, [g, g], square);
        enc.set_pipeline(&self.mip);
        for (level, pair) in (1u32..).zip(self.field_mips.windows(2)) {
            let [src, dst] = pair else {
                continue;
            };
            enc.set_texture(src, SRC_FIELD_INDEX);
            enc.set_texture(dst, DST_FIELD_INDEX);
            let g = wake_groups(level);
            dispatch(enc, [g, g], square);
        }
        true
    }
}

// The field a sea binds where no wake runs: one still texel.
pub(in crate::metal) fn stand_in_field(
    device: &ProtocolObject<dyn MTLDevice>,
) -> RenderResult<Texture> {
    field_texture(device, 1, 1)
}

// A `size`-square field with `mips` levels.
fn field_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    size: u32,
    mips: u32,
) -> RenderResult<Texture> {
    let desc = TextureDesc {
        kind: MTLTextureType::Type2D,
        format: FIELD_FORMAT,
        width: size as usize,
        height: size as usize,
        mip_count: mips as usize,
        usage: MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
        storage: MTLStorageMode::Private,
        ..Default::default()
    }
    .build();
    device
        .newTextureWithDescriptor(&desc)
        .ok_or_else(|| allocation_failed("the wake field"))
}

// Mip `level` of `field`, for a kernel to read or write.
fn mip_view(field: &Texture, level: u32) -> RenderResult<Texture> {
    // SAFETY: the view covers one existing level of the field's one slice and
    // keeps its format, so it reinterprets nothing.
    unsafe {
        field.newTextureViewWithPixelFormat_textureType_levels_slices(
            FIELD_FORMAT,
            MTLTextureType::Type2D,
            NSRange::new(level as usize, 1),
            NSRange::new(0, 1),
        )
    }
    .ok_or_else(|| RenderError::Other("failed to view a wake field mip".into()))
}

fn dispatch(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    groups: [u32; 2],
    threads: [u32; 2],
) {
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: groups[0] as usize,
            height: groups[1] as usize,
            depth: 1,
        },
        MTLSize {
            width: threads[0] as usize,
            height: threads[1] as usize,
            depth: 1,
        },
    );
}
