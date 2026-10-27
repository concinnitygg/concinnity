//! The atmosphere (see `concinnity_engine::render::environment::atmosphere`): its tables, the
//! compute pass that builds them, the haze pass that blends the aerial volume
//! over the opaque scene, and the sky pipeline that draws the sky-view table
//! behind it. The irradiance cube the last kernel writes is the environment's
//! irradiance cube, and the sky cube the reflection prefilter convolves is
//! its prefiltered cube, so every pass that takes ambient light or
//! reflections from the environment takes the atmosphere's.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::environment::atmosphere::{
    AERIAL_SIZE, AirTables, AtmosphereFrame, AtmosphereParams, AtmosphereSettings, AtmosphereView,
    IRRADIANCE_FACE, MULTISCATTER_SIZE, SKY_CUBE_FACE, SKY_CUBE_PLAN, SKY_VIEW_HEIGHT,
    SKY_VIEW_WIDTH, SkyCubeKey, SkyCubeSchedule, SkyCubeWork, TRANSMITTANCE_HEIGHT,
    TRANSMITTANCE_WIDTH, TableReadiness,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, ns_string};
use objc2_metal::{
    MTLCommandBuffer, MTLComputeCommandEncoder as _, MTLComputePipelineState, MTLDevice,
    MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder as _,
    MTLRenderPassDescriptor, MTLRenderPipelineState, MTLSize, MTLStoreAction, MTLTexture,
    MTLTextureType, MTLTextureUsage,
};

use super::allocator::{DeviceAllocator, PooledTexture};
use super::builtin_shaders::{
    AERIAL_PERSPECTIVE_FRAG, ATMOSPHERE_AERIAL, ATMOSPHERE_IRRADIANCE, ATMOSPHERE_MULTISCATTER,
    ATMOSPHERE_SKY_CUBE, ATMOSPHERE_SKY_VIEW, ATMOSPHERE_TRANSMITTANCE, SKY_ATMOSPHERE_FRAG,
    compute_pipeline,
};
use super::context::MtlContext;
use super::descriptors::TextureDesc;
use super::encode::{ComputeEncode, RenderEncode};
use super::error::allocation_failed;
use super::post::fullscreen::{
    FullscreenBlend, build_fullscreen_pipeline, set_fragment_sampler_range,
};
use super::probe_prefilter::{
    PROBE_CUBE_FORMAT, PrefilterGpu, ProbePrefilterPipelines, create_capture_cube, dispatch_cube,
    encode_ggx_mip, encode_pyramid,
};
use super::scoped_encoder::ScopedEncoder;

// The format every table and the irradiance cube are stored in.
const TABLE_FORMAT: MTLPixelFormat = MTLPixelFormat::RGBA16Float;
// Absolute radiance, which a night sky takes below half float's range: the
// environment maps' irradiance cubes are full float too.
const IRRADIANCE_FORMAT: MTLPixelFormat = MTLPixelFormat::RGBA32Float;

type Texture = Retained<ProtocolObject<dyn MTLTexture>>;
type Kernel = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

// The atmosphere of a world that declares one.
pub(super) struct AtmosphereState {
    pub(super) settings: AtmosphereSettings,
    // This frame's constants, `None` until a frame carries the atmosphere's
    // inputs; the passes are left out of the graph until then.
    pub(super) params: Option<AtmosphereParams>,
    transmittance: Texture,
    multiscatter: Texture,
    pub(super) sky_view: Texture,
    aerial: Texture,
    // The environment's irradiance cube seen as six array slices, for the
    // kernel to write.
    irradiance_slices: Texture,
    kernels: Kernels,
    pub(super) sky_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    aerial_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    sky_cube: SkyCube,
    // The ground the tables that depend on the air alone were built over.
    air_tables: AirTables,
    // The frames that recorded the tables, for work recorded outside the frame.
    readiness: TableReadiness,
}

// The sky drawn into a capture cube and convolved by the reflection-probe
// kernels into the environment's prefiltered cube, a few steps a frame.
struct SkyCube {
    kernel: Kernel,
    prefilter: ProbePrefilterPipelines,
    gpu: PrefilterGpu,
    schedule: SkyCubeSchedule,
    // This frame's steps, planned when its constants are.
    work: SkyCubeWork,
}

/// The cubes an atmosphere writes, which the caller installs as the
/// environment's before anything binds them.
pub(super) struct AtmosphereCubes {
    pub(super) irradiance: PooledTexture,
    pub(super) prefilter: PooledTexture,
}

struct Kernels {
    transmittance: Kernel,
    multiscatter: Kernel,
    sky_view: Kernel,
    aerial: Kernel,
    irradiance: Kernel,
}

impl AtmosphereState {
    // The tables, the pipelines, and the cubes the atmosphere writes.
    pub(super) fn build(
        device: &ProtocolObject<dyn MTLDevice>,
        allocator: &DeviceAllocator,
        settings: &AtmosphereSettings,
        hdr_samples: u32,
        hot_reload: bool,
    ) -> RenderResult<(Self, AtmosphereCubes)> {
        let table = |kind, width: u32, height: u32, depth: u32, label: &str| {
            let desc = TextureDesc {
                kind,
                format: TABLE_FORMAT,
                width: width as usize,
                height: height as usize,
                depth: depth as usize,
                usage: MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
                ..Default::default()
            }
            .build();
            device
                .newTextureWithDescriptor(&desc)
                .ok_or_else(|| allocation_failed(format_args!("the atmosphere {label} table")))
        };
        let transmittance = table(
            MTLTextureType::Type2D,
            TRANSMITTANCE_WIDTH,
            TRANSMITTANCE_HEIGHT,
            1,
            "transmittance",
        )?;
        let multiscatter = table(
            MTLTextureType::Type2D,
            MULTISCATTER_SIZE,
            MULTISCATTER_SIZE,
            1,
            "multiple-scattering",
        )?;
        let sky_view = table(
            MTLTextureType::Type2D,
            SKY_VIEW_WIDTH,
            SKY_VIEW_HEIGHT,
            1,
            "sky-view",
        )?;
        let aerial = table(
            MTLTextureType::Type3D,
            AERIAL_SIZE,
            AERIAL_SIZE,
            AERIAL_SIZE,
            "aerial-perspective",
        )?;
        let irradiance = allocator.alloc_texture(
            &TextureDesc {
                kind: MTLTextureType::TypeCube,
                format: IRRADIANCE_FORMAT,
                width: IRRADIANCE_FACE as usize,
                height: IRRADIANCE_FACE as usize,
                usage: MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
                ..Default::default()
            }
            .build(),
        )?;
        // SAFETY: the view covers the cube's one level and its six faces, and
        // keeps the cube's own format, so it reinterprets nothing.
        let irradiance_slices = unsafe {
            irradiance.newTextureViewWithPixelFormat_textureType_levels_slices(
                IRRADIANCE_FORMAT,
                MTLTextureType::Type2DArray,
                NSRange::new(0, 1),
                NSRange::new(0, 6),
            )
        }
        .ok_or_else(|| RenderError::Other("failed to view the irradiance cube's faces".into()))?;
        let prefilter = allocator.alloc_texture(
            &TextureDesc {
                kind: MTLTextureType::TypeCube,
                format: PROBE_CUBE_FORMAT,
                width: SKY_CUBE_FACE as usize,
                height: SKY_CUBE_FACE as usize,
                mip_count: SKY_CUBE_PLAN.mips() as usize,
                usage: MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
                ..Default::default()
            }
            .build(),
        )?;
        let sky_cube = SkyCube {
            kernel: compute_pipeline(device, &ATMOSPHERE_SKY_CUBE, hot_reload)?,
            prefilter: ProbePrefilterPipelines::new(device, hot_reload)?,
            gpu: PrefilterGpu::new(
                create_capture_cube(device, &SKY_CUBE_PLAN)?,
                &prefilter,
                0,
                &SKY_CUBE_PLAN,
            )?,
            schedule: SkyCubeSchedule::new(SKY_CUBE_PLAN.mips()),
            work: SkyCubeWork::default(),
        };
        let kernels = Kernels {
            transmittance: compute_pipeline(device, &ATMOSPHERE_TRANSMITTANCE, hot_reload)?,
            multiscatter: compute_pipeline(device, &ATMOSPHERE_MULTISCATTER, hot_reload)?,
            sky_view: compute_pipeline(device, &ATMOSPHERE_SKY_VIEW, hot_reload)?,
            aerial: compute_pipeline(device, &ATMOSPHERE_AERIAL, hot_reload)?,
            irradiance: compute_pipeline(device, &ATMOSPHERE_IRRADIANCE, hot_reload)?,
        };
        let sky_pipeline =
            super::sky::build_sky_pipeline(device, &SKY_ATMOSPHERE_FRAG, hdr_samples, hot_reload)?;
        // `(in-scatter, 1 - T)` over the scene -> `scene * T + in-scatter`.
        let aerial_pipeline = build_fullscreen_pipeline(
            device,
            &AERIAL_PERSPECTIVE_FRAG,
            MTLPixelFormat::RGBA16Float,
            FullscreenBlend::PremultipliedOver,
            hot_reload,
        )?;
        let state = Self {
            settings: *settings,
            params: None,
            transmittance,
            multiscatter,
            sky_view,
            aerial,
            irradiance_slices,
            kernels,
            sky_pipeline,
            aerial_pipeline,
            sky_cube,
            air_tables: AirTables::default(),
            readiness: TableReadiness::default(),
        };
        Ok((
            state,
            AtmosphereCubes {
                irradiance,
                prefilter,
            },
        ))
    }
}

// One thread per texel over a `width` x `height` x `depth` grid, in 8 x 8
// groups.
fn dispatch_texels(
    enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    width: u32,
    height: u32,
) {
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: width as usize,
            height: height as usize,
            depth: 1,
        },
        MTLSize {
            width: 8,
            height: 8,
            depth: 1,
        },
    );
}

// `groups` groups of 64 threads.
fn dispatch_groups(
    enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    groups: [u32; 2],
) {
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: groups[0] as usize,
            height: groups[1] as usize,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
}

impl MtlContext {
    // Build this frame's tables: the sky view, the aerial volume and the
    // irradiance cube, after the transmittance and multiple-scattering tables
    // when the ground they are built over has changed; then this frame's steps
    // of the sky cube. One encoder; its resources are hazard-tracked, so each
    // dispatch waits for the writes the one before it made.
    pub(in crate::metal) fn encode_atmosphere(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> RenderResult<u32> {
        let Some(atmo) = self.atmosphere.as_ref() else {
            return Ok(0);
        };
        let Some(params) = atmo.params.as_ref() else {
            return Ok(0);
        };
        let desc = objc2_metal::MTLComputePassDescriptor::new();
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_compute(&desc, super::pass_timing::PassId::Atmosphere);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .computeCommandEncoderWithDescriptor(&desc)
                .ok_or_else(|| {
                    RenderError::Other("failed to get atmosphere compute encoder".into())
                })?,
            ns_string!("atmosphere"),
        );
        enc.set_value(params, 0);
        enc.set_sampler(&self.composite.sampler, 0);
        atmo.readiness.note_recorded();
        if atmo.air_tables.take_rebuild(params.radii[0]) {
            enc.set_pipeline(&atmo.kernels.transmittance);
            enc.set_texture(&atmo.transmittance, 3);
            dispatch_texels(&enc, TRANSMITTANCE_WIDTH, TRANSMITTANCE_HEIGHT);
            enc.set_pipeline(&atmo.kernels.multiscatter);
            enc.set_texture(&atmo.transmittance, 0);
            enc.set_texture(&atmo.multiscatter, 3);
            dispatch_groups(&enc, [MULTISCATTER_SIZE, MULTISCATTER_SIZE]);
        }
        enc.set_texture(&atmo.transmittance, 0);
        enc.set_texture(&atmo.multiscatter, 1);
        enc.set_pipeline(&atmo.kernels.sky_view);
        enc.set_texture(&atmo.sky_view, 3);
        dispatch_texels(&enc, SKY_VIEW_WIDTH, SKY_VIEW_HEIGHT);
        enc.set_pipeline(&atmo.kernels.aerial);
        enc.set_texture(&atmo.aerial, 3);
        dispatch_texels(&enc, AERIAL_SIZE, AERIAL_SIZE);
        enc.set_pipeline(&atmo.kernels.irradiance);
        enc.set_texture(&atmo.sky_view, 2);
        enc.set_texture(&atmo.irradiance_slices, 3);
        dispatch_groups(&enc, [1, 1]);
        let cube = &atmo.sky_cube;
        if cube.work.capture {
            enc.set_pipeline(&cube.kernel);
            enc.set_texture(cube.gpu.capture_top(), 3);
            dispatch_cube(&enc, SKY_CUBE_FACE);
            encode_pyramid(&enc, &cube.prefilter, &cube.gpu, &SKY_CUBE_PLAN);
        }
        for mip in cube.work.ggx_mips() {
            encode_ggx_mip(
                &enc,
                &cube.prefilter,
                &self.scene.cube_sampler,
                &cube.gpu,
                &SKY_CUBE_PLAN,
                mip,
            );
        }
        Ok(0)
    }

    // Blend the air in front of each opaque surface over the resolved scene.
    pub(in crate::metal) fn encode_aerial_perspective(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> RenderResult<u32> {
        let Some(atmo) = self.atmosphere.as_ref() else {
            return Ok(0);
        };
        let Some(params) = atmo.params.as_ref() else {
            return Ok(0);
        };
        let pass_desc = MTLRenderPassDescriptor::new();
        // SAFETY: plain descriptor property setters; the subscripted slot is one this descriptor
        // declares.
        unsafe {
            let ca = pass_desc.colorAttachments().objectAtIndexedSubscript(0);
            ca.setTexture(Some(self.targets.hdr.hdr_resolve.as_ref()));
            ca.setLoadAction(MTLLoadAction::Load);
            ca.setStoreAction(MTLStoreAction::Store);
        }
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_render(&pass_desc, super::pass_timing::PassId::AerialPerspective);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&pass_desc)
                .ok_or_else(|| {
                    RenderError::Other("failed to get aerial perspective encoder".into())
                })?,
            ns_string!("aerial perspective"),
        );
        enc.set_pipeline(&atmo.aerial_pipeline);
        enc.set_fragment_value(params, 0);
        enc.set_fragment_texture(self.targets.hdr.depth_resolve.as_ref(), 0);
        enc.set_fragment_texture(&atmo.aerial, 1);
        set_fragment_sampler_range(&enc, &self.composite.sampler, 0, 1);
        // SAFETY: the vertex shader generates all three vertices, so the draw reads no bound
        // vertex buffer.
        unsafe {
            enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
        Ok(1)
    }
}

impl MtlContext {
    // This frame's atmosphere constants, from the frame's inputs and the view
    // the scene is drawn from. A frame without inputs leaves the passes out.
    pub(super) fn prepare_atmosphere(
        &mut self,
        frame: Option<&AtmosphereFrame>,
        view: AtmosphereView,
    ) {
        if let Some(atmo) = self.atmosphere.as_mut() {
            atmo.params = frame.map(|f| AtmosphereParams::new(&atmo.settings, f, &view));
            atmo.sky_cube.work = match frame {
                Some(f) => {
                    atmo.sky_cube
                        .schedule
                        .plan(SkyCubeKey::new(&atmo.settings, f, &view.sky_rot))
                }
                None => SkyCubeWork::default(),
            };
        }
    }
}

impl MtlContext {
    // The view block's `irradiance` lane: whether the irradiance cube holds
    // real diffuse light this run.
    pub(super) fn irradiance_lane(&self) -> f32 {
        concinnity_engine::render::environment::sky::irradiance_lane(
            self.scene.env_map.prefilter_mip_count > 0,
            self.atmosphere.is_some(),
        )
    }

    // Whether work recorded outside the frame graph, such as a probe face,
    // reads tables this atmosphere has written. Always true without one.
    pub(super) fn atmosphere_tables_ready(&self) -> bool {
        self.atmosphere
            .as_ref()
            .is_none_or(|atmo| atmo.readiness.ready(self.frames_in_flight))
    }
}
