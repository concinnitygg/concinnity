//! The atmosphere's clouds (see `concinnity_engine::render::environment::clouds`): the
//! noise volume the layers are carved from, built by the first frame's
//! atmosphere pass; the shadow map and the panorama that pass writes each frame
//! for the shaded passes and the sky's cubes; the half-resolution targets the
//! march and the reconstruction fill, one frame's and the previous frame's in
//! turn; and the composite that blends it over the scene behind the haze.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::environment::atmosphere::AtmosphereParams;
use concinnity_engine::render::environment::atmosphere::{
    AtmosphereFrame, AtmosphereSettings, AtmosphereView,
};
use concinnity_engine::render::environment::clouds::{
    CloudParams, CloudSchedule, NOISE_SIZE, PANORAMA_SIZE, SHADOW_SIZE,
};
use concinnity_engine::render::shader::programs::abi::clouds::metal::{composite, march, noise};
use concinnity_engine::render::shader::programs::abi::clouds::{
    MARCH_GROUP, NOISE_GROUP, PANORAMA_GROUP, RECONSTRUCT_GROUP, REFLECT_GROUP, SHADOW_GROUP,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLCommandBuffer, MTLComputeCommandEncoder as _, MTLComputePipelineState, MTLDevice,
    MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder as _,
    MTLRenderPassDescriptor, MTLRenderPipelineState, MTLSamplerAddressMode, MTLSamplerDescriptor,
    MTLSamplerMinMagFilter, MTLSamplerState, MTLStoreAction, MTLTexture, MTLTextureType,
    MTLTextureUsage,
};

use super::atmosphere::AtmosphereState;
use super::builtin_shaders::{
    CLOUD_COMPOSITE_FRAG, CLOUD_MARCH, CLOUD_NOISE, CLOUD_PANORAMA, CLOUD_RECONSTRUCT,
    CLOUD_REFLECT, CLOUD_SHADOW, compute_pipeline,
};
use super::context::MtlContext;
use super::descriptors::TextureDesc;
use super::dispatch::dispatch_threads;
use super::encode::{ComputeEncode, RenderEncode};
use super::error::allocation_failed;
use super::post::fullscreen::{FullscreenBlend, build_fullscreen_pipeline};
use super::scoped_encoder::ScopedEncoder;

type Texture = Retained<ProtocolObject<dyn MTLTexture>>;
type Kernel = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

// What the reflected sky reads of the atmosphere's and the clouds' state.
#[derive(Clone, Copy)]
pub(in crate::metal) struct ReflectedSky<'a> {
    atmo: &'a AtmosphereState,
    atmo_params: &'a AtmosphereParams,
    clouds: &'a CloudState,
    params: &'a CloudParams,
}

// The clouds of an atmosphere with cloud layers.
pub(super) struct CloudState {
    // This frame's constants, `None` until a frame carries the clouds' inputs.
    params: Option<CloudParams>,
    schedule: CloudSchedule,
    noise: Texture,
    pub(super) shadow: Texture,
    pub(super) panorama: Texture,
    // The march's texels, one of each 2x2 block of the target's.
    marched: Texture,
    // This frame's target and the previous frame's, as `schedule` swaps them.
    targets: [Texture; 2],
    noise_kernel: Kernel,
    shadow_kernel: Kernel,
    panorama_kernel: Kernel,
    march_kernel: Kernel,
    reconstruct_kernel: Kernel,
    reflect_kernel: Kernel,
    composite: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    noise_sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
}

impl CloudState {
    // The volume, the kernels and the composite, and a target to be sized by
    // the first frame.
    pub(super) fn build(
        device: &ProtocolObject<dyn MTLDevice>,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let noise = texture(
            device,
            MTLTextureType::Type3D,
            MTLPixelFormat::RGBA8Unorm,
            [NOISE_SIZE; 3],
            "the cloud noise volume",
        )?;
        // `(light, 1 - T)` over the scene -> `scene * T + light`.
        let composite = build_fullscreen_pipeline(
            device,
            &CLOUD_COMPOSITE_FRAG,
            MTLPixelFormat::RGBA16Float,
            FullscreenBlend::PremultipliedOver,
            hot_reload,
        )?;
        Ok(Self {
            params: None,
            schedule: CloudSchedule::new([1, 1]),
            noise,
            shadow: texture(
                device,
                MTLTextureType::Type2D,
                MTLPixelFormat::RGBA16Float,
                [SHADOW_SIZE, SHADOW_SIZE, 1],
                "the cloud shadow map",
            )?,
            panorama: texture(
                device,
                MTLTextureType::Type2D,
                MTLPixelFormat::RGBA16Float,
                [PANORAMA_SIZE[0], PANORAMA_SIZE[1], 1],
                "the cloud panorama",
            )?,
            marched: target(device, [1, 1])?,
            targets: [target(device, [1, 1])?, target(device, [1, 1])?],
            noise_kernel: compute_pipeline(device, &CLOUD_NOISE, hot_reload)?,
            shadow_kernel: compute_pipeline(device, &CLOUD_SHADOW, hot_reload)?,
            panorama_kernel: compute_pipeline(device, &CLOUD_PANORAMA, hot_reload)?,
            march_kernel: compute_pipeline(device, &CLOUD_MARCH, hot_reload)?,
            reconstruct_kernel: compute_pipeline(device, &CLOUD_RECONSTRUCT, hot_reload)?,
            reflect_kernel: compute_pipeline(device, &CLOUD_REFLECT, hot_reload)?,
            composite,
            noise_sampler: repeat_sampler(device)?,
        })
    }

    // This frame's constants for `settings` and `frame` seen from `view`,
    // the targets made again when the scene's size changes, and this frame's
    // target the other one. A command buffer still in flight keeps the old
    // targets alive.
    pub(super) fn prepare(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        settings: &AtmosphereSettings,
        frame: Option<&AtmosphereFrame>,
        view: &AtmosphereView,
    ) -> RenderResult<()> {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a pixel count"
        )]
        let scene = view.viewport.map(|v| v.max(1.0) as u32);
        if let Some(size) = self.schedule.resize(scene) {
            self.marched = target(device, self.schedule.march_size())?;
            self.targets = [target(device, size)?, target(device, size)?];
        }
        self.params = self.schedule.advance(settings, frame, view);
        Ok(())
    }

    // The target this frame's composite reads.
    fn target(&self) -> &Texture {
        &self.targets[self.schedule.write()]
    }

    // Whether this frame has the clouds' constants.
    pub(super) fn live(&self) -> bool {
        self.params.is_some()
    }

    // Write this frame's shadow map and panorama inside the atmosphere's
    // encoder, after building the noise volume on the first frame. The
    // atmosphere's tables are bound at the indices the cloud kernels read them
    // at by `bind_tables`.
    pub(super) fn encode_maps(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        atmo_params: &AtmosphereParams,
        bind_tables: impl Fn(&ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>),
    ) {
        let Some(params) = self.params.as_ref() else {
            return;
        };
        let threads = self.schedule.dispatch();
        if self.schedule.take_noise_build() {
            enc.set_pipeline(&self.noise_kernel);
            enc.set_texture(&self.noise, noise::NOISE_OUT.index);
            dispatch_threads(enc, threads.noise, NOISE_GROUP);
        }
        enc.set_value(atmo_params, march::ATMOSPHERE.index);
        enc.set_value(params, march::CLOUDS.index);
        bind_tables(enc);
        enc.set_texture(&self.noise, march::NOISE.index);
        enc.set_sampler(&self.noise_sampler, march::NOISE_SAMPLER.index);
        enc.set_pipeline(&self.shadow_kernel);
        enc.set_texture(&self.shadow, march::OUTPUT.index);
        dispatch_threads(enc, threads.shadow, SHADOW_GROUP);
        enc.set_pipeline(&self.panorama_kernel);
        enc.set_texture(&self.panorama, march::OUTPUT.index);
        dispatch_threads(enc, threads.panorama, PANORAMA_GROUP);
    }
}

fn texture(
    device: &ProtocolObject<dyn MTLDevice>,
    kind: MTLTextureType,
    format: MTLPixelFormat,
    size: [u32; 3],
    label: &str,
) -> RenderResult<Texture> {
    let desc = TextureDesc {
        kind,
        format,
        width: size[0] as usize,
        height: size[1] as usize,
        depth: size[2] as usize,
        usage: MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
        ..Default::default()
    }
    .build();
    device
        .newTextureWithDescriptor(&desc)
        .ok_or_else(|| allocation_failed(label))
}

fn target(device: &ProtocolObject<dyn MTLDevice>, size: [u32; 2]) -> RenderResult<Texture> {
    texture(
        device,
        MTLTextureType::Type2D,
        MTLPixelFormat::RGBA16Float,
        [size[0], size[1], 1],
        "the cloud target",
    )
}

// Trilinear and wrapping: the noise volume tiles the sky.
fn repeat_sampler(
    device: &ProtocolObject<dyn MTLDevice>,
) -> RenderResult<Retained<ProtocolObject<dyn MTLSamplerState>>> {
    let desc = MTLSamplerDescriptor::new();
    desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
    desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
    desc.setSAddressMode(MTLSamplerAddressMode::Repeat);
    desc.setTAddressMode(MTLSamplerAddressMode::Repeat);
    desc.setRAddressMode(MTLSamplerAddressMode::Repeat);
    device
        .newSamplerStateWithDescriptor(&desc)
        .ok_or_else(|| RenderError::Other("failed to create the cloud noise sampler".into()))
}

impl MtlContext {
    // March one texel of each 2x2 block of the clouds' target, then fill this
    // frame's target from the march and the previous frame's.
    pub(in crate::metal) fn encode_clouds(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> RenderResult<u32> {
        let Some(atmo) = self.atmosphere.as_ref() else {
            return Ok(0);
        };
        let (Some(atmo_params), Some(clouds)) = (atmo.params.as_ref(), atmo.clouds.as_ref()) else {
            return Ok(0);
        };
        let Some(params) = clouds.params.as_ref() else {
            return Ok(0);
        };
        let desc = objc2_metal::MTLComputePassDescriptor::new();
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_compute(&desc, super::pass_timing::PassId::Clouds);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .computeCommandEncoderWithDescriptor(&desc)
                .ok_or_else(|| RenderError::Other("failed to get cloud compute encoder".into()))?,
            ns_string!("clouds"),
        );
        enc.set_pipeline(&clouds.march_kernel);
        enc.set_value(atmo_params, march::ATMOSPHERE.index);
        enc.set_value(params, march::CLOUDS.index);
        enc.set_texture(
            self.targets.hdr.depth_resolve.as_ref(),
            march::SCENE_DEPTH.index,
        );
        enc.set_texture(atmo.aerial(), march::AERIAL_VOLUME.index);
        enc.set_texture(atmo.transmittance(), march::TRANSMITTANCE.index);
        enc.set_texture(atmo.multiscatter(), march::MULTISCATTER.index);
        enc.set_texture(&clouds.noise, march::NOISE.index);
        enc.set_texture(&clouds.marched, march::OUTPUT.index);
        enc.set_sampler(&self.composite.sampler, march::LUT_SAMPLER.index);
        enc.set_sampler(&clouds.noise_sampler, march::NOISE_SAMPLER.index);
        let threads = clouds.schedule.dispatch();
        dispatch_threads(&enc, threads.march, MARCH_GROUP);
        enc.set_pipeline(&clouds.reconstruct_kernel);
        enc.set_texture(clouds.target(), march::OUTPUT.index);
        enc.set_texture(&clouds.marched, march::MARCHED.index);
        let previous = &clouds.targets[clouds.schedule.previous()];
        enc.set_texture(previous, march::PREVIOUS.index);
        dispatch_threads(&enc, threads.reconstruct, RECONSTRUCT_GROUP);
        Ok(0)
    }

    // What the reflected sky reads this frame, when the clouds have this
    // frame's constants: without them the trace lists no ray.
    pub(in crate::metal) fn reflected_sky(&self) -> Option<ReflectedSky<'_>> {
        let atmo = self.atmosphere.as_ref()?;
        let clouds = atmo.clouds.as_ref()?;
        Some(ReflectedSky {
            atmo,
            atmo_params: atmo.params.as_ref()?,
            clouds,
            params: clouds.params.as_ref()?,
        })
    }

    // Shade the reflection rays the trace listed as leaving the scene: the sky
    // along each, its clouds marched, added into `reflection` by the ray's
    // weight. One thread per listed ray, dispatched by the list's own
    // arguments.
    pub(in crate::metal) fn encode_reflected_sky(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        sky: &ReflectedSky<'_>,
        reflection: &ProtocolObject<dyn MTLTexture>,
        misses: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
        miss_args: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
    ) {
        let ReflectedSky {
            atmo,
            atmo_params,
            clouds,
            params,
        } = *sky;
        enc.set_pipeline(&clouds.reflect_kernel);
        enc.set_value(atmo_params, march::ATMOSPHERE.index);
        enc.set_value(params, march::CLOUDS.index);
        enc.set_texture(atmo.transmittance(), march::TRANSMITTANCE.index);
        enc.set_texture(atmo.multiscatter(), march::MULTISCATTER.index);
        enc.set_texture(&atmo.sky_view, march::SKY_VIEW.index);
        enc.set_texture(&clouds.noise, march::NOISE.index);
        enc.set_texture(reflection, march::OUTPUT.index);
        enc.set_sampler(&self.composite.sampler, march::LUT_SAMPLER.index);
        enc.set_sampler(&clouds.noise_sampler, march::NOISE_SAMPLER.index);
        enc.set_buffer(misses, 0, march::MISSES.index);
        enc.set_buffer(miss_args, 0, march::MISS_ARGS.index);
        enc.set_texture(&clouds.panorama, march::PANORAMA.index);
        // SAFETY: the argument buffer holds the three group counts the trace
        // grew to cover its list, at offset 0, written before this dispatch.
        unsafe {
            enc.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                miss_args,
                0,
                objc2_metal::MTLSize {
                    width: REFLECT_GROUP.size[0] as usize,
                    height: 1,
                    depth: 1,
                },
            );
        }
    }

    // Blend the cloud target over the pre-TAA scene `scene`.
    pub(in crate::metal) fn encode_cloud_composite(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        scene: Option<&Texture>,
    ) -> RenderResult<u32> {
        let scene = super::graph_exec::pass_input(
            scene,
            super::pass_timing::PassId::CloudComposite,
            "scene_pre_taa",
        )?;
        let Some(clouds) = self.atmosphere.as_ref().and_then(|a| a.clouds.as_ref()) else {
            return Ok(0);
        };
        let Some(params) = clouds.params.as_ref() else {
            return Ok(0);
        };
        let pass_desc = MTLRenderPassDescriptor::new();
        // SAFETY: plain descriptor property setters; the subscripted slot is one this descriptor
        // declares.
        unsafe {
            let ca = pass_desc.colorAttachments().objectAtIndexedSubscript(0);
            ca.setTexture(Some(scene));
            ca.setLoadAction(MTLLoadAction::Load);
            ca.setStoreAction(MTLStoreAction::Store);
        }
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_render(&pass_desc, super::pass_timing::PassId::CloudComposite);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&pass_desc)
                .ok_or_else(|| {
                    RenderError::Other("failed to get cloud composite encoder".into())
                })?,
            ns_string!("cloud composite"),
        );
        enc.set_pipeline(&clouds.composite);
        enc.set_fragment_value(params, composite::CLOUDS.index);
        enc.set_fragment_texture(
            self.targets.hdr.depth_resolve.as_ref(),
            composite::SCENE_DEPTH.index,
        );
        enc.set_fragment_texture(clouds.target(), composite::TARGET.index);
        // SAFETY: the vertex shader generates all three vertices, so the draw reads no bound
        // vertex buffer.
        unsafe {
            enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
        Ok(1)
    }
}

impl MtlContext {
    // The clouds' shadow map the main pass reads. Without clouds the shadow
    // lanes are off and the slot is never read, so any 2D texture fills it.
    pub(in crate::metal) fn cloud_shadow_texture(&self) -> &ProtocolObject<dyn MTLTexture> {
        match self.atmosphere.as_ref().and_then(|a| a.clouds.as_ref()) {
            Some(clouds) => &clouds.shadow,
            None => &self.scene.ltc_matrix_texture,
        }
    }
}
