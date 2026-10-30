//! The atmosphere's clouds (see `concinnity_engine::render::environment::clouds`): the
//! noise volume the layers are carved from, built by the first frame's
//! atmosphere pass; the shadow map and the panorama that pass writes each frame
//! for the shaded passes and the sky's cubes; the half-resolution target the
//! march writes; and the composite that blends it over the scene behind the
//! haze.
#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::Cell;

use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::environment::atmosphere::AtmosphereParams;
use concinnity_engine::render::environment::atmosphere::{AtmosphereFrame, AtmosphereSettings};
use concinnity_engine::render::environment::clouds::{
    CloudParams, NOISE_SIZE, PANORAMA_SIZE, SHADOW_SIZE, cloud_target_size,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLCommandBuffer, MTLComputeCommandEncoder as _, MTLComputePipelineState, MTLDevice,
    MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder as _,
    MTLRenderPassDescriptor, MTLRenderPipelineState, MTLSamplerAddressMode, MTLSamplerDescriptor,
    MTLSamplerMinMagFilter, MTLSamplerState, MTLSize, MTLStoreAction, MTLTexture, MTLTextureType,
    MTLTextureUsage,
};

use super::builtin_shaders::{
    CLOUD_COMPOSITE_FRAG, CLOUD_MARCH, CLOUD_NOISE, CLOUD_PANORAMA, CLOUD_SHADOW, compute_pipeline,
};
use super::context::MtlContext;
use super::descriptors::TextureDesc;
use super::encode::{ComputeEncode, RenderEncode};
use super::error::allocation_failed;
use super::post::fullscreen::{FullscreenBlend, build_fullscreen_pipeline};
use super::scoped_encoder::ScopedEncoder;

type Texture = Retained<ProtocolObject<dyn MTLTexture>>;
type Kernel = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

// The clouds of an atmosphere with cloud layers.
pub(super) struct CloudState {
    // This frame's constants, `None` until a frame carries the clouds' inputs.
    params: Option<CloudParams>,
    noise: Texture,
    // Whether an atmosphere pass has built the noise volume yet.
    noise_built: Cell<bool>,
    pub(super) shadow: Texture,
    pub(super) panorama: Texture,
    target: Texture,
    target_size: [u32; 2],
    noise_kernel: Kernel,
    shadow_kernel: Kernel,
    panorama_kernel: Kernel,
    march_kernel: Kernel,
    composite: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    noise_sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    frames: u64,
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
            noise,
            noise_built: Cell::new(false),
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
            target: target(device, [1, 1])?,
            target_size: [1, 1],
            noise_kernel: compute_pipeline(device, &CLOUD_NOISE, hot_reload)?,
            shadow_kernel: compute_pipeline(device, &CLOUD_SHADOW, hot_reload)?,
            panorama_kernel: compute_pipeline(device, &CLOUD_PANORAMA, hot_reload)?,
            march_kernel: compute_pipeline(device, &CLOUD_MARCH, hot_reload)?,
            composite,
            noise_sampler: repeat_sampler(device)?,
            frames: 0,
        })
    }

    // This frame's constants for `settings` and `frame` over a scene
    // `viewport` pixels across, the target made again when that size changes.
    // A command buffer still in flight keeps the old target alive.
    pub(super) fn prepare(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        settings: &AtmosphereSettings,
        frame: Option<&AtmosphereFrame>,
        viewport: [f32; 2],
    ) -> RenderResult<()> {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a pixel count"
        )]
        let scene = viewport.map(|v| v.max(1.0) as u32);
        let size = cloud_target_size(scene[0], scene[1]);
        if size != self.target_size {
            self.target = target(device, size)?;
            self.target_size = size;
        }
        self.params = match (
            settings.clouds.as_ref(),
            frame.and_then(|f| f.clouds.as_ref()),
        ) {
            (Some(clouds), Some(cloud_frame)) => Some(CloudParams::new(
                clouds,
                settings.bottom_radius,
                cloud_frame,
                scene,
                self.frames,
            )),
            _ => None,
        };
        self.frames = self.frames.wrapping_add(1);
        Ok(())
    }

    // Whether this frame has the clouds' constants.
    pub(super) fn live(&self) -> bool {
        self.params.is_some()
    }

    // Write this frame's shadow map, and the panorama when `panorama` (the
    // sky cube redraws), inside the atmosphere's encoder, after building the
    // noise volume on the first frame. The atmosphere's tables are bound at
    // the indices the cloud kernels read them at by `bind_tables`.
    pub(super) fn encode_maps(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        atmo_params: &AtmosphereParams,
        panorama: bool,
        bind_tables: impl Fn(&ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>),
    ) {
        let Some(params) = self.params.as_ref() else {
            return;
        };
        if !self.noise_built.get() {
            enc.set_pipeline(&self.noise_kernel);
            enc.set_texture(&self.noise, 0);
            dispatch(enc, [NOISE_SIZE; 3], [4, 4, 4]);
            self.noise_built.set(true);
        }
        enc.set_value(atmo_params, 0);
        enc.set_value(params, 1);
        bind_tables(enc);
        enc.set_texture(&self.noise, 4);
        enc.set_sampler(&self.noise_sampler, 1);
        enc.set_pipeline(&self.shadow_kernel);
        enc.set_texture(&self.shadow, 5);
        dispatch(enc, [SHADOW_SIZE, SHADOW_SIZE, 1], [8, 8, 1]);
        if panorama {
            enc.set_pipeline(&self.panorama_kernel);
            enc.set_texture(&self.panorama, 5);
            dispatch(enc, [PANORAMA_SIZE[0], PANORAMA_SIZE[1], 1], [8, 8, 1]);
        }
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

// One thread per texel over a `size` grid in `group` groups.
fn dispatch(
    enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    size: [u32; 3],
    group: [u32; 3],
) {
    let s = |v: [u32; 3]| MTLSize {
        width: v[0] as usize,
        height: v[1] as usize,
        depth: v[2] as usize,
    };
    enc.dispatchThreads_threadsPerThreadgroup(s(size), s(group));
}

impl MtlContext {
    // March the clouds into their target.
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
        enc.set_value(atmo_params, 0);
        enc.set_value(params, 1);
        enc.set_texture(self.targets.hdr.depth_resolve.as_ref(), 0);
        enc.set_texture(atmo.aerial(), 1);
        enc.set_texture(atmo.transmittance(), 2);
        enc.set_texture(atmo.multiscatter(), 3);
        enc.set_texture(&clouds.noise, 4);
        enc.set_texture(&clouds.target, 5);
        enc.set_sampler(&self.composite.sampler, 0);
        enc.set_sampler(&clouds.noise_sampler, 1);
        let [w, h] = clouds.target_size;
        dispatch(&enc, [w, h, 1], [8, 8, 1]);
        Ok(0)
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
        enc.set_fragment_value(params, 0);
        enc.set_fragment_texture(self.targets.hdr.depth_resolve.as_ref(), 0);
        enc.set_fragment_texture(&clouds.target, 1);
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
