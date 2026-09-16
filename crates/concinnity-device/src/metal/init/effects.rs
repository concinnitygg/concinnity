//! Screen-space effect construction: the MetalFX upscaler, TAA, SSAO, SSR, the
//! unified G-buffer pre-pass, SSGI, and auto-exposure. Each is gated on its
//! world setting so a world that disables an effect pays zero construction cost,
//! and the runtime quality rebuild calls the same builders.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::gfx::auto_exposure;
use concinnity_core::gfx::auto_exposure::{AutoExposureSettings, AutoExposureState};
use concinnity_core::render::backend_init::PostSettings;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::post::device::PostExtent;
use concinnity_core::render::post::rt_reflections::RtReflectionSettings;
use concinnity_core::render::post::ssao::SsaoSettings;
use concinnity_core::render::post::ssgi::SsgiPass;
use concinnity_core::render::post::ssgi::settings::SsgiSettings;
use concinnity_core::render::post::ssr::SsrPass;
use concinnity_core::render::post::ssr::settings::SsrSettings;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::InitGpu;
use crate::metal::allocator::DeviceAllocator;
use crate::metal::auto_exposure::{AutoExposureGpu, build_auto_exposure_pipelines};
use crate::metal::context::{CompositeState, MtlSceneAssets};
use crate::metal::error::allocation_failed;
use crate::metal::post::post_device::MtlPostDevice;
use crate::metal::post::{
    GBufferState, MetalFXUpscaler, SsaoState, SsgiState, SsrState, TaaState, UpscaleState,
    build_gbuffer_bindless_pipeline, build_reflection_blur_pipeline,
    build_reflection_composite_pipeline, build_ssao_pipeline, build_taa_pass,
    create_gbuffer_targets, create_ssao_targets, create_ssr_targets, temporal_scaler_supported,
};
use crate::metal::slang_builtins::{SSAO_BLUR, SSAO_KERNEL};
use crate::metal::texture::create_fallback_texture;

// The toggle-controlled feature settings the screen-space builders gate on:
// the world's post settings at init, a quality push at runtime. Each is the
// world-resolved `Option` that gates whether that feature's pipelines + targets
// are built at all. `reflection_blur_scale` and `auto_exposure_bias_ev` ride
// along because they only make sense paired with their feature (SSR/RT and
// auto-exposure respectively).
pub(in crate::metal) struct EffectSettings<'a> {
    pub ssao: &'a Option<SsaoSettings>,
    pub ssr: &'a Option<SsrSettings>,
    pub ssgi: &'a Option<SsgiSettings>,
    pub rt_reflection: &'a Option<RtReflectionSettings>,
    pub auto_exposure: &'a Option<AutoExposureSettings>,
    // Per-axis divisor for the roughness-aware reflection blur target, resolved
    // from the world's `reflection_blur_resolution`. Sizes the blur target at
    // render / this; stored on `SsrState` so resize reuses it.
    pub reflection_blur_scale: u32,
    pub auto_exposure_bias_ev: f32,
}

impl<'a> EffectSettings<'a> {
    pub(super) fn from_post(post: &'a PostSettings) -> Self {
        Self {
            ssao: &post.ssao,
            ssr: &post.ssr,
            ssgi: &post.ssgi,
            rt_reflection: &post.rt_reflections,
            auto_exposure: &post.auto_exposure,
            reflection_blur_scale: post.reflection_blur_scale,
            auto_exposure_bias_ev: post.auto_exposure_bias_ev,
        }
    }

    // Whether SSR, SSGI or RT reflections run: all three need the reflection
    // targets and the G-buffer the unified pre-pass produces.
    fn reflections(&self) -> bool {
        self.ssr.is_some() || self.ssgi.is_some() || self.rt_reflection.is_some()
    }

    // Whether the unified G-buffer pre-pass runs at all. It gates two things
    // that must agree: the pool (which owns the pre-pass's three color
    // channels) and the pre-pass's own targets + pipelines. If they disagreed,
    // a consumer would read a label the pool never created. `needs_velocity` is
    // true when TAA is on or temporal upscaling is on (the MetalFX scaler
    // consumes motion vectors).
    pub(in crate::metal) fn gbuffer_needed(&self, needs_velocity: bool) -> bool {
        self.reflections() || self.ssao.is_some() || needs_velocity
    }
}

// MetalFX temporal upscaler. Built ahead of the scene targets because the
// resolved input size (clamped to the device's supported scale range)
// determines the render resolution every other 3D-scene target uses; bloom +
// composite stay at the drawable (output) resolution. Failure or unsupported
// hardware falls back silently to native-resolution rendering: the entire
// `temporal_upscaling` feature is asset-driven so a world that doesn't author
// it pays no construction cost either way. Metal always uses MetalFX, whatever
// upscaler backend the world names.
pub(super) fn build_upscale(
    gpu: &InitGpu<'_>,
    output: (u32, u32),
    post: &PostSettings,
) -> UpscaleState {
    let device = &*gpu.hw.device;
    let upscaler = if post.temporal_upscaling {
        if temporal_scaler_supported(device) {
            match MetalFXUpscaler::new(device, output.0, output.1, post.upscale_scale) {
                Ok(u) => {
                    tracing::info!(
                        "MetalFX: temporal upscaling on: render {}x{} → present {}x{} ({}x scale)",
                        u.input_width,
                        u.input_height,
                        u.output_width,
                        u.output_height,
                        (u.input_width as f32) / (u.output_width.max(1) as f32),
                    );
                    Some(u)
                }
                Err(e) => {
                    tracing::warn!(
                        "MetalFX: temporal scaler creation failed ({}); falling back to native resolution",
                        e
                    );
                    None
                }
            }
        } else {
            tracing::warn!(
                "MetalFX: temporal scaler not supported on this GPU; falling back to native resolution"
            );
            None
        }
    } else {
        None
    };
    // The stored scale is the *requested* one, not `input / output`: that ratio
    // is rounded to whole pixels, so feeding it back into the next rebuild
    // shrinks the input a little further every resize.
    let scale = if upscaler.is_some() {
        post.upscale_scale
    } else {
        1.0
    };
    UpscaleState {
        scaler: upscaler,
        scale,
        jitter: Default::default(),
        reset_pending: std::sync::atomic::AtomicBool::new(true),
    }
}

// The shared post-pass device every effect drawn through the seam is built on.
// Pipelines and targets only: nothing encodes before the context exists, so
// the device needs no probe set.
pub(super) fn post_device<'a>(
    gpu: &InitGpu<'a>,
    composite: &'a CompositeState,
    scene: &'a MtlSceneAssets,
) -> MtlPostDevice<'a> {
    MtlPostDevice {
        device: &gpu.hw.device,
        sampler: &composite.sampler,
        cube_sampler: &scene.cube_sampler,
        probes: None,
        timing: None,
        hot_reload: gpu.hot_reload,
    }
}

// The shared temporal resolve: pipeline plus ping-pong history buffers.
// Built only when TAA is on; upscaling-on worlds skip the TAA pass entirely
// (the MetalFX scaler does temporal accumulation itself). Its targets are
// sized at render-resolution to match the scene texture they sample.
pub(in crate::metal) fn build_taa(
    post_device: &MtlPostDevice,
    enabled: bool,
    render: (u32, u32),
) -> RenderResult<TaaState> {
    let pass = if enabled {
        Some(build_taa_pass(post_device, render.0, render.1)?)
    } else {
        None
    };
    Ok(TaaState {
        enabled,
        pass,
        frame: 0,
    })
}

// SSAO (GTAO): the horizon-search kernel, the depth-aware blur, and their
// occlusion targets, built only when SSAO is on. The depth + normal the kernel
// reads come from the unified G-buffer pre-pass, so SSAO builds no pre-pass of
// its own; the white fallback is always present.
pub(in crate::metal) fn build_ssao(
    alloc: &DeviceAllocator,
    settings: &EffectSettings,
    render: (u32, u32),
    hot_reload: bool,
) -> RenderResult<SsaoState> {
    let device = alloc.device();
    let (ssao_targets, ssao_kernel_pipeline, ssao_blur_pipeline) = if settings.ssao.is_some() {
        (
            Some(create_ssao_targets(device, render.0, render.1)?),
            Some(build_ssao_pipeline(device, &SSAO_KERNEL, hot_reload)?),
            Some(build_ssao_pipeline(device, &SSAO_BLUR, hot_reload)?),
        )
    } else {
        (None, None, None)
    };
    Ok(SsaoState {
        settings: *settings.ssao,
        targets: ssao_targets,
        kernel_pipeline: ssao_kernel_pipeline,
        blur_pipeline: ssao_blur_pipeline,
        white: create_fallback_texture(alloc)?,
    })
}

// SSR: the reflection targets, built when SSR *or* SSGI *or* RT reflections
// is on (all three need the G-buffer the unified pre-pass produces; RT
// reuses `ssr_targets.reflection`). The shared resolve is built only when
// SSR itself is on.
pub(in crate::metal) fn build_ssr(
    post_device: &MtlPostDevice,
    settings: &EffectSettings,
    render: (u32, u32),
) -> RenderResult<SsrState> {
    let (device, hot_reload) = (post_device.device, post_device.hot_reload);
    let (ssr_targets, ssr_resolve, ssr_composite_pipeline, ssr_blur_pipeline) =
        if settings.reflections() {
            let ssr_resolve = if settings.ssr.is_some() {
                Some(SsrPass::new(post_device)?)
            } else {
                None
            };
            // The reflection composite (roughness blur + blend over the scene)
            // runs for both SSR and RT reflections; both write the reflection
            // target it reads. SSGI alone needs the G-buffer but no composite.
            // The blur is its reduced-resolution first pass.
            let (composite, blur) = if settings.ssr.is_some() || settings.rt_reflection.is_some() {
                (
                    Some(build_reflection_composite_pipeline(device, hot_reload)?),
                    Some(build_reflection_blur_pipeline(device, hot_reload)?),
                )
            } else {
                (None, None)
            };
            (
                Some(create_ssr_targets(
                    device,
                    render.0,
                    render.1,
                    settings.reflection_blur_scale,
                )?),
                ssr_resolve,
                composite,
                blur,
            )
        } else {
            (None, None, None, None)
        };
    Ok(SsrState {
        settings: *settings.ssr,
        targets: ssr_targets,
        resolve: ssr_resolve,
        composite_pipeline: ssr_composite_pipeline,
        blur_pipeline: ssr_blur_pipeline,
        blur_scale: settings.reflection_blur_scale.max(1),
    })
}

// Unified G-buffer pre-pass (Metal): the shared targets and the one
// GPU-driven pipeline that fills them, built when any consumer (SSR / SSGI /
// RT / SSAO / velocity) is on. The pipeline is one engine-internal shader,
// independent of the world's fragment, so it builds the same in init and the
// runtime quality rebuild; the encode gates on the cull-produced object
// buffer, so a world with nothing in the cull records draws nothing here. The
// skinned variant is built later by `upload_skinned`.
pub(in crate::metal) fn build_gbuffer(
    device: &ProtocolObject<dyn MTLDevice>,
    enabled: bool,
    render: (u32, u32),
    hot_reload: bool,
) -> RenderResult<GBufferState> {
    let (targets, bindless_pipeline, history_pipeline) = if enabled {
        (
            Some(create_gbuffer_targets(device, render.0, render.1)?),
            Some(build_gbuffer_bindless_pipeline(device, hot_reload)?),
            Some(crate::metal::model_history::build_model_history_pipeline(
                device, hot_reload,
            )?),
        )
    } else {
        (None, None, None)
    };
    Ok(GBufferState {
        targets,
        bindless_pipeline,
        history_pipeline,
    })
}

// SSGI: the shared gather + composite. Built only when SSGI is on; the
// gather reads the G-buffer the pre-pass fills.
pub(in crate::metal) fn build_ssgi(
    post_device: &MtlPostDevice,
    settings: &EffectSettings,
    render: (u32, u32),
) -> RenderResult<SsgiState> {
    let pass = match settings.ssgi {
        Some(s) => Some(SsgiPass::new(
            post_device,
            s.gi_scale,
            PostExtent {
                width: render.0,
                height: render.1,
            },
        )?),
        None => None,
    };
    Ok(SsgiState {
        settings: *settings.ssgi,
        pass,
    })
}

// Auto-exposure pipelines + persistent compute buffers. Every buffer is
// zero-initialized so the build kernel's first dispatch sees an empty
// histogram and the readback ring's first reads see a finite average.
// `frames_in_flight` sizes the readback ring, one slot per frame the CPU may
// queue ahead of the GPU.
pub(in crate::metal) fn build_auto_exposure(
    device: &ProtocolObject<dyn MTLDevice>,
    settings: &EffectSettings,
    frames_in_flight: usize,
    hot_reload: bool,
) -> RenderResult<AutoExposureGpu> {
    let (pipelines, histogram, outputs, state, bias_ev) =
        if let Some(ae_settings) = settings.auto_exposure.as_ref() {
            let pipelines = build_auto_exposure_pipelines(device, hot_reload)?;
            let hist = make_auto_exposure_histogram(device)?;
            let outputs = (0..frames_in_flight.max(1))
                .map(|_| make_auto_exposure_output(device))
                .collect::<Result<Vec<_>, _>>()?;
            let state = AutoExposureState::new(ae_settings);
            (
                Some(pipelines),
                Some(hist),
                outputs,
                Some(state),
                settings.auto_exposure_bias_ev,
            )
        } else {
            (None, None, Vec::new(), None, 0.0)
        };
    Ok(AutoExposureGpu {
        settings: *settings.auto_exposure,
        state,
        bias_ev,
        pipelines,
        histogram,
        outputs,
        last_elapsed: 0.0,
    })
}

// Shared storage so the average kernel's writes are visible to the CPU
// readback at the top of the next frame without an explicit GPU<->CPU sync.
fn make_auto_exposure_histogram(
    device: &ProtocolObject<dyn MTLDevice>,
) -> RenderResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let hist_bytes = vec![0u8; std::mem::size_of::<u32>() * auto_exposure::HISTOGRAM_BINS];
    // SAFETY: the pointer and length describe the live `hist_bytes` allocation, and Metal copies
    // those bytes into the new buffer before the call returns.
    unsafe {
        let ptr = std::ptr::NonNull::new(hist_bytes.as_ptr() as *mut _).ok_or_else(|| {
            RenderError::Other("auto-exposure histogram bytes pointer is null".to_string())
        })?;
        device
            .newBufferWithBytes_length_options(
                ptr,
                hist_bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| allocation_failed("auto-exposure histogram buffer"))
    }
}

fn make_auto_exposure_output(
    device: &ProtocolObject<dyn MTLDevice>,
) -> RenderResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let out_bytes = vec![0u8; std::mem::size_of::<f32>()];
    // SAFETY: the pointer and length describe the live `out_bytes` allocation, and Metal copies
    // those bytes into the new buffer before the call returns.
    unsafe {
        let ptr = std::ptr::NonNull::new(out_bytes.as_ptr() as *mut _).ok_or_else(|| {
            RenderError::Other("auto-exposure output bytes pointer is null".to_string())
        })?;
        device
            .newBufferWithBytes_length_options(
                ptr,
                out_bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| allocation_failed("auto-exposure output buffer"))
    }
}
