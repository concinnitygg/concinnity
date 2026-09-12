// src/directx/post/ssr.rs
//
// DirectX's share of screen-space reflections: the settings, the reflection
// target the resolve writes, where the resolve's inputs come from this frame,
// and which scene texture the post stack consumes once a reflection path has
// run. The resolve itself -- its pipeline and its draw -- is written once in
// `concinnity_core::render::post::ssr` and reaches D3D12 through
// `DxPostDevice`, which also binds the probe cube table and the frame's
// ProbeSet a missed ray falls back to.

use concinnity_core::gfx::ssr;
use concinnity_core::render::post::device::{PostExtent, PostPassDevice};
use concinnity_core::render::post::ssr::{SsrInputs, SsrPass, target_desc};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::directx::context::DxContext;
use crate::directx::post::post_device::{DxPostDevice, PostPipeline, PostTarget};

// HDR-format reflection targets: the resolve's and the reflection composite's.
pub(crate) const SSR_OUTPUT_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;

// The reflection target's graph label, carried into debug naming.
const TARGET_LABEL: &str = "ssr_reflection";

// The SSR resolve, `Some` only when SSR itself is authored on.
pub(in crate::directx) struct SsrResolve {
    // Resolved authored tunables; turned into a per-frame `SsrParams` block.
    pub(in crate::directx) settings: ssr::SsrSettings,
    pass: SsrPass<PostPipeline>,
    // Reflected radiance + composite weight, which the reflection composite
    // blurs by roughness and blends over the scene.
    pub(in crate::directx) output: PostTarget,
}

// SSR resources held by `DxContext` when `PostProcessConfig.ssr` is on, or when
// SSGI or RT reflections are (all three reuse the unified G-buffer pre-pass).
// The resolve half is `Some` only when the SSR resolve itself is authored on.
pub(in crate::directx) struct SsrResources {
    pub(in crate::directx) resolve: Option<SsrResolve>,
}

impl SsrResources {
    // Build the resolve pipeline and its reflection target when
    // `resolve_settings` is `Some`; a SSGI-only or RT-only build holds neither.
    pub(in crate::directx) fn new(
        device: &DxPostDevice,
        width: u32,
        height: u32,
        resolve_settings: Option<ssr::SsrSettings>,
    ) -> Result<Self, String> {
        let resolve = match resolve_settings {
            Some(settings) => Some(SsrResolve {
                settings,
                pass: SsrPass::new(device)?,
                output: device.create_target(
                    TARGET_LABEL,
                    &target_desc(),
                    PostExtent { width, height },
                )?,
            }),
            None => None,
        };
        Ok(Self { resolve })
    }

    // Recreate the reflection target at a new render resolution. The reflection
    // composite reads its descriptor per frame, so the slot it lands on is free
    // to move.
    pub(in crate::directx) fn resize_to(
        &mut self,
        device: &DxPostDevice,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        if let Some(r) = self.resolve.as_mut() {
            r.output =
                device.create_target(TARGET_LABEL, &target_desc(), PostExtent { width, height })?;
        }
        Ok(())
    }

    // Swap in a freshly built resolve pipeline. Driven by shader hot reload; the
    // caller has already idled the device.
    pub(in crate::directx) fn swap_pipeline(&mut self, pipeline: PostPipeline) {
        if let Some(r) = self.resolve.as_mut() {
            r.pass.swap_pipeline(pipeline);
        }
    }
}

impl DxContext {
    // The scene texture the post stack consumes *before* the upscaler: the
    // reflection composite's output when a resolve ran, otherwise the HDR spine.
    // These two are the graph's `scene_pre_taa` and `hdr_resolve`, and this
    // predicate is the one the graph builder branches on.
    //
    // It exists because `scene_srv_for_post` below picks the same texture and
    // the two drifted apart once: when the SSR / RT resolves stopped
    // compositing inline and started writing radiance + weight into their own
    // target, the upscaler kept reading that target as if it were the scene, so
    // a world with both SSR and temporal upscaling upscaled the reflection
    // buffer. Anything that needs the resource rather than its SRV goes through
    // here.
    pub(in crate::directx) fn post_scene_target(&self) -> &ID3D12Resource {
        match self
            .reflection_composite
            .as_ref()
            .filter(|_| self.reflection_resolve_active())
        {
            Some(rc) => &rc.output,
            None => self.hdr_scene_target(),
        }
    }

    // Render-target view of `post_scene_target`, for the passes that blend into
    // it rather than sample it.
    pub(in crate::directx) fn post_scene_rtv(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        match self
            .reflection_composite
            .as_ref()
            .filter(|_| self.reflection_resolve_active())
        {
            Some(rc) => rc.output_rtv,
            None => self.hdr_scene_rtv(),
        }
    }

    // GPU descriptor handle of the SRV the TAA / bloom / composite passes
    // sample as the "scene": the upscaler's output when temporal upscaling is
    // on, the reflection composite's output when a reflection resolve ran,
    // otherwise the raw `hdr_resolve` SRV.
    pub(in crate::directx) fn scene_srv_for_post(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        if let Some(up) = &self.upscale.backend {
            return up.output_srv_gpu();
        }
        // Both SSR and RT feed the same composite (RT takes precedence at the
        // graph level, so at most one resolve runs). A SSGI-only world runs no
        // resolve, so it samples `hdr_resolve` directly (SSGI already composited
        // its bounce in).
        if let Some(rc) = self.reflection_composite.as_ref()
            && self.reflection_resolve_active()
        {
            return rc.output_srv_gpu;
        }
        self.hdr.srv_gpu
    }

    // GPU descriptor handle of the IBL prefilter cubemap SRV. Fixed at heap
    // slot 2; the SSR resolve and RT-reflection resolve both bind it as a miss
    // fallback. With no `EnvironmentMap` declared, the slot holds a 1x1 gray
    // fallback cube and `prefilter_mip_count == 0` tells the resolve to skip it.
    pub(in crate::directx) fn prefilter_cube_srv_gpu(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a property query on a live descriptor heap; it only reads.
        let srv_gpu_base = unsafe {
            self.descriptors
                .srv_heap
                .GetGPUDescriptorHandleForHeapStart()
        };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: srv_gpu_base.ptr + (2 * self.descriptors.srv_descriptor_size) as u64,
        }
    }

    // Encode the SSR resolve into `ssr.resolve.output`, then blur it by
    // roughness and composite it over the scene into the reflection composite's
    // output, which `scene_srv_for_post` hands the post stack. No-op when the
    // resolve half is absent (a SSGI-only build) or the G-buffer is missing.
    pub(in crate::directx) fn encode_ssr_resolve(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        fov_y_radians: f32,
        aspect: f32,
        cam_pos: [f32; 3],
    ) {
        let Some(resolve) = self.ssr.as_ref().and_then(|s| s.resolve.as_ref()) else {
            return;
        };
        let Some(gbuffer) = &self.gbuffer else { return };
        // The view-to-world rotation is the transpose of the view matrix's
        // orthonormal 3x3, embedded in a 4x4.
        let v = self.view.matrix;
        let inv_view_rot = [
            [v[0][0], v[1][0], v[2][0], 0.0],
            [v[0][1], v[1][1], v[2][1], 0.0],
            [v[0][2], v[1][2], v[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let params = resolve.settings.params(
            fov_y_radians,
            aspect,
            inv_view_rot,
            cam_pos,
            self.env_map.prefilter_mip_count as f32,
            self.view.sky_rot,
        );
        let device = self.post_device(frame_idx);
        if let Err(e) = resolve.pass.encode(
            &device,
            cmd,
            SsrInputs {
                target: device.target_attachment(&resolve.output),
                scene: self.hdr.srv_gpu,
                normal_depth: gbuffer.normal_depth_srv_gpu,
                roughness: gbuffer.roughness_srv_gpu,
                prefilter: self.prefilter_cube_srv_gpu(),
            },
            &params,
        ) {
            tracing::error!("SSR resolve: {e}");
            return;
        }
        self.encode_reflection_composite(cmd, resolve.output.srv_gpu());
    }
}
