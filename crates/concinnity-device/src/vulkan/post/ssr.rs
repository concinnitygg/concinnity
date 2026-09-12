// src/vulkan/post/ssr.rs
//
// Vulkan's share of screen-space reflections: the settings, the reflection
// target the resolve writes, and where the resolve's inputs come from this
// frame. The resolve itself -- its pipeline and its draw -- is written once in
// `concinnity_core::render::post::ssr` and reaches Vulkan through
// `VkPostDevice`, which also binds the forward global set it reads the
// reflection probes from.

use ash::vk;
use concinnity_core::gfx::ssr;
use concinnity_core::render::post::device::PostPassDevice;
use concinnity_core::render::post::ssr::{SsrInputs, SsrPass, target_desc};

use crate::vulkan::context::VkContext;
use crate::vulkan::post::post_device::{PostPipeline, PostTarget, VkPostDevice, post_extent};

// The reflection target's graph label, carried into debug naming.
const TARGET_LABEL: &str = "ssr_reflection";

// SSR resources held by `VkContext` whenever SSR, SSGI or RT reflections are on
// (all three share the G-buffer pre-pass). The resolve only runs when SSR itself
// is authored.
pub(in crate::vulkan) struct SsrResources {
    // Resolved authored tunables; turned into a per-frame `SsrParams` push.
    pub(in crate::vulkan) settings: ssr::SsrSettings,
    pass: SsrPass<PostPipeline>,
    // Reflected radiance + composite weight, which the reflection composite
    // blurs by roughness and blends over the scene.
    pub(in crate::vulkan) output: PostTarget,
}

impl SsrResources {
    // Build the resolve pipeline and the reflection target at `extent`.
    pub(in crate::vulkan) fn new(
        device: &VkPostDevice,
        settings: ssr::SsrSettings,
        extent: vk::Extent2D,
    ) -> Result<Self, String> {
        Ok(Self {
            settings,
            pass: SsrPass::new(device)?,
            output: device.create_target(TARGET_LABEL, &target_desc(), post_extent(extent))?,
        })
    }

    // Recreate the reflection target at a new render extent. The caller has
    // already idled the device and dropped the framebuffers naming the old view.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        device: &VkPostDevice,
        extent: vk::Extent2D,
    ) -> Result<(), String> {
        self.output = device.create_target(TARGET_LABEL, &target_desc(), post_extent(extent))?;
        Ok(())
    }

    // Swap in a freshly built pipeline. Driven by shader hot reload; the caller
    // has already idled the device.
    pub(in crate::vulkan) fn swap_pipeline(&mut self, pipeline: PostPipeline) {
        self.pass.swap_pipeline(pipeline);
    }
}

impl VkContext {
    // Encode the SSR resolve into `ssr.output`, then blur it by roughness and
    // composite it over the scene into the reflection composite's output, which
    // the post stack consumes. No-op when SSR is disabled.
    pub(in crate::vulkan) fn encode_ssr_resolve(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        fov_y_radians: f32,
        aspect: f32,
        cam_pos: [f32; 3],
    ) {
        let Some(ssr) = &self.ssr else { return };
        let Some(gbuffer) = &self.gbuffer else {
            tracing::error!("SSR resolve enabled but the G-buffer pre-pass is missing");
            return;
        };
        // The view-to-world rotation is the transpose of the view matrix's
        // orthonormal 3x3, embedded in a 4x4.
        let v = self.view.matrix;
        let inv_view_rot = [
            [v[0][0], v[1][0], v[2][0], 0.0],
            [v[0][1], v[1][1], v[2][1], 0.0],
            [v[0][2], v[1][2], v[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let params = ssr.settings.params(
            fov_y_radians,
            aspect,
            inv_view_rot,
            cam_pos,
            self.prefilter_mip_count as f32,
            self.view.sky_rot,
        );
        let device = self.post_device(frame_idx);
        let scene = &self.hdr_resolve_images[frame_idx % self.hdr_resolve_images.len()];
        if let Err(e) = ssr.pass.encode(
            &device,
            &cmd,
            SsrInputs {
                target: device.target_attachment(&ssr.output),
                scene: scene.view,
                normal_depth: gbuffer.normal_depth_view(frame_idx),
                roughness: gbuffer.roughness_view(frame_idx),
                prefilter: self.env_map.prefilter.view,
            },
            &params,
        ) {
            tracing::error!("SSR resolve: {e}");
            return;
        }
        self.encode_reflection_composite(cmd, ssr.output.view(), frame_idx);
    }
}
