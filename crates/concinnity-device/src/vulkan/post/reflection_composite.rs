//! Vulkan's share of the reflection composite: when it exists, and where its
//! inputs come from this frame. Which resolve feeds it is the shared
//! `ReflectionPath` table, which this backend reads with `rt_node`: its post
//! stack reads the composite's output as the scene every frame, so the RT node
//! keeps writing it while the BVH is missing. The roughness blur
//! and the composite -- their pipelines, their targets and both draws -- are
//! written once in `concinnity_core::render::post::reflection_composite` and
//! reach Vulkan through `VkPostDevice`.
//!
//! Both reflection paths feed one composite: `encode_ssr_resolve` /
//! `encode_rt_reflections` each render their resolve target, then call
//! `encode_reflection_composite` with that target's view.

use ash::vk;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::post::reflection_composite::{
    ReflectionCompositeInputs, ReflectionCompositePass,
};
use concinnity_core::render::post::reflection_path::ReflectionPath;

use super::super::context::VkContext;
use super::super::texture::GpuImage;
use crate::vulkan::post::post_device::{PostPipeline, PostTarget, VkPostDevice, post_extent};

// The shared composite, holding Vulkan's own pipeline and target types.
pub(in crate::vulkan) type VkReflectionCompositePass =
    ReflectionCompositePass<PostPipeline, PostTarget>;

// Build the composite at `extent` with its blur divided by `blur_scale`.
pub(in crate::vulkan) fn build_reflection_composite(
    device: &VkPostDevice,
    blur_scale: u32,
    extent: vk::Extent2D,
) -> RenderResult<VkReflectionCompositePass> {
    ReflectionCompositePass::new(device, blur_scale, post_extent(extent))
}

impl VkContext {
    // Which reflection stages run this frame, from the authored SSR and the live
    // RT pass and BVH.
    pub(in crate::vulkan) fn reflection_path(&self) -> ReflectionPath {
        ReflectionPath::new(
            self.ssr_authored(),
            self.rt_reflections.is_some(),
            self.rt.accel.is_some(),
        )
    }

    // Whether the SSR resolve runs: SSR is authored and RT did not take its slot.
    pub(in crate::vulkan) fn ssr_resolve_active(&self) -> bool {
        self.reflection_path().ssr_resolve
    }

    // The pre-TAA scene image for frame slot `frame`: the composite's output when
    // one exists, which is exactly when a resolve node writes it every frame.
    pub(in crate::vulkan) fn post_scene_image(&self, frame: usize) -> &GpuImage {
        match self.reflection_composite.as_ref() {
            Some(rc) => rc.output().image(),
            None => &self.targets.hdr_resolve_images[frame % self.targets.hdr_resolve_images.len()],
        }
    }

    // Drop the composite once no resolve feeds it, reporting whether it went.
    // The caller rebuilds the swapchain when it did, which re-points every scene
    // reader at the HDR resolve.
    fn release_unfed_reflection_composite(&mut self) -> bool {
        if self.reflection_path().composite || self.reflection_composite.is_none() {
            return false;
        }
        // The cached framebuffers name the targets about to drop.
        self.post.cache.forget_views();
        self.reflection_composite = None;
        true
    }

    // Bring the composite in line with the authored SSR and the live RT state,
    // building it at `blur_scale` when a resolve feeds it, or moving a live one
    // to `blur_scale`. The caller idles the device first and rebuilds the
    // swapchain after.
    pub(in crate::vulkan) fn reconcile_reflection_composite(
        &mut self,
        blur_scale: u32,
    ) -> RenderResult<()> {
        self.release_unfed_reflection_composite();
        if !self.reflection_path().composite {
            return Ok(());
        }
        if let Some(mut rc) = self.reflection_composite.take() {
            if rc.blur_scale_differs(blur_scale) {
                // The cached framebuffers name the blur about to drop.
                self.post.cache.forget_views();
            }
            let rescaled = rc.set_blur_scale(
                &self.post_device(0),
                blur_scale,
                post_extent(self.targets.render_extent),
            );
            self.reflection_composite = Some(rc);
            return rescaled.map(|_| ());
        }
        let rc = build_reflection_composite(
            &self.post_device(0),
            blur_scale,
            self.targets.render_extent,
        )?;
        self.reflection_composite = Some(rc);
        Ok(())
    }

    // Blur `reflection_view`, the resolve target the SSR / RT pass just wrote
    // (radiance + weight), by surface roughness and composite it over this
    // slot's HDR scene into the composite's output. Encoded inline at the tail
    // of `encode_ssr_resolve` / `encode_rt_reflections`. No-op when the
    // composite is absent (no reflection path active).
    pub(in crate::vulkan) fn encode_reflection_composite(
        &self,
        cmd: vk::CommandBuffer,
        reflection_view: vk::ImageView,
        frame_idx: usize,
    ) {
        let Some(rc) = &self.reflection_composite else {
            return;
        };
        let Some(gbuffer) = &self.gbuffer else {
            tracing::error!("reflection composite enabled but the G-buffer pre-pass is missing");
            return;
        };
        let scene =
            &self.targets.hdr_resolve_images[frame_idx % self.targets.hdr_resolve_images.len()];
        if let Err(e) = rc.encode(
            &self.post_device(frame_idx),
            &cmd,
            ReflectionCompositeInputs {
                reflection: reflection_view,
                scene: scene.view,
                normal_depth: gbuffer.normal_depth_view(frame_idx),
                roughness: gbuffer.roughness_view(frame_idx),
            },
        ) {
            tracing::error!("reflection composite: {e}");
        }
    }
}
