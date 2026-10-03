//! Vulkan's share of the reflection composite: which reflection path feeds it,
//! when it exists, and where its inputs come from this frame. The roughness blur
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

// Which reflection stages run. The composite exists whenever the RT pass or an
// authored SSR resolve can feed it, so a BVH coming or going never rebuilds it.
// RT takes the resolve slot while its BVH is live. Without one the SSR resolve
// covers when authored, and otherwise the RT node keeps the composite fed with
// an empty reflection, which leaves the scene as it was.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::vulkan) struct ReflectionPath {
    pub(in crate::vulkan) rt_trace: bool,
    pub(in crate::vulkan) rt_node: bool,
    pub(in crate::vulkan) ssr_resolve: bool,
    pub(in crate::vulkan) composite: bool,
}

impl ReflectionPath {
    pub(in crate::vulkan) fn new(ssr_authored: bool, rt_pass: bool, bvh_live: bool) -> Self {
        let rt_trace = rt_pass && bvh_live;
        Self {
            rt_trace,
            rt_node: rt_trace || (rt_pass && !ssr_authored),
            ssr_resolve: ssr_authored && !rt_trace,
            composite: rt_pass || ssr_authored,
        }
    }

    // Whether a resolve composites reflections over the scene this frame, which
    // is when the forward pass hands it the glossy dielectric specular.
    pub(in crate::vulkan) fn resolves(&self) -> bool {
        self.rt_trace || self.ssr_resolve
    }
}

impl VkContext {
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

#[cfg(test)]
mod tests {
    use super::ReflectionPath;

    #[test]
    fn a_live_trace_takes_the_resolve_slot_from_authored_ssr() {
        let path = ReflectionPath::new(true, true, true);
        assert!(path.rt_trace && path.rt_node);
        assert!(!path.ssr_resolve);
        assert!(path.composite && path.resolves());
    }

    #[test]
    fn authored_ssr_covers_while_the_rt_pass_has_no_bvh() {
        let path = ReflectionPath::new(true, true, false);
        assert!(!path.rt_trace && !path.rt_node);
        assert!(path.ssr_resolve);
        assert!(path.composite && path.resolves());
    }

    #[test]
    fn authored_ssr_resolves_without_rt() {
        for bvh_live in [false, true] {
            let path = ReflectionPath::new(true, false, bvh_live);
            assert!(!path.rt_trace && !path.rt_node);
            assert!(path.ssr_resolve);
            assert!(path.composite && path.resolves());
        }
    }

    #[test]
    fn rt_alone_traces_its_live_bvh() {
        let path = ReflectionPath::new(false, true, true);
        assert!(path.rt_trace && path.rt_node);
        assert!(!path.ssr_resolve);
        assert!(path.composite && path.resolves());
    }

    #[test]
    fn rt_alone_without_a_bvh_feeds_the_composite_nothing() {
        // The RT node still runs, so the composite stays fed, but it traces
        // nothing and the forward pass keeps its own specular.
        let path = ReflectionPath::new(false, true, false);
        assert!(!path.rt_trace && path.rt_node);
        assert!(!path.ssr_resolve);
        assert!(path.composite && !path.resolves());
    }

    #[test]
    fn no_resolve_leaves_no_composite() {
        // RT off or unsupported without authored SSR, or a SSGI-only world.
        for bvh_live in [false, true] {
            let path = ReflectionPath::new(false, false, bvh_live);
            assert!(!path.rt_trace && !path.rt_node && !path.ssr_resolve);
            assert!(!path.composite && !path.resolves());
        }
    }

    #[test]
    fn at_most_one_resolve_node_runs() {
        for bits in 0..8u8 {
            let path = ReflectionPath::new(bits & 1 != 0, bits & 2 != 0, bits & 4 != 0);
            assert!(!(path.rt_node && path.ssr_resolve), "{bits:03b}: {path:?}");
            assert_eq!(
                path.composite,
                path.rt_node || path.ssr_resolve,
                "{bits:03b}"
            );
        }
    }
}
