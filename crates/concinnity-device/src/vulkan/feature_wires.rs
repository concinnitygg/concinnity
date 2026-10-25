//! Per-frame-slot re-pointing of the descriptor sets and framebuffers a frame
//! records against but that are written ahead of it, wherever they name the
//! output of a feature a settings change can rebuild: the composite's scene
//! input and channels, the main camera's screen-space indirect light, the
//! post-chain scene the transparent and particle passes draw into, and the
//! glass reflection layers.
//!
//! A settings change never rewrites them itself: a frame still in flight may be
//! reading the set it would write. Each slot is checked right after its own
//! fence instead, when nothing in flight can name it, and rewritten only when
//! what it should name moved, so the steady state is a few comparisons and no
//! writes.

use ash::vk;
use concinnity_engine::render::backend::error::RenderResult;

use super::context::VkContext;
use super::descriptor_layout::SCREEN_INDIRECT_BINDING;
use super::post::bloom::composite_bloom_view;
use super::swapchain::{composite_channels, write_composite_channel_set, write_composite_set};
use super::wire_cache::WireCache;

// The images one composite set binds, in binding order.
#[derive(Clone, Copy, PartialEq)]
struct CompositeWire {
    scene: vk::ImageView,
    bloom: vk::ImageView,
    lut: vk::ImageView,
    channels: [vk::ImageView; 5],
}

// What each frame slot was last wired with.
pub(in crate::vulkan) struct FeatureWires {
    composite: WireCache<CompositeWire>,
    screen_indirect: WireCache<vk::ImageView>,
    post_scene: WireCache<vk::ImageView>,
}

impl FeatureWires {
    pub(in crate::vulkan) fn new(frames: usize) -> Self {
        Self {
            composite: WireCache::new(frames),
            screen_indirect: WireCache::new(frames),
            post_scene: WireCache::new(frames),
        }
    }
}

impl VkContext {
    fn composite_wire(&self, frame: usize) -> CompositeWire {
        let white = self.scene.ssao_white.view;
        let channels = composite_channels(
            &self.targets,
            self.gbuffer.as_ref(),
            self.ssao.is_some(),
            white,
            frame,
        );
        CompositeWire {
            scene: self.scene_color_view(frame),
            bloom: composite_bloom_view(&self.targets.transient_pool, white, frame),
            lut: self.scene.color_lut.view,
            channels: [
                channels.normal_depth,
                channels.roughness,
                channels.ao,
                channels.motion,
                channels.reactive,
            ],
        }
    }

    // Record every slot as wired with what it names now, for a caller that has
    // just written every set itself (init, a swapchain rebuild).
    pub(in crate::vulkan) fn mark_features_wired(&mut self) {
        for frame in 0..self.frames_in_flight {
            let composite = self.composite_wire(frame);
            let indirect = self.global_bindings().screen_indirect(frame);
            let scene = self.post_scene_image(frame).view;
            let wires = &mut self.feature_wires;
            wires.composite.changed(frame, composite);
            wires.screen_indirect.changed(frame, indirect);
            wires.post_scene.changed(frame, scene);
        }
    }

    // Re-point frame slot `frame`'s sets and framebuffers at what they should
    // name now. Called past the slot's fence, before the frame records.
    pub(in crate::vulkan) fn rewire_features(&mut self, frame: usize) -> RenderResult<()> {
        let composite = self.composite_wire(frame);
        if self.feature_wires.composite.changed(frame, composite)
            && let Some(&set) = self.composite.sets.get(frame)
        {
            let device = &self.hw.device;
            write_composite_set(device, set, composite.scene, composite.bloom, composite.lut);
            let [normal_depth, roughness, ao, motion, reactive] = composite.channels;
            write_composite_channel_set(
                device,
                set,
                &super::swapchain::CompositeChannels {
                    normal_depth,
                    roughness,
                    ao,
                    motion,
                    reactive,
                },
            );
        }

        let indirect = self.global_bindings().screen_indirect(frame);
        if self.feature_wires.screen_indirect.changed(frame, indirect)
            && let Some(&set) = self.descriptors.global_sets.get(frame)
        {
            self.global_bindings().frame(frame).write_binding(
                &self.hw.device,
                set,
                SCREEN_INDIRECT_BINDING,
            );
        }

        let scene = self.post_scene_image(frame);
        let scene = (scene.view, scene.image);
        let mask = self
            .targets
            .reactive_mask_images
            .get(frame)
            .map_or(self.scene.ssao_white.view, |m| m.view);
        let extent = self.targets.render_extent;
        if self.feature_wires.post_scene.changed(frame, scene.0) {
            if let Some(p) = self.particle.resources.as_mut() {
                p.rescene(&self.hw.device, frame, (scene.0, mask), extent)?;
            }
            if let Some(t) = self.transparent.as_mut() {
                t.rescene(&self.hw.device, frame, scene, mask, extent)?;
            }
        }

        if let (Some(t), Some(depth)) = (
            self.transparent.as_mut(),
            self.targets.depth_images.get(frame),
        ) {
            t.rewire_reflection_views(&self.hw.device, frame, depth.view);
        }
        Ok(())
    }
}
