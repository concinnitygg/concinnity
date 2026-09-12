// src/vulkan/post/mod.rs
//
// Screen-space and post-process passes for the Vulkan frame encoder. Each
// effect lives in its own file with its pipeline builder(s), target
// allocator(s), and per-frame encoder(s) co-located; mirrors the Metal
// `metal/post/` shape:
//
//   bloom.rs    prefilter + downsample/upsample mip chain
//   reflection_composite.rs  roughness blur + composite of the SSR/RT reflection
//   ssao.rs     GTAO depth+normal pre-pass + horizon-search kernel + blur
//   ssgi.rs     the settings + inputs of the shared SSGI gather and composite
//   ssr.rs      the reflection target + inputs of the shared SSR resolve
//   taa.rs      the TAA jitter counter + inputs over the shared resolve
//
// The three files below the effects are the shared fullscreen post-pass seam's
// Vulkan half: `post_device.rs` implements `gfx::post::PostPassDevice`,
// `pass_cache.rs` caches the render passes and framebuffers a draw needs, and
// `set_arena.rs` hands out descriptor sets per frame instead of per effect.
//   upscale/    temporal upscaling (FSR / DLSS / XeSS) behind VkUpscaleBackend

pub(in crate::vulkan) mod bloom;
pub(in crate::vulkan) mod fullscreen;
pub(in crate::vulkan) mod gbuffer;
pub(in crate::vulkan) mod reflection_composite;
pub(in crate::vulkan) mod rt_reflections;
pub(in crate::vulkan) mod ssao;
pub(in crate::vulkan) mod ssgi;
pub(in crate::vulkan) mod ssr;
pub(in crate::vulkan) mod taa;

pub(in crate::vulkan) mod pass_cache;
pub(in crate::vulkan) mod post_device;
pub(in crate::vulkan) mod set_arena;
pub(in crate::vulkan) mod upscale;

pub(in crate::vulkan) use gbuffer::GbufferResources;
pub(in crate::vulkan) use reflection_composite::ReflectionCompositeResources;
pub(in crate::vulkan) use rt_reflections::RtReflectionsResources;
pub(in crate::vulkan) use ssao::SsaoResources;
pub(in crate::vulkan) use ssgi::SsgiResources;
pub(in crate::vulkan) use ssr::SsrResources;
pub(in crate::vulkan) use taa::TaaResources;

/// The resources every shared post pass draws through, held once for the
/// backend rather than once per effect.
pub(in crate::vulkan) struct PostSupport {
    /// Render passes and framebuffers, keyed by attachment shape and view.
    pub(in crate::vulkan) cache: pass_cache::PostPassCache,
    /// Descriptor sets, one pool per frame in flight.
    pub(in crate::vulkan) arena: set_arena::PostSetArena,
}

impl PostSupport {
    /// Build the support for `frames` frames in flight.
    pub(in crate::vulkan) fn new(
        device: &crate::vulkan::owned::VkDevice,
        frames: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            cache: pass_cache::PostPassCache::new(),
            arena: set_arena::PostSetArena::new(device, frames)?,
        })
    }
}
pub(in crate::vulkan) use upscale::{
    ResolvedBackend, UpscaleSdk, VkUpscaleBackend, build_upscaler,
};
