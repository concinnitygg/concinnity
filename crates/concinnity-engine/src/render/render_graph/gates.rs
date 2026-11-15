//! The settings gates and frame facts every backend's pass selection reads.
//!
//! A feature the settings switch at runtime (TAA, SSAO, SSR, ray-traced
//! reflections, SSGI, auto-exposure) runs when the resolved settings ask for it
//! and the backend built it when the settings last applied ([`FeatureGates`]).
//! A feature sized or chosen once at launch (shadows, the upscaler, two-pass
//! occlusion, the Hi-Z pyramid) runs when it was built. Everything else follows
//! what the world holds this frame ([`FrameFacts`]).

use crate::components::IndirectTechnique;
use crate::render::config::ResolvedGraphics;

/// What a backend's graphics API decides for it, the same on every frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackendTraits {
    /// The reactive mask has a target to write: the device can blend each
    /// render target independently.
    pub reactive_mask: bool,
    /// The post chain reads the reflection composite's output as its scene
    /// through bindings written ahead of the frame, so the composite must be
    /// written every frame, by an empty ray-traced pass while there is no BVH.
    pub composite_scene_fixed: bool,
}

/// The runtime-switched features a settings apply built, each `true` when its
/// resources exist after the apply. A feature asked for but not built (a
/// shader that failed to compile, a device that cannot run it) stays off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one flag per buildable render feature"
)]
pub struct BuildStatus {
    /// The TAA resolve.
    pub taa: bool,
    /// The SSAO kernel and blur.
    pub ssao: bool,
    /// The SSR resolve.
    pub ssr_resolve: bool,
    /// The ray-traced reflection pass.
    pub rt_pass: bool,
    /// The SSGI trace.
    pub ssgi: bool,
    /// The auto-exposure histogram.
    pub auto_exposure: bool,
    /// The unified G-buffer pre-pass targets.
    pub gbuffer: bool,
}

/// The runtime-switched features that run, from the resolved settings and what
/// the last apply built. Computed when settings apply, not per frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one gate per runtime-switched render feature"
)]
pub struct FeatureGates {
    /// TAA is the anti-aliasing mode.
    pub taa: bool,
    /// SSAO runs.
    pub ssao: bool,
    /// SSR is authored, so its resolve covers whenever ray tracing does not.
    pub ssr: bool,
    /// The ray-traced reflection pass runs.
    pub rt_pass: bool,
    /// SSGI runs.
    pub ssgi: bool,
    /// SSGI is the technique and built, so the main pass samples its resolved
    /// visibility and bounce. Outlasts `ssgi` while a live change has the trace
    /// off, which leaves the resolve carrying the occlusion with no bounce.
    pub ssgi_resolve: bool,
    /// Auto-exposure runs.
    pub auto_exposure: bool,
    /// The G-buffer pre-pass targets exist.
    pub gbuffer: bool,
    /// The backend's fixed traits.
    pub traits: BackendTraits,
}

impl FeatureGates {
    /// The gates `graphics` asks for, confirmed by `built`.
    pub fn new(graphics: &ResolvedGraphics, built: &BuildStatus, traits: BackendTraits) -> Self {
        Self {
            taa: graphics.taa() && built.taa,
            ssao: graphics.ssao().is_some() && built.ssao,
            ssr: graphics.ssr().is_some() && built.ssr_resolve,
            rt_pass: graphics.rt_reflections().is_some() && built.rt_pass,
            ssgi: graphics.ssgi().is_some() && built.ssgi,
            ssgi_resolve: graphics.indirect_lighting.technique == IndirectTechnique::Ssgi
                && built.ssgi,
            auto_exposure: graphics.auto_exposure().is_some() && built.auto_exposure,
            gbuffer: built.gbuffer,
            traits,
        }
    }
}

/// What the backend holds this frame: target sizes, what launch built, and
/// what the world's content puts on screen.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent per-frame render facts"
)]
pub struct FrameFacts {
    /// HDR target width in pixels.
    pub hdr_width: u32,
    /// HDR target height in pixels.
    pub hdr_height: u32,
    /// HDR target sample count.
    pub hdr_sample_count: u32,
    /// The cascade shadow pipeline and map were built.
    pub shadow: bool,
    /// Cascade shadow map edge in texels.
    pub shadow_map_size: u32,
    /// Far shadow map edge in texels; 0 when none was built.
    pub far_shadow_map_size: u32,
    /// The far shadow redraws this frame.
    pub far_shadow_drawn: bool,
    /// Shadowed spot lights this frame.
    pub shadowed_spot_count: u32,
    /// Spot shadow slice edge in texels.
    pub spot_shadow_slice_size: u32,
    /// An opaque menu backdrop hides the world.
    pub world_hidden: bool,
    /// The GPU-driven cull can run: its pipeline exists (and, where they are
    /// built per frame, so do its record buffers).
    pub cull_ready: bool,
    /// Records the cull processes this frame.
    pub cull_records: usize,
    /// Two-pass occlusion was built: the phase-two cull and the Hi-Z pyramid.
    pub two_pass_ready: bool,
    /// The Hi-Z pyramid exists.
    pub hiz: bool,
    /// The shadow plan measures this frame's receivers and the reduction was
    /// built.
    pub shadow_receivers: bool,
    /// The bloom chain exists.
    pub bloom_chain: bool,
    /// The live bloom intensity.
    pub bloom_intensity: f32,
    /// A temporal upscaler is running.
    pub upscaler: bool,
    /// The running upscaler reads the reactive mask.
    pub upscaler_reads_reactive: bool,
    /// The disocclusion pass was built.
    pub disocclusion: bool,
    /// A ray-tracing acceleration structure is live.
    pub bvh_live: bool,
    /// A particle emitter is live and the particle pipeline is built.
    pub particles: bool,
    /// A live particle emitter draws after the see-through surfaces.
    pub late_particles: bool,
    /// Fog is built and contributes.
    pub fog: bool,
    /// Decals exist and the decal pipeline is built.
    pub decals: bool,
    /// A water surface or glass panel is visible.
    pub visible_surfaces: bool,
    /// A see-through mesh is visible and its pipelines are built.
    pub visible_see_through: bool,
    /// Every transparent producer's ray-traced pipeline is built.
    pub rt_transparent_ready: bool,
    /// Planar mirror targets exist.
    pub planar_targets: bool,
    /// A visible water surface holds a mirror slot.
    pub water_planar_slot: bool,
    /// Lines were published and their pipeline exists.
    pub lines: bool,
    /// An SDF volume is visible.
    pub visible_volumes: bool,
    /// The atmosphere has this frame's inputs.
    pub atmosphere: bool,
    /// The atmosphere's clouds are built and have this frame's inputs.
    pub clouds: bool,
    /// A visible surface samples a simulated ocean.
    pub water_sim: bool,
    /// Grass is planted.
    pub grass: bool,
    /// Grass casts into the cascades this frame.
    pub grass_casts: bool,
    /// Local lights or probes are clustered.
    pub clustered: bool,
    /// A prop reflection probe rides this frame and its capture is built.
    pub prop_probe: bool,
}

#[cfg(test)]
mod tests;
