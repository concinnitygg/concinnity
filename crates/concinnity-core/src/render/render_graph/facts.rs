//! What a backend reports to gate its frame graph, split by what changes it:
//! [`ResidentFeatures`] follows what the backend built, [`FrameContent`] what
//! the world holds this frame. [`FrameFacts::derive`] combines them, once, for
//! every backend.

use super::gates::FrameFacts;
use crate::gfx::render_types::spot_shadow_slice_size;

/// What the backend has built: targets and the pipelines a pass needs to run.
/// Changes only at init, on a settings apply, or on a resize.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidentFeatures {
    /// HDR target width in pixels.
    pub hdr_width: u32,
    /// HDR target height in pixels.
    pub hdr_height: u32,
    /// HDR target sample count.
    pub hdr_sample_count: u32,
    /// The cascade shadow map and its pipeline.
    pub shadow: bool,
    /// Cascade shadow map edge in texels; spot slices are sized from it.
    pub shadow_map_size: u32,
    /// The GPU-driven cull pipeline.
    pub cull: bool,
    /// Two-pass occlusion: the phase-two cull and the Hi-Z pyramid.
    pub two_pass: bool,
    /// The Hi-Z pyramid.
    pub hiz: bool,
    /// The bloom chain.
    pub bloom_chain: bool,
    /// A temporal upscaler.
    pub upscaler: Option<ResidentUpscaler>,
    /// The particle pipelines.
    pub particles: bool,
    /// The fog pipelines.
    pub fog: bool,
    /// The decal pipeline.
    pub decals: bool,
    /// The line pipeline.
    pub lines: bool,
    /// The atmosphere's tables and passes.
    pub atmosphere: bool,
    /// Every transparent producer's ray-traced pipeline.
    pub rt_transparent: bool,
    /// The TAA resolve.
    pub taa: bool,
    /// The G-buffer pre-pass's targets and its model-history snapshot.
    pub model_history: bool,
    /// The reflection composite, which blends a resolve's reflections over
    /// the scene.
    pub reflection_composite: bool,
}

/// The temporal upscaler the backend built.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidentUpscaler {
    /// It reads the reactive mask.
    pub reads_reactive: bool,
}

/// What the world puts on screen this frame. Each field is one read of the
/// state that holds it, with no pipeline test: [`FrameFacts::derive`] pairs
/// them with [`ResidentFeatures`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameContent {
    /// An opaque menu backdrop hides the world.
    pub world_hidden: bool,
    /// Records the cull processes this frame.
    pub cull_records: usize,
    /// Shadowed spot lights.
    pub shadowed_spot_count: u32,
    /// The live bloom intensity.
    pub bloom_intensity: f32,
    /// A ray-tracing acceleration structure is live.
    pub bvh_live: bool,
    /// A particle emitter is live.
    pub emitters: bool,
    /// The live fog settings contribute to the image.
    pub fog: bool,
    /// A decal exists.
    pub decals: bool,
    /// A water surface or glass panel is visible.
    pub visible_surfaces: bool,
    /// A see-through mesh is visible and its pipelines are built.
    pub visible_see_through: bool,
    /// Planar mirror targets exist.
    pub planar_targets: bool,
    /// A visible water surface holds a mirror slot.
    pub water_planar_slot: bool,
    /// Lines were published this frame.
    pub lines: bool,
    /// An SDF volume is visible.
    pub visible_volumes: bool,
    /// The atmosphere has this frame's inputs.
    pub atmosphere: bool,
    /// A visible surface samples a simulated ocean.
    pub water_sim: bool,
    /// Grass is planted.
    pub grass: bool,
    /// The SSGI accumulation is built and contributes, so it reprojects.
    pub ssgi_motion: bool,
}

/// What the planner decides ahead of the graph gates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlannedFacts {
    /// The cascades the shadow schedule refreshes this frame, one bit each.
    pub cascade_mask: u32,
    /// Local lights or probes are clustered.
    pub clustered: bool,
}

impl FrameFacts {
    /// The facts a backend holding `resident` sees over `content`, with what
    /// the planner decided in `planned`.
    pub fn derive(
        resident: &ResidentFeatures,
        content: &FrameContent,
        planned: &PlannedFacts,
    ) -> Self {
        let r = resident;
        let c = content;
        Self {
            hdr_width: r.hdr_width,
            hdr_height: r.hdr_height,
            hdr_sample_count: r.hdr_sample_count,
            shadow: r.shadow,
            shadow_map_size: r.shadow_map_size,
            shadowed_spot_count: c.shadowed_spot_count,
            spot_shadow_slice_size: spot_shadow_slice_size(r.shadow_map_size),
            world_hidden: c.world_hidden,
            cull_ready: r.cull,
            cull_records: c.cull_records,
            two_pass_ready: r.two_pass,
            hiz: r.hiz,
            bloom_chain: r.bloom_chain,
            bloom_intensity: c.bloom_intensity,
            upscaler: r.upscaler.is_some(),
            upscaler_reads_reactive: r.upscaler.is_some_and(|u| u.reads_reactive),
            bvh_live: c.bvh_live,
            particles: r.particles && c.emitters,
            fog: r.fog && c.fog,
            decals: r.decals && c.decals,
            visible_surfaces: c.visible_surfaces,
            visible_see_through: c.visible_see_through,
            rt_transparent_ready: r.rt_transparent,
            planar_targets: c.planar_targets,
            water_planar_slot: c.water_planar_slot,
            lines: r.lines && c.lines,
            visible_volumes: c.visible_volumes,
            atmosphere: r.atmosphere && c.atmosphere,
            water_sim: c.water_sim,
            grass: c.grass,
            grass_casts: c.grass && r.shadow && planned.cascade_mask & 1 != 0,
            clustered: planned.clustered,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLANNED: PlannedFacts = PlannedFacts {
        cascade_mask: 0b1,
        clustered: true,
    };

    fn everything() -> (ResidentFeatures, FrameContent) {
        let resident = ResidentFeatures {
            hdr_width: 64,
            hdr_height: 32,
            hdr_sample_count: 1,
            shadow: true,
            shadow_map_size: 2048,
            cull: true,
            two_pass: true,
            hiz: true,
            bloom_chain: true,
            upscaler: Some(ResidentUpscaler {
                reads_reactive: true,
            }),
            particles: true,
            fog: true,
            decals: true,
            lines: true,
            atmosphere: true,
            rt_transparent: true,
            taa: true,
            model_history: true,
            reflection_composite: true,
        };
        let content = FrameContent {
            world_hidden: false,
            cull_records: 3,
            shadowed_spot_count: 2,
            bloom_intensity: 0.5,
            bvh_live: true,
            emitters: true,
            fog: true,
            decals: true,
            visible_surfaces: true,
            visible_see_through: true,
            planar_targets: true,
            water_planar_slot: true,
            lines: true,
            visible_volumes: true,
            atmosphere: true,
            water_sim: true,
            grass: true,
            ssgi_motion: true,
        };
        (resident, content)
    }

    // A pass whose content is live runs only once its pipeline is built, and a
    // built pipeline runs nothing without content.
    #[test]
    fn content_runs_only_over_a_built_pipeline() {
        type Unbuild = fn(&mut ResidentFeatures);
        type Empty = fn(&mut FrameContent);
        type Read = fn(&FrameFacts) -> bool;
        let cases: [(&str, Unbuild, Empty, Read); 5] = [
            (
                "particles",
                |r| r.particles = false,
                |c| c.emitters = false,
                |f| f.particles,
            ),
            ("fog", |r| r.fog = false, |c| c.fog = false, |f| f.fog),
            (
                "decals",
                |r| r.decals = false,
                |c| c.decals = false,
                |f| f.decals,
            ),
            (
                "lines",
                |r| r.lines = false,
                |c| c.lines = false,
                |f| f.lines,
            ),
            (
                "atmosphere",
                |r| r.atmosphere = false,
                |c| c.atmosphere = false,
                |f| f.atmosphere,
            ),
        ];
        for (name, unbuild, empty, read) in cases {
            let (r, c) = everything();
            assert!(read(&FrameFacts::derive(&r, &c, &PLANNED)), "{name} runs");
            let mut unbuilt = r;
            unbuild(&mut unbuilt);
            assert!(
                !read(&FrameFacts::derive(&unbuilt, &c, &PLANNED)),
                "{name} unbuilt"
            );
            let mut emptied = c;
            empty(&mut emptied);
            assert!(
                !read(&FrameFacts::derive(&r, &emptied, &PLANNED)),
                "{name} empty"
            );
        }
    }

    #[test]
    fn grass_casts_only_on_a_frame_that_refreshes_the_first_cascade() {
        let (r, mut c) = everything();
        assert!(FrameFacts::derive(&r, &c, &PLANNED).grass_casts);
        let second_only = PlannedFacts {
            cascade_mask: 0b10,
            ..PLANNED
        };
        assert!(!FrameFacts::derive(&r, &c, &second_only).grass_casts);
        let no_shadow = ResidentFeatures { shadow: false, ..r };
        assert!(!FrameFacts::derive(&no_shadow, &c, &PLANNED).grass_casts);
        c.grass = false;
        assert!(!FrameFacts::derive(&r, &c, &PLANNED).grass_casts);
    }

    #[test]
    fn spot_slices_are_sized_from_the_cascade_map() {
        let (r, c) = everything();
        let facts = FrameFacts::derive(&r, &c, &PLANNED);
        assert_eq!(facts.spot_shadow_slice_size, spot_shadow_slice_size(2048));
        assert_eq!(facts.shadowed_spot_count, 2);
    }

    #[test]
    fn the_upscaler_reads_the_mask_only_when_built() {
        let (r, c) = everything();
        let facts = FrameFacts::derive(&r, &c, &PLANNED);
        assert!(facts.upscaler && facts.upscaler_reads_reactive);
        let none = ResidentFeatures {
            upscaler: None,
            ..r
        };
        let facts = FrameFacts::derive(&none, &c, &PLANNED);
        assert!(!facts.upscaler && !facts.upscaler_reads_reactive);
    }
}
