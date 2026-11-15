//! Which passes a frame runs and why, decided by one table for every backend.
//!
//! `PASS_TABLE` holds one row per [`PassId`]: the conditions the frame must
//! meet for the pass to be asked for (a settings gate, something the world
//! holds, the view mode, a show flag) and the passes it reads and cannot run
//! without. [`PassDecisions::select`] walks the table once a frame into a
//! [`PassDecision`] per pass, and [`PassDecisions::graph_inputs`] fills the
//! [`FrameGraphInputs`] the frame graph is built from.

mod need;
#[cfg(test)]
mod reference;
#[cfg(test)]
pub(super) mod samples;
mod table;
#[cfg(test)]
mod tests;

use core::fmt;

pub use need::{Fact, Layer, Need};
use table::{PASS_TABLE, PassRow};

use super::FrameGraphInputs;
use super::gates::{FeatureGates, FrameFacts};
use super::passes::{PASS_COUNT, PassId};
use crate::gfx::view_modes::{ShowFlags, ViewMode};
use crate::render::post::reflection_path::ReflectionPath;

/// Everything a frame's pass selection reads.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PassInputs {
    /// The runtime-switched features the settings and the last apply leave on.
    pub gates: FeatureGates,
    /// What the backend holds this frame.
    pub facts: FrameFacts,
    /// The viewport's view mode.
    pub mode: ViewMode,
    /// The viewport's show flags.
    pub show: ShowFlags,
}

impl PassInputs {
    /// Which reflection resolve takes the slot.
    fn reflection_path(&self) -> ReflectionPath {
        ReflectionPath::new(self.gates.ssr, self.gates.rt_pass, self.facts.bvh_live)
    }

    /// The see-through surfaces are traced: a live trace and every
    /// transparent producer's traced pipeline.
    fn rt_transparent(&self) -> bool {
        self.reflection_path().rt_trace() && self.facts.rt_transparent_ready
    }

    /// A pass reprojects through the pre-pass's motion.
    fn velocity(&self) -> bool {
        !self.mode.is_flat()
            && (self.gates.taa
                || self.facts.upscaler
                || self.gates.ssgi
                || self.reflection_path().rt_trace())
    }

    /// A screen-space pass reads the pre-pass's normal and depth, or the view
    /// mode shows one of its channels.
    fn screen_space_read(&self) -> bool {
        let g = &self.gates;
        self.mode.is_gbuffer_channel()
            || (!self.mode.is_flat() && (g.ssr || g.ssgi || g.ssgi_resolve || g.rt_pass))
    }

    /// Anything reads the pre-pass's output.
    fn prepass_read(&self) -> bool {
        self.screen_space_read() || (self.gates.ssao && !self.mode.is_flat()) || self.velocity()
    }
}

/// What decided a pass's decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// Every condition holds and every pass feeding it runs.
    Runs,
    /// An opaque menu backdrop hides the world it draws.
    WorldHidden,
    /// The first condition in its row that the frame does not meet.
    Needs(Need),
    /// A pass whose output it reads is off.
    FedBy(PassId),
}

impl Reason {
    /// The layer that decided it, `None` for a pass that runs.
    pub fn layer(self) -> Option<Layer> {
        match self {
            Self::Runs => None,
            Self::WorldHidden => Some(Layer::WorldHidden),
            Self::Needs(need) => Some(need.layer()),
            Self::FedBy(_) => Some(Layer::Input),
        }
    }
}

impl fmt::Display for Reason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runs => f.write_str("runs"),
            Self::WorldHidden => f.write_str("the world is hidden"),
            Self::Needs(need) => need.fmt(f),
            Self::FedBy(pass) => write!(f, "its input {} is off", pass.name()),
        }
    }
}

/// Whether one pass runs this frame, and why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassDecision {
    /// Every condition in its row holds.
    pub asked: bool,
    /// The pass is in the frame graph: it is asked for, every pass feeding it
    /// runs, and the world it draws is shown.
    pub runs: bool,
    /// What decided it.
    pub reason: Reason,
}

impl PassDecision {
    // A slot the selection has not reached yet.
    const PENDING: Self = Self {
        asked: false,
        runs: false,
        reason: Reason::Runs,
    };
}

/// One [`PassDecision`] per pass, indexed by [`PassId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassDecisions([PassDecision; PASS_COUNT]);

impl PassDecisions {
    /// Walk `PASS_TABLE` over `inputs`.
    pub fn select(inputs: &PassInputs) -> Self {
        let mut decisions = Self([PassDecision::PENDING; PASS_COUNT]);
        for row in &PASS_TABLE {
            decisions.0[row.pass as usize] = decisions.decide(row, inputs);
        }
        decisions
    }

    // One row's decision, over the rows already decided.
    fn decide(&self, row: &PassRow, inputs: &PassInputs) -> PassDecision {
        let unmet = row.needs.iter().copied().find(|need| !need.holds(inputs));
        let asked = unmet.is_none();
        let reason = if row.draws_world && inputs.facts.world_hidden {
            Reason::WorldHidden
        } else if let Some(need) = unmet {
            Reason::Needs(need)
        } else if let Some(&input) = row.fed_by.iter().find(|&&p| !self.get(p).runs) {
            Reason::FedBy(input)
        } else {
            Reason::Runs
        };
        PassDecision {
            asked,
            runs: reason == Reason::Runs,
            reason,
        }
    }

    /// The decision for `pass`.
    pub fn get(&self, pass: PassId) -> PassDecision {
        self.0[pass as usize]
    }

    // Whether every condition in `pass`'s row holds.
    fn asked(&self, pass: PassId) -> bool {
        self.get(pass).asked
    }

    /// Every pass with its decision, in [`PassId`] order.
    pub fn iter(&self) -> impl Iterator<Item = (PassId, PassDecision)> + '_ {
        PassId::ALL.iter().map(|&pass| (pass, self.get(pass)))
    }

    /// The frame graph inputs these decisions ask for over `inputs`.
    pub fn graph_inputs(&self, inputs: &PassInputs) -> FrameGraphInputs {
        let f = &inputs.facts;
        let asked = |pass| self.asked(pass);
        FrameGraphInputs {
            shadow_enabled: asked(PassId::Shadow),
            shadow_map_size: f.shadow_map_size,
            far_shadow_map_size: if f.shadow { f.far_shadow_map_size } else { 0 },
            far_shadow_drawn: f.shadow && f.far_shadow_drawn,
            hdr_width: f.hdr_width,
            hdr_height: f.hdr_height,
            hdr_sample_count: f.hdr_sample_count,
            bindless_cull_enabled: asked(PassId::Cull),
            auto_exposure_enabled: asked(PassId::AutoExposure),
            bloom_enabled: asked(PassId::Bloom),
            velocity_enabled: inputs.velocity(),
            taa_enabled: asked(PassId::TaaResolve),
            ssr_enabled: asked(PassId::SsrResolve),
            particles_enabled: asked(PassId::ParticlesSim),
            late_particles_enabled: asked(PassId::ParticlesLateDraw),
            fog_enabled: asked(PassId::Fog),
            decals_enabled: asked(PassId::Decals),
            ssr_prepass_enabled: inputs.screen_space_read(),
            ssao_enabled: asked(PassId::SsaoBlur),
            upscale_enabled: asked(PassId::Upscale),
            transparent_enabled: asked(PassId::Transparent),
            planar_reflection_enabled: asked(PassId::PlanarReflection),
            prop_probe_enabled: asked(PassId::PropProbe),
            lines_enabled: asked(PassId::Lines),
            raymarch_enabled: asked(PassId::Raymarch),
            two_pass_occlusion_enabled: asked(PassId::MainLate),
            ssgi_enabled: asked(PassId::Ssgi),
            ssgi_resolve_enabled: asked(PassId::SsgiResolve),
            rt_reflections_enabled: asked(PassId::RtReflections),
            gbuffer_prepass_enabled: inputs.gates.gbuffer,
            world_hidden: f.world_hidden,
            clustering_enabled: asked(PassId::LightCull),
            composite_reads_ao: inputs.mode == ViewMode::Occlusion && asked(PassId::SsaoBlur),
            composite_reads_motion: inputs.mode == ViewMode::Motion,
            composite_reads_reactive: inputs.mode == ViewMode::Reactive,
            shadowed_spot_count: if asked(PassId::SpotShadow) {
                f.shadowed_spot_count
            } else {
                0
            },
            spot_shadow_slice_size: f.spot_shadow_slice_size,
            hiz_build_enabled: asked(PassId::HizFinal),
            shadow_receivers_enabled: asked(PassId::ShadowReceivers),
            reactive_mask_enabled: Fact::ReactiveTarget.holds(inputs)
                && inputs.show.contains(ShowFlags::REACTIVE),
            upscale_reads_reactive: Fact::UpscalerReadsReactive.holds(inputs),
            disocclusion_enabled: f.disocclusion,
            grass_enabled: asked(PassId::Grass),
            grass_shadow_enabled: asked(PassId::GrassShadow),
            atmosphere_enabled: asked(PassId::Atmosphere),
            clouds_enabled: asked(PassId::Clouds),
            water_sim_enabled: asked(PassId::WaterSim),
        }
    }
}
