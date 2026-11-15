// The conditions a pass row names, and the layer each one belongs to.

use core::fmt;

use super::PassInputs;
use crate::gfx::view_modes::ShowFlags;
use crate::render::planar_reflection::planar_pass_needed;

/// Where a pass decision was made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    /// The resolved settings and what the last settings apply built.
    Setting,
    /// The backend's graphics API.
    Backend,
    /// What the world holds this frame and what launch built for it.
    Content,
    /// Several of the layers above at once, such as which reflection resolve
    /// takes the slot.
    Combined,
    /// The viewport's view mode.
    ViewMode,
    /// The viewport's show flags.
    ShowFlag,
    /// A pass whose output this one reads is off.
    Input,
    /// An opaque menu backdrop hides the world.
    WorldHidden,
}

impl Layer {
    /// The layer's name in reports.
    pub fn name(self) -> &'static str {
        match self {
            Self::Setting => "setting",
            Self::Backend => "backend",
            Self::Content => "content",
            Self::Combined => "combined",
            Self::ViewMode => "view_mode",
            Self::ShowFlag => "show_flag",
            Self::Input => "input",
            Self::WorldHidden => "world_hidden",
        }
    }
}

/// One condition a frame can meet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fact {
    /// TAA is the anti-aliasing mode and its resolve was built.
    Taa,
    /// SSAO is on and built.
    Ssao,
    /// The SSGI trace is on and built.
    Ssgi,
    /// SSGI is the indirect technique and built, so the main pass samples its
    /// resolve.
    SsgiResolve,
    /// Auto-exposure is on and built.
    AutoExposure,
    /// The G-buffer pre-pass targets exist.
    GBuffer,
    /// The device can write the reactive mask beside the scene.
    ReactiveTarget,
    /// The cascade shadow pipeline and map were built.
    Shadow,
    /// A spot light casts shadows.
    SpotShadows,
    /// The far shadow map was built beside the cascades and redraws this
    /// frame.
    FarShadowRedraw,
    /// The GPU-driven cull runs: its pipeline is ready, it has records, and
    /// the world is shown.
    Cull,
    /// Two-pass occlusion was built and the cull has records.
    TwoPass,
    /// The Hi-Z pyramid exists.
    Hiz,
    /// The shadow plan measures this frame's receivers over a built cascade
    /// map and pyramid.
    ShadowReceivers,
    /// The view sampling was built over the pyramid.
    ViewSamples,
    /// The bloom chain exists and its intensity is positive.
    Bloom,
    /// A temporal upscaler runs.
    Upscaler,
    /// The running upscaler reads the reactive mask.
    UpscalerReadsReactive,
    /// The disocclusion mark was built.
    Disocclusion,
    /// A particle emitter is live.
    Particles,
    /// A live particle emitter draws over the see-through surfaces.
    LateParticles,
    /// Fog is built and contributes.
    Fog,
    /// A decal exists.
    Decals,
    /// A water surface or glass panel is visible, or a see-through mesh whose
    /// trace runs.
    Transparent,
    /// A flat reflector's mirror is sampled this frame.
    PlanarMirrors,
    /// Lines were published.
    Lines,
    /// An SDF volume is visible.
    Volumes,
    /// The atmosphere has this frame's inputs.
    Atmosphere,
    /// The atmosphere has cloud layers with this frame's inputs.
    Clouds,
    /// A visible surface samples a simulated ocean.
    WaterSim,
    /// Grass is planted.
    Grass,
    /// Grass casts into a cascade this frame.
    GrassCasts,
    /// Local lights or probes are clustered.
    Clustered,
    /// A prop reflection probe rides this frame.
    PropProbe,
    /// The screen-space reflection resolve takes the reflection slot.
    SsrResolve,
    /// The ray-traced reflection node takes the reflection slot.
    RtNode,
    /// Ray-traced reflections trace a live BVH.
    RtTrace,
    /// A pass reads the G-buffer pre-pass's output.
    PrepassRead,
}

impl Fact {
    /// Whether the frame `inputs` describes meets it.
    pub fn holds(self, inputs: &PassInputs) -> bool {
        let g = &inputs.gates;
        let f = &inputs.facts;
        match self {
            Self::Taa => g.taa,
            Self::Ssao => g.ssao,
            Self::Ssgi => g.ssgi,
            Self::SsgiResolve => g.ssgi_resolve,
            Self::AutoExposure => g.auto_exposure,
            Self::GBuffer => g.gbuffer,
            Self::ReactiveTarget => g.traits.reactive_mask,
            Self::Shadow => f.shadow,
            Self::SpotShadows => f.shadowed_spot_count > 0,
            Self::FarShadowRedraw => f.shadow && f.far_shadow_map_size > 0 && f.far_shadow_drawn,
            Self::Cull => f.cull_ready && f.cull_records > 0 && !f.world_hidden,
            Self::TwoPass => f.two_pass_ready && f.cull_records > 0,
            Self::Hiz => f.hiz,
            Self::ShadowReceivers => f.shadow && f.hiz && f.shadow_receivers,
            Self::ViewSamples => f.hiz && f.view_samples,
            Self::Bloom => f.bloom_chain && f.bloom_intensity > 0.0,
            Self::Upscaler => f.upscaler,
            Self::UpscalerReadsReactive => f.upscaler && f.upscaler_reads_reactive,
            Self::Disocclusion => f.disocclusion,
            Self::Particles => f.particles,
            Self::LateParticles => f.late_particles,
            Self::Fog => f.fog,
            Self::Decals => f.decals,
            Self::Transparent => {
                f.visible_surfaces || (f.visible_see_through && inputs.rt_transparent())
            }
            Self::PlanarMirrors => planar_pass_needed(
                f.planar_targets,
                f.water_planar_slot,
                inputs.rt_transparent(),
            ),
            Self::Lines => f.lines,
            Self::Volumes => f.visible_volumes,
            Self::Atmosphere => f.atmosphere,
            Self::Clouds => f.atmosphere && f.clouds,
            Self::WaterSim => f.water_sim,
            Self::Grass => f.grass,
            Self::GrassCasts => f.grass_casts,
            Self::Clustered => f.clustered,
            Self::PropProbe => f.prop_probe,
            Self::SsrResolve => inputs.reflection_path().ssr_resolve(),
            Self::RtNode => {
                let path = inputs.reflection_path();
                if g.traits.composite_scene_fixed {
                    path.rt_node()
                } else {
                    path.rt_trace()
                }
            }
            Self::RtTrace => inputs.reflection_path().rt_trace(),
            Self::PrepassRead => inputs.prepass_read(),
        }
    }

    /// The layer that decides it.
    pub fn layer(self) -> Layer {
        match self {
            Self::Taa
            | Self::Ssao
            | Self::Ssgi
            | Self::SsgiResolve
            | Self::AutoExposure
            | Self::GBuffer => Layer::Setting,
            Self::ReactiveTarget | Self::ViewSamples => Layer::Backend,
            Self::Transparent
            | Self::PlanarMirrors
            | Self::SsrResolve
            | Self::RtNode
            | Self::RtTrace
            | Self::PrepassRead => Layer::Combined,
            _ => Layer::Content,
        }
    }

    /// The fact's name in reports.
    pub fn name(self) -> &'static str {
        match self {
            Self::Taa => "taa",
            Self::Ssao => "ssao",
            Self::Ssgi => "ssgi",
            Self::SsgiResolve => "ssgi_resolve",
            Self::AutoExposure => "auto_exposure",
            Self::GBuffer => "gbuffer_targets",
            Self::ReactiveTarget => "reactive_target",
            Self::Shadow => "shadow",
            Self::SpotShadows => "spot_shadows",
            Self::FarShadowRedraw => "far_shadow_redraw",
            Self::Cull => "gpu_cull",
            Self::TwoPass => "two_pass_occlusion",
            Self::Hiz => "hiz",
            Self::ShadowReceivers => "shadow_receivers",
            Self::ViewSamples => "view_samples",
            Self::Bloom => "bloom",
            Self::Upscaler => "upscaler",
            Self::UpscalerReadsReactive => "upscaler_reads_reactive",
            Self::Disocclusion => "disocclusion",
            Self::Particles => "particles",
            Self::LateParticles => "late_particles",
            Self::Fog => "fog",
            Self::Decals => "decals",
            Self::Transparent => "transparent_surfaces",
            Self::PlanarMirrors => "planar_mirrors",
            Self::Lines => "lines",
            Self::Volumes => "sdf_volumes",
            Self::Atmosphere => "atmosphere",
            Self::Clouds => "clouds",
            Self::WaterSim => "water_sim",
            Self::Grass => "grass",
            Self::GrassCasts => "grass_casts",
            Self::Clustered => "clustered_lights",
            Self::PropProbe => "prop_probe",
            Self::SsrResolve => "ssr_resolve",
            Self::RtNode => "rt_node",
            Self::RtTrace => "rt_trace",
            Self::PrepassRead => "prepass_read",
        }
    }
}

/// One condition a pass row asks of the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Need {
    /// The fact holds.
    Has(Fact),
    /// The fact does not hold: another pass takes this one's place while it
    /// does.
    Lacks(Fact),
    /// The show flag is set.
    Show(ShowFlags),
    /// The view mode shades surfaces: not Unlit or Wireframe.
    ShadedView,
    /// The view mode presents the scene, not one channel of it.
    SceneView,
}

impl Need {
    /// Whether the frame `inputs` describes meets it.
    pub fn holds(self, inputs: &PassInputs) -> bool {
        match self {
            Self::Has(fact) => fact.holds(inputs),
            Self::Lacks(fact) => !fact.holds(inputs),
            Self::Show(flag) => inputs.show.contains(flag),
            Self::ShadedView => !inputs.mode.is_flat(),
            Self::SceneView => !inputs.mode.is_gbuffer_channel(),
        }
    }

    /// The layer that decides it.
    pub fn layer(self) -> Layer {
        match self {
            Self::Has(fact) | Self::Lacks(fact) => fact.layer(),
            Self::Show(_) => Layer::ShowFlag,
            Self::ShadedView | Self::SceneView => Layer::ViewMode,
        }
    }
}

impl fmt::Display for Need {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Has(fact) => write!(f, "needs {}", fact.name()),
            Self::Lacks(fact) => write!(f, "not with {}", fact.name()),
            Self::Show(flag) => {
                let name = ShowFlags::NAMED
                    .iter()
                    .find(|(named, _)| *named == flag)
                    .map_or("unnamed", |(_, name)| *name);
                write!(f, "needs the {name} show flag")
            }
            Self::ShadedView => f.write_str("needs a shaded view mode"),
            Self::SceneView => f.write_str("needs a view mode that shows the scene"),
        }
    }
}
