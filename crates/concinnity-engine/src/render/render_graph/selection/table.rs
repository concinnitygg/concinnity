// One row per render graph pass: what it needs and what it reads.

use super::need::Fact::{
    Atmosphere, AutoExposure, Bloom, Clouds, Clustered, Cull, Decals, Disocclusion,
    FarShadowRedraw, Fog, GBuffer, Grass, GrassCasts, Hiz, LateParticles, Lines, Particles,
    PlanarMirrors, PrepassRead, PropProbe, ReactiveTarget, RtNode, RtTrace, Shadow,
    ShadowReceivers, SpotShadows, Ssao, Ssgi, SsgiResolve, SsrResolve, Taa, Transparent, TwoPass,
    Upscaler, UpscalerReadsReactive, ViewSamples, Volumes, WaterSim,
};
use super::need::Need::{self, Has, Lacks, Relevant, SceneView, ShadedView, Show};
use crate::gfx::view_modes::ShowFlags;
use crate::render::relevance::Effect;
use crate::render::render_graph::passes::{PASS_COUNT, PassId};

/// What one pass needs to run.
#[derive(Clone, Copy, Debug)]
pub(super) struct PassRow {
    /// The pass.
    pub(super) pass: PassId,
    /// Every condition the frame must meet for the pass to be asked for.
    pub(super) needs: &'static [Need],
    /// The passes whose output this one reads and cannot run without. Each
    /// is decided by a row earlier in [`PASS_TABLE`].
    pub(super) fed_by: &'static [PassId],
    /// It draws or feeds the world, so it stays off while a menu backdrop
    /// hides it.
    pub(super) draws_world: bool,
}

const fn world(pass: PassId, needs: &'static [Need], fed_by: &'static [PassId]) -> PassRow {
    PassRow {
        pass,
        needs,
        fed_by,
        draws_world: true,
    }
}

const fn frame_pass(pass: PassId, needs: &'static [Need]) -> PassRow {
    PassRow {
        pass,
        needs,
        fed_by: &[],
        draws_world: false,
    }
}

const SHADOWS: Need = Show(ShowFlags::SHADOWS);
const FOG: Need = Show(ShowFlags::FOG);
const BLOOM: Need = Show(ShowFlags::BLOOM);
const SSGI: Need = Show(ShowFlags::SSGI);
const SSR: Need = Show(ShowFlags::SSR);
const LINES: Need = Show(ShowFlags::LINES);
const REACTIVE: Need = Show(ShowFlags::REACTIVE);

/// Every render graph pass, in the order the selection decides them: a pass
/// comes after each pass it is fed by.
pub(super) const PASS_TABLE: [PassRow; PASS_COUNT] = [
    world(PassId::Cull, &[Has(Cull)], &[]),
    world(
        PassId::Grass,
        &[Has(Grass), Has(Cull), Relevant(Effect::Grass)],
        &[],
    ),
    world(
        PassId::GrassBend,
        &[Has(Grass), Has(Cull), Relevant(Effect::Grass)],
        &[],
    ),
    world(
        PassId::Shadow,
        &[Has(Shadow), SHADOWS, Relevant(Effect::ShadowCascades)],
        &[],
    ),
    world(
        PassId::GrassShadow,
        &[Has(GrassCasts)],
        &[PassId::Grass, PassId::Shadow],
    ),
    world(PassId::SpotShadow, &[Has(SpotShadows), SHADOWS], &[]),
    world(PassId::FarShadow, &[Has(FarShadowRedraw)], &[]),
    world(PassId::LightCull, &[Has(Clustered)], &[]),
    world(PassId::Atmosphere, &[Has(Atmosphere)], &[]),
    world(PassId::AerialPerspective, &[Has(Atmosphere)], &[]),
    world(PassId::ParticlesSim, &[Has(Particles)], &[]),
    world(PassId::ParticlesDraw, &[Has(Particles)], &[]),
    world(
        PassId::ParticlesLateDraw,
        &[Has(LateParticles)],
        &[PassId::ParticlesSim],
    ),
    world(
        PassId::PropProbe,
        &[
            Has(PropProbe),
            Has(Cull),
            Lacks(RtTrace),
            Relevant(Effect::PropProbe),
        ],
        &[],
    ),
    world(
        PassId::GBufferPrepass,
        &[Has(GBuffer), Has(PrepassRead)],
        &[],
    ),
    world(
        PassId::SsaoBlur,
        &[Has(Ssao), ShadedView, Relevant(Effect::AmbientOcclusion)],
        &[],
    ),
    world(
        PassId::SsgiResolve,
        &[
            Has(SsgiResolve),
            ShadedView,
            Relevant(Effect::AmbientOcclusion),
        ],
        &[],
    ),
    frame_pass(PassId::Main, &[]),
    world(PassId::HizEarly, &[Has(TwoPass)], &[PassId::Cull]),
    world(PassId::CullLate, &[Has(TwoPass)], &[PassId::Cull]),
    world(PassId::MainLate, &[Has(TwoPass)], &[PassId::Cull]),
    world(
        PassId::GBufferLate,
        &[Has(TwoPass)],
        &[PassId::Cull, PassId::GBufferPrepass],
    ),
    world(PassId::AutoExposure, &[Has(AutoExposure)], &[]),
    world(PassId::Raymarch, &[Has(Volumes)], &[]),
    world(
        PassId::Ssgi,
        &[
            Has(Ssgi),
            SSGI,
            ShadedView,
            Relevant(Effect::IndirectLighting),
            Relevant(Effect::AmbientOcclusion),
        ],
        &[],
    ),
    world(PassId::Decals, &[Has(Decals), ShadedView], &[]),
    world(PassId::FogFroxel, &[Has(Fog), FOG, ShadedView], &[]),
    world(PassId::Fog, &[Has(Fog), FOG, ShadedView], &[]),
    world(PassId::Lines, &[Has(Lines), LINES], &[]),
    world(PassId::Transparent, &[Has(Transparent)], &[]),
    world(
        PassId::PlanarReflection,
        &[Has(PlanarMirrors)],
        &[PassId::Transparent],
    ),
    world(
        PassId::WaterSim,
        &[Has(WaterSim), Relevant(Effect::WaterSim)],
        &[PassId::Transparent],
    ),
    world(
        PassId::RtReflections,
        &[Has(RtNode), SSR, ShadedView, Relevant(Effect::Reflections)],
        &[],
    ),
    world(
        PassId::SsrResolve,
        &[
            Has(SsrResolve),
            SSR,
            ShadedView,
            Relevant(Effect::Reflections),
        ],
        &[],
    ),
    world(PassId::Clouds, &[Has(Clouds)], &[]),
    world(PassId::CloudComposite, &[Has(Clouds)], &[]),
    world(PassId::Upscale, &[Has(Upscaler), ShadedView], &[]),
    world(
        PassId::Disocclusion,
        &[
            Has(Disocclusion),
            Has(ReactiveTarget),
            REACTIVE,
            Has(UpscalerReadsReactive),
            Relevant(Effect::Disocclusion),
        ],
        &[PassId::GBufferPrepass, PassId::Upscale],
    ),
    world(
        PassId::TaaResolve,
        &[Has(Taa), Lacks(Upscaler), ShadedView],
        &[],
    ),
    world(
        PassId::Bloom,
        &[Has(Bloom), BLOOM, ShadedView, SceneView],
        &[],
    ),
    frame_pass(PassId::HizFinal, &[Has(Hiz)]),
    world(
        PassId::ShadowReceivers,
        &[Has(ShadowReceivers)],
        &[PassId::HizFinal, PassId::Shadow],
    ),
    world(
        PassId::ViewSamples,
        &[Has(ViewSamples)],
        &[PassId::HizFinal],
    ),
    frame_pass(PassId::Composite, &[]),
];
