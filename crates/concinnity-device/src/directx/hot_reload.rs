//! D3D12 shader hot-reload: `DxContext::reload_shaders` rebuilds every live
//! built-in PSO from the checkout's shader sources. The `reload-shaders` debug
//! command sets the shared flag, and the main thread polls it at the top of
//! `draw_frame`.
//!
//! Each feature's PSOs are one set below: how they are built from the current
//! sources and where they are installed. A reload builds every live set before
//! installing any, so a compile error leaves the running PSOs in place.

use concinnity_engine::render::backend::RenderBackend;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::post::reload::{self, PostPipelines};
use concinnity_engine::render::shader::pipeline_set::{self, Gated, PipelineSet, Pipelines};
use concinnity_engine::render::shader::programs::surface::SurfaceVariant;
use concinnity_engine::render::shader::world_pipelines::{
    BucketSource, check_rebuild, replace_bucket,
};
use windows::Win32::Graphics::Direct3D12::{ID3D12PipelineState, ID3D12RootSignature};

use super::atmosphere::{AtmospherePipelines, AtmosphereState};
use super::auto_exposure::AutoExposureResources;
use super::context::{DxContext, dump_on_err};
use super::cull::CullKernels;
use super::decal::DecalResources;
use super::fog::FogResources;
use super::grass::GrassPipelines;
use super::hiz::HiZResources;
use super::init::pipelines::{
    BindlessMainShaders, BucketPipelineTargets, BucketPsos, build_bucket_pipeline,
};
use super::line::LineResources;
use super::post::gbuffer::GbufferResources;
use super::post::post_device::{DxPostDevice, PostPipeline};
use super::post::rt_reflections::{
    RebuiltRtPipelines, RtReflectionsResources, rebuild_rt_reflections_pipelines,
    swap_rt_reflections_pipelines,
};
use super::pso::compute_pso;
use super::shadow_receivers::ShadowReceiversGpu;
use super::transparent::TransparentResources;
use super::view_samples::ViewSamplesGpu;
use crate::directx::builtin_shaders::{CULL_EARLY, CULL_LATE, CompileProgram as _};

type Pso = ID3D12PipelineState;

// Every feature's PSO set, in the order a reload builds them.
const PIPELINE_SETS: &[&dyn PipelineSet<DxContext>] = &[
    &COMPOSITE,
    &TEXT,
    &ENGINE_POST,
    &MAIN,
    &CULL,
    &HIZ,
    &AUTO_EXPOSURE,
    &SHADOW_RECEIVERS,
    &VIEW_SAMPLES,
    &SKY,
    &ATMOSPHERE,
    &GBUFFER_SKY,
    &GRASS,
    &DECAL,
    &LINES,
    &TRANSPARENT,
    &FOG,
    &RT_REFLECTIONS,
];

const COMPOSITE: Pipelines<DxContext, Pso> = Pipelines {
    build: |c, hr| {
        Some(
            super::pipeline::compile_composite_shaders(hr).and_then(|(vs, ps)| {
                dump_on_err(
                    c.hw.info_queue.as_ref(),
                    super::pipeline::create_composite_pso(
                        &c.hw.device,
                        &c.composite.root_sig,
                        &vs,
                        &ps,
                        c.swapchain.format,
                    ),
                )
            }),
        )
    },
    install: |c, p| c.composite.pso = p,
};

const TEXT: Pipelines<DxContext, Pso> = Pipelines {
    build: |c, hr| {
        c.text.pso.is_some().then(|| {
            let (vs, ps) = super::pipeline::compile_text_shaders(hr)?;
            dump_on_err(
                c.hw.info_queue.as_ref(),
                super::pipeline::create_text_pso(
                    &c.hw.device,
                    &c.text.root_sig,
                    &vs,
                    &ps,
                    c.swapchain.format,
                    1,
                ),
            )
        })
    },
    install: |c, p| c.text.pso = Some(p),
};

// The engine-owned post passes that are live.
const ENGINE_POST: Pipelines<DxContext, Vec<PostPipelines<PostPipeline>>> = Pipelines {
    build: |c, hr| {
        let device = DxPostDevice {
            hot_reload: hr,
            ..c.post_device(0)
        };
        Some(reload::build_live(&device, |pass| c.has_post_pass(pass)))
    },
    install: |c, built| {
        for pipelines in built {
            c.install_post_pipelines(pipelines);
        }
    },
};

// Every shader bucket's main and pre-pass PSOs, rebuilt as pairs so they never
// compile from different templates.
type MainPsos = (BucketPsos, Vec<(usize, BucketPsos)>, BindlessMainShaders);
const MAIN: Pipelines<DxContext, MainPsos> = Pipelines {
    build: |c, hr| {
        (c.cull.main_bindless_root_sig.is_some() && c.cull.main_bindless_pso.is_some()).then(|| {
            let engine = super::init::pipelines::compile_main_bindless_shaders(hr)?;
            let main = c.build_world_main_pso(c.world_shader.as_ref(), &engine, hr)?;
            let world_buckets = c.rebuild_world_buckets(&engine, hr)?;
            Ok((main, world_buckets, engine))
        })
    },
    install: |c, (main, world_buckets, engine)| {
        c.cull.main_bindless_pso = Some(main.main);
        c.cull.main_prepass_pso = main.prepass;
        c.swap_world_buckets(world_buckets);
        c.cull.bindless_main_shaders = engine;
        // The wireframe twins were built from the old sources; the next
        // wireframe frame rebuilds them.
        c.invalidate_wireframe_pipelines();
    },
};

// A compute PSO from `cs` against `root_sig`, its failures dumped from the
// info queue.
fn compute(
    c: &DxContext,
    root_sig: &ID3D12RootSignature,
    cs: &[u8],
    label: &str,
) -> RenderResult<Pso> {
    dump_on_err(
        c.hw.info_queue.as_ref(),
        compute_pso(&c.hw.device, root_sig, cs, label),
    )
}

// The cull PSO, and its late twin (two-pass occlusion) when built, both
// against the shared root signature.
const CULL: Gated<DxContext, CullKernels, (Pso, Option<Pso>)> = Gated {
    state: |c| c.cull.cull_kernels.as_ref(),
    state_mut: |c| c.cull.cull_kernels.as_mut(),
    build: |c, kernels, hr| {
        let create = |cs: &[u8]| {
            dump_on_err(
                c.hw.info_queue.as_ref(),
                super::cull::create_cull_pso(&c.hw.device, &kernels.root_sig, cs),
            )
        };
        let pso = create(&CULL_EARLY.compile(hr)?)?;
        let late = kernels
            .pso_late
            .as_ref()
            .map(|_| create(&CULL_LATE.compile(hr)?))
            .transpose()?;
        Ok((pso, late))
    },
    install: |kernels, (pso, late)| {
        kernels.pso = pso;
        kernels.pso_late = late;
    },
};

// The three SPD kernels, against the existing root signatures so the live
// cull root binding stays valid. The tail takes the signature without the
// depth SRV table.
const HIZ: Gated<DxContext, HiZResources, (Pso, Pso, Pso)> = Gated {
    state: |c| c.cull.hiz.as_ref(),
    state_mut: |c| c.cull.hiz.as_mut(),
    build: |c, hiz, hr| {
        let (single, msaa, tail) = super::hiz::compile_hiz_shaders(hr)?;
        Ok((
            compute(c, &hiz.root_sig, &single, "hiz spd_single")?,
            compute(c, &hiz.root_sig, &msaa, "hiz spd_msaa")?,
            compute(c, &hiz.tail_root_sig, &tail, "hiz spd_tail")?,
        ))
    },
    install: |hiz, (single, msaa, tail)| hiz.swap_pipelines(single, msaa, tail),
};

const AUTO_EXPOSURE: Gated<DxContext, AutoExposureResources, (Pso, Pso)> = Gated {
    state: |c| c.auto_exposure.resources.as_ref(),
    state_mut: |c| c.auto_exposure.resources.as_mut(),
    build: |c, ae, hr| {
        let (build, average) = super::auto_exposure::compile_auto_exposure_shaders(hr)?;
        Ok((
            compute(c, ae.build_root_sig(), &build, "auto-exposure build")?,
            compute(c, ae.average_root_sig(), &average, "auto-exposure average")?,
        ))
    },
    install: |ae, (build, average)| ae.swap_pipelines(build, average),
};

const SHADOW_RECEIVERS: Gated<DxContext, ShadowReceiversGpu, (Pso, Pso)> = Gated {
    state: |c| c.shadow_receivers.as_ref(),
    state_mut: |c| c.shadow_receivers.as_mut(),
    build: |c, r, hr| {
        dump_on_err(
            c.hw.info_queue.as_ref(),
            r.rebuild_pipelines(&c.hw.device, hr),
        )
    },
    install: |r, p| r.swap_pipelines(p),
};

const VIEW_SAMPLES: Gated<DxContext, ViewSamplesGpu, Pso> = Gated {
    state: |c| c.view_samples.as_ref(),
    state_mut: |c| c.view_samples.as_mut(),
    build: |c, v, hr| {
        dump_on_err(
            c.hw.info_queue.as_ref(),
            v.rebuild_pipeline(&c.hw.device, hr),
        )
    },
    install: |v, p| v.swap_pipeline(p),
};

const SKY: Pipelines<DxContext, Pso> = Pipelines {
    build: |c, hr| {
        Some(dump_on_err(
            c.hw.info_queue.as_ref(),
            super::sky::build_sky_pso(
                &c.hw.device,
                &super::builtin_shaders::SKY_FRAG,
                c.sky.root_sig(),
                c.targets.hdr.msaa_samples,
                hr,
            ),
        ))
    },
    install: |c, p| c.sky.swap_pso(p),
};

const ATMOSPHERE: Gated<DxContext, AtmosphereState, AtmospherePipelines> = Gated {
    state: |c| c.atmosphere.as_ref(),
    state_mut: |c| c.atmosphere.as_mut(),
    build: |c, a, hr| a.build_pipelines(&c.hw.device, c.targets.hdr.msaa_samples, hr),
    install: AtmosphereState::swap_pipelines,
};

// The sky's motion behind the G-buffer pre-pass.
const GBUFFER_SKY: Gated<DxContext, GbufferResources, Pso> = Gated {
    state: |c| c.gbuffer.as_ref(),
    state_mut: |c| c.gbuffer.as_mut(),
    build: |c, gb, hr| {
        gb.sky
            .rebuild_pso(&c.hw.device, c.hw.info_queue.as_ref(), hr)
    },
    install: |gb, p| gb.sky.swap_pso(p),
};

const GRASS: Pipelines<DxContext, GrassPipelines> = Pipelines {
    build: |c, hr| match (
        c.grass.as_ref(),
        c.cull.main_bindless_root_sig.as_ref(),
        c.cull.prepass_root_sig.as_ref(),
    ) {
        (Some(_), Some(main), Some(prepass)) => Some(dump_on_err(
            c.hw.info_queue.as_ref(),
            GrassPipelines::build(
                &c.hw.device,
                super::grass::GrassRootSigs { main, prepass },
                c.targets.hdr.msaa_samples,
                hr,
            ),
        )),
        _ => None,
    },
    install: |c, p| {
        if let Some(grass) = c.grass.as_mut() {
            grass.pipelines = p;
        }
    },
};

const DECAL: Gated<DxContext, DecalResources, Pso> = Gated {
    state: |c| c.decal.state.as_ref(),
    state_mut: |c| c.decal.state.as_mut(),
    build: |c, decals, hr| {
        super::decal::rebuild_decal_pso(
            &c.hw.device,
            &decals.root_sig,
            c.targets.hdr.msaa_samples,
            hr,
            c.hw.info_queue.as_ref(),
        )
    },
    install: |decals, p| decals.pso = p,
};

// Built once a frame published lines.
const LINES: Gated<DxContext, LineResources, Pso> = Gated {
    state: |c| c.lines.resources.as_ref(),
    state_mut: |c| c.lines.resources.as_mut(),
    build: |c, lines, hr| {
        super::line::rebuild_line_pso(
            &c.hw.device,
            lines,
            c.targets.hdr.msaa_samples,
            hr,
            c.hw.info_queue.as_ref(),
        )
    },
    install: |lines, p| lines.pso = p,
};

// The transparent pass's two producers, each only when the world declared its
// asset, against the pass's shared root signature.
const TRANSPARENT: Gated<DxContext, TransparentResources, (Option<Pso>, Option<Pso>)> = Gated {
    state: |c| c.transparent.as_ref(),
    state_mut: |c| c.transparent.as_mut(),
    build: |c, t, hr| {
        let device = &c.hw.device;
        let (samples, info_queue) = (c.targets.hdr.msaa_samples, c.hw.info_queue.as_ref());
        let glass = t
            .has_glass()
            .then(|| super::glass::rebuild_glass_pso(device, t.root_sig(), samples, hr, info_queue))
            .transpose()?;
        let water = t
            .has_water()
            .then(|| super::water::rebuild_water_pso(device, t.root_sig(), samples, hr, info_queue))
            .transpose()?;
        Ok((glass, water))
    },
    install: |t, (glass, water)| t.swap_pipelines(glass, water),
};

// The fullscreen fog draw and the froxel-volume kernel.
const FOG: Gated<DxContext, FogResources, (Pso, Pso)> = Gated {
    state: |c| c.fog.resources.as_ref(),
    state_mut: |c| c.fog.resources.as_mut(),
    build: |c, fog, hr| {
        let (device, info_queue) = (&c.hw.device, c.hw.info_queue.as_ref());
        let samples = c.targets.hdr.msaa_samples;
        let render = super::fog::rebuild_fog_pso(device, &fog.root_sig, samples, hr, info_queue)?;
        let froxel =
            super::fog::rebuild_fog_froxel_pso(device, &fog.froxel_root_sig, hr, info_queue)?;
        Ok((render, froxel))
    },
    install: |fog, (render, froxel)| {
        fog.pso = render;
        fog.froxel_pso = froxel;
    },
};

// Live only when DXR, the DXC compile and the acceleration build all
// succeeded at init.
const RT_REFLECTIONS: Gated<DxContext, RtReflectionsResources, RebuiltRtPipelines> = Gated {
    state: |c| c.rt_reflections.as_ref(),
    state_mut: |c| c.rt_reflections.as_mut(),
    build: |c, rt, hr| {
        rebuild_rt_reflections_pipelines(&c.hw.device, rt, hr, c.hw.info_queue.as_ref())
    },
    install: |rt, p| swap_rt_reflections_pipelines(rt, p),
};

impl DxContext {
    // Rebuild every live PSO set from the current shader sources.
    pub(super) fn reload_shaders(&mut self) -> RenderResult<()> {
        if !self.hot_reload.enabled {
            return Ok(());
        }
        self.hot_reload.generation += 1;
        pipeline_set::rebuild(self, PIPELINE_SETS, true)
    }
}

// World-Shader runtime hot-swap (RenderBackend::update_world_shader)

// cn-debug-only runtime-mutation surface; dead from the FFI lib crate's roots,
// live in the concinnity binary. See the note on the analogous block in
// [directx/particle.rs].
impl DxContext {
    // Rebuild bucket 0's main and pre-pass PSOs from the world default Shader, or
    // install `prepared`. Built first, swapped only on success.
    pub(in crate::directx) fn update_default_world_shader(
        &mut self,
        programs: &concinnity_engine::components::ShaderPrograms,
        prepared: Option<BucketPsos>,
    ) -> RenderResult<()> {
        let new = replace_bucket(
            0,
            self.cull.main_prepass_pso.is_some(),
            prepared,
            |p: &BucketPsos| p.prepass.is_some(),
            || {
                let engine = &self.cull.bindless_main_shaders;
                self.build_world_main_pso(Some(programs), engine, self.hot_reload.enabled)
            },
        )?;
        // Drain the GPU before the swap releases the displaced PSOs: a command
        // list does not keep one alive, and the debug reload drive does not
        // wait for us.
        self.wait_idle();
        self.cull.main_bindless_pso = Some(new.main);
        self.cull.main_prepass_pso = new.prepass;
        self.world_shader = Some(programs.clone());
        self.invalidate_wireframe_pipelines();
        Ok(())
    }

    // Bucket 0's PSOs: the world default Shader's programs, else
    // `engine_default`, from the sources `hot_reload` selects. Errors when the
    // GPU-driven pass is not live, or the rebuild would drop the live pre-pass.
    pub(super) fn build_world_main_pso(
        &self,
        world: Option<&concinnity_engine::components::ShaderPrograms>,
        engine_default: &BindlessMainShaders,
        hot_reload: bool,
    ) -> RenderResult<BucketPsos> {
        let targets = self
            .world_pso_targets()
            .ok_or_else(|| RenderError::Other("the GPU-driven main pass is not live".into()))?;
        let psos = build_bucket_pipeline(
            &self.hw.device,
            self.hw.info_queue.as_ref(),
            BucketPipelineTargets {
                root_sigs: &targets.root_sigs,
                msaa_samples: self.targets.hdr.msaa_samples,
                engine_default,
                hot_reload,
            },
            0,
            &BucketSource {
                programs: world,
                deferred: false,
                variant: SurfaceVariant::Plain,
            },
        )?;
        check_rebuild(
            0,
            self.cull.main_prepass_pso.is_some(),
            psos.prepass.is_some(),
        )?;
        Ok(psos)
    }
}
