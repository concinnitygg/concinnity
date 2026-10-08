// Each shader bucket's pipelines: the main pass that shades its draws, and the
// G-buffer pre-pass that lays down their depth, normal, roughness and motion
// from the same vertex hook. The main pipeline is the bucket's to have; the
// pre-pass exists only while a G-buffer consumer is on, and a pre-pass that
// fails to build costs that bucket its G-buffer draws, never its shading.

use concinnity_core::components::ShaderPrograms;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::shader_programs::ShaderProgram;
use concinnity_core::render::world_pipelines::replace_bucket;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLFunction, MTLRenderPipelineState, MTLVertexDescriptor};

use super::MtlContext;
use super::builtin_shaders::{self as engine, entry_function};
use super::init::pipelines::{build_main_pipeline, make_vertex_descriptor};
use super::pipeline::{
    WORLD_FRAGMENT_ENTRY, WORLD_PREPASS_FRAGMENT_ENTRY, WORLD_PREPASS_VERTEX_ENTRY,
    WORLD_VERTEX_ENTRY, world_function,
};

type Pipeline = Retained<ProtocolObject<dyn MTLRenderPipelineState>>;

pub(crate) struct BucketPipelines {
    pub main: Pipeline,
    // `None` while no G-buffer consumer is on, or after this bucket's pre-pass
    // failed to build; the pre-pass then skips the bucket's draws.
    pub prepass: Option<Pipeline>,
    // The world Shader the bucket compiles, `None` for the engine's own
    // programs, kept so the pre-pass can be built or rebuilt later.
    pub programs: Option<ShaderPrograms>,
}

// What every bucket's pipelines are built under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BucketBuild {
    pub hot_reload: bool,
    pub sample_count: u32,
    // A G-buffer consumer is on, so the bucket gets a pre-pass pipeline.
    pub prepass: bool,
    // The engine-template reload the pipelines compile from.
    pub template_generation: u64,
}

// A pipeline's two stages.
type StagePair = (
    Retained<ProtocolObject<dyn MTLFunction>>,
    Retained<ProtocolObject<dyn MTLFunction>>,
);

// The two stages of one pass for a bucket: the engine's programs, or the world
// Shader's compile of the same file with its hooks spliced in.
fn bucket_stages(
    device: &ProtocolObject<dyn MTLDevice>,
    world: Option<&ShaderPrograms>,
    engine: [&ShaderProgram; 2],
    entries: [&str; 2],
    hot_reload: bool,
) -> RenderResult<StagePair> {
    Ok(match world {
        None => (
            entry_function(device, engine[0], hot_reload)?,
            entry_function(device, engine[1], hot_reload)?,
        ),
        Some(programs) => (
            world_function(device, hot_reload, programs, entries[0])?,
            world_function(device, hot_reload, programs, entries[1])?,
        ),
    })
}

fn bucket_context(bucket: usize) -> impl Fn(RenderError) -> RenderError {
    move |e| match bucket {
        0 => e.context("the world's main pass"),
        b => e.context(format_args!("shader bucket {b}")),
    }
}

// Build shader bucket `bucket`'s pipelines: the engine's own programs for
// bucket 0 of a world without a default Shader, else `world`'s. Both opt into
// indirect command buffers, since every bucket draws through its ICB. A
// pre-pass that fails to build leaves the bucket without one.
pub(crate) fn build_bucket_pipelines(
    device: &ProtocolObject<dyn MTLDevice>,
    vert_desc: &MTLVertexDescriptor,
    bucket: usize,
    world: Option<&ShaderPrograms>,
    build: BucketBuild,
) -> RenderResult<BucketPipelines> {
    let main = build_bucket_main(device, vert_desc, bucket, world, build)?;
    let prepass = match build.prepass {
        true => build_bucket_prepass(device, bucket, world, build.hot_reload),
        false => None,
    };
    Ok(BucketPipelines {
        main,
        prepass,
        programs: world.cloned(),
    })
}

// Shader bucket `bucket`'s pipelines to replace `live`: `prepared` when it
// keeps what `live` draws with, else ones built here. Refused when the build
// would drop a pre-pass `live` draws with, so the caller keeps `live`.
pub(crate) fn replacement(
    device: &ProtocolObject<dyn MTLDevice>,
    bucket: usize,
    world: Option<&ShaderPrograms>,
    build: BucketBuild,
    live: Option<&BucketPipelines>,
    prepared: Option<BucketPipelines>,
) -> RenderResult<BucketPipelines> {
    replace_bucket(
        bucket,
        live.is_some_and(|l| l.prepass.is_some()),
        prepared,
        |p: &BucketPipelines| p.prepass.is_some(),
        || build_bucket_pipelines(device, &make_vertex_descriptor(), bucket, world, build),
    )
}

fn build_bucket_main(
    device: &ProtocolObject<dyn MTLDevice>,
    vert_desc: &MTLVertexDescriptor,
    bucket: usize,
    world: Option<&ShaderPrograms>,
    build: BucketBuild,
) -> RenderResult<Pipeline> {
    let (vert_fn, frag_fn) = bucket_stages(
        device,
        world,
        [&engine::MAIN_BINDLESS_VERT, &engine::MAIN_BINDLESS_FRAG],
        [WORLD_VERTEX_ENTRY, WORLD_FRAGMENT_ENTRY],
        build.hot_reload,
    )
    .map_err(bucket_context(bucket))?;
    build_main_pipeline(device, vert_desc, &vert_fn, &frag_fn, build.sample_count)
        .map_err(bucket_context(bucket))
}

// Bucket `bucket`'s pre-pass pipeline, or `None` with a warning when it fails
// to build: the bucket then shades without contributing to the G-buffer.
fn build_bucket_prepass(
    device: &ProtocolObject<dyn MTLDevice>,
    bucket: usize,
    world: Option<&ShaderPrograms>,
    hot_reload: bool,
) -> Option<Pipeline> {
    try_build_bucket_prepass(device, world, hot_reload)
        .map_err(|e| {
            tracing::warn!("shader bucket {bucket}'s G-buffer pre-pass did not build: {e}")
        })
        .ok()
}

fn try_build_bucket_prepass(
    device: &ProtocolObject<dyn MTLDevice>,
    world: Option<&ShaderPrograms>,
    hot_reload: bool,
) -> RenderResult<Pipeline> {
    let (vert_fn, frag_fn) = bucket_stages(
        device,
        world,
        [&engine::MAIN_PREPASS_VERT, &engine::MAIN_PREPASS_FRAG],
        [WORLD_PREPASS_VERTEX_ENTRY, WORLD_PREPASS_FRAGMENT_ENTRY],
        hot_reload,
    )?;
    crate::metal::post::build_gbuffer_prepass_pipeline(device, &vert_fn, &frag_fn)
}

impl MtlContext {
    // What every bucket's pipelines are built under now.
    pub(super) fn bucket_build(&self) -> BucketBuild {
        BucketBuild {
            hot_reload: self.hot_reload.enabled,
            sample_count: self.targets.hdr.sample_count,
            prepass: self.gbuffer.targets.is_some(),
            template_generation: self.hot_reload.generation,
        }
    }

    // Bring every bucket's pre-pass in line with the G-buffer: build the
    // missing ones while it exists and drop them while it does not.
    pub(super) fn sync_prepass_pipelines(&mut self) {
        let build = self.bucket_build();
        let device = self.hw.device.clone();
        let world_default = self.world_shader.clone();
        if let Some(pipelines) = self.cull.main_pipeline.as_mut() {
            sync_prepass(&device, build, 0, pipelines, world_default.as_ref());
        }
        let buckets: Vec<usize> = self.cull.world_pipelines.resident_buckets().collect();
        for bucket in buckets {
            if let Some(pipelines) = self.cull.world_pipelines.get_mut(bucket) {
                let programs = pipelines.programs.clone();
                sync_prepass(&device, build, bucket, pipelines, programs.as_ref());
            }
        }
    }

    // Every resident material bucket's pipelines rebuilt from the current
    // templates. Fails as a whole when any one bucket's would drop a live
    // pre-pass or its main pipeline fails, so the caller keeps every live pair.
    pub(super) fn rebuild_world_buckets(
        &self,
        build: BucketBuild,
    ) -> RenderResult<Vec<(usize, BucketPipelines)>> {
        let mut rebuilt = Vec::new();
        for bucket in self.cull.world_pipelines.resident_buckets() {
            let Some(live) = self.cull.world_pipelines.get(bucket) else {
                continue;
            };
            let programs = live.programs.as_ref();
            let fresh = replacement(&self.hw.device, bucket, programs, build, Some(live), None)?;
            rebuilt.push((bucket, fresh));
        }
        Ok(rebuilt)
    }
}

// One bucket's half of `sync_prepass_pipelines`.
fn sync_prepass(
    device: &ProtocolObject<dyn MTLDevice>,
    build: BucketBuild,
    bucket: usize,
    pipelines: &mut BucketPipelines,
    world: Option<&ShaderPrograms>,
) {
    if !build.prepass {
        pipelines.prepass = None;
    } else if pipelines.prepass.is_none() {
        pipelines.prepass = build_bucket_prepass(device, bucket, world, build.hot_reload);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::ShaderSource;
    use concinnity_core::components::compiled_programs::CompiledProgram;
    use concinnity_core::platform::Platform;
    use concinnity_core::render::shader_programs::surface;
    use concinnity_core::render::shader_source;
    use objc2_metal::MTLCreateSystemDefaultDevice;

    const SHADE: &str =
        "float4 shade(VertexOut v, GpuObjectData od) { return shade_surface(v, od); }";
    const SWAY: &str = "VertexOut transform(float4x4 model, float3 pos, float3 normal, \
        float3 tangent, float3 color, float2 uv)\n{\n    return project_vertex(model, pos + \
        float3(0.1 * sin(VIEW.elapsed), 0.0, 0.0), normal, tangent, color, uv);\n}\n";

    // A world Shader with a vertex hook and no cooked artifacts, so every
    // entry compiles here from the current templates.
    fn world() -> ShaderPrograms {
        ShaderPrograms {
            name: "reeds".to_string(),
            vertex: Some(ShaderSource {
                path: "shaders/sway.hlsl".to_string(),
                text: SWAY.to_string(),
            }),
            fragment: ShaderSource {
                path: "shaders/lit.hlsl".to_string(),
                text: SHADE.to_string(),
            },
            programs: Vec::new(),
        }
    }

    // `world()` with a cooked pre-pass vertex entry that matches the template
    // but does not compile, so only the pre-pass fails to build.
    fn broken_prepass() -> ShaderPrograms {
        let mut programs = world();
        let program = surface::program(surface::PREPASS_VERTEX.entry).expect("a pre-pass entry");
        let source = surface::source(program, Platform::Metal, &programs.sources());
        programs.programs.push(CompiledProgram {
            entry: surface::PREPASS_VERTEX.entry.to_string(),
            source_digest: shader_source::source_digest(&source),
            artifact: b"not a shader".to_vec(),
        });
        programs
    }

    fn build(prepass: bool) -> BucketBuild {
        BucketBuild {
            hot_reload: false,
            sample_count: 1,
            prepass,
            template_generation: 0,
        }
    }

    fn pipeline_id(
        pipeline: &Option<Pipeline>,
    ) -> Option<*const ProtocolObject<dyn MTLRenderPipelineState>> {
        pipeline.as_ref().map(Retained::as_ptr)
    }

    // A fresh bucket whose pre-pass fails still shades. Rebuilt, it may
    // replace a live bucket that had no pre-pass either, but not one drawing
    // with a pre-pass, which the caller then keeps.
    #[test]
    fn a_failed_prepass_replaces_only_a_bucket_without_one() {
        concinnity_shader::require_dxc!();
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            return;
        };
        let broken = broken_prepass();
        let vert_desc = make_vertex_descriptor();
        let fresh = build_bucket_pipelines(&device, &vert_desc, 1, Some(&broken), build(true))
            .expect("the main pipeline builds");
        assert!(fresh.prepass.is_none());
        let healthy = build_bucket_pipelines(&device, &vert_desc, 1, Some(&world()), build(true))
            .expect("a world bucket builds");
        assert!(healthy.prepass.is_some());

        let rebuild = |live: &BucketPipelines, prepared| {
            replacement(&device, 1, Some(&broken), build(true), Some(live), prepared)
        };
        assert!(rebuild(&healthy, None).is_err());
        assert!(rebuild(&fresh, None).is_ok());
        // A prepared pair that would drop the live pre-pass is rebuilt here,
        // and refused when that build drops it too.
        assert!(rebuild(&healthy, Some(fresh)).is_err());
        let rebuilt = replacement(
            &device,
            1,
            Some(&world()),
            build(true),
            Some(&healthy),
            None,
        )
        .expect("a healthy rebuild replaces");
        assert!(rebuilt.prepass.is_some());
    }

    // Syncing drops every pre-pass without a G-buffer, builds a missing one
    // with it, and leaves a live one in place.
    #[test]
    fn syncing_follows_the_gbuffer_and_keeps_a_live_prepass() {
        concinnity_shader::require_dxc!();
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            return;
        };
        let world = world();
        let mut pipelines = build_bucket_pipelines(
            &device,
            &make_vertex_descriptor(),
            1,
            Some(&world),
            build(false),
        )
        .expect("a world bucket builds");
        assert!(pipelines.prepass.is_none());

        sync_prepass(&device, build(true), 1, &mut pipelines, Some(&world));
        let built = pipeline_id(&pipelines.prepass);
        assert!(built.is_some());

        sync_prepass(&device, build(true), 1, &mut pipelines, Some(&world));
        assert_eq!(pipeline_id(&pipelines.prepass), built);

        sync_prepass(&device, build(false), 1, &mut pipelines, Some(&world));
        assert!(pipelines.prepass.is_none());
    }

    // A world Shader's bucket builds both pipelines when a G-buffer consumer is
    // on and only the main one when none is, and keeps its programs either way.
    #[test]
    fn a_world_bucket_builds_its_prepass_only_with_a_gbuffer() {
        concinnity_shader::require_dxc!();
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            return;
        };
        let world = world();
        let with = build_bucket_pipelines(
            &device,
            &make_vertex_descriptor(),
            1,
            Some(&world),
            build(true),
        )
        .expect("a world bucket builds");
        assert!(with.prepass.is_some());
        assert_eq!(
            with.programs.as_ref().map(|p| p.name.as_str()),
            Some("reeds")
        );
        let without = build_bucket_pipelines(
            &device,
            &make_vertex_descriptor(),
            1,
            Some(&world),
            build(false),
        )
        .expect("a world bucket builds");
        assert!(without.prepass.is_none());
    }
}
