//! A Shader's files as a save left them, and the compile a worker runs on them.

use concinnity_cook::compile::shader::CompiledShader;
use concinnity_core::components::{ShaderSource, ShaderStage};
use concinnity_core::render::backend::{PipelineBuilder, PreparedPipelines};
use concinnity_core::render::shader_programs::surface::Sources;
use concinnity_engine::live_edit::shader_sources::ShaderSourceEntry;

use crate::debug::hot_reload::report::ReloadFailure;

pub(in crate::debug::hot_reload) type CompileResult = Result<CompiledShader, ReloadFailure>;

// A Shader's files as read at the moment of the save, each under the resolved
// path it was read from, which is the path its diagnostics then name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::debug::hot_reload) struct ShaderTexts {
    pub vertex: Option<ShaderSource>,
    pub fragment: ShaderSource,
}

impl ShaderTexts {
    // Read `entry`'s files. The error names the file that could not be read.
    pub(in crate::debug::hot_reload) fn read(entry: &ShaderSourceEntry) -> Result<Self, String> {
        let read = |stage| {
            entry
                .path(stage)
                .map(|path| {
                    concinnity_cook::compile::shader::read_shader_source(path)
                        .map(|text| ShaderSource {
                            path: path.to_string(),
                            text,
                        })
                        .map_err(|e| e.to_string())
                })
                .transpose()
        };
        Ok(Self {
            vertex: read(ShaderStage::Vertex)?,
            fragment: read(ShaderStage::Fragment)?
                .ok_or_else(|| "the Shader declares no fragment file".to_string())?,
        })
    }

    pub(in crate::debug::hot_reload) fn sources(&self) -> Sources<'_> {
        Sources {
            vertex: self.vertex.as_ref().map(ShaderSource::as_file),
            fragment: self.fragment.as_file(),
        }
    }
}

// Compile `texts` exactly as `cn build` would for this host's backend.
pub(in crate::debug::hot_reload) fn compile(name: &str, texts: &ShaderTexts) -> CompileResult {
    // Inside the bounded job pool, so the compile's rayon fan-out over the
    // programs does not claim every core the frame loop also needs.
    Ok(concinnity_host::thread::jobs::pool().install(|| {
        concinnity_cook::compile::shader::compile_world_shader(
            name,
            &texts.sources(),
            crate::cook_platform(),
        )
    })?)
}

// A compile, with the pipeline the backend's builder made from it.
pub(in crate::debug::hot_reload) struct Rebuilt {
    pub compiled: CompiledShader,
    pub prepared: Option<PreparedPipelines>,
}

pub(in crate::debug::hot_reload) type RebuildResult = Result<Rebuilt, ReloadFailure>;

// Build bucket `bucket`'s pipeline from `compiled` through `builder`, so the
// swap on the frame thread does not. Without a builder the swap builds it.
pub(in crate::debug::hot_reload) fn prepare(
    compiled: CompiledShader,
    builder: Option<&dyn PipelineBuilder>,
    bucket: u32,
) -> RebuildResult {
    let prepared = builder
        .map(|b| b.world_shader(bucket, &compiled.programs))
        .transpose()
        .map_err(|e| ReloadFailure::Rejected(e.to_string()))?;
    Ok(Rebuilt { compiled, prepared })
}
