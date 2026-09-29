//! An SdfVolume's field as a save left it, the key a compile of it runs under,
//! and the compile and pipeline builds a worker runs.

use concinnity_cook::compile::sdf_field::CompiledField;
use concinnity_core::components::ShaderSource;
use concinnity_core::render::backend::{PipelineBuilder, PreparedPipelines};
use concinnity_core::render::shader_programs::raymarch::VolumeFlags;
use concinnity_engine::gfx::system::sdf_field_sources::SdfFieldEntry;
use std::collections::HashMap;

use crate::debug::hot_reload::report::ReloadFailure;

pub(in crate::debug::hot_reload) type FieldResult = Result<CompiledField, ReloadFailure>;

// One compile: a field file and the flags that decide which entries it
// compiles. Every volume with the same key shares one compile.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::debug::hot_reload) struct FieldKey {
    pub path: String,
    pub volumetric: bool,
    pub cast_shadows: bool,
}

impl FieldKey {
    pub(in crate::debug::hot_reload) fn of(entry: &SdfFieldEntry) -> Self {
        Self {
            path: entry.resolved_path.clone(),
            volumetric: entry.volumetric,
            cast_shadows: entry.cast_shadows,
        }
    }

    pub(in crate::debug::hot_reload) fn flags(&self) -> VolumeFlags {
        VolumeFlags {
            volumetric: self.volumetric,
            cast_shadows: self.cast_shadows,
        }
    }
}

// Read the field at `path`, under that path, which is the path its diagnostics
// then name. The error names the file.
pub(in crate::debug::hot_reload) fn read_field(path: &str) -> Result<ShaderSource, String> {
    concinnity_cook::compile::shader::read_shader_source(path)
        .map(|text| ShaderSource {
            path: path.to_string(),
            text,
        })
        .map_err(|e| e.to_string())
}

// Compile `field` exactly as `cn build` would for this host's backend.
pub(in crate::debug::hot_reload) fn compile(
    name: &str,
    key: &FieldKey,
    field: &ShaderSource,
) -> FieldResult {
    // Inside the bounded job pool, so the compile's rayon fan-out over the
    // entries does not claim every core the frame loop also needs.
    Ok(concinnity_host::thread::jobs::pool().install(|| {
        concinnity_cook::compile::sdf_field::compile_sdf_field(
            name,
            field.as_file(),
            crate::cook_platform(),
            key.volumetric,
            key.cast_shadows,
        )
    })?)
}

// A compile, with the pipelines the backend's builder made from it for each
// volume of the group, by the volume's position in the backend.
pub(in crate::debug::hot_reload) struct Rebuilt {
    pub compiled: CompiledField,
    pub prepared: HashMap<usize, PreparedPipelines>,
}

pub(in crate::debug::hot_reload) type RebuildResult = Result<Rebuilt, ReloadFailure>;

// Build every pipeline each of `volumes` draws with from `compiled` through
// `builder`, so the swap on the frame thread does not. Each volume gets its
// own, since a swap takes its pipelines by value. Without a builder the swap
// builds them.
pub(in crate::debug::hot_reload) fn prepare(
    compiled: CompiledField,
    builder: Option<&dyn PipelineBuilder>,
    key: &FieldKey,
    volumes: &[(usize, String)],
) -> RebuildResult {
    let mut prepared = HashMap::new();
    if let Some(builder) = builder {
        for (volume, name) in volumes {
            let pipelines = builder
                .sdf_volume(&compiled.programs, key.flags(), name)
                .map_err(|e| ReloadFailure::Rejected(e.to_string()))?;
            prepared.insert(*volume, pipelines);
        }
    }
    Ok(Rebuilt { compiled, prepared })
}
