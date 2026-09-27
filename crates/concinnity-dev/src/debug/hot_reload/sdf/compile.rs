//! An SdfVolume's field as a save left it, the key a compile of it runs under,
//! and the compile a worker runs.

use concinnity_cook::compile::sdf_field::CompiledField;
use concinnity_core::components::ShaderSource;
use concinnity_engine::gfx::system::sdf_field_sources::SdfFieldEntry;

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

// Compile `field` exactly as `cn build` would for this host's backend, then do
// the device-free part of the pipeline build so the swap on the frame thread
// is short.
pub(in crate::debug::hot_reload) fn compile(
    name: &str,
    key: &FieldKey,
    field: &ShaderSource,
) -> FieldResult {
    // Inside the bounded job pool, so the compile's rayon fan-out over the
    // entries does not claim every core the frame loop also needs.
    let compiled = concinnity_host::thread::jobs::pool().install(|| {
        concinnity_cook::compile::sdf_field::compile_sdf_field(
            name,
            field.as_file(),
            crate::cook_platform(),
            key.volumetric,
            key.cast_shadows,
        )
    })?;
    // A field catalog is only captured in a dev-loop session, whose backend is
    // always built with hot reload on.
    if let Err(e) = concinnity_engine::warm_sdf_field(
        &compiled.programs,
        key.volumetric,
        key.cast_shadows,
        true,
    ) {
        tracing::warn!(
            "SdfVolume hot-reload: '{name}' could not be prepared off the frame thread ({e}); \
             the swap will build it"
        );
    }
    Ok(compiled)
}
