// Build-time precompilation of the engine's static Metal shaders.
//
// The device crate's build script hands this module its own `.metal` shader
// directory and the single-source Metal program table. Every `.metal` file is
// compiled to a `.metallib` through `concinnity_shader::metallib`, the same
// path a renderer's runtime compile takes, and a lookup function mapping shader
// name to embedded bytes is generated beside them.
//
// The single-source half never touches a directory: its programs and their text
// come from `concinnity_core::render`, which embeds every shader, so this
// compiles the same bytes whether the crate is built from the workspace or
// unpacked from a registry tarball with no sibling checkout beside it. Each
// program takes the dxc / spirv-cross route to MSL, and that MSL is kept for
// the build script's ABI check rather than translated a second time.
//
// Without the Metal compiler (Command Line Tools without a full Xcode) a build
// for this host fails, naming what is missing. A cross build warns instead and
// generates a lookup that returns `None` for every name.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use concinnity_core::platform::Platform;
use concinnity_core::render::shader_programs::{Table, Variant};
use concinnity_core::render::shader_source;
use concinnity_shader::metallib;

use crate::shader_artifacts::{embed, stub_lookup_source};

// The function the generated file defines.
const LOOKUP: &str = "embedded_metallib";

/// The MSL each single-source program compiled from, by artifact name.
#[derive(Default)]
pub struct EmittedMsl(BTreeMap<String, String>);

impl EmittedMsl {
    /// The MSL the variant filed under `name` compiled from.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&str> {
        self.0.get(name).map(String::as_str)
    }
}

/// Precompile every `.metal` under `shaders_dir`, plus every variant of the
/// single-source programs in `table`, into OUT_DIR and generate
/// `engine_metallibs.rs` there, looking each library up by its artifact name. Returns the MSL each
/// program compiled from, or `None` when the host has no Metal toolchain and a
/// cross build embedded nothing.
/// Panics if the Metal toolchain is present but a shader fails to compile: a
/// broken shader must fail the build, not surface at renderer init.
pub fn precompile_metal_shaders(shaders_dir: &Path, table: &Table) -> Option<EmittedMsl> {
    println!("cargo:rerun-if-changed={}", shaders_dir.display());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let generated = out_dir.join("engine_metallibs.rs");

    let shaders = metal_sources(shaders_dir);
    if !metallib::toolchain_present() {
        crate::embedded_shaders::cannot_embed("Metal compiler not found (full Xcode required)");
        std::fs::write(&generated, stub_lookup_source(LOOKUP)).expect("write engine_metallibs.rs");
        return None;
    }
    // dxc is demanded here rather than earlier because the route ends by handing
    // MSL to the Metal toolchain: a host without that toolchain embeds nothing
    // whether or not it has dxc, so a build that has already stubbed for its
    // absence must not also fail for a compiler that could not have helped.
    // That is what a cross build off macOS does, docs.rs included.
    let variants: Vec<Variant<'static>> = table.variants().collect();
    if !variants.is_empty() {
        crate::embedded_shaders::require_dxc();
    }

    let lib_dir = out_dir.join("engine_shaders");
    std::fs::create_dir_all(&lib_dir).expect("create engine_shaders dir");
    let jobs: Vec<Job> = shaders
        .iter()
        .map(|path| Job::hand_written(path))
        .chain(variants.iter().map(|v| Job::program(*v)))
        .collect();
    let msl = embed(
        &jobs,
        |job| {
            (
                job.name.clone(),
                shader_source::source_digest(&job.source),
                lib_path(&lib_dir, &job.name),
            )
        },
        |job| {
            job.compile(&lib_dir)
                .map_err(|e| format!("metallib precompile failed: {e}"))
        },
        &generated,
        LOOKUP,
    );

    let mut emitted = EmittedMsl::default();
    for (job, msl) in jobs.iter().zip(msl) {
        if job.program.is_some()
            && let Some(msl) = msl
        {
            emitted.0.insert(job.name.clone(), msl);
        }
    }
    Some(emitted)
}

// One library to build: a hand-written `.metal`, or a single-source program
// with its assembled HLSL. `name` is the lookup key and `source` the text its
// digest is taken over.
struct Job {
    name: String,
    source: String,
    program: Option<Variant<'static>>,
}

impl Job {
    fn hand_written(path: &Path) -> Self {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .expect("utf8 shader filename")
            .to_string();
        let source = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        Job {
            name,
            source,
            program: None,
        }
    }

    fn program(variant: Variant<'static>) -> Self {
        let file = variant.program.file;
        // Assembly answers an unknown name with empty text, which would reach
        // the compiler as a missing entry point rather than a typo.
        assert!(
            concinnity_core::render::shaders::embedded(file).is_some(),
            "no embedded shader named {file}"
        );
        Job {
            name: variant.artifact_name(),
            source: variant.assemble(Platform::Metal),
            program: Some(variant),
        }
    }

    // The library's bytes, and for a single-source program the MSL it was
    // built from.
    fn compile(&self, work_dir: &Path) -> Result<(Vec<u8>, Option<String>), String> {
        let Some(program) = self.program else {
            let lib = metallib::compile(&self.source, work_dir)
                .map_err(|e| format!("{}: {e}", self.name))?;
            return Ok((lib, None));
        };
        let msl = compile_msl(&self.name, program, &self.source, work_dir)?;
        let lib = metallib::compile(&msl, work_dir).map_err(|e| format!("{}: {e}", self.name))?;
        Ok((lib, Some(msl)))
    }
}

// The MSL leg links spirv-cross, which this crate's manifest enables for a
// macOS host alone; that is also the only host with the Metal toolchain this is
// gated on.
fn compile_msl(
    name: &str,
    variant: Variant<'_>,
    source: &str,
    work_dir: &Path,
) -> Result<String, String> {
    let job = concinnity_shader::HlslJob {
        source,
        file_name: variant.program.file,
        entry: variant.program.entry,
        target: concinnity_shader::HlslTarget::Msl,
    };
    let bytes = concinnity_shader::compile(&job, work_dir)?;
    String::from_utf8(bytes).map_err(|e| format!("{name}: MSL is not UTF-8: {e}"))
}

// Where one shader's metallib lands. The registered name keeps its extension
// rather than trading it for `.metallib`: a `.metal` shader and a single-source
// library may share a stem, and replacing the extension would give the two one
// path, so the second compiled would silently serve both.
fn lib_path(lib_dir: &Path, name: &str) -> PathBuf {
    lib_dir.join(format!("{name}.metallib"))
}

// Every `.metal` in the directory, sorted for a deterministic generated file.
fn metal_sources(shaders_dir: &Path) -> Vec<PathBuf> {
    let mut shaders: Vec<PathBuf> = std::fs::read_dir(shaders_dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", shaders_dir.display()))
        .filter_map(|entry| Some(entry.ok()?.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "metal"))
        .collect();
    shaders.sort();
    shaders
}

#[cfg(test)]
mod tests {
    use super::*;

    // One stem in two source kinds must not share an artifact: the second
    // compiled would overwrite the first and the lookup would serve one
    // library under both names, which is a pipeline that fails to link.
    #[test]
    fn two_sources_with_one_stem_get_separate_artifacts() {
        let dir = Path::new("/out");
        assert_ne!(
            lib_path(dir, "fullscreen_vert.metal"),
            lib_path(dir, "fullscreen_vert.hlsl")
        );
        assert_eq!(
            lib_path(dir, "fullscreen_vert.hlsl"),
            Path::new("/out/fullscreen_vert.hlsl.metallib")
        );
    }

    #[test]
    fn metal_sources_takes_only_metal_files_in_order() {
        let tree = concinnity_testing::TempTree::new();
        for name in ["b.metal", "a.metal", "notes.txt"] {
            tree.write(name, "");
        }
        let shaders = metal_sources(tree.path());
        let names: Vec<_> = shaders
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names, ["a.metal", "b.metal"]);
    }

    // A hand-written shader is looked up by its file name and digested over
    // the text on disk, which is what the renderer compares against.
    #[test]
    fn a_hand_written_job_is_keyed_by_file_name_over_its_text() {
        let tree = concinnity_testing::TempTree::new();
        tree.write("k.metal", "kernel void k() {}\n");
        let job = Job::hand_written(&tree.path().join("k.metal"));
        assert_eq!(job.name, "k.metal");
        assert_eq!(job.source, "kernel void k() {}\n");
        assert!(job.program.is_none());
    }

    // A program is looked up by its artifact name and digested over the text
    // the renderer assembles for it, so an unedited shader is a content hit.
    #[test]
    fn a_program_job_is_keyed_by_artifact_name_over_the_assembled_source() {
        use concinnity_core::render::shader_programs::shared;
        let variant = shared::FOG_FRAG.at(false);
        let job = Job::program(variant);
        assert_eq!(job.name, "fog_frag.hlsl");
        assert_eq!(
            job.source,
            shader_source::assemble("fog.hlsl", Platform::Metal, &[("USE_MSAA", "0")])
        );
    }

    #[test]
    fn emitted_msl_answers_by_label() {
        let mut emitted = EmittedMsl::default();
        emitted.0.insert("a.hlsl".to_string(), "msl".to_string());
        assert_eq!(emitted.get("a.hlsl"), Some("msl"));
        assert_eq!(emitted.get("b.hlsl"), None);
    }
}
