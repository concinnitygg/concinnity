use super::compile::{CompileQueue, Finished, Generations};
use super::*;
use crate::debug::hot_reload::pending::PendingShaders;
use concinnity_cook::compile::program::{CompileFailure, EntryFailure, Severity};
use concinnity_cook::compile::shader::CompiledShader;
use concinnity_core::components::ShaderSource;
use concinnity_core::components::ShaderStage;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_engine::gfx::system::shader_sources::ShaderFile;
use std::collections::HashSet;
use std::path::Path;
use std::sync::mpsc::channel;
use std::time::{Duration, Instant};

// A backend that records every Shader update, reports the buckets in
// `resident` as installed, and rejects every build when `reject` is set.
#[derive(Default)]
struct ShaderBackend {
    resident: HashSet<u32>,
    reject: bool,
    updates: Vec<(u32, String)>,
}

impl LiveEdit for ShaderBackend {
    fn update_world_shader(
        &mut self,
        bucket: u32,
        programs: &ShaderPrograms,
    ) -> RenderResult<WorldShaderSwap> {
        self.updates.push((bucket, programs.fragment.text.clone()));
        if self.reject {
            return Err(RenderError::ShaderCompile("pipeline build failed".into()));
        }
        Ok(if bucket == 0 || self.resident.contains(&bucket) {
            WorldShaderSwap::Swapped
        } else {
            WorldShaderSwap::NotResident
        })
    }
}

// Stands in for dxc: a fragment containing "error" fails with an error on its
// first line, anything else "compiles" to programs carrying the texts.
fn fake_compiler() -> Compiler {
    Arc::new(|name: &str, texts: &ShaderTexts| {
        if texts.fragment.text.contains("error") {
            let output = format!("{}:1:1: error: syntax error\n", texts.fragment.path);
            return Err(ShaderReloadFailure::Compile(CompileFailure {
                owner: format!("Shader '{name}'"),
                diagnostics: fake_diagnostics(&output),
                failures: vec![EntryFailure {
                    entry: "fragment_main".to_string(),
                    output,
                }],
                hint: "",
            }));
        }
        Ok(CompiledShader {
            programs: ShaderPrograms {
                name: name.to_string(),
                vertex: texts.vertex.clone(),
                fragment: texts.fragment.clone(),
                programs: Vec::new(),
            },
            warnings: Vec::new(),
        })
    })
}

// The one diagnostic the fake compiler reports.
fn fake_diagnostics(output: &str) -> Vec<Diagnostic> {
    let (path, message) = output.split_once(":1:1: error: ").unwrap();
    vec![Diagnostic {
        path: path.to_string(),
        line: 1,
        column: 1,
        severity: Severity::Error,
        message: message.trim_end().to_string(),
        context: String::new(),
    }]
}

fn entry(id: u32, bucket: u32, files: &[(ShaderStage, &Path)]) -> ShaderSourceEntry {
    ShaderSourceEntry {
        id: AssetId(id),
        name: format!("shader{id}"),
        bucket,
        files: files
            .iter()
            .map(|&(stage, path)| ShaderFile {
                stage,
                resolved_path: path.to_string_lossy().into_owned(),
            })
            .collect(),
    }
}

fn pending(ids: &[u32]) -> PendingShaders {
    PendingShaders {
        all: false,
        ids: ids.iter().map(|&id| AssetId(id)).collect(),
    }
}

// Poll until `count` reports arrived; bounded so a lost result fails the test
// instead of hanging it.
fn poll_for(
    reload: &mut ShaderReload,
    backend: &mut ShaderBackend,
    count: usize,
) -> Vec<ShaderReloadReport> {
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut reports = Vec::new();
    while reports.len() < count && Instant::now() < deadline {
        reports.extend(reload.poll(backend));
        std::thread::sleep(Duration::from_millis(5));
    }
    reports
}

// Three Shaders: the world default, a Material-named one, and one owned by an
// unloaded scene, the last two sharing a vertex file.
struct Fixture {
    dir: tempfile::TempDir,
    reload: ShaderReload,
    overrides: ShaderOverrides,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        for (name, text) in [
            ("lit.hlsl", "lit v1"),
            ("water.hlsl", "water v1"),
            ("cave.hlsl", "cave v1"),
            ("sway.hlsl", "sway v1"),
        ] {
            std::fs::write(dir.path().join(name), text).unwrap();
        }
        let p = |name: &str| dir.path().join(name);
        let catalog = ShaderSourceMap {
            entries: vec![
                entry(1, 0, &[(ShaderStage::Fragment, &p("lit.hlsl"))]),
                entry(
                    2,
                    1,
                    &[
                        (ShaderStage::Vertex, &p("sway.hlsl")),
                        (ShaderStage::Fragment, &p("water.hlsl")),
                    ],
                ),
                entry(
                    3,
                    2,
                    &[
                        (ShaderStage::Vertex, &p("sway.hlsl")),
                        (ShaderStage::Fragment, &p("cave.hlsl")),
                    ],
                ),
            ],
        };
        let overrides = ShaderOverrides::default();
        let reload = ShaderReload::with_compiler(catalog, overrides.clone(), fake_compiler());
        Self {
            dir,
            reload,
            overrides,
        }
    }

    fn write(&self, name: &str, text: &str) {
        std::fs::write(self.dir.path().join(name), text).unwrap();
    }
}

// Each report as its Shader's name and what became of it, frame time aside.
fn outcomes(reports: &[ShaderReloadReport]) -> Vec<(&str, &'static str)> {
    let mut out: Vec<(&str, &'static str)> = reports
        .iter()
        .map(|r| {
            let kind = match r.outcome {
                ShaderReloadOutcome::Swapped { .. } => "swapped",
                ShaderReloadOutcome::AppliesOnLoad { .. } => "applies on load",
                ShaderReloadOutcome::Failed(_) => "failed",
            };
            (r.name.as_str(), kind)
        })
        .collect();
    out.sort_unstable();
    out
}

// Only the Shader a request names is recompiled, and a resident non-default
// bucket swaps its own pipeline with the edited text.
#[test]
fn a_request_rebuilds_just_the_named_shader() {
    let mut f = Fixture::new();
    let mut backend = ShaderBackend {
        resident: [1].into(),
        ..Default::default()
    };
    f.write("water.hlsl", "water v2");
    assert!(f.reload.request(&pending(&[2])).is_empty());
    let reports = poll_for(&mut f.reload, &mut backend, 1);
    assert_eq!(outcomes(&reports), [("shader2", "swapped")]);
    assert_eq!(backend.updates, [(1, "water v2".to_string())]);
    // The edit is also kept for the next install of the bucket.
    assert_eq!(f.overrides.get(1).unwrap().fragment.text, "water v2");
}

// The world default rebuilds through bucket 0 and needs no override, since
// nothing ever reinstalls it.
#[test]
fn the_world_default_swaps_without_an_override() {
    let mut f = Fixture::new();
    let mut backend = ShaderBackend::default();
    f.write("lit.hlsl", "lit v2");
    f.reload.request(&pending(&[1]));
    let reports = poll_for(&mut f.reload, &mut backend, 1);
    assert_eq!(outcomes(&reports), [("shader1", "swapped")]);
    assert_eq!(backend.updates, [(0, "lit v2".to_string())]);
    assert!(f.overrides.get(0).is_none());
}

// A Shader whose scene is unloaded builds nothing now; its programs wait as
// the override the scene's load installs.
#[test]
fn a_non_resident_shader_is_kept_for_its_scene_load() {
    let mut f = Fixture::new();
    let mut backend = ShaderBackend::default();
    f.write("cave.hlsl", "cave v2");
    f.reload.request(&pending(&[3]));
    let reports = poll_for(&mut f.reload, &mut backend, 1);
    assert_eq!(outcomes(&reports), [("shader3", "applies on load")]);
    assert_eq!(f.overrides.get(2).unwrap().fragment.text, "cave v2");
}

// Saving a file two Shaders share, which the watcher turns into a request for
// both, rebuilds each of them.
#[test]
fn a_shared_file_reloads_every_shader_reading_it() {
    let mut f = Fixture::new();
    let mut backend = ShaderBackend {
        resident: [1, 2].into(),
        ..Default::default()
    };
    f.write("sway.hlsl", "sway v2");
    f.reload.request(&pending(&[2, 3]));
    let reports = poll_for(&mut f.reload, &mut backend, 2);
    assert_eq!(
        outcomes(&reports),
        [("shader2", "swapped"), ("shader3", "swapped")]
    );
    let mut buckets: Vec<u32> = backend.updates.iter().map(|&(b, _)| b).collect();
    buckets.sort_unstable();
    assert_eq!(buckets, [1, 2]);
    for bucket in [1, 2] {
        let programs = f.overrides.get(bucket).unwrap();
        assert_eq!(
            programs.vertex.as_ref().map(|v| v.text.as_str()),
            Some("sway v2")
        );
    }
}

// `reload-assets` asks for every Shader.
#[test]
fn a_request_for_all_reaches_every_shader() {
    let mut f = Fixture::new();
    let mut backend = ShaderBackend::default();
    f.reload.request(&PendingShaders {
        all: true,
        ids: Default::default(),
    });
    assert_eq!(poll_for(&mut f.reload, &mut backend, 3).len(), 3);
}

// A compile error reaches neither the backend nor the override: the live
// pipeline and any earlier edit stay as they were.
#[test]
fn a_compile_error_leaves_the_pipeline_and_the_override_alone() {
    let mut f = Fixture::new();
    let mut backend = ShaderBackend {
        resident: [1].into(),
        ..Default::default()
    };
    f.write("water.hlsl", "water v2");
    f.reload.request(&pending(&[2]));
    poll_for(&mut f.reload, &mut backend, 1);
    f.write("water.hlsl", "water error");
    f.reload.request(&pending(&[2]));
    let reports = poll_for(&mut f.reload, &mut backend, 1);
    let [
        ShaderReloadReport {
            outcome: ShaderReloadOutcome::Failed(ShaderReloadFailure::Compile(failed)),
            ..
        },
    ] = &reports[..]
    else {
        panic!("one compile failure: {reports:?}");
    };
    // The error names the file by the catalog's resolved path.
    let water = f.dir.path().join("water.hlsl");
    assert_eq!(failed.diagnostics[0].path, water.to_string_lossy());
    assert_eq!(backend.updates.len(), 1);
    assert_eq!(f.overrides.get(1).unwrap().fragment.text, "water v2");
}

// A pipeline the backend refuses to build is reported and not kept.
#[test]
fn a_rejected_pipeline_is_not_kept_as_an_override() {
    let mut f = Fixture::new();
    let mut backend = ShaderBackend {
        resident: [1].into(),
        reject: true,
        ..Default::default()
    };
    f.reload.request(&pending(&[2]));
    let reports = poll_for(&mut f.reload, &mut backend, 1);
    assert!(matches!(
        reports[0].outcome,
        ShaderReloadOutcome::Failed(ShaderReloadFailure::Rejected(ref e)) if e.contains("pipeline build failed")
    ));
    let ShaderReloadOutcome::Failed(e) = &reports[0].outcome else {
        unreachable!()
    };
    assert!(
        e.to_string().starts_with("pipeline rebuild rejected: "),
        "{e}"
    );
    assert!(f.overrides.get(1).is_none());
}

// A file that cannot be read fails the request at once, with no compile.
#[test]
fn an_unreadable_file_fails_before_compiling() {
    let mut f = Fixture::new();
    let mut backend = ShaderBackend::default();
    std::fs::remove_file(f.dir.path().join("water.hlsl")).unwrap();
    let reports = f.reload.request(&pending(&[2]));
    assert!(matches!(
        &reports[..],
        [ShaderReloadReport { name, outcome: ShaderReloadOutcome::Failed(ShaderReloadFailure::Unstarted(e)) }]
            if name == "shader2" && e.contains("water.hlsl")
    ));
    std::thread::sleep(Duration::from_millis(50));
    assert!(f.reload.poll(&mut backend).is_empty());
    assert!(backend.updates.is_empty());
}

// An empty catalog makes every request and poll a no-op.
#[test]
fn an_empty_catalog_reloads_nothing() {
    let mut reload = ShaderReload::with_compiler(
        ShaderSourceMap::default(),
        ShaderOverrides::default(),
        fake_compiler(),
    );
    let mut backend = ShaderBackend::default();
    assert!(
        reload
            .request(&PendingShaders {
                all: true,
                ids: Default::default()
            })
            .is_empty()
    );
    assert!(reload.poll(&mut backend).is_empty());
    assert!(backend.updates.is_empty());
}

fn compiled(programs: ShaderPrograms) -> CompiledShader {
    CompiledShader {
        programs,
        warnings: Vec::new(),
    }
}

fn finished(id: u32, generation: u64) -> Finished {
    Finished {
        id: AssetId(id),
        generation,
        result: Ok(compiled(ShaderPrograms::default())),
    }
}

// Whatever order two compiles of one Shader finish in, only the newer applies.
#[test]
fn only_a_shaders_newest_request_is_accepted() {
    let mut generations = Generations::default();
    let older = generations.begin(AssetId(1));
    let newer = generations.begin(AssetId(1));
    assert!(generations.accept(finished(1, older)).is_none());
    assert!(generations.accept(finished(1, newer)).is_some());
    // A late older result after the newer one applied is still stale.
    assert!(generations.accept(finished(1, older)).is_none());
}

// Requests for different Shaders never supersede each other.
#[test]
fn requests_for_different_shaders_are_independent() {
    let mut generations = Generations::default();
    let water = generations.begin(AssetId(1));
    let cave = generations.begin(AssetId(2));
    assert!(generations.accept(finished(1, water)).is_some());
    assert!(generations.accept(finished(2, cave)).is_some());
    assert!(generations.accept(finished(3, cave)).is_none());
}

// On real workers: an older compile that finishes after a newer one is
// dropped instead of overwriting it.
#[test]
fn a_stale_compile_finishing_late_is_dropped() {
    let mut queue = CompileQueue::new();
    let (release_old, old_gate) = channel::<()>();
    let (old_done, old_finished) = channel::<()>();
    let program = |fragment: &str| {
        compiled(ShaderPrograms {
            fragment: ShaderSource {
                path: "f.hlsl".to_string(),
                text: fragment.to_string(),
            },
            ..Default::default()
        })
    };
    let old = program("old");
    queue
        .submit(AssetId(7), move || {
            let _ = old_gate.recv();
            let _ = old_done.send(());
            Ok(old)
        })
        .unwrap();
    let new = program("new");
    queue.submit(AssetId(7), move || Ok(new)).unwrap();

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut applied = Vec::new();
    while applied.is_empty() && Instant::now() < deadline {
        applied = queue.drain();
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(applied.len(), 1);
    assert_eq!(applied[0].1.as_ref().unwrap().programs.fragment.text, "new");

    release_old.send(()).unwrap();
    old_finished
        .recv_timeout(Duration::from_secs(10))
        .expect("the old compile ran");
    std::thread::sleep(Duration::from_millis(50));
    assert!(queue.drain().is_empty());
}

// With the real compiler: a broken file read through the catalog fails with
// its error at the line it is on, under the path the catalog resolved, so a
// caller finds the entry's file by comparing paths.
#[test]
fn a_real_compile_error_names_the_catalogs_path_and_line() {
    if !concinnity_shader::dxc_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("broken.hlsl");
    std::fs::write(
        &path,
        "float4 shade(VertexOut v, GpuObjectData od)\n{\n    return missing_tint;\n}\n",
    )
    .unwrap();
    let catalog_entry = entry(9, 0, &[(ShaderStage::Fragment, &path)]);
    let texts = ShaderTexts::read(&catalog_entry).unwrap();
    let Err(ShaderReloadFailure::Compile(failed)) = compile::compile("broken", &texts) else {
        panic!("a compile failure");
    };
    let errors: Vec<(&str, u32)> = failed.errors().map(|d| (d.path.as_str(), d.line)).collect();
    assert_eq!(
        errors,
        [(catalog_entry.path(ShaderStage::Fragment).unwrap(), 3)]
    );
    let failure = ShaderReloadFailure::Compile(failed);
    assert_eq!(failure.first_error_at().as_deref(), Some("broken.hlsl:3"));
}
