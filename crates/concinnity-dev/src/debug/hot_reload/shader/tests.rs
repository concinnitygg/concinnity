use super::*;
use crate::debug::hot_reload::signals::PendingShaders;
use concinnity_cook::compile::program::{CompileFailure, Diagnostic, EntryFailure, Severity};
use concinnity_cook::compile::shader::CompiledShader;
use concinnity_core::components::sdf_programs::SdfPrograms;
use concinnity_core::components::{ShaderPrograms, ShaderStage};
use concinnity_core::render::backend::PreparedPipelines;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::scene_state::SceneState;
use concinnity_core::render::shader_programs::raymarch::VolumeFlags;
use concinnity_engine::live_edit::shader_sources::ShaderFile;
use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, Instant};

// A backend that records every Shader update with the pipeline it was handed,
// reports the buckets in `resident` as installed, and rejects every build when
// `reject` is set.
#[derive(Default)]
struct ShaderBackend {
    resident: HashSet<u32>,
    reject: bool,
    updates: Vec<(u32, String)>,
    prepared: Vec<Option<FakePipeline>>,
}

impl concinnity_core::render::backend::SceneHost for ShaderBackend {
    fn scene(&self) -> Option<&SceneState> {
        None
    }
    fn scene_mut(&mut self) -> Option<&mut SceneState> {
        None
    }
    fn edit_geometry(
        &mut self,
        _: concinnity_core::render::backend::GeometryEdit<'_>,
    ) -> Option<RenderResult<()>> {
        None
    }
}

impl LiveEdit for ShaderBackend {
    fn update_world_shader(
        &mut self,
        bucket: u32,
        programs: &ShaderPrograms,
        prepared: Option<PreparedPipelines>,
    ) -> RenderResult<PipelineSwap> {
        self.updates.push((bucket, programs.fragment.text.clone()));
        self.prepared
            .push(prepared.and_then(PreparedPipelines::downcast::<FakePipeline>));
        if self.reject {
            return Err(RenderError::ShaderCompile("pipeline build failed".into()));
        }
        Ok(if bucket == 0 || self.resident.contains(&bucket) {
            PipelineSwap::Swapped
        } else {
            PipelineSwap::NotResident
        })
    }
}

// What the fake builder builds: the bucket and fragment text it was built from.
#[derive(Debug, PartialEq, Eq)]
struct FakePipeline(u32, String);

// Stands in for a backend's builder, refusing every build when `reject` is set.
struct FakeBuilder {
    reject: bool,
}

impl PipelineBuilder for FakeBuilder {
    fn world_shader(
        &self,
        bucket: u32,
        programs: &ShaderPrograms,
    ) -> RenderResult<PreparedPipelines> {
        if self.reject {
            return Err(RenderError::ShaderCompile("pipeline state failed".into()));
        }
        Ok(PreparedPipelines::new(FakePipeline(
            bucket,
            programs.fragment.text.clone(),
        )))
    }

    fn sdf_volume(
        &self,
        _: &SdfPrograms,
        _: VolumeFlags,
        _: &str,
    ) -> RenderResult<PreparedPipelines> {
        unreachable!("a Shader reload builds no volume")
    }
}

fn builder(reject: bool) -> Option<Arc<dyn PipelineBuilder>> {
    Some(Arc::new(FakeBuilder { reject }))
}

// Stands in for dxc: a fragment containing "error" fails with an error on its
// first line, anything else "compiles" to programs carrying the texts.
fn fake_compiler() -> Compiler {
    Arc::new(|name: &str, texts: &ShaderTexts| {
        if texts.fragment.text.contains("error") {
            let output = format!("{}:1:1: error: syntax error\n", texts.fragment.path);
            return Err(ReloadFailure::Compile(CompileFailure {
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
) -> Vec<ReloadReport> {
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
fn outcomes(reports: &[ReloadReport]) -> Vec<(&str, &'static str)> {
    let mut out: Vec<(&str, &'static str)> = reports
        .iter()
        .map(|r| {
            let kind = match r.outcome {
                ReloadOutcome::Swapped { .. } => "swapped",
                ReloadOutcome::AppliesOnLoad { .. } => "applies on load",
                ReloadOutcome::Failed(_) => "failed",
            };
            assert_eq!(
                r.subject.kind,
                crate::debug::hot_reload::report::SubjectKind::Shader
            );
            (r.subject.name.as_str(), kind)
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
    assert!(f.reload.request(&pending(&[2]), None).is_empty());
    let reports = poll_for(&mut f.reload, &mut backend, 1);
    assert_eq!(outcomes(&reports), [("shader2", "swapped")]);
    assert_eq!(backend.updates, [(1, "water v2".to_string())]);
    // Without a builder the swap is handed nothing and builds for itself.
    assert_eq!(backend.prepared, [None]);
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
    f.reload.request(&pending(&[1]), None);
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
    f.reload.request(&pending(&[3]), None);
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
    f.reload.request(&pending(&[2, 3]), None);
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
    f.reload.request(
        &PendingShaders {
            all: true,
            ids: Default::default(),
        },
        None,
    );
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
    f.reload.request(&pending(&[2]), None);
    poll_for(&mut f.reload, &mut backend, 1);
    f.write("water.hlsl", "water error");
    f.reload.request(&pending(&[2]), None);
    let reports = poll_for(&mut f.reload, &mut backend, 1);
    let [
        ReloadReport {
            outcome: ReloadOutcome::Failed(ReloadFailure::Compile(failed)),
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
    f.reload.request(&pending(&[2]), None);
    let reports = poll_for(&mut f.reload, &mut backend, 1);
    assert!(matches!(
        reports[0].outcome,
        ReloadOutcome::Failed(ReloadFailure::Rejected(ref e)) if e.contains("pipeline build failed")
    ));
    let ReloadOutcome::Failed(e) = &reports[0].outcome else {
        unreachable!()
    };
    assert!(
        e.to_string().starts_with("pipeline rebuild rejected: "),
        "{e}"
    );
    assert!(f.overrides.get(1).is_none());
}

// With a builder, the worker builds each Shader's pipeline from the compile it
// just ran, for the Shader's own bucket, and the swap is handed that pipeline.
#[test]
fn a_worker_built_pipeline_reaches_the_swap() {
    let mut f = Fixture::new();
    let mut backend = ShaderBackend {
        resident: [1].into(),
        ..Default::default()
    };
    f.write("lit.hlsl", "lit v2");
    f.write("water.hlsl", "water v2");
    f.reload.request(&pending(&[1, 2]), builder(false));
    let reports = poll_for(&mut f.reload, &mut backend, 2);
    assert_eq!(
        outcomes(&reports),
        [("shader1", "swapped"), ("shader2", "swapped")]
    );
    let mut prepared: Vec<FakePipeline> = backend.prepared.into_iter().flatten().collect();
    prepared.sort_by_key(|p| p.0);
    assert_eq!(
        prepared,
        [
            FakePipeline(0, "lit v2".to_string()),
            FakePipeline(1, "water v2".to_string())
        ]
    );
}

// A pipeline the builder refuses fails the reload on the worker: nothing
// reaches the backend and the edit is not kept for a later install.
#[test]
fn a_pipeline_the_builder_refuses_is_never_swapped() {
    let mut f = Fixture::new();
    let mut backend = ShaderBackend {
        resident: [1].into(),
        ..Default::default()
    };
    f.reload.request(&pending(&[2]), builder(true));
    let reports = poll_for(&mut f.reload, &mut backend, 1);
    assert!(matches!(
        &reports[0].outcome,
        ReloadOutcome::Failed(ReloadFailure::Rejected(e)) if e.contains("pipeline state failed")
    ));
    assert!(backend.updates.is_empty());
    assert!(f.overrides.get(1).is_none());
}

// A file that cannot be read fails the request at once, with no compile.
#[test]
fn an_unreadable_file_fails_before_compiling() {
    let mut f = Fixture::new();
    let mut backend = ShaderBackend::default();
    std::fs::remove_file(f.dir.path().join("water.hlsl")).unwrap();
    let reports = f.reload.request(&pending(&[2]), None);
    assert!(matches!(
        &reports[..],
        [ReloadReport { subject, outcome: ReloadOutcome::Failed(ReloadFailure::Unstarted(e)) }]
            if subject.name == "shader2" && e.contains("water.hlsl")
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
            .request(
                &PendingShaders {
                    all: true,
                    ids: Default::default()
                },
                None
            )
            .is_empty()
    );
    assert!(reload.poll(&mut backend).is_empty());
    assert!(backend.updates.is_empty());
}

// With the real compiler: a broken file read through the catalog fails with
// its error at the line it is on, under the path the catalog resolved, so a
// caller finds the entry's file by comparing paths.
#[test]
fn a_real_compile_error_names_the_catalogs_path_and_line() {
    concinnity_shader::require_dxc!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("broken.hlsl");
    std::fs::write(
        &path,
        "float4 shade(VertexOut v, GpuObjectData od)\n{\n    return missing_tint;\n}\n",
    )
    .unwrap();
    let catalog_entry = entry(9, 0, &[(ShaderStage::Fragment, &path)]);
    let texts = ShaderTexts::read(&catalog_entry).unwrap();
    let Err(ReloadFailure::Compile(failed)) = compile::compile("broken", &texts) else {
        panic!("a compile failure");
    };
    let errors: Vec<(&str, u32)> = failed.errors().map(|d| (d.path.as_str(), d.line)).collect();
    assert_eq!(
        errors,
        [(catalog_entry.path(ShaderStage::Fragment).unwrap(), 3)]
    );
    let failure = ReloadFailure::Compile(failed);
    assert_eq!(failure.first_error_at().as_deref(), Some("broken.hlsl:3"));
}
