use super::*;
use crate::debug::hot_reload::report::SubjectKind;
use concinnity_cook::compile::program::{CompileFailure, Diagnostic, EntryFailure, Severity};
use concinnity_cook::compile::sdf_field::CompiledField;
use concinnity_core::components::ShaderPrograms;
use concinnity_core::components::sdf_programs::SdfPrograms;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::shader_programs::raymarch::VolumeFlags;
use std::collections::HashSet;
use std::sync::Mutex;
use std::time::Duration;

// A backend that records every volume swap with the pipelines it was handed,
// holds the volumes in `resident`, and rejects every build when `reject` is
// set.
#[derive(Default)]
struct VolumeBackend {
    resident: HashSet<usize>,
    reject: bool,
    swaps: Vec<(usize, String)>,
    prepared: Vec<(usize, Option<FakePipelines>)>,
}

impl LiveEdit for VolumeBackend {
    fn replace_sdf_volume_pipelines(
        &mut self,
        volume: usize,
        programs: &SdfPrograms,
        prepared: Option<PreparedPipelines>,
    ) -> RenderResult<PipelineSwap> {
        if !self.resident.contains(&volume) {
            return Ok(PipelineSwap::NotResident);
        }
        self.swaps.push((volume, programs.field.text.clone()));
        self.prepared.push((
            volume,
            prepared.and_then(PreparedPipelines::downcast::<FakePipelines>),
        ));
        if self.reject {
            return Err(RenderError::ShaderCompile("pipeline build failed".into()));
        }
        Ok(PipelineSwap::Swapped)
    }
}

// What the fake builder builds: the volume it was labeled for, the flags and
// the field text.
#[derive(Debug, PartialEq, Eq)]
struct FakePipelines(String, VolumeFlags, String);

// Stands in for a backend's builder, refusing every build when `reject` is set.
struct FakeBuilder {
    reject: bool,
}

impl PipelineBuilder for FakeBuilder {
    fn world_shader(&self, _: u32, _: &ShaderPrograms) -> RenderResult<PreparedPipelines> {
        unreachable!("an SdfVolume reload builds no world Shader")
    }

    fn sdf_volume(
        &self,
        programs: &SdfPrograms,
        flags: VolumeFlags,
        label: &str,
    ) -> RenderResult<PreparedPipelines> {
        if self.reject {
            return Err(RenderError::ShaderCompile("pipeline state failed".into()));
        }
        Ok(PreparedPipelines::new(FakePipelines(
            label.to_string(),
            flags,
            programs.field.text.clone(),
        )))
    }
}

fn builder(reject: bool) -> Option<Arc<dyn PipelineBuilder>> {
    Some(Arc::new(FakeBuilder { reject }))
}

// Stands in for dxc and records each compile as (name, flags). A field
// containing "error" fails with an error on its first line; anything else
// "compiles" to programs carrying the field.
fn fake_compiler(calls: Arc<Mutex<Vec<(String, bool, bool)>>>) -> Compiler {
    Arc::new(move |name: &str, key: &FieldKey, field: &ShaderSource| {
        calls
            .lock()
            .unwrap()
            .push((name.to_string(), key.volumetric, key.cast_shadows));
        if field.text.contains("error") {
            let output = format!("{}:1:1: error: syntax error\n", field.path);
            return Err(ReloadFailure::Compile(CompileFailure {
                owner: format!("SdfVolume '{name}'"),
                diagnostics: vec![Diagnostic {
                    path: field.path.clone(),
                    line: 1,
                    column: 1,
                    severity: Severity::Error,
                    message: "syntax error".to_string(),
                    context: String::new(),
                }],
                failures: vec![EntryFailure {
                    entry: "raymarch_fragment".to_string(),
                    output,
                }],
                hint: "",
            }));
        }
        Ok(CompiledField {
            programs: SdfPrograms {
                field: field.clone(),
                programs: Vec::new(),
            },
            warnings: Vec::new(),
        })
    })
}

fn entry(name: &str, volume: usize, path: &std::path::Path, flags: (bool, bool)) -> SdfFieldEntry {
    SdfFieldEntry {
        name: name.to_string(),
        volume,
        volumetric: flags.0,
        cast_shadows: flags.1,
        resolved_path: path.to_string_lossy().into_owned(),
    }
}

fn pending(names: &[&str]) -> PendingSdfVolumes {
    PendingSdfVolumes {
        all: false,
        ids: names.iter().map(|n| n.to_string()).collect(),
    }
}

// Five volumes: three share `cloud.hlsl`, two of them with the same flags and
// the third casting; `blob` reads its own field; `lost` is not resident.
struct Fixture {
    dir: tempfile::TempDir,
    reload: SdfReload,
    calls: Arc<Mutex<Vec<(String, bool, bool)>>>,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        for (name, text) in [("cloud.hlsl", "cloud v1"), ("blob.hlsl", "blob v1")] {
            std::fs::write(dir.path().join(name), text).unwrap();
        }
        let p = |name: &str| dir.path().join(name);
        let catalog = SdfFieldMap {
            entries: vec![
                entry("cloud_a", 0, &p("cloud.hlsl"), (false, false)),
                entry("cloud_b", 1, &p("cloud.hlsl"), (false, false)),
                entry("cloud_caster", 2, &p("cloud.hlsl"), (false, true)),
                entry("blob", 3, &p("blob.hlsl"), (false, false)),
                entry("lost", 7, &p("blob.hlsl"), (true, false)),
            ],
        };
        let calls = Arc::new(Mutex::new(Vec::new()));
        let reload = SdfReload::with_compiler(catalog, fake_compiler(Arc::clone(&calls)));
        Self { dir, reload, calls }
    }

    fn write(&self, name: &str, text: &str) {
        std::fs::write(self.dir.path().join(name), text).unwrap();
    }

    fn compiles(&self) -> Vec<(String, bool, bool)> {
        let mut calls = self.calls.lock().unwrap().clone();
        calls.sort();
        calls
    }
}

fn backend(resident: &[usize]) -> VolumeBackend {
    VolumeBackend {
        resident: resident.iter().copied().collect(),
        ..Default::default()
    }
}

// Poll until `count` reports arrived; bounded so a lost result fails the test
// instead of hanging it.
fn poll_for(
    reload: &mut SdfReload,
    backend: &mut VolumeBackend,
    count: usize,
) -> Vec<ReloadReport> {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let mut reports = Vec::new();
    while reports.len() < count && std::time::Instant::now() < deadline {
        reports.extend(reload.poll(backend));
        std::thread::sleep(Duration::from_millis(5));
    }
    reports
}

// Each report as its volume's name and what became of it, frame time aside.
fn outcomes(reports: &[ReloadReport]) -> Vec<(&str, &'static str)> {
    let mut out: Vec<(&str, &'static str)> = reports
        .iter()
        .map(|r| {
            assert_eq!(r.subject.kind, SubjectKind::SdfVolume);
            let kind = match r.outcome {
                ReloadOutcome::Swapped { .. } => "swapped",
                ReloadOutcome::AppliesOnLoad { .. } => "applies on load",
                ReloadOutcome::Failed(_) => "failed",
            };
            (r.subject.name.as_str(), kind)
        })
        .collect();
    out.sort_unstable();
    out
}

// A field three volumes share compiles once per distinct flag set, and each
// compile swaps into every volume it was compiled for.
#[test]
fn a_shared_field_compiles_once_per_flag_set_and_swaps_every_volume() {
    let mut f = Fixture::new();
    let mut backend = backend(&[0, 1, 2, 3]);
    f.write("cloud.hlsl", "cloud v2");
    assert!(
        f.reload
            .request(&pending(&["cloud_a", "cloud_b", "cloud_caster"]), None)
            .is_empty()
    );
    let reports = poll_for(&mut f.reload, &mut backend, 3);
    assert_eq!(
        outcomes(&reports),
        [
            ("cloud_a", "swapped"),
            ("cloud_b", "swapped"),
            ("cloud_caster", "swapped")
        ]
    );
    assert_eq!(
        f.compiles(),
        [
            ("cloud_a".to_string(), false, false),
            ("cloud_caster".to_string(), false, true)
        ]
    );
    backend.swaps.sort();
    let text = "cloud v2".to_string();
    assert_eq!(
        backend.swaps,
        [(0, text.clone()), (1, text.clone()), (2, text)]
    );
}

// Only the volumes a request names compile.
#[test]
fn a_request_compiles_just_the_named_volumes() {
    let mut f = Fixture::new();
    let mut backend = backend(&[0, 1, 2, 3]);
    f.write("blob.hlsl", "blob v2");
    f.reload.request(&pending(&["blob"]), None);
    let reports = poll_for(&mut f.reload, &mut backend, 1);
    assert_eq!(outcomes(&reports), [("blob", "swapped")]);
    assert_eq!(backend.swaps, [(3, "blob v2".to_string())]);
    // Without a builder the swap is handed nothing and builds for itself.
    assert_eq!(backend.prepared, [(3, None)]);
    assert_eq!(f.compiles(), [("blob".to_string(), false, false)]);
}

// A volume the backend does not hold builds nothing and says when the edit
// lands instead.
#[test]
fn a_volume_the_backend_does_not_hold_applies_on_load() {
    let mut f = Fixture::new();
    let mut backend = backend(&[3]);
    f.reload.request(&pending(&["lost"]), None);
    let reports = poll_for(&mut f.reload, &mut backend, 1);
    assert_eq!(outcomes(&reports), [("lost", "applies on load")]);
    assert!(backend.swaps.is_empty());
}

// `reload-assets` asks for every volume.
#[test]
fn a_request_for_all_reaches_every_volume() {
    let mut f = Fixture::new();
    let mut backend = backend(&[0, 1, 2, 3]);
    f.reload.request(
        &PendingSdfVolumes {
            all: true,
            ids: Default::default(),
        },
        None,
    );
    assert_eq!(poll_for(&mut f.reload, &mut backend, 5).len(), 5);
    assert_eq!(f.compiles().len(), 4, "one per field and flag set");
}

// A compile error reaches no backend: every volume of the group keeps its
// live pipelines, and each is told why, at the catalog's path.
#[test]
fn a_compile_error_keeps_every_volume_as_it_was() {
    let mut f = Fixture::new();
    let mut backend = backend(&[0, 1]);
    f.write("cloud.hlsl", "cloud error");
    f.reload.request(&pending(&["cloud_a", "cloud_b"]), None);
    let reports = poll_for(&mut f.reload, &mut backend, 2);
    assert_eq!(
        outcomes(&reports),
        [("cloud_a", "failed"), ("cloud_b", "failed")]
    );
    let ReloadOutcome::Failed(ReloadFailure::Compile(failed)) = &reports[0].outcome else {
        panic!("a compile failure: {reports:?}");
    };
    let cloud = f.dir.path().join("cloud.hlsl");
    assert_eq!(failed.diagnostics[0].path, cloud.to_string_lossy());
    assert!(backend.swaps.is_empty());
}

// A pipeline the backend refuses to build is reported as refused.
#[test]
fn a_rejected_pipeline_is_reported() {
    let mut f = Fixture::new();
    let mut backend = VolumeBackend {
        reject: true,
        ..backend(&[3])
    };
    f.reload.request(&pending(&["blob"]), None);
    let reports = poll_for(&mut f.reload, &mut backend, 1);
    assert!(matches!(
        &reports[0].outcome,
        ReloadOutcome::Failed(ReloadFailure::Rejected(e)) if e.contains("pipeline build failed")
    ));
}

// With a builder, the worker builds each volume of a group its own pipelines
// from the group's one compile, under the volume's name and the group's flags,
// and each swap is handed its volume's.
#[test]
fn every_volume_of_a_group_is_handed_its_own_worker_built_pipelines() {
    let mut f = Fixture::new();
    let mut backend = backend(&[0, 1, 2]);
    f.write("cloud.hlsl", "cloud v2");
    f.reload.request(
        &pending(&["cloud_a", "cloud_b", "cloud_caster"]),
        builder(false),
    );
    let reports = poll_for(&mut f.reload, &mut backend, 3);
    assert_eq!(outcomes(&reports).len(), 3);
    assert_eq!(f.compiles().len(), 2, "one per flag set");
    backend.prepared.sort_by_key(|p| p.0);
    let built = |name: &str, cast_shadows| {
        let flags = VolumeFlags {
            volumetric: false,
            cast_shadows,
        };
        Some(FakePipelines(
            name.to_string(),
            flags,
            "cloud v2".to_string(),
        ))
    };
    assert_eq!(
        backend.prepared,
        [
            (0, built("cloud_a", false)),
            (1, built("cloud_b", false)),
            (2, built("cloud_caster", true))
        ]
    );
}

// Pipelines the builder refuses fail every volume of the group on the worker,
// and none reaches the backend.
#[test]
fn pipelines_the_builder_refuses_are_never_swapped() {
    let mut f = Fixture::new();
    let mut backend = backend(&[0, 1]);
    f.reload
        .request(&pending(&["cloud_a", "cloud_b"]), builder(true));
    let reports = poll_for(&mut f.reload, &mut backend, 2);
    assert_eq!(
        outcomes(&reports),
        [("cloud_a", "failed"), ("cloud_b", "failed")]
    );
    assert!(matches!(
        &reports[0].outcome,
        ReloadOutcome::Failed(ReloadFailure::Rejected(e)) if e.contains("pipeline state failed")
    ));
    assert!(backend.swaps.is_empty());
}

// A field that cannot be read fails every volume of its group at once, with
// no compile.
#[test]
fn an_unreadable_field_fails_before_compiling() {
    let mut f = Fixture::new();
    let mut backend = backend(&[0, 1]);
    std::fs::remove_file(f.dir.path().join("cloud.hlsl")).unwrap();
    let reports = f.reload.request(&pending(&["cloud_a", "cloud_b"]), None);
    assert_eq!(
        outcomes(&reports),
        [("cloud_a", "failed"), ("cloud_b", "failed")]
    );
    assert!(matches!(
        &reports[0].outcome,
        ReloadOutcome::Failed(ReloadFailure::Unstarted(e)) if e.contains("cloud.hlsl")
    ));
    std::thread::sleep(Duration::from_millis(50));
    assert!(f.reload.poll(&mut backend).is_empty());
    assert!(f.compiles().is_empty());
}

// The groups a request compiles: by field and flags, in a stable order.
#[test]
fn volumes_group_by_field_and_flags() {
    let f = Fixture::new();
    let grouped = groups(
        &f.reload.catalog,
        &pending(&["cloud_caster", "cloud_b", "cloud_a", "lost"]),
    );
    let names: Vec<Vec<&str>> = grouped
        .values()
        .map(|g| g.iter().map(|e| e.name.as_str()).collect())
        .collect();
    let mut want = vec![
        vec!["cloud_a", "cloud_b"],
        vec!["cloud_caster"],
        vec!["lost"],
    ];
    want.sort_by_key(|g| FieldKey::of(f.reload.catalog.get(g[0]).unwrap()));
    assert_eq!(names, want);
}

// An empty catalog makes every request and poll a no-op.
#[test]
fn an_empty_catalog_reloads_nothing() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut reload = SdfReload::with_compiler(SdfFieldMap::default(), fake_compiler(calls));
    let mut backend = VolumeBackend::default();
    assert!(
        reload
            .request(
                &PendingSdfVolumes {
                    all: true,
                    ids: Default::default()
                },
                None
            )
            .is_empty()
    );
    assert!(reload.poll(&mut backend).is_empty());
}

// With the real compiler: a broken field read through the catalog fails with
// its error at the line it is on, under the path the catalog resolved.
#[test]
fn a_real_compile_error_names_the_catalogs_path_and_line() {
    if !concinnity_shader::dxc_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("broken.hlsl");
    std::fs::write(
        &path,
        "float map(float3 p, SdfParams q, float t)\n{\n    return missing_radius;\n}\n",
    )
    .unwrap();
    let e = entry("broken", 0, &path, (false, false));
    let field = read_field(&e.resolved_path).unwrap();
    let Err(ReloadFailure::Compile(failed)) = compile::compile("broken", &FieldKey::of(&e), &field)
    else {
        panic!("a compile failure");
    };
    let errors: Vec<(&str, u32)> = failed.errors().map(|d| (d.path.as_str(), d.line)).collect();
    assert_eq!(errors, [(e.resolved_path.as_str(), 3)]);
    let failure = ReloadFailure::Compile(failed);
    assert_eq!(failure.first_error_at().as_deref(), Some("broken.hlsl:3"));
}
