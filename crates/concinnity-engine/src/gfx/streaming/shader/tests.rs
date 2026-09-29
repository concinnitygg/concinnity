use super::build::ShaderInstall;
use super::*;
use crate::gfx::mock_backend::{Call, MockBackend, MockPipeline, MockState, recording_backend};
use concinnity_core::components::sdf_programs::SdfPrograms;
use concinnity_core::components::{ShaderPrograms, ShaderSource, compiled_programs};
use concinnity_core::render::backend::{PipelineBuilder, PreparedPipelines};
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::shader_programs::raymarch::VolumeFlags;
use std::io::Write;
use std::sync::Mutex;
use std::time::Duration;

// Upper bound on pumps while waiting for a build thread, a few seconds at one
// millisecond apart.
const MAX_PUMPS: usize = 5000;

fn programs(name: &str) -> ShaderPrograms {
    ShaderPrograms {
        name: name.into(),
        vertex: None,
        fragment: ShaderSource {
            path: "shaders/wall.hlsl".into(),
            text: "float4 shade(VertexOut v, GpuObjectData od) { return 1.0; }".into(),
        },
        programs: vec![compiled_programs::CompiledProgram {
            entry: "fragment_main_bindless".into(),
            source_digest: 1,
            artifact: vec![4, 5],
        }],
    }
}

fn payload_bytes() -> Vec<u8> {
    programs("wall").encode().expect("encode")
}

fn edited(name: &str) -> Arc<ShaderPrograms> {
    Arc::new(ShaderPrograms {
        name: name.into(),
        ..Default::default()
    })
}

fn deferred(bucket: u32, source: ShaderPayloadSource) -> DeferredBucket {
    DeferredBucket { bucket, source }
}

// Prepares a `MockPipeline` named after the programs. A gated builder waits
// for one release per build; a rejecting one fails every build.
#[derive(Default)]
struct TestBuilder {
    gate: Option<Mutex<std::sync::mpsc::Receiver<()>>>,
    reject: bool,
}

impl TestBuilder {
    fn gated() -> (Arc<Self>, Sender<()>) {
        let (release, gate) = channel();
        let builder = Self {
            gate: Some(Mutex::new(gate)),
            reject: false,
        };
        (Arc::new(builder), release)
    }
}

impl PipelineBuilder for TestBuilder {
    fn world_shader(
        &self,
        _bucket: u32,
        programs: &ShaderPrograms,
    ) -> RenderResult<PreparedPipelines> {
        if let Some(gate) = &self.gate {
            let _ = gate.lock().unwrap().recv();
        }
        if self.reject {
            return Err(RenderError::ShaderCompile("pipeline state failed".into()));
        }
        Ok(PreparedPipelines::new(MockPipeline(programs.name.clone())))
    }

    fn sdf_volume(
        &self,
        _: &SdfPrograms,
        _: VolumeFlags,
        _: &str,
    ) -> RenderResult<PreparedPipelines> {
        unreachable!("a scene install builds no volume")
    }
}

// A warmup over buckets 1 and 2 driving a recording backend, with every
// residency change it reported.
struct Harness {
    warmup: ShaderWarmup,
    backend: MockBackend,
    state: Arc<Mutex<MockState>>,
    changes: Vec<(u32, bool)>,
}

impl Harness {
    fn new(overrides: Option<ShaderOverrides>, builder: Option<Arc<TestBuilder>>) -> Self {
        Self::with(
            vec![
                deferred(1, ShaderPayloadSource::Bytes(payload_bytes())),
                deferred(2, ShaderPayloadSource::Bytes(payload_bytes())),
            ],
            overrides,
            builder,
        )
    }

    fn with(
        deferred: Vec<DeferredBucket>,
        overrides: Option<ShaderOverrides>,
        builder: Option<Arc<TestBuilder>>,
    ) -> Self {
        let (state, backend) = recording_backend();
        state.lock().unwrap().pipeline_builder = builder.map(|b| b as Arc<dyn PipelineBuilder>);
        Self {
            warmup: ShaderWarmup::new(deferred, overrides),
            backend,
            state,
            changes: Vec::new(),
        }
    }

    // Pump once, returning what it recorded without replaying it.
    fn pump(&mut self) -> RenderOps {
        let mut ops = RenderOps::default();
        let changes = &mut self.changes;
        self.warmup.pump(&mut ops, |bucket, resident| {
            changes.push((bucket, resident))
        });
        ops
    }

    // Pump once and replay what it recorded.
    fn step(&mut self) {
        self.pump().replay(&mut self.backend);
    }

    // Step until `done` holds, pausing between steps while a build thread runs.
    fn step_until(&mut self, done: impl Fn(&Self) -> bool) {
        for _ in 0..MAX_PUMPS {
            self.step();
            if done(self) {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("the shader warmup never reached the expected state");
    }

    fn step_until_resident(&mut self, bucket: u32) {
        self.step_until(|h| h.changes.contains(&(bucket, true)));
    }

    fn step_until_idle(&mut self) {
        self.step_until(|h| h.warmup.entries.iter().all(|e| !e.building));
    }

    fn installs(&self) -> Vec<Call> {
        self.state
            .lock()
            .unwrap()
            .calls
            .iter()
            .filter(|c| matches!(c, Call::InstallWorldShader { .. }))
            .cloned()
            .collect()
    }
}

fn install(bucket: u32, name: &str, prepared: Option<&str>) -> Call {
    Call::InstallWorldShader {
        bucket,
        name: name.into(),
        prepared: prepared.map(Into::into),
    }
}

#[test]
fn blocked_buckets_record_nothing() {
    let mut h = Harness::new(None, Some(Arc::default()));
    assert!(!h.warmup.is_empty());
    assert!(h.pump().is_empty());
    assert!(h.changes.is_empty());
}

// The frame thread only swaps in what the build thread prepared, and the bucket
// turns resident with that install rather than with the dispatch.
#[test]
fn a_pinned_bucket_installs_the_pipeline_its_build_thread_prepared() {
    let mut h = Harness::new(None, Some(Arc::default()));
    h.warmup.set_blocked(1, false);
    h.step();
    assert!(h.changes.is_empty(), "the dispatch alone is not residency");
    h.step_until_resident(1);
    assert_eq!(h.installs(), [install(1, "wall", Some("wall"))]);
    assert_eq!(h.changes, [(1, true)]);
}

#[test]
fn without_a_builder_the_install_builds_for_itself() {
    let mut h = Harness::new(None, None);
    h.warmup.set_blocked(1, false);
    h.step_until_resident(1);
    assert_eq!(h.installs(), [install(1, "wall", None)]);
}

#[test]
fn the_scene_keeps_loading_while_the_build_runs() {
    let (builder, release) = TestBuilder::gated();
    let mut h = Harness::new(None, Some(builder));
    h.warmup.set_blocked(1, false);
    for _ in 0..3 {
        h.step();
    }
    assert!(h.changes.is_empty());
    assert!(h.installs().is_empty());
    release.send(()).unwrap();
    h.step_until_resident(1);
    assert_eq!(h.installs(), [install(1, "wall", Some("wall"))]);
}

#[test]
fn every_pinned_bucket_builds_at_once() {
    let mut h = Harness::new(None, Some(Arc::default()));
    h.warmup.set_blocked(1, false);
    h.warmup.set_blocked(2, false);
    h.step();
    assert!(h.warmup.entries.iter().all(|e| e.building));
    h.step_until(|h| h.changes.len() == 2);
    assert_eq!(h.installs().len(), 2);
}

// An edit that lands while the cooked programs build is built in their place,
// still off the frame thread.
#[test]
fn an_edit_landing_mid_build_is_built_instead() {
    let overrides = ShaderOverrides::default();
    let (builder, release) = TestBuilder::gated();
    let mut h = Harness::new(Some(overrides.clone()), Some(builder));
    h.warmup.set_blocked(1, false);
    h.step();
    overrides.set(1, edited("edited"));
    release.send(()).unwrap();
    release.send(()).unwrap();
    h.step_until_resident(1);
    assert_eq!(h.installs(), [install(1, "edited", Some("edited"))]);
}

// An edit that lands after the install was recorded, but before the frame
// thread replays it, is what installs; the stale pipeline is not.
#[test]
fn an_edit_landing_after_the_build_is_installed_in_its_place() {
    let overrides = ShaderOverrides::default();
    let mut h = Harness::new(Some(overrides.clone()), Some(Arc::default()));
    h.warmup.set_blocked(1, false);
    for _ in 0..MAX_PUMPS {
        let mut ops = h.pump();
        if h.changes.contains(&(1, true)) {
            overrides.set(1, edited("edited"));
            ops.replay(&mut h.backend);
            break;
        }
        ops.replay(&mut h.backend);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(h.installs(), [install(1, "edited", None)]);
}

#[test]
fn a_scene_unpinned_mid_build_discards_the_build() {
    let (builder, release) = TestBuilder::gated();
    let mut h = Harness::new(None, Some(builder));
    h.warmup.set_blocked(1, false);
    h.step();
    h.warmup.set_blocked(1, true);
    release.send(()).unwrap();
    h.step_until_idle();
    assert!(h.installs().is_empty());
    assert!(h.changes.is_empty());

    h.warmup.set_blocked(1, false);
    release.send(()).unwrap();
    h.step_until_resident(1);
    assert_eq!(h.installs(), [install(1, "wall", Some("wall"))]);
}

// A pin that returns before the build does takes that build rather than
// starting a second.
#[test]
fn a_scene_repinned_mid_build_takes_the_build_in_flight() {
    let (builder, release) = TestBuilder::gated();
    let mut h = Harness::new(None, Some(builder));
    h.warmup.set_blocked(1, false);
    h.step();
    h.warmup.set_blocked(1, true);
    h.step();
    h.warmup.set_blocked(1, false);
    release.send(()).unwrap();
    h.step_until_resident(1);
    assert_eq!(h.installs(), [install(1, "wall", Some("wall"))]);
}

#[test]
fn unpinning_a_resident_bucket_evicts_it() {
    let mut h = Harness::new(None, Some(Arc::default()));
    h.warmup.set_blocked(1, false);
    h.step_until_resident(1);
    h.warmup.set_blocked(1, true);
    h.step();
    assert_eq!(h.changes, [(1, true), (1, false)]);
    assert!(h.state.lock().unwrap().saw(&Call::EvictWorldShader(1)));
    h.step();
    assert_eq!(h.changes.len(), 2, "an evicted bucket stays evicted");
}

// A bucket whose install can never succeed still lets its scene finish loading.
#[test]
fn a_failed_build_is_resident_with_nothing_installed() {
    let builder = TestBuilder {
        reject: true,
        ..Default::default()
    };
    let mut h = Harness::new(None, Some(Arc::new(builder)));
    h.warmup.set_blocked(1, false);
    h.step_until_resident(1);
    assert!(h.installs().is_empty());
}

#[test]
fn an_unreadable_payload_is_resident_with_nothing_installed() {
    let mut h = Harness::with(
        vec![deferred(1, ShaderPayloadSource::Bytes(vec![0xff; 4]))],
        None,
        Some(Arc::default()),
    );
    h.warmup.set_blocked(1, false);
    h.step_until_resident(1);
    assert!(h.installs().is_empty());
}

#[test]
fn an_unknown_bucket_is_ignored() {
    let mut h = Harness::new(None, Some(Arc::default()));
    h.warmup.set_blocked(9, false);
    assert!(h.pump().is_empty());
}

#[test]
fn a_disk_backed_payload_is_read_from_its_range() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("blob").to_string_lossy().into_owned();
    let bytes = payload_bytes();
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(b"header").unwrap();
    file.write_all(&bytes).unwrap();
    let source = ShaderPayloadSource::Disk {
        path,
        offset: 6,
        len: bytes.len() as u64,
    };
    assert_eq!(
        source
            .decode()
            .expect("decode")
            .artifact("fragment_main_bindless", 1),
        Some(&[4u8, 5][..])
    );
}

#[test]
fn a_corrupt_payload_fails_to_decode() {
    assert!(ShaderPayloadSource::Bytes(vec![0xff; 4]).decode().is_err());
}

// A hot-reloaded edit wins over the cooked programs for its own bucket only,
// and is read when asked rather than when the install was made.
#[test]
fn an_override_wins_over_the_cooked_programs_for_its_bucket() {
    let overrides = ShaderOverrides::default();
    let install = |bucket| ShaderInstall {
        bucket,
        cooked: Arc::new(programs("wall")),
        overrides: Some(overrides.clone()),
    };
    let (one, two) = (install(1), install(2));
    assert_eq!(one.programs().name, "wall");
    overrides.set(1, edited("first"));
    overrides.set(1, edited("second"));
    assert_eq!(one.programs().name, "second");
    assert_eq!(two.programs().name, "wall");
}
