// One deferred bucket's install, built away from the frame thread: the payload
// read and decode, the choice between the cooked programs and a hot-reloaded
// edit, and the pipeline build, all on a thread of its own. What comes back is
// installed on the frame thread by `BuiltShader::install`.

use concinnity_core::components::ShaderPrograms;
use concinnity_core::render::backend::{PipelineBuilder, PreparedPipelines, RenderBackend};
use concinnity_core::render::error::RenderResult;
use std::sync::Arc;
use std::sync::mpsc::Sender;
use std::time::Instant;

use crate::gfx::system::parked::ShaderOverrides;

// Where a deferred bucket's compiled stage container is read from.
pub(crate) enum ShaderPayloadSource {
    // RAM-backed world (`cn debug`, `cn editor`): the payload bytes.
    Bytes(Vec<u8>),
    // Disk-backed world (`cn run`): the payload's absolute range in its blob.
    Disk { path: String, offset: u64, len: u64 },
}

impl ShaderPayloadSource {
    // Read and decode the cooked programs.
    pub(super) fn decode(&self) -> Result<ShaderPrograms, String> {
        let read;
        let bytes = match self {
            Self::Bytes(b) => b.as_slice(),
            Self::Disk { path, offset, len } => {
                read = super::super::file_range::read_at(path, *offset, *len)?;
                read.as_slice()
            }
        };
        ShaderPrograms::decode(bytes).map_err(|e| format!("payload decode: {e:?}"))
    }
}

// A bucket's cooked programs and the hot-reloaded edits that win over them.
pub(super) struct ShaderInstall {
    pub bucket: u32,
    pub cooked: Arc<ShaderPrograms>,
    pub overrides: Option<ShaderOverrides>,
}

impl ShaderInstall {
    // The programs to build: the bucket's hot-reloaded edit as of this call,
    // else the cooked ones.
    pub(super) fn programs(&self) -> Arc<ShaderPrograms> {
        self.overrides
            .as_ref()
            .and_then(|o| o.get(self.bucket))
            .unwrap_or_else(|| Arc::clone(&self.cooked))
    }
}

// A dispatched build, waiting for the frame thread to supply the backend's
// pipeline builder.
pub(super) struct BuildRequest {
    pub bucket: u32,
    pub source: Arc<ShaderPayloadSource>,
    pub overrides: Option<ShaderOverrides>,
    pub done: Sender<Built>,
}

// A finished build. `Err` when the payload could not be read or decoded, or the
// build thread could not start.
pub(super) struct Built {
    pub bucket: u32,
    pub outcome: Result<BuiltShader, String>,
}

pub(super) struct BuiltShader {
    pub install: ShaderInstall,
    // The programs the pipeline was built from.
    pub programs: Arc<ShaderPrograms>,
    // `None` when the backend offers no builder: the install builds for itself.
    pub prepared: Option<RenderResult<PreparedPipelines>>,
}

impl BuildRequest {
    // Run the build on a thread of its own, reporting through `done`. A build
    // that cannot be started reports that instead, so the bucket is never left
    // waiting on a result that will not come.
    pub(super) fn spawn(self, builder: Option<Arc<dyn PipelineBuilder>>) {
        let bucket = self.bucket;
        let done = self.done.clone();
        let spawned = std::thread::Builder::new()
            .name("shader-warmup".into())
            .spawn(move || {
                let built = self.run(builder.as_deref());
                let _ = self.done.send(built);
            });
        if let Err(e) = spawned {
            let _ = done.send(Built {
                bucket,
                outcome: Err(format!("build thread: {e}")),
            });
        }
    }

    pub(super) fn run(&self, builder: Option<&dyn PipelineBuilder>) -> Built {
        let bucket = self.bucket;
        let outcome = self.source.decode().map(|cooked| {
            let install = ShaderInstall {
                bucket,
                cooked: Arc::new(cooked),
                overrides: self.overrides.clone(),
            };
            let programs = install.programs();
            if !Arc::ptr_eq(&programs, &install.cooked) {
                tracing::info!(
                    "StreamingSystem: shader bucket {} installs its hot-reloaded edit",
                    bucket
                );
            }
            let prepared = builder.map(|builder| {
                let started = Instant::now();
                let prepared = builder.world_shader(bucket, &programs);
                tracing::info!(
                    "StreamingSystem: shader bucket {} pipeline built off the frame \
                     thread ({:.1} ms)",
                    bucket,
                    started.elapsed().as_secs_f32() * 1000.0
                );
                prepared
            });
            BuiltShader {
                install,
                programs,
                prepared,
            }
        });
        Built { bucket, outcome }
    }
}

impl BuiltShader {
    // Whether the programs built are still the ones to install: a hot-reloaded
    // edit may have landed since.
    pub(super) fn is_current(&self) -> bool {
        Arc::ptr_eq(&self.install.programs(), &self.programs)
    }

    // What to install, or the pipeline build's failure.
    pub(super) fn ready(self) -> RenderResult<ReadyShader> {
        Ok(ReadyShader {
            prepared: self.prepared.transpose()?,
            install: self.install,
            programs: self.programs,
        })
    }
}

// A build ready to install on the frame thread.
pub(super) struct ReadyShader {
    install: ShaderInstall,
    programs: Arc<ShaderPrograms>,
    prepared: Option<PreparedPipelines>,
}

impl ReadyShader {
    // Install the pipeline. An edit that landed after the build was checked is
    // installed in its place, built here.
    pub(super) fn install(self, backend: &mut dyn RenderBackend) {
        let bucket = self.install.bucket;
        let programs = self.install.programs();
        let prepared = self
            .prepared
            .filter(|_| Arc::ptr_eq(&programs, &self.programs));
        let started = Instant::now();
        match backend.install_world_shader(bucket, &programs, prepared) {
            Ok(()) => tracing::info!(
                "StreamingSystem: shader bucket {} pipeline ready ({:.1} ms)",
                bucket,
                started.elapsed().as_secs_f32() * 1000.0
            ),
            Err(e) => tracing::error!(
                "StreamingSystem: shader bucket {} pipeline build failed: {}",
                bucket,
                e
            ),
        }
    }
}
