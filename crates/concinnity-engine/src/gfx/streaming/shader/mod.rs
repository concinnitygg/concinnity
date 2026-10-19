// Scene-resident shader programs: the deferred payload source for each shader
// bucket a scene exclusively owns, plus the install / evict work the streaming
// pump applies to the backend as scenes pin and unpin.
//
// An install is built off the frame thread. The pump records a dispatch that,
// when the frame thread replays it, takes the backend's pipeline builder and
// starts a build thread; that thread reads and decodes the payload and builds
// the pipeline. The pump records the install once the build comes back, so the
// frame thread only swaps a finished pipeline in. A bucket counts as resident
// when its install is recorded, which keeps its scene loading until then.

mod build;
#[cfg(test)]
mod tests;

use concinnity_core::render::ops::RenderOps;
use concinnity_core::render::stream_pools::{Due, Finished, ShaderPump, ShaderResidency};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use crate::gfx::system::parked::ShaderOverrides;
use build::{BuildRequest, Built};

pub(crate) use build::ShaderPayloadSource;

// One deferred bucket as init recorded it.
pub(crate) struct DeferredBucket {
    pub bucket: u32,
    pub source: ShaderPayloadSource,
}

pub(crate) struct ShaderWarmup {
    residency: ShaderResidency<Arc<ShaderPayloadSource>>,
    // Hot-reloaded programs that win over the cooked payload, under
    // hot-reload capture only.
    overrides: Option<ShaderOverrides>,
    built_tx: Sender<Built>,
    built_rx: Receiver<Built>,
}

impl ShaderWarmup {
    pub(crate) fn new(deferred: Vec<DeferredBucket>, overrides: Option<ShaderOverrides>) -> Self {
        let (built_tx, built_rx) = channel();
        Self {
            residency: ShaderResidency::new(
                deferred.into_iter().map(|d| (d.bucket, Arc::new(d.source))),
            ),
            overrides,
            built_tx,
            built_rx,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.residency.is_empty()
    }
}

impl ShaderPump for ShaderWarmup {
    fn set_blocked(&mut self, bucket: u32, blocked: bool) {
        self.residency.set_blocked(bucket, blocked);
    }

    // Record this frame's shader work: the install of every build that came
    // back, a build for every bucket whose scene pinned, and the evict of every
    // bucket whose scene unpinned. `on_resident` hears each residency change.
    fn pump(&mut self, ops: &mut RenderOps, on_resident: &mut dyn FnMut(u32, bool)) {
        while let Ok(built) = self.built_rx.try_recv() {
            if let Some(bucket) = self.finish(built, ops) {
                on_resident(bucket, true);
            }
        }
        let (overrides, done) = (&self.overrides, &self.built_tx);
        self.residency.sweep(|due| match due {
            Due::Build(bucket, source) => dispatch(bucket, source, overrides, done, ops),
            Due::Evict(bucket) => {
                ops.record(move |backend| {
                    backend.evict_world_shader(bucket);
                    tracing::info!("shader bucket {} pipeline released", bucket);
                });
                on_resident(bucket, false);
            }
        });
    }
}

// Record a build of `bucket`, started when the frame thread replays it.
fn dispatch(
    bucket: u32,
    source: &Arc<ShaderPayloadSource>,
    overrides: &Option<ShaderOverrides>,
    done: &Sender<Built>,
    ops: &mut RenderOps,
) {
    let request = BuildRequest {
        bucket,
        source: Arc::clone(source),
        overrides: overrides.clone(),
        done: done.clone(),
    };
    ops.record(move |backend| request.spawn(backend.pipeline_builder()));
}

impl ShaderWarmup {
    // Apply one finished build, returning the bucket it made resident. A
    // bucket that cannot be installed (unreadable payload, a failed pipeline
    // build) is resident anyway after the error: its draws stay skipped.
    fn finish(&mut self, built: Built, ops: &mut RenderOps) -> Option<u32> {
        let bucket = built.bucket;
        let stale = built
            .outcome
            .as_ref()
            .is_ok_and(|shader| !shader.is_current());
        match self.residency.finish(bucket, stale) {
            Finished::Dropped => None,
            Finished::Rebuild(source) => {
                dispatch(bucket, source, &self.overrides, &self.built_tx, ops);
                None
            }
            Finished::Resident => {
                match built.outcome {
                    Err(e) => tracing::error!("shader bucket {} could not be built: {}", bucket, e),
                    Ok(shader) => match shader.ready() {
                        Ok(ready) => ops.record(move |backend| ready.install(backend)),
                        Err(e) => {
                            tracing::error!("shader bucket {} pipeline build failed: {}", bucket, e)
                        }
                    },
                }
                Some(bucket)
            }
        }
    }
}
