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

struct Entry {
    bucket: u32,
    source: Arc<ShaderPayloadSource>,
    // Set while the owning scene is unpinned.
    blocked: bool,
    resident: bool,
    // A build is in flight.
    building: bool,
}

impl Entry {
    fn wants_install(&self) -> bool {
        !self.blocked && !self.resident && !self.building
    }

    fn wants_evict(&self) -> bool {
        self.blocked && self.resident
    }
}

pub(crate) struct ShaderWarmup {
    entries: Vec<Entry>,
    // Hot-reloaded programs that win over the cooked payload, under
    // hot-reload capture only.
    overrides: Option<ShaderOverrides>,
    built_tx: Sender<Built>,
    built_rx: Receiver<Built>,
}

impl ShaderWarmup {
    // Every deferred bucket starts blocked and non-resident, matching every
    // scene starting unpinned: the first pin sync unblocks the start scene's.
    pub(crate) fn new(deferred: Vec<DeferredBucket>, overrides: Option<ShaderOverrides>) -> Self {
        let (built_tx, built_rx) = channel();
        Self {
            entries: deferred
                .into_iter()
                .map(|d| Entry {
                    bucket: d.bucket,
                    source: Arc::new(d.source),
                    blocked: true,
                    resident: false,
                    building: false,
                })
                .collect(),
            overrides,
            built_tx,
            built_rx,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn set_blocked(&mut self, bucket: u32, blocked: bool) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.bucket == bucket) {
            e.blocked = blocked;
        }
    }

    // Record this frame's shader work: the install of every build that came
    // back, a build for every bucket whose scene pinned, and the evict of every
    // bucket whose scene unpinned. `on_resident` hears each residency change.
    //
    // A bucket that cannot be installed (unreadable payload, a failed pipeline
    // build) is reported resident anyway after the error: its draws stay
    // skipped, but the owning scene finishes loading instead of holding its
    // loading screen open forever on work that will never succeed.
    pub(crate) fn pump(&mut self, ops: &mut RenderOps, mut on_resident: impl FnMut(u32, bool)) {
        while let Ok(built) = self.built_rx.try_recv() {
            if let Some(bucket) = self.finish(built, ops) {
                on_resident(bucket, true);
            }
        }
        for i in 0..self.entries.len() {
            let entry = &self.entries[i];
            if entry.wants_install() {
                self.dispatch(i, ops);
            } else if entry.wants_evict() {
                let bucket = entry.bucket;
                ops.record(move |backend| {
                    backend.evict_world_shader(bucket);
                    tracing::info!(
                        "StreamingSystem: shader bucket {} pipeline released",
                        bucket
                    );
                });
                self.entries[i].resident = false;
                on_resident(bucket, false);
            }
        }
    }

    // Record a build of entry `i`, started when the frame thread replays it.
    fn dispatch(&mut self, i: usize, ops: &mut RenderOps) {
        let entry = &mut self.entries[i];
        entry.building = true;
        let request = BuildRequest {
            bucket: entry.bucket,
            source: Arc::clone(&entry.source),
            overrides: self.overrides.clone(),
            done: self.built_tx.clone(),
        };
        ops.record(move |backend| request.spawn(backend.pipeline_builder()));
    }

    // Apply one finished build, returning the bucket it made resident.
    fn finish(&mut self, built: Built, ops: &mut RenderOps) -> Option<u32> {
        let i = self.entries.iter().position(|e| e.bucket == built.bucket)?;
        self.entries[i].building = false;
        if self.entries[i].blocked {
            // The scene unpinned mid-build; its next pin builds afresh.
            return None;
        }
        let bucket = built.bucket;
        match built.outcome {
            Err(e) => tracing::error!(
                "StreamingSystem: shader bucket {} could not be built: {}",
                bucket,
                e
            ),
            Ok(shader) if !shader.is_current() => {
                // A hot-reloaded edit landed mid-build: build that instead.
                self.dispatch(i, ops);
                return None;
            }
            Ok(shader) => match shader.ready() {
                Ok(ready) => ops.record(move |backend| ready.install(backend)),
                Err(e) => tracing::error!(
                    "StreamingSystem: shader bucket {} pipeline build failed: {}",
                    bucket,
                    e
                ),
            },
        }
        self.entries[i].resident = true;
        Some(bucket)
    }
}
