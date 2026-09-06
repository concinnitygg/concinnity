// Scene-resident shader programs: the deferred payload source for each shader
// bucket a scene exclusively owns, plus the install / evict work the streaming
// pump applies to the backend as scenes pin and unpin.
//
// Unlike the texture and mesh pools there is no byte budget and no worker
// thread. A world declares a handful of shaders, and what is being deferred is
// pipeline-state creation on the render device, which has to happen on the
// thread that owns the device. The pump takes one bucket per frame, so a scene
// with several shaders spreads its warmup across the loading screen instead of
// building every pipeline in one frame.

use crate::components::ShaderPrograms;

// Where a deferred bucket's compiled stage container is read from.
pub(crate) enum ShaderPayloadSource {
    // RAM-backed world (`cn debug`, `cn editor`): the payload bytes.
    Bytes(Vec<u8>),
    // Disk-backed world (`cn run`): the payload's absolute range in its blob.
    Disk { path: String, offset: u64, len: u64 },
}

// One deferred bucket as init recorded it.
pub(crate) struct DeferredBucket {
    pub bucket: u32,
    pub source: ShaderPayloadSource,
}

struct Entry {
    bucket: u32,
    source: ShaderPayloadSource,
    // Set while the owning scene is unpinned.
    blocked: bool,
    resident: bool,
}

pub(crate) struct ShaderWarmup {
    entries: Vec<Entry>,
}

impl ShaderWarmup {
    // Every deferred bucket starts blocked and non-resident, matching every
    // scene starting unpinned: the first pin sync unblocks the start scene's.
    pub(crate) fn new(deferred: Vec<DeferredBucket>) -> Self {
        Self {
            entries: deferred
                .into_iter()
                .map(|d| Entry {
                    bucket: d.bucket,
                    source: d.source,
                    blocked: true,
                    resident: false,
                })
                .collect(),
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

    // The next bucket whose residency disagrees with its scene's pin state, as
    // `(bucket, want_resident)`. One per call: the pump applies it and comes
    // back next frame for the rest.
    pub(crate) fn next_pending(&self) -> Option<(u32, bool)> {
        self.entries
            .iter()
            .find(|e| e.resident == e.blocked)
            .map(|e| (e.bucket, !e.blocked))
    }

    // Read and decode one bucket's compiled programs.
    pub(crate) fn load(&self, bucket: u32) -> Result<ShaderPrograms, String> {
        let entry = self
            .entries
            .iter()
            .find(|e| e.bucket == bucket)
            .ok_or_else(|| format!("shader bucket {bucket} is not deferred"))?;
        let bytes = match &entry.source {
            ShaderPayloadSource::Bytes(b) => b.clone(),
            ShaderPayloadSource::Disk { path, offset, len } => {
                super::file_range::read_at(path, *offset, *len)?
            }
        };
        ShaderPrograms::decode(&bytes)
            .map_err(|e| format!("shader bucket {bucket}: payload decode: {e:?}"))
    }

    pub(crate) fn note_resident(&mut self, bucket: u32, resident: bool) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.bucket == bucket) {
            e.resident = resident;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn payload_bytes() -> Vec<u8> {
        ShaderPrograms {
            name: "wall".into(),
            vertex: None,
            fragment: "float4 shade(VertexOut in, GpuObjectData od) { return 1.0; }".into(),
            programs: vec![crate::components::compiled_programs::CompiledProgram {
                entries: vec!["fragment_main_bindless".into()],
                source_digest: 1,
                artifact: vec![4, 5],
            }],
        }
        .encode()
        .expect("encode")
    }

    fn deferred(bucket: u32, source: ShaderPayloadSource) -> DeferredBucket {
        DeferredBucket { bucket, source }
    }

    fn warmup() -> ShaderWarmup {
        ShaderWarmup::new(vec![
            deferred(1, ShaderPayloadSource::Bytes(payload_bytes())),
            deferred(2, ShaderPayloadSource::Bytes(payload_bytes())),
        ])
    }

    #[test]
    fn buckets_start_blocked_with_nothing_pending() {
        let w = warmup();
        assert!(!w.is_empty());
        assert_eq!(w.next_pending(), None);
    }

    #[test]
    fn unblocking_queues_an_install_that_residency_clears() {
        let mut w = warmup();
        w.set_blocked(1, false);
        assert_eq!(w.next_pending(), Some((1, true)));
        w.note_resident(1, true);
        assert_eq!(w.next_pending(), None);
    }

    #[test]
    fn reblocking_a_resident_bucket_queues_an_evict() {
        let mut w = warmup();
        w.set_blocked(1, false);
        w.note_resident(1, true);
        w.set_blocked(1, true);
        assert_eq!(w.next_pending(), Some((1, false)));
        w.note_resident(1, false);
        assert_eq!(w.next_pending(), None);
    }

    #[test]
    fn pending_work_is_served_one_bucket_at_a_time() {
        let mut w = warmup();
        w.set_blocked(1, false);
        w.set_blocked(2, false);
        assert_eq!(w.next_pending(), Some((1, true)));
        w.note_resident(1, true);
        assert_eq!(w.next_pending(), Some((2, true)));
    }

    #[test]
    fn load_decodes_the_programs() {
        let programs = warmup().load(1).expect("load");
        assert_eq!(programs.programs.len(), 1);
        assert_eq!(
            programs.artifact("fragment_main_bindless", 1),
            Some(&[4u8, 5][..])
        );
        assert!(programs.vertex.is_none());
    }

    #[test]
    fn load_reads_a_disk_backed_payload_range() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob").to_string_lossy().into_owned();
        let bytes = payload_bytes();
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(b"header").unwrap();
        file.write_all(&bytes).unwrap();
        let w = ShaderWarmup::new(vec![deferred(
            3,
            ShaderPayloadSource::Disk {
                path,
                offset: 6,
                len: bytes.len() as u64,
            },
        )]);
        assert_eq!(
            w.load(3)
                .expect("load")
                .artifact("fragment_main_bindless", 1),
            Some(&[4u8, 5][..])
        );
    }

    #[test]
    fn load_reports_an_unknown_bucket_and_a_corrupt_payload() {
        assert!(warmup().load(9).is_err());
        let w = ShaderWarmup::new(vec![deferred(1, ShaderPayloadSource::Bytes(vec![0xff; 4]))]);
        assert!(w.load(1).is_err());
    }

    #[test]
    fn notes_for_unknown_buckets_are_ignored() {
        let mut w = warmup();
        w.note_resident(9, true);
        w.set_blocked(9, false);
        assert_eq!(w.next_pending(), None);
    }
}
