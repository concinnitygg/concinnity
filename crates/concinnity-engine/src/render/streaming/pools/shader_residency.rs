// Which scene-owned shader buckets to build, install and release as their
// scenes pin and unpin. The builds themselves are the host's: this decides
// only what is due and what a finished build means.

use alloc::vec::Vec;

struct Entry<S> {
    bucket: u32,
    source: S,
    // Set while the owning scene is unpinned.
    blocked: bool,
    resident: bool,
    // A build is in flight.
    building: bool,
}

/// What a finished build means for its bucket.
#[derive(Debug, PartialEq, Eq)]
pub enum Finished<'a, S> {
    /// Discard it: the bucket's scene unpinned mid-build (its next pin builds
    /// afresh), or the bucket is unknown.
    Dropped,
    /// It built stale programs: build again from this source.
    Rebuild(&'a S),
    /// Install it (or report its error): the bucket is resident either way, so
    /// its scene finishes loading rather than waiting on work that will never
    /// succeed.
    Resident,
}

/// Work a sweep finds due.
#[derive(Debug, PartialEq, Eq)]
pub enum Due<'a, S> {
    /// Start a build of the bucket from its source.
    Build(u32, &'a S),
    /// Release the bucket's pipeline.
    Evict(u32),
}

/// Scene-owned shader buckets, each with the host's source of its programs.
/// Every bucket starts blocked and not resident, matching every scene starting
/// unpinned.
pub struct ShaderResidency<S> {
    entries: Vec<Entry<S>>,
}

impl<S> ShaderResidency<S> {
    /// One entry per `(bucket, source)`.
    pub fn new(buckets: impl IntoIterator<Item = (u32, S)>) -> Self {
        Self {
            entries: buckets
                .into_iter()
                .map(|(bucket, source)| Entry {
                    bucket,
                    source,
                    blocked: true,
                    resident: false,
                    building: false,
                })
                .collect(),
        }
    }

    /// Whether no bucket is deferred.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many builds are in flight.
    pub fn building(&self) -> usize {
        self.entries.iter().filter(|e| e.building).count()
    }

    /// Block or unblock `bucket` as its scene unpins or pins.
    pub fn set_blocked(&mut self, bucket: u32, blocked: bool) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.bucket == bucket) {
            e.blocked = blocked;
        }
    }

    /// Take a finished build of `bucket`, `stale` when what it built is no
    /// longer the bucket's current programs.
    pub fn finish(&mut self, bucket: u32, stale: bool) -> Finished<'_, S> {
        let Some(e) = self.entries.iter_mut().find(|e| e.bucket == bucket) else {
            return Finished::Dropped;
        };
        if e.blocked {
            e.building = false;
            return Finished::Dropped;
        }
        if stale {
            return Finished::Rebuild(&e.source);
        }
        e.building = false;
        e.resident = true;
        Finished::Resident
    }

    /// Start every due build (a pinned bucket neither resident nor building)
    /// and release every resident bucket whose scene unpinned.
    pub fn sweep(&mut self, mut due: impl FnMut(Due<'_, S>)) {
        for e in &mut self.entries {
            if !e.blocked && !e.resident && !e.building {
                e.building = true;
                due(Due::Build(e.bucket, &e.source));
            } else if e.blocked && e.resident {
                e.resident = false;
                due(Due::Evict(e.bucket));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn sweep(r: &mut ShaderResidency<&'static str>) -> (Vec<(u32, &'static str)>, Vec<u32>) {
        let (mut built, mut evicted) = (Vec::new(), Vec::new());
        r.sweep(|due| match due {
            Due::Build(b, s) => built.push((b, *s)),
            Due::Evict(b) => evicted.push(b),
        });
        (built, evicted)
    }

    fn residency() -> ShaderResidency<&'static str> {
        ShaderResidency::new([(1, "one"), (2, "two")])
    }

    #[test]
    fn blocked_buckets_are_never_due() {
        let mut r = residency();
        assert!(!r.is_empty());
        assert_eq!(sweep(&mut r), (vec![], vec![]));
    }

    // A pinned bucket builds once; a second sweep while it builds starts none.
    #[test]
    fn a_pinned_bucket_builds_once() {
        let mut r = residency();
        r.set_blocked(2, false);
        assert_eq!(sweep(&mut r), (vec![(2, "two")], vec![]));
        assert_eq!(sweep(&mut r), (vec![], vec![]));
        assert_eq!(r.building(), 1);
        assert_eq!(r.finish(2, false), Finished::Resident);
        assert_eq!(r.building(), 0);
        assert_eq!(sweep(&mut r), (vec![], vec![]));
    }

    #[test]
    fn a_stale_build_is_built_again_and_stays_building() {
        let mut r = residency();
        r.set_blocked(1, false);
        sweep(&mut r);
        assert_eq!(r.finish(1, true), Finished::Rebuild(&"one"));
        assert_eq!(sweep(&mut r), (vec![], vec![]), "the rebuild is in flight");
        assert_eq!(r.finish(1, false), Finished::Resident);
    }

    // An unpin mid-build discards the build; the next pin builds afresh.
    #[test]
    fn a_build_finishing_after_its_scene_unpinned_is_dropped() {
        let mut r = residency();
        r.set_blocked(1, false);
        sweep(&mut r);
        r.set_blocked(1, true);
        assert_eq!(r.finish(1, false), Finished::Dropped);
        r.set_blocked(1, false);
        assert_eq!(sweep(&mut r), (vec![(1, "one")], vec![]));
    }

    #[test]
    fn unpinning_a_resident_bucket_releases_it_once() {
        let mut r = residency();
        r.set_blocked(1, false);
        sweep(&mut r);
        r.finish(1, false);
        r.set_blocked(1, true);
        assert_eq!(sweep(&mut r), (vec![], vec![1]));
        assert_eq!(sweep(&mut r), (vec![], vec![]));
    }

    #[test]
    fn an_unknown_bucket_is_dropped() {
        assert_eq!(residency().finish(9, false), Finished::Dropped);
    }
}
