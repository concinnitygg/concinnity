//! The emitter table the backends share: one slot per emitter id holding its
//! record, the cached cull bounds, the spawn ring, and the backend's own
//! per-emitter GPU state `G`. The backend keeps pipelines and encoding; this
//! owns which emitters exist, how far each has spawned, and which are seen.

use crate::gfx::frustum::Frustum;
use crate::gfx::render_types::ParticleParams;
use crate::render::particles::{
    MAX_EMITTERS, ParticleEmitterRecord, ParticleSpawnState, ParticleSpawns, spawn_seed,
};
use crate::render::slots::table::{AtCapacity, RemoveError, SlotTable};

/// The per-frame clock [`ParticleSystem::prepare`] advanced: the integration
/// step and the frame counter the kernel's RNG seed mixes in. Each emitter's
/// spawn run for the same frame lives in the system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleFrame {
    /// Seconds since the previous prepared frame, never negative.
    pub dt: f32,
    /// Monotonic, wrapping frame counter.
    pub frame_index: u32,
}

struct Emitter<G> {
    record: ParticleEmitterRecord,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    spawn_state: ParticleSpawnState,
    spawns: ParticleSpawns,
    gpu: G,
}

/// The emitters a backend's particle passes simulate and draw, addressed by the
/// slot id [`ParticleSystem::insert`] returns. That id is stable until
/// [`ParticleSystem::remove`] tombstones it, and indexes any parallel
/// per-emitter resource the backend reserves (a descriptor, a uniform slot).
///
/// The table holds at most [`MAX_EMITTERS`] live emitters on every backend.
///
/// ```rust
/// # use concinnity_core::components::ParticleEmitter;
/// # use concinnity_core::render::particles::{ParticleEmitterRecord, ParticleSystem};
/// let record = ParticleEmitterRecord::new(&ParticleEmitter::default(), 0);
/// let mut particles = ParticleSystem::new();
/// let id = particles.insert(record, "pool").expect("room for one");
/// let frame = particles.prepare(0.5).expect("an emitter exists");
/// assert_eq!(particles.emitters(&frame).count(), 1);
/// assert_eq!(particles.remove(id), Ok("pool"));
/// ```
pub struct ParticleSystem<G> {
    emitters: SlotTable<Emitter<G>>,
    last_elapsed: f32,
    frame_index: u32,
}

impl<G> Default for ParticleSystem<G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G> ParticleSystem<G> {
    /// An empty system holding at most [`MAX_EMITTERS`] live emitters.
    pub fn new() -> Self {
        Self::with_capacity(MAX_EMITTERS)
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            emitters: SlotTable::new(capacity),
            last_elapsed: 0.0,
            frame_index: 0,
        }
    }

    /// Whether another [`Self::insert`] would succeed, checked before the
    /// backend allocates an emitter's GPU state.
    pub fn has_room(&self) -> bool {
        self.emitters.has_room()
    }

    /// Place `record` with its GPU state in a slot and return the id, reusing a
    /// tombstoned slot before growing. The emitter starts with an empty pool.
    pub fn insert(&mut self, record: ParticleEmitterRecord, gpu: G) -> Result<usize, AtCapacity> {
        let (aabb_min, aabb_max) = record.aabb();
        self.emitters.insert(Emitter {
            record,
            aabb_min,
            aabb_max,
            spawn_state: ParticleSpawnState::default(),
            spawns: ParticleSpawns::default(),
            gpu,
        })
    }

    /// Tombstone the emitter at `id`, handing its GPU state back so the backend
    /// can release it on its own terms.
    pub fn remove(&mut self, id: usize) -> Result<G, RemoveError> {
        self.emitters.remove(id).map(|e| e.gpu)
    }

    /// Whether no emitter is live, so both passes can be skipped.
    pub fn is_empty(&self) -> bool {
        self.emitters.is_empty()
    }

    /// Whether no emitter has ever been inserted, tombstoned or not.
    pub fn never_used(&self) -> bool {
        self.emitters.never_used()
    }

    /// Advance the frame clock to `elapsed` seconds and take every live
    /// emitter's spawn run for this frame. Run exactly once per frame, before
    /// either pass encodes, so the simulation and the draw see one frame.
    /// `None` until an emitter has ever been inserted; the clock keeps running
    /// while every slot is tombstoned.
    pub fn prepare(&mut self, elapsed: f32) -> Option<ParticleFrame> {
        if self.emitters.never_used() {
            return None;
        }
        let dt = (elapsed - self.last_elapsed).max(0.0);
        self.last_elapsed = elapsed;
        self.frame_index = self.frame_index.wrapping_add(1);
        for (_, e) in self.emitters.iter_mut() {
            e.spawns = e.spawn_state.take_spawns(dt, &e.record);
        }
        Some(ParticleFrame {
            dt,
            frame_index: self.frame_index,
        })
    }

    /// Every live emitter with this frame's simulation uniform, in id order.
    /// The simulation ticks all of them, seen or not, so an off-screen pool
    /// keeps evolving for when the camera turns back.
    pub fn emitters<'a>(
        &'a self,
        frame: &'a ParticleFrame,
    ) -> impl Iterator<Item = ActiveEmitter<'a, G>> + 'a {
        self.emitters
            .iter()
            .map(move |(id, e)| ActiveEmitter::new(id, e, *frame))
    }

    /// The live emitters whose spawn bounds meet `frustum`, in id order. Each
    /// is tested once: a caller that needs to know whether the draw runs at all
    /// peeks this iterator rather than counting it first.
    pub fn visible<'a>(
        &'a self,
        frame: &'a ParticleFrame,
        frustum: &'a Frustum,
    ) -> impl Iterator<Item = ActiveEmitter<'a, G>> + 'a {
        self.emitters
            .iter()
            .filter(|(_, e)| frustum.intersects_aabb(e.aabb_min, e.aabb_max))
            .map(move |(id, e)| ActiveEmitter::new(id, e, *frame))
    }

    /// Every live emitter's GPU state, in id order.
    pub fn gpu_states(&self) -> impl Iterator<Item = &G> {
        self.emitters.iter().map(|(_, e)| &e.gpu)
    }

    /// Drop every emitter, live or tombstoned, with its GPU state.
    pub fn clear(&mut self) {
        self.emitters.clear();
    }
}

/// A live emitter for one pass: its id, record, GPU state, and this frame's
/// simulation uniform.
pub struct ActiveEmitter<'a, G> {
    /// The emitter's slot id.
    pub id: usize,
    /// The emitter's record.
    pub record: &'a ParticleEmitterRecord,
    /// The backend's per-emitter GPU state.
    pub gpu: &'a G,
    /// This frame's uniform: the record's fields plus the frame step, the
    /// emitter's spawn run, and its RNG seed. The simulation and the draw bind
    /// the same block; the draw reads only the record's size and color ramps.
    pub params: ParticleParams,
}

impl<'a, G> ActiveEmitter<'a, G> {
    fn new(id: usize, e: &'a Emitter<G>, frame: ParticleFrame) -> Self {
        Self {
            id,
            record: &e.record,
            gpu: &e.gpu,
            params: e
                .record
                .params(frame.dt, e.spawns, spawn_seed(frame.frame_index, id)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::ParticleEmitter;
    use crate::gfx::frustum::Plane;
    use alloc::vec;
    use alloc::vec::Vec;

    fn record_at(position: [f32; 3]) -> ParticleEmitterRecord {
        let emitter = ParticleEmitter {
            position,
            spawn_rate: 10.0,
            speed_min: 0.0,
            speed_max: 0.0,
            gravity: [0.0; 3],
            size_start: 0.0,
            size_end: 0.0,
            ..Default::default()
        };
        ParticleEmitterRecord::new(&emitter, 0)
    }

    fn frustum_left_of_ten() -> Frustum {
        let wide = |normal: [f32; 3]| Plane { normal, d: 1.0e6 };
        Frustum {
            planes: [
                Plane {
                    normal: [-1.0, 0.0, 0.0],
                    d: 10.0,
                },
                wide([1.0, 0.0, 0.0]),
                wide([0.0, 1.0, 0.0]),
                wide([0.0, -1.0, 0.0]),
                wide([0.0, 0.0, 1.0]),
                wide([0.0, 0.0, -1.0]),
            ],
        }
    }

    #[test]
    fn a_never_used_system_prepares_nothing() {
        let mut s = ParticleSystem::<()>::with_capacity(4);
        assert!(s.is_empty());
        assert_eq!(s.prepare(1.0), None);
    }

    #[test]
    fn prepare_steps_the_clock_and_the_frame_counter() {
        let mut s = ParticleSystem::with_capacity(4);
        s.insert(record_at([0.0; 3]), ()).unwrap();
        let a = s.prepare(0.25).unwrap();
        assert_eq!((a.dt, a.frame_index), (0.25, 1));
        let b = s.prepare(0.75).unwrap();
        assert_eq!((b.dt, b.frame_index), (0.5, 2));
        // A clock that went backwards steps by zero.
        assert_eq!(s.prepare(0.5).unwrap().dt, 0.0);
    }

    #[test]
    fn the_clock_keeps_running_while_every_emitter_is_removed() {
        let mut s = ParticleSystem::with_capacity(4);
        let id = s.insert(record_at([0.0; 3]), ()).unwrap();
        s.prepare(1.0);
        s.remove(id).unwrap();
        assert!(s.is_empty());
        assert!(!s.never_used());
        assert_eq!(s.prepare(2.0).unwrap().frame_index, 2);
        s.insert(record_at([0.0; 3]), ()).unwrap();
        assert_eq!(s.prepare(2.5).unwrap().dt, 0.5);
    }

    #[test]
    fn the_uniform_carries_the_spawn_run_and_seed() {
        let mut s = ParticleSystem::with_capacity(4);
        let rec = record_at([0.0; 3]);
        s.insert(rec, ()).unwrap();
        s.insert(rec, ()).unwrap();
        let frame = s.prepare(0.5).unwrap();
        let all: Vec<_> = s.emitters(&frame).collect();
        assert_eq!(all.len(), 2);
        for e in &all {
            assert_eq!(e.params.dt, 0.5);
            assert_eq!(e.params.spawn_count, 5);
            assert_eq!(e.params.spawn_first, 0);
            assert_eq!(e.params.random_seed, spawn_seed(1, e.id));
        }
        // The next frame continues each emitter's own ring.
        let frame = s.prepare(1.0).unwrap();
        let first: Vec<u32> = s.emitters(&frame).map(|e| e.params.spawn_first).collect();
        assert_eq!(first, vec![5, 5]);
    }

    #[test]
    fn only_emitters_meeting_the_frustum_are_visible() {
        let mut s = ParticleSystem::with_capacity(4);
        let near = s.insert(record_at([0.0; 3]), 'n').unwrap();
        s.insert(record_at([100.0, 0.0, 0.0]), 'f').unwrap();
        let frame = s.prepare(0.1).unwrap();
        let seen: Vec<(usize, char)> = s
            .visible(&frame, &frustum_left_of_ten())
            .map(|e| (e.id, *e.gpu))
            .collect();
        assert_eq!(seen, vec![(near, 'n')]);
        // The culled emitter still simulates.
        assert_eq!(s.emitters(&frame).count(), 2);
    }

    #[test]
    fn a_reused_slot_starts_a_fresh_ring() {
        let mut s = ParticleSystem::with_capacity(1);
        let id = s.insert(record_at([0.0; 3]), 1).unwrap();
        s.prepare(1.0);
        assert!(!s.has_room());
        assert_eq!(s.insert(record_at([0.0; 3]), 2), Err(AtCapacity));
        assert_eq!(s.remove(id), Ok(1));
        assert_eq!(s.insert(record_at([0.0; 3]), 3), Ok(id));
        let frame = s.prepare(1.5).unwrap();
        let e = s.emitters(&frame).next().unwrap();
        assert_eq!((e.params.spawn_first, e.params.spawn_count), (0, 5));
        assert_eq!(s.gpu_states().copied().collect::<Vec<_>>(), vec![3]);
    }

    #[test]
    fn remove_rejects_unknown_and_repeated_ids() {
        let mut s = ParticleSystem::with_capacity(2);
        assert_eq!(s.remove(0), Err(RemoveError::OutOfRange));
        let id = s.insert(record_at([0.0; 3]), ()).unwrap();
        s.remove(id).unwrap();
        assert_eq!(s.remove(id), Err(RemoveError::AlreadyRemoved));
    }

    #[test]
    fn the_cap_counts_live_emitters() {
        let mut s = ParticleSystem::new();
        let ids: Vec<usize> = (0..MAX_EMITTERS)
            .map(|_| s.insert(record_at([0.0; 3]), ()).unwrap())
            .collect();
        assert_eq!(s.insert(record_at([0.0; 3]), ()), Err(AtCapacity));
        s.remove(ids[7]).unwrap();
        assert!(s.has_room());
        assert_eq!(s.insert(record_at([0.0; 3]), ()), Ok(ids[7]));
    }

    #[test]
    fn clear_drops_every_emitter() {
        let mut s = ParticleSystem::with_capacity(2);
        s.insert(record_at([0.0; 3]), ()).unwrap();
        s.clear();
        assert_eq!(s.prepare(1.0), None);
    }
}
