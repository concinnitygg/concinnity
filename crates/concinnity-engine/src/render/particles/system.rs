//! The emitter table the backends share: one slot per emitter id holding its
//! record, the cached cull bounds, the spawn ring, and the backend's own
//! per-emitter GPU state `G`. The backend keeps pipelines and encoding; this
//! owns which emitters exist, how far each has spawned, and which are seen.

use crate::gfx::frustum::Frustum;
use crate::gfx::render_types::ParticleParams;
use crate::planet::Rebase;
use crate::render::particles::burst::{FrameRows, MAX_EMITTER_BURSTS};
use crate::render::particles::{
    EmitterBursts, MAX_EMITTERS, ParticleBurst, ParticleEmitterRecord, ParticleSpawnState,
    ParticleSpawns, spawn_seed,
};
use crate::render::slots::table::{AtCapacity, RemoveError, SlotTable};
use concinnity_core::math::SaturatingFrom;

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
    bursts: EmitterBursts,
    // Whether it has ever spawned a burst, which can lie anywhere, so its
    // bounds no longer hold its particles.
    fed: bool,
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
/// # use concinnity_engine::components::ParticleEmitter;
/// # use concinnity_engine::render::particles::{ParticleEmitterRecord, ParticleSystem};
/// let record = ParticleEmitterRecord::new(&ParticleEmitter::default(), 0);
/// let mut particles = ParticleSystem::new();
/// let id = particles.insert(record, "pool").expect("room for one");
/// let frame = particles.prepare(0.5, &[]).expect("an emitter exists");
/// assert_eq!(particles.emitters(&frame).count(), 1);
/// assert_eq!(particles.remove(id), Ok("pool"));
/// ```
pub struct ParticleSystem<G> {
    emitters: SlotTable<Emitter<G>>,
    last_elapsed: f32,
    frame_index: u32,
    rows: FrameRows,
    // Whether a dropped burst has been reported.
    dropped_reported: bool,
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
            rows: FrameRows::default(),
            dropped_reported: false,
        }
    }

    /// Whether another [`Self::insert`] would succeed, checked before the
    /// backend allocates an emitter's GPU state.
    pub fn has_room(&self) -> bool {
        self.emitters.has_room()
    }

    /// Place `record` with its GPU state in a slot and return the id, reusing a
    /// tombstoned slot before growing. The emitter starts with an empty pool.
    ///
    /// # Errors
    ///
    /// Returns `AtCapacity` if the table has no free slot and cannot grow.
    pub fn insert(&mut self, record: ParticleEmitterRecord, gpu: G) -> Result<usize, AtCapacity> {
        let (aabb_min, aabb_max) = record.aabb();
        self.emitters.insert(Emitter {
            record,
            aabb_min,
            aabb_max,
            spawn_state: ParticleSpawnState::default(),
            spawns: ParticleSpawns::default(),
            bursts: EmitterBursts::default(),
            fed: false,
            gpu,
        })
    }

    /// Tombstone the emitter at `id`, handing its GPU state back so the backend
    /// can release it on its own terms.
    ///
    /// # Errors
    ///
    /// Returns an error if `id` was never handed out or is already removed.
    pub fn remove(&mut self, id: usize) -> Result<G, RemoveError> {
        self.emitters.remove(id).map(|e| e.gpu)
    }

    /// Whether no emitter is live, so both passes can be skipped.
    pub fn is_empty(&self) -> bool {
        self.emitters.is_empty()
    }

    /// Whether a live emitter draws after the see-through surfaces.
    pub fn draws_after_transparent(&self) -> bool {
        self.emitters
            .iter()
            .any(|(_, e)| e.record.draw_after_transparent)
    }

    /// Whether no emitter has ever been inserted, tombstoned or not.
    pub fn never_used(&self) -> bool {
        self.emitters.never_used()
    }

    /// Carry every emitter, and on the next frame every live particle, into
    /// the frame the world moved to. Gravity stays as it is: a moved frame's
    /// down is down where it now lies.
    pub fn rebase(&mut self, rebase: &Rebase) {
        for (_, e) in self.emitters.iter_mut() {
            let r = &mut e.record;
            r.position = rebase.apply_point(r.position);
            r.direction = rebase.apply_vector(r.direction);
            (e.aabb_min, e.aabb_max) = r.aabb();
        }
        self.rows.rebase(rebase);
    }

    /// Advance the frame clock to `elapsed` seconds and take every live
    /// emitter's spawn run for this frame, with the `bursts` asked of it:
    /// at most [`MAX_EMITTER_BURSTS`] an emitter and
    /// [`MAX_FRAME_BURSTS`](super::burst::MAX_FRAME_BURSTS) a frame, as far
    /// as its pool has dead slots, the rest dropped. Run exactly once per
    /// frame, before either pass encodes, so the simulation and the draw see
    /// one frame. `None` until an emitter has ever been inserted; the clock
    /// keeps running while every slot is tombstoned.
    pub fn prepare(&mut self, elapsed: f32, bursts: &[ParticleBurst]) -> Option<ParticleFrame> {
        if self.emitters.never_used() {
            return None;
        }
        let dt = (elapsed - self.last_elapsed).max(0.0);
        self.last_elapsed = elapsed;
        self.frame_index = self.frame_index.wrapping_add(1);
        self.rows.begin();
        // Bursts past the caps, and particles a full pool had no room for.
        let mut capped = false;
        let mut crowded = None;
        for (_, e) in self.emitters.iter_mut() {
            let first = self.rows.bursts();
            let mut asked = 0u32;
            if let Some(id) = e.record.id {
                let mine = bursts.iter().filter(|b| b.emitter == id && b.count > 0);
                for (n, burst) in mine.enumerate() {
                    let count = burst.count.min(e.record.max_particles);
                    if n >= MAX_EMITTER_BURSTS || !self.rows.push(burst, count) {
                        capped = true;
                        break;
                    }
                    asked = asked.saturating_add(count);
                }
            }
            let count = self.rows.bursts() - first;
            e.spawns = e.spawn_state.take_spawns(dt, &e.record, asked);
            // The bursts go first: the rate's spawns give way to them, and
            // those the pool has no room for are cut from the last.
            let mut room = e.spawns.count;
            for i in first..first + count {
                let n = self.rows.count(i).min(room);
                if n < self.rows.count(i) {
                    crowded = Some(e.record.max_particles);
                }
                self.rows.set_count(i, n);
                room -= n;
            }
            e.bursts = EmitterBursts {
                first: u32::saturating_from(first),
                count: u32::saturating_from(count),
                spawns: e.spawns.count - room,
            };
            e.fed |= count > 0;
        }
        if (capped || crowded.is_some()) && !core::mem::replace(&mut self.dropped_reported, true) {
            if capped {
                tracing::warn!(
                    "particle bursts dropped: an emitter takes {MAX_EMITTER_BURSTS} a frame and \
                     the frame {}",
                    super::burst::MAX_FRAME_BURSTS
                );
            }
            if let Some(pool) = crowded {
                tracing::warn!(
                    "particle bursts cut: an emitter's pool of {pool} had no room for them"
                );
            }
        }
        Some(ParticleFrame {
            dt,
            frame_index: self.frame_index,
        })
    }

    /// The frame's rows [`Self::prepare`] wrote, which every emitter's
    /// simulation reads: the frame's move, then the bursts.
    pub fn rows(&self) -> &[[f32; 4]] {
        self.rows.rows()
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

    /// The live emitters that draw before the see-through surfaces, or after
    /// them for `after_transparent`, whose spawn bounds meet `frustum`, and
    /// those that spawn bursts, which can lie anywhere, in id order. Each
    /// is tested once: a caller that needs to know whether the draw runs at all
    /// peeks this iterator rather than counting it first.
    pub fn visible<'a>(
        &'a self,
        frame: &'a ParticleFrame,
        frustum: &'a Frustum,
        after_transparent: bool,
    ) -> impl Iterator<Item = ActiveEmitter<'a, G>> + 'a {
        self.emitters
            .iter()
            .filter(move |(_, e)| e.record.draw_after_transparent == after_transparent)
            .filter(|(_, e)| e.fed || frustum.intersects_aabb(e.aabb_min, e.aabb_max))
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
            params: e.record.params(
                frame.dt,
                e.spawns,
                e.bursts,
                spawn_seed(frame.frame_index, id),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::ParticleEmitter;
    use crate::ecs::asset_id::AssetId;
    use crate::gfx::frustum::Plane;
    use crate::render::particles::burst::{
        PARTICLE_BURST_ROWS, PARTICLE_FRAME_HEADER, PARTICLE_FRAME_ROWS,
    };
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
        assert_eq!(s.prepare(1.0, &[]), None);
    }

    #[test]
    fn prepare_steps_the_clock_and_the_frame_counter() {
        let mut s = ParticleSystem::with_capacity(4);
        s.insert(record_at([0.0; 3]), ()).unwrap();
        let a = s.prepare(0.25, &[]).unwrap();
        assert_eq!((a.dt, a.frame_index), (0.25, 1));
        let b = s.prepare(0.75, &[]).unwrap();
        assert_eq!((b.dt, b.frame_index), (0.5, 2));
        // A clock that went backwards steps by zero.
        assert_eq!(s.prepare(0.5, &[]).unwrap().dt, 0.0);
    }

    #[test]
    fn the_clock_keeps_running_while_every_emitter_is_removed() {
        let mut s = ParticleSystem::with_capacity(4);
        let id = s.insert(record_at([0.0; 3]), ()).unwrap();
        s.prepare(1.0, &[]);
        s.remove(id).unwrap();
        assert!(s.is_empty());
        assert!(!s.never_used());
        assert_eq!(s.prepare(2.0, &[]).unwrap().frame_index, 2);
        s.insert(record_at([0.0; 3]), ()).unwrap();
        assert_eq!(s.prepare(2.5, &[]).unwrap().dt, 0.5);
    }

    #[test]
    fn the_uniform_carries_the_spawn_run_and_seed() {
        let mut s = ParticleSystem::with_capacity(4);
        let rec = record_at([0.0; 3]);
        s.insert(rec, ()).unwrap();
        s.insert(rec, ()).unwrap();
        let frame = s.prepare(0.5, &[]).unwrap();
        let all: Vec<_> = s.emitters(&frame).collect();
        assert_eq!(all.len(), 2);
        for e in &all {
            assert_eq!(e.params.dt, 0.5);
            assert_eq!(e.params.spawn_count, 5);
            assert_eq!(e.params.spawn_first, 0);
            assert_eq!(e.params.random_seed, spawn_seed(1, e.id));
        }
        // The next frame continues each emitter's own ring.
        let frame = s.prepare(1.0, &[]).unwrap();
        let first: Vec<u32> = s.emitters(&frame).map(|e| e.params.spawn_first).collect();
        assert_eq!(first, vec![5, 5]);
    }

    #[test]
    fn only_emitters_meeting_the_frustum_are_visible() {
        let mut s = ParticleSystem::with_capacity(4);
        let near = s.insert(record_at([0.0; 3]), 'n').unwrap();
        s.insert(record_at([100.0, 0.0, 0.0]), 'f').unwrap();
        let frame = s.prepare(0.1, &[]).unwrap();
        let seen: Vec<(usize, char)> = s
            .visible(&frame, &frustum_left_of_ten(), false)
            .map(|e| (e.id, *e.gpu))
            .collect();
        assert_eq!(seen, vec![(near, 'n')]);
        // The culled emitter still simulates.
        assert_eq!(s.emitters(&frame).count(), 2);
    }

    // Each draw stage sees only its own emitters; both simulate.
    #[test]
    fn each_draw_stage_draws_its_own_emitters() {
        let mut s = ParticleSystem::with_capacity(4);
        let before = s.insert(record_at([0.0; 3]), 'b').unwrap();
        assert!(!s.draws_after_transparent());
        let late = ParticleEmitterRecord {
            draw_after_transparent: true,
            ..record_at([0.0; 3])
        };
        let after = s.insert(late, 'a').unwrap();
        assert!(s.draws_after_transparent());
        let frame = s.prepare(0.1, &[]).unwrap();
        let ids = |late| -> Vec<usize> {
            s.visible(&frame, &frustum_left_of_ten(), late)
                .map(|e| e.id)
                .collect()
        };
        assert_eq!(ids(false), vec![before]);
        assert_eq!(ids(true), vec![after]);
        assert_eq!(s.emitters(&frame).count(), 2);
        s.remove(after).unwrap();
        assert!(!s.draws_after_transparent());
    }

    #[test]
    fn a_reused_slot_starts_a_fresh_ring() {
        let mut s = ParticleSystem::with_capacity(1);
        let id = s.insert(record_at([0.0; 3]), 1).unwrap();
        s.prepare(1.0, &[]);
        assert!(!s.has_room());
        assert_eq!(s.insert(record_at([0.0; 3]), 2), Err(AtCapacity));
        assert_eq!(s.remove(id), Ok(1));
        assert_eq!(s.insert(record_at([0.0; 3]), 3), Ok(id));
        let frame = s.prepare(1.5, &[]).unwrap();
        let e = s.emitters(&frame).next().unwrap();
        assert_eq!((e.params.spawn_first, e.params.spawn_count), (0, 5));
        assert_eq!(s.gpu_states().copied().collect::<Vec<_>>(), vec![3]);
    }

    fn burst_for(emitter: AssetId, x: f32, count: u32) -> ParticleBurst {
        ParticleBurst {
            emitter,
            position: [x, 0.0, 0.0],
            direction: [0.0, 1.0, 0.0],
            speed: 1.0,
            carry: [0.0; 3],
            floor: None,
            count,
        }
    }

    fn named(position: [f32; 3], id: u32, spawn_rate: f32) -> ParticleEmitterRecord {
        ParticleEmitterRecord {
            id: Some(AssetId(id)),
            spawn_rate,
            ..record_at(position)
        }
    }

    // A burst goes to the emitter it names, ahead of that emitter's rate,
    // and shows in its uniform; an emitter no burst names has none.
    #[test]
    fn bursts_reach_the_emitter_they_name() {
        let mut s = ParticleSystem::with_capacity(4);
        let spray = s.insert(named([0.0; 3], 9, 0.0), ()).unwrap();
        let other = s.insert(record_at([0.0; 3]), ()).unwrap();
        let bursts = [
            burst_for(AssetId(9), 1.0, 3),
            burst_for(AssetId(4), 2.0, 5),
            burst_for(AssetId(9), 3.0, 2),
        ];
        let frame = s.prepare(0.5, &bursts).unwrap();
        let params: Vec<_> = s.emitters(&frame).map(|e| (e.id, e.params)).collect();
        let p = params.iter().find(|(id, _)| *id == spray).unwrap().1;
        assert_eq!([p.burst_first, p.burst_count, p.burst_spawns], [0, 2, 5]);
        assert_eq!(p.spawn_count, 5);
        let q = params.iter().find(|(id, _)| *id == other).unwrap().1;
        assert_eq!([q.burst_count, q.burst_spawns], [0, 0]);
        assert_eq!(q.spawn_count, 5);
        let rows = s.rows();
        assert_eq!(rows.len(), PARTICLE_FRAME_HEADER + 2 * PARTICLE_BURST_ROWS);
        assert_eq!(rows[PARTICLE_FRAME_HEADER], [1.0, 0.0, 0.0, 3.0]);
        assert_eq!(
            rows[PARTICLE_FRAME_HEADER + PARTICLE_BURST_ROWS],
            [3.0, 0.0, 0.0, 2.0]
        );
        // The next frame without bursts has none.
        let frame = s.prepare(1.0, &[]).unwrap();
        assert!(s.emitters(&frame).all(|e| e.params.burst_count == 0));
        assert_eq!(s.rows().len(), PARTICLE_FRAME_HEADER);
    }

    // Past the cap an emitter's bursts are dropped, and so is what its pool
    // has no room for, the last bursts first; nothing panics.
    #[test]
    fn bursts_past_the_caps_are_dropped() {
        let mut s = ParticleSystem::with_capacity(2);
        let mut rec = named([0.0; 3], 9, 0.0);
        rec.max_particles = 40;
        s.insert(rec, ()).unwrap();
        let many: Vec<_> = (0..MAX_EMITTER_BURSTS + 5)
            .map(|i| burst_for(AssetId(9), i as f32, 2))
            .collect();
        let frame = s.prepare(0.5, &many).unwrap();
        let p = s.emitters(&frame).next().unwrap().params;
        assert_eq!(p.burst_count as usize, MAX_EMITTER_BURSTS);
        assert_eq!(p.burst_spawns as usize, 2 * MAX_EMITTER_BURSTS);
        // Only 8 slots stay dead for the next frame: the last bursts lose out.
        let frame = s.prepare(0.6, &many).unwrap();
        let p = s.emitters(&frame).next().unwrap().params;
        assert_eq!(p.burst_spawns, 8);
        assert_eq!(p.spawn_count, 8);
        let counts: Vec<f32> = s.rows()[PARTICLE_FRAME_HEADER..]
            .iter()
            .step_by(PARTICLE_BURST_ROWS)
            .map(|r| r[3])
            .collect();
        assert_eq!(counts[..4], [2.0; 4]);
        assert!(counts[4..].iter().all(|c| *c == 0.0));
        // The frame as a whole holds its own cap.
        let mut wide = ParticleSystem::with_capacity(MAX_EMITTERS);
        for id in 0..20 {
            wide.insert(named([0.0; 3], id, 0.0), ()).unwrap();
        }
        let flood: Vec<_> = (0..20)
            .flat_map(|id| (0..MAX_EMITTER_BURSTS).map(move |_| burst_for(AssetId(id), 0.0, 1)))
            .collect();
        wide.prepare(0.5, &flood).unwrap();
        assert_eq!(wide.rows().len(), PARTICLE_FRAME_ROWS);
    }

    // An emitter that has spawned a burst is drawn wherever the camera looks,
    // since its particles can lie anywhere.
    #[test]
    fn a_fed_emitter_is_never_culled() {
        let mut s = ParticleSystem::with_capacity(2);
        s.insert(named([100.0, 0.0, 0.0], 9, 0.0), ()).unwrap();
        let frame = s.prepare(0.1, &[]).unwrap();
        assert_eq!(s.visible(&frame, &frustum_left_of_ten(), false).count(), 0);
        let frame = s.prepare(0.2, &[burst_for(AssetId(9), 0.0, 1)]).unwrap();
        assert_eq!(s.visible(&frame, &frustum_left_of_ten(), false).count(), 1);
        let frame = s.prepare(0.3, &[]).unwrap();
        assert_eq!(s.visible(&frame, &frustum_left_of_ten(), false).count(), 1);
    }

    // A move of the frame carries the emitters and reaches the next frame's
    // header, so the live particles follow.
    #[test]
    fn a_frame_move_carries_the_emitters_and_the_live_particles() {
        let mut s = ParticleSystem::with_capacity(2);
        s.insert(record_at([5.0, 0.0, 0.0]), ()).unwrap();
        s.rebase(&Rebase {
            rotation: [0.0, 0.0, 0.0, 1.0],
            translation: [-5.0, 0.0, 0.0],
        });
        let frame = s.prepare(0.1, &[]).unwrap();
        let e = s.emitters(&frame).next().unwrap();
        assert_eq!(e.params.position, [0.0; 3]);
        assert_eq!(s.rows()[0], [1.0, 0.0, 0.0, -5.0]);
        assert_eq!(s.rows()[3][0], 1.0);
        s.prepare(0.2, &[]);
        assert_eq!(s.rows()[3][0], 0.0);
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
        assert_eq!(s.prepare(1.0, &[]), None);
    }
}
