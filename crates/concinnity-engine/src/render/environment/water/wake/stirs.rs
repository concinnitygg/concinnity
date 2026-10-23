//! Pairing each floating piece with where the wake last took it in, so it can
//! push in the change: a piece seen for the first time, or one that jumped
//! farther than it could have moved, starts where it is and pushes nothing.

use super::window::WakeWindow;
use super::{STIR_SPREAD, WAKE_COUPLING, WAKE_JUMP_RADII};
use crate::planet::Rebase;
use crate::render::uniforms::wake::{MAX_WAKE_STIRS, WaterWakeStirGpu};
use crate::water::stir::{StirPiece, WaterStir};
use alloc::vec::Vec;

/// The stirs the wake has taken in, kept to pair the next frame's with.
#[derive(Debug, Clone, Default)]
pub struct StirMemory {
    // What the wake last took in from each body, in this frame's
    // coordinates, by entity.
    last: Vec<WaterStir>,
    // This frame's, built beside it.
    next: Vec<WaterStir>,
    found: Vec<Found>,
}

// One piece's pair this frame.
#[derive(Debug, Clone, Copy)]
struct Found {
    // Squared distance from the camera.
    distance: f32,
    pair: WaterWakeStirGpu,
    // The piece's stir in this frame's list, the piece, and where it was.
    stir: usize,
    piece: usize,
    prev: StirPiece,
}

impl StirMemory {
    /// Carry what it remembers into the frame the world moved to.
    pub fn rebase(&mut self, rebase: &Rebase) {
        for stir in &mut self.last {
            for piece in &mut stir.pieces {
                piece.at = rebase.apply_point(piece.at);
            }
        }
    }

    /// Fill `out` with this frame's pairs for `stirs` that reach `window`,
    /// nearest `camera` first, skipping the ones that push nothing. Returns
    /// how many it wrote. A pair past the cap keeps its piece where the wake
    /// last took it in, so its push comes whole once there is room. A body
    /// gone from `stirs` is forgotten; the waves it made die out with the
    /// rest.
    pub fn gather(
        &mut self,
        stirs: &[WaterStir],
        window: &WakeWindow,
        camera: [f32; 3],
        out: &mut [WaterWakeStirGpu; MAX_WAKE_STIRS],
    ) -> usize {
        self.found.clear();
        for (s, stir) in stirs.iter().enumerate() {
            let last = self
                .last
                .binary_search_by_key(&stir.entity, |l| l.entity)
                .ok()
                .and_then(|i| self.last.get(i))
                .filter(|l| l.count == stir.count);
            for (i, now) in stir.pieces().iter().enumerate() {
                let prev = last.and_then(|l| l.pieces.get(i)).unwrap_or(now);
                let prev = if jumped(now, prev) { now } else { prev };
                if prev == now || (now.volume <= 0.0 && prev.volume <= 0.0) {
                    continue;
                }
                let reach = STIR_SPREAD * now.radius.max(prev.radius);
                if !window.touches(xz(now), reach) && !window.touches(xz(prev), reach) {
                    continue;
                }
                let d = [now.at[0] - camera[0], now.at[2] - camera[2]];
                self.found.push(Found {
                    distance: d[0] * d[0] + d[1] * d[1],
                    pair: WaterWakeStirGpu {
                        now: lane(now),
                        prev: lane(prev),
                    },
                    stir: s,
                    piece: i,
                    prev: *prev,
                });
            }
        }
        self.found
            .sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));
        let count = self.found.len().min(MAX_WAKE_STIRS);
        for (slot, found) in out.iter_mut().zip(&self.found) {
            *slot = found.pair;
        }
        self.next.clear();
        self.next.extend_from_slice(stirs);
        for dropped in self.found.iter().skip(MAX_WAKE_STIRS) {
            if let Some(piece) = self
                .next
                .get_mut(dropped.stir)
                .and_then(|s| s.pieces.get_mut(dropped.piece))
            {
                *piece = dropped.prev;
            }
        }
        self.next.sort_unstable_by_key(|s| s.entity);
        core::mem::swap(&mut self.last, &mut self.next);
        count
    }
}

// Whether a piece moved farther since last frame than a body can between
// frames: respawned, or carried across a move the frame missed.
fn jumped(now: &StirPiece, prev: &StirPiece) -> bool {
    let d = [now.at[0] - prev.at[0], now.at[2] - prev.at[2]];
    let reach = WAKE_JUMP_RADII * now.radius.max(prev.radius);
    d[0] * d[0] + d[1] * d[1] > reach * reach
}

fn xz(p: &StirPiece) -> [f32; 2] {
    [p.at[0], p.at[2]]
}

// The piece's lane, holding the share of its volume that makes waves.
fn lane(p: &StirPiece) -> [f32; 4] {
    [p.at[0], p.at[2], p.radius, p.volume * WAKE_COUPLING]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::Entity;
    use crate::physics::MAX_FLUID_PIECES;

    fn piece(x: f32, volume: f32) -> StirPiece {
        StirPiece {
            at: [x, 0.0, 0.0],
            radius: 0.5,
            volume,
            ..StirPiece::default()
        }
    }

    fn stir(id: u32, pieces: &[StirPiece]) -> WaterStir {
        let mut all = [StirPiece::default(); MAX_FLUID_PIECES];
        all[..pieces.len()].copy_from_slice(pieces);
        WaterStir {
            entity: Entity::from_bits((u64::from(id) << 32) | 1).unwrap_or_else(Entity::dangling),
            up: [0.0, 1.0, 0.0],
            velocity: [0.0; 3],
            pieces: all,
            count: pieces.len(),
        }
    }

    fn run(memory: &mut StirMemory, stirs: &[WaterStir]) -> Vec<WaterWakeStirGpu> {
        let mut out = [WaterWakeStirGpu::default(); MAX_WAKE_STIRS];
        let n = memory.gather(stirs, &WakeWindow::around([0.0; 2]), [0.0; 3], &mut out);
        out[..n].to_vec()
    }

    // A body seen for the first time pushes nothing; from then on it pushes
    // the change since last frame.
    #[test]
    fn a_new_body_starts_still() {
        let mut m = StirMemory::default();
        assert!(run(&mut m, &[stir(1, &[piece(1.0, 0.2)])]).is_empty());
        let pairs = run(&mut m, &[stir(1, &[piece(1.1, 0.25)])]);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].now, [1.1, 0.0, 0.5, 0.25 * WAKE_COUPLING]);
        assert_eq!(pairs[0].prev, [1.0, 0.0, 0.5, 0.2 * WAKE_COUPLING]);
        // Resting where it was, it pushes nothing.
        assert!(run(&mut m, &[stir(1, &[piece(1.1, 0.25)])]).is_empty());
    }

    // A body that jumps farther than a few of its radii between frames
    // starts again where it landed instead of pushing a huge pair.
    #[test]
    fn a_jump_starts_over() {
        let mut m = StirMemory::default();
        run(&mut m, &[stir(1, &[piece(0.0, 0.2)])]);
        assert!(run(&mut m, &[stir(1, &[piece(10.0, 0.2)])]).is_empty());
        assert_eq!(run(&mut m, &[stir(1, &[piece(10.5, 0.2)])]).len(), 1);
    }

    // Leaving the water pushes the volume back out; dry pieces push nothing.
    #[test]
    fn leaving_the_water_pulls_back_and_dry_pieces_rest() {
        let mut m = StirMemory::default();
        run(&mut m, &[stir(1, &[piece(0.0, 0.2), piece(0.0, 0.0)])]);
        let pairs = run(&mut m, &[stir(1, &[piece(0.1, 0.0), piece(0.1, 0.0)])]);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].now[3], 0.0);
        assert_eq!(pairs[0].prev[3], 0.2 * WAKE_COUPLING);
    }

    // The nearest pairs to the camera win when there are too many, and
    // bodies beyond the window are left out.
    #[test]
    fn the_nearest_fill_the_cap() {
        let mut m = StirMemory::default();
        let many: Vec<WaterStir> = (0..u32::try_from(MAX_WAKE_STIRS + 8).unwrap_or(u32::MAX))
            .map(|i| stir(i, &[piece(i as f32 * 0.4, 0.2)]))
            .collect();
        let far = stir(1000, &[piece(500.0, 0.2)]);
        let mut first = many.clone();
        first.push(far);
        run(&mut m, &first);
        let mut moved: Vec<WaterStir> = many
            .iter()
            .map(|s| stir(s.entity.index(), &[piece(s.pieces[0].at[0] + 0.1, 0.2)]))
            .collect();
        moved.push(stir(1000, &[piece(500.1, 0.2)]));
        moved.reverse();
        let pairs = run(&mut m, &moved);
        assert_eq!(pairs.len(), MAX_WAKE_STIRS);
        assert_eq!(pairs[0].now[0], 0.1);
        assert!(pairs.iter().all(|p| p.now[0] < 30.0));
    }

    // A body past the cap keeps its last taken-in piece, so a body that
    // enters while dropped pushes its whole mound once it gets room.
    #[test]
    fn a_dropped_body_pushes_its_change_once_it_has_room() {
        let mut m = StirMemory::default();
        let crowd = |x: f32| -> Vec<WaterStir> {
            (1..=u32::try_from(MAX_WAKE_STIRS).unwrap_or(u32::MAX))
                .map(|i| stir(i, &[piece(i as f32 * 0.3 + x, 0.2)]))
                .collect()
        };
        let far = |volume: f32| stir(5000, &[piece(25.0, volume)]);
        let mut all = crowd(0.0);
        all.push(far(0.0));
        run(&mut m, &all);
        // The crowd moves and fills the cap while the far body enters.
        let mut all = crowd(0.1);
        all.push(far(0.3));
        assert!(run(&mut m, &all).iter().all(|p| p.now[0] < 25.0));
        // Once the crowd rests, the far body pushes the whole entry.
        let mut all = crowd(0.1);
        all.push(far(0.3));
        let pairs = run(&mut m, &all);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].prev[3], 0.0);
        assert_eq!(pairs[0].now[3], 0.3 * WAKE_COUPLING);
    }

    // A move of the frame carries last frame's pieces with it, so a body at
    // rest pushes nothing across it.
    #[test]
    fn a_frame_move_carries_the_last_pieces() {
        let mut m = StirMemory::default();
        run(&mut m, &[stir(1, &[piece(1.0, 0.2)])]);
        let rebase = Rebase {
            rotation: [0.0, 0.0, 0.0, 1.0],
            translation: [-1.0, 0.0, 0.0],
        };
        m.rebase(&rebase);
        assert!(run(&mut m, &[stir(1, &[piece(0.0, 0.2)])]).is_empty());
    }
}
