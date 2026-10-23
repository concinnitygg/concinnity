//! Particles spawned on request: any system can ask an authored emitter for a
//! burst at a point this frame, through [`ParticleBursts`]. The bursts reach
//! the simulation in the frame's rows, a small buffer every emitter's kernel
//! reads, which also carries a move of the simulated frame so the particles
//! alive across it keep their place in the world.

use crate::ecs::asset_id::AssetId;
use crate::planet::Rebase;
use crate::transform::{Mat4, mat4_mul};
use alloc::vec::Vec;
use concinnity_core::math::SaturatingFrom;

/// Bursts one emitter takes in one frame; later ones are dropped. Mirrors
/// `PARTICLE_MAX_EMITTER_BURSTS` in `particle_simulate.hlsl`.
pub const MAX_EMITTER_BURSTS: usize = 16;

/// Bursts every emitter together takes in one frame; later ones are dropped.
pub const MAX_FRAME_BURSTS: usize = 256;

/// Rows ahead of the bursts in the frame's rows: the frame's move, three rows
/// of `[R | t]`, then `[moved, _, _, _]`. Mirrors `PARTICLE_FRAME_HEADER` in
/// `particle_simulate.hlsl`.
pub const PARTICLE_FRAME_HEADER: usize = 4;

/// The frame's rows at most: the header, then two per burst.
pub const PARTICLE_FRAME_ROWS: usize = PARTICLE_FRAME_HEADER + 2 * MAX_FRAME_BURSTS;

/// A burst of particles one emitter spawns this frame, at `position`, sent
/// along `direction` within the emitter's cone at its speeds times `speed`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleBurst {
    /// The emitter, by its asset id.
    pub emitter: AssetId,
    /// Where the particles start, in the simulated frame.
    pub position: [f32; 3],
    /// The axis of the emitter's cone for this burst.
    pub direction: [f32; 3],
    /// What the emitter's speeds are scaled by.
    pub speed: f32,
    /// How many particles.
    pub count: u32,
}

/// The bursts systems ask for this frame, at most [`MAX_FRAME_BURSTS`]
/// until the renderer takes them, so a world nothing draws holds no more.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParticleBursts(Vec<ParticleBurst>);

impl ParticleBursts {
    /// Ask for `burst`, unless the frame is full. Returns whether it was
    /// taken.
    pub fn push(&mut self, burst: ParticleBurst) -> bool {
        if self.0.len() >= MAX_FRAME_BURSTS {
            return false;
        }
        self.0.push(burst);
        true
    }

    /// The bursts asked for so far.
    pub fn as_slice(&self) -> &[ParticleBurst] {
        &self.0
    }

    /// Move every burst asked for onto the end of `out`.
    pub fn take_into(&mut self, out: &mut Vec<ParticleBurst>) {
        out.append(&mut self.0);
    }
}

/// One emitter's share of the frame's bursts: `count` bursts from burst
/// `first` of the frame's rows, which spawn `spawns` particles.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EmitterBursts {
    /// The emitter's first burst.
    pub first: u32,
    /// How many bursts it has.
    pub count: u32,
    /// The particles they spawn together.
    pub spawns: u32,
}

/// The frame's rows: the header with the frame's move, then each burst's
/// `[position, count]` and `[direction, speed]`.
#[derive(Debug, Clone)]
pub(super) struct FrameRows {
    rows: Vec<[f32; 4]>,
    // The move since the last frame, column-major.
    moved: Option<Mat4>,
}

impl Default for FrameRows {
    fn default() -> Self {
        Self {
            rows: Vec::with_capacity(PARTICLE_FRAME_ROWS),
            moved: None,
        }
    }
}

impl FrameRows {
    /// Take the move of the simulated frame `rebase`, after any taken since
    /// the last frame.
    pub(super) fn rebase(&mut self, rebase: &Rebase) {
        let m = rebase.matrix();
        self.moved = Some(match self.moved {
            Some(before) => mat4_mul(m, before),
            None => m,
        });
    }

    /// Start the frame's rows with the move taken since the last frame.
    pub(super) fn begin(&mut self) {
        self.rows.clear();
        let m = self.moved.take();
        let mut header = [[0.0f32; 4]; PARTICLE_FRAME_HEADER];
        for (i, row) in header.iter_mut().take(3).enumerate() {
            *row = match &m {
                Some(m) => [m[0][i], m[1][i], m[2][i], m[3][i]],
                None => core::array::from_fn(|c| if c == i { 1.0 } else { 0.0 }),
            };
        }
        header[3][0] = if m.is_some() { 1.0 } else { 0.0 };
        self.rows.extend_from_slice(&header);
    }

    /// Bursts written so far.
    pub(super) fn bursts(&self) -> usize {
        (self.rows.len() - PARTICLE_FRAME_HEADER) / 2
    }

    /// Write `burst` with `count` particles, if the frame has room.
    pub(super) fn push(&mut self, burst: &ParticleBurst, count: u32) -> bool {
        if self.bursts() >= MAX_FRAME_BURSTS {
            return false;
        }
        let [x, y, z] = burst.position;
        let [dx, dy, dz] = burst.direction;
        self.rows.push([x, y, z, count as f32]);
        self.rows.push([dx, dy, dz, burst.speed]);
        true
    }

    /// Cut burst `index`'s count to `count`.
    pub(super) fn set_count(&mut self, index: usize, count: u32) {
        if let Some(row) = self.rows.get_mut(PARTICLE_FRAME_HEADER + 2 * index) {
            row[3] = count as f32;
        }
    }

    /// Burst `index`'s count.
    pub(super) fn count(&self, index: usize) -> u32 {
        self.rows
            .get(PARTICLE_FRAME_HEADER + 2 * index)
            .map_or(0, |row| u32::saturating_from(row[3]))
    }

    /// The rows, header first.
    pub(super) fn rows(&self) -> &[[f32; 4]] {
        &self.rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn burst(count: u32) -> ParticleBurst {
        ParticleBurst {
            emitter: AssetId(7),
            position: [1.0, 2.0, 3.0],
            direction: [0.0, 1.0, 0.0],
            speed: 1.5,
            count,
        }
    }

    // A still frame's header leaves every particle where it is; a moved one
    // carries the move, two moves in a row composed.
    #[test]
    fn the_header_carries_the_frames_move() {
        let mut rows = FrameRows::default();
        rows.begin();
        assert_eq!(rows.rows()[0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(rows.rows()[2], [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(rows.rows()[3][0], 0.0);
        let step = Rebase {
            rotation: [0.0, 0.0, 0.0, 1.0],
            translation: [-10.0, 0.0, 2.0],
        };
        rows.rebase(&step);
        rows.rebase(&step);
        rows.begin();
        assert_eq!(rows.rows()[0], [1.0, 0.0, 0.0, -20.0]);
        assert_eq!(rows.rows()[2], [0.0, 0.0, 1.0, 4.0]);
        assert_eq!(rows.rows()[3][0], 1.0);
        rows.begin();
        assert_eq!(rows.rows()[3][0], 0.0);
    }

    // Bursts nothing takes stop at the frame's cap.
    #[test]
    fn untaken_bursts_stop_at_the_cap() {
        let mut bursts = ParticleBursts::default();
        for _ in 0..MAX_FRAME_BURSTS {
            assert!(bursts.push(burst(1)));
        }
        assert!(!bursts.push(burst(1)));
        assert_eq!(bursts.as_slice().len(), MAX_FRAME_BURSTS);
        let mut out = Vec::new();
        bursts.take_into(&mut out);
        assert_eq!(out.len(), MAX_FRAME_BURSTS);
        assert!(bursts.as_slice().is_empty());
        assert!(bursts.push(burst(1)));
    }

    // The frame holds at most `MAX_FRAME_BURSTS`, and a count can be cut.
    #[test]
    fn the_frame_holds_its_cap_of_bursts() {
        let mut rows = FrameRows::default();
        rows.begin();
        for _ in 0..MAX_FRAME_BURSTS {
            assert!(rows.push(&burst(3), 3));
        }
        assert!(!rows.push(&burst(3), 3));
        assert_eq!(rows.rows().len(), PARTICLE_FRAME_ROWS);
        assert_eq!(rows.rows()[PARTICLE_FRAME_HEADER], [1.0, 2.0, 3.0, 3.0]);
        assert_eq!(rows.rows()[PARTICLE_FRAME_HEADER + 1], [0.0, 1.0, 0.0, 1.5]);
        rows.set_count(0, 1);
        assert_eq!(rows.count(0), 1);
        assert_eq!(rows.count(MAX_FRAME_BURSTS), 0);
    }
}
