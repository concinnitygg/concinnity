//! Particles spawned on request, as the simulation reads them: the bursts
//! systems ask for through
//! [`ParticleBursts`](crate::protocol::ParticleBursts) reach every emitter's kernel in
//! the frame's rows, a small buffer that also carries a move of the simulated
//! frame so the particles alive across it keep their place in the world.

use crate::environment::planet::Rebase;
use crate::protocol::{MAX_FRAME_BURSTS, ParticleBurst};
use crate::transform::{Mat4, mat4_mul};
use alloc::vec::Vec;
use concinnity_core::math::SaturatingFrom;

/// Bursts one emitter takes in one frame; later ones are dropped. Mirrors
/// `PARTICLE_MAX_EMITTER_BURSTS` in `particle_simulate.hlsl`.
pub const MAX_EMITTER_BURSTS: usize = 16;

/// Rows ahead of the bursts in the frame's rows: the frame's move, three rows
/// of `[R | t]`, then `[moved, _, _, _]`. Mirrors `PARTICLE_FRAME_HEADER` in
/// `particle_simulate.hlsl`.
pub const PARTICLE_FRAME_HEADER: usize = 4;

/// Rows each burst takes in the frame's rows. Mirrors `PARTICLE_BURST_ROWS`
/// in `particle_simulate.hlsl`.
pub const PARTICLE_BURST_ROWS: usize = 4;

/// The frame's rows at most: the header, then each burst's.
pub const PARTICLE_FRAME_ROWS: usize =
    PARTICLE_FRAME_HEADER + PARTICLE_BURST_ROWS * MAX_FRAME_BURSTS;

/// One emitter's share of the frame's bursts: `count` bursts from burst
/// `first` of the frame's rows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EmitterBursts {
    /// The emitter's first burst.
    pub first: u32,
    /// How many bursts it has.
    pub count: u32,
}

/// The frame's rows: the header with the frame's move, then each burst's
/// `[position, count]`, `[direction, speed]`, `[carry, ring]` and
/// `[floor, flare]`, the floor zero for none.
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
        (self.rows.len() - PARTICLE_FRAME_HEADER) / PARTICLE_BURST_ROWS
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
        let [cx, cy, cz] = burst.carry;
        self.rows.push([cx, cy, cz, burst.ring]);
        let [fx, fy, fz] = burst.floor.unwrap_or_default();
        self.rows.push([fx, fy, fz, burst.flare]);
        true
    }

    /// Cut burst `index`'s count to `count`.
    pub(super) fn set_count(&mut self, index: usize, count: u32) {
        if let Some(row) = self
            .rows
            .get_mut(PARTICLE_FRAME_HEADER + PARTICLE_BURST_ROWS * index)
        {
            row[3] = count as f32;
        }
    }

    /// Burst `index`'s count.
    pub(super) fn count(&self, index: usize) -> u32 {
        self.rows
            .get(PARTICLE_FRAME_HEADER + PARTICLE_BURST_ROWS * index)
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
    use crate::ecs::asset_id::AssetId;

    fn burst(count: u32) -> ParticleBurst {
        ParticleBurst {
            emitter: AssetId(7),
            position: [1.0, 2.0, 3.0],
            direction: [0.0, 1.0, 0.0],
            speed: 1.5,
            carry: [0.5, 0.0, -0.5],
            ring: 0.25,
            flare: 0.5,
            floor: Some([0.0, 1.0, 0.0]),
            count,
        }
    }

    #[test]
    fn the_constants_match_the_shader() {
        let src = crate::render::shader::embedded::PARTICLE_SIMULATE;
        let uint = |name| crate::render::shader::consts::uint(src, name);
        assert_eq!(uint("PARTICLE_FRAME_HEADER"), PARTICLE_FRAME_HEADER);
        assert_eq!(uint("PARTICLE_BURST_ROWS"), PARTICLE_BURST_ROWS);
        assert_eq!(uint("PARTICLE_MAX_EMITTER_BURSTS"), MAX_EMITTER_BURSTS);
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
        assert_eq!(
            rows.rows()[PARTICLE_FRAME_HEADER + 2],
            [0.5, 0.0, -0.5, 0.25]
        );
        assert_eq!(rows.rows()[PARTICLE_FRAME_HEADER + 3], [0.0, 1.0, 0.0, 0.5]);
        assert_eq!(
            rows.rows()[PARTICLE_FRAME_HEADER + PARTICLE_BURST_ROWS][3],
            3.0
        );
        rows.set_count(0, 1);
        assert_eq!(rows.count(0), 1);
        assert_eq!(rows.count(MAX_FRAME_BURSTS), 0);
    }
}
