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

/// Rows each burst takes in the frame's rows. Mirrors `PARTICLE_BURST_ROWS`
/// in `particle_simulate.hlsl`.
pub const PARTICLE_BURST_ROWS: usize = 4;

/// The frame's rows at most: the header, then each burst's.
pub const PARTICLE_FRAME_ROWS: usize =
    PARTICLE_FRAME_HEADER + PARTICLE_BURST_ROWS * MAX_FRAME_BURSTS;

/// A burst of particles one emitter spawns this frame, at `position`, sent
/// along `direction` within the emitter's cone at its speeds times `speed`,
/// on top of the velocity `carry` every one of them starts with, and dying
/// as it falls back through its `floor`, if any.
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
    /// The velocity every particle starts with on top of its own, in meters
    /// per second: what it keeps of the motion of whatever threw it.
    pub carry: [f32; 3],
    /// An up direction: when set, each particle dies as it falls back through
    /// the plane through `position` across it, as droplets do landing on
    /// water.
    pub floor: Option<[f32; 3]>,
    /// How many particles.
    pub count: u32,
}

/// The bursts systems ask for this frame, at most [`MAX_FRAME_BURSTS`]
/// until the renderer takes them, so a world nothing draws holds no more.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParticleBursts(Vec<ParticleBurst>);

/// How near a burst must start to one already asked of the same emitter to
/// join it, in meters.
pub const BURST_MERGE_REACH: f32 = 0.5;

// How closely two bursts' axes and floors must agree for them to join.
const BURST_MERGE_ALIGN: f32 = 0.9;

impl ParticleBursts {
    /// Ask for `burst`, joining it to one asked of the same emitter within
    /// [`BURST_MERGE_REACH`] along much the same axis over the same floor, or
    /// else unless the frame is full. Returns whether it was taken.
    pub fn push(&mut self, burst: ParticleBurst) -> bool {
        if let Some(near) = self.0.iter_mut().find(|b| b.joins(&burst)) {
            near.join(&burst);
            return true;
        }
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

impl ParticleBurst {
    // Whether `other` may join this burst.
    fn joins(&self, other: &Self) -> bool {
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let gap: [f32; 3] = core::array::from_fn(|i| self.position[i] - other.position[i]);
        let aligned = |a: [f32; 3], b: [f32; 3]| {
            dot(a, b) >= BURST_MERGE_ALIGN * (dot(a, a) * dot(b, b)).sqrt()
        };
        let floors = match (self.floor, other.floor) {
            (Some(a), Some(b)) => aligned(a, b),
            (None, None) => true,
            _ => false,
        };
        self.emitter == other.emitter
            && dot(gap, gap) <= BURST_MERGE_REACH * BURST_MERGE_REACH
            && aligned(self.direction, other.direction)
            && floors
    }

    // Take in `other`'s particles, the starts and motions weighed by count.
    fn join(&mut self, other: &Self) {
        let total = self.count.saturating_add(other.count);
        let w = other.count as f32 / (total.max(1) as f32);
        let mix = |a: [f32; 3], b: [f32; 3]| -> [f32; 3] {
            core::array::from_fn(|i| a[i] + (b[i] - a[i]) * w)
        };
        self.position = mix(self.position, other.position);
        self.direction = mix(self.direction, other.direction);
        self.carry = mix(self.carry, other.carry);
        self.speed += (other.speed - self.speed) * w;
        self.count = total;
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
/// `[position, count]`, `[direction, speed]`, `[carry, _]` and
/// `[floor, has floor]`.
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
        self.rows.push([cx, cy, cz, 0.0]);
        let [fx, fy, fz] = burst.floor.unwrap_or_default();
        self.rows
            .push([fx, fy, fz, if burst.floor.is_some() { 1.0 } else { 0.0 }]);
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

    fn burst(count: u32) -> ParticleBurst {
        ParticleBurst {
            emitter: AssetId(7),
            position: [1.0, 2.0, 3.0],
            direction: [0.0, 1.0, 0.0],
            speed: 1.5,
            carry: [0.5, 0.0, -0.5],
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

    // Bursts nothing takes stop at the frame's cap.
    #[test]
    fn untaken_bursts_stop_at_the_cap() {
        let mut bursts = ParticleBursts::default();
        let apart = |i: usize| {
            let mut b = burst(1);
            b.position[0] = i as f32;
            b
        };
        for i in 0..MAX_FRAME_BURSTS {
            assert!(bursts.push(apart(i)));
        }
        assert!(!bursts.push(apart(MAX_FRAME_BURSTS)));
        assert_eq!(bursts.as_slice().len(), MAX_FRAME_BURSTS);
        let mut out = Vec::new();
        bursts.take_into(&mut out);
        assert_eq!(out.len(), MAX_FRAME_BURSTS);
        assert!(bursts.as_slice().is_empty());
        assert!(bursts.push(apart(0)));
    }

    // A burst near one of the same emitter's, along much the same axis,
    // joins it with the counts summed; one farther off, turned away or over
    // no floor stays apart.
    #[test]
    fn near_bursts_of_one_emitter_join() {
        let mut bursts = ParticleBursts::default();
        assert!(bursts.push(burst(2)));
        let mut near = burst(6);
        near.position[0] += 0.4;
        assert!(bursts.push(near));
        assert_eq!(bursts.as_slice().len(), 1);
        let joined = bursts.as_slice()[0];
        assert_eq!(joined.count, 8);
        assert!((joined.position[0] - 1.3).abs() < 1e-5);
        let mut far = burst(1);
        far.position[0] += 1.0;
        let mut turned = burst(1);
        turned.direction = [1.0, 0.0, 0.0];
        let mut floorless = burst(1);
        floorless.floor = None;
        let mut other = burst(1);
        other.emitter = AssetId(8);
        for b in [far, turned, floorless, other] {
            assert!(bursts.push(b));
        }
        assert_eq!(bursts.as_slice().len(), 5);
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
            [0.5, 0.0, -0.5, 0.0]
        );
        assert_eq!(rows.rows()[PARTICLE_FRAME_HEADER + 3], [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(
            rows.rows()[PARTICLE_FRAME_HEADER + PARTICLE_BURST_ROWS][3],
            3.0
        );
        rows.set_count(0, 1);
        assert_eq!(rows.count(0), 1);
        assert_eq!(rows.count(MAX_FRAME_BURSTS), 0);
    }
}
