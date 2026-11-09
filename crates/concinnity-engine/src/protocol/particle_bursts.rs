// Particles spawned on request: any system can ask an authored emitter for a
// burst at a point this frame, and the renderer takes the frame's bursts.

use alloc::vec::Vec;

use crate::ecs::asset_id::AssetId;

/// Bursts every emitter together takes in one frame; later ones are dropped.
pub const MAX_FRAME_BURSTS: usize = 256;

/// A burst of particles one emitter spawns this frame, at `position`, sent
/// along `direction` within the emitter's cone at its speeds times `speed`,
/// on top of the velocity `carry` every one of them starts with, and dying
/// as it falls back through its `floor`, if any. With a `ring`, they start
/// around a circle of that radius across `direction` instead, each sent
/// along `direction` leaned out through its own point of the ring by
/// `flare`, as the crown of a splash rises from the rim of what made it.
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
    /// The radius of the ring the particles start around, in meters; `0`
    /// starts them all at `position`.
    pub ring: f32,
    /// How far each particle's axis leans out through its point of the ring,
    /// as a share of `direction`: `1` sends it out at 45 degrees.
    pub flare: f32,
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
        // `cos >= ALIGN`, squared so it needs no root.
        let aligned = |a: [f32; 3], b: [f32; 3]| {
            let d = dot(a, b);
            d >= 0.0 && d * d >= BURST_MERGE_ALIGN * BURST_MERGE_ALIGN * dot(a, a) * dot(b, b)
        };
        if self.emitter != other.emitter {
            return false;
        }
        let gap: [f32; 3] = core::array::from_fn(|i| self.position[i] - other.position[i]);
        dot(gap, gap) <= BURST_MERGE_REACH * BURST_MERGE_REACH
            && aligned(self.direction, other.direction)
            && match (self.floor, other.floor) {
                (Some(a), Some(b)) => aligned(a, b),
                (None, None) => true,
                _ => false,
            }
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
        self.ring += (other.ring - self.ring) * w;
        self.flare += (other.flare - self.flare) * w;
        self.count = total;
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
            ring: 0.25,
            flare: 0.5,
            floor: Some([0.0, 1.0, 0.0]),
            count,
        }
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
        near.ring = 0.65;
        assert!(bursts.push(near));
        assert_eq!(bursts.as_slice().len(), 1);
        let joined = bursts.as_slice()[0];
        assert_eq!(joined.count, 8);
        assert!((joined.position[0] - 1.3).abs() < 1e-5);
        assert!((joined.ring - 0.55).abs() < 1e-5);
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
}
