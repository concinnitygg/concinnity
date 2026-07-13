// src/assets/anim_params.rs

use crate::ecs::SkinnedMeshHandle;

/// Runtime-only parameter block for one animation graph target.
///
/// `AnimationSystem` publishes one `AnimParams` per `AnimGraph` at init,
/// seeded to the graph's declared parameter defaults. Gameplay systems write
/// values into it each frame (matching on `target`); `AnimationSystem` reads
/// it back during its step and evaluates the graph's transitions against the
/// values. Entries are indexed by the graph's parameter declaration order --
/// parameter names are compiled away at build time, so nothing resolves them
/// at runtime.
///
/// Not authored in world files: it has no `args`.
#[derive(Debug, Clone)]
pub struct AnimParams {
    /// The `SkinnedMesh` resource whose graph these parameters drive.
    pub target: SkinnedMeshHandle,
    /// One value per graph parameter, in declaration order.
    pub values: Vec<f32>,
}

impl AnimParams {
    /// A parameter block for `target`, seeded with the graph's defaults.
    pub fn new(target: SkinnedMeshHandle, values: Vec<f32>) -> Self {
        Self { target, values }
    }

    /// Set one parameter by index; out-of-range writes are ignored so a
    /// stale writer cannot grow the block.
    pub fn set(&mut self, index: usize, value: f32) {
        if let Some(slot) = self.values.get_mut(index) {
            *slot = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_writes_in_range_and_ignores_out_of_range() {
        let mut p = AnimParams::new(SkinnedMeshHandle(1), vec![0.0, 1.0]);
        p.set(0, 3.5);
        assert_eq!(p.values, vec![3.5, 1.0]);
        p.set(5, 9.0);
        assert_eq!(p.values.len(), 2, "out-of-range write must not grow");
    }
}
