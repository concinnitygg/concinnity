//! The live light block a backend draws with: the `LightUniforms` its passes
//! bind, which frame-in-flight copies of it are stale, and the local lights'
//! reach the cluster grid is sliced to. Every live light change goes through
//! here, so the copies are re-armed the same way on every backend.

use core::cell::Cell;

use crate::components::DirectionalLight;
use crate::gfx::render_types::{GpuLight, LightUniforms};
use crate::render::cluster_range::ClusterReach;
use crate::render::frame_dirty::FrameDirty;
use crate::render::lights::{directional_light_data, sun_color, sun_direction};

/// The first directional light after a change, which the shadow cascades and
/// any fog sun follow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sun {
    /// Direction toward the light, or the fallback when there is none.
    pub direction: [f32; 3],
    /// Radiance of the light, or black when there is none.
    pub color: [f32; 3],
}

impl Sun {
    fn of(uniforms: &LightUniforms) -> Self {
        Self {
            direction: sun_direction(uniforms),
            color: sun_color(uniforms),
        }
    }
}

/// The live light block and its frames-in-flight bookkeeping.
pub struct LightState {
    uniforms: LightUniforms,
    sun: Sun,
    dirty: Cell<FrameDirty>,
    cluster_reach: ClusterReach,
}

impl LightState {
    /// The block `uniforms` over the packed `local_lights`, every one of
    /// `frames` copies stale so the ring seeds itself.
    pub fn new(uniforms: LightUniforms, local_lights: &[GpuLight], frames: usize) -> Self {
        Self {
            sun: Sun::of(&uniforms),
            uniforms,
            dirty: Cell::new(FrameDirty::new(frames)),
            cluster_reach: ClusterReach::new(local_lights),
        }
    }

    /// The block the passes bind.
    pub fn uniforms(&self) -> &LightUniforms {
        &self.uniforms
    }

    /// The local lights' reach, which places the far end of the cluster grid.
    pub fn cluster_reach(&self) -> &ClusterReach {
        &self.cluster_reach
    }

    /// The first directional light as the block holds it now, which the
    /// shadow cascades, the fog and every reflection trace light with.
    pub fn sun(&self) -> Sun {
        self.sun
    }

    /// Set the ambient (IBL) scale. Edge-triggered: an unchanged value
    /// re-arms nothing.
    pub fn set_ambient(&mut self, value: f32) {
        if self.uniforms.ambient_intensity != value {
            self.uniforms.ambient_intensity = value;
            self.mark_all();
        }
    }

    /// Replace the directional lights, returning the new sun when the set
    /// changed. An unchanged set touches nothing.
    pub fn set_directional(&mut self, lights: &[DirectionalLight]) -> Option<Sun> {
        let (directional, num_directional) = directional_light_data(lights);
        if self.uniforms.directional == directional
            && self.uniforms.num_directional == num_directional
        {
            return None;
        }
        self.uniforms.directional = directional;
        self.uniforms.num_directional = num_directional;
        self.sun = Sun::of(&self.uniforms);
        self.mark_all();
        Some(self.sun)
    }

    /// Whether `local_lights` can move in place: the packed count is the one
    /// the block was built with.
    pub fn same_local_count(&self, local_lights: &[GpuLight]) -> bool {
        local_lights.len() as i32 == self.uniforms.num_local_lights
    }

    /// Take moved local lights: their reach, and the point array of
    /// `uniforms`.
    pub fn move_local(&mut self, local_lights: &[GpuLight], uniforms: &LightUniforms) {
        self.cluster_reach = ClusterReach::new(local_lights);
        self.uniforms.point = uniforms.point;
        self.uniforms.num_point = uniforms.num_point;
        self.mark_all();
    }

    /// Whether frame `frame`'s copy is stale, clearing the flag if so.
    pub fn take_dirty(&self, frame: usize) -> bool {
        let mut dirty = self.dirty.get();
        let pending = dirty.take(frame);
        self.dirty.set(dirty);
        pending
    }

    fn mark_all(&self) {
        let mut dirty = self.dirty.get();
        dirty.mark_all();
        self.dirty.set(dirty);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(frames: usize) -> LightState {
        let s = LightState::new(LightUniforms::DEFAULT, &[], frames);
        for f in 0..frames {
            assert!(s.take_dirty(f), "frame {f} seeds stale");
        }
        s
    }

    fn sun_light(direction: [f32; 3]) -> DirectionalLight {
        DirectionalLight {
            direction,
            ..Default::default()
        }
    }

    #[test]
    fn an_unchanged_ambient_rearms_nothing() {
        let mut s = state(2);
        let current = s.uniforms().ambient_intensity;
        s.set_ambient(current);
        assert!(!s.take_dirty(0));
        s.set_ambient(current + 1.0);
        assert!(s.take_dirty(0) && s.take_dirty(1));
        assert!(!s.take_dirty(0));
        assert_eq!(s.uniforms().ambient_intensity, current + 1.0);
    }

    #[test]
    fn a_new_sun_is_returned_once() {
        let mut s = state(2);
        let lights = [sun_light([0.0, -1.0, 0.0])];
        let sun = s.set_directional(&lights).expect("a changed set");
        assert_eq!(sun, s.sun());
        assert_eq!(s.uniforms().num_directional, 1);
        assert!(s.take_dirty(0) && s.take_dirty(1));
        assert_eq!(s.set_directional(&lights), None);
        assert!(!s.take_dirty(0) && !s.take_dirty(1));
    }

    #[test]
    fn moved_lights_take_the_point_array_and_rearm() {
        let mut s = state(1);
        let mut moved = LightUniforms::DEFAULT;
        moved.num_point = 3;
        moved.num_directional += 1;
        s.move_local(&[], &moved);
        assert_eq!(s.uniforms().num_point, 3);
        assert_eq!(
            s.uniforms().num_directional,
            LightUniforms::DEFAULT.num_directional,
            "only the point array moves"
        );
        assert!(s.take_dirty(0));
        assert!(s.same_local_count(&[]));
    }
}
