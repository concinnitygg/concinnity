//! The sun's shadow a prop reflection probe's faces shade with: the near
//! cascades and the far shadow past them. A face is a few hundred texels
//! across, so past a short distance a cascade's detail is finer than its
//! pixels, while every cascade the faces read must hold the casters around the
//! probe. Reading only the near ones leaves the rest cut to the camera's view.

use crate::gfx::render_types::ShadowUniforms;
use crate::render::shadow::csm::{cascade_fade_start, cascades_sampled_within};

/// Which sun shadow an off-camera face shades with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceShadow {
    /// The frame's: every cascade, and the far shadow past them.
    Frame,
    /// The prop reflection probe's: the near cascades, and the far shadow
    /// past them ([`probe_shadow_uniforms`]).
    PropProbe,
}

/// How far from a prop reflection probe its faces take the sun's cascades, in
/// meters: past it they take the far shadow, or none.
pub const PROP_PROBE_SHADOW_DISTANCE: f32 = 24.0;

/// The shadow block a prop probe's faces shade with, and how far from the
/// probe its cascades reach: those of `uniforms` sampled within
/// [`PROP_PROBE_SHADOW_DISTANCE`], fading over the last of them into the far
/// shadow as the full set fades at its reach. All of them when they already
/// end within it, or when there is no far shadow to take over past it.
pub fn probe_shadow_uniforms(uniforms: &ShadowUniforms) -> (ShadowUniforms, f32) {
    let near = cascades_sampled_within(uniforms, PROP_PROBE_SHADOW_DISTANCE);
    let count = near.count_ones().max(1);
    let far_shadow = uniforms.far.center[3] > 0.0;
    if count >= uniforms.active_cascades || !far_shadow {
        let last = (uniforms.active_cascades as usize).saturating_sub(1);
        return (*uniforms, uniforms.cascade_splits[last.min(3)]);
    }
    let reach = uniforms.cascade_splits[count as usize - 1];
    let probe = ShadowUniforms {
        active_cascades: count,
        fade_start: cascade_fade_start(reach),
        ..*uniforms
    };
    (probe, reach)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::projection::look_at;
    use crate::gfx::render_types::FarShadowLanes;
    use crate::render::shadow::csm::{ShadowUniformInputs, compute_shadow_uniforms};

    // The cascades out to `reach`, with a far shadow past them.
    fn uniforms(reach: f32) -> ShadowUniforms {
        ShadowUniforms {
            far: FarShadowLanes {
                center: [0.0, 0.0, 0.0, 4000.0],
                ..FarShadowLanes::OFF
            },
            ..cascades(reach)
        }
    }

    // The cascades out to `reach` alone.
    fn cascades(reach: f32) -> ShadowUniforms {
        compute_shadow_uniforms(ShadowUniformInputs {
            view: look_at([0.0, 1.6, 0.0], [0.0, 1.4, -1.0], [0.0, 1.0, 0.0]),
            cam_pos: [0.0, 1.6, 0.0],
            fov_y_rad: 70.0_f32.to_radians(),
            aspect: 16.0 / 9.0,
            near: 0.05,
            shadow_distance: reach,
            light_dir_to_source: [0.3, 1.0, 0.2],
            shadow_map_size: 2048,
            active_cascades: 4,
        })
    }

    // At the planet's 250 m the probe takes the cascades sampled within 24 m,
    // fading out over the last of them, and reaches no farther than that one.
    #[test]
    fn a_probe_takes_the_near_cascades_and_fades_at_their_end() {
        let full = uniforms(250.0);
        let (probe, reach) = probe_shadow_uniforms(&full);
        assert!(probe.active_cascades < full.active_cascades);
        let last = probe.active_cascades as usize - 1;
        assert_eq!(reach, full.cascade_splits[last]);
        assert!(reach >= PROP_PROBE_SHADOW_DISTANCE * 0.5);
        assert!(probe.fade_start < reach && probe.fade_start > 0.8 * reach);
        assert_eq!(probe.light_vps, full.light_vps);
        assert_eq!(probe.cascade_splits, full.cascade_splits);
        assert_eq!(probe.far, full.far);
        assert_eq!(
            cascades_sampled_within(&full, PROP_PROBE_SHADOW_DISTANCE),
            (1 << probe.active_cascades) - 1
        );
    }

    // Without a far shadow nothing takes over past the near cascades, so the
    // probe keeps every cascade.
    #[test]
    fn without_a_far_shadow_the_probe_keeps_every_cascade() {
        let full = cascades(250.0);
        assert!(cascades_sampled_within(&full, PROP_PROBE_SHADOW_DISTANCE).count_ones() < 4);
        let (probe, reach) = probe_shadow_uniforms(&full);
        assert_eq!(bytemuck::bytes_of(&probe), bytemuck::bytes_of(&full));
        assert_eq!(reach, full.cascade_splits[3]);
    }

    // Cascades that already end within the distance are all the probe's.
    #[test]
    fn a_short_reach_leaves_the_probe_every_cascade() {
        let full = uniforms(20.0);
        let (probe, reach) = probe_shadow_uniforms(&full);
        assert_eq!(bytemuck::bytes_of(&probe), bytemuck::bytes_of(&full));
        assert_eq!(reach, full.cascade_splits[3]);
    }
}
