//! The participating media the cascades shade along view rays: fog from the
//! camera outward, and raymarched volumes where they lie.

use crate::gfx::frustum::{Aabb, Frustum};
use crate::render::environment::volumetric_fog::FogSettings;
use concinnity_core::math::exp;
use concinnity_core::math::vec3::length;

/// The optical depth a medium must reach within the cascades' reach for
/// their shadow in it to show: a percent of the light it scatters.
const SHADOWED_DEPTH: f32 = 0.01;

/// How deep below its reference height the fog keeps thickening, in meters,
/// as `fog.hlsl` clamps it.
const FOG_FLOOR: f32 = 50.0;

/// Whether `fog`, seen from a camera at height `eye_y`, is thick enough within
/// `reach` meters of it for the cascades' shadow in it to show: its densest
/// part within reach, across that reach.
pub(super) fn fog_shadowed(fog: &FogSettings, eye_y: f32, reach: f32) -> bool {
    let lowest = eye_y - reach - fog.height_reference;
    let densest = fog.density * exp(-lowest.max(-FOG_FLOOR) * fog.height_falloff);
    densest * reach >= SHADOWED_DEPTH
}

/// How far from a camera at `eye` seeing through `frustum` the nearest medium
/// the cascades reaching `reach` meters shade lies: `fog` at the camera when
/// thick enough, a raymarched volume within `volumes` at its distance when in
/// view; `None` for neither.
pub fn shadowed_media(
    fog: Option<&FogSettings>,
    volumes: Option<Aabb>,
    eye: [f32; 3],
    frustum: &Frustum,
    reach: f32,
) -> Option<f32> {
    let fog = fog
        .filter(|f| f.contributes() && fog_shadowed(f, eye[1], reach))
        .map(|_| 0.0_f32);
    let volume = volumes
        .filter(|(lo, hi)| frustum.intersects_aabb(*lo, *hi))
        .map(|(lo, hi)| {
            let gap: [f32; 3] =
                core::array::from_fn(|k| (lo[k] - eye[k]).max(eye[k] - hi[k]).max(0.0));
            length(gap)
        });
    match (fog, volume) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::depth::camera_projection;
    use crate::transform::mat4_mul;

    fn fog(density: f32, height_reference: f32) -> FogSettings {
        FogSettings {
            color: [1.0; 3],
            density,
            height_falloff: 0.1,
            height_reference,
            max_distance: 500.0,
            phase_g: 0.0,
            ambient: 0.0,
        }
    }

    // A camera at the origin looking down -z.
    fn frustum() -> Frustum {
        let view = crate::transform::IDENTITY;
        Frustum::from_camera(mat4_mul(camera_projection(1.0, 1.0, 0.1), view), None)
    }

    #[test]
    fn fog_counts_while_its_densest_part_within_reach_scatters_a_percent() {
        // Thickening under the camera to 0.15 per meter within reach.
        assert!(fog_shadowed(&fog(0.001, 0.0), 0.0, 100.0));
        // So thin that its densest part within reach scatters a thousandth.
        assert!(!fog_shadowed(&fog(1e-7, 0.0), 0.0, 100.0));
        // Far above its layer, the fog under the camera is gone.
        assert!(!fog_shadowed(&fog(0.001, 0.0), 2000.0, 100.0));
        // Reaching down into it brings it back.
        assert!(fog_shadowed(&fog(0.001, 0.0), 2000.0, 2100.0));
    }

    #[test]
    fn the_nearest_shaded_medium_is_fog_at_the_camera_or_a_volume_in_view() {
        let reach = 100.0;
        let thick = fog(0.01, 0.0);
        let ahead = Some(([-1.0, -1.0, -21.0], [1.0, 1.0, -19.0]));
        let behind = Some(([-1.0, -1.0, 19.0], [1.0, 1.0, 21.0]));
        let f = frustum();
        assert_eq!(shadowed_media(None, None, [0.0; 3], &f, reach), None);
        assert_eq!(
            shadowed_media(Some(&thick), ahead, [0.0; 3], &f, reach),
            Some(0.0)
        );
        let volume = shadowed_media(None, ahead, [0.0; 3], &f, reach).unwrap();
        assert!((volume - 19.0).abs() < 1e-4, "{volume}");
        assert_eq!(shadowed_media(None, behind, [0.0; 3], &f, reach), None);
        let clear = FogSettings {
            density: 0.0,
            ..thick
        };
        assert_eq!(
            shadowed_media(Some(&clear), None, [0.0; 3], &f, reach),
            None
        );
    }
}
