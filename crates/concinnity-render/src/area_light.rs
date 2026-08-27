//! Packs authored `RectAreaLight`s into the per-scene `AreaLightData` table the
//! forward pass reads alongside the `GpuLight` buffer.
//!
//! The GpuLight record carries the panel's centre (`position`), emitting
//! direction (`direction`), colour, intensity, and range; only the two in-plane
//! edge vectors and the sidedness flag need the parallel table, indexed by
//! `GpuLight.data_index`. The edge vectors are pre-scaled by the half-extents, so
//! the shader reconstructs the four corners as `centre +/- right +/- up` without
//! needing the sizes separately.
//!
//! The tangent frame comes from `geometry::glass_quad::plane_basis`, shared with
//! the glass-panel quad builder, so a panel and an area light with the same
//! normal agree on which way is "across".

use crate::components::RectAreaLight;
use crate::geometry::glass_quad::plane_basis;
use crate::render_types::{AreaLightData, MAX_AREA_LIGHTS};
use alloc::vec;
use alloc::vec::Vec;

// Per-rect table index: `indices[i]` is the `AreaLightData` slot rect `i` owns,
// or -1 once the table is full. The value is what `GpuLight.data_index` carries.
pub(crate) fn assign_area_light_slots(rect_lights: &[RectAreaLight]) -> Vec<i32> {
    if rect_lights.len() > MAX_AREA_LIGHTS {
        tracing::warn!(
            "GraphicsSystem: {} area lights declared; only {} are supported -- extras ignored",
            rect_lights.len(),
            MAX_AREA_LIGHTS
        );
    }
    (0..rect_lights.len())
        .map(|i| if i < MAX_AREA_LIGHTS { i as i32 } else { -1 })
        .collect()
}

// The `AreaLightData` for each assigned slot, ordered by slot index.
pub(crate) fn build_area_light_data(
    rect_lights: &[RectAreaLight],
    slots: &[i32],
) -> Vec<AreaLightData> {
    let mut out = vec![AreaLightData::ZERO; count_area_lights(slots)];
    for (light, &slot) in rect_lights.iter().zip(slots) {
        if slot >= 0 {
            out[slot as usize] = area_light_data(light);
        }
    }
    out
}

// How many slots `assign_area_light_slots` handed out.
pub(crate) fn count_area_lights(slots: &[i32]) -> usize {
    slots.iter().filter(|s| **s >= 0).count()
}

// One rect's edge vectors, pre-scaled by its half-extents. The authored normal
// is already unit length and the half-extents already positive (the
// `rect_area_light` validator guarantees both), so no re-clamping here.
fn area_light_data(light: &RectAreaLight) -> AreaLightData {
    let (tangent, bitangent) = plane_basis(light.normal);
    let hw = light.half_size[0];
    let hh = light.half_size[1];
    AreaLightData {
        right: [tangent[0] * hw, tangent[1] * hw, tangent[2] * hw],
        two_sided: u32::from(light.two_sided),
        up: [bitangent[0] * hh, bitangent[1] * hh, bitangent[2] * hh],
        _pad: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::math::vec3::{dot, length};

    fn rect(normal: [f32; 3], half_size: [f32; 2]) -> RectAreaLight {
        RectAreaLight {
            normal,
            half_size,
            ..RectAreaLight::default()
        }
    }

    #[test]
    fn slots_are_handed_out_in_declaration_order() {
        let lights = vec![rect([0.0, 0.0, 1.0], [1.0, 1.0]); 3];
        assert_eq!(assign_area_light_slots(&lights), vec![0, 1, 2]);
    }

    #[test]
    fn slots_past_the_cap_are_dropped() {
        let lights = vec![rect([0.0, 0.0, 1.0], [1.0, 1.0]); MAX_AREA_LIGHTS + 2];
        let slots = assign_area_light_slots(&lights);
        assert_eq!(count_area_lights(&slots), MAX_AREA_LIGHTS);
        assert!(slots[MAX_AREA_LIGHTS..].iter().all(|s| *s == -1));
    }

    // The edge vectors carry the half-extents, so the shader can rebuild the
    // corners without the sizes.
    #[test]
    fn edge_vectors_are_scaled_by_the_half_extents() {
        let d = area_light_data(&rect([0.0, 0.0, 1.0], [3.0, 0.5]));
        assert!((length(d.right) - 3.0).abs() < 1e-5);
        assert!((length(d.up) - 0.5).abs() < 1e-5);
    }

    // The two edges and the normal must stay mutually perpendicular, or the
    // reconstructed quad is skewed.
    #[test]
    fn the_edge_frame_stays_orthogonal_for_any_normal() {
        for n in [
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.577, 0.577, 0.577],
            [-0.3, 0.9, 0.31],
        ] {
            let len = length(n);
            let unit = [n[0] / len, n[1] / len, n[2] / len];
            let d = area_light_data(&rect(unit, [2.0, 2.0]));
            assert!(
                dot(d.right, d.up).abs() < 1e-4,
                "edges perpendicular: {n:?}"
            );
            assert!(dot(d.right, unit).abs() < 1e-4, "right in plane: {n:?}");
            assert!(dot(d.up, unit).abs() < 1e-4, "up in plane: {n:?}");
            assert!(d.right.iter().chain(&d.up).all(|v| v.is_finite()));
        }
    }

    #[test]
    fn two_sided_flag_is_carried() {
        let mut l = rect([0.0, 0.0, 1.0], [1.0, 1.0]);
        assert_eq!(area_light_data(&l).two_sided, 0);
        l.two_sided = true;
        assert_eq!(area_light_data(&l).two_sided, 1);
    }

    #[test]
    fn data_is_indexed_by_slot() {
        let lights = vec![
            rect([0.0, 0.0, 1.0], [5.0, 1.0]),
            rect([0.0, 0.0, 1.0], [1.0, 7.0]),
        ];
        let slots = assign_area_light_slots(&lights);
        let data = build_area_light_data(&lights, &slots);
        assert_eq!(data.len(), 2);
        assert!((length(data[0].right) - 5.0).abs() < 1e-5);
        assert!((length(data[1].up) - 7.0).abs() < 1e-5);
    }
}
