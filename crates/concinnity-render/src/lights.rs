// src/lights.rs
//
// Converts drained DirectionalLight and PointLight asset components into the
// GPU data the renderer consumes: the fixed LightUniforms uniform (directional
// lights, ambient, and the legacy point array the raymarch / fog / probe paths
// read) and the GpuLight storage buffer the clustered forward pass iterates.

use crate::assets::{DirectionalLight, PointLight};
use crate::render_types::{
    DirectionalLightData, GpuLight, LIGHT_KIND_POINT, LightUniforms, MAX_DIRECTIONAL_LIGHTS,
    MAX_LOCAL_LIGHTS, MAX_POINT_LIGHTS, PointLightData,
};

// Packs point lights into the per-scene GpuLight storage buffer the forward
// pass reads. Bounded by MAX_LOCAL_LIGHTS (not the 8-entry LightUniforms array);
// extras past the cap are dropped with a warning. Point lights carry no cone or
// shadow data, so those fields stay at their neutral GpuLight::ZERO values.
pub fn build_light_buffer(pt_lights: &[PointLight]) -> Vec<GpuLight> {
    if pt_lights.len() > MAX_LOCAL_LIGHTS {
        tracing::warn!(
            "GraphicsSystem: {} local lights declared; only {} are supported -- extras ignored",
            pt_lights.len(),
            MAX_LOCAL_LIGHTS
        );
    }
    pt_lights
        .iter()
        .take(MAX_LOCAL_LIGHTS)
        .map(|l| GpuLight {
            position: l.position,
            range: l.range,
            color: l.color,
            intensity: l.intensity,
            kind: LIGHT_KIND_POINT,
            ..GpuLight::ZERO
        })
        .collect()
}

pub fn build_light_uniforms(
    dir_lights: Vec<DirectionalLight>,
    pt_lights: Vec<PointLight>,
    ambient_intensity: f32,
) -> LightUniforms {
    if dir_lights.is_empty() && pt_lights.is_empty() {
        return LightUniforms {
            ambient_intensity,
            ..LightUniforms::DEFAULT
        };
    }

    const ZERO_DIR: DirectionalLightData = DirectionalLightData {
        direction: [0.0; 3],
        intensity: 0.0,
        color: [0.0; 3],
        _pad: 0.0,
    };
    const ZERO_PT: PointLightData = PointLightData {
        position: [0.0; 3],
        range: 0.0,
        color: [0.0; 3],
        intensity: 0.0,
    };

    let mut directional = [ZERO_DIR; MAX_DIRECTIONAL_LIGHTS];
    // The `point` array is the legacy subset the raymarch / fog / probe paths
    // read; the forward pass reads every light from the GpuLight buffer instead
    // (see build_light_buffer), so exceeding MAX_POINT_LIGHTS is not an error.
    let mut point = [ZERO_PT; MAX_POINT_LIGHTS];
    let num_directional = dir_lights.len().min(MAX_DIRECTIONAL_LIGHTS);
    let num_point = pt_lights.len().min(MAX_POINT_LIGHTS);
    let num_local = pt_lights.len().min(MAX_LOCAL_LIGHTS);

    if dir_lights.len() > MAX_DIRECTIONAL_LIGHTS {
        tracing::warn!(
            "GraphicsSystem: {} directional lights declared; only {} are supported -- extras ignored",
            dir_lights.len(),
            MAX_DIRECTIONAL_LIGHTS
        );
    }

    for (i, l) in dir_lights
        .into_iter()
        .take(MAX_DIRECTIONAL_LIGHTS)
        .enumerate()
    {
        directional[i] = DirectionalLightData {
            direction: l.direction,
            intensity: l.intensity,
            color: l.color,
            _pad: 0.0,
        };
    }
    for (i, l) in pt_lights.into_iter().take(MAX_POINT_LIGHTS).enumerate() {
        point[i] = PointLightData {
            position: l.position,
            range: l.range,
            color: l.color,
            intensity: l.intensity,
        };
    }

    LightUniforms {
        directional,
        point,
        num_directional: num_directional as i32,
        num_point: num_point as i32,
        ambient_intensity,
        num_local_lights: num_local as i32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(direction: [f32; 3], color: [f32; 3], intensity: f32) -> DirectionalLight {
        DirectionalLight {
            direction,
            color,
            intensity,
        }
    }

    fn pt(position: [f32; 3], color: [f32; 3], intensity: f32, range: f32) -> PointLight {
        PointLight {
            position,
            color,
            intensity,
            range,
        }
    }

    #[test]
    fn empty_inputs_return_default() {
        let u = build_light_uniforms(vec![], vec![], 1.0);
        assert_eq!(u.num_directional, LightUniforms::DEFAULT.num_directional);
        assert_eq!(u.num_point, LightUniforms::DEFAULT.num_point);
    }

    #[test]
    fn ambient_intensity_carried_in_both_branches() {
        // Empty (DEFAULT) branch and the populated branch both honour the
        // authored multiplier.
        let empty = build_light_uniforms(vec![], vec![], 2.5);
        assert!((empty.ambient_intensity - 2.5).abs() < 1e-6);
        let populated =
            build_light_uniforms(vec![dir([-0.3, 0.85, 0.4], [1.0; 3], 1.0)], vec![], 3.0);
        assert!((populated.ambient_intensity - 3.0).abs() < 1e-6);
    }

    #[test]
    fn single_directional_light_fields_mapped() {
        let u = build_light_uniforms(
            vec![dir([-0.3, 0.85, 0.4], [1.0, 0.95, 0.8], 1.5)],
            vec![],
            1.0,
        );
        assert_eq!(u.num_directional, 1);
        assert_eq!(u.num_point, 0);
        assert_eq!(u.directional[0].direction, [-0.3, 0.85, 0.4]);
        assert_eq!(u.directional[0].color, [1.0, 0.95, 0.8]);
        assert!((u.directional[0].intensity - 1.5).abs() < 1e-6);
    }

    #[test]
    fn single_point_light_fields_mapped() {
        let u = build_light_uniforms(
            vec![],
            vec![pt([2.0, 3.0, 4.0], [1.0, 0.8, 0.5], 8.0, 6.0)],
            1.0,
        );
        assert_eq!(u.num_directional, 0);
        assert_eq!(u.num_point, 1);
        assert_eq!(u.point[0].position, [2.0, 3.0, 4.0]);
        assert_eq!(u.point[0].color, [1.0, 0.8, 0.5]);
        assert!((u.point[0].intensity - 8.0).abs() < 1e-6);
        assert!((u.point[0].range - 6.0).abs() < 1e-6);
    }

    #[test]
    fn excess_directional_lights_clamped_to_max() {
        let lights: Vec<DirectionalLight> = (0..MAX_DIRECTIONAL_LIGHTS + 2)
            .map(|i| dir([i as f32, 0.0, 0.0], [1.0; 3], 1.0))
            .collect();
        let u = build_light_uniforms(lights, vec![], 1.0);
        assert_eq!(u.num_directional, MAX_DIRECTIONAL_LIGHTS as i32);
    }

    #[test]
    fn excess_point_lights_clamped_to_max() {
        // The legacy `point` array (raymarch / fog / probe) still caps at 8, but
        // num_local_lights carries the full count for the forward pass.
        let lights: Vec<PointLight> = (0..MAX_POINT_LIGHTS + 2)
            .map(|i| pt([i as f32, 0.0, 0.0], [1.0; 3], 1.0, 5.0))
            .collect();
        let u = build_light_uniforms(vec![], lights, 1.0);
        assert_eq!(u.num_point, MAX_POINT_LIGHTS as i32);
        assert_eq!(u.num_local_lights, (MAX_POINT_LIGHTS + 2) as i32);
    }

    #[test]
    fn light_buffer_maps_point_light_fields() {
        let buf = build_light_buffer(&[pt([2.0, 3.0, 4.0], [1.0, 0.8, 0.5], 8.0, 6.0)]);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0].position, [2.0, 3.0, 4.0]);
        assert_eq!(buf[0].color, [1.0, 0.8, 0.5]);
        assert!((buf[0].intensity - 8.0).abs() < 1e-6);
        assert!((buf[0].range - 6.0).abs() < 1e-6);
        assert_eq!(buf[0].kind, crate::render_types::LIGHT_KIND_POINT);
        // Point lights carry no cone or shadow data.
        assert_eq!(buf[0].shadow_index, -1);
        assert_eq!(buf[0].direction, [0.0; 3]);
        assert_eq!(buf[0].cos_inner, 0.0);
        assert_eq!(buf[0].cos_outer, 0.0);
    }

    #[test]
    fn light_buffer_carries_more_than_the_legacy_cap() {
        let lights: Vec<PointLight> = (0..MAX_POINT_LIGHTS + 50)
            .map(|i| pt([i as f32, 0.0, 0.0], [1.0; 3], 1.0, 5.0))
            .collect();
        let buf = build_light_buffer(&lights);
        assert_eq!(buf.len(), MAX_POINT_LIGHTS + 50);
    }

    #[test]
    fn light_buffer_clamped_to_capacity() {
        let lights: Vec<PointLight> = (0..MAX_LOCAL_LIGHTS + 10)
            .map(|i| pt([i as f32, 0.0, 0.0], [1.0; 3], 1.0, 5.0))
            .collect();
        let buf = build_light_buffer(&lights);
        assert_eq!(buf.len(), MAX_LOCAL_LIGHTS);
    }
}
