//! Converts drained DirectionalLight, PointLight, SpotLight, and RectAreaLight
//! asset components into the GPU data the renderer consumes: the fixed
//! LightUniforms uniform (directional lights, ambient, and the legacy point array
//! the raymarch / fog / probe paths read), the GpuLight storage buffer the
//! clustered forward pass iterates, the per-slice spot shadow projections, and
//! the rect area-light extents.

use crate::area_light;
use crate::components::{
    DirectionalLight, PointLight, RectAreaLight, SpotLight, SpotLightGeometry,
};
use crate::render_types::{
    AreaLightData, DirectionalLightData, GpuLight, LIGHT_KIND_AREA, LIGHT_KIND_POINT,
    LIGHT_KIND_SPOT, LightUniforms, MAX_DIRECTIONAL_LIGHTS, MAX_LOCAL_LIGHTS, MAX_POINT_LIGHTS,
    PointLightData, SpotShadowData,
};
use crate::spot_shadow;

/// The per-scene GPU light data: the storage buffer the clustered forward pass
/// iterates, plus the side tables it indexes into. Kept together because
/// `GpuLight.shadow_index` indexes `spot_shadows` and `GpuLight.data_index`
/// indexes `area_lights` -- invariants that would be easy to break if the three
/// were built independently.
pub struct LightData {
    /// Every local light for the clustered forward pass.
    pub lights: Vec<GpuLight>,
    /// One entry per shadowed spot light.
    pub spot_shadows: Vec<SpotShadowData>,
    /// One entry per rectangular area light.
    pub area_lights: Vec<AreaLightData>,
}

/// Packs point, spot, and rect area lights into the GpuLight storage buffer and
/// assigns their side-table slots. All three share the MAX_LOCAL_LIGHTS budget
/// (not the 8-entry LightUniforms array); extras past the cap are dropped with a
/// warning. Unused fields stay at their neutral GpuLight::ZERO values, so a point
/// light carries no cone, shadow, or area data.
pub fn build_light_data(
    pt_lights: &[PointLight],
    spot_lights: &[SpotLight],
    rect_lights: &[RectAreaLight],
) -> LightData {
    let total = pt_lights.len() + spot_lights.len() + rect_lights.len();
    if total > MAX_LOCAL_LIGHTS {
        tracing::warn!(
            "GraphicsSystem: {} local lights declared; only {} are supported -- extras ignored",
            total,
            MAX_LOCAL_LIGHTS
        );
    }
    let slices = spot_shadow::assign_spot_shadow_slices(spot_lights);
    let spot_shadows = spot_shadow::build_spot_shadow_data(spot_lights, &slices);
    let slots = area_light::assign_area_light_slots(rect_lights);
    let area_lights = area_light::build_area_light_data(rect_lights, &slots);

    let points = pt_lights.iter().map(|l| GpuLight {
        position: l.position,
        range: l.range,
        color: l.color,
        intensity: l.intensity,
        kind: LIGHT_KIND_POINT,
        ..GpuLight::ZERO
    });
    let spots = spot_lights
        .iter()
        .zip(&slices)
        .map(|(l, &shadow_index)| GpuLight {
            position: l.position,
            range: l.range,
            color: l.color,
            intensity: l.intensity,
            direction: l.unit_direction(),
            kind: LIGHT_KIND_SPOT,
            cos_inner: l.cos_inner(),
            cos_outer: l.cos_outer(),
            shadow_index,
            ..GpuLight::ZERO
        });
    let areas = rect_lights
        .iter()
        .zip(&slots)
        .map(|(l, &data_index)| GpuLight {
            position: l.centre,
            range: l.range,
            color: l.color,
            intensity: l.intensity,
            direction: l.normal,
            kind: LIGHT_KIND_AREA,
            data_index,
            ..GpuLight::ZERO
        });
    let lights: Vec<GpuLight> = points
        .chain(spots)
        .chain(areas)
        .take(MAX_LOCAL_LIGHTS)
        .collect();

    // A spot dropped by the MAX_LOCAL_LIGHTS clamp must not leave a slice
    // reserved for a light the forward pass will never see.
    let kept = lights.iter().filter(|l| l.shadow_index >= 0).count();
    let spot_shadows = if kept < spot_shadows.len() {
        spot_shadows[..kept].to_vec()
    } else {
        spot_shadows
    };

    // An area light dropped by the MAX_LOCAL_LIGHTS clamp must not leave a table
    // entry the forward pass will never reach.
    let kept_areas = lights.iter().filter(|l| l.data_index >= 0).count();
    let area_lights = if kept_areas < area_lights.len() {
        area_lights[..kept_areas].to_vec()
    } else {
        area_lights
    };

    LightData {
        lights,
        spot_shadows,
        area_lights,
    }
}

/// `local_lights` is the buffer `build_light_data` produced; its length is the
/// authoritative `num_local_lights` the forward pass iterates.
pub fn build_light_uniforms(
    dir_lights: Vec<DirectionalLight>,
    pt_lights: Vec<PointLight>,
    local_lights: &[GpuLight],
    ambient_intensity: f32,
) -> LightUniforms {
    if dir_lights.is_empty() && pt_lights.is_empty() && local_lights.is_empty() {
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
    // (see build_light_data), so exceeding MAX_POINT_LIGHTS is not an error.
    let mut point = [ZERO_PT; MAX_POINT_LIGHTS];
    let num_directional = dir_lights.len().min(MAX_DIRECTIONAL_LIGHTS);
    let num_point = pt_lights.len().min(MAX_POINT_LIGHTS);

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
        num_local_lights: local_lights.len() as i32,
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

    fn spot(position: [f32; 3], direction: [f32; 3], inner: f32, outer: f32) -> SpotLight {
        SpotLight {
            position,
            direction,
            inner_angle: inner,
            outer_angle: outer,
            ..SpotLight::default()
        }
    }

    // The uniforms every test that does not exercise the local buffer wants.
    fn uniforms(dir_lights: Vec<DirectionalLight>, pt_lights: Vec<PointLight>) -> LightUniforms {
        let local = build_light_data(&pt_lights, &[], &[]).lights;
        build_light_uniforms(dir_lights, pt_lights, &local, 1.0)
    }

    #[test]
    fn empty_inputs_return_default() {
        let u = uniforms(vec![], vec![]);
        assert_eq!(u.num_directional, LightUniforms::DEFAULT.num_directional);
        assert_eq!(u.num_point, LightUniforms::DEFAULT.num_point);
    }

    #[test]
    fn ambient_intensity_carried_in_both_branches() {
        // Empty (DEFAULT) branch and the populated branch both honour the
        // authored multiplier.
        let empty = build_light_uniforms(vec![], vec![], &[], 2.5);
        assert!((empty.ambient_intensity - 2.5).abs() < 1e-6);
        let populated = build_light_uniforms(
            vec![dir([-0.3, 0.85, 0.4], [1.0; 3], 1.0)],
            vec![],
            &[],
            3.0,
        );
        assert!((populated.ambient_intensity - 3.0).abs() < 1e-6);
    }

    #[test]
    fn single_directional_light_fields_mapped() {
        let u = uniforms(vec![dir([-0.3, 0.85, 0.4], [1.0, 0.95, 0.8], 1.5)], vec![]);
        assert_eq!(u.num_directional, 1);
        assert_eq!(u.num_point, 0);
        assert_eq!(u.directional[0].direction, [-0.3, 0.85, 0.4]);
        assert_eq!(u.directional[0].color, [1.0, 0.95, 0.8]);
        assert!((u.directional[0].intensity - 1.5).abs() < 1e-6);
    }

    #[test]
    fn single_point_light_fields_mapped() {
        let u = uniforms(vec![], vec![pt([2.0, 3.0, 4.0], [1.0, 0.8, 0.5], 8.0, 6.0)]);
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
        let u = uniforms(lights, vec![]);
        assert_eq!(u.num_directional, MAX_DIRECTIONAL_LIGHTS as i32);
    }

    #[test]
    fn excess_point_lights_clamped_to_max() {
        // The legacy `point` array (raymarch / fog / probe) still caps at 8, but
        // num_local_lights carries the full count for the forward pass.
        let lights: Vec<PointLight> = (0..MAX_POINT_LIGHTS + 2)
            .map(|i| pt([i as f32, 0.0, 0.0], [1.0; 3], 1.0, 5.0))
            .collect();
        let u = uniforms(vec![], lights);
        assert_eq!(u.num_point, MAX_POINT_LIGHTS as i32);
        assert_eq!(u.num_local_lights, (MAX_POINT_LIGHTS + 2) as i32);
    }

    // Spot lights live only in the local buffer, so a spot-only scene still has
    // to report them through num_local_lights.
    #[test]
    fn num_local_lights_counts_spot_lights() {
        let local =
            build_light_data(&[], &[spot([0.0; 3], [0.0, -1.0, 0.0], 10.0, 20.0)], &[]).lights;
        let u = build_light_uniforms(vec![], vec![], &local, 1.0);
        assert_eq!(u.num_point, 0);
        assert_eq!(u.num_local_lights, 1);
    }

    #[test]
    fn light_buffer_maps_point_light_fields() {
        let buf =
            build_light_data(&[pt([2.0, 3.0, 4.0], [1.0, 0.8, 0.5], 8.0, 6.0)], &[], &[]).lights;
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0].position, [2.0, 3.0, 4.0]);
        assert_eq!(buf[0].color, [1.0, 0.8, 0.5]);
        assert!((buf[0].intensity - 8.0).abs() < 1e-6);
        assert!((buf[0].range - 6.0).abs() < 1e-6);
        assert_eq!(buf[0].kind, LIGHT_KIND_POINT);
        // Point lights carry no cone or shadow data.
        assert_eq!(buf[0].shadow_index, -1);
        assert_eq!(buf[0].direction, [0.0; 3]);
        assert_eq!(buf[0].cos_inner, 0.0);
        assert_eq!(buf[0].cos_outer, 0.0);
    }

    #[test]
    fn light_buffer_maps_spot_light_fields() {
        let buf = build_light_data(
            &[],
            &[spot([1.0, 5.0, 2.0], [0.0, -2.0, 0.0], 15.0, 30.0)],
            &[],
        )
        .lights;
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0].position, [1.0, 5.0, 2.0]);
        assert_eq!(buf[0].kind, LIGHT_KIND_SPOT);
        // The authored direction is normalised into the record.
        assert_eq!(buf[0].direction, [0.0, -1.0, 0.0]);
        assert!((buf[0].cos_inner - 15.0f32.to_radians().cos()).abs() < 1e-6);
        assert!((buf[0].cos_outer - 30.0f32.to_radians().cos()).abs() < 1e-6);
        // A wider inner cone than outer would invert the falloff; it is clamped.
        assert!(buf[0].cos_inner >= buf[0].cos_outer);
    }

    #[test]
    fn spot_inner_cone_clamped_to_the_outer_cone() {
        let buf =
            build_light_data(&[], &[spot([0.0; 3], [0.0, -1.0, 0.0], 60.0, 20.0)], &[]).lights;
        assert!((buf[0].cos_inner - buf[0].cos_outer).abs() < 1e-6);
    }

    #[test]
    fn spot_lights_follow_the_point_lights_in_the_buffer() {
        let buf = build_light_data(
            &[
                pt([0.0; 3], [1.0; 3], 1.0, 5.0),
                pt([1.0; 3], [1.0; 3], 1.0, 5.0),
            ],
            &[spot([2.0; 3], [0.0, -1.0, 0.0], 10.0, 20.0)],
            &[],
        )
        .lights;
        assert_eq!(buf.len(), 3);
        assert_eq!(buf[0].kind, LIGHT_KIND_POINT);
        assert_eq!(buf[1].kind, LIGHT_KIND_POINT);
        assert_eq!(buf[2].kind, LIGHT_KIND_SPOT);
    }

    // GpuLight.shadow_index indexes spot_shadows, so the two must agree.
    #[test]
    fn shadow_indices_point_at_real_spot_shadow_entries() {
        let mut casting = spot([0.0, 5.0, 0.0], [0.0, -1.0, 0.0], 10.0, 20.0);
        casting.cast_shadows = true;
        let mut dark = spot([3.0, 5.0, 0.0], [0.0, -1.0, 0.0], 10.0, 20.0);
        dark.cast_shadows = false;
        let data = build_light_data(
            &[pt([0.0; 3], [1.0; 3], 1.0, 5.0)],
            &[
                casting,
                dark,
                spot([6.0, 5.0, 0.0], [0.0, -1.0, 0.0], 10.0, 20.0),
            ],
            &[],
        );
        // Point lights never cast; the two casting spots take slices 0 and 1.
        assert_eq!(data.lights[0].shadow_index, -1);
        assert_eq!(data.lights[1].shadow_index, 0);
        assert_eq!(data.lights[2].shadow_index, -1);
        assert_eq!(data.lights[3].shadow_index, 1);
        assert_eq!(data.spot_shadows.len(), 2);
        for l in &data.lights {
            assert!(
                l.shadow_index < data.spot_shadows.len() as i32,
                "shadow_index stays in bounds of spot_shadows"
            );
        }
    }

    // A spot dropped by the MAX_LOCAL_LIGHTS clamp must not leave a shadow slice
    // reserved for a light the forward pass never sees.
    #[test]
    fn clamped_spots_do_not_strand_shadow_slices() {
        let points: Vec<PointLight> = (0..MAX_LOCAL_LIGHTS - 1)
            .map(|i| pt([i as f32, 0.0, 0.0], [1.0; 3], 1.0, 5.0))
            .collect();
        let spots: Vec<SpotLight> = (0..4)
            .map(|i| {
                let mut s = spot([i as f32, 5.0, 0.0], [0.0, -1.0, 0.0], 10.0, 20.0);
                s.cast_shadows = true;
                s
            })
            .collect();
        let data = build_light_data(&points, &spots, &[]);
        assert_eq!(data.lights.len(), MAX_LOCAL_LIGHTS);
        // Only one spot survived the clamp, so only its slice is kept.
        assert_eq!(data.spot_shadows.len(), 1);
        let max_index = data.lights.iter().map(|l| l.shadow_index).max().unwrap();
        assert_eq!(max_index, 0);
    }

    fn area(centre: [f32; 3], half_size: [f32; 2]) -> RectAreaLight {
        RectAreaLight {
            centre,
            half_size,
            ..RectAreaLight::default()
        }
    }

    // GpuLight.data_index indexes area_lights, so the two must agree, and area
    // lights must not disturb the spot shadow indices.
    #[test]
    fn area_lights_follow_the_other_kinds_and_index_their_table() {
        let mut casting = spot([0.0, 5.0, 0.0], [0.0, -1.0, 0.0], 10.0, 20.0);
        casting.cast_shadows = true;
        let data = build_light_data(
            &[pt([0.0; 3], [1.0; 3], 1.0, 5.0)],
            &[casting],
            &[
                area([2.0, 3.0, 0.0], [1.0, 2.0]),
                area([5.0; 3], [1.0, 1.0]),
            ],
        );
        assert_eq!(data.lights.len(), 4);
        assert_eq!(data.lights[2].kind, LIGHT_KIND_AREA);
        assert_eq!(data.lights[3].kind, LIGHT_KIND_AREA);
        assert_eq!(data.lights[2].data_index, 0);
        assert_eq!(data.lights[3].data_index, 1);
        assert_eq!(data.area_lights.len(), 2);
        // Point and spot lights carry no area data; the spot keeps its slice.
        assert_eq!(data.lights[0].data_index, -1);
        assert_eq!(data.lights[1].data_index, -1);
        assert_eq!(data.lights[1].shadow_index, 0);
        for l in &data.lights {
            assert!(l.data_index < data.area_lights.len() as i32);
        }
    }

    // The area light's centre and emitting direction ride the GpuLight record.
    #[test]
    fn area_light_centre_and_normal_map_onto_the_gpu_light() {
        let mut l = area([1.0, 2.0, 3.0], [1.0, 1.0]);
        l.normal = [0.0, 0.0, 1.0];
        let data = build_light_data(&[], &[], &[l]);
        assert_eq!(data.lights[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(data.lights[0].direction, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn clamped_area_lights_do_not_strand_table_entries() {
        let points: Vec<PointLight> = (0..MAX_LOCAL_LIGHTS - 1)
            .map(|i| pt([i as f32, 0.0, 0.0], [1.0; 3], 1.0, 5.0))
            .collect();
        let areas: Vec<RectAreaLight> = (0..4).map(|_| area([0.0; 3], [1.0, 1.0])).collect();
        let data = build_light_data(&points, &[], &areas);
        assert_eq!(data.lights.len(), MAX_LOCAL_LIGHTS);
        assert_eq!(data.area_lights.len(), 1);
    }

    #[test]
    fn light_buffer_carries_more_than_the_legacy_cap() {
        let lights: Vec<PointLight> = (0..MAX_POINT_LIGHTS + 50)
            .map(|i| pt([i as f32, 0.0, 0.0], [1.0; 3], 1.0, 5.0))
            .collect();
        let buf = build_light_data(&lights, &[], &[]).lights;
        assert_eq!(buf.len(), MAX_POINT_LIGHTS + 50);
    }

    #[test]
    fn light_buffer_clamped_to_capacity() {
        let lights: Vec<PointLight> = (0..MAX_LOCAL_LIGHTS + 10)
            .map(|i| pt([i as f32, 0.0, 0.0], [1.0; 3], 1.0, 5.0))
            .collect();
        let buf = build_light_data(&lights, &[], &[]).lights;
        assert_eq!(buf.len(), MAX_LOCAL_LIGHTS);
    }

    // Point and spot lights share one budget: the spots are what overflow.
    #[test]
    fn point_and_spot_lights_share_the_capacity() {
        let points: Vec<PointLight> = (0..MAX_LOCAL_LIGHTS - 1)
            .map(|i| pt([i as f32, 0.0, 0.0], [1.0; 3], 1.0, 5.0))
            .collect();
        let spots: Vec<SpotLight> = (0..4)
            .map(|i| spot([i as f32, 0.0, 0.0], [0.0, -1.0, 0.0], 10.0, 20.0))
            .collect();
        let buf = build_light_data(&points, &spots, &[]).lights;
        assert_eq!(buf.len(), MAX_LOCAL_LIGHTS);
        assert_eq!(buf[MAX_LOCAL_LIGHTS - 1].kind, LIGHT_KIND_SPOT);
    }
}
