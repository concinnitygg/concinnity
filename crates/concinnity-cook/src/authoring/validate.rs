//! The bake-time validators, re-exported from their home in
//! `concinnity_core::components::validate` under the `crate::authoring::validate::<fn>`
//! paths the registry's `validate:` / `validate_for:` entries name. The build-side
//! `RegisteredType::reserialize_args` applies them while baking the blob
//! record; the runtime never runs these on a loaded world.

pub use concinnity_core::components::validate::volumetric_fog;
pub(crate) use concinnity_core::components::validate::{
    decal, directional_light, glass_panel, instanced_prop, joint, material, particle_emitter,
    point_light, prop, rect_area_light, reflection_probe, rigid_body, sdf_volume, spot_light,
    voxel_chunk, water_surface,
};

#[cfg(test)]
mod tests {
    use crate::components::*;

    mod material {
        use super::*;

        #[test]
        fn default_is_opaque_and_not_see_through() {
            let m = Material::default();
            assert!(!m.transparent);
            assert!(!m.see_through);
            assert_eq!(m.opacity, 1.0);
        }

        #[test]
        fn see_through_implies_transparent() {
            // A material that opts into see-through but leaves `transparent` at
            // its default must still route through the transparent pass.
            let m = super::super::material(Material {
                see_through: true,
                ..Material::default()
            });
            assert!(m.see_through);
            assert!(m.transparent);
        }

        #[test]
        fn transparent_without_see_through_stays_opaque_layer() {
            // The importer's glass detection sets `transparent` only; that
            // material stays Layer 1 (opaque reflective) and keeps see-through off.
            let m = super::super::material(Material {
                transparent: true,
                ..Material::default()
            });
            assert!(m.transparent);
            assert!(!m.see_through);
        }
    }

    mod decal {
        use super::*;

        #[test]
        fn deserialises_with_defaults() {
            let d: Decal = serde_json::from_str("{}").unwrap();
            assert_eq!(d.position, [0.0, 0.0, 0.0]);
            assert_eq!(d.size, [1.0, 1.0, 1.0]);
            assert_eq!(d.tint, [1.0, 1.0, 1.0, 1.0]);
            assert!(d.visible);
            assert!(d.texture.is_none());
        }

        #[test]
        fn deserialises_with_all_fields() {
            crate::ecs::asset_id::reset_interner();
            let json = r#"{
                "texture":"tex_bullet",
                "position":[1.0,2.0,3.0],
                "rotation_deg":[0,90,0],
                "size":[0.4,0.2,0.4],
                "tint":[0.9,0.2,0.1,0.8],
                "visible":false
            }"#;
            let d: Decal = serde_json::from_str(json).unwrap();
            assert_eq!(d.position, [1.0, 2.0, 3.0]);
            assert_eq!(d.rotation_deg, [0.0, 90.0, 0.0]);
            assert_eq!(d.size, [0.4, 0.2, 0.4]);
            assert_eq!(d.tint, [0.9, 0.2, 0.1, 0.8]);
            assert!(!d.visible);
            assert!(d.texture.is_some());
        }

        #[test]
        fn clamps_alpha_through_from_args() {
            let json = r#"{"tint":[1,1,1,5.0]}"#;
            let parsed: Decal = serde_json::from_str(json).unwrap();
            let normalised = super::super::decal(parsed);
            assert_eq!(normalised.tint[3], 1.0);

            let json = r#"{"tint":[1,1,1,-0.5]}"#;
            let parsed: Decal = serde_json::from_str(json).unwrap();
            let normalised = super::super::decal(parsed);
            assert_eq!(normalised.tint[3], 0.0);
        }
    }

    mod glass_panel {
        use super::*;

        #[test]
        fn from_args_normalizes_normal() {
            let g = super::super::glass_panel(GlassPanel {
                normal: [0.0, 0.0, 4.0],
                ..Default::default()
            });
            let len = (g.normal[0].powi(2) + g.normal[1].powi(2) + g.normal[2].powi(2)).sqrt();
            assert!((len - 1.0).abs() < 1e-5);
            assert!((g.normal[2] - 1.0).abs() < 1e-5);
        }

        #[test]
        fn from_args_falls_back_on_degenerate_normal() {
            let g = super::super::glass_panel(GlassPanel {
                normal: [0.0, 0.0, 0.0],
                ..Default::default()
            });
            assert_eq!(g.normal, [0.0, 0.0, 1.0]);
        }

        #[test]
        fn from_args_clamps_ranges() {
            let g = super::super::glass_panel(GlassPanel {
                half_size: [-2.0, 0.0],
                opacity: 1.5,
                refraction_strength: -0.1,
                fresnel_power: -3.0,
                ..Default::default()
            });
            assert!(g.half_size[0] > 0.0 && g.half_size[1] > 0.0);
            assert_eq!(g.opacity, 1.0);
            assert_eq!(g.refraction_strength, 0.0);
            assert_eq!(g.fresnel_power, 0.0);
        }
    }

    mod joint {
        use super::*;

        #[test]
        fn deserialises_with_defaults() {
            let j: PhysicsJoint = serde_json::from_str("{}").unwrap();
            assert_eq!(j.kind, "fixed");
            assert_eq!(j.anchor_a, [0.0, 0.0, 0.0]);
            assert_eq!(j.axis, [0.0, 1.0, 0.0]);
            assert!(!j.limits_enabled);
            assert_eq!(j.motor_max_force, 0.0);
        }

        #[test]
        fn deserialises_all_fields() {
            crate::ecs::asset_id::reset_interner();
            let json = r#"{
                "kind":"revolute",
                "body_a":"door",
                "body_b":"wall",
                "anchor_a":[0.5,1.0,0.0],
                "anchor_b":[1.0,1.0,0.0],
                "axis":[0,1,0],
                "limits_enabled":true,
                "limits":[-90,90],
                "motor_target_velocity":30.0,
                "motor_max_force":50.0
            }"#;
            let j: PhysicsJoint = serde_json::from_str(json).unwrap();
            assert_eq!(j.parsed_kind(), PhysicsJointKind::Revolute);
            assert!(j.body_a.is_some());
            assert!(j.body_b.is_some());
            assert!(j.limits_enabled);
        }

        #[test]
        fn aliases_resolve_to_canonical_kind() {
            assert_eq!(
                PhysicsJointKind::from_str_norm("hinge"),
                Some(PhysicsJointKind::Revolute)
            );
            assert_eq!(
                PhysicsJointKind::from_str_norm("WELD"),
                Some(PhysicsJointKind::Fixed)
            );
            assert_eq!(
                PhysicsJointKind::from_str_norm("ball"),
                Some(PhysicsJointKind::Spherical)
            );
            assert_eq!(
                PhysicsJointKind::from_str_norm("slider"),
                Some(PhysicsJointKind::Prismatic)
            );
        }

        #[test]
        fn from_args_normalises_kind_string() {
            let json = r#"{"kind":"HINGE"}"#;
            let parsed: PhysicsJoint = serde_json::from_str(json).unwrap();
            let normalised = super::super::joint(parsed);
            assert_eq!(normalised.kind, "revolute");
        }

        #[test]
        fn unknown_kind_falls_back_to_fixed() {
            let j = PhysicsJoint {
                kind: "frumpus".to_string(),
                ..Default::default()
            };
            assert_eq!(j.parsed_kind(), PhysicsJointKind::Fixed);
        }
    }

    mod particle_emitter {
        use super::*;

        #[test]
        fn deserialises_with_defaults() {
            let p: ParticleEmitter = serde_json::from_str("{}").unwrap();
            assert_eq!(p.position, [0.0, 0.0, 0.0]);
            assert_eq!(p.direction, [0.0, 1.0, 0.0]);
            assert_eq!(p.max_particles, 256);
            assert!(p.visible);
            assert!(p.texture.is_none());
        }

        #[test]
        fn deserialises_with_all_fields() {
            crate::ecs::asset_id::reset_interner();
            let json = r#"{
                "texture":"tex_spark","position":[1,2,3],"direction":[0,1,0],
                "spread_deg":30,"speed_min":1.5,"speed_max":4.0,
                "lifetime_min":0.5,"lifetime_max":1.0,"gravity":[0,-1,0],
                "spawn_rate":60,"max_particles":128,"size_start":0.1,"size_end":0.02,
                "color_start":[1,0.5,0,1],"color_end":[1,0,0,0],"visible":false
            }"#;
            let p: ParticleEmitter = serde_json::from_str(json).unwrap();
            assert_eq!(p.position, [1.0, 2.0, 3.0]);
            assert_eq!(p.max_particles, 128);
            assert_eq!(p.color_start, [1.0, 0.5, 0.0, 1.0]);
            assert!(!p.visible);
            assert!(p.texture.is_some());
        }

        #[test]
        fn from_args_clamps_invalid_inputs() {
            let a = ParticleEmitter {
                spread_deg: 300.0,
                speed_min: -1.0,
                speed_max: -5.0,
                lifetime_min: -0.4,
                lifetime_max: -2.0,
                spawn_rate: -10.0,
                max_particles: 0,
                size_start: -0.5,
                size_end: -0.1,
                ..Default::default()
            };
            let n = super::super::particle_emitter(a);
            assert_eq!(n.spread_deg, 180.0);
            assert_eq!(n.speed_min, 0.0);
            assert_eq!(n.speed_max, 0.0);
            assert!(n.lifetime_min > 0.0);
            assert!(n.lifetime_max >= n.lifetime_min);
            assert_eq!(n.spawn_rate, 0.0);
            assert_eq!(n.max_particles, 1);
            assert_eq!(n.size_start, 0.0);
            assert_eq!(n.size_end, 0.0);
        }

        #[test]
        fn from_args_lifts_speed_max_to_speed_min() {
            let a = ParticleEmitter {
                speed_min: 5.0,
                speed_max: 2.0,
                ..Default::default()
            };
            let n = super::super::particle_emitter(a);
            assert!((n.speed_max - n.speed_min).abs() < 1e-6);
        }

        #[test]
        fn from_args_clamps_max_particles_upper() {
            let a = ParticleEmitter {
                max_particles: 200_000,
                ..Default::default()
            };
            let n = super::super::particle_emitter(a);
            assert_eq!(n.max_particles, 65_536);
        }
    }

    mod volumetric_fog {
        use super::*;

        #[test]
        fn deserialises_with_defaults() {
            let f: VolumetricFog = serde_json::from_str("{}").unwrap();
            assert!(f.enabled);
            assert_eq!(f.color, [0.7, 0.78, 0.85]);
            assert!((f.density - 0.05).abs() < 1e-6);
            assert!((f.max_distance - 200.0).abs() < 1e-6);
        }

        #[test]
        fn deserialises_with_explicit_fields() {
            let json = r#"{
                "enabled":false,"density":0.12,"color":[0.5,0.6,0.7],
                "height_falloff":0.3,"height_reference":1.5,
                "max_distance":80.0,"phase_g":0.7,"ambient":0.25
            }"#;
            let f: VolumetricFog = serde_json::from_str(json).unwrap();
            assert!(!f.enabled);
            assert_eq!(f.color, [0.5, 0.6, 0.7]);
            assert!((f.phase_g - 0.7).abs() < 1e-6);
        }

        #[test]
        fn from_args_clamps_invalid_inputs() {
            let a = VolumetricFog {
                density: -1.0,
                height_falloff: -0.4,
                ambient: -2.0,
                max_distance: -1.0,
                phase_g: 1.4,
                ..Default::default()
            };
            let n = super::super::volumetric_fog(a);
            assert_eq!(n.density, 0.0);
            assert_eq!(n.height_falloff, 0.0);
            assert_eq!(n.ambient, 0.0);
            assert!(n.max_distance > 0.0);
            assert!(n.phase_g <= 0.95 && n.phase_g > 0.0);
        }

        #[test]
        fn from_args_passes_through_valid_inputs() {
            let a = VolumetricFog {
                density: 0.08,
                phase_g: -0.3,
                ..Default::default()
            };
            let n = super::super::volumetric_fog(a);
            assert!((n.density - 0.08).abs() < 1e-6);
            assert!((n.phase_g - (-0.3)).abs() < 1e-6);
        }
    }

    mod instanced_prop {
        use super::*;
        use crate::components::{InstanceTransform, InstancedPropGeometry};
        use crate::ecs::asset_id::AssetId;

        fn empty() -> InstancedProp {
            InstancedProp {
                asset_id: AssetId::default(),
                mesh: None,
                material: None,
                texture: None,
                instances: Vec::new(),
                cull_distance: 0.0,
            }
        }

        #[test]
        fn instance_model_matrix_default_is_identity() {
            let mut p = empty();
            p.instances.push(InstanceTransform::default());
            let m = p.instance_model_matrix(0).unwrap();
            assert_eq!(m[3], [0.0, 0.0, 0.0, 1.0]);
            assert!((m[0][0] - 1.0).abs() < 1e-5);
            assert!((m[1][1] - 1.0).abs() < 1e-5);
            assert!((m[2][2] - 1.0).abs() < 1e-5);
        }

        #[test]
        fn instance_model_matrix_translates() {
            let mut p = empty();
            p.instances.push(InstanceTransform {
                position: [5.0, -2.0, 3.0],
                ..InstanceTransform::default()
            });
            let m = p.instance_model_matrix(0).unwrap();
            assert_eq!(m[3], [5.0, -2.0, 3.0, 1.0]);
        }

        #[test]
        fn instance_model_matrix_scales() {
            let mut p = empty();
            p.instances.push(InstanceTransform {
                scale: [2.0, 3.0, 4.0],
                ..InstanceTransform::default()
            });
            let m = p.instance_model_matrix(0).unwrap();
            // diagonal entries should be the scale factors (no rotation)
            assert!((m[0][0] - 2.0).abs() < 1e-5);
            assert!((m[1][1] - 3.0).abs() < 1e-5);
            assert!((m[2][2] - 4.0).abs() < 1e-5);
        }

        #[test]
        fn instance_model_matrix_out_of_range_returns_none() {
            let p = empty();
            assert!(p.instance_model_matrix(0).is_none());
        }

        #[test]
        fn from_args_clamps_negative_cull_distance() {
            let args = InstancedProp {
                cull_distance: -5.0,
                ..InstancedProp::default()
            };
            let p = super::super::instanced_prop(args);
            assert_eq!(p.cull_distance, 0.0);
        }
    }

    mod sdf_volume {
        use super::*;
        use crate::components::sdf_volume::{SDF_MAX_STEPS_CEILING, SDF_MAX_STEPS_FLOOR};

        use concinnity_core::platform::Platform;

        #[test]
        fn clamps_steps() {
            let mut a = SdfVolume {
                max_steps: 1,
                ..Default::default()
            };
            let fixed = super::super::sdf_volume(a.clone(), Platform::Metal);
            assert_eq!(fixed.max_steps, SDF_MAX_STEPS_FLOOR);

            a.max_steps = 9999;
            let fixed = super::super::sdf_volume(a, Platform::Metal);
            assert_eq!(fixed.max_steps, SDF_MAX_STEPS_CEILING);
        }

        #[test]
        fn repairs_bad_extent() {
            let a = SdfVolume {
                extent: [0.0, -1.0, f32::NAN],
                ..Default::default()
            };
            let fixed = super::super::sdf_volume(a, Platform::Metal);
            assert_eq!(fixed.extent, [1.0, 1.0, 1.0]);
        }

        #[test]
        fn repairs_bad_gradient_and_distance() {
            let a = SdfVolume {
                max_gradient: -0.5,
                max_distance: f32::NAN,
                ..Default::default()
            };
            let fixed = super::super::sdf_volume(a, Platform::Metal);
            assert_eq!(fixed.max_gradient, 1.0);
            assert_eq!(fixed.max_distance, 0.1);
        }

        #[test]
        fn collapses_map_to_the_requested_backend() {
            // The runtime struct should carry the cooked backend's path in
            // `fragment_shader` so the DirectX path-extension filter still works
            // for map-authored volumes.
            let mut map = std::collections::BTreeMap::new();
            map.insert("metal".to_string(), "shaders/blob.metal".to_string());
            map.insert("hlsl".to_string(), "shaders/blob.hlsl".to_string());
            map.insert("glsl".to_string(), "shaders/blob.glsl".to_string());
            let a = SdfVolume {
                fragment_shaders: Some(map),
                ..Default::default()
            };
            for (platform, expected) in [
                (Platform::Metal, "shaders/blob.metal"),
                (Platform::Hlsl, "shaders/blob.hlsl"),
                (Platform::Glsl, "shaders/blob.glsl"),
            ] {
                let resolved = super::super::sdf_volume(a.clone(), platform);
                assert_eq!(resolved.fragment_shader, expected);
            }
        }

        #[test]
        fn a_source_for_another_backend_is_left_alone() {
            // A single path whose extension names a different backend is not
            // this platform's source, so the collapse leaves the field as
            // authored rather than adopting it.
            let a = SdfVolume {
                fragment_shader: "shaders/blob.metal".to_string(),
                ..Default::default()
            };
            let fixed = super::super::sdf_volume(a, Platform::Hlsl);
            assert_eq!(fixed.fragment_shader, "shaders/blob.metal");
        }

        #[test]
        fn volumetric_forces_cast_shadows_off() {
            let a = SdfVolume {
                volumetric: true,
                cast_shadows: true,
                ..Default::default()
            };
            let fixed = super::super::sdf_volume(a, Platform::Metal);
            assert!(fixed.volumetric);
            assert!(
                !fixed.cast_shadows,
                "volumetric SDFs are translucent and must not cast hard shadows"
            );
        }

        #[test]
        fn roundtrip_through_args() {
            let mut v = SdfVolume {
                centre: [1.0, 2.0, 3.0],
                extent: [4.0, 5.0, 6.0],
                fragment_shader: "shaders/foo.metal".to_string(),
                ..Default::default()
            };
            v.params[7] = 0.42;
            let json = serde_json::to_value(v.clone()).expect("serialises");
            let back: SdfVolume = serde_json::from_value(json).expect("deserialises");
            let back = super::super::sdf_volume(back, Platform::Metal);
            assert_eq!(back.centre, [1.0, 2.0, 3.0]);
            assert_eq!(back.extent, [4.0, 5.0, 6.0]);
            assert_eq!(back.fragment_shader, "shaders/foo.metal");
            assert_eq!(back.params[7], 0.42);
        }
    }
}
