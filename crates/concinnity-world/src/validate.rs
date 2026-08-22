//! Named bake-time validators for the data-only assets. Each function clamps or
//! normalizes an asset's authored value into a self-consistent runtime value.
//! The registry entry names the function via `validate: <fn>`; the build-side
//! `ComponentType::reserialize_args` applies it while baking the blob record.
//! The runtime never runs these -- a baked record is already validated.

use crate::assets::{
    Decal, DirectionalLight, GlassPanel, GlassPanelGeometry, InstancedProp, Joint, JointKind,
    Material, ParticleEmitter, PointLight, Prop, RectAreaLight, ReflectionProbe, RigidBody,
    SPOT_MAX_ANGLE_DEG, SdfVolume, SpotLight, SpotLightGeometry, VolumetricFog, VoxelChunk,
    WaterSurface, WaterWave,
};

// The wave ceiling lives with the schema in concinnity-asset and is shared with
// the render backends; re-imported here for the clamp.
use crate::assets::MAX_WATER_WAVES;

// Resolve the fragment shader source path for the current build backend from a
// volume's `fragment_shaders` map (preferred) or its `fragment_shader`
// fallback. Mirrors the source selection in `source_args`.
fn sdf_current_platform_source(v: &SdfVolume) -> Option<String> {
    let platform = crate::platform::Platform::current();
    if let Some(map) = &v.fragment_shaders
        && let Some(src) = map.get(platform.key()).filter(|s| !s.is_empty())
    {
        return Some(src.clone());
    }
    if v.fragment_shader.is_empty() {
        return None;
    }
    let ext = std::path::Path::new(&v.fragment_shader)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    if platform.accepts_ext(ext) {
        Some(v.fragment_shader.clone())
    } else {
        None
    }
}

/// Normalize an authored volume for the runtime: clamp the raymarch knobs to
/// sane bounds, force shadows off for translucent volumetrics (they write no
/// depth), and collapse the per-backend `fragment_shaders` map to the current
/// backend's `fragment_shader` (the DirectX raymarch pass filters volumes by
/// that path's extension). The step-count bounds stay in core: they double as
/// the runtime kernel's loop bound.
pub fn sdf_volume(mut v: SdfVolume) -> SdfVolume {
    use crate::assets::sdf_volume::{SDF_MAX_STEPS_CEILING, SDF_MAX_STEPS_FLOOR};
    // Extents must be positive: a zero or negative extent would produce an
    // inside-out bounding box no fragment ever enters.
    for axis in v.extent.iter_mut() {
        if !axis.is_finite() || *axis <= 0.0 {
            *axis = 1.0;
        }
    }
    if !v.max_gradient.is_finite() || v.max_gradient <= 0.0 {
        v.max_gradient = 1.0;
    }
    v.max_steps = v
        .max_steps
        .clamp(SDF_MAX_STEPS_FLOOR, SDF_MAX_STEPS_CEILING);
    if !v.max_distance.is_finite() || v.max_distance < 0.1 {
        v.max_distance = 0.1;
    }
    if v.volumetric {
        v.cast_shadows = false;
    }
    if let Some(src) = sdf_current_platform_source(&v) {
        v.fragment_shader = src;
    }
    v
}

/// Clamp a `PointLight`'s authored fields into their valid ranges.
pub fn point_light(mut args: PointLight) -> PointLight {
    args.intensity = args.intensity.max(0.0);
    args.range = args.range.max(0.0);
    args
}

pub(crate) fn spot_light(mut args: SpotLight) -> SpotLight {
    args.intensity = args.intensity.max(0.0);
    args.range = args.range.max(0.0);
    args.direction = args.unit_direction();
    args.outer_angle = args.outer_angle.clamp(0.0, SPOT_MAX_ANGLE_DEG);
    args.inner_angle = args.inner_angle.clamp(0.0, args.outer_angle);
    args
}

pub(crate) fn rect_area_light(mut args: RectAreaLight) -> RectAreaLight {
    args.intensity = args.intensity.max(0.0);
    args.range = args.range.max(0.0);
    // A degenerate normal would collapse the panel's tangent frame; a zero
    // half-extent would collapse its area and divide by zero in the integrator.
    let n = args.normal;
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    args.normal = if len < 1e-6 {
        [0.0, 0.0, 1.0]
    } else {
        [n[0] / len, n[1] / len, n[2] / len]
    };
    args.half_size[0] = args.half_size[0].max(1e-3);
    args.half_size[1] = args.half_size[1].max(1e-3);
    args
}

pub(crate) fn directional_light(mut args: DirectionalLight) -> DirectionalLight {
    args.intensity = args.intensity.max(0.0);
    args
}

/// Public so the cook-side Material data-resource compiler can apply the same
/// clamps: Material left the component registry, so its generated `from_args` (the
/// usual caller of this validator) no longer runs -- cook must call it explicitly
/// before baking the material into its `data_bytes`.
pub fn material(mut args: Material) -> Material {
    args.roughness = args.roughness.clamp(0.0, 1.0);
    args.metallic = args.metallic.clamp(0.0, 1.0);
    args.macro_variation = args.macro_variation.clamp(0.0, 1.0);
    args.terrain_blend = args.terrain_blend.clamp(0.0, 1.0);
    args.secondary_blend_sharpness = args.secondary_blend_sharpness.clamp(0.0, 1.0);
    args.alpha_cutoff = args.alpha_cutoff.clamp(0.0, 1.0);
    args.opacity = args.opacity.clamp(0.0, 1.0);
    // See-through glass is by definition transparent; opting into it implies
    // the transparent pass even if the author only set `see_through`.
    if args.see_through {
        args.transparent = true;
    }
    args
}

pub(crate) fn glass_panel(mut args: GlassPanel) -> GlassPanel {
    args.normal = args.unit_normal();
    args.half_size[0] = args.half_size[0].max(1e-3);
    args.half_size[1] = args.half_size[1].max(1e-3);
    args.opacity = args.opacity.clamp(0.0, 1.0);
    args.refraction_strength = args.refraction_strength.max(0.0);
    args.fresnel_power = args.fresnel_power.max(0.0);
    args
}

pub(crate) fn water_surface(mut args: WaterSurface) -> WaterSurface {
    args.subdivisions = args.subdivisions.clamp(8, 255);
    if args.waves.len() > MAX_WATER_WAVES {
        args.waves.truncate(MAX_WATER_WAVES);
    }
    if args.waves.is_empty() {
        args.waves.push(WaterWave::default());
    }
    args
}

/// Clamp a `Joint`'s authored fields into their valid ranges.
pub fn joint(mut args: Joint) -> Joint {
    // Normalise the kind string so `to_args` round-trips cleanly.
    if let Some(k) = JointKind::from_str_norm(&args.kind) {
        args.kind = k.as_str().to_string();
    }
    args
}

/// Clamp a `Decal`'s authored fields into their valid ranges.
pub fn decal(mut args: Decal) -> Decal {
    // Clamp the alpha to [0, 1] so a stray > 1 doesn't blow out the
    // composite. The size components are left as-authored: a non-positive
    // value silently disables the decal in the gfx-side resolver below.
    args.tint[3] = args.tint[3].clamp(0.0, 1.0);
    args
}

pub(crate) fn reflection_probe(mut args: ReflectionProbe) -> ReflectionProbe {
    // Half-extents are sizes: keep them non-negative so the influence box is
    // never inverted.
    for e in &mut args.half_extents {
        *e = e.max(0.0);
    }
    args
}

pub(crate) fn rigid_body(mut args: RigidBody) -> RigidBody {
    // Runtime state is always reset on construction.
    args.is_grounded = true;
    args
}

/// Clamp a `Prop`'s authored fields into their valid ranges.
pub fn prop(mut args: Prop) -> Prop {
    args.cull_distance = args.cull_distance.max(0.0);
    args.is_held = false;
    args
}

pub(crate) fn particle_emitter(mut args: ParticleEmitter) -> ParticleEmitter {
    // Asset-side floor: keep every authored field in a self-consistent
    // range. The gfx-side `build_particle_records` adds its own clamps
    // for fields that affect GPU buffer sizing.
    args.spread_deg = args.spread_deg.clamp(0.0, 180.0);
    args.speed_min = args.speed_min.max(0.0);
    if !args.speed_max.is_finite() || args.speed_max < args.speed_min {
        args.speed_max = args.speed_min;
    }
    if !args.lifetime_min.is_finite() || args.lifetime_min <= 0.0 {
        args.lifetime_min = 0.001;
    }
    if !args.lifetime_max.is_finite() || args.lifetime_max < args.lifetime_min {
        args.lifetime_max = args.lifetime_min;
    }
    args.spawn_rate = args.spawn_rate.max(0.0);
    args.max_particles = args.max_particles.clamp(1, 65_536);
    args.size_start = args.size_start.max(0.0);
    args.size_end = args.size_end.max(0.0);
    for c in args.color_start.iter_mut().chain(args.color_end.iter_mut()) {
        if !c.is_finite() {
            *c = 0.0;
        }
    }
    args
}

/// Clamp a `VolumetricFog`'s authored fields into their valid ranges.
pub fn volumetric_fog(mut args: VolumetricFog) -> VolumetricFog {
    // Density / falloff / ambient floor at 0; max_distance must stay
    // positive so the gfx-side resolver does not divide by zero when
    // computing the per-step length.
    args.density = args.density.max(0.0);
    args.height_falloff = args.height_falloff.max(0.0);
    args.ambient = args.ambient.max(0.0);
    if args.max_distance <= 0.0 || !args.max_distance.is_finite() {
        args.max_distance = 1.0;
    }
    // Henyey-Greenstein blows up at |g| = 1; clamp inside the open
    // interval so the closed-form `(1 - g²)` factor stays positive.
    args.phase_g = args.phase_g.clamp(-0.95, 0.95);
    args
}

pub(crate) fn instanced_prop(mut args: InstancedProp) -> InstancedProp {
    args.cull_distance = args.cull_distance.max(0.0);
    args
}

/// Clamp a `VoxelChunk`'s authored fields into their valid ranges.
pub fn voxel_chunk(mut args: VoxelChunk) -> VoxelChunk {
    args.block_size = args.block_size.max(0.0);
    if args.lod_levels == 0 {
        args.lod_levels = 1;
    }
    args.lod_levels = args.lod_levels.min(8);
    args
}

#[cfg(test)]
mod tests {
    use crate::assets::*;

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
            let j: Joint = serde_json::from_str("{}").unwrap();
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
            let j: Joint = serde_json::from_str(json).unwrap();
            assert_eq!(j.parsed_kind(), JointKind::Revolute);
            assert!(j.body_a.is_some());
            assert!(j.body_b.is_some());
            assert!(j.limits_enabled);
        }

        #[test]
        fn aliases_resolve_to_canonical_kind() {
            assert_eq!(JointKind::from_str_norm("hinge"), Some(JointKind::Revolute));
            assert_eq!(JointKind::from_str_norm("WELD"), Some(JointKind::Fixed));
            assert_eq!(JointKind::from_str_norm("ball"), Some(JointKind::Spherical));
            assert_eq!(
                JointKind::from_str_norm("slider"),
                Some(JointKind::Prismatic)
            );
        }

        #[test]
        fn from_args_normalises_kind_string() {
            let json = r#"{"kind":"HINGE"}"#;
            let parsed: Joint = serde_json::from_str(json).unwrap();
            let normalised = super::super::joint(parsed);
            assert_eq!(normalised.kind, "revolute");
        }

        #[test]
        fn unknown_kind_falls_back_to_fixed() {
            let j = Joint {
                kind: "frumpus".to_string(),
                ..Default::default()
            };
            assert_eq!(j.parsed_kind(), JointKind::Fixed);
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
        use crate::assets::{InstanceTransform, InstancedPropGeometry};
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
        use crate::assets::sdf_volume::{SDF_MAX_STEPS_CEILING, SDF_MAX_STEPS_FLOOR};

        // File extension matching the backend these tests compile against, so a
        // single `fragment_shader` path resolves as current-platform-compatible
        // on Metal, DirectX, and Vulkan alike.
        fn platform_ext() -> &'static str {
            crate::platform::Platform::current().key()
        }

        #[test]
        fn clamps_steps() {
            let mut a = SdfVolume {
                max_steps: 1,
                ..Default::default()
            };
            let fixed = super::super::sdf_volume(a.clone());
            assert_eq!(fixed.max_steps, SDF_MAX_STEPS_FLOOR);

            a.max_steps = 9999;
            let fixed = super::super::sdf_volume(a);
            assert_eq!(fixed.max_steps, SDF_MAX_STEPS_CEILING);
        }

        #[test]
        fn repairs_bad_extent() {
            let a = SdfVolume {
                extent: [0.0, -1.0, f32::NAN],
                ..Default::default()
            };
            let fixed = super::super::sdf_volume(a);
            assert_eq!(fixed.extent, [1.0, 1.0, 1.0]);
        }

        #[test]
        fn repairs_bad_gradient_and_distance() {
            let a = SdfVolume {
                max_gradient: -0.5,
                max_distance: f32::NAN,
                ..Default::default()
            };
            let fixed = super::super::sdf_volume(a);
            assert_eq!(fixed.max_gradient, 1.0);
            assert_eq!(fixed.max_distance, 0.1);
        }

        #[test]
        fn collapses_map_to_current_backend() {
            // The runtime struct should carry the current backend's path in
            // `fragment_shader` so the DirectX path-extension filter still works
            // for map-authored volumes.
            // Include every backend so the collapse resolves regardless of which
            // backend this test build targets (metal / hlsl / glsl).
            let mut map = std::collections::BTreeMap::new();
            map.insert("metal".to_string(), "shaders/blob.metal".to_string());
            map.insert("hlsl".to_string(), "shaders/blob.hlsl".to_string());
            map.insert("glsl".to_string(), "shaders/blob.glsl".to_string());
            let a = SdfVolume {
                fragment_shaders: Some(map),
                ..Default::default()
            };
            let resolved = super::super::sdf_volume(a);
            assert_eq!(
                resolved.fragment_shader,
                format!("shaders/blob.{}", platform_ext())
            );
        }

        #[test]
        fn volumetric_forces_cast_shadows_off() {
            let a = SdfVolume {
                volumetric: true,
                cast_shadows: true,
                ..Default::default()
            };
            let fixed = super::super::sdf_volume(a);
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
            let back = super::super::sdf_volume(back);
            assert_eq!(back.centre, [1.0, 2.0, 3.0]);
            assert_eq!(back.extent, [4.0, 5.0, 6.0]);
            assert_eq!(back.fragment_shader, "shaders/foo.metal");
            assert_eq!(back.params[7], 0.42);
        }
    }
}
