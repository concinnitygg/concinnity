//! Named bake-time validators for the data-only assets. Each function clamps or
//! normalizes an asset's authored value into a self-consistent runtime value.
//! The authoring registry names the function via `validate: <fn>` (or
//! `validate_for: <fn>` when the clamp also reads the shader platform the world
//! is cooked for) and applies it while baking the blob record; a runtime bake
//! applies the same function before installing the value. The runtime never runs these on a loaded
//! world -- a baked record is already validated.

use alloc::string::{String, ToString};

use crate::components::{
    Decal, DirectionalLight, GlassPanel, GlassPanelGeometry, InstancedProp, MAX_WATER_WAVES,
    Material, ParticleEmitter, PhysicsJoint, PhysicsJointKind, PointLight, Prop, RectAreaLight,
    ReflectionProbe, RigidBody, SPOT_MAX_ANGLE_DEG, SdfVolume, SkyRotation, SpotLight,
    SpotLightGeometry, VolumetricFog, VoxelChunk, WaterSurface, WaterWave,
};
use crate::math::sqrt;
use crate::platform::Platform;

// Extension of the file name at the end of `path`, or "" when it has none.
// The no_std stand-in for `std::path::Path::extension`.
fn path_extension(path: &str) -> &str {
    let file = path.rsplit(['/', '\\']).next().unwrap_or(path);
    match file.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() => ext,
        _ => "",
    }
}

// Resolve the fragment shader source path `platform` selects from a volume's
// `fragment_shaders` map (preferred) or its `fragment_shader` fallback.
fn sdf_source_for(v: &SdfVolume, platform: Platform) -> Option<String> {
    if let Some(map) = &v.fragment_shaders
        && let Some(src) = map.get(platform.key()).filter(|s| !s.is_empty())
    {
        return Some(src.clone());
    }
    if v.fragment_shader.is_empty() {
        return None;
    }
    if platform.accepts_ext(path_extension(&v.fragment_shader)) {
        Some(v.fragment_shader.clone())
    } else {
        None
    }
}

/// Normalize an authored volume for the runtime: clamp the raymarch knobs to
/// sane bounds, force shadows off for translucent volumetrics (they write no
/// depth), and collapse the per-backend `fragment_shaders` map to
/// `platform`'s `fragment_shader` (the DirectX raymarch pass filters volumes by
/// that path's extension). The step-count bounds stay with the schema: they
/// double as the runtime kernel's loop bound.
pub fn sdf_volume(mut v: SdfVolume, platform: Platform) -> SdfVolume {
    use crate::components::sdf_volume::{SDF_MAX_STEPS_CEILING, SDF_MAX_STEPS_FLOOR};
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
    if let Some(src) = sdf_source_for(&v, platform) {
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

/// Clamp a `SpotLight`'s authored fields into their valid ranges.
pub fn spot_light(mut args: SpotLight) -> SpotLight {
    args.intensity = args.intensity.max(0.0);
    args.range = args.range.max(0.0);
    args.direction = args.unit_direction();
    args.outer_angle = args.outer_angle.clamp(0.0, SPOT_MAX_ANGLE_DEG);
    args.inner_angle = args.inner_angle.clamp(0.0, args.outer_angle);
    args
}

/// Clamp a `RectAreaLight`'s authored fields into their valid ranges.
pub fn rect_area_light(mut args: RectAreaLight) -> RectAreaLight {
    args.intensity = args.intensity.max(0.0);
    args.range = args.range.max(0.0);
    // A degenerate normal would collapse the panel's tangent frame; a zero
    // half-extent would collapse its area and divide by zero in the integrator.
    let n = args.normal;
    let len = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    args.normal = if len < 1e-6 {
        [0.0, 0.0, 1.0]
    } else {
        [n[0] / len, n[1] / len, n[2] / len]
    };
    args.half_size[0] = args.half_size[0].max(1e-3);
    args.half_size[1] = args.half_size[1].max(1e-3);
    args
}

/// Clamp a `DirectionalLight`'s authored fields into their valid ranges.
pub fn directional_light(mut args: DirectionalLight) -> DirectionalLight {
    args.intensity = args.intensity.max(0.0);
    args
}

/// Replace a `SkyRotation`'s degenerate axis with the default pole: an axis
/// with no direction would leave the sky in a fixed orientation whatever the
/// rate said.
pub fn sky_rotation(mut args: SkyRotation) -> SkyRotation {
    let [x, y, z] = args.axis;
    if sqrt(x * x + y * y + z * z) < 1e-6 {
        args.axis = SkyRotation::default().axis;
    }
    args
}

/// Clamp a `Material`'s authored fields into their valid ranges. Material is a
/// data resource, not a registered component, so no generated `from_args` runs
/// this -- the material compilers call it explicitly before baking the bytes.
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

/// Clamp a `GlassPanel`'s authored fields into their valid ranges.
pub fn glass_panel(mut args: GlassPanel) -> GlassPanel {
    args.normal = args.unit_normal();
    args.half_size[0] = args.half_size[0].max(1e-3);
    args.half_size[1] = args.half_size[1].max(1e-3);
    args.opacity = args.opacity.clamp(0.0, 1.0);
    args.refraction_strength = args.refraction_strength.max(0.0);
    args.fresnel_power = args.fresnel_power.max(0.0);
    args
}

/// Clamp a `WaterSurface`'s authored fields into their valid ranges.
pub fn water_surface(mut args: WaterSurface) -> WaterSurface {
    args.subdivisions = args.subdivisions.clamp(8, 255);
    if args.waves.len() > MAX_WATER_WAVES {
        args.waves.truncate(MAX_WATER_WAVES);
    }
    if args.waves.is_empty() {
        args.waves.push(WaterWave::default());
    }
    args
}

/// Clamp a `PhysicsJoint`'s authored fields into their valid ranges.
pub fn joint(mut args: PhysicsJoint) -> PhysicsJoint {
    // Normalise the kind string so `to_args` round-trips cleanly.
    if let Some(k) = PhysicsJointKind::from_str_norm(&args.kind) {
        args.kind = k.as_str().to_string();
    }
    args
}

/// Clamp a `Decal`'s authored fields into their valid ranges.
pub fn decal(mut args: Decal) -> Decal {
    // Clamp the alpha to [0, 1] so a stray > 1 doesn't blow out the
    // composite. The size components are left as-authored: a non-positive
    // value silently disables the decal in the gfx-side resolver.
    args.tint[3] = args.tint[3].clamp(0.0, 1.0);
    args
}

/// Clamp a `ReflectionProbe`'s authored fields into their valid ranges.
pub fn reflection_probe(mut args: ReflectionProbe) -> ReflectionProbe {
    // Half-extents are sizes: keep them non-negative so the influence box is
    // never inverted.
    for e in &mut args.half_extents {
        *e = e.max(0.0);
    }
    args
}

/// Reset a `RigidBody`'s runtime state on construction.
pub fn rigid_body(mut args: RigidBody) -> RigidBody {
    args.is_grounded = true;
    args
}

/// Clamp a `Prop`'s authored fields into their valid ranges.
pub fn prop(mut args: Prop) -> Prop {
    args.cull_distance = args.cull_distance.max(0.0);
    args.is_held = false;
    args
}

/// Clamp a `ParticleEmitter`'s authored fields into their valid ranges.
pub fn particle_emitter(mut args: ParticleEmitter) -> ParticleEmitter {
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

/// Clamp an `InstancedProp`'s authored fields into their valid ranges.
pub fn instanced_prop(mut args: InstancedProp) -> InstancedProp {
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
    use super::*;

    #[test]
    fn material_clamps_and_see_through_implies_transparent() {
        let m = material(Material {
            roughness: 2.0,
            metallic: -1.0,
            opacity: 1.5,
            see_through: true,
            ..Material::default()
        });
        assert_eq!(m.roughness, 1.0);
        assert_eq!(m.metallic, 0.0);
        assert_eq!(m.opacity, 1.0);
        assert!(m.transparent);
    }

    #[test]
    fn water_surface_clamps_subdivisions_and_guarantees_a_wave() {
        let w = water_surface(WaterSurface {
            subdivisions: 3,
            waves: alloc::vec::Vec::new(),
            ..WaterSurface::default()
        });
        assert_eq!(w.subdivisions, 8);
        assert_eq!(w.waves.len(), 1);
    }

    #[test]
    fn lights_floor_intensity_and_range_at_zero() {
        let p = point_light(PointLight {
            intensity: -2.0,
            range: -1.0,
            ..PointLight::default()
        });
        assert_eq!((p.intensity, p.range), (0.0, 0.0));
        let d = directional_light(DirectionalLight {
            intensity: -1.0,
            ..DirectionalLight::default()
        });
        assert_eq!(d.intensity, 0.0);
    }

    #[test]
    fn path_extension_matches_file_name_semantics() {
        assert_eq!(path_extension("shaders/blob.metal"), "metal");
        assert_eq!(path_extension("blob.hlsl"), "hlsl");
        assert_eq!(path_extension("dir.v2/shader"), "");
        assert_eq!(path_extension(".hidden"), "");
        assert_eq!(path_extension("noext"), "");
    }
}
