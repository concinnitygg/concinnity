//! Named bake-time validators for the data-only assets. Each function clamps or
//! normalizes an asset's authored value into a self-consistent runtime value.
//! The authoring registry names the function via `validate: <fn>` and applies
//! it while baking the blob record; a runtime bake
//! applies the same function before installing the value. The runtime never runs these on a loaded
//! world -- a baked record is already validated.

use crate::components::{
    Atmosphere, CameraTrackArgs, Decal, DirectionalLight, GlassPanel, Grass, InstancedProp,
    MAX_WATER_WAVES, Material, NightSky, ParticleEmitter, Planet, PointLight, Prop, RectAreaLight,
    ReflectionProbe, RigidBody, SPOT_MAX_ANGLE_DEG, SdfVolume, SkyRotation, SpotLight, Terrain,
    VolumetricFog, VoxelChunk, WaterShape, WaterSurface, WaterWave, Wind,
};
use crate::math::SaturatingFrom;
use crate::math::sqrt;
use crate::math::vec3::length;

/// Normalize an authored volume for the runtime: clamp the raymarch knobs to
/// sane bounds and force shadows off for translucent volumetrics, which write
/// no depth to cast from. The step-count bounds stay with the schema: they
/// double as the runtime kernel's loop bound.
pub fn sdf_volume(mut v: SdfVolume) -> SdfVolume {
    use crate::render::sdf::volume::{SDF_MAX_STEPS_CEILING, SDF_MAX_STEPS_FLOOR};
    // Extents must be positive: a zero or negative extent would produce an
    // inside-out bounding box no fragment ever enters.
    for axis in &mut v.extent {
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
    let len = length(n);
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

/// Normalize an authored `CameraTrack` so every leg has a duration the
/// timeline can lay out: negative and non-finite quantities become zero, and a
/// leg with no direction to travel in becomes a hold of the duration its
/// distance and speed implied, so the legs after it stay where the author put
/// them on the clock. A turn angle that is not a real one is dropped, leaving
/// that leg holding the angle it started from. `CameraTrack` bakes divergently, so no generated
/// `from_args` runs this -- `CameraTrack::bake` calls it.
pub fn camera_track(mut args: CameraTrackArgs) -> CameraTrackArgs {
    for leg in &mut args.travel {
        leg.distance = finite_non_negative(leg.distance);
        leg.speed = finite_non_negative(leg.speed);
        leg.seconds = finite_non_negative(leg.seconds);
        for axis in &mut leg.direction {
            if !axis.is_finite() {
                *axis = 0.0;
            }
        }
        if length(leg.direction) < 1e-6 {
            leg.seconds = crate::camera::track::travel_seconds(leg);
            leg.distance = 0.0;
        }
    }
    for leg in &mut args.turn {
        leg.yaw_deg = leg.yaw_deg.filter(|a| a.is_finite());
        leg.pitch_deg = leg.pitch_deg.filter(|a| a.is_finite());
        leg.degrees_per_second = finite_non_negative(leg.degrees_per_second);
        leg.seconds = finite_non_negative(leg.seconds);
    }
    args
}

// Zero for anything that is not a usable non-negative magnitude.
fn finite_non_negative(v: f32) -> f32 {
    if v.is_finite() { v.max(0.0) } else { 0.0 }
}

/// Clamp a `Material`'s authored fields into their valid ranges. Material is a
/// data resource, not a registered component, so no generated `from_args` runs
/// this -- the material compilers call it explicitly before baking the bytes.
pub fn material(mut args: Material) -> Material {
    args.roughness = args.roughness.clamp(0.0, 1.0);
    args.metallic = args.metallic.clamp(0.0, 1.0);
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
    args.fetch_meters = finite_at_least(args.fetch_meters, 1.0, 1000.0);
    args.wind_response = finite_at_least(args.wind_response, 0.0, 1.0);
    args.whitecaps = finite_at_least(args.whitecaps, 0.0, 0.0).min(1.0);
    args.sea_level = finite_at_least(args.sea_level, 0.0, 0.0);
    if args.shape == WaterShape::PlanetSea {
        args.waves.clear();
    }
    args.waves.truncate(MAX_WATER_WAVES);
    for w in &mut args.waves {
        *w = water_wave(*w);
    }
    args
}

// Clamp one authored swell wave: a non-negative height, a wavelength of at
// least a centimeter, a unit direction (+X when it has no length) and a
// steepness in [0, 1].
fn water_wave(mut w: WaterWave) -> WaterWave {
    w.amplitude = finite_at_least(w.amplitude, 0.0, 0.0);
    w.wavelength = finite_at_least(w.wavelength, 0.01, 4.0);
    let [x, z] = w.direction;
    let len = crate::math::hypot(x, z);
    w.direction = if len > 1e-6 && len.is_finite() {
        [x / len, z / len]
    } else {
        [1.0, 0.0]
    };
    w.steepness = finite_at_least(w.steepness, 0.0, 0.0).min(1.0);
    w
}

/// Clamp a `Wind`'s authored fields into their valid ranges.
pub fn wind(mut args: Wind) -> Wind {
    args.direction = args.unit_direction();
    args.strength = finite_at_least(args.strength, 0.0, 0.0);
    args.gustiness = finite_at_least(args.gustiness, 0.0, 0.0).min(1.0);
    args.gust_scale = finite_at_least(args.gust_scale, 0.5, 20.0);
    args
}

/// Clamp a `Grass` field's authored values into their valid ranges.
pub fn grass(mut args: Grass) -> Grass {
    args.height = finite_at_least(args.height, 0.01, 0.5);
    args.height_variance = finite_at_least(args.height_variance, 0.0, 0.0).min(1.0);
    args.width = finite_at_least(args.width, 0.001, 0.03);
    args.density = finite_at_least(args.density, 0.0, 0.0);
    args.clump_size = finite_at_least(args.clump_size, 0.05, 0.6);
    args.clumping = finite_at_least(args.clumping, 0.0, 0.0).min(1.0);
    args.stiffness = finite_at_least(args.stiffness, 0.0, 0.0).min(1.0);
    args.color_variation = finite_at_least(args.color_variation, 0.0, 0.0).min(1.0);
    args.macro_variation = finite_at_least(args.macro_variation, 0.0, 0.0).min(1.0);
    for c in args
        .root_color
        .iter_mut()
        .chain(args.tip_color.iter_mut())
        .chain(args.litter_color.iter_mut().flatten())
    {
        *c = finite_at_least(*c, 0.0, 0.0);
    }
    args
}

/// Clamp a `Terrain`'s authored values into their valid ranges.
pub fn terrain(mut args: Terrain) -> Terrain {
    for c in &mut args.center {
        *c = finite_or(*c, 0.0);
    }
    for e in &mut args.extent {
        *e = finite_at_least(*e, crate::terrain::MIN_TERRAIN_EXTENT, 50.0);
    }
    args.resolution = crate::terrain::clamp_resolution(args.resolution);
    args.amplitude = finite_at_least(args.amplitude, 0.0, 0.0);
    args.elevation_min = finite_or(args.elevation_min, 0.0);
    args.elevation_max = finite_or(args.elevation_max, args.elevation_min);
    args
}

/// Clamp a `Planet`'s authored values into their valid ranges.
pub fn planet(mut args: Planet) -> Planet {
    let defaults = Planet::default();
    for (c, d) in args.center.iter_mut().zip(defaults.center) {
        *c = finite_or(*c, d);
    }
    args.radius = finite_at_least(
        args.radius,
        crate::planet::MIN_PLANET_RADIUS,
        defaults.radius,
    );
    args.amplitude = finite_at_least(args.amplitude, 0.0, 0.0).min(args.radius * 0.25);
    args.feature_size = finite_at_least(args.feature_size, 1.0, defaults.feature_size);
    args.octaves = args
        .octaves
        .clamp(crate::planet::MIN_OCTAVES, crate::planet::MAX_OCTAVES);
    args.gravity = finite_at_least(args.gravity, 0.0, defaults.gravity);
    args
}

/// Clamp an `Atmosphere`'s authored values into their valid ranges: lengths
/// positive, coefficients and albedo non-negative, the haze's anisotropy short
/// of a singular phase function and the sun's disc a real size.
pub fn atmosphere(mut args: Atmosphere) -> Atmosphere {
    let defaults = Atmosphere::default();
    args.height_km = finite_at_least(args.height_km, 1.0, defaults.height_km);
    args.planet_radius_km = finite_at_least(args.planet_radius_km, 1.0, defaults.planet_radius_km);
    args.rayleigh_scale_height_km = finite_at_least(
        args.rayleigh_scale_height_km,
        0.01,
        defaults.rayleigh_scale_height_km,
    );
    args.mie_scale_height_km =
        finite_at_least(args.mie_scale_height_km, 0.01, defaults.mie_scale_height_km);
    args.ozone_center_km = finite_at_least(args.ozone_center_km, 0.0, defaults.ozone_center_km);
    args.ozone_width_km = finite_at_least(args.ozone_width_km, 0.01, defaults.ozone_width_km);
    for c in args
        .rayleigh_scattering
        .iter_mut()
        .chain(args.mie_scattering.iter_mut())
        .chain(args.mie_absorption.iter_mut())
        .chain(args.ozone_absorption.iter_mut())
        .chain(args.ground_albedo.iter_mut())
    {
        *c = finite_at_least(*c, 0.0, 0.0);
    }
    args.mie_anisotropy =
        finite_or(args.mie_anisotropy, defaults.mie_anisotropy).clamp(-0.99, 0.99);
    args.sun_angular_diameter_deg = finite_or(
        args.sun_angular_diameter_deg,
        defaults.sun_angular_diameter_deg,
    )
    .clamp(0.05, 10.0);
    args.sun_disc_intensity = finite_at_least(args.sun_disc_intensity, 0.0, 0.0);
    args.apparent_radius_km = args
        .apparent_radius_km
        .filter(|r| r.is_finite())
        .map(|r| r.max(1.0));
    let [start, end] = args.apparent_radius_fade_km;
    let start = finite_at_least(start, 0.0, defaults.apparent_radius_fade_km[0]);
    let end = finite_at_least(end, start + 0.01, start + 0.01);
    args.apparent_radius_fade_km = [start, end];
    args
}

/// Clamp a `NightSky`'s authored values into their valid ranges:
/// brightnesses non-negative, the faintest magnitude within what the star
/// grid holds, at most the galaxies the sky's shaders carry, and the moon a
/// real size on a path that runs forward.
pub fn night_sky(mut args: NightSky) -> NightSky {
    let defaults = NightSky::default();
    let limits = crate::render::environment::night_sky::LIMIT_RANGE;
    args.star_brightness = finite_at_least(args.star_brightness, 0.0, 0.0);
    args.star_limit_magnitude = finite_or(args.star_limit_magnitude, defaults.star_limit_magnitude)
        .clamp(limits[0], limits[1]);
    args.milky_way_brightness = finite_at_least(args.milky_way_brightness, 0.0, 0.0);
    args.milky_way_tilt_deg =
        finite_or(args.milky_way_tilt_deg, defaults.milky_way_tilt_deg).clamp(0.0, 180.0);
    args.milky_way_turn_deg = finite_or(args.milky_way_turn_deg, 0.0);
    args.milky_way_core_deg = finite_or(args.milky_way_core_deg, 0.0);
    args.galaxies = args.galaxies.min(u32::saturating_from(
        crate::render::environment::night_sky::MAX_GALAXIES,
    ));
    args.moon_start_deg = finite_or(args.moon_start_deg, defaults.moon_start_deg);
    args.moon_period_days = finite_at_least(args.moon_period_days, 0.0, 0.0);
    args.moon_inclination_deg = finite_or(args.moon_inclination_deg, 0.0).clamp(-90.0, 90.0);
    args.moon_angular_diameter_deg = finite_or(
        args.moon_angular_diameter_deg,
        defaults.moon_angular_diameter_deg,
    )
    .clamp(0.05, 20.0);
    args.moon_brightness = finite_at_least(args.moon_brightness, 0.0, 0.0);
    args.moon_disc_brightness = finite_at_least(args.moon_disc_brightness, 0.0, 0.0);
    args
}

// `value`, or `fallback` when it is not a finite number.
fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

// `value` floored at `min`, or `fallback` when it is not a finite number.
fn finite_at_least(value: f32, min: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.max(min)
    } else {
        fallback
    }
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
    args.capture_distance = args.capture_distance.map(|d| d.max(0.0));
    args.blend_distance = args.blend_distance.map(|d| d.max(0.0));
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
    fn water_surface_clamps_its_fields_and_swell() {
        let swell = WaterWave {
            amplitude: -1.0,
            wavelength: 0.0,
            direction: [0.0, 0.0],
            steepness: 3.0,
        };
        let w = water_surface(WaterSurface {
            subdivisions: 3,
            fetch_meters: f32::NAN,
            wind_response: -2.0,
            whitecaps: 3.0,
            waves: alloc::vec![swell; 6],
            ..WaterSurface::default()
        });
        assert_eq!(w.subdivisions, 8);
        assert_eq!(w.fetch_meters, 1000.0);
        assert_eq!(w.wind_response, 0.0);
        assert_eq!(w.whitecaps, 1.0);
        assert_eq!(w.waves.len(), MAX_WATER_WAVES);
        let s = w.waves[0];
        assert_eq!(
            (s.amplitude, s.wavelength, s.direction, s.steepness),
            (0.0, 0.01, [1.0, 0.0], 1.0)
        );
        assert!(WaterSurface::default().waves.is_empty());
        for (authored, clamped) in [(-0.5, 0.0), (f32::NAN, 0.0), (0.4, 0.4)] {
            let w = water_surface(WaterSurface {
                whitecaps: authored,
                ..WaterSurface::default()
            });
            assert_eq!(w.whitecaps, clamped, "{authored}");
        }
    }

    // A planet sea's height is never below the planet's radius, and it
    // carries no swell.
    #[test]
    fn a_planet_sea_clamps_its_level_and_drops_swell() {
        for (authored, clamped) in [(-3.0, 0.0), (f32::NAN, 0.0), (12.5, 12.5)] {
            let w = water_surface(WaterSurface {
                shape: WaterShape::PlanetSea,
                sea_level: authored,
                waves: alloc::vec![WaterWave::default()],
                ..WaterSurface::default()
            });
            assert_eq!(w.sea_level, clamped, "{authored}");
            assert!(w.waves.is_empty());
        }
        let rect = water_surface(WaterSurface {
            waves: alloc::vec![WaterWave::default()],
            ..WaterSurface::default()
        });
        assert_eq!(rect.shape, WaterShape::Rectangle);
        assert_eq!(rect.waves.len(), 1);
    }

    // A surface whitecaps only when it asks to.
    #[test]
    fn water_surface_whitecaps_are_off_by_default() {
        assert_eq!(WaterSurface::default().whitecaps, 0.0);
    }

    #[test]
    fn wind_normalizes_its_direction_and_floors_its_speeds() {
        let w = wind(Wind {
            direction: [3.0, 4.0],
            strength: -1.0,
            gustiness: 2.0,
            gust_scale: 0.0,
        });
        assert_eq!(w.direction, [0.6, 0.8]);
        assert_eq!(w.strength, 0.0);
        assert_eq!(w.gustiness, 1.0);
        assert_eq!(w.gust_scale, 0.5);
        let still = wind(Wind {
            direction: [0.0, 0.0],
            strength: f32::NAN,
            ..Wind::default()
        });
        assert_eq!(still.direction, [1.0, 0.0]);
        assert_eq!(still.strength, 0.0);
    }

    #[test]
    fn grass_clamps_every_blade_parameter() {
        let g = grass(Grass {
            height: 0.0,
            height_variance: 3.0,
            width: -0.5,
            density: -10.0,
            clump_size: 0.0,
            clumping: 2.0,
            stiffness: 1.5,
            root_color: [-1.0, f32::NAN, 0.5],
            tip_color: [0.2, 0.3, 0.4],
            litter_color: Some([f32::INFINITY, -0.1, 0.05]),
            color_variation: -0.2,
            ..Grass::default()
        });
        assert_eq!(g.height, 0.01);
        assert_eq!(g.height_variance, 1.0);
        assert_eq!(g.width, 0.001);
        assert_eq!(g.density, 0.0);
        assert_eq!(g.clump_size, 0.05);
        assert_eq!(g.clumping, 1.0);
        assert_eq!(g.litter_color, Some([0.0, 0.0, 0.05]));
        assert_eq!(g.stiffness, 1.0);
        assert_eq!(g.root_color, [0.0, 0.0, 0.5]);
        assert_eq!(g.tip_color, [0.2, 0.3, 0.4]);
        assert_eq!(g.color_variation, 0.0);
    }

    #[test]
    fn grass_defaults_pass_validation_unchanged() {
        let d = Grass::default();
        let g = grass(d.clone());
        assert_eq!(
            (g.height, g.width, g.density, g.clump_size, g.clumping),
            (d.height, d.width, d.density, d.clump_size, d.clumping)
        );
        assert_eq!(g.litter_color, None);
        // The default field gathers, short of the strongest clumping.
        assert!(d.clumping > 0.0 && d.clumping < 1.0);
        let w = wind(Wind::default());
        assert_eq!(
            (w.direction, w.strength, w.gustiness, w.gust_scale),
            (
                Wind::default().direction,
                Wind::default().strength,
                Wind::default().gustiness,
                Wind::default().gust_scale
            )
        );
    }

    #[test]
    fn terrain_clamps_its_grid_and_heights() {
        let t = terrain(Terrain {
            center: [f32::NAN, 2.0, 3.0],
            extent: [-4.0, f32::INFINITY],
            resolution: 1,
            amplitude: -1.0,
            elevation_min: f32::NAN,
            elevation_max: f32::INFINITY,
            ..Terrain::default()
        });
        assert_eq!(t.center, [0.0, 2.0, 3.0]);
        assert_eq!(
            t.extent,
            [crate::terrain::MIN_TERRAIN_EXTENT, 50.0],
            "a negative extent floors, a non-finite one takes the default"
        );
        assert_eq!(t.resolution, crate::terrain::MIN_TERRAIN_RESOLUTION);
        assert_eq!(t.amplitude, 0.0);
        assert_eq!((t.elevation_min, t.elevation_max), (0.0, 0.0));
        let huge = terrain(Terrain {
            resolution: u32::MAX,
            ..Terrain::default()
        });
        assert_eq!(huge.resolution, crate::terrain::MAX_TERRAIN_RESOLUTION);
    }

    #[test]
    fn terrain_defaults_pass_validation_unchanged() {
        let d = Terrain::default();
        let t = terrain(d.clone());
        assert_eq!(
            (t.extent, t.resolution, t.amplitude, t.elevation_max),
            (d.extent, d.resolution, d.amplitude, d.elevation_max)
        );
    }

    #[test]
    fn a_night_sky_is_clamped_into_range() {
        let n = night_sky(NightSky {
            star_limit_magnitude: 30.0,
            galaxies: 99,
            moon_period_days: -4.0,
            moon_angular_diameter_deg: f32::NAN,
            milky_way_brightness: -1.0,
            ..NightSky::default()
        });
        assert_eq!(n.star_limit_magnitude, 8.0);
        assert_eq!(n.galaxies, 16);
        assert_eq!(n.moon_period_days, 0.0);
        assert_eq!(n.moon_angular_diameter_deg, 0.52);
        assert_eq!(n.milky_way_brightness, 0.0);
        assert_eq!(night_sky(NightSky::default()), NightSky::default());
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

    // An apparent radius is floored at 1 km and a non-finite one dropped; the
    // fade never ends before it starts.
    #[test]
    fn atmosphere_apparent_radius_is_sane() {
        let a = atmosphere(Atmosphere {
            apparent_radius_km: Some(0.2),
            apparent_radius_fade_km: [5.0, 2.0],
            ..Atmosphere::default()
        });
        assert_eq!(a.apparent_radius_km, Some(1.0));
        assert_eq!(a.apparent_radius_fade_km, [5.0, 5.0 + 0.01]);
        let a = atmosphere(Atmosphere {
            apparent_radius_km: Some(f32::NAN),
            apparent_radius_fade_km: [f32::NAN, 30.0],
            ..Atmosphere::default()
        });
        assert_eq!(a.apparent_radius_km, None);
        assert_eq!(a.apparent_radius_fade_km, [1.0, 30.0]);
    }
}
