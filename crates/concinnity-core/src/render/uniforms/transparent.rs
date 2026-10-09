//! What the transparent pass binds: the shared per-frame view block, and the
//! per-record tunables of its producers -- glass panes, see-through glass
//! meshes, and water surfaces.

use super::PassCamera;
use crate::components::{GlassPanel, WaterSurface, WaterWave};
use crate::gfx::render_types::LightUniforms;
use crate::render::lights::glint_sun;

/// Per-frame view inputs shared by every draw in the transparent pass (water,
/// glass), bound once for the whole pass. Matches `TransparentView` in
/// `shaders/glass.hlsl` and `shaders/water.hlsl`. 240 bytes.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct TransparentView {
    /// View-projection matrix, column-major.
    pub vp: [[f32; 4]; 4],
    /// Inverse view-projection matrix, column-major.
    pub inv_vp: [[f32; 4]; 4],
    /// World-space camera position (xyz). `.w` is ignored by the shader.
    pub camera_pos: [f32; 4],
    /// Render-target width / height in pixels: the shader uses this to
    /// turn its fragment position into a normalized screen UV.
    pub viewport: [f32; 2],
    /// Wall-clock seconds since startup, fed to the Gerstner sum.
    pub time: f32,
    /// Mip count of the bound IBL prefilter cube; 0 signals "no environment map",
    /// where the glass reflection falls back to a white rim. Per-frame state, so
    /// it rides the shared view rather than a per-draw params block.
    pub prefilter_mip_count: f32,
    /// Rows of the rotation taking a world-space direction into the environment
    /// cubemap's baked frame, mirroring `ViewUniforms.sky_rot` so this pass's
    /// sky taps turn with the main one. One `float4` per row; `w` is unused.
    pub sky_rot: [[f32; 4]; 3],
    /// `[x, y, z, _]`: unit direction toward the scene's sun, the first
    /// directional light. Zero when the world declares none.
    pub sun_dir: [f32; 4],
    /// `[r, g, b, _]`: that light's color times its intensity, which the water
    /// glint scales by. Zero when the world declares no directional light, and
    /// the shader draws no glint.
    pub sun_color: [f32; 4],
}

impl TransparentView {
    /// The view block for `camera`, lit by the sun `lights` glints with.
    pub fn new(camera: &PassCamera, lights: &LightUniforms) -> Self {
        let c = camera.cam_pos;
        let (sun_dir, sun_color) = glint_sun(lights);
        Self {
            vp: camera.vp,
            inv_vp: camera.inv_vp,
            camera_pos: [c[0], c[1], c[2], 0.0],
            viewport: camera.viewport,
            time: camera.time,
            prefilter_mip_count: camera.prefilter_mip_count,
            sky_rot: camera.sky_rot,
            sun_dir,
            sun_color,
        }
    }
}

/// Per-panel tunables for a `GlassPanel`, uploaded once per panel per frame.
/// The vec3-ish fields are `[f32; 4]` so the layout is byte-identical to the
/// shader's `float4`. Matches `GlassParams` in `shaders/glass.hlsl`. 64 bytes.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct GlassParams {
    /// `[x, y, z, _]`: world-space panel center.
    pub center: [f32; 4],
    /// `[nx, ny, nz, _]`: unit panel normal (facing direction).
    pub normal: [f32; 4],
    /// `[r, g, b, _]`: color multiplied into the refracted scene.
    pub tint: [f32; 4],
    /// Base alpha at normal incidence.
    pub opacity: f32,
    /// Screen-space refraction offset strength.
    pub refraction_strength: f32,
    /// Schlick-Fresnel exponent for the grazing-angle rim.
    pub fresnel_power: f32,
    /// Planar reflection strength: `> 0.5` selects the sharp planar reflection
    /// (the scene re-rendered mirrored across this pane's plane, sampled
    /// projectively at screen UV) over the probe / sky cube. 0 when planar is off
    /// (RT on, no planar slot, or the plane overflowed the budget), keeping the
    /// probe / sky path. Patched per-frame in `collect_glass_transparent_draws`.
    pub planar: f32,
}

impl GlassParams {
    /// The `planar` lane: 1 selects the pane's planar reflection, 0 keeps the
    /// probe / sky path.
    pub fn planar_lane(mirrored: bool) -> f32 {
        if mirrored { 1.0 } else { 0.0 }
    }

    /// The per-panel block for an authored panel. `planar` is whether the pane
    /// holds a planar reflection slot, see [`GlassParams::planar_lane`].
    pub fn from_panel(panel: &GlassPanel, planar: bool) -> Self {
        let n = panel.normal; // already unit-length from GlassPanel::from_args
        Self {
            center: [panel.center[0], panel.center[1], panel.center[2], 0.0],
            normal: [n[0], n[1], n[2], 0.0],
            tint: [panel.tint[0], panel.tint[1], panel.tint[2], 0.0],
            opacity: panel.opacity,
            refraction_strength: panel.refraction_strength,
            fresnel_power: panel.fresnel_power,
            planar: Self::planar_lane(planar),
        }
    }
}

/// Per-draw tunables for a see-through glass MESH: a `Material` flagged
/// `see_through` on an RT-capable device, drawn in the transparent pass instead
/// of the opaque one. Unlike `GlassParams` (a pre-baked world-space pane), a
/// mesh is LOCAL-space, so this carries the model matrix the vertex stage
/// applies; the fragment uses the interpolated per-vertex world normal. Matches
/// `GlassMeshParams` in `shaders/glass_mesh.hlsl`. 96 bytes (model is the first
/// field, so its 16-byte GPU alignment is satisfied at offset 0).
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct GlassMeshParams {
    /// Column-major local-to-world model matrix.
    pub model: [[f32; 4]; 4],
    /// `[r, g, b, _]`: color multiplied into the refracted scene (material tint).
    pub tint: [f32; 4],
    /// Base alpha at normal incidence (from `Material.opacity`).
    pub opacity: f32,
    /// Screen-space refraction offset strength.
    pub refraction_strength: f32,
    /// Schlick-Fresnel exponent for the grazing-angle rim.
    pub fresnel_power: f32,
    /// Mip count of the bound IBL prefilter cube (ray-miss fallback); 0 = none.
    pub prefilter_mip_count: f32,
}

/// Maximum waves summed per `WaterParams`: the `WaterSurface` asset's limit.
/// Mirrors `MAX_WATER_WAVES` in `shaders/water.hlsl`.
pub const WATER_MAX_WAVES: usize = crate::components::MAX_WATER_WAVES;

/// One Gerstner wave coefficient set, packed into two `float4` lanes so the
/// layout is identical on every target. Matches `WaterWave` in
/// `shaders/water.hlsl`. 32 bytes.
#[derive(Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct WaterWaveGpu {
    /// `[direction.x, direction.y, amplitude, wavelength]`.
    pub dir_amp_wave: [f32; 4],
    /// `[speed, steepness, _, _]`.
    pub speed_steep_pad: [f32; 4],
}

impl WaterWaveGpu {
    /// The shader-side lanes for one authored wave.
    pub fn from_wave(w: &WaterWave) -> Self {
        Self {
            dir_amp_wave: [w.direction[0], w.direction[1], w.amplitude, w.wavelength],
            speed_steep_pad: [w.speed, w.steepness, 0.0, 0.0],
        }
    }
}

/// Per-surface tunables for a `WaterSurface`, uploaded once per surface. The
/// vec3-ish fields are `[f32; 4]` so the layout is byte-identical to the
/// shader's `float4`. Matches `WaterParams` in `shaders/water.hlsl`. 224 bytes.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct WaterParams {
    /// `[x, y, z, _]`: world-space surface center.
    pub center: [f32; 4],
    /// `[r, g, b, _]`: water tint at full column depth.
    pub deep_color: [f32; 4],
    /// `[r, g, b, _]`: water tint just above the seabed.
    pub shallow_color: [f32; 4],
    /// Depth over which the tint blends from shallow to deep, in meters.
    pub depth_falloff: f32,
    /// Width of the shoreline foam band, in world units.
    pub foam_width: f32,
    /// Foam brightness multiplier.
    pub foam_intensity: f32,
    /// Exponent of the Fresnel reflectance curve.
    pub fresnel_power: f32,
    /// Perceptual roughness in `[0, 1]`; picks the reflection's prefilter mip.
    pub roughness: f32,
    /// How far refraction offsets the sampled background.
    pub refraction_strength: f32,
    /// Live entries in `waves`.
    pub wave_count: u32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad: f32,
    /// Wave coefficients; the first `wave_count` entries are live.
    pub waves: [WaterWaveGpu; WATER_MAX_WAVES],
    /// Planar reflection control: `[strength, distortion, _, _]`. `strength >
    /// 0.5` selects the sharp planar reflection (the scene re-rendered mirrored
    /// across this surface's rest plane, sampled at screen UV) over the probe /
    /// sky cube; `distortion` scales the wave-normal ripple offset of that
    /// lookup, see [`WaterParams::planar_lane`]. 0 when planar is off (RT on,
    /// no planar slot, or the plane overflowed the budget), keeping the probe /
    /// sky path.
    pub planar: [f32; 4],
}

// How far the mirror lookup is pushed per unit of wave slope, per unit of the
// surface's authored roughness, and the ceiling a very rough surface stops at.
// The planar target is a flat-plane render, so the offset only fakes the
// ripple; a near-mirror surface barely moves it, a choppy one moves it more.
const PLANAR_DISTORTION_PER_ROUGHNESS: f32 = 0.6;
const PLANAR_DISTORTION_MAX: f32 = 0.06;

impl WaterParams {
    /// The `planar` lane for a surface of `roughness`: the mirror selected
    /// with its ripple offset scaled by the roughness when `mirrored`, else
    /// zeroed so the shader keeps the probe / sky path.
    pub fn planar_lane(roughness: f32, mirrored: bool) -> [f32; 4] {
        if !mirrored {
            return [0.0; 4];
        }
        let distortion =
            (roughness.max(0.0) * PLANAR_DISTORTION_PER_ROUGHNESS).min(PLANAR_DISTORTION_MAX);
        [1.0, distortion, 0.0, 0.0]
    }

    /// The per-surface block for an authored surface, keeping at most
    /// [`WATER_MAX_WAVES`] of its waves. `planar` is whether the surface holds
    /// a planar reflection slot, see [`WaterParams::planar_lane`].
    pub fn from_surface(surface: &WaterSurface, planar: bool) -> Self {
        let mut waves = [WaterWaveGpu::default(); WATER_MAX_WAVES];
        for (slot, src) in waves.iter_mut().zip(surface.waves.iter()) {
            *slot = WaterWaveGpu::from_wave(src);
        }
        Self {
            center: [surface.center[0], surface.center[1], surface.center[2], 0.0],
            deep_color: [
                surface.deep_color[0],
                surface.deep_color[1],
                surface.deep_color[2],
                0.0,
            ],
            shallow_color: [
                surface.shallow_color[0],
                surface.shallow_color[1],
                surface.shallow_color[2],
                0.0,
            ],
            depth_falloff: surface.depth_falloff_meters,
            foam_width: surface.foam_width_meters,
            foam_intensity: surface.foam_intensity,
            fresnel_power: surface.fresnel_power,
            roughness: surface.roughness,
            refraction_strength: surface.refraction_strength,
            wave_count: surface.waves.len().min(WATER_MAX_WAVES) as u32,
            _pad: 0.0,
            waves,
            planar: Self::planar_lane(surface.roughness, planar),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use core::mem::{offset_of, size_of};

    #[test]
    fn the_wave_count_matches_the_shader() {
        let src = crate::render::shaders::WATER;
        assert_eq!(
            crate::render::shader_consts::uint(src, "MAX_WATER_WAVES"),
            WATER_MAX_WAVES
        );
    }

    // Every backend binds this block under the same layout, so it is checked
    // here rather than per backend: a float4x4 model, a float4 tint, then four
    // scalars. `model` is first, so its 16-byte GPU alignment is satisfied at
    // offset 0 and the Rust `[[f32; 4]; 4]` matches byte-for-byte.
    #[test]
    fn glass_mesh_params_layout_matches_shader() {
        assert_eq!(size_of::<GlassMeshParams>(), 96);
        assert_eq!(offset_of!(GlassMeshParams, model), 0);
        assert_eq!(offset_of!(GlassMeshParams, tint), 64);
        assert_eq!(offset_of!(GlassMeshParams, opacity), 80);
        assert_eq!(offset_of!(GlassMeshParams, refraction_strength), 84);
        assert_eq!(offset_of!(GlassMeshParams, fresnel_power), 88);
        assert_eq!(offset_of!(GlassMeshParams, prefilter_mip_count), 92);
        assert_eq!(size_of::<GlassMeshParams>() % 16, 0);
    }

    #[test]
    fn the_view_block_carries_the_camera_and_the_glint_sun() {
        let camera = super::super::view::test_camera();
        let lights = LightUniforms::DEFAULT;
        let view = TransparentView::new(&camera, &lights);
        assert_eq!(view.vp, camera.vp);
        assert_eq!(view.inv_vp, camera.inv_vp);
        assert_eq!(view.camera_pos, [3.0, 4.0, 5.0, 0.0]);
        assert_eq!(view.viewport, camera.viewport);
        assert_eq!(view.time, camera.time);
        assert_eq!(view.prefilter_mip_count, camera.prefilter_mip_count);
        assert_eq!(view.sky_rot, camera.sky_rot);
        assert_eq!((view.sun_dir, view.sun_color), glint_sun(&lights));
    }

    // The per-frame block every transparent draw shares. `sky_rot` is a float4
    // array, so it needs the 16-byte boundary the scalars ahead of it land on.
    #[test]
    fn transparent_view_layout_matches_shader() {
        assert_eq!(size_of::<TransparentView>(), 240);
        assert_eq!(offset_of!(TransparentView, vp), 0);
        assert_eq!(offset_of!(TransparentView, inv_vp), 64);
        assert_eq!(offset_of!(TransparentView, camera_pos), 128);
        assert_eq!(offset_of!(TransparentView, viewport), 144);
        assert_eq!(offset_of!(TransparentView, time), 152);
        assert_eq!(offset_of!(TransparentView, prefilter_mip_count), 156);
        assert_eq!(offset_of!(TransparentView, sky_rot), 160);
        assert_eq!(offset_of!(TransparentView, sun_dir), 208);
        assert_eq!(offset_of!(TransparentView, sun_color), 224);
        assert_eq!(size_of::<TransparentView>() % 16, 0);
    }

    // The ripple offset follows the surface's roughness: a mirror barely moves
    // its lookup, a rough surface moves it up to the ceiling, and a surface
    // with no mirror slot carries nothing.
    #[test]
    fn the_planar_lane_scales_the_ripple_offset_by_roughness() {
        assert_eq!(WaterParams::planar_lane(0.0, true), [1.0, 0.0, 0.0, 0.0]);
        let default_roughness = WaterParams::planar_lane(0.05, true);
        assert!(
            (default_roughness[1] - 0.03).abs() < 1e-6,
            "{default_roughness:?}"
        );
        let mirror = WaterParams::planar_lane(0.01, true)[1];
        let rough = WaterParams::planar_lane(0.2, true)[1];
        assert!(mirror < default_roughness[1] && default_roughness[1] < rough);
        assert_eq!(
            WaterParams::planar_lane(1.0, true)[1],
            PLANAR_DISTORTION_MAX
        );
        assert_eq!(WaterParams::planar_lane(-1.0, true)[1], 0.0);
        assert_eq!(WaterParams::planar_lane(0.5, false), [0.0; 4]);
    }

    // Two float4s and four scalars, in one 16-byte-aligned block.
    #[test]
    fn glass_params_layout_matches_shader() {
        assert_eq!(size_of::<GlassParams>(), 64);
        assert_eq!(offset_of!(GlassParams, center), 0);
        assert_eq!(offset_of!(GlassParams, normal), 16);
        assert_eq!(offset_of!(GlassParams, tint), 32);
        assert_eq!(offset_of!(GlassParams, opacity), 48);
        assert_eq!(offset_of!(GlassParams, refraction_strength), 52);
        assert_eq!(offset_of!(GlassParams, fresnel_power), 56);
        assert_eq!(offset_of!(GlassParams, planar), 60);
    }

    // `waves` is an array of float4-carrying structs, so the shader aligns it to
    // 16 and the seven live scalars ahead of it need `_pad` to reach that
    // boundary. Nothing else pins the pad, so it is asserted here.
    #[test]
    fn water_params_layout_matches_shader() {
        assert_eq!(size_of::<WaterParams>(), 224);
        assert_eq!(offset_of!(WaterParams, center), 0);
        assert_eq!(offset_of!(WaterParams, deep_color), 16);
        assert_eq!(offset_of!(WaterParams, shallow_color), 32);
        assert_eq!(offset_of!(WaterParams, depth_falloff), 48);
        assert_eq!(offset_of!(WaterParams, foam_width), 52);
        assert_eq!(offset_of!(WaterParams, foam_intensity), 56);
        assert_eq!(offset_of!(WaterParams, fresnel_power), 60);
        assert_eq!(offset_of!(WaterParams, roughness), 64);
        assert_eq!(offset_of!(WaterParams, refraction_strength), 68);
        assert_eq!(offset_of!(WaterParams, wave_count), 72);
        assert_eq!(offset_of!(WaterParams, _pad), 76);
        assert_eq!(offset_of!(WaterParams, waves), 80);
        assert_eq!(offset_of!(WaterParams, planar), 208);
        assert_eq!(size_of::<WaterWaveGpu>(), 32);
    }

    #[test]
    fn from_wave_packs_the_lanes() {
        let w = WaterWave {
            amplitude: 0.25,
            wavelength: 3.0,
            speed: 1.5,
            direction: [0.6, -0.8],
            steepness: 0.4,
        };
        let g = WaterWaveGpu::from_wave(&w);
        assert_eq!(g.dir_amp_wave, [0.6, -0.8, 0.25, 3.0]);
        assert_eq!(g.speed_steep_pad, [1.5, 0.4, 0.0, 0.0]);
    }

    #[test]
    fn from_surface_maps_fields() {
        let surface = WaterSurface {
            center: [1.0, 2.0, 3.0],
            deep_color: [0.02, 0.05, 0.12],
            shallow_color: [0.1, 0.3, 0.4],
            depth_falloff_meters: 3.0,
            foam_width_meters: 0.2,
            foam_intensity: 0.5,
            fresnel_power: 4.0,
            roughness: 0.08,
            refraction_strength: 0.05,
            waves: vec![WaterWave::default(), WaterWave::default()],
            ..Default::default()
        };
        let p = WaterParams::from_surface(&surface, true);
        assert_eq!(p.center, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(p.deep_color, [0.02, 0.05, 0.12, 0.0]);
        assert_eq!(p.shallow_color, [0.1, 0.3, 0.4, 0.0]);
        assert_eq!(p.depth_falloff, 3.0);
        assert_eq!(p.foam_width, 0.2);
        assert_eq!(p.foam_intensity, 0.5);
        assert_eq!(p.fresnel_power, 4.0);
        assert_eq!(p.roughness, 0.08);
        assert_eq!(p.refraction_strength, 0.05);
        assert_eq!(p.wave_count, 2);
        assert_eq!(p.planar, WaterParams::planar_lane(0.08, true));
        assert!(p.planar[0] > 0.5 && p.planar[1] > 0.0);
        // A slotless surface keeps the probe / sky path.
        assert_eq!(WaterParams::from_surface(&surface, false).planar, [0.0; 4]);
    }

    // More authored waves than the shader's array can hold must clamp rather
    // than overflow the fixed lane count.
    #[test]
    fn from_surface_clamps_the_wave_count() {
        let surface = WaterSurface {
            waves: vec![WaterWave::default(); WATER_MAX_WAVES + 3],
            ..Default::default()
        };
        assert_eq!(
            WaterParams::from_surface(&surface, false).wave_count,
            WATER_MAX_WAVES as u32
        );
    }

    #[test]
    fn from_panel_maps_fields() {
        let panel = GlassPanel {
            center: [1.0, 2.0, 3.0],
            normal: [0.0, 0.0, 1.0],
            tint: [0.6, 0.85, 0.9],
            opacity: 0.45,
            refraction_strength: 0.04,
            fresnel_power: 4.0,
            ..Default::default()
        };
        let p = GlassParams::from_panel(&panel, true);
        assert_eq!(p.center, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(p.normal, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(p.tint, [0.6, 0.85, 0.9, 0.0]);
        assert_eq!(p.opacity, 0.45);
        assert_eq!(p.refraction_strength, 0.04);
        assert_eq!(p.fresnel_power, 4.0);
        assert_eq!(p.planar, 1.0);
        // A slotless pane keeps the probe / sky path.
        assert_eq!(GlassParams::from_panel(&panel, false).planar, 0.0);
    }
}
