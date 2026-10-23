//! What the transparent pass binds: the shared per-frame view block, and the
//! per-record tunables of its producers -- glass panes, see-through glass
//! meshes, and water surfaces.

use super::PassCamera;
use super::daylight::{DaylightBlock, VIEW_BLOCK_LIMIT};
use super::shore::WaterShoreGpu;
use super::wake::WaterWakeGpu;
use crate::components::GlassPanel;
use crate::gfx::render_types::LightUniforms;
use crate::render::environment::water::motion::BodyMotion;
use crate::render::environment::water::sea::{SEA_FACES, SeaFaceMotion, SeaMotion};
use crate::render::light::lights::glint_sun;
use crate::water::ocean::foam::FOAM_CASCADES;
use crate::water::ocean::{BodyOcean, MAX_OCEAN_CASCADES};
use crate::water::ripples::{RIPPLES, Ripple};
use crate::water::sea::SeaSphere;
use crate::water::{BodyWave, WaterBody};
use concinnity_core::math::{SaturatingFrom, narrow_f32};

/// Per-frame view inputs shared by every draw in the transparent pass (water,
/// glass), bound once for the whole pass. Matches `TransparentView` in
/// `shaders/glass.hlsl` and `shaders/water.hlsl`.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct TransparentView {
    /// View-projection matrix, column-major.
    pub vp: [[f32; 4]; 4],
    /// Inverse view-projection matrix, column-major.
    pub inv_vp: [[f32; 4]; 4],
    /// World-space camera position (xyz); `w` is the scale the HDR chain is
    /// written at: the scene snapshot is divided by it, the pass's output
    /// multiplied.
    pub camera_pos: [f32; 4],
    /// Render-target width / height in pixels: the shader uses this to
    /// turn its fragment position into a normalized screen UV.
    pub viewport: [f32; 2],
    /// Seconds since the world started, which the water's gusts follow.
    pub time: f32,
    /// Mip count of the bound IBL prefilter cube; 0 signals "no environment map",
    /// where the glass reflection falls back to a white rim. Per-frame state, so
    /// it rides the shared view rather than a per-draw params block.
    pub prefilter_mip_count: f32,
    /// Rows of the rotation taking a world-space direction into the environment
    /// cubemap's baked frame, mirroring `ViewUniforms.sky_rot` so this pass's
    /// sky taps turn with the main one, and read the cube at its storage scale.
    pub sky_rot: [[f32; 4]; 3],
    /// `[x, y, z, _]`: unit direction toward the scene's sun, the first
    /// directional light. Zero when the world declares none.
    pub sun_dir: [f32; 4],
    /// `[r, g, b, _]`: that light's color times its intensity, which the water
    /// glint and the water body's lighting scale by. Zero when the world
    /// declares no directional light, and the shader draws no glint.
    pub sun_color: [f32; 4],
    /// `[irradiance, intensity, _, _]`: 1 in `irradiance` when the bound
    /// irradiance cube holds the sky's diffuse light (else the water body takes
    /// the main pass's flat ambient), and the authored indirect-light multiplier
    /// (`IndirectLighting.ambient_intensity`) in `intensity`.
    pub ambient: [f32; 4],
    /// The planet's per-point sky, [`DaylightBlock::OFF`] for a world
    /// without one.
    pub daylight: DaylightBlock,
}

const _: () = assert!(size_of::<TransparentView>() <= VIEW_BLOCK_LIMIT);

impl TransparentView {
    /// The view block for `camera`, lit by the sun `lights` glints with and
    /// the planet's per-point sky `daylight`.
    pub fn new(camera: &PassCamera, lights: &LightUniforms, daylight: &DaylightBlock) -> Self {
        let c = camera.cam_pos;
        let (sun_dir, sun_color) = glint_sun(lights);
        Self {
            vp: camera.vp,
            inv_vp: camera.inv_vp,
            camera_pos: [c[0], c[1], c[2], camera.pre_exposure],
            viewport: camera.viewport,
            time: camera.time,
            prefilter_mip_count: camera.prefilter_mip_count,
            sky_rot: camera.sky_rot,
            sun_dir,
            sun_color,
            ambient: [camera.irradiance, lights.ambient_intensity, 0.0, 0.0],
            daylight: *daylight,
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

/// Maximum waves summed per `WaterParams`: the wind sea plus the authored
/// swell. Mirrors `WATER_MAX_WAVES` in `shaders/water.hlsl`.
pub const WATER_MAX_WAVES: usize = crate::water::MAX_WAVES;

/// One wave of the sum, packed into two `float4` lanes so the layout is
/// identical on every target. Matches `WaterWave` in `shaders/water.hlsl`.
/// 32 bytes.
#[derive(Copy, Clone, Default, Debug, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct WaterWaveGpu {
    /// `[direction.x, direction.z, amplitude, wavenumber]`.
    pub dir_amp_k: [f32; 4],
    /// `[pinch, phase, _, _]`, the phase at the anchor this frame
    /// ([`BodyMotion::wave_phases`]).
    pub motion: [f32; 4],
}

impl WaterWaveGpu {
    /// The shader-side lanes for one wave of a body at `phase`.
    fn from_wave(w: &BodyWave, phase: f32) -> Self {
        let wave = &w.wave;
        Self {
            dir_amp_k: [
                wave.direction[0],
                wave.direction[1],
                wave.amplitude,
                wave.wavenumber(),
            ],
            motion: [w.pinch, phase, 0.0, 0.0],
        }
    }
}

/// One ripple octave, packed into two `float4` lanes. Matches `WaterRipple` in
/// `shaders/water.hlsl`. 32 bytes.
#[derive(Copy, Clone, Default, Debug, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct WaterRippleGpu {
    /// `[heading.x, heading.z, wavelength, _]`.
    pub shape: [f32; 4],
    /// `[along, across, _, _]`: the noise offset at the anchor this
    /// frame, in lattice cells ([`BodyMotion::ripple_offsets`]).
    pub offset: [f32; 4],
}

impl WaterRippleGpu {
    /// The shader-side lanes for one ripple octave at `offset`.
    fn from_ripple(r: &Ripple, offset: [f32; 2]) -> Self {
        Self {
            shape: [r.direction[0], r.direction[1], r.wavelength, 0.0],
            offset: [offset[0], offset[1], 0.0, 0.0],
        }
    }
}

/// Where a body samples its simulated ocean. Matches `WaterOcean` in
/// `shaders/water.hlsl`. 64 bytes.
#[derive(Copy, Clone, Default, Debug, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct WaterOceanGpu {
    /// `[cascades, first layer, texels per side, whitecapping cascades]`; no
    /// cascades for a body whose sea is summed, and no whitecapping ones for
    /// a body without whitecaps.
    pub layers: [u32; 4],
    /// One over each cascade's side, per meter; zero past the cascades.
    pub inv_lengths: [f32; 4],
    /// Where the anchor falls in each cascade's tile, `[u0, v0, u1, v1]` and
    /// `[u2, v2, u3, v3]` ([`BodyMotion::ocean_offsets`]).
    pub offsets: [[f32; 4]; 2],
}

impl WaterOceanGpu {
    /// The block for a body's share of an ocean, at no offset, sampling the
    /// foam of its longest cascades when it `whitecaps`.
    fn from_ocean(ocean: &BodyOcean, whitecaps: bool) -> Self {
        let c = &ocean.cascades;
        let mut inv_lengths = [0.0; 4];
        for (slot, &l) in inv_lengths.iter_mut().zip(c.lengths()) {
            *slot = 1.0 / l;
        }
        Self {
            layers: [
                u32::saturating_from(c.count()),
                ocean.first_layer,
                c.size(),
                if whitecaps {
                    u32::saturating_from(c.count().min(FOAM_CASCADES))
                } else {
                    0
                },
            ],
            inv_lengths,
            offsets: [[0.0; 4]; 2],
        }
    }

    fn offsets_lane(offsets: &[[f32; 2]; MAX_OCEAN_CASCADES]) -> [[f32; 4]; 2] {
        let [a, b, c, d] = *offsets;
        [[a[0], a[1], b[0], b[1]], [c[0], c[1], d[0], d[1]]]
    }
}

/// One cube face's view of a planet sea. Matches `WaterSeaFace` in
/// `shaders/water.hlsl`. 128 bytes.
#[derive(Copy, Clone, Default, Debug, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct WaterSeaFaceGpu {
    /// The rotation from the simulated frame into the face's `[u, normal,
    /// v]`, one row per face axis, each `w` the anchor's direction from the
    /// planet's center along that axis ([`SeaFaceMotion::rows`]).
    pub rows: [[f32; 4]; 3],
    /// `[x, z, _, _]`: the ripples' domain warp offset at the anchor's place
    /// on the face.
    pub warp: [f32; 4],
    /// Where the anchor's place on the face falls in each cascade's tile,
    /// laid out as [`WaterOceanGpu::offsets`].
    pub ocean_offsets: [[f32; 4]; 2],
    /// Each ripple octave's noise offset there, `[along, across]` pairs, two
    /// octaves a row.
    pub ripple_offsets: [[f32; 4]; 2],
}

impl WaterSeaFaceGpu {
    /// The block for one face at one instant.
    fn from_motion(face: &SeaFaceMotion) -> Self {
        let m = &face.motion;
        let [a, b, c, d] = m.ripple_offsets;
        Self {
            rows: face.rows,
            warp: warp_lane(m),
            ocean_offsets: WaterOceanGpu::offsets_lane(&m.ocean_offsets),
            ripple_offsets: [[a[0], a[1], b[0], b[1]], [c[0], c[1], d[0], d[1]]],
        }
    }
}

/// Where a planet sea lies and how its grid is laid on it. Matches
/// `WaterSea` in `shaders/water.hlsl`; zero for a rectangle, which never
/// reads it. 2096 bytes.
#[derive(Copy, Clone, Default, Debug, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct WaterSeaGpu {
    /// `[up, radius]`: the sea's radial up at the anchor in the simulated
    /// frame, and the sea's radius in meters.
    pub up: [f32; 4],
    /// `[x axis, grid scale]`: the direction the grid's `x` runs along the
    /// sea at the anchor, and meters along the sea per meter of the grid.
    pub tangent_u: [f32; 4],
    /// `[z axis, reach]`: the direction the grid's `z` runs, and how far
    /// from the anchor along the tangent plane the grid is drawn, in meters.
    pub tangent_v: [f32; 4],
    /// `[1 / radius, face blend, _, _]`: a face's share where it meets a
    /// neighbor is `saturate(0.5 + margin * face blend)`, `margin` its lead
    /// over the neighbor in the direction's components.
    pub shape: [f32; 4],
    /// Where the wind's gusts are laid out, as rows over this frame's points
    /// (the same rows the grass reads), so the two gust together.
    pub wind_map: [[f32; 4]; 2],
    /// The faces around the cube corner nearest the anchor, one per axis.
    pub faces: [WaterSeaFaceGpu; SEA_FACES],
    /// The waves at its shores.
    pub shore: WaterShoreGpu,
    /// Where its wake lies; zero while none runs.
    pub wake: WaterWakeGpu,
}

impl WaterSeaGpu {
    /// The block for a sea on `sphere` at the instant `motion` describes,
    /// its gusts laid out by `wind_map`.
    pub fn new(sphere: &SeaSphere, motion: &SeaMotion, wind_map: [[f32; 4]; 2]) -> Self {
        let [tu, tv] = motion.tangents;
        let r = sphere.radius;
        let blend = sphere.blend_rate();
        Self {
            up: [motion.up[0], motion.up[1], motion.up[2], narrow_f32(r)],
            tangent_u: [tu[0], tu[1], tu[2], motion.grid_scale],
            tangent_v: [tv[0], tv[1], tv[2], motion.reach],
            shape: [narrow_f32(1.0 / r), narrow_f32(blend), 0.0, 0.0],
            wind_map,
            faces: motion.faces.each_ref().map(WaterSeaFaceGpu::from_motion),
            shore: WaterShoreGpu::default(),
            wake: WaterWakeGpu::default(),
        }
    }
}

/// Per-surface tunables for a `WaterSurface`, uploaded once per surface per
/// frame: the anchor, the wave phases, the ripple offsets and the ocean tile
/// offsets move with time and the camera (see [`WaterParams::advance`]). The
/// vec3-ish fields are `[f32; 4]` so the layout is byte-identical to the
/// shader's `float4`. Matches `WaterParams` in `shaders/water.hlsl`. 3232
/// bytes.
#[derive(Copy, Clone, Debug, bytemuck::NoUninit)]
#[repr(C)]
pub struct WaterParams {
    /// `[x, y, z, _]`: the world-space point the grid and everything that
    /// moves on it are measured from ([`BodyMotion::anchor`]).
    pub anchor: [f32; 4],
    /// `[r, g, b, _]`: body reflectance in `[0, 1]` at full column depth.
    pub deep_color: [f32; 4],
    /// `[r, g, b, _]`: body reflectance in `[0, 1]` just above the seabed, and
    /// the filter on the seabed seen through the shallows.
    pub shallow_color: [f32; 4],
    /// Depth over which the tint blends from shallow to deep, in meters.
    pub depth_falloff: f32,
    /// Width of the shoreline foam band, in world units.
    pub foam_width: f32,
    /// Foam brightness multiplier.
    pub foam_intensity: f32,
    /// Exponent of the Fresnel reflectance curve.
    pub fresnel_power: f32,
    /// Perceptual roughness in `[0, 1]` of the surface at rest; the waves
    /// filtered out of a pixel and the ripples finer than the shader draws
    /// add to it.
    pub roughness: f32,
    /// How far refraction offsets the sampled background.
    pub refraction_strength: f32,
    /// Live entries in `waves`.
    pub wave_count: u32,
    /// How many of the live `waves`, from the first, displace the grid.
    pub displace_count: u32,
    /// The wind, `WindField::gpu_rows()[0]`: `[direction.x, direction.z,
    /// strength, gustiness]`.
    pub wind: [f32; 4],
    /// `[1 / gust_scale, wind_response, whitecaps, _]`: the gusts' scale, how
    /// strongly the wind roughens this surface, and how strongly its breaking
    /// crests foam.
    pub wind_gust: [f32; 4],
    /// Wave coefficients, longest first; the first `wave_count` are live.
    pub waves: [WaterWaveGpu; WATER_MAX_WAVES],
    /// Planar reflection control: `[strength, distortion, _, _]`. `strength >
    /// 0.5` selects the sharp planar reflection (the scene re-rendered mirrored
    /// across this surface's rest plane, sampled at screen UV) over the probe /
    /// sky cube; `distortion` scales the wave-normal ripple offset of that
    /// lookup, see [`WaterParams::planar_lane`]. 0 when planar is off (RT on,
    /// no planar slot, or the plane overflowed the budget), keeping the probe /
    /// sky path.
    pub planar: [f32; 4],
    /// The ripple octaves, longest first.
    pub ripples: [WaterRippleGpu; RIPPLES],
    /// `[x, z, _, _]`: the ripples' domain warp offset at the anchor this
    /// frame, in its noise cells ([`BodyMotion::warp_offset`]).
    pub ripple_warp: [f32; 4],
    /// The surface's rectangle relative to the anchor, `[min x, min z, max
    /// x, max z]`, which the grid is clamped to.
    pub rest_bounds: [f32; 4],
    /// `[inner cell, inner half-width, widening, _]`: the grid's local cell
    /// size is `inner + max(|x| - half, 0) * widening` per axis.
    pub grid: [f32; 4],
    /// The simulated ocean the body samples.
    pub ocean: WaterOceanGpu,
    /// The planet sea the body is; zero for a rectangle.
    pub sea: WaterSeaGpu,
}

// Metal binds `WaterParams` inline (`setBytes`), which takes at most 4 KiB.
const WATER_PARAMS_INLINE_LIMIT: usize = 4096;
const _: () = assert!(size_of::<WaterParams>() <= WATER_PARAMS_INLINE_LIMIT);

// How far the mirror lookup is pushed per unit of wave slope, per unit of the
// surface's authored roughness, and the ceiling a very rough surface stops at.
// The planar target is a flat-plane render, so the offset only fakes the
// ripple; a near-mirror surface barely moves it, a choppy one moves it more.
const PLANAR_DISTORTION_PER_ROUGHNESS: f32 = 0.6;
const PLANAR_DISTORTION_MAX: f32 = 0.06;

// An authored body color as the shader's reflectance lane, each channel held to
// [0, 1] so the column never returns more light than reaches it. A NaN channel
// reads as black.
fn reflectance(rgb: [f32; 3]) -> [f32; 4] {
    let r = |c: f32| if c.is_nan() { 0.0 } else { c.clamp(0.0, 1.0) };
    [r(rgb[0]), r(rgb[1]), r(rgb[2]), 0.0]
}

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

    /// The per-surface block for a resolved body at the world's start,
    /// keeping at most [`WATER_MAX_WAVES`] of its waves. `planar` is whether
    /// the surface holds a planar reflection slot, see
    /// [`WaterParams::planar_lane`].
    pub fn from_body(body: &WaterBody, planar: bool) -> Self {
        let surface = &body.surface;
        let motion = body.motion_at(0.0, body.anchor(surface.center));
        let mut waves = [WaterWaveGpu::default(); WATER_MAX_WAVES];
        for ((slot, src), &phase) in waves.iter_mut().zip(&body.waves).zip(&motion.wave_phases) {
            *slot = WaterWaveGpu::from_wave(src, phase);
        }
        let ripples = core::array::from_fn(|r| {
            WaterRippleGpu::from_ripple(&body.ripples[r], motion.ripple_offsets[r])
        });
        let wave_count = body.waves.len().min(WATER_MAX_WAVES);
        let [wind, gust] = body.wind.gpu_rows();
        Self {
            anchor: anchor_lane(&motion),
            deep_color: reflectance(surface.deep_color),
            shallow_color: reflectance(surface.shallow_color),
            depth_falloff: surface.depth_falloff_meters,
            foam_width: surface.foam_width_meters,
            foam_intensity: surface.foam_intensity,
            fresnel_power: surface.fresnel_power,
            roughness: surface.roughness,
            refraction_strength: surface.refraction_strength,
            wave_count: u32::saturating_from(wave_count),
            displace_count: u32::saturating_from(body.displacing.min(wave_count)),
            wind,
            wind_gust: [gust[0], surface.wind_response, surface.whitecaps, 0.0],
            waves,
            planar: Self::planar_lane(surface.roughness, planar),
            ripples,
            ripple_warp: warp_lane(&motion),
            rest_bounds: motion.rest_bounds,
            grid: body.grid_lane(),
            ocean: body
                .ocean
                .as_ref()
                .map_or_else(WaterOceanGpu::default, |o| WaterOceanGpu {
                    offsets: WaterOceanGpu::offsets_lane(&motion.ocean_offsets),
                    ..WaterOceanGpu::from_ocean(o, surface.whitecaps > 0.0)
                }),
            sea: WaterSeaGpu {
                shore: body
                    .sea
                    .as_ref()
                    .and_then(|s| s.shore.as_ref())
                    .map_or_else(WaterShoreGpu::default, WaterShoreGpu::from_shore),
                ..WaterSeaGpu::default()
            },
        }
    }

    /// The block for a planet sea at the instant `motion` describes, its
    /// gusts laid out by `wind_map`: the anchor and every face's offsets.
    pub fn advance_sea(&mut self, body: &WaterBody, motion: &SeaMotion, wind_map: [[f32; 4]; 2]) {
        let Some(sea) = &body.sea else {
            return;
        };
        let [x, y, z] = motion.anchor;
        self.anchor = [x, y, z, 0.0];
        let shore = self.sea.shore;
        self.sea = WaterSeaGpu::new(&sea.sphere, motion, wind_map);
        self.sea.shore = shore;
        if let Some(m) = &motion.shore {
            self.sea.shore.advance(m);
        }
    }

    /// Moves the block to the instant and anchor `motion` describes: each
    /// wave's phase, each ripple's and each cascade's offset, all small
    /// numbers the shader adds to what varies across the surface.
    pub fn advance(&mut self, motion: &BodyMotion) {
        self.anchor = anchor_lane(motion);
        self.rest_bounds = motion.rest_bounds;
        self.ocean.offsets = WaterOceanGpu::offsets_lane(&motion.ocean_offsets);
        for (wave, &phase) in self.waves.iter_mut().zip(&motion.wave_phases) {
            wave.motion[1] = phase;
        }
        for (ripple, offset) in self.ripples.iter_mut().zip(&motion.ripple_offsets) {
            ripple.offset = [offset[0], offset[1], 0.0, 0.0];
        }
        self.ripple_warp = warp_lane(motion);
    }
}

fn warp_lane(motion: &BodyMotion) -> [f32; 4] {
    [motion.warp_offset[0], motion.warp_offset[1], 0.0, 0.0]
}

fn anchor_lane(motion: &BodyMotion) -> [f32; 4] {
    let [x, y, z] = motion.anchor;
    [x, y, z, 0.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{WaterSurface, WaterWave, Wind};
    use crate::water::spectrum::Wave;
    use crate::wind::WindField;
    use alloc::vec;
    use core::mem::{offset_of, size_of};

    #[test]
    fn the_wave_count_matches_the_shader() {
        let src = crate::render::shader::embedded::WATER;
        assert_eq!(
            crate::render::shader::consts::uint(src, "WATER_MAX_WAVES"),
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
        let view = TransparentView::new(&camera, &lights, &DaylightBlock::OFF);
        assert_eq!(view.vp, camera.vp);
        assert_eq!(view.inv_vp, camera.inv_vp);
        assert_eq!(view.camera_pos, [3.0, 4.0, 5.0, camera.pre_exposure]);
        assert_eq!(view.viewport, camera.viewport);
        assert_eq!(view.time, camera.time);
        assert_eq!(view.prefilter_mip_count, camera.prefilter_mip_count);
        assert_eq!(view.sky_rot, camera.sky_rot);
        assert_eq!((view.sun_dir, view.sun_color), glint_sun(&lights));
        assert_eq!(
            view.ambient,
            [camera.irradiance, lights.ambient_intensity, 0.0, 0.0]
        );
    }

    // The per-frame block every transparent draw shares. `sky_rot` is a float4
    // array, so it needs the 16-byte boundary the scalars ahead of it land on.
    #[test]
    fn transparent_view_layout_matches_shader() {
        assert_eq!(size_of::<TransparentView>(), 256 + 1664);
        assert_eq!(offset_of!(TransparentView, vp), 0);
        assert_eq!(offset_of!(TransparentView, inv_vp), 64);
        assert_eq!(offset_of!(TransparentView, camera_pos), 128);
        assert_eq!(offset_of!(TransparentView, viewport), 144);
        assert_eq!(offset_of!(TransparentView, time), 152);
        assert_eq!(offset_of!(TransparentView, prefilter_mip_count), 156);
        assert_eq!(offset_of!(TransparentView, sky_rot), 160);
        assert_eq!(offset_of!(TransparentView, sun_dir), 208);
        assert_eq!(offset_of!(TransparentView, sun_color), 224);
        assert_eq!(offset_of!(TransparentView, ambient), 240);
        assert_eq!(offset_of!(TransparentView, daylight), 256);
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

    // `waves` and `ripples` are arrays of float4-carrying structs, so the
    // shader aligns them to 16; the eight scalars ahead of the wind rows fill
    // two 16-byte rows.
    #[test]
    fn water_params_layout_matches_shader() {
        assert_eq!(size_of::<WaterParams>(), 3232);
        assert_eq!(offset_of!(WaterParams, anchor), 0);
        assert_eq!(offset_of!(WaterParams, deep_color), 16);
        assert_eq!(offset_of!(WaterParams, shallow_color), 32);
        assert_eq!(offset_of!(WaterParams, depth_falloff), 48);
        assert_eq!(offset_of!(WaterParams, foam_width), 52);
        assert_eq!(offset_of!(WaterParams, foam_intensity), 56);
        assert_eq!(offset_of!(WaterParams, fresnel_power), 60);
        assert_eq!(offset_of!(WaterParams, roughness), 64);
        assert_eq!(offset_of!(WaterParams, refraction_strength), 68);
        assert_eq!(offset_of!(WaterParams, wave_count), 72);
        assert_eq!(offset_of!(WaterParams, displace_count), 76);
        assert_eq!(offset_of!(WaterParams, wind), 80);
        assert_eq!(offset_of!(WaterParams, wind_gust), 96);
        assert_eq!(offset_of!(WaterParams, waves), 112);
        assert_eq!(offset_of!(WaterParams, planar), 880);
        assert_eq!(offset_of!(WaterParams, ripples), 896);
        assert_eq!(offset_of!(WaterParams, ripple_warp), 1024);
        assert_eq!(offset_of!(WaterParams, rest_bounds), 1040);
        assert_eq!(offset_of!(WaterParams, grid), 1056);
        assert_eq!(offset_of!(WaterParams, ocean), 1072);
        assert_eq!(offset_of!(WaterParams, sea), 1136);
        assert_eq!(size_of::<WaterSeaGpu>(), 2096);
        assert_eq!(offset_of!(WaterSeaGpu, up), 0);
        assert_eq!(offset_of!(WaterSeaGpu, tangent_u), 16);
        assert_eq!(offset_of!(WaterSeaGpu, tangent_v), 32);
        assert_eq!(offset_of!(WaterSeaGpu, shape), 48);
        assert_eq!(offset_of!(WaterSeaGpu, wind_map), 64);
        assert_eq!(offset_of!(WaterSeaGpu, faces), 96);
        assert_eq!(offset_of!(WaterSeaGpu, shore), 480);
        assert_eq!(offset_of!(WaterSeaGpu, wake), 2064);
        assert_eq!(size_of::<WaterSeaFaceGpu>(), 128);
        assert_eq!(offset_of!(WaterSeaFaceGpu, rows), 0);
        assert_eq!(offset_of!(WaterSeaFaceGpu, warp), 48);
        assert_eq!(offset_of!(WaterSeaFaceGpu, ocean_offsets), 64);
        assert_eq!(offset_of!(WaterSeaFaceGpu, ripple_offsets), 96);
        assert_eq!(size_of::<WaterOceanGpu>(), 64);
        assert_eq!(offset_of!(WaterOceanGpu, layers), 0);
        assert_eq!(offset_of!(WaterOceanGpu, inv_lengths), 16);
        assert_eq!(offset_of!(WaterOceanGpu, offsets), 32);
        assert_eq!(size_of::<WaterWaveGpu>(), 32);
        assert_eq!(offset_of!(WaterWaveGpu, dir_amp_k), 0);
        assert_eq!(offset_of!(WaterWaveGpu, motion), 16);
        assert_eq!(size_of::<WaterRippleGpu>(), 32);
        assert_eq!(offset_of!(WaterRippleGpu, shape), 0);
        assert_eq!(offset_of!(WaterRippleGpu, offset), 16);
    }

    // An ocean's block names its cascades, where they sit and how many of
    // them whitecap: the longest ones, never more than it has, and none for
    // a body without whitecaps.
    #[test]
    fn an_ocean_block_names_its_whitecapping_cascades() {
        use crate::water::ocean::cascades::Cascades;
        use crate::water::ocean::grid::OceanGrid;
        use crate::water::ocean::{OceanQuality, foam::FOAM_CASCADES};
        let body = |cascades: u32| BodyOcean {
            simulation: 0,
            cascades: Cascades::new(
                56.0,
                OceanQuality {
                    size: 256,
                    cascades,
                    ..OceanQuality::FULL
                },
            ),
            first_layer: 3,
            grid: OceanGrid::new(56.0, [6000.0, 6000.0]),
        };
        let full = WaterOceanGpu::from_ocean(&body(3), true);
        assert_eq!(
            full.layers,
            [3, 3, 256, u32::saturating_from(FOAM_CASCADES)]
        );
        assert!((full.inv_lengths[0] * 560.0 - 1.0).abs() < 1e-5);
        assert_eq!(full.inv_lengths[3], 0.0);
        assert_eq!(WaterOceanGpu::from_ocean(&body(1), true).layers[3], 1);
        let calm = WaterOceanGpu::from_ocean(&body(3), false);
        assert_eq!(calm.layers, [3, 3, 256, 0]);
        assert_eq!(calm.inv_lengths, full.inv_lengths);
    }

    // The body colors are reflectances: whatever is authored, each channel the
    // shader sees lies in [0, 1].
    #[test]
    fn the_body_colors_are_held_to_reflectances() {
        let surface = WaterSurface {
            deep_color: [-0.5, 0.25, 3.0],
            shallow_color: [f32::NAN, 1.0, f32::INFINITY],
            ..Default::default()
        };
        let p = WaterParams::from_body(&body(&surface), false);
        assert_eq!(p.deep_color, [0.0, 0.25, 1.0, 0.0]);
        assert_eq!(p.shallow_color, [0.0, 1.0, 1.0, 0.0]);
        let defaults = WaterParams::from_body(&body(&WaterSurface::default()), false);
        let d = WaterSurface::default();
        assert_eq!(defaults.deep_color[..3], d.deep_color);
        assert_eq!(defaults.shallow_color[..3], d.shallow_color);
    }

    fn body(surface: &WaterSurface) -> WaterBody {
        WaterBody::resolve(surface, &WindField::new(Some(&Wind::default())))
    }

    #[test]
    fn from_wave_packs_the_lanes() {
        let w = BodyWave {
            wave: Wave {
                direction: [0.6, -0.8],
                amplitude: 0.25,
                wavelength: 3.0,
                phase: 1.5,
                steepness: 0.4,
            },
            pinch: 0.7,
        };
        let g = WaterWaveGpu::from_wave(&w, 2.5);
        assert_eq!(g.dir_amp_k, [0.6, -0.8, 0.25, w.wave.wavenumber()]);
        assert_eq!(g.motion, [0.7, 2.5, 0.0, 0.0]);
    }

    #[test]
    fn from_ripple_packs_the_lanes() {
        let r = Ripple {
            direction: [0.6, 0.8],
            wavelength: 0.05,
            speed: 0.3,
            seed: 17.31,
        };
        let g = WaterRippleGpu::from_ripple(&r, [3.0, -4.0]);
        assert_eq!(g.shape, [0.6, 0.8, 0.05, 0.0]);
        assert_eq!(g.offset, [3.0, -4.0, 0.0, 0.0]);
    }

    // Advancing moves every phase and offset to the given instant and leaves
    // the rest of the block as built.
    #[test]
    fn advance_moves_only_the_phases_and_offsets() {
        let b = body(&WaterSurface {
            center: [120.0, 0.0, -40.0],
            ..Default::default()
        });
        let built = WaterParams::from_body(&b, false);
        let mut p = built;
        let center = b.anchor([0.0; 3]);
        let motion = b.motion_at(37.5, center);
        p.advance(&motion);
        for (i, w) in b.waves.iter().enumerate() {
            assert_eq!(
                p.waves[i],
                WaterWaveGpu::from_wave(w, motion.wave_phases[i])
            );
        }
        for (r, ripple) in b.ripples.iter().enumerate() {
            let want = WaterRippleGpu::from_ripple(ripple, motion.ripple_offsets[r]);
            assert_eq!(p.ripples[r], want);
        }
        assert_eq!(p.ripple_warp[..2], motion.warp_offset);
        assert_eq!(p.anchor, built.anchor);
        assert_eq!(p.rest_bounds, built.rest_bounds);
        assert_eq!(p.wind_gust, built.wind_gust);
        assert_ne!(p.waves[0], built.waves[0]);
        // Advancing back to the start gives the block as built.
        p.advance(&b.motion_at(0.0, center));
        assert_eq!(bytemuck::bytes_of(&p), bytemuck::bytes_of(&built));
    }

    #[test]
    fn from_body_maps_fields() {
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
            wind_response: 0.5,
            whitecaps: 0.75,
            ..Default::default()
        };
        let b = body(&surface);
        let p = WaterParams::from_body(&b, true);
        assert_eq!(p.anchor, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(p.deep_color, [0.02, 0.05, 0.12, 0.0]);
        assert_eq!(p.shallow_color, [0.1, 0.3, 0.4, 0.0]);
        assert_eq!(p.depth_falloff, 3.0);
        assert_eq!(p.foam_width, 0.2);
        assert_eq!(p.foam_intensity, 0.5);
        assert_eq!(p.fresnel_power, 4.0);
        assert_eq!(p.roughness, 0.08);
        assert_eq!(p.refraction_strength, 0.05);
        assert_eq!(p.wave_count as usize, b.waves.len());
        assert_eq!(p.displace_count as usize, b.displacing);
        let motion = b.motion_at(0.0, b.anchor([0.0; 3]));
        assert_eq!(p.rest_bounds, motion.rest_bounds);
        assert_eq!(p.grid, b.grid_lane());
        assert_eq!(p.ocean, WaterOceanGpu::default());
        assert_eq!(
            p.waves[0],
            WaterWaveGpu::from_wave(&b.waves[0], motion.wave_phases[0])
        );
        assert_eq!(p.waves[b.waves.len()], WaterWaveGpu::default());
        assert_eq!(
            p.ripples[0],
            WaterRippleGpu::from_ripple(&b.ripples[0], motion.ripple_offsets[0])
        );
        assert_eq!(p.ripple_warp[..2], motion.warp_offset);
        let [wind, gust] = b.wind.gpu_rows();
        assert_eq!(p.wind, wind);
        assert_eq!(p.wind_gust, [gust[0], 0.5, 0.75, 0.0]);
        assert_eq!(p.planar, WaterParams::planar_lane(0.08, true));
        assert!(p.planar[0] > 0.5 && p.planar[1] > 0.0);
        // A slotless surface keeps the probe / sky path.
        assert_eq!(WaterParams::from_body(&b, false).planar, [0.0; 4]);
    }

    // The fullest sum a body can hold, a whole wind sea and every swell wave
    // the asset allows, fits the shader's array.
    #[test]
    fn the_fullest_body_fits_the_wave_array() {
        let surface = WaterSurface {
            waves: vec![WaterWave::default(); crate::components::MAX_WATER_WAVES + 3],
            ..Default::default()
        };
        assert_eq!(
            WaterParams::from_body(&body(&surface), false).wave_count,
            u32::saturating_from(WATER_MAX_WAVES)
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
