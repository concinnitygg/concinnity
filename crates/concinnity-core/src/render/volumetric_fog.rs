//! Backend-agnostic resolution of the authored `VolumetricFog` asset into a
//! clamped settings struct plus the per-frame `FogParams` uniform the Metal
//! fog fragment shader consumes. Pure CPU; unit-testable without a GPU.

use crate::gfx::render_types::FogParams;

// Upper bound on the volumetric density. The integral
// `1 - exp(-density * step)` saturates near 1.0 well before this cap, so
// anything higher just wastes precision and risks numeric blowups for the
// Henyey-Greenstein factor. 10/world-unit is already pea-soup territory.
const MAX_DENSITY: f32 = 10.0;
// Largest sensible height-falloff rate. Beyond this the density drops to
// nothing within centimeters above the reference height, which is not
// useful (and is rounding-error fragile in the shader's `exp`).
const MAX_HEIGHT_FALLOFF: f32 = 4.0;
// Cap on the ray-march distance. The marcher takes a fixed number of steps,
// so a longer ray spends more world units per step rather than more samples.
// Going past this trades shadow / phase accuracy for distance with no
// real visual win.
const MAX_DISTANCE_CAP: f32 = 2_000.0;
// Floor on the ray-march distance. The shader divides by it, and the
// per-step length collapses to zero past about a millimeter.
const MIN_DISTANCE: f32 = 1.0;
// Floor on the viewport short edge so a zero-sized swapchain (initial layout)
// cannot poison the reciprocal the shader uses to convert screen to NDC.
const MIN_VIEWPORT: f32 = 1.0;

/// Resolved and clamped fog tunables, threaded into the backend at init.
/// `None` from `FogSettings::resolve_optional` means the world declared no
/// `VolumetricFog`: the renderer then skips the fog pass entirely.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FogSettings {
    /// Linear RGB color.
    pub color: [f32; 3],
    /// Fog density at the height reference.
    pub density: f32,
    /// How fast density falls off with height.
    pub height_falloff: f32,
    /// World height at which density equals `density`.
    pub height_reference: f32,
    /// Furthest world distance the march travels.
    pub max_distance: f32,
    /// Henyey-Greenstein anisotropy in `(-1, 1)`; 0 is isotropic.
    pub phase_g: f32,
    /// Ambient radiance added to the in-scattered term.
    pub ambient: f32,
}

impl FogSettings {
    /// Clamp the authored fields into a safe range. Mirrors `VolumetricFog::from_args`;
    /// those clamps are the asset-side floor; this is the gfx-side ceiling.
    pub fn resolve(
        color: [f32; 3],
        density: f32,
        height_falloff: f32,
        height_reference: f32,
        max_distance: f32,
        phase_g: f32,
        ambient: f32,
    ) -> Self {
        let max_distance = if max_distance.is_finite() {
            max_distance.clamp(MIN_DISTANCE, MAX_DISTANCE_CAP)
        } else {
            MIN_DISTANCE
        };
        let color = [color[0].max(0.0), color[1].max(0.0), color[2].max(0.0)];
        Self {
            color,
            density: density.clamp(0.0, MAX_DENSITY),
            height_falloff: height_falloff.clamp(0.0, MAX_HEIGHT_FALLOFF),
            height_reference,
            max_distance,
            // Mirror the asset clamp so a settings built from out-of-range
            // raw floats (e.g. in tests) still produces stable HG output.
            phase_g: phase_g.clamp(-0.95, 0.95),
            ambient: ambient.clamp(0.0, MAX_DENSITY),
        }
    }

    /// Whether the medium can affect the frame. The froxel kernel integrates
    /// `tau = density * step_len` per slab, so a zero density leaves every slab
    /// at `exp(0) = 1`: the volume stores `(0, 0)` everywhere, `ambient` never
    /// enters (it is scaled by `1 - slab_T`), and the premultiplied `over`
    /// blend resolves to the scene untouched. Backends gate
    /// `FrameGraphInputs::fog_enabled` on this so the graph drops the froxel
    /// compute and the fullscreen resolve rather than paying both to composite
    /// a transparent black.
    pub fn contributes(&self) -> bool {
        self.density > 0.0
    }

    /// Build the per-frame GPU uniform from these settings and the active
    /// camera. `inv_vp` is the inverse view-projection used to reconstruct
    /// world positions from depth; `cam_pos` is the camera origin; `sun_dir`
    /// and `sun_color` are the first directional light's direction (toward
    /// the light) and `intensity * color`. `viewport` is the HDR resolve
    /// target's pixel dimensions.
    pub fn params(
        &self,
        inv_vp: [[f32; 4]; 4],
        cam_pos: [f32; 3],
        sun_dir: [f32; 3],
        sun_color: [f32; 3],
        viewport: [f32; 2],
    ) -> FogParams {
        let viewport = [viewport[0].max(MIN_VIEWPORT), viewport[1].max(MIN_VIEWPORT)];
        FogParams {
            inv_vp,
            color: [self.color[0], self.color[1], self.color[2], 1.0],
            cam_pos,
            _pad0: 0.0,
            sun_dir,
            _pad1: 0.0,
            sun_color,
            _pad2: 0.0,
            density: self.density,
            height_falloff: self.height_falloff,
            height_reference: self.height_reference,
            max_distance: self.max_distance,
            phase_g: self.phase_g,
            ambient: self.ambient,
            viewport,
            inv_max_distance: 1.0 / self.max_distance,
            _pad3: [0.0; 3],
        }
    }
}

/// Resolve an authored `VolumetricFog` into clamped [`FogSettings`], or `None`
/// when the asset's `enabled` toggle is off.
///
/// The one place the asset becomes settings. Init, the `VolumetricFog`
/// hot-reload, and the lighting preview all route through here, so `None` means
/// the same thing on every path and none of them can drift on which fields gate
/// the pass. Mirrors `PostProcessConfig`'s `*_settings()` resolvers.
///
/// Deliberately NOT filtered on [`FogSettings::contributes`], unlike the SSAO
/// and SSGI resolvers. Fog is live-editable: the settings decide whether the
/// backend builds the fog resources at all, and `SettingsState::fog_built`
/// refuses a later runtime enable on a world that never built them. Resolving a
/// zero density away here would leave an author who starts at 0 unable to raise
/// it. The per-frame skip is the backends' `fog_enabled` gate instead, which
/// costs one allocation to keep the slider live.
pub fn resolve_asset(fog: &crate::components::VolumetricFog) -> Option<FogSettings> {
    fog.enabled.then(|| {
        FogSettings::resolve(
            fog.color,
            fog.density,
            fog.height_falloff,
            fog.height_reference,
            fog.max_distance,
            fog.phase_g,
            fog.ambient,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn authored(enabled: bool, density: f32) -> crate::components::VolumetricFog {
        crate::components::VolumetricFog {
            enabled,
            density,
            ..Default::default()
        }
    }

    #[test]
    fn resolve_asset_skips_a_disabled_medium() {
        assert!(resolve_asset(&authored(false, 0.02)).is_none());
    }

    #[test]
    fn resolve_asset_keeps_an_enabled_but_inert_medium() {
        // A zero density still resolves, so the backend builds the fog
        // resources and a live density raise has a pass to land on; the
        // per-frame `fog_enabled` gate is what skips the work. Dropping it here
        // would strand an author who starts the slider at zero.
        let s = resolve_asset(&authored(true, 0.0)).expect("enabled");
        assert!(!s.contributes());
    }

    #[test]
    fn resolve_asset_clamps_a_live_medium() {
        let s = resolve_asset(&authored(true, 1.0e6)).expect("enabled and dense");
        assert_eq!(s.density, MAX_DENSITY);
        assert!(s.contributes());
    }

    const IDENTITY: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    #[test]
    fn resolve_clamps_density_falloff_and_distance() {
        let s = FogSettings::resolve([1.0, 1.0, 1.0], 100.0, 20.0, 0.0, 1e9, 1.5, -1.0);
        assert_eq!(s.density, MAX_DENSITY);
        assert_eq!(s.height_falloff, MAX_HEIGHT_FALLOFF);
        assert_eq!(s.max_distance, MAX_DISTANCE_CAP);
        assert!(s.phase_g <= 0.95 && s.phase_g > 0.0);
        assert_eq!(s.ambient, 0.0);
    }

    #[test]
    fn resolve_passes_through_in_range_values() {
        let s = FogSettings::resolve([0.6, 0.7, 0.8], 0.08, 0.25, 1.5, 120.0, 0.4, 0.2);
        assert_eq!(s.color, [0.6, 0.7, 0.8]);
        assert!((s.density - 0.08).abs() < 1e-6);
        assert!((s.phase_g - 0.4).abs() < 1e-6);
        assert!((s.max_distance - 120.0).abs() < 1e-6);
    }

    #[test]
    fn resolve_handles_non_finite_distance() {
        let s = FogSettings::resolve([0.6; 3], 0.05, 0.2, 0.0, f32::NAN, 0.4, 0.15);
        assert!(s.max_distance.is_finite());
        assert!(s.max_distance >= MIN_DISTANCE);
    }

    #[test]
    fn zero_density_does_not_contribute() {
        // `VolumetricFog { enabled: true, density: 0.0 }` resolves to settings,
        // so presence alone cannot gate the pass.
        let off = FogSettings::resolve([0.6, 0.7, 0.8], 0.0, 0.1, 0.0, 200.0, 0.3, 0.2);
        assert!(!off.contributes());

        let on = FogSettings::resolve([0.6, 0.7, 0.8], 0.02, 0.1, 0.0, 200.0, 0.3, 0.2);
        assert!(on.contributes());
    }

    #[test]
    fn a_negative_density_clamps_to_no_contribution() {
        let s = FogSettings::resolve([0.6, 0.7, 0.8], -1.0, 0.1, 0.0, 200.0, 0.3, 0.2);
        assert!(!s.contributes());
    }

    #[test]
    fn ambient_alone_does_not_make_zero_density_contribute() {
        // The in-scatter is scaled by `1 - exp(-tau)`, so a bright ambient with
        // no medium to scatter in is still nothing.
        let s = FogSettings::resolve([1.0, 1.0, 1.0], 0.0, 0.1, 0.0, 200.0, 0.0, 10.0);
        assert!(!s.contributes());
    }

    #[test]
    fn params_derive_inverse_max_distance() {
        let s = FogSettings::resolve([0.7; 3], 0.05, 0.2, 0.0, 50.0, 0.4, 0.15);
        let p = s.params(
            IDENTITY,
            [0.0; 3],
            [0.0, 1.0, 0.0],
            [1.0; 3],
            [1280.0, 720.0],
        );
        assert!((p.inv_max_distance - (1.0 / 50.0)).abs() < 1e-6);
        assert_eq!(p.viewport, [1280.0, 720.0]);
    }

    #[test]
    fn params_floor_a_degenerate_viewport() {
        let s = FogSettings::resolve([0.7; 3], 0.05, 0.2, 0.0, 50.0, 0.4, 0.15);
        let p = s.params(IDENTITY, [0.0; 3], [0.0, 1.0, 0.0], [1.0; 3], [0.0, 0.0]);
        assert!(p.viewport[0] >= MIN_VIEWPORT);
        assert!(p.viewport[1] >= MIN_VIEWPORT);
    }

    #[test]
    fn params_zero_padding_words_are_zero() {
        let s = FogSettings::resolve([0.7; 3], 0.05, 0.2, 0.0, 50.0, 0.4, 0.15);
        let p = s.params(IDENTITY, [0.0; 3], [0.0, 1.0, 0.0], [1.0; 3], [1.0, 1.0]);
        assert_eq!(p._pad0, 0.0);
        assert_eq!(p._pad1, 0.0);
        assert_eq!(p._pad2, 0.0);
        assert_eq!(p._pad3, [0.0; 3]);
    }
}
