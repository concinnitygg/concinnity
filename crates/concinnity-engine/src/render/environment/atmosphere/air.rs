//! The air's vertical structure: how thick it is, how its scattering and
//! absorption are spread with height, and the same air scaled to another
//! ground.

use concinnity_core::math::exp;

/// The medium of an atmosphere over its ground: its thickness, the Rayleigh
/// and Mie layers and the ozone layer. Lengths are kilometers, coefficients
/// per kilometer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AirProfile {
    /// Thickness of the air above the ground.
    pub height: f32,
    /// Rayleigh scattering at the ground.
    pub rayleigh_scattering: [f32; 3],
    /// Rayleigh scale height.
    pub rayleigh_scale_height: f32,
    /// Mie scattering at the ground.
    pub mie_scattering: [f32; 3],
    /// Mie scattering plus absorption at the ground.
    pub mie_extinction: [f32; 3],
    /// Mie scale height.
    pub mie_scale_height: f32,
    /// Ozone absorption at the layer's center.
    pub ozone_absorption: [f32; 3],
    /// Height of the ozone layer's center.
    pub ozone_center: f32,
    /// Half the ozone layer's width.
    pub ozone_half_width: f32,
}

/// The air at one height, per kilometer and color.
pub(super) struct AirMedium {
    /// Rayleigh scattering.
    pub(super) rayleigh: [f32; 3],
    /// Mie scattering.
    pub(super) mie: [f32; 3],
    /// What the air takes out of a beam crossing it.
    pub(super) extinction: [f32; 3],
}

impl AirProfile {
    /// This air with every height multiplied by `k` and every coefficient
    /// divided by it: the same air around a ground `k` times the size, so a
    /// path across it scaled by `k` loses the same light. 1 returns it
    /// unchanged.
    #[must_use]
    pub fn scaled(&self, k: f32) -> Self {
        if k == 1.0 {
            return *self;
        }
        let thin = |c: [f32; 3]| c.map(|v| v / k);
        Self {
            height: self.height * k,
            rayleigh_scattering: thin(self.rayleigh_scattering),
            rayleigh_scale_height: self.rayleigh_scale_height * k,
            mie_scattering: thin(self.mie_scattering),
            mie_extinction: thin(self.mie_extinction),
            mie_scale_height: self.mie_scale_height * k,
            ozone_absorption: thin(self.ozone_absorption),
            ozone_center: self.ozone_center * k,
            ozone_half_width: self.ozone_half_width * k,
        }
    }

    /// Extinction per kilometer at `height` kilometers above the ground: what
    /// the air takes out of a beam crossing it. Mirrors `medium_extinction` in
    /// `atmosphere_common.hlsl`.
    pub fn extinction_at(&self, height: f32) -> [f32; 3] {
        let rayleigh = exp(-height.max(0.0) / self.rayleigh_scale_height);
        let mie = exp(-height.max(0.0) / self.mie_scale_height);
        let ozone = self.ozone_density(height);
        core::array::from_fn(|i| {
            self.rayleigh_scattering[i] * rayleigh
                + self.mie_extinction[i] * mie
                + self.ozone_absorption[i] * ozone
        })
    }

    /// The air at `height` kilometers above the ground. Mirrors `atmo_medium`
    /// in `atmosphere_common.hlsl`.
    pub(super) fn medium_at(&self, height: f32) -> AirMedium {
        let rayleigh = exp(-height.max(0.0) / self.rayleigh_scale_height);
        let mie = exp(-height.max(0.0) / self.mie_scale_height);
        let ozone = self.ozone_density(height);
        AirMedium {
            rayleigh: self.rayleigh_scattering.map(|c| c * rayleigh),
            mie: self.mie_scattering.map(|c| c * mie),
            extinction: core::array::from_fn(|i| {
                self.rayleigh_scattering[i] * rayleigh
                    + self.mie_extinction[i] * mie
                    + self.ozone_absorption[i] * ozone
            }),
        }
    }

    // The ozone layer's relative density: a tent peaking at its center.
    fn ozone_density(&self, height: f32) -> f32 {
        (1.0 - (height - self.ozone_center).abs() / self.ozone_half_width).max(0.0)
    }

    /// Optical depth straight up from the ground to the top of the air, per
    /// color, in closed form.
    pub fn zenith_depth(&self) -> [f32; 3] {
        let layer = |h: f32| h * (1.0 - exp(-self.height / h));
        let rayleigh = layer(self.rayleigh_scale_height);
        let mie = layer(self.mie_scale_height);
        core::array::from_fn(|i| {
            self.rayleigh_scattering[i] * rayleigh
                + self.mie_extinction[i] * mie
                + self.ozone_absorption[i] * self.ozone_half_width
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Atmosphere;
    use crate::render::environment::atmosphere::AtmosphereSettings;

    fn earth() -> AirProfile {
        AtmosphereSettings::resolve(&Atmosphere::default(), None).air
    }

    #[test]
    fn a_unit_scale_is_the_same_air() {
        let air = earth();
        assert_eq!(air.scaled(1.0), air);
    }

    // Every length shrinks and every coefficient grows by the same factor, so
    // the column above the ground keeps its optical depth.
    #[test]
    fn scaling_keeps_the_column_depth() {
        let air = earth();
        let ground = air.zenith_depth();
        for k in [0.5, 0.1, 50.0 / 6360.0] {
            let small = air.scaled(k);
            assert!((small.height - air.height * k).abs() < 1e-5);
            assert!((small.rayleigh_scale_height - 8.0 * k).abs() < 1e-5);
            assert!((small.ozone_center - 25.0 * k).abs() < 1e-5);
            let depth = small.zenith_depth();
            for i in 0..3 {
                assert!(
                    (depth[i] - ground[i]).abs() < ground[i] * 1e-4,
                    "{k} {i}: {} vs {}",
                    depth[i],
                    ground[i]
                );
            }
            // The extinction at a scaled height is the original's over k.
            let at = small.extinction_at(3.0 * k);
            let full = air.extinction_at(3.0);
            for i in 0..3 {
                assert!((at[i] * k - full[i]).abs() < full[i] * 1e-4, "{k} {i}");
            }
        }
    }
}
