// Exposure schema: a fixed offset, or auto-exposure that adapts to the scene.

use crate::gfx::auto_exposure::{AutoExposureSettings, EV_LIMIT};
use concinnity_core::math::exp2;

/// Scene exposure: a fixed offset in photographic stops, or auto-exposure that
/// measures the scene's average brightness each frame and adapts toward a
/// balanced mid-tone. The `exposure` section of
/// [`GraphicsConfig`](#graphicsconfig); a field left unset takes the default
/// below.
///
/// When `auto` is on, `ev` acts as an additive bias (in stops) on top of the
/// adapted value.
///
/// ```rust
/// # use concinnity_engine::components::{ExposureOverrides, GraphicsConfig};
/// GraphicsConfig {
///     exposure: ExposureOverrides {
///         auto: Some(true),
///         auto_max_ev: Some(24.0),
///         ..Default::default()
///     },
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[asset(section)]
#[serde(default)]
pub struct Exposure {
    /// Exposure offset in photographic stops. Each +1 doubles scene
    /// brightness before bloom and tonemapping; 0 is neutral.
    #[asset(min = -EV_LIMIT, max = EV_LIMIT, apply = live)]
    pub ev: f32,
    /// Auto-exposure. Adapts exposure each frame toward a balanced mid-tone;
    /// `ev` then acts as an additive bias in stops on top of the adapted value.
    #[asset(apply = rebuild)]
    pub auto: bool,
    /// Lower bound on the adapted exposure (EV). The `ev` bias is applied
    /// before this clamp.
    #[asset(default = -8.0, min = -EV_LIMIT, max = EV_LIMIT, apply = live)]
    pub auto_min_ev: f32,
    /// Upper bound on the adapted exposure (EV): how far a dark scene is
    /// brightened, at most 24. A night under a [`NightSky`](#nightsky) adapts
    /// at around 19 to 20, so wants 24.
    #[asset(default = 8.0, min = -EV_LIMIT, max = EV_LIMIT, apply = live)]
    pub auto_max_ev: f32,
    /// How quickly exposure chases a new target (per second). Higher converges
    /// faster but can pump under flickering content; 1-3 is comfortable.
    #[asset(default = 1.5, min = 0.001, max = 20.0, apply = live)]
    pub auto_speed: f32,
    /// Extra stops auto-exposure adds once a scene is dark, on top of `ev`.
    /// Fully adapted, a moonlit night otherwise shows about as bright as a
    /// day; a negative value keeps it darker, so night reads as night. It
    /// eases in as the scene's mean luminance falls from the end of dusk to
    /// moonlight (from about 2^-15 to 2^-22, for a sun of a few units'
    /// intensity), so day and dusk are untouched. 0 adapts a night like a day.
    #[asset(min = -EV_LIMIT, max = EV_LIMIT, apply = live)]
    pub auto_night_ev: f32,
    /// How strongly, in `[0, 1]`, a dark scene is shown as night vision sees
    /// it, with `auto` on. At 1 a scene dark enough for the eye's rods alone
    /// (under about 0.01 cd/m2) loses its color: it shows in the rods'
    /// brightness, under which blues stay lighter and reds go dark, with a
    /// slight blue cast. Through twilight levels, up to about 3 cd/m2, it
    /// blends in from full color. The level is the scene's mean luminance the
    /// exposure has adapted to, read in cd/m2 by taking the world's sun (its
    /// first [`DirectionalLight`](#directionallight)) as 100000 lux of
    /// sunlight, so a day stays in full color whatever the sun's intensity.
    /// 0 keeps full color at any level.
    #[asset(min = 0.0, max = 1.0, apply = live)]
    pub night_vision: f32,
    /// How many stops over white the brightest 0.1% of the frame may sit, with
    /// `auto` on. Auto-exposure brightens no further than this allows, so a
    /// small bright subject in a mostly dark frame, such as a sunlit planet
    /// against black space, keeps its detail instead of blowing out. 24
    /// leaves exposure to the frame's average alone.
    #[asset(default = 2.0, min = 0.0, max = EV_LIMIT, apply = live)]
    pub auto_highlight_ev: f32,
}

impl Exposure {
    /// The linear multiplier `ev` stops resolve to, clamped to a finite range.
    pub fn multiplier(&self) -> f32 {
        exp2(self.ev.clamp(-EV_LIMIT, EV_LIMIT))
    }

    /// Resolve the auto-exposure tunables into clamped `AutoExposureSettings`,
    /// or `None` when `auto` is off so the backend can skip the histogram
    /// passes. `hdr_display` shifts the metering pivot from scene white to
    /// perceptual middle gray, since an HDR panel applies no implicit tonemap.
    pub fn auto_settings(&self, hdr_display: bool) -> Option<AutoExposureSettings> {
        self.auto.then(|| {
            AutoExposureSettings::resolve(
                self.auto_min_ev,
                self.auto_max_ev,
                self.auto_speed,
                hdr_display,
            )
            .with_night_bias(self.auto_night_ev)
            .with_night_vision(self.night_vision)
            .with_highlight_headroom(self.auto_highlight_ev)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::auto_exposure::HIGHLIGHT_HEADROOM_EV;

    #[test]
    fn ev_resolves_to_power_of_two_multiplier() {
        let up = Exposure {
            ev: 2.0,
            ..Default::default()
        };
        assert_eq!(up.multiplier(), 4.0);
        let down = Exposure {
            ev: -1.0,
            ..Default::default()
        };
        assert_eq!(down.multiplier(), 0.5);
    }

    #[test]
    fn ev_is_clamped_to_a_finite_multiplier() {
        let cfg = Exposure {
            ev: 1.0e9,
            ..Default::default()
        };
        assert!(cfg.multiplier().is_finite());
        assert_eq!(cfg.multiplier(), EV_LIMIT.exp2());
    }

    #[test]
    fn auto_settings_are_none_when_off() {
        assert!(Exposure::default().auto_settings(false).is_none());
    }

    #[test]
    fn auto_settings_resolve_when_enabled() {
        let cfg = Exposure {
            auto: true,
            auto_min_ev: -4.0,
            auto_max_ev: 6.0,
            auto_speed: 2.0,
            ..Default::default()
        };
        let s = cfg.auto_settings(false).expect("auto-exposure on");
        assert_eq!(s.min_ev, -4.0);
        assert_eq!(s.max_ev, 6.0);
        assert_eq!(s.speed, 2.0);
        assert_eq!(s.night_bias_ev, 0.0);
        let night = Exposure {
            auto: true,
            auto_night_ev: -3.0,
            ..Default::default()
        };
        let s = night.auto_settings(false).expect("auto-exposure on");
        assert_eq!(s.night_bias_ev, -3.0);
        assert_eq!(s.night_vision, 0.0);
        let rods = Exposure {
            auto: true,
            night_vision: 1.0,
            ..Default::default()
        };
        let s = rods.auto_settings(false).expect("auto-exposure on");
        assert_eq!(s.night_vision, 1.0);
        let over = Exposure {
            night_vision: 4.0,
            ..rods
        };
        assert_eq!(over.auto_settings(false).unwrap().night_vision, 1.0);
    }

    #[test]
    fn auto_settings_carry_the_highlight_headroom() {
        let cfg = Exposure {
            auto: true,
            ..Default::default()
        };
        assert_eq!(cfg.auto_highlight_ev, HIGHLIGHT_HEADROOM_EV);
        let s = cfg.auto_settings(false).expect("auto-exposure on");
        assert_eq!(s.highlight_headroom_ev, HIGHLIGHT_HEADROOM_EV);
        let wide = Exposure {
            auto_highlight_ev: 5.0,
            ..cfg
        };
        assert_eq!(
            wide.auto_settings(false).unwrap().highlight_headroom_ev,
            5.0
        );
    }
}
