// Billboard particle-emitter schema.

use crate::{AssetId, TextureHandle, de_opt_texture_handle};

/// A billboard particle emitter.
///
/// Particles spawn from `position` in a cone centred on `direction` (half-angle
/// `spread_deg`), with a speed drawn from `[speed_min, speed_max]` and a
/// lifetime from `[lifetime_min, lifetime_max]`. Over each particle's life its
/// size interpolates from `size_start` to `size_end` and its colour from
/// `color_start` to `color_end`. Each particle is drawn as a camera-facing quad
/// textured by `texture`.
///
/// The pool holds `max_particles` particles; new ones spawn at `spawn_rate` per
/// second, reusing slots as old particles die.
///
/// ```rust
/// # use concinnity_asset::ParticleEmitter;
/// ParticleEmitter {
///     position: [0.0, 1.0, 0.0],
///     direction: [0.0, 1.0, 0.0],
///     spread_deg: 25.0,
///     speed_min: 2.0,
///     speed_max: 5.0,
///     lifetime_min: 0.5,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ParticleEmitter {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// [Texture](#texture) sampled per particle. `None` uses a white fallback so
    /// the colour gradient still shows.
    #[serde(deserialize_with = "de_opt_texture_handle")]
    pub texture: Option<TextureHandle>,
    /// World-space spawn origin.
    pub position: [f32; 3],
    /// Mean emission direction. The cone of width `spread_deg` is centred on
    /// this vector. Normalised on load; a zero vector falls back to `[0, 1, 0]`.
    pub direction: [f32; 3],
    /// Cone half-angle in degrees around `direction`. `0` emits a straight
    /// jet; `180` emits in all directions.
    pub spread_deg: f32,
    /// Lower bound on initial speed (m/s). Floored at 0.
    pub speed_min: f32,
    /// Upper bound on initial speed (m/s). Lifted to at least `speed_min`.
    pub speed_max: f32,
    /// Lower bound on particle lifetime (seconds). Must be > 0.
    pub lifetime_min: f32,
    /// Upper bound on particle lifetime (seconds). Lifted to at least
    /// `lifetime_min`.
    pub lifetime_max: f32,
    /// Constant acceleration applied to each particle, in world units per second
    /// squared.
    pub gravity: [f32; 3],
    /// Particles spawned per second. `0` produces a one-shot burst that then
    /// empties as particles age out.
    pub spawn_rate: f32,
    /// Maximum number of particles alive at once. Clamped to `[1, 65536]`.
    pub max_particles: u32,
    /// Billboard side length at spawn, in world units.
    pub size_start: f32,
    /// Billboard side length at death, in world units.
    pub size_end: f32,
    /// Linear-space RGBA multiplier applied to the texture at spawn.
    pub color_start: [f32; 4],
    /// Linear-space RGBA multiplier applied to the texture at death.
    pub color_end: [f32; 4],
    /// When false the emitter is skipped each frame.
    pub visible: bool,
}

impl Default for ParticleEmitter {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            texture: None,
            position: [0.0, 0.0, 0.0],
            direction: [0.0, 1.0, 0.0],
            spread_deg: 15.0,
            speed_min: 1.0,
            speed_max: 2.0,
            lifetime_min: 1.0,
            lifetime_max: 2.0,
            gravity: [0.0, -9.8, 0.0],
            spawn_rate: 32.0,
            max_particles: 256,
            size_start: 0.2,
            size_end: 0.05,
            color_start: [1.0, 1.0, 1.0, 1.0],
            color_end: [1.0, 1.0, 1.0, 0.0],
            visible: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_emitter_sprays_upward_and_fades_out() {
        let e = ParticleEmitter::default();
        assert_eq!(e.direction, [0.0, 1.0, 0.0]);
        assert_eq!(e.gravity, [0.0, -9.8, 0.0]);
        assert_eq!(e.spread_deg, 15.0);
        assert!(e.speed_min <= e.speed_max);
        assert!(e.lifetime_min <= e.lifetime_max);
        // Particles shrink and fade over their life rather than popping out.
        assert!(e.size_end < e.size_start);
        assert_eq!(e.color_start[3], 1.0);
        assert_eq!(e.color_end[3], 0.0);
        assert_eq!(e.spawn_rate, 32.0);
        assert_eq!(e.max_particles, 256);
        assert!(e.visible);
        assert!(e.texture.is_none());
    }

    #[test]
    fn an_authored_emitter_parses_and_round_trips_through_postcard() {
        crate::test_support::install_resolvers();
        let e: ParticleEmitter = serde_json::from_str(
            r#"{"texture":"tex_spark","position":[0,1,0],"direction":[0,0,1],"spread_deg":45,
                "speed_min":2,"speed_max":6,"lifetime_min":0.5,"lifetime_max":1.5,
                "gravity":[0,0,0],"spawn_rate":120,"max_particles":2048,
                "size_start":0.05,"size_end":0.2,"color_start":[1,0.6,0.2,1],
                "color_end":[1,0,0,0],"visible":false}"#,
        )
        .unwrap();
        assert_eq!(e.texture, Some(TextureHandle(9)));
        assert!(!e.visible);
        // A spark grows as it cools, so size_end above size_start is allowed.
        assert!(e.size_end > e.size_start);

        let bytes = postcard::to_allocvec(&e).unwrap();
        let back: ParticleEmitter = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.texture, Some(TextureHandle(9)));
        assert_eq!(back.direction, [0.0, 0.0, 1.0]);
        assert_eq!(back.gravity, [0.0, 0.0, 0.0]);
        assert_eq!(back.max_particles, 2048);
        assert_eq!(back.color_start, [1.0, 0.6, 0.2, 1.0]);
        assert_eq!(back.asset_id, AssetId::default());
    }
}
