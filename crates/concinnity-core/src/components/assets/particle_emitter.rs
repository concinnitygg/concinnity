// Billboard particle-emitter schema.

use crate::ecs::TextureHandle;

/// A billboard particle emitter.
///
/// Particles spawn from `position` in a cone centered on `direction` (half-angle
/// `spread_deg`), with a speed drawn from `[speed_min, speed_max]` and a
/// lifetime from `[lifetime_min, lifetime_max]`. Over each particle's life its
/// size interpolates from `size_start` to `size_end` and its color from
/// `color_start` to `color_end`. Each particle is drawn as a camera-facing quad
/// textured by `texture`.
///
/// The pool holds `max_particles` particles; new ones spawn at `spawn_rate` per
/// second, taking the pool's slots in turn. A slot is reused only once the
/// particle in it has certainly died, so a pool smaller than `spawn_rate` times
/// `lifetime_max` fills up and drops new particles until slots free up. Under a
/// fixed frame rate the emitter plays out identically on every run.
///
/// Particles carry no motion vectors. Where one covers a pixel, temporal
/// anti-aliasing and upscaling favor the current frame over their history in
/// proportion to its opacity, or to the light it adds where a bright particle
/// fading out adds more, so a moving particle does not smear.
///
/// ```rust
/// # use concinnity_core::components::ParticleEmitter;
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
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct ParticleEmitter {
    /// [Texture](#texture) sampled per particle. `None` uses a white fallback so
    /// the color gradient still shows.
    pub texture: Option<TextureHandle>,
    /// World-space spawn origin.
    pub position: [f32; 3],
    /// Mean emission direction. The cone of width `spread_deg` is centered on
    /// this vector. Normalized on load; a zero vector falls back to `[0, 1, 0]`.
    #[asset(default = [0.0, 1.0, 0.0])]
    pub direction: [f32; 3],
    /// Cone half-angle in degrees around `direction`. `0` emits a straight
    /// jet; `180` emits in all directions.
    #[asset(default = 15.0)]
    pub spread_deg: f32,
    /// Lower bound on initial speed (m/s). Floored at 0.
    #[asset(default = 1.0)]
    pub speed_min: f32,
    /// Upper bound on initial speed (m/s). Lifted to at least `speed_min`.
    #[asset(default = 2.0)]
    pub speed_max: f32,
    /// Lower bound on particle lifetime (seconds). Must be > 0.
    #[asset(default = 1.0)]
    pub lifetime_min: f32,
    /// Upper bound on particle lifetime (seconds). Lifted to at least
    /// `lifetime_min`.
    #[asset(default = 2.0)]
    pub lifetime_max: f32,
    /// Constant acceleration applied to each particle, in world units per second
    /// squared.
    #[asset(default = [0.0, -9.8, 0.0])]
    pub gravity: [f32; 3],
    /// Particles spawned per second. `0` produces a one-shot burst that then
    /// empties as particles age out.
    #[asset(default = 32.0)]
    pub spawn_rate: f32,
    /// Maximum number of particles alive at once; while the pool is full, new
    /// particles are dropped. Emitting at the full `spawn_rate` takes
    /// `spawn_rate * lifetime_max` slots. Clamped to `[1, 65536]`.
    #[asset(default = 256)]
    pub max_particles: u32,
    /// Billboard side length at spawn, in world units.
    #[asset(default = 0.2)]
    pub size_start: f32,
    /// Billboard side length at death, in world units.
    #[asset(default = 0.05)]
    pub size_end: f32,
    /// Linear-space RGBA multiplier applied to the texture at spawn.
    #[asset(default = [1.0, 1.0, 1.0, 1.0])]
    pub color_start: [f32; 4],
    /// Linear-space RGBA multiplier applied to the texture at death.
    #[asset(default = [1.0, 1.0, 1.0, 0.0])]
    pub color_end: [f32; 4],
    /// When false the emitter is skipped each frame.
    #[asset(default = true)]
    pub visible: bool,
}
