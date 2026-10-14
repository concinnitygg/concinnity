// Night sky schema.

/// The sky beyond the air: stars, the Milky Way, distant galaxies and the
/// moon, fixed to the celestial sphere so they rise and set as a
/// [SkyRotation](#skyrotation) turns it.
///
/// Under an [Atmosphere](#atmosphere) the night sky is seen through the air:
/// it dims and hides toward the horizon, and the daylight sky outshines it,
/// so it comes out as the sun sets and the twilight fades. Without one it is
/// seen as from airless ground: black between the stars, with the sun's disc
/// in it by day, and it lights the scene in place of an
/// [EnvironmentMap](#environmentmap).
///
/// The celestial pole is the [SkyRotation](#skyrotation)'s axis (straight up
/// without one). The Milky Way is a band around the sky whose own pole leans
/// `milky_way_tilt_deg` away from the celestial pole, toward the side
/// `milky_way_turn_deg` around the pole from the sun, with its bright core
/// `milky_way_core_deg` along the band. Stars and galaxies are placed by
/// `seed`: each seed is a different sky.
///
/// The moon circles the sky once every `moon_period_days` days (a day being
/// one full turn of the sky), drifting east against the stars, so it rises
/// a little later each day. Its lit side faces the sun, so its phase follows
/// its angle from the sun: new beside it, full opposite it. At night it is a
/// second, dim [DirectionalLight](#directionallight), dimmed and reddened by
/// the air like the sun. Only one directional light casts shadows: once the
/// sun has fully set the moon takes them over, its light fading in over the
/// next 6 degrees of the sun's descent, and it hands them back the same way
/// before sunrise, so neither light pops.
///
/// The defaults are the real sky's: a full moon lights the ground about
/// 400000 times less than the sun, and the Milky Way is fainter still. A
/// world with auto-exposure that should adapt to the night needs its
/// [Exposure](#exposure) to allow it: an `auto_max_ev` of 24, since a moonlit
/// or starlit scene adapts at around 19 to 20, and an `ev` about 4 stops under, so moonlit
/// ground reads while the night stays darker than day and dusk stays dim.
///
/// One per world. Brightnesses are relative to the sunlight, so the night
/// keeps its balance with any sun.
///
/// ```rust
/// # use concinnity_core::components::NightSky;
/// NightSky {
///     galaxies: 6,
///     moon_start_deg: 0.0,
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct NightSky {
    /// Brightness of the stars and galaxies, as a multiple of the default
    /// sky's. 0 hides them.
    #[asset(default = 1.0)]
    pub star_brightness: f32,
    /// Magnitude of the faintest stars drawn, in `[2, 8]`. Each step up draws
    /// about three times as many: 6.5 is the naked eye's ~9000 stars.
    #[asset(default = 6.5)]
    pub star_limit_magnitude: f32,
    /// Places the stars and galaxies: each seed is a different sky.
    pub seed: u32,
    /// Brightness of the Milky Way, as a multiple of the default sky's. 0
    /// hides it.
    #[asset(default = 1.0)]
    pub milky_way_brightness: f32,
    /// Angle between the Milky Way's pole and the celestial pole, in degrees:
    /// 0 lays the band along the celestial equator, 90 runs it over the
    /// pole.
    #[asset(default = 62.9)]
    pub milky_way_tilt_deg: f32,
    /// Which way around the celestial pole the Milky Way's pole leans, in
    /// degrees east of the sun.
    pub milky_way_turn_deg: f32,
    /// Where along the band its bright core sits, in degrees.
    pub milky_way_core_deg: f32,
    /// Distant galaxies drawn, spiral and elliptical, each a faint smudge
    /// under a few degrees across. At most 16.
    #[asset(default = 3)]
    pub galaxies: u32,
    /// Whether the sky has a moon.
    #[asset(default = true)]
    pub moon: bool,
    /// The moon's angle east of the sun along its path while the sky's angle
    /// is 0, in degrees: 0 is a new moon, 90 the first quarter, 180 full.
    #[asset(default = 180.0)]
    pub moon_start_deg: f32,
    /// Days, each a full turn of the sky, the moon takes to come back to the
    /// same phase. 0 holds it still against the stars.
    #[asset(default = 29.53)]
    pub moon_period_days: f32,
    /// Tilt of the moon's path, in degrees. Untilted, the path is the great
    /// circle that leaves the sun heading due east; the tilt turns it about
    /// the line to the sun, carrying the moon north and south of that circle.
    #[asset(default = 5.1)]
    pub moon_inclination_deg: f32,
    /// Angular diameter of the moon's disc, in degrees. Clamped to
    /// `[0.05, 20]`.
    #[asset(default = 0.52)]
    pub moon_angular_diameter_deg: f32,
    /// The full moon's light as a fraction of the sunlight. The default is
    /// the real moon's, 14 magnitudes fainter than the sun.
    #[asset(default = 0.0000025)]
    pub moon_brightness: f32,
    /// Brightness of the moon's face as a multiple of the face whose light,
    /// over the whole disc, is `moon_brightness`. At 1 the full moon's face
    /// is as bright as a gray rock in sunlight, far brighter than the ground
    /// it lights.
    #[asset(default = 1.0)]
    pub moon_disc_brightness: f32,
}
