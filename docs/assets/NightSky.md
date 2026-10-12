<!-- Auto-generated - do not edit. -->

# NightSky

The sky beyond the air: stars, the Milky Way, distant galaxies and the
moon, fixed to the celestial sphere so they rise and set as a
[SkyRotation](SkyRotation.md) turns it.

Under an [Atmosphere](Atmosphere.md) the night sky is seen through the air:
it dims and hides toward the horizon, and the daylight sky outshines it,
so it comes out as the sun sets and the twilight fades. Without one it is
seen as from airless ground: black between the stars, with the sun's disc
in it by day, and it lights the scene in place of an
[EnvironmentMap](EnvironmentMap.md).

The celestial pole is the [SkyRotation](SkyRotation.md)'s axis (straight up
without one). The Milky Way is a band around the sky whose own pole leans
`milky_way_tilt_deg` away from the celestial pole, toward the side
`milky_way_turn_deg` around the pole from the sun, with its bright core
`milky_way_core_deg` along the band. Stars and galaxies are placed by
`seed`: each seed is a different sky.

The moon circles the sky once every `moon_period_days` days (a day being
one full turn of the sky), drifting east against the stars, so it rises
a little later each day. Its lit side faces the sun, so its phase follows
its angle from the sun: new beside it, full opposite it. At night it is a
second, dim [DirectionalLight](DirectionalLight.md), dimmed and reddened by
the air like the sun. Only one directional light casts shadows: once the
sun has fully set the moon takes them over, its light fading in over the
next 6 degrees of the sun's descent, and it hands them back the same way
before sunrise, so neither light pops.

The defaults are the real sky's: a full moon lights the ground about
400000 times less than the sun, and the Milky Way is fainter still. A
world with auto-exposure that should adapt to the night needs its
[PostProcessConfig](PostProcessConfig.md) to allow it: an
`auto_exposure_max_ev` of 24, since a moonlit or starlit scene adapts at
around 19 to 20, and an `exposure_ev` about 4 stops under, so moonlit
ground reads while the night stays darker than day and dusk stays dim.

One per world. Brightnesses are relative to the sunlight, so the night
keeps its balance with any sun.

## Parameters

- `star_brightness`: A float. Brightness of the stars and galaxies, as a multiple of the default sky's. 0 hides them.
- `star_limit_magnitude`: A float. Magnitude of the faintest stars drawn, in `[2, 8]`. Each step up draws about three times as many: 6.5 is the naked eye's ~9000 stars. Defaults to `6.5`.
- `seed`: An integer. Places the stars and galaxies: each seed is a different sky. Defaults to `0`.
- `milky_way_brightness`: A float. Brightness of the Milky Way, as a multiple of the default sky's. 0 hides it.
- `milky_way_tilt_deg`: A float. Angle between the Milky Way's pole and the celestial pole, in degrees: 0 lays the band along the celestial equator, 90 runs it over the pole. Defaults to `62.9`.
- `milky_way_turn_deg`: A float. Which way around the celestial pole the Milky Way's pole leans, in degrees east of the sun. Defaults to `0.0`.
- `milky_way_core_deg`: A float. Where along the band its bright core sits, in degrees. Defaults to `0.0`.
- `galaxies`: An integer. Distant galaxies drawn, spiral and elliptical, each a faint smudge under a few degrees across. At most 16. Defaults to `3`.
- `moon`: A boolean. Whether the sky has a moon. Defaults to `true`.
- `moon_start_deg`: A float. The moon's angle east of the sun along its path while the sky's angle is 0, in degrees: 0 is a new moon, 90 the first quarter, 180 full. Defaults to `180.0`.
- `moon_period_days`: A float. Days, each a full turn of the sky, the moon takes to come back to the same phase. 0 holds it still against the stars. Defaults to `29.53`.
- `moon_inclination_deg`: A float. Tilt of the moon's path, in degrees. Untilted, the path is the great circle that leaves the sun heading due east; the tilt turns it about the line to the sun, carrying the moon north and south of that circle. Defaults to `5.1`.
- `moon_angular_diameter_deg`: A float. Angular diameter of the moon's disc, in degrees. Clamped to `[0.05, 20]`. Defaults to `0.52`.
- `moon_brightness`: A float. The full moon's light as a fraction of the sunlight. The default is the real moon's, 14 magnitudes fainter than the sun.
- `moon_disc_brightness`: A float. Brightness of the moon's face as a multiple of the face whose light, over the whole disc, is `moon_brightness`. At 1 the full moon's face is as bright as a gray rock in sunlight, far brighter than the ground it lights. Defaults to `1.0`.
