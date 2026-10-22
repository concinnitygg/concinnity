<!-- Auto-generated - do not edit. -->

# Atmosphere

The air above the ground: a physically based sky that scatters sunlight
into blue overhead, white haze at the horizon and red at sunset, and that
looks right from the ground to high above it.

The air is a shell `height_km` thick around the planet. With a
[`Planet`](Planet.md) in the world it wraps that planet, at its center and
radius. Without one, it wraps a sphere of `planet_radius_km` whose top is at
height 0, so a flat world sits on a gently curved ground far below it.

Two kinds of particle scatter the light. Air molecules (Rayleigh) scatter
short wavelengths most, which makes the sky blue and the low sun red; they
thin by a factor of e every `rayleigh_scale_height_km` up. Haze and dust
(Mie) scatter all colors alike, mostly forward, which whitens the horizon
and draws a glow around the sun; they hug the ground more closely. An ozone
layer `ozone_center_km` up absorbs a little orange, which keeps the
twilight sky blue rather than gray. Coefficients are per kilometer of air
at the ground (for ozone, at the layer's center).

The sun is the first [`DirectionalLight`](DirectionalLight.md): its direction
is where the sun is, and its `color * intensity` is the sunlight above the
air. The light that reaches the camera is that sunlight dimmed and
reddened by the air it crossed, and that is what lights the scene: white at
noon, orange and weak at sunset, gone once the sun is below the horizon. A
[`SkyRotation`](SkyRotation.md) is the day cycle: with its axis along the
planet's pole the sun rises and sets as the sky turns, and on a planet its
height also follows where on the planet the camera stands.

On a [`Planet`](Planet.md), its ground and sea are lit point by point: each
takes the sun (and a [`NightSky`](NightSky.md)'s moon) through the air above
it and the light of its own sky, so seen from high above, the day side is
lit, a reddened band runs along the terminator and the night side is dark,
lit by the moon where it is up.

The sky is drawn behind everything, with the sun's disc in it. Distant
surfaces fade into the air's color (aerial perspective), behind any
[`VolumetricFog`](VolumetricFog.md), which stays the nearer layer. The sky
lights the scene in place of an [`EnvironmentMap`](EnvironmentMap.md): its
light fills the shadows, and shiny surfaces reflect it, following the sun
through the day.

One per world. The defaults are Earth's.

## Parameters

- `height_km`: A float. Thickness of the air above the lowest ground, in kilometers. At least 1. Defaults to `100.0`.
- `planet_radius_km`: A float. Radius of the ground the air wraps, in kilometers, when the world has no `Planet`; a planet's own radius is used when it has one. At least 1. Defaults to `6360.0`.
- `rayleigh_scattering`: An array of 3 floats. Rayleigh (air molecule) scattering per kilometer at the ground, for red, green and blue. Defaults to `[0.005802, 0.013558, 0.0331]`.
- `rayleigh_scale_height_km`: A float. Height over which the air thins by a factor of e, in kilometers. Defaults to `8.0`.
- `mie_scattering`: An array of 3 floats. Mie (haze) scattering per kilometer at the ground. Defaults to `[0.003996, 0.003996, 0.003996]`.
- `mie_absorption`: An array of 3 floats. Mie (haze) absorption per kilometer at the ground: haze that darkens rather than brightens. Defaults to `[0.000444, 0.000444, 0.000444]`.
- `mie_scale_height_km`: A float. Height over which the haze thins by a factor of e, in kilometers. Defaults to `1.2`.
- `mie_anisotropy`: A float. How strongly haze scatters forward, in `[-0.99, 0.99]`: higher draws a tighter, brighter glow around the sun. Defaults to `0.8`.
- `ozone_absorption`: An array of 3 floats. Ozone absorption per kilometer at the center of the ozone layer. Defaults to `[0.00065, 0.001881, 0.000085]`.
- `ozone_center_km`: A float. Height of the ozone layer's densest point, in kilometers. Defaults to `25.0`.
- `ozone_width_km`: A float. Full width of the ozone layer, in kilometers: its density falls off linearly to nothing half this far above and below its center. Defaults to `30.0`.
- `ground_albedo`: An array of 3 floats. Linear-space color of the ground as seen from far above: the light it bounces back into the air. Defaults to `[0.3, 0.3, 0.3]`.
- `sun_angular_diameter_deg`: A float. Angular diameter of the sun's disc, in degrees. Clamped to `[0.05, 10]`. Defaults to `0.53`.
- `sun_disc_intensity`: A float. Brightness of the sun's disc, as a multiple of the sunlight it casts. 0 hides the disc. Defaults to `1000.0`.
- `apparent_radius_km`: A float. Radius, in kilometers, of the planet the sky is drawn over while the camera is near the ground, in place of the true one. A small planet's own sky is a flat deep blue with a pale sun at sunset, since a ray along its horizon crosses little air; drawn over an Earth-sized radius it gets a pale horizon and red sunsets. The haze over the ground keeps its true distances, and the horizon still meets the ground where it really is. At least 1. `null` (the default) draws the sky over the true radius.
- `apparent_radius_fade_km`: An array of 2 floats. Heights above the ground, in kilometers, between which the sky fades from `apparent_radius_km` to the true radius as the camera climbs, so the planet's limb seen from high up is its own. The air's heights describe it over the apparent ground and shrink in proportion with the radius, its coefficients growing to keep its optical depth. Defaults to `[1.0, 20.0]`.
