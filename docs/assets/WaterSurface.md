<!-- Auto-generated - do not edit. -->

# WaterSurface

A translucent animated water surface.

A flat, subdivided horizontal surface moved by the world's `Wind`. The wind
raises a field of waves of many lengths, mostly running downwind with some
crossing it, each traveling at the speed deep water carries a wave of its
length, plus fine ripples that roughen the surface. Gusts roll across it in
step with the grass: rough dark patches where a gust touches down, glassier
water in the lulls. A stronger wind and a longer `fetch_meters` (the open
water the wind blows across) build longer, taller waves; with no `Wind`
the water lies almost mirror-still. Authored `waves` add swell on top.

Large open water, a surface hundreds of meters to kilometers across, is a
simulated ocean: its wind waves rise, cross and fade continuously over the
whole surface with no repeating pattern, crests sharpen and troughs
broaden, and the water stays as detailed beside the camera as anywhere,
however large the surface or far from the world origin. Pools, ponds and
small lakes keep a lighter sum of waves. The graphics quality setting
trades the ocean's finest detail for speed.

A simulated ocean can whitecap, as strongly as `whitecaps` asks (none by
default): where a crest grows steep enough to break, a patch of white foam
appears on it and stays behind as the wave runs on, thinning into lacy
streaks over a few seconds. How much of the sea is white follows the wind
as at sea: almost none below about 4 m/s, about one percent at 10 m/s, a
few percent at 14 m/s and more in a gale. Whitecaps hide the reflection
and the sun's glint beneath them. Smaller water never whitecaps.

Far off, waves too fine to draw fold into a rougher surface, so the sun or
moon spreads into a glittering path instead of shimmering.

It refracts and reflects the scene, and its body is lit like any
other surface: the sun or moon (in shadow where the scene casts one) and the
sky's light scatter back out of the water, off a reflectance that blends
from `shallow_color` to `deep_color` as the water deepens. Shoreline foam
and whitecaps are a lit white surface.

A [`PropBody`](PropBody.md) with `buoyancy` floats in the water on the waves
it draws, buoyed by the water's `density`.

The surface is a rectangle positioned by `center` and sized by `extent` (XZ
half-widths). The mesh itself is flat; all height variation comes from the
animated waves.

A [`PlanetSea`](PlanetSea.md) takes its look from the surface it names, and
that surface is drawn as the sea rather than as a rectangle.

Its sun glint rides the waves, and its reflection and refraction move with
neither the surface nor what they show, so temporal anti-aliasing and
upscaling favor the current frame where the glint shows and, within about
16 meters of the camera, where the reflection is strong. Farther off they
keep their history, which holds distant waves still.

## Parameters

- `center`: An array of 3 floats. World-space position of a rectangle's center. Defaults to `[0.0, 0.0, 0.0]`.
- `extent`: An array of 2 floats. Half-width and half-depth of a rectangle `[x, z]`, in world units. Defaults to `[10.0, 10.0]`.
- `subdivisions`: An integer. Grid subdivisions across a rectangle. Higher gives smoother waves. Clamped to [8, 255]. Defaults to `64`.
- `fetch_meters`: A float. The open water the wind blows across, in meters: how large a sea it can build. A pond is tens of meters, a lake one to a few thousand, an open sea a hundred thousand or more. At least 1. Defaults to `1000.0`.
- `wind_response`: A float. How strongly the wind moves this surface, at least 0. 0 is sheltered water the wind never touches, 1 the full wind; above 1 exaggerates it. Defaults to `1.0`.
- `density`: A float. The water's density, in kilograms per cubic meter, at least 1: how strongly it buoys what floats in it. Seawater by default; fresh water is 1000.
- `depth_meters`: A float. How deep a rectangle's water is, in meters below its rest, at least 0: anything lower is out of the water, so a cave under a pond stays dry. Defaults to `10.0`.
- `waves`: An array of [WaterWave](WaterWave.md) objects. Extra swell rolling over the wind's waves, up to 4. None by default, and none on a planet's sea.
- `deep_color`: An array of 3 floats. Linear-space RGB reflectance of deep water's body, each channel in [0, 1]: the share of the light reaching the surface that a deep column scatters back out. Values outside [0, 1] are clamped. Defaults to `[0.02, 0.05, 0.15]`.
- `shallow_color`: An array of 3 floats. Linear-space RGB reflectance of shallow water's body near the shore, each channel in [0, 1]. The seabed seen through the shallows is filtered by it too. Values outside [0, 1] are clamped. Defaults to `[0.2, 0.5, 0.55]`.
- `depth_falloff_meters`: A float. Depth over which the reflectance blends from shallow to deep, in meters. Defaults to `4.0`.
- `foam_width_meters`: A float. Width of the shoreline foam band, in meters. Defaults to `0.3`.
- `foam_intensity`: A float. Strength of the shoreline foam, in [0, 1]. Defaults to `0.8`.
- `whitecaps`: A float. Strength of the foam on a simulated ocean's breaking crests, in [0, 1]. 0 is none; smaller water never whitecaps. Values outside [0, 1] are clamped. Defaults to `0.0`.
- `spray`: A string. The particle emitter that throws spray where floating bodies plunge into the water, leave it, or plow through it fast, a burst at a time from where they meet the surface. None throws no spray.
- `fresnel_power`: A float. Sharpness of the grazing-angle reflection. Higher confines reflections to steeper viewing angles. Defaults to `5.0`.
- `roughness`: A float. Surface roughness in [0, 1]. Higher gives blurrier reflections, and pushes a mirrored reflection further off its line with each wave: a near-mirror surface keeps its reflection almost still. Defaults to `0.05`.
- `refraction_strength`: A float. How strongly the surface bends the view of what's beneath it. Defaults to `0.15`.
- `visible`: A boolean. When false the surface is skipped each frame. Defaults to `true`.
