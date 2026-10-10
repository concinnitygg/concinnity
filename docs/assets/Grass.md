<!-- Auto-generated - do not edit. -->

# Grass

The look of a grass cover: individual blades, grown on the GPU each frame
and swayed by the world's [Wind](Wind.md).

A `Grass` grows only where a [Terrain](Terrain.md) layer names it: the
terrain sets where the blades root and which way the ground slopes, and the
layer's mask sets where they grow. Blades are scattered evenly at `density`
per square meter, gathered into clumps about `clump_size` meters across
whose blades share their height and lean, and placed the same way on every
run. Blades on a slope lean downhill.

Each blade is real geometry: a tapered strip that curves under its own
weight and the wind, shaded from `root_color` at the ground to `tip_color`
at the tip, with light passing through it when the sun is behind. Blades are
drawn out to a fixed distance from the camera and thin out toward it.

## Parameters

- `height`: A float. Mean blade height, in meters. Defaults to `0.5`.
- `height_variance`: A float. How much blade heights vary around `height`, as a fraction of it in [0, 1]. Defaults to `0.4`.
- `width`: A float. Blade width at the root, in meters. Blades taper to a point. Defaults to `0.03`.
- `density`: A float. Blades per square meter. Defaults to `120.0`.
- `clump_size`: A float. Typical clump diameter, in meters. Defaults to `0.6`.
- `stiffness`: A float. How firmly blades resist bending, in [0, 1]. 0 lies flat in a strong wind; 1 barely moves. Defaults to `0.5`.
- `root_color`: An array of 3 floats. Linear-space RGB color at the blade root. Defaults to `[0.025, 0.06, 0.012]`.
- `tip_color`: An array of 3 floats. Linear-space RGB color at the blade tip. Defaults to `[0.2, 0.32, 0.065]`.
- `color_variation`: A float. How much hue and brightness vary between clumps and blades, in [0, 1]. Defaults to `0.35`.
- `visible`: A boolean. When false no terrain grows this grass. Defaults to `true`.
