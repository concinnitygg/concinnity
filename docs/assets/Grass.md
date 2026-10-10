<!-- Auto-generated - do not edit. -->

# Grass

A field of individual grass blades, grown on the GPU each frame and swayed
by the world's [Wind](Wind.md).

The field covers a flat rectangle: `center` sets its middle (and the
ground height the blades root at), `extent` its half-width and half-depth.
Blades are scattered evenly at `density` per square meter, gathered into
clumps about `clump_size` meters across whose blades share their height and
lean, and placed the same way on every run.

Each blade is real geometry: a tapered strip that curves under its own
weight and the wind, shaded from `root_color` at the ground to `tip_color`
at the tip, with light passing through it when the sun is behind. Blades are
drawn out to a fixed distance from the camera and thin out toward it.

One field is drawn per world: the first visible `Grass` declared.

## Parameters

- `center`: An array of 3 floats. World-space center of the field; its height is the ground the blades grow from. Defaults to `[0.0, 0.0, 0.0]`.
- `extent`: An array of 2 floats. Half-width and half-depth of the field `[x, z]`, in meters. Defaults to `[10.0, 10.0]`.
- `height`: A float. Mean blade height, in meters. Defaults to `0.5`.
- `height_variance`: A float. How much blade heights vary around `height`, as a fraction of it in [0, 1]. Defaults to `0.4`.
- `width`: A float. Blade width at the root, in meters. Blades taper to a point. Defaults to `0.03`.
- `density`: A float. Blades per square meter. Defaults to `120.0`.
- `clump_size`: A float. Typical clump diameter, in meters. Defaults to `0.6`.
- `stiffness`: A float. How firmly blades resist bending, in [0, 1]. 0 lies flat in a strong wind; 1 barely moves. Defaults to `0.5`.
- `root_color`: An array of 3 floats. Linear-space RGB color at the blade root. Defaults to `[0.025, 0.06, 0.012]`.
- `tip_color`: An array of 3 floats. Linear-space RGB color at the blade tip. Defaults to `[0.2, 0.32, 0.065]`.
- `color_variation`: A float. How much hue and brightness vary between clumps and blades, in [0, 1]. Defaults to `0.35`.
- `visible`: A boolean. When false the field is not drawn. Defaults to `true`.
