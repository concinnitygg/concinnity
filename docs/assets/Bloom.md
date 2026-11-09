<!-- Auto-generated - do not edit. -->

# Bloom

Bloom: light from the brightest parts of the image spilling softly into
their surroundings. The `bloom` section of [`GraphicsConfig`](GraphicsConfig.md);
a field left unset takes the default below.

## Parameters

- `intensity`: A float. Additive bloom contribution. 0 skips bloom entirely. Defaults to `0.6`. At least `0`. Applies live.
- `threshold`: A float. Brightness threshold. Pixels brighter than this contribute fully; pixels within `knee` below it ramp in softly. Defaults to `1.0`. At least `0`. Applies live.
- `knee`: A float. Width of the soft knee just below `threshold`. Defaults to `0.5`. At least `0`. Applies live.
