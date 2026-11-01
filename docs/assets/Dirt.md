<!-- Auto-generated - do not edit. -->

# Dirt

A dirt surface, drawn by the cook into seamless one-meter textures.

Expands into a [Material](Material.md) of the same name, so a
[Planet](Planet.md)'s `material` or a cover's `material` names it directly,
and the two [Texture](Texture.md)s that material samples:
`<name>_albedo` (base color, with the relief's height in alpha) and
`<name>_normal`. The textures repeat every UV unit; a planet's ground maps
one unit to a meter.

## Parameters

- `color`: An array of 3 floats. Mean linear albedo of the soil. Defaults to `[0.1, 0.075, 0.052]`.
- `roughness`: A float. Perceptual roughness of the soil. Defaults to `0.95`.
- `clod_size`: A float. Typical width of a clod, in meters. Defaults to `0.04`.
- `clods`: A float. How lumpy the clods stand, in [0, 1]. Defaults to `0.6`.
- `stones`: A float. How thickly stones lie in the soil, in [0, 1]. Defaults to `0.25`.
- `stone_size`: A float. Typical width of a stone, in meters. Defaults to `0.035`.
- `stone_color`: An array of 3 floats. Linear albedo of the stones. Defaults to `[0.17, 0.16, 0.14]`.
- `humus`: A float. How dark and patchy the humus mottling is, in [0, 1]. Defaults to `0.5`.
- `cracks`: A float. How deeply the soil has dried and cracked, in [0, 1]. 0 leaves it uncracked. Defaults to `0.0`.
- `seed`: An integer. Seed of the pattern; two dirts with different seeds never repeat each other. Defaults to `0`.
- `resolution`: An integer. Texels across each one-meter tile of its textures. Defaults to `1024`.
