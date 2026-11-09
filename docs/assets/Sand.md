<!-- Auto-generated - do not edit. -->

# Sand

A sand surface, drawn by the cook into seamless one-meter textures.

Expands into a [Material](Material.md) of the same name, so a
[Planet](Planet.md)'s `material` or a cover's `material` names it directly,
and the two [Texture](Texture.md)s that material samples:
`<name>_albedo` (base color, with the relief's height in alpha) and
`<name>_normal`. The textures repeat every UV unit; a planet's ground maps
one unit to a meter.

## Parameters

- `color`: An array of 3 floats. Mean linear albedo of the dry sand. Defaults to `[0.45, 0.38, 0.27]`.
- `roughness`: A float. Perceptual roughness of the dry sand. Defaults to `0.9`.
- `ripple_spacing`: A float. Crest-to-crest spacing of the wind ripples, in meters. Defaults to `0.09`.
- `ripple_height`: A float. Trough-to-crest height of the ripples, in meters. 0 leaves the sand unrippled. Defaults to `0.006`.
- `ripple_direction_deg`: A float. Direction the wind blew the ripples along, in degrees from the texture's u axis. The crests run across it. Defaults to `0.0`.
- `grain`: A float. How coarse and loose the grains look, in [0, 1]. Defaults to `0.5`.
- `dark_grains`: A float. Share of dark mineral grains speckling the sand and gathering in the ripples' troughs, in [0, 1]. Defaults to `0.25`.
- `shells`: A float. How thickly shell fragments are strewn, in [0, 1]. Defaults to `0.1`.
- `shell_color`: An array of 3 floats. Linear albedo of the shell fragments. Defaults to `[0.78, 0.72, 0.63]`.
- `seed`: An integer. Seed of the pattern; two sands with different seeds never repeat each other. Defaults to `0`.
- `resolution`: An integer. Texels across each one-meter tile of its textures. Defaults to `1024`.
