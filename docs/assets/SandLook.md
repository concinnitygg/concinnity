<!-- Auto-generated - do not edit. -->

# SandLook

The look of a patch of sand, one meter across and repeating seamlessly:
wind ripples over loose grains, dark mineral grains, and scattered shell
fragments.

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
