<!-- Auto-generated - do not edit. -->

# WaterWave

One authored swell wave on a water surface, on top of the waves the wind
raises. Swell is the long, regular motion left by distant weather: it rolls
along `direction`, rising and falling `amplitude` around the rest height,
`wavelength` between crests, at the speed deep water carries a wave that
long (about 1.25 m/s times the square root of the wavelength in meters).
`steepness` in [0, 1] pinches the crests and broadens the troughs.

## Parameters

- `amplitude`: A float. Peak height of the wave, in world units. Defaults to `0.15`.
- `wavelength`: A float. Distance between successive crests, in world units. Defaults to `4.0`.
- `direction`: An array of 2 floats. Horizontal travel direction `[x, z]`. Does not need to be normalized. Defaults to `[1.0, 0.0]`.
- `steepness`: A float. Crest sharpness in [0, 1]. 0 is a smooth sine; higher pinches crests and broadens troughs. Defaults to `0.4`.
