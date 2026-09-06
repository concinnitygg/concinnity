<!-- Auto-generated - do not edit. -->

# SdfVolume

A raymarched signed-distance-field volume. It occupies a world-space
bounding box; a user-authored fragment shader sphere-traces an SDF inside
the box, composites correctly with the surrounding scene through the depth
buffer, and shades hits with the engine's lighting helpers.

The distance field is one `.slang` file for every backend. The build
compiles it, so a field that does not compile fails `cn build` rather than
the renderer, and a shipped player needs no shader compiler of its own.

## Parameters

- `centre`: An array of 3 floats. World-space centre of the bounding box. Defaults to `[0.0, 0.0, 0.0]`.
- `extent`: An array of 3 floats. XYZ half-widths of the bounding box. The raymarch is clipped to the box, so the SDF only has to be well-defined inside this region. Defaults to `[1.0, 1.0, 1.0]`.
- `fragment_shader`: A string. Distance-field source path (e.g. `"shaders/chrome_blob.slang"`), resolved relative to the project's `assets/` at build time. The file defines `map` and `shade`, or `sampleVolume` for a volumetric volume.
- `max_gradient`: A float. Worst-case gradient of the SDF, used to size the cone-march step. `1.0` is correct for any well-formed SDF; higher values shorten the step but stay safe. Must be > 0. Defaults to `1.0`.
- `max_steps`: An integer. Maximum cone-march steps per pixel. Clamped to `[8, 256]`. Defaults to `64`.
- `max_distance`: A float. Maximum march distance in metres. Must be ≥ 0.1. Defaults to `30.0`.
- `params`: An array of floats. Generic parameter block passed to the shader as a uniform buffer; the shader interprets it however it likes. Up to 32 values.
- `cast_shadows`: A boolean. When true, the volume casts shadows onto the surrounding scene. Disable for translucent / volumetric effects that shouldn't block light. Defaults to `false`.
- `receive_shadows`: A boolean. When true (the default), the volume is shadowed by the scene. Set to false for unlit / always-bright effects (energy fields, etc.).
- `volumetric`: A boolean. When true, the volume renders as a participating medium (clouds, smoke, fog blobs, energy fields) instead of an opaque surface. The shader must define `sampleVolume(p, params, time)` returning per-point density, scattering colour, and emission instead of `map` / `shade`. Volumetrics never cast shadows (`cast_shadows` is forced off). The medium fills the whole bounding box, so don't overlap it with geometry it should render behind. Defaults to `false`.
- `visible`: A boolean. When false the volume is skipped each frame. Defaults to `true`.
