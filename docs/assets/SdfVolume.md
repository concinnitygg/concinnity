<!-- Auto-generated - do not edit. -->

# SdfVolume

A raymarched signed-distance-field volume. It occupies a world-space
bounding box; a user-authored fragment shader sphere-traces an SDF inside
the box, composites correctly with the surrounding scene through the depth
buffer, and shades hits with the engine's lighting helpers.

The distance field is one `.hlsl` file for every backend. The build
compiles it, so a field that does not compile fails `cn build` rather than
the renderer, and a shipped player needs no shader compiler of its own.

# The distance field

A field file defines functions, not an entry point. The engine owns the
march, the lighting and every binding, and calls the field from inside its
own pass. A surface volume's file defines the shape and its material:

```hlsl
float map(float3 p, SdfParams params, float time);
SdfSurface shade(float3 p, float3 normal, SdfParams params, float time,
                 float2 frag_uv);
```

and a volumetric one defines the medium instead:

```hlsl
VolumeSample sampleVolume(float3 p, SdfParams params, float time);
```

`p` is a world-space point. Volumes may share one field file, so a field
that places its shape at `volume_center()` draws it in whichever volume
reads it:

```hlsl
float map(float3 p, SdfParams params, float time)
{
    return sdSphere(p - volume_center(), 0.5);
}
```

Besides its own functions, a field can call:

- `sdf_param(params, i)`: parameter `i`, 0 to 31, of this volume's
  `params`.
- `volume_center()`: the world-space center of the volume being drawn.
- `volume_extent()`: its half-widths.
- `sdSphere(p, r)`, `sdBox(p, b)`, `sdRoundBox(p, b, r)`, `sdTorus(p, t)`,
  `sdCapsule(p, a, b, r)` and `sdPlane(p, n, h)`: distances to primitives
  around the origin.
- `opSmoothUnion(a, b, k)`, `opSmoothSubtraction(d1, d2, k)` and
  `opSmoothIntersection(a, b, k)`: two distances combined with a blend of
  width `k`.
- `sampleSceneRefracted(frag_uv, normal, strength)`: the scene behind the
  surface, bent by the normal, for a refractive `shade`.

`shade` returns an `SdfSurface`: `albedo`, `roughness`, `metallic`,
`emissive`, and `transmitted`, a color shown through the surface on top of
its lighting (zero for an opaque one). `sampleVolume` returns a
`VolumeSample`: `density` (0 is empty), `scattering`, which the sun's
light multiplies, and `emission`.

## Parameters

- `center`: An array of 3 floats. World-space center of the bounding box. Defaults to `[0.0, 0.0, 0.0]`.
- `extent`: An array of 3 floats. XYZ half-widths of the bounding box. The raymarch is clipped to the box, so the SDF only has to be well-defined inside this region. Defaults to `[1.0, 1.0, 1.0]`.
- `fragment_shader`: A string. Distance-field source path (e.g. `"shaders/chrome_blob.hlsl"`), resolved relative to the project's `assets/` at build time. The file defines `map` and `shade`, or `sampleVolume` for a volumetric volume. Defaults to `""`.
- `max_gradient`: A float. Worst-case gradient of the SDF, used to size the cone-march step. `1.0` is correct for any well-formed SDF; higher values shorten the step but stay safe. Must be > 0. Defaults to `1.0`.
- `max_steps`: An integer. Maximum cone-march steps per pixel. Clamped to `[8, 256]`. Defaults to `64`.
- `max_distance`: A float. Maximum march distance in meters. Must be ≥ 0.1. Defaults to `30.0`.
- `params`: An array of 32 floats. Generic parameter block passed to the shader as a uniform buffer; the shader interprets it however it likes. Up to 32 values. Defaults to `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`.
- `cast_shadows`: A boolean. When true, the volume casts shadows onto the surrounding scene. Disable for translucent / volumetric effects that shouldn't block light. Defaults to `false`.
- `receive_shadows`: A boolean. When true (the default), the volume is shadowed by the scene. Set to false for unlit / always-bright effects (energy fields, etc.).
- `volumetric`: A boolean. When true, the volume renders as a participating medium (clouds, smoke, fog blobs, energy fields) instead of an opaque surface. The shader must define `sampleVolume(p, params, time)` returning per-point density, scattering color, and emission instead of `map` / `shade`. Volumetrics never cast shadows (`cast_shadows` is forced off). The medium fills the whole bounding box, so don't overlap it with geometry it should render behind. Defaults to `false`.
- `visible`: A boolean. When false the volume is skipped each frame. Defaults to `true`.
