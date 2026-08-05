<!-- Auto-generated - do not edit. -->

# EnvironmentMap

A baked lighting environment built from an equirectangular source (or a
built-in generator). It provides the scene's ambient image-based lighting
(soft diffuse fill plus glossy reflections that follow surface roughness)
and the on-screen sky.

**Source formats:** a Radiance `.hdr`, or a panorama-sphere `.glb` /
`.gltf` -- the packaging where an environment image is painted on the
emissive channel of a sphere you stand inside. `cn add` recognises the
latter and produces an EnvironmentMap instead of scene geometry.

**Dynamic range:** a `.hdr` carries real radiance, so its sun can be
thousands of times brighter than the sky and bakes into a bright key light
with a hot specular highlight. A panorama inside a `.glb` is a display
image whose brightest value is white; it is read literally, with the sRGB
curve inverted and white landing at 1.0 radiance. That makes it an exact
backdrop and a soft, low-contrast fill light, never a key light. Raise
[PostProcessConfig](PostProcessConfig.md)'s `ambient_intensity` to lift the
level rather than expecting the bake to invent range the file lacks.

**`prefilter_face_size` note:** this controls both the reflection detail and
the on-screen sky sharpness. 512 is the default balance: 256 visibly
pixelates a 4K-source sky, 1024 sharpens it further at 4× the size.

**Built-in generators:** `sky` produces a procedural blue sky with a soft
sun, useful when no source file is available.

The sky mesh that displays the map (a skybox
[ProceduralMesh](ProceduralMesh.md) plus its [Material](Material.md) and
[Prop](Prop.md)) is injected at build time when the world declares no skybox
mesh of its own. Declare an [EngineDefaults](EngineDefaults.md) with
`"sky": false` to use the map for image-based lighting only, with the
background left to `clear_color` or your own geometry.

```jsonl
{"name":"env_studio","type":"EnvironmentMap","args":{"source":"assets/hdri/studio.hdr"}}
{"name":"env_outdoor","type":"EnvironmentMap","args":{"source":"assets/hdri/sky.hdr","prefilter_face_size":512}}
{"name":"env_galaxy","type":"EnvironmentMap","args":{"source":"assets/hdri/galaxy.glb"}}
{"name":"env_proc","type":"EnvironmentMap","args":{"generator":"sky"}}
```

## Parameters

- `source`: A string. Path to the source equirectangular panorama -- a Radiance `.hdr`, or a panorama-sphere `.glb` / `.gltf` -- relative to the project root. Mutually exclusive with `generator`.
- `generator`: A string. Built-in source name (e.g. "sky"). Mutually exclusive with `source`.
- `prefilter_face_size`: An integer. Face size of the reflection/sky cubemap, in pixels. Higher is sharper but larger. Defaults to `512`.
- `irradiance_face_size`: An integer. Face size of the diffuse ambient cubemap, in pixels. Defaults to `8`.
- `prefilter_samples`: An integer. Number of samples used to filter each reflection texel. Higher reduces noise at the cost of build time. Defaults to `1024`.
- `prefilter_clamp`: A float. Upper bound on how bright a single source texel may count while building the glossy reflection mips. A clear-sky HDR holds a few sun or sky texels thousands of times brighter than their surroundings; left unbounded they survive into the small (coarse) reflection mips as lone hot texels and smear across glossy floors as hard bright squares. This caps each sampled texel so that energy spreads smoothly across the reflection instead. It affects reflections only, never the on-screen sky. Set to `0` to disable (no cap); lower values clamp harder. Defaults to `12.0`.
