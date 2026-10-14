<!-- Auto-generated - do not edit. -->

# GraphicsConfig

The world's graphics settings, one section per feature. One per world; a
world that draws gets an empty one when it declares none.

Every field of every section is optional. A field left unset follows the
graphics quality preset: under `Auto` the detected GPU picks a tier, and the
tier caps the quality-governed fields (shadow resolution and reach, texture
filtering, ambient occlusion, reflections, the screen-space bounce,
anti-aliasing and the upscaler's render scale) below the section defaults.
A field set here replaces the default but stays under the same cap on
weaker hardware, and a choice the player makes in the settings menu
replaces both.

```json
["GraphicsConfig",{"shadows":{"map_size":2048,"distance":120},"anti_aliasing":{"mode":"fxaa"},"background":{"color":[0.1,0.1,0.15,1.0]}}]
```

## Parameters

- `frame_pacing`: A [FramePacing](FramePacing.md) object. Frames in flight, display sync and the frame-rate cap.
- `shadows`: A [Shadows](Shadows.md) object. Cascaded shadow maps for directional lights.
- `texture_filtering`: A [TextureFiltering](TextureFiltering.md) object. Surface texture filtering.
- `background`: A [Background](Background.md) object. The color shown where nothing draws.
- `ambient_occlusion`: A [AmbientOcclusion](AmbientOcclusion.md) object. Screen-space ambient occlusion.
- `reflections`: A [Reflections](Reflections.md) object. Screen-space and ray-traced reflections.
- `indirect_lighting`: A [IndirectLighting](IndirectLighting.md) object. The ambient term and the screen-space bounce.
- `anti_aliasing`: A [AntiAliasing](AntiAliasing.md) object. Anti-aliasing and temporal upscaling.
- `exposure`: A [Exposure](Exposure.md) object. Fixed and automatic exposure.
- `bloom`: A [Bloom](Bloom.md) object. Bloom.
- `vignette`: A [Vignette](Vignette.md) object. Corner darkening.
- `display`: A [Display](Display.md) object. HDR and PQ output.
- `culling`: A [Culling](Culling.md) object. Visibility culling.
