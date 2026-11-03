<!-- Auto-generated - do not edit. -->

# Graphics settings

These are the settings a player can change while the app runs, from the settings
menu or the stored settings file, that change how the scene is rendered or shown.
Each heading is the key a settings row's action names. A setting that overrides
an asset field starts from the world's authored value, and leaving it unchanged
keeps that value.

## Quality presets

`graphics_quality` picks a preset: `Auto`, a named tier (Low, Medium, High,
Ultra), or `Custom`. A tier is a ceiling over the world's authored look: it lowers
the settings below where a world asks for more than the tier allows, and never
turns a feature on. Changing any setting the preset governs switches the preset
to `Custom`, which keeps the ceiling of the preset it left, so only that setting
changes. Picking `Custom` itself keeps every current value.

`Auto` picks the tier from the GPU, each launch:

| GPU | Auto picks |
| --- | --- |
| An integrated or low-power GPU | Low |
| A discrete GPU that reports no VRAM | Medium |
| A discrete GPU with under 6 GB of VRAM, or a base Apple M-series chip | Medium |
| A discrete GPU with 6 GB to 12 GB of VRAM, an Apple Pro chip, an M1 or M2 Max, or an M1 Ultra | High |
| A discrete GPU with 12 GB of VRAM or more, an Apple Max from M3 on, or an Apple Ultra from M2 on | Ultra |

The finest value each tier allows:

| Setting | Low | Medium | High | Ultra |
| --- | --- | --- | --- | --- |
| `render_scale` | Performance | Balanced | Quality | Quality |
| `aa_mode` | FXAA | TAA | TAA | TAA |
| `ssgi_resolution` | Quarter | Quarter | Quarter | Full |
| `ssgi_rays` | 1 | 1 | 1 | 2 |
| `rt_reflection_resolution` | Quarter | Half | Half | Full |
| `reflection_blur_resolution` | Quarter | Half | Half | Full |
| `shadow_map_size` | 1024 | 2048 | 4096 | No cap |
| `shadow_update` | Hybrid | Hybrid | Hybrid | Every Frame |
| `shadow_distance` | 40 m | 80 m | 160 m | No cap |
| `shadow_cascades` | 2 | 3 | 4 | 4 |
| `anisotropy` | 4x | 8x | 16x | 16x |
| `temporal_upscaling` | On above 2560x1440 | On above 2560x1440 | On above 2560x1440 | Native |
| `ssao` | Off | On | On | On |
| `ssr` | Off | Off | On | On |
| `ray_traced_reflections` | Off | Off | Off | On |
| `rt_reflection_shadows` | Off | Off | On | On |
| `ssgi` | Off | Off | On | On |
| `auto_exposure` | On | On | On | On |

## Display

### `graphics_api`

The graphics API a launch renders with.

- **Options:** Metal, DirectX 12, Vulkan, where the build compiled them
- **Available with:** A build with more than one graphics API.

### `vsync`

Vertical sync.

- **Options:** Off, On
- **Overrides:** [`FramePacing.vsync`](../assets/FramePacing.md). A boolean. Cap the frame rate to the display refresh (vsync). Defaults to `false`: presentation is not tied to the refresh (DirectX presents with tearing allowed, Vulkan uses a mailbox present mode) and `fps_cap` alone bounds the frame rate. Set to `true` to lock presentation to the monitor refresh, eliminating tearing and the wasted frames that never reach the screen. Applies live.

### `window_mode`

Windowed, borderless, or fullscreen.

- **Options:** Windowed, Borderless, Fullscreen
- **Overrides:** [`Window.mode`](../assets/Window.md). A string (see [WindowMode](../assets/WindowMode.md)). How the window is displayed. Defaults to `"windowed"`.

### `resolution`

The display mode, enumerated from the display at runtime.

- **Options:** The display's own modes
- **Available with:** Fullscreen window mode.

### `fps_cap`

The frame-rate cap.

- **Options:** Unlimited, 30, 60, 120, 144, 240, 300, 360
- **Overrides:** [`FramePacing.fps_cap`](../assets/FramePacing.md). An integer. Cap the frame rate to this many frames per second, `300` by default. `0` leaves the loop uncapped, which is what GPU cost measurement wants: a capped GPU idles between frames and lowers its clocks. The cap is a CPU-side frame pacer, so it composes with `vsync`: the more restrictive of the two wins. Lower caps limit heat, fan noise, and power draw. Applies live.

### `frames_in_flight`

The frames buffered ahead of presentation.

- **Options:** 1, 2, 3
- **Overrides:** [`FramePacing.frames_in_flight`](../assets/FramePacing.md). An integer. Preferred number of frames in flight. Higher can smooth pacing at the cost of input latency. Defaults to `2`. Clamped to `[1, 3]`. Applies at the next launch.

### `hdr_display`

HDR display output.

- **Options:** Off, On
- **Overrides:** [`Display.hdr`](../assets/Display.md). A boolean. HDR display output. On a capable display, emits extended-range HDR instead of the standard tonemapped output. Falls back to standard output when the display or platform doesn't support HDR. Auto-exposure meters toward perceptual middle gray while it is on. Defaults to `false`. Applies at the next launch.

### `hdr_pq`

The PQ transfer function for HDR output.

- **Options:** Off, On
- **Overrides:** [`Display.pq`](../assets/Display.md). A boolean. PQ (HDR10) output. When true, and `hdr` is on, and the display has HDR headroom, output is PQ-encoded for HDR10 panels. No effect when `hdr` is off. Defaults to `false`. Applies at the next launch.

### `fov`

The camera field of view.

- **Range:** 50° to 100°
- **Overrides:** [`Camera3D.fov_y_degrees`](../assets/Camera3D.md). A float. Vertical field-of-view in degrees. Defaults to `75.0`.

## Quality

### `graphics_quality`

The master graphics-quality preset.

- **Options:** Auto, Low, Medium, High, Ultra, Custom

### `render_scale`

The upscaling quality tier.

- **Options:** Quality, Balanced, Performance, Ultra
- **Overrides:** [`AntiAliasing.upscale_quality`](../assets/AntiAliasing.md). A string (see [UpscaleQuality](../assets/UpscaleQuality.md)). Render-scale preset for `upscaling`; each step progressively lowers the internal resolution. No effect when `upscaling` is off. Defaults to `"quality"`. Applies at the next launch.
- **Preset caps:** Low Performance, Medium Balanced, High Quality, Ultra Quality

### `upscale_backend`

The upscaler implementation.

- **Options:** Auto, FSR 3, DLSS, XeSS
- **Overrides:** [`AntiAliasing.upscaler`](../assets/AntiAliasing.md). A string (see [UpscalerBackend](../assets/UpscalerBackend.md)). Which upscaler `upscaling` uses. `auto` (default) picks the best available at runtime (DLSS on NVIDIA RTX, else XeSS, else FSR3); `fsr3` / `dlss` / `xess` request a specific one and fall back when it is unavailable on the current GPU or build. No effect when `upscaling` is off. Metal treats every value as `MetalFX`. Applies at the next launch.
- **Available with:** DirectX 12 or Vulkan. Metal always upscales through MetalFX.

### `dlss_preset`

The DLSS render preset.

- **Options:** Default, K, L, M
- **Available with:** An NVIDIA GPU, on DirectX 12 or Vulkan.

### `aa_mode`

The anti-aliasing mode.

- **Options:** Off, FXAA, TAA
- **Overrides:** [`AntiAliasing.mode`](../assets/AntiAliasing.md). A string (see [AaMode](../assets/AaMode.md)). Anti-aliasing mode. `fxaa` applies a cheap composite-pass edge filter; `taa` (default) adds a temporal pass that jitters the projection and accumulates detail across frames for the cleanest edges, at the cost of a velocity pre-pass and a history buffer; `off` disables edge smoothing. Applies live, rebuilding the passes it gates.
- **Preset caps:** Low FXAA, Medium TAA, High TAA, Ultra TAA

### `ssgi_resolution`

The SSGI trace resolution.

- **Options:** Full, Half, Quarter
- **Overrides:** [`IndirectLighting.ssgi_resolution`](../assets/IndirectLighting.md). A string (see [PassResolution](../assets/PassResolution.md)). Internal resolution of the SSGI trace. `full` (default) is native; `half` trades a little sharpness for a large performance saving; `quarter` is the cheapest. Only matters when `technique` is `ssgi`. Applies live, rebuilding the passes it gates.
- **Preset caps:** Low Quarter, Medium Quarter, High Quarter, Ultra Full

### `ssgi_rays`

The SSGI rays per pixel per frame.

- **Options:** 1, 2, 4
- **Overrides:** [`IndirectLighting.ssgi_rays`](../assets/IndirectLighting.md). An integer. Hemisphere rays traced per pixel each frame. The indirect light is accumulated over frames, so more rays settle it faster after the camera reveals new surfaces or the lighting changes, at a higher cost. Only matters when `technique` is `ssgi`. Defaults to `2`. Clamped to `[1, 4]`. Applies live, rebuilding the passes it gates.
- **Preset caps:** Low 1, Medium 1, High 1, Ultra 2

### `rt_reflection_resolution`

The ray-traced reflection trace resolution.

- **Options:** Full, Half, Quarter
- **Overrides:** [`Reflections.ray_traced_resolution`](../assets/Reflections.md). A string (see [PassResolution](../assets/PassResolution.md)). Internal resolution the ray-traced reflection rays are cast at. `full` (default) traces every pixel for the sharpest mirrors; `half` traces a quarter of the pixels and upsamples with a depth- and normal-aware filter; `quarter` is the cheapest. Only matters when `ray_traced` is on. Applies live, rebuilding the passes it gates.
- **Available with:** A GPU with hardware ray tracing.
- **Preset caps:** Low Quarter, Medium Half, High Half, Ultra Full

### `reflection_blur_resolution`

The reflection blur resolution.

- **Options:** Full, Half, Quarter
- **Overrides:** [`Reflections.blur_resolution`](../assets/Reflections.md). A string (see [PassResolution](../assets/PassResolution.md)). Internal resolution of the roughness-aware reflection blur. `full` (default) blurs at native resolution; `half` blurs at a quarter of the pixels for a large saving and bilinearly upsamples; `quarter` is the cheapest. Smooth mirror surfaces stay sharp at any setting (the composite keeps the sharp reflection for low roughness). Applies live, rebuilding the passes it gates.
- **Preset caps:** Low Quarter, Medium Half, High Half, Ultra Full

### `shadow_map_size`

The shadow-map cascade resolution, or shadows off.

- **Options:** Off, 1024, 2048, 4096
- **Overrides:** [`Shadows.map_size`](../assets/Shadows.md). An integer. Shadow map resolution in texels. `4096` by default, capped by the quality preset down to 1024 on the lowest tier. Set to 0 to disable shadows. Applies at the next launch.
- **Preset caps:** Low 1024, Medium 2048, High 4096, Ultra No cap

### `shadow_update`

The shadow re-render cadence.

- **Options:** Every Frame, Hybrid
- **Overrides:** [`Shadows.update`](../assets/Shadows.md). A string (see [ShadowUpdate](../assets/ShadowUpdate.md)). How often shadow cascades are re-rendered. `every_frame` (default) refreshes them all every frame; `hybrid` amortizes the far cascades across frames. Only the top quality tier permits `every_frame`, so everything below it runs `hybrid`. Applies live.
- **Preset caps:** Low Hybrid, Medium Hybrid, High Hybrid, Ultra Every Frame

### `shadow_distance`

The distance the shadow cascades cover.

- **Options:** 40 m, 80 m, 160 m, 320 m
- **Overrides:** [`Shadows.distance`](../assets/Shadows.md). An integer. How far from the camera shadows are cast, in world units (e.g. 80). The cascades cover from the near plane out to this distance; a larger value shadows more of the scene but spreads the same shadow-map resolution over more area (softer, blockier shadows). Capped at the camera far plane. Defaults to `80`. Applies live.
- **Preset caps:** Low 40 m, Medium 80 m, High 160 m, Ultra No cap

### `shadow_cascades`

The shadow cascade count.

- **Options:** 2, 3, 4
- **Overrides:** [`Shadows.cascades`](../assets/Shadows.md). An integer. Number of shadow cascades (`4` is the default and the maximum). More cascades keep distant shadows sharper by splitting the view range into finer slices, at the cost of an extra shadow-map render per cascade; fewer is cheaper but blockier far from the camera. The slice count covers the same `distance` regardless. Clamped to `[1, 4]`. Applies live.
- **Preset caps:** Low 2, Medium 3, High 4, Ultra 4

### `anisotropy`

The scene sampler's anisotropic filtering degree.

- **Options:** Off, 2x, 4x, 8x, 16x
- **Overrides:** [`TextureFiltering.anisotropy`](../assets/TextureFiltering.md). An integer. Maximum anisotropic-filtering degree, e.g. 8. Higher keeps textures viewed at a grazing angle (floors, walls receding into the distance) sharp instead of blurring along the minor axis, at a small sampling cost. `1` disables anisotropy (plain trilinear). `16` by default, capped by the quality preset down to 4 on the lowest tier. Clamped to `[1, 16]`. Applies at the next launch.
- **Preset caps:** Low 4x, Medium 8x, High 16x, Ultra 16x

### `texture_quality`

The texture streaming pool and upload budget tier.

- **Options:** Low, Medium, High, Ultra

### `temporal_upscaling`

Temporal upscaling.

- **Options:** Off, On
- **Overrides:** [`AntiAliasing.upscaling`](../assets/AntiAliasing.md). A boolean. Temporal upscaling. Renders the 3D scene at a lower resolution (set by `upscale_quality`) and reconstructs a full-resolution image, trading some sharpness for performance. Replaces TAA while on. Defaults to `false`. Applies at the next launch.
- **Preset caps:** Low On above 2560x1440, Medium On above 2560x1440, High On above 2560x1440, Ultra Native

### `occlusion_two_pass`

Two-pass occlusion culling.

- **Options:** Off, On
- **Overrides:** [`Culling.two_pass_occlusion`](../assets/Culling.md). A boolean. Two-pass occlusion culling. Reduces objects popping in a frame late when they're revealed by camera or occluder motion, at the cost of extra culling work each frame. Needs the bindless GPU-cull path. Defaults to `true`. Applies at the next launch.

### `ssao`

Screen-space ambient occlusion.

- **Options:** Off, On
- **Overrides:** [`AmbientOcclusion.enabled`](../assets/AmbientOcclusion.md). A boolean. Whether the occlusion pass runs. Defaults to `true`. Applies live, rebuilding the passes it gates.
- **Preset caps:** Low Off, Medium On, High On, Ultra On

### `ssr`

Screen-space reflections.

- **Options:** Off, On
- **Overrides:** [`Reflections.screen_space`](../assets/Reflections.md). A boolean. Screen-space reflections: mix reflected scene color over glossy surfaces, from what is already on screen. Defaults to `true`. Applies live, rebuilding the passes it gates.
- **Available with:** Runs only where ray-traced reflections are off or unsupported; the row is grayed while they run.
- **Preset caps:** Low Off, Medium Off, High On, Ultra On

### `ray_traced_reflections`

Hardware ray-traced reflections.

- **Options:** Off, On
- **Overrides:** [`Reflections.ray_traced`](../assets/Reflections.md). A boolean. Hardware ray-traced reflections. When the GPU supports ray tracing, traces real reflection rays so off-screen geometry still appears. Takes precedence over `screen_space`, falling back to it where ray tracing isn't available. Defaults to `true`. Applies live, rebuilding the passes it gates.
- **Available with:** A GPU with hardware ray tracing.
- **Preset caps:** Low Off, Medium Off, High Off, Ultra On

### `rt_reflection_shadows`

Sun shadows inside ray-traced reflections.

- **Options:** Off, On
- **Overrides:** [`Reflections.ray_traced_shadows`](../assets/Reflections.md). A boolean. Whether surfaces seen in a ray-traced reflection are shadowed from the sun. Each reflected hit then casts a second ray toward the sun, which roughly doubles the trace cost; off lights every reflected surface as if the sun reached it. Only matters when `ray_traced` is on. Defaults to `true`. Applies live, rebuilding the passes it gates.
- **Available with:** A GPU with hardware ray tracing.
- **Preset caps:** Low Off, Medium Off, High On, Ultra On

### `ssgi`

Screen-space global illumination.

- **Options:** Off, On
- **Overrides:** [`IndirectLighting.technique`](../assets/IndirectLighting.md). A string (see [IndirectTechnique](../assets/IndirectTechnique.md)). Indirect-diffuse lighting source. `ibl` uses the environment map's ambient alone. `ssgi` (default) adds light bounced off nearby surfaces on screen, tinted by the surface it lands on, so lit surfaces bleed color onto one another; where the bounce finds a surface in the way, the environment ambient from that direction is held back, and off-screen and open-sky directions keep it. Applies live, rebuilding the passes it gates.
- **Preset caps:** Low Off, Medium Off, High On, Ultra On

### `auto_exposure`

Automatic exposure adaptation.

- **Options:** Off, On
- **Overrides:** [`Exposure.auto`](../assets/Exposure.md). A boolean. Auto-exposure. Adapts exposure each frame toward a balanced mid-tone; `ev` then acts as an additive bias in stops on top of the adapted value. Defaults to `false`. Applies live, rebuilding the passes it gates.
- **Preset caps:** Low On, Medium On, High On, Ultra On

## Image

### `exposure`

Exposure compensation in EV.

- **Range:** -3.0 EV to +3.0 EV
- **Overrides:** [`Exposure.ev`](../assets/Exposure.md). A float. Exposure offset in photographic stops. Each +1 doubles scene brightness before bloom and tonemapping; 0 is neutral. Defaults to `0.0`. Clamped to `[-24, 24]`. Applies live.

### `bloom_intensity`

Bloom strength.

- **Range:** 0.00 to 2.00
- **Overrides:** [`Bloom.intensity`](../assets/Bloom.md). A float. Additive bloom contribution. 0 skips bloom entirely. Defaults to `0.6`. At least `0`. Applies live.

### `bloom_threshold`

The luminance bloom starts at.

- **Range:** 0.00 to 4.00
- **Overrides:** [`Bloom.threshold`](../assets/Bloom.md). A float. Brightness threshold. Pixels brighter than this contribute fully; pixels within `knee` below it ramp in softly. Defaults to `1.0`. At least `0`. Applies live.

### `bloom_knee`

The softness of the bloom threshold.

- **Range:** 0.00 to 1.00
- **Overrides:** [`Bloom.knee`](../assets/Bloom.md). A float. Width of the soft knee just below `threshold`. Defaults to `0.5`. At least `0`. Applies live.

### `vignette`

Vignette strength.

- **Range:** 0% to 100%
- **Overrides:** [`Vignette.strength`](../assets/Vignette.md). A float. Vignette strength in `[0, 1]`. 0 disables the corner darkening. Defaults to `0.0`. Clamped to `[0, 1]`. Applies live.

### `lut_strength`

Color-grading LUT strength.

- **Range:** 0% to 100%
- **Overrides:** [`ColorLut.strength`](../assets/ColorLut.md). A float. Blend in `[0, 1]`: mixes the graded color over the ungraded one by this amount. Defaults to `1.0`. Clamped to `[0, 1]`. Applies live.

### `ambient_intensity`

Image-based ambient light scale.

- **Range:** 0.00 to 4.00
- **Overrides:** [`IndirectLighting.ambient_intensity`](../assets/IndirectLighting.md). A float. Multiplier on the indirect (ambient / IBL) lighting term. 1.0 (default) leaves the environment-derived ambient at its physical level. Raising it lifts fill light in areas the directional light cannot reach (shadowed facades, alleys) without brightening directly lit surfaces, which the sun already dominates. Scales the diffuse and specular IBL together, so reflections stay consistent with the brighter ambient. Useful for high-contrast exterior scenes where a strong sun would otherwise crush shadows to black. Clamped to `[0, 16]`. Applies live.

### `ssao_radius`

The SSAO sample radius.

- **Range:** 0.1 m to 2.0 m
- **Overrides:** [`AmbientOcclusion.radius`](../assets/AmbientOcclusion.md). A float. How far the search reaches for occluders, in world units. Larger values pick up broader, softer occlusion. Defaults to `0.5`. At least `0.001`. Applies live.

### `ssao_intensity`

SSAO strength.

- **Range:** 0.00 to 4.00
- **Overrides:** [`AmbientOcclusion.intensity`](../assets/AmbientOcclusion.md). A float. Occlusion strength. 1.0 is the natural amount; higher values exaggerate the contact darkening. Defaults to `1.0`. Clamped to `[0, 4]`. Applies live.

### `ssr_intensity`

SSR strength.

- **Range:** 0.00 to 1.00
- **Overrides:** [`Reflections.intensity`](../assets/Reflections.md). A float. Reflection blend strength. Scales the Fresnel-weighted reflection mixed over the base shading. Defaults to `0.7`. Clamped to `[0, 1]`. Applies live.

### `ssr_max_distance`

The SSR ray length.

- **Range:** 1.0 m to 200.0 m
- **Overrides:** [`Reflections.max_distance`](../assets/Reflections.md). A float. How far a reflection reaches, in world units. Longer reaches catch more distant reflections, more coarsely. Defaults to `40.0`. Clamped to `[1, 1000]`. Applies live.

### `ssgi_intensity`

SSGI strength.

- **Range:** 0.00 to 4.00
- **Overrides:** [`IndirectLighting.ssgi_intensity`](../assets/IndirectLighting.md). A float. Indirect-bounce strength. 1.0 (default) is the physical level: a surface receives the light its neighbors reflect, times its own color. Lower values weaken the bounce; 0 turns it off. Only matters when `technique` is `ssgi`. Clamped to `[0, 1]`. Applies live.

### `ssgi_max_distance`

The SSGI ray length.

- **Range:** 0.5 m to 40.0 m
- **Overrides:** [`IndirectLighting.ssgi_max_distance`](../assets/IndirectLighting.md). A float. How far the indirect-light rays reach, in world units. A near-field effect, so it defaults well below the reflection reach. Only matters when `technique` is `ssgi`. Clamped to `[0.5, 100]`. Applies live.

### `auto_exposure_min_ev`

The darkest exposure auto exposure adapts to.

- **Range:** -24.0 EV to +24.0 EV
- **Overrides:** [`Exposure.auto_min_ev`](../assets/Exposure.md). A float. Lower bound on the adapted exposure (EV). The `ev` bias is applied before this clamp. Defaults to `-8.0`. Clamped to `[-24, 24]`. Applies live.

### `auto_exposure_max_ev`

The brightest exposure auto exposure adapts to.

- **Range:** -24.0 EV to +24.0 EV
- **Overrides:** [`Exposure.auto_max_ev`](../assets/Exposure.md). A float. Upper bound on the adapted exposure (EV): how far a dark scene is brightened, at most 24. A night under a [`NightSky`](../assets/NightSky.md) adapts at around 19 to 20, so wants 24. Defaults to `8.0`. Clamped to `[-24, 24]`. Applies live.

### `auto_exposure_speed`

How fast auto exposure adapts.

- **Range:** 0.10 to 6.00
- **Overrides:** [`Exposure.auto_speed`](../assets/Exposure.md). A float. How quickly exposure chases a new target (per second). Higher converges faster but can pump under flickering content; 1-3 is comfortable. As the eye does, it adapts to a brighter view three times as fast as to a darker one. Defaults to `1.5`. Clamped to `[0.001, 20]`. Applies live.
