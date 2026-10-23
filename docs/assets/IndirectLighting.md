<!-- Auto-generated - do not edit. -->

# IndirectLighting

Indirect (ambient) lighting: how strong the environment's ambient term is,
and whether a screen-space bounce lets nearby lit surfaces bleed color onto
one another. The `indirect_lighting` section of
[`GraphicsConfig`](GraphicsConfig.md); a field left unset takes the default
below.

The screen-space bounce is on by default, which is the look on capable
hardware: the `Auto` graphics quality preset clamps it back to `ibl` below
the high tier.

## Parameters

- `technique`: A string (see [IndirectTechnique](IndirectTechnique.md)). Indirect-diffuse lighting source. `ibl` uses the environment map's ambient alone. `ssgi` (default) adds light bounced off nearby surfaces on screen, tinted by the surface it lands on, so lit surfaces bleed color onto one another; where the bounce finds a surface in the way, the environment ambient from that direction is held back, and off-screen and open-sky directions keep it. Applies live, rebuilding the passes it gates.
- `ambient_intensity`: A float. Multiplier on the indirect (ambient / IBL) lighting term. 1.0 (default) leaves the environment-derived ambient at its physical level. Raising it lifts fill light in areas the directional light cannot reach (shadowed facades, alleys) without brightening directly lit surfaces, which the sun already dominates. Scales the diffuse and specular IBL together, so reflections stay consistent with the brighter ambient. Useful for high-contrast exterior scenes where a strong sun would otherwise crush shadows to black. Clamped to `[0, 16]`. Applies live.
- `ssgi_intensity`: A float. Indirect-bounce strength. 1.0 (default) is the physical level: a surface receives the light its neighbors reflect, times its own color. Lower values weaken the bounce; 0 turns it off. Only matters when `technique` is `ssgi`. Clamped to `[0, 1]`. Applies live.
- `ssgi_max_distance`: A float. How far the indirect-light rays reach, in world units. A near-field effect, so it defaults well below the reflection reach. Only matters when `technique` is `ssgi`. Clamped to `[0.5, 100]`. Applies live.
- `ssgi_resolution`: A string (see [PassResolution](PassResolution.md)). Internal resolution of the SSGI trace. `half` (default) trades a little sharpness for a large performance saving; `full` is native; `quarter` is the cheapest. Only matters when `technique` is `ssgi`. Applies live, rebuilding the passes it gates.
- `ssgi_rays`: An integer. Hemisphere rays traced per pixel each frame. The indirect light is accumulated over frames, so more rays settle it faster after the camera reveals new surfaces or the lighting changes, at a higher cost. Only matters when `technique` is `ssgi`. Defaults to `1`. Clamped to `[1, 4]`. Applies live, rebuilding the passes it gates.
