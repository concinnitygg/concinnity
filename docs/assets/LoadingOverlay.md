<!-- Auto-generated - do not edit. -->

# LoadingOverlay

Requests the scene-loading overlay: a full-window backdrop with a progress
bar, shown while a scene jump waits for its streamed content and faded out
once the destination scene is fully resident.

The overlay is assembled from ordinary UI elements the fields reference: a
[Screen](Screen.md) that hosts them, a backdrop [Sprite](Sprite.md) covering
the canvas, a progress-bar track and fill [Sprite](Sprite.md) pair, and a
[TextLabel](TextLabel.md) above the bar. Each frame the engine widens the
fill to the destination scene's load progress and rewrites the label with
the percentage; restyle any piece by declaring it yourself under the same
name.

While the overlay's screen is up the world pauses and its render is
skipped, exactly like an opaque menu, but streaming keeps running so the
load it reports can finish. Scenes whose content is already resident jump
without the overlay ever appearing.

Every rendering world that declares [Scene](Scene.md)s and a
[StreamingConfig](StreamingConfig.md) receives a `LoadingOverlay` and its
pieces at build time when it declares none, so the example below is only
needed to restyle them. Declare an [EngineDefaults](EngineDefaults.md) with
`"loading_overlay": false` to remove it from the build entirely.

## Parameters

- `screen`: A string. [Screen](Screen.md) the overlay shows while a scene loads. Its `pauses_world` (on by default) freezes the world beneath the overlay so the destination scene starts fresh when revealed.
- `backdrop`: A string. Backdrop [Sprite](Sprite.md) covering the canvas. Its tint alpha is animated to reveal the scene once loading completes; an opaque tint hides the still-loading world completely. Optional.
- `track`: A string. Progress-bar track [Sprite](Sprite.md); its width is the bar's full extent the fill is measured against. Optional.
- `fill`: A string. Progress-bar fill [Sprite](Sprite.md); the engine sets its width to the track width times the destination scene's load progress each frame. Optional.
- `label`: A string. [TextLabel](TextLabel.md) rewritten each frame with the load percentage. Optional.
