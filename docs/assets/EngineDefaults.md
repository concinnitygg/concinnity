<!-- Auto-generated - do not edit. -->

# EngineDefaults

Opts a world out of individual engine-injected defaults.

A world is completed at build time with standard assets it does not declare
itself: the [DebugHud](DebugHud.md) with its chip [TextLabel](TextLabel.md)s
and font, the [StatHud](StatHud.md) and its chips when the world declares a
[MainMenu](MainMenu.md), the [PhysicsConfig](PhysicsConfig.md) a world with
physics content simulates on, and, when an
[EnvironmentMap](EnvironmentMap.md) is present, the sky mesh that displays
it. Declaring the same asset yourself replaces the injected one; declaring
`EngineDefaults` with a flag set to `false` removes it entirely.

The build records every injected asset in `world-lock.json`; copy an entry
from there (or from `cn explain <name>`) into `world.jsonl` to override it.

## Parameters

- `hud`: A boolean. Inject the [StatHud](StatHud.md) with its chip labels and font when the world declares a [MainMenu](MainMenu.md) but no `StatHud`. Defaults to `true`.
- `debug_hud`: A boolean. Inject the [DebugHud](DebugHud.md) with its chip labels when the world declares no `DebugHud`. Defaults to `true`.
- `sky`: A boolean. Inject the sky mesh (a skybox [ProceduralMesh](ProceduralMesh.md), [Material](Material.md), and [Prop](Prop.md)) when the world has an [EnvironmentMap](EnvironmentMap.md) but no skybox mesh. Disable to use an `EnvironmentMap` for image-based lighting only, with the background left to `clear_color` or your own geometry. Defaults to `true`.
- `story_pause_menu`: A boolean. Inject an Escape-toggled pause [MainMenu](MainMenu.md) when the world plays a [Story](Story.md) but declares no `MainMenu`: Resume, Save, Load, a trimmed Settings screen, and Quit to the story's title. Disable to leave a story with no pause menu, or declare your own `MainMenu` to replace it. Defaults to `true`.
- `loading_overlay`: A boolean. Inject the [LoadingOverlay](LoadingOverlay.md) with its screen, backdrop, progress bar, and label when the world declares [Scene](Scene.md)s and a [StreamingConfig](StreamingConfig.md) but no `LoadingOverlay`. Disable to jump between scenes with no loading screen while their content streams in. Defaults to `true`.
- `physics_config`: A boolean. Inject a [PhysicsConfig](PhysicsConfig.md) with the engine's own values when the world has physics content -- a [RigidBody](RigidBody.md), a [PropBody](PropBody.md), a [TriggerVolume](TriggerVolume.md), or a [SkinnedMesh](SkinnedMesh.md) with a `capsule` -- but declares no `PhysicsConfig`. Physics runs on those values either way; the injected asset is what makes them visible in `world-lock.json` and editable, `spawn_headroom` above all. Disable to leave them implicit. Defaults to `true`.
