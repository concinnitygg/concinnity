<!-- Auto-generated - do not edit. -->

# Screen

A named full-screen layer of UI drawn over the world: a pause menu, a
settings page, a console, a score overlay.

UI elements ([Sprite](Sprite.md), [TextLabel](TextLabel.md),
[TextInput](TextInput.md), [HitRegion](HitRegion.md)) belong to a screen by
name prefix `<screen_name>_*`, mirroring the [Scene](Scene.md) →
[Prop](Prop.md) convention. Active screens form a stack; each is shown /
hidden via [HitRegion](HitRegion.md) or [KeyBinding](KeyBinding.md) actions:
- `screen:show:<name>` replaces the top of the stack (menu navigation)
- `screen:push:<name>` opens on top of what is already showing
- `screen:hide` closes the top screen, revealing what was beneath
- `screen:toggle:<name>` closes the screen if it is on top, opens it otherwise

Screens draw in stack order (later on top); `layer` orders a screen
against the always-on HUD and other screens independent of stack position.
While any active screen has `pauses_world` set, the world freezes exactly
as today's pause menu does. A `toggle_key` opens and closes the screen from
anywhere. `focus` names a [TextInput](TextInput.md) that receives keyboard
focus whenever the screen reaches the top of the stack. Worlds that need no
menus simply declare no screens.

## Parameters

- `initial`: A boolean. When true, this screen is shown as soon as the world loads. Defaults to `false`.
- `fade_in_secs`: A float. Seconds to fade the screen in when it's shown. 0 shows it instantly. Defaults to `0.0`.
- `toggle_key`: A string. InputKey that toggles this screen open / closed from anywhere, by the same canonical key names a [KeyBinding](KeyBinding.md) uses (e.g. "Escape", "Backtick"). Empty leaves the screen action-driven only.
- `input`: A string (see [ScreenInput](ScreenInput.md)). Input policy while the screen is active.
- `pauses_world`: A boolean. When true (the default), the world pauses beneath this screen while it is active: gameplay input, physics, and animation freeze.
- `focus`: A string. [TextInput](TextInput.md) that receives keyboard focus whenever this screen reaches the top of the stack. Optional.
- `layer`: An integer. Draw-order bias against the always-on HUD and other screens. Screens default above the HUD in stack order; a negative layer draws beneath the HUD, a higher layer stays above later-pushed screens.
