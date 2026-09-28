<!-- Auto-generated - do not edit. -->

# KeyBinding

Maps a keyboard key to an action.

When the bound key is pressed, the action fires once per press (like a
[HitRegion](HitRegion.md) click). Bindings only run while the cursor is free:
they're inactive in worlds that capture the cursor for camera control.
While a [TextInput](TextInput.md) has keyboard focus, bindings are suspended
so typing cannot trigger actions; a [Screen](Screen.md)'s `toggle_key` stays
live.

InputKey names are case-sensitive canonical names (e.g. `"Escape"`, `"Space"`,
`"Enter"`).

## Parameters

- `key`: A string. The key name to bind (e.g. `"Escape"`). Defaults to `""`.
- `action`: A string or single-key object (see [Action](Action.md)). What a press of the key fires. Unset fires nothing.
- `screen`: A string. [Screen](Screen.md) this binding is scoped to: the binding only fires while that screen is on top of the stack. Unset, the binding is global.
