<!-- Auto-generated - do not edit. -->

# KeyBinding

Maps a keyboard key to an action string.

When the bound key is pressed, the action fires once per press (like a
[HitRegion](HitRegion.md) click). Bindings only run while the cursor is free:
they're inactive in worlds that capture the cursor for camera control.
While a [TextInput](TextInput.md) has keyboard focus, bindings are suspended
so typing cannot trigger actions; a [Screen](Screen.md)'s `toggle_key` stays
live.

The action vocabulary is the same as [HitRegion](HitRegion.md)'s:
- `"scene:<name>"`:         jump to the named [Scene](Scene.md)
- `"quit"`:                 stop the application
- `"screen:show:<name>"`:   show the named [Screen](Screen.md), replacing the top of the stack
- `"screen:push:<name>"`:   open the named [Screen](Screen.md) on top of what is showing
- `"screen:hide"`:          close the top [Screen](Screen.md)
- `"screen:toggle:<name>"`: toggle the named [Screen](Screen.md)

Key names are case-sensitive canonical names (e.g. `"Escape"`, `"Space"`,
`"Enter"`).

```jsonl
{"name":"esc_binding","type":"KeyBinding","args":{"key":"Escape","action":"screen:toggle:pause_menu"}}
```

## Parameters

- `key`: A string. The key name to bind (e.g. `"Escape"`).
- `action`: A string. The action to fire when the key is pressed.
- `screen`: A string. [Screen](Screen.md) this binding is scoped to: the binding only fires while that screen is on top of the stack. Unset, the binding is global.
