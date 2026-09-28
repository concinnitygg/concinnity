<!-- Auto-generated - do not edit. -->

# Action

What a [HitRegion](HitRegion.md) click or a [KeyBinding](KeyBinding.md) press does.

An action that names nothing is written as its name; one that names
something is an object with the action as its only key, e.g.
`{"show": "pause"}`.

## Values

- `"quit"`: Stop the application.
- `{"scene": "<Scene name>"}`: Change to the named [Scene](Scene.md), dismissing every open screen.
- `{"show": "<Screen name>"}`: Show the named [Screen](Screen.md), replacing the top of the stack.
- `{"push": "<Screen name>"}`: Open the named [Screen](Screen.md) on top of what is showing.
- `{"toggle": "<Screen name>"}`: Close the named [Screen](Screen.md) if it is on top, open it otherwise.
- `"hide"`: Close the top [Screen](Screen.md).
- `{"story": <StoryCommand>}`: Drive the story system. See [StoryCommand](StoryCommand.md).
- `{"group_toggle": <integer>}`: Expand or collapse a settings-screen group by index. Emitted by generated settings menus.
- `{"setting": {...}}`: Operate a settings row: `{"key": "<setting>", "verb": "<verb>"}`. Emitted by generated settings menus.
