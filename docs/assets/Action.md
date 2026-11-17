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
- `"toggle_hud"`: Hide the [StatHud](StatHud.md) chips if they are showing, show them otherwise.
- `{"screenshot": <integer>}`: Save this many consecutive presented frames, starting with the next, as lossless PNG images in the run's screenshot folder: `screenshots` under the working directory, unless the app was launched with another (`--screenshots-dir`). A burst of several frames follows something that moves from frame to frame.
- `{"story": <StoryCommand>}`: Drive the story system. See [StoryCommand](StoryCommand.md).
- `{"set": {...}}`: Set the value a binding (`setting:`, `var:`, or `state:`) names: `{"binding": "state:mode", "value": 2}`.
- `{"flip": "<name>"}`: Turn the boolean a binding (`setting:`, `var:`, or `state:`) names over.
- `{"add": {...}}`: Add to the number a binding (`setting:`, `var:`, or `state:`) names: `{"binding": "var:score", "by": 1}`.
- `{"step": {...}}`: Step the setting or whole number a binding (`setting:`, `var:`, or `state:`) names: `{"binding": "setting:vsync", "by": -1}`.
- `{"emit": "<name>"}`: Tell the app's own systems, which read it as a `UiEvent`: a name, `"save"`, or a name and a value, `{"name": "pick", "value": 3}`.
- `{"setting": {...}}`: Operate a settings row: `{"key": "<setting>", "verb": "<verb>"}`. Emitted by generated settings menus.
