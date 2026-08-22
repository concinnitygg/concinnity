<!-- Auto-generated - do not edit. -->

# Window

Declares the application window.

```json
{
  "name": "main_window",
  "type": "Window",
  "args": {
    "title": "Game",
    "width": 1280,
    "height": 720,
    "mode": "windowed",
    "resizable": true,
    "title_bar": true
  }
}
```

## Parameters

- `title`: A string. Window title shown in the title bar. Defaults to `"Concinnity"`.
- `width`: An integer. Initial window width in pixels. Defaults to `1024`.
- `height`: An integer. Initial window height in pixels. Defaults to `768`.
- `mode`: A string (see [WindowMode](WindowMode.md)). How the window is displayed.
- `resizable`: A boolean. Whether the user can resize the window. Defaults to `false`.
- `title_bar`: A boolean. Whether the title bar is drawn, letting content fill the frame when it is not. Only applies to `windowed` mode: `borderless` has no title bar by definition, and `fullscreen` leaves the chrome to the OS. Platforms differ in what survives the title bar. macOS keeps the close / minimize / zoom buttons floating over the content, so the window stays movable and closable. Windows and Linux draw their controls *in* the title bar, so turning it off there also removes them: the window can still be resized from its border, but offers no close button and cannot be dragged. Defaults to `true`.
