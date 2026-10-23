<!-- Auto-generated - do not edit. -->

# Screenshot

Saves the frame drawn at a moment of the run as a PNG image, in a run
that takes screenshots.

The moment is counted on the fixed simulation clock from when the world
starts, a benchmark's warm-up included. A camera track, a benchmark's
route and anything else driven by that clock is in the same pose at the
same moment on every machine, so a screenshot shows the same thing however
fast the machine draws. The frame saved is the first one drawn at or after
the moment. A moment the run never reaches is reported when the run ends.

`file` is a relative path under the run's screenshot directory. It cannot
leave that directory or name a directory: an absolute path, one with a
`..` part or a `:`, or one ending in `/` is refused, and so are two
screenshots naming the same file.

```json
["Screenshot",{"seconds":25.0,"file":"beach.png","hud":false}]
```

## Parameters

- `seconds`: A float. Seconds from the start of the run, on the simulation clock. Defaults to `0.0`.
- `file`: A string. The image's path under the screenshot directory. Defaults to `""`.
- `hud`: A boolean. Whether the HUD and other UI show in the image. Defaults to `true`.
