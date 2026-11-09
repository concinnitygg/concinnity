# concinnity-runtime

The std runtime for [Concinnity](https://crates.io/crates/concinnity): what
runs a compiled world on an operating system.

`concinnity-engine` decides everything a frame does; this crate drives it. It
holds the run loop and the host entry points, the graphics driver that builds
a `Box<dyn RenderBackend>` from `concinnity-device` and submits each frame
(never naming a concrete backend), and the adapters core's systems are lent:
the kira audio output, gilrs gamepads, crash capture, the allocator, the
macOS and Windows run loops, and the file stores behind saves and settings.

## Constraints

- Runtime only: depends on `concinnity-engine`, `concinnity-host`, and
  `concinnity-device`. No `concinnity-cook`, no importers, no image
  decoders; a shipped game plays compiled blobs and never sees authored
  input.
- No decisions: anything that would compile without std once time and I/O are
  passed in belongs in `concinnity-engine`.
- The dev tooling (`concinnity-dev`) drives this crate's `Runtime` and
  renderer through the public API; internals stay `pub(crate)` unless the
  editor specifically needs them.

Most users want the [`concinnity`](https://crates.io/crates/concinnity)
facade crate rather than this one.
