# concinnity-cook

The build side of the [Concinnity](https://crates.io/crates/concinnity)
engine: the authored world model, validation, and the asset cook pipeline
that bakes a world into the binary blobs the runtime reads.

The stages read in order: `authoring` is the authored model and the type
vocabulary it is written in, `build_only` expands the types that never
reach a blob, `check` validates the expanded world, and the compile path
turns what is left into payloads (`import` reads artist-supplied source
files, `codec` decodes the container and image formats they carry,
`compile` produces each asset's payload, and `pipeline`/`blob`/`cache`
drive the run).

## Constraints

- Build-side only: the runtime tier (`core`/`device`/`engine`/`host`) must
  never depend on this crate. A shipped game plays compiled blobs and never
  sees authored input.
- Keeps the build-only dependencies (FBX/glTF importers, image and audio
  decoders, hashing) out of the runtime foundation.
- Importers parse artist-supplied files, so a panic here is a crash on a
  malformed asset rather than a bug; errors are reported, not panicked.

## Terminology

- **World**: the authored `world.jsonl`, a list of typed asset entries.
- **Expansion**: build-time rewriting that resolves build-only types into
  the runtime vocabulary.
- **Blob (`.cnb`)**: the compiled binary container the runtime loads.

Most users want the [`concinnity`](https://crates.io/crates/concinnity)
facade crate rather than this one.
