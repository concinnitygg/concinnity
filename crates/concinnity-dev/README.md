# concinnity-dev

The [Concinnity](https://crates.io/crates/concinnity) dev tooling library:
world authoring, the in-engine editor, the debug server, docs generation,
and packaging.

Everything the `concinnity` binary does, minus the argv it does it from:
the world authoring and in-memory build code, the implementation behind
each CLI subcommand, the asset-reference generator, bundle packaging and
export, the in-engine editor HUD with save-back, and the localhost debug
server. The binary itself is just a clap command tree that dispatches into
this crate, which is what lets the same entry points serve an out-of-tree
host with no argv at all.

## Constraints

- Development only: this crate (and its dependency on `concinnity-cook`)
  never appears in a shipped game's graph. The `concinnity` facade gates it
  behind the `editor` feature.

Most users want the [`concinnity`](https://crates.io/crates/concinnity)
facade crate rather than this one.
