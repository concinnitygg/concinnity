# concinnity-dev

The [Concinnity](https://crates.io/crates/concinnity) dev tooling library:
world authoring, the in-engine editor, the debug server, docs generation,
and packaging.

Everything the `concinnity` binary does, minus the argv it does it from:
the world authoring and in-memory build code, the implementation behind
each CLI subcommand, the asset-reference generator, bundle packaging and
export, the in-engine editor HUD with save-back, and the localhost debug
server. The binary itself is just a clap command tree that dispatches into
this crate.

## Constraints

- Development only: a shipped game never runs this crate's code. The
  `concinnity` facade pulls it in through its default `editor` feature,
  which a build that wants a smaller graph turns off.
