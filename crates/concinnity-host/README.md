# concinnity-host

Host services for the [Concinnity](https://crates.io/crates/concinnity)
engine: what the engine needs from the machine it runs on.

Two charters that share nothing but this crate:

- **`store`**: the project's on-disk state tree and the reads over it.
- **`thread`**: the worker pool and the per-thread interner.

They stay separate all the way down: nothing in `store` reaches into
`thread`, and nothing in `thread` names a path.

Most users want the [`concinnity`](https://crates.io/crates/concinnity)
facade crate rather than this one.
