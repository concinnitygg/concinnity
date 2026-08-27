# Crates

The `concinnity` and `concinnity-run` binaries are targets of the root package,
over these libraries.

| Crate                  | Kind      |      no_std?       | Role                                                                                  |
| ---------------------- | --------- | :----------------: | ------------------------------------------------------------------------------------- |
| `concinnity-cli`       | lib       |                    | The `concinnity` CLI's command tree: build, run, add, export, debug.                  |
| `concinnity-editor`    | lib       |                    | In-engine world editor: live preview, draggable panels, hot-reload.                   |
| `concinnity-cook`      | lib       |                    | Asset cook pipeline that bakes an authored world into a blob.                         |
| `concinnity-world`     | lib       |                    | Authored world source, args schema, validation, and spec builders.                    |
| `concinnity-engine`    | lib       |                    | Runtime engine: ECS schedule, graphics/audio ([kira])/spawn, allocator.               |
| `concinnity-device`    | lib       |                    | GPU backends (Metal/Vulkan/DX) behind a device facade.                                |
| `concinnity-shader`    | lib       |                    | Build-time shader compilers for the backend being built.                              |
| `concinnity-host`      | lib       |                    | Host services: the on-disk state tree, the worker pool, the build-time name interner. |
| `concinnity-render`    | lib       | :white_check_mark: | GPU-free render preparation.                                                          |
| `concinnity-core`      | lib       | :white_check_mark: | Runtime vocabulary, the compute over it, and the headless sim drivers.                |
| `concinnity-asset`     | lib       | :white_check_mark: | User-facing asset schema (the single home for asset types).                           |
| `concinnity-physics`   | lib       | :white_check_mark: | Rigid-body simulation: bodies, contacts, solver, queries.                             |
| `concinnity-memory`    | lib       | :white_check_mark: | Allocation layer: tracking allocator, tagged budgets, arenas, pools.                  |
| `concinnity-toolchain` | build-dep |                    | Build-script support: cfgs, SDKs, source hashing, doc extraction.                     |

[kira]: https://docs.rs/kira/latest/kira
