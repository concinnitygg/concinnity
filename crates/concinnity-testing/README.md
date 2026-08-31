# concinnity-testing

Test scaffolding shared by the workspace. A dev-dependency only: nothing here
ships, and `publish = false` keeps it out of a release.

It depends on `concinnity-core` and nothing else in the workspace, so any crate
can take it as a dev-dependency without a cycle and without pulling an engine
tier into its graph.

- `TempTree` and `write_into`: every path a test writes, under a directory that
  deletes itself.
- `fixtures`: synthetic PNG, GLB and WAV bytes, built in memory, so a test needs
  no checked-in binary.
- `exclusive` / `shared`: the one reader/writer lock over a test binary's
  process-global state. `GlobalState` is that exclusive guard plus the working
  directory, state-root and window-policy moves, all put back on drop.
- `forbid_windows`: arms the tripwire that turns a stray windowed run into a
  panic naming the backend, rather than a hang.
- `shared_cache_dir`: the one thing a test may keep between runs, for a
  content-addressed cache that exists to avoid recompiling.
- `source`: reading the workspace's own sources, for the guard tests that forbid
  a shape which at runtime would hang rather than fail.
