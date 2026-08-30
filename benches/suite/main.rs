//! The benchmark suite: every measured subsystem in one `harness = false`
//! target, one binary. Bare-word arguments select benchmarks by substring and
//! every name is prefixed with its module, so `-- anim/` runs one module's
//! set; `--json <path>` writes the records for tooling. See
//! [README.md](README.md) for the full invocation list.

mod support;

mod anim;
mod cook;
mod engine;
mod render;

use support::Bench;

fn main() {
    let mut bench = Bench::from_env();
    anim::benches(&mut bench);
    cook::benches(&mut bench);
    engine::benches(&mut bench);
    render::benches(&mut bench);
    bench.finish();
}
