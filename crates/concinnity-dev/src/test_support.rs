// src/test_support.rs
//
// Tests that touch process-global state -- the session's open project, the
// debug/hot-reload statics, and the engine's development flags -- must not run
// concurrently, because Cargo runs a binary's tests in parallel threads within
// one process.
//
// The guard is the workspace's one process-global lock, so a test here writing
// a development flag excludes the engine's own readers of that same flag
// rather than racing them under a second, private lock.

use concinnity_core::ecs::World;
use concinnity_host::thread::asset_id;
pub(crate) fn lock() -> concinnity_testing::ExclusiveAccess {
    prepare();
    concinnity_testing::exclusive()
}

// What every guard here sets up before taking the lock.
fn prepare() {
    // Route tracing output through the harness's per-test capture so expected
    // ERROR logs stay hidden on pass but replay on failure. First install wins.
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    // Nothing in this crate's tests has a window to open, so a call that
    // reaches one is a regression: panic naming the backend rather than block.
    concinnity_testing::forbid_windows();
}

// Open a project away from the working directory, with the caches on a root of
// their own.
//
// The cache root persists and is shared, so a shader that has not changed is
// compiled once for the machine rather than once per test. The content and
// writable roots are this process's own directory, emptied when it first asks:
// the blobs, the editor's session store, the saves and the settings are state,
// not cache, and a test that inherited them would assert against what another
// left. Per process rather than per machine because the guard below is per
// process, so under a runner that gives each test one -- `cargo nextest` --
// nothing here excludes the test in the process beside it.
pub(crate) fn isolate_state_dir() {
    crate::project::open(
        concinnity_host::store::paths::StateTree::at(concinnity_testing::shared_state_dir(
            "concinnity-dev-tests",
        ))
        .with_cache(concinnity_testing::shared_cache_dir(
            "concinnity-dev-tests-cache",
        )),
    );
}

// A world carrying the editor's injected HUD assets: one blank `Sprite` per
// sprite id, one blank `TextLabel` per label id, and one blank `TextInput` per
// field id. Panels that declare no text fields pass an empty `fields` slice.
pub(crate) fn injected_world(
    sprites: &[asset_id::AssetId],
    labels: &[asset_id::AssetId],
    fields: &[asset_id::AssetId],
) -> World {
    use concinnity_core::components::{Sprite, TextInput, TextLabel};
    let mut world = World::new();
    for &id in sprites {
        world.add_component(Sprite {
            asset_id: id,
            ..Default::default()
        });
    }
    for &id in labels {
        world.add_component(TextLabel {
            asset_id: id,
            ..Default::default()
        });
    }
    for &id in fields {
        world.add_component(TextInput {
            asset_id: id,
            ..Default::default()
        });
    }
    world
}
