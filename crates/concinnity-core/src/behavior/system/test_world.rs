// The world the unit tests and the benchmarks drive: no compiled blob and no
// renderer behind it, with each behavior pushed exactly as a load would.

use alloc::vec::Vec;

use crate::components::Behavior;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

pub(super) fn world_with(behaviors: Vec<Behavior>) -> World {
    let mut world = World::new();
    // Each behavior is identified by its position, from 1.
    for (b, id) in behaviors.into_iter().zip(1..) {
        world.push_identified(AssetId(id), b);
    }
    world
}
