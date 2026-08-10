// src/bench/transforms.rs
//
// Per-frame transform propagation: the pass GraphicsSystem runs before it
// builds a draw list. The whole-frame probe cannot read this axis -- its cost
// is attributed to GraphicsSystem, whose timing also carries the present wait,
// so at any load light enough to see the difference the pacing slack absorbs
// it. Driving `propagate_transforms_cached` directly is the only way to get a
// number, which is why these live here.
//
// The pairs that matter are cached-vs-dirty: a static scene must early-out on
// the column change ticks and cost nothing, and a scene whose Transform column
// moved must walk the hierarchy. Both are measured so the cache's worth is a
// subtraction rather than a claim.

use super::{BenchWorld, bench};
use crate::assets::{GlobalTransform, Parent, Prop, Transform};
use crate::ecs::Entity;
use crate::gfx::draw_list::{TransformCache, propagate_transforms_cached};

const FLAT: usize = 10_000;
const CHAINS: usize = 1_250;
const CHAIN_DEPTH: usize = 8;

fn transform_at(i: usize) -> Transform {
    Transform {
        position: [i as f32, 0.45, 0.0],
        rotation_deg: [0.0; 3],
        scale: [1.0; 3],
    }
}

// An entity carrying the three components propagation reads and writes.
fn spawn(world: &mut BenchWorld, i: usize, parent: Option<Entity>) -> Entity {
    let entity = world.components.push_typed(Prop::default());
    world.components.insert_typed(entity, transform_at(i));
    world
        .components
        .insert_typed(entity, GlobalTransform::default());
    if let Some(p) = parent {
        world.components.insert_typed(entity, Parent(p));
    }
    entity
}

// `count` unparented entities: the flat case, where every entity is a root and
// the hierarchy walk is one level deep.
fn flat_world(count: usize) -> (BenchWorld, Vec<Entity>) {
    let mut world = BenchWorld::new();
    let entities = (0..count).map(|i| spawn(&mut world, i, None)).collect();
    (world, entities)
}

// `chains` chains of `depth` entities, each link parented to the one above, so
// resolving a leaf composes `depth` matrices.
fn chained_world(chains: usize, depth: usize) -> (BenchWorld, Vec<Entity>) {
    let mut world = BenchWorld::new();
    let mut entities = Vec::with_capacity(chains * depth);
    for c in 0..chains {
        let mut parent = None;
        for d in 0..depth {
            let entity = spawn(&mut world, c * depth + d, parent);
            entities.push(entity);
            parent = Some(entity);
        }
    }
    (world, entities)
}

#[test]
#[ignore = "microbench; see crate::bench for the command"]
fn transform_propagation() {
    println!("\ntransform propagation");

    // A world nothing touches: every frame after the first must early-out on
    // the change ticks. This is the common case in a mostly-static scene, and
    // the reason the cache exists.
    {
        let (mut world, _) = flat_world(FLAT);
        let mut cache = TransformCache::default();
        propagate_transforms_cached(&mut world.ctx(), &mut cache);
        bench("cached_static/10k", FLAT as u64, || {
            propagate_transforms_cached(&mut world.ctx(), &mut cache);
        });
    }

    // One Transform written per frame, so the tick moves and the whole pass
    // reruns. The delta against the row above is what propagation costs.
    {
        let (mut world, entities) = flat_world(FLAT);
        let mut cache = TransformCache::default();
        let first = entities[0];
        bench("dirty_flat/10k", FLAT as u64, || {
            if let Some(t) = world.ctx().get_mut::<Transform>(first) {
                t.position[1] += 0.001;
            }
            propagate_transforms_cached(&mut world.ctx(), &mut cache);
        });
    }

    // The same entity count arranged as chains, so each leaf composes through
    // its ancestors instead of standing alone.
    {
        let (mut world, entities) = chained_world(CHAINS, CHAIN_DEPTH);
        let mut cache = TransformCache::default();
        let first = entities[0];
        let count = (CHAINS * CHAIN_DEPTH) as u64;
        bench(
            &format!("dirty_chains_depth{CHAIN_DEPTH}/10k"),
            count,
            || {
                if let Some(t) = world.ctx().get_mut::<Transform>(first) {
                    t.position[1] += 0.001;
                }
                propagate_transforms_cached(&mut world.ctx(), &mut cache);
            },
        );
    }
}
