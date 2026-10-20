use alloc::vec;
use alloc::vec::Vec;

use super::SpawnSystem;
use crate::animation::skeleton::Skeleton;
use crate::components::{
    Children, DespawnRequest, GlobalTransform, Hidden, Lifetime, Parent, RenderHandle,
    ReparentRequest, SkeletonPose, SpawnRequest, Spawner, Transform, VisibilityRequest,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{Entity, MenuActive, SkinnedMeshHandle, System, World};
use crate::gfx::render_types::{DrawIndex, SkinnedIndex};
use crate::render::backend::recording::{Call, RecordingBackend, recording_backend};
use crate::render::frame::ops::RenderOps;
use crate::render::frame::queues::{ActiveRenderQueues, RenderQueues};
use crate::render::slots::render_slots::RenderSlots;
use crate::transform::propagation;

const PROP: AssetId = AssetId(4);
const OTHER: AssetId = AssetId(8);

fn at(position: [f32; 3]) -> Transform {
    Transform {
        position,
        rotation_deg: [0.0; 3],
        scale: [1.0; 3],
    }
}

// An identified placement at (1, 2, 3) owning draw slot `slot`.
fn place(world: &mut World, id: AssetId, slot: u32) -> Entity {
    let mut ctx = world.context();
    let entity = ctx.components.spawn();
    let transform = at([1.0, 2.0, 3.0]);
    ctx.insert(entity, GlobalTransform(transform.model_matrix()));
    ctx.insert(entity, transform);
    ctx.insert(
        entity,
        RenderHandle {
            draws: [DrawIndex(slot)].into(),
        },
    );
    ctx.identify(entity, id);
    entity
}

// A world with the one PROP placement at slot 0 and the recording surfaces
// graphics init publishes, over a backend that built `draws` slots.
fn world_with_queues(draws: usize) -> World {
    let mut world = World::new();
    place(&mut world, PROP, 0);
    ActiveRenderQueues::put(
        world.context().resources,
        RenderQueues {
            ops: RenderOps::default(),
            slots: RenderSlots::new(draws, true, &[]),
        },
    );
    world
}

// One SpawnSystem step with its recorded ops replayed onto `backend`, the
// way submission applies them.
fn churn(spawn: &mut SpawnSystem, world: &mut World, backend: &mut RecordingBackend) {
    spawn.step(&mut world.context());
    if let Some(mut queues) = ActiveRenderQueues::take(world.context().resources) {
        queues.ops.replay(backend);
        ActiveRenderQueues::put(world.context().resources, queues);
    }
}

fn clones(calls: &[Call]) -> usize {
    calls
        .iter()
        .filter(|c| matches!(c, Call::CloneStaticDrawObject { .. }))
        .count()
}

fn due_spawner(template: AssetId, lifetime: f32) -> Spawner {
    // Already at the interval, so the first tick is due at any dt >= 0.
    Spawner {
        template,
        interval: 1.0,
        lifetime,
        elapsed: 1.0,
        count: 0,
    }
}

#[test]
fn a_spawn_request_clones_the_template_draw_slot() {
    let (state, mut backend) = recording_backend();
    let mut world = world_with_queues(1);
    world
        .context()
        .events_mut::<SpawnRequest>()
        .send(SpawnRequest {
            template: PROP,
            name: Some(AssetId(900)),
            transform: at([5.0, 0.0, 0.0]),
            lifetime_secs: None,
        });
    churn(&mut SpawnSystem::new(), &mut world, &mut backend);

    assert!(state.lock().saw(&Call::CloneStaticDrawObject {
        src: DrawIndex(0),
        new_idx: DrawIndex(1)
    }));
    let mut handles: Vec<Vec<DrawIndex>> = world
        .context()
        .query::<RenderHandle>()
        .map(|h| h.draws.to_vec())
        .collect();
    handles.sort();
    assert_eq!(
        handles,
        vec![vec![DrawIndex(0)], vec![DrawIndex(1)]],
        "the spawned copy owns slot 1"
    );
}

#[test]
fn a_visibility_request_switches_slots_and_the_hidden_tag() {
    let (state, mut backend) = recording_backend();
    let mut world = world_with_queues(1);
    let mut spawn = SpawnSystem::new();
    for visible in [false, true] {
        world
            .context()
            .events_mut::<VisibilityRequest>()
            .send(VisibilityRequest {
                target: PROP.into(),
                visible,
            });
        churn(&mut spawn, &mut world, &mut backend);
        assert!(state.lock().saw(&Call::UpdateVisibility {
            draw_idx: DrawIndex(0),
            visible
        }));
        assert_eq!(
            world.context().query::<Hidden>().count(),
            usize::from(!visible),
            "hide tags the entity and show clears it"
        );
    }
}

#[test]
fn a_despawn_request_retires_the_draw_slots() {
    let (state, mut backend) = recording_backend();
    let mut world = world_with_queues(1);
    world
        .context()
        .events_mut::<DespawnRequest>()
        .send(DespawnRequest {
            target: PROP.into(),
        });
    churn(&mut SpawnSystem::new(), &mut world, &mut backend);

    assert!(state.lock().saw(&Call::RetireDrawObject(DrawIndex(0))));
    assert_eq!(world.context().query::<RenderHandle>().count(), 0);
}

// With no recording surfaces (graphics never inited) there are no draw slots
// to retire or clone, so the churn waits rather than running.
#[test]
fn without_recording_surfaces_the_churn_stays_pending() {
    let mut world = World::new();
    place(&mut world, PROP, 0);
    world
        .context()
        .events_mut::<DespawnRequest>()
        .send(DespawnRequest {
            target: PROP.into(),
        });
    SpawnSystem::new().step(&mut world.context());
    assert_eq!(world.context().query::<RenderHandle>().count(), 1);
}

// A ReparentRequest naming both ends re-points the child's Parent edge and
// recomposes its world matrix under the new parent.
#[test]
fn a_reparent_request_repoints_the_child_under_the_named_parent() {
    let (_state, mut backend) = recording_backend();
    let mut world = world_with_queues(2);
    let child = place(&mut world, OTHER, 1);
    let parent = world.context().entity_of(PROP).unwrap();
    world
        .context()
        .events_mut::<ReparentRequest>()
        .send(ReparentRequest {
            child: OTHER.into(),
            parent: Some(PROP.into()),
        });
    churn(&mut SpawnSystem::new(), &mut world, &mut backend);

    let ctx = world.context();
    assert_eq!(ctx.get::<Parent>(child).map(|p| p.0), Some(parent));
    // Both are placed at (1, 2, 3), so the child's world position is the
    // parent's translation applied twice.
    let global = ctx.get::<GlobalTransform>(child).unwrap();
    assert_eq!(
        [global.0[3][0], global.0[3][1], global.0[3][2]],
        [2.0, 4.0, 6.0]
    );
}

// A named parent that does not resolve skips the request: falling through to
// no parent would detach the child, so a typo'd name would teleport it.
#[test]
fn a_reparent_request_with_an_unresolved_parent_is_skipped() {
    let (_state, mut backend) = recording_backend();
    let mut world = world_with_queues(2);
    let child = place(&mut world, OTHER, 1);
    let parent = world.context().entity_of(PROP).unwrap();
    {
        let mut ctx = world.context();
        propagation::reparent(&mut ctx, child, Some(parent));
        ctx.events_mut::<ReparentRequest>().send(ReparentRequest {
            child: OTHER.into(),
            parent: Some(AssetId(900).into()),
        });
    }
    churn(&mut SpawnSystem::new(), &mut world, &mut backend);

    assert_eq!(
        world.context().get::<Parent>(child).map(|p| p.0),
        Some(parent)
    );
}

// A ReparentRequest naming no parent is a detach: the child becomes a root and
// its world matrix falls back to its own transform.
#[test]
fn a_reparent_request_without_a_parent_detaches_the_child() {
    let (_state, mut backend) = recording_backend();
    let mut world = world_with_queues(2);
    let child = place(&mut world, OTHER, 1);
    let parent = world.context().entity_of(PROP).unwrap();
    {
        let mut ctx = world.context();
        propagation::reparent(&mut ctx, child, Some(parent));
        ctx.events_mut::<ReparentRequest>().send(ReparentRequest {
            child: OTHER.into(),
            parent: None,
        });
    }
    churn(&mut SpawnSystem::new(), &mut world, &mut backend);

    let ctx = world.context();
    assert!(ctx.get::<Parent>(child).is_none());
    let global = ctx.get::<GlobalTransform>(child).unwrap();
    assert_eq!(
        [global.0[3][0], global.0[3][1], global.0[3][2]],
        [1.0, 2.0, 3.0]
    );
}

// A child name that does not resolve finds nothing to move.
#[test]
fn a_reparent_request_with_an_unresolved_child_is_skipped() {
    let (_state, mut backend) = recording_backend();
    let mut world = world_with_queues(1);
    world
        .context()
        .events_mut::<ReparentRequest>()
        .send(ReparentRequest {
            child: AssetId(901).into(),
            parent: Some(PROP.into()),
        });
    churn(&mut SpawnSystem::new(), &mut world, &mut backend);

    let parent = world.context().entity_of(PROP).unwrap();
    assert!(
        world
            .context()
            .get::<Children>(parent)
            .is_none_or(|c| c.0.is_empty())
    );
}

// A Lifetime already at zero expires on the next tick through the despawn
// cascade: the entity leaves the world and its slot is retired.
#[test]
fn an_expired_lifetime_despawns_the_entity_and_retires_its_slot() {
    let (state, mut backend) = recording_backend();
    let mut world = world_with_queues(1);
    let entity = world.context().entity_of(PROP).unwrap();
    world.context().insert(entity, Lifetime { remaining: 0.0 });
    churn(&mut SpawnSystem::new(), &mut world, &mut backend);

    assert!(state.lock().saw(&Call::RetireDrawObject(DrawIndex(0))));
    assert_eq!(world.context().query::<RenderHandle>().count(), 0);
}

// A due Spawner clones its template's slot at the spawner's own transform, and
// the copy carries the spawner's Lifetime.
#[test]
fn a_due_spawner_clones_its_template_at_its_own_transform() {
    let (state, mut backend) = recording_backend();
    let mut world = world_with_queues(1);
    {
        let mut ctx = world.context();
        let spawner = ctx.components.spawn();
        ctx.insert(spawner, due_spawner(PROP, 5.0));
        ctx.insert(spawner, at([9.0, 0.0, 0.0]));
    }
    churn(&mut SpawnSystem::new(), &mut world, &mut backend);

    assert!(state.lock().saw(&Call::CloneStaticDrawObject {
        src: DrawIndex(0),
        new_idx: DrawIndex(1)
    }));
    let ctx = world.context();
    let lifetimes: Vec<f32> = ctx.query::<Lifetime>().map(|l| l.remaining).collect();
    assert_eq!(lifetimes, [5.0]);
    assert_eq!(ctx.query::<Spawner>().map(|s| s.count).next(), Some(1));
}

// The world clock freezes behind an open menu: neither a due Lifetime nor a
// due Spawner fires while `MenuActive` is set, and both resume once it clears.
#[test]
fn an_open_menu_freezes_lifetimes_and_spawners() {
    let (state, mut backend) = recording_backend();
    let mut world = world_with_queues(1);
    {
        let mut ctx = world.context();
        let entity = ctx.entity_of(PROP).unwrap();
        ctx.insert(entity, Lifetime { remaining: 0.0 });
        let spawner = ctx.components.spawn();
        ctx.insert(spawner, due_spawner(PROP, 0.0));
    }
    world.insert_resource(MenuActive(true));
    let mut spawn = SpawnSystem::new();
    churn(&mut spawn, &mut world, &mut backend);

    {
        let s = state.lock();
        assert!(!s.saw(&Call::RetireDrawObject(DrawIndex(0))));
        assert_eq!(clones(&s.calls), 0);
    }
    assert_eq!(
        world.context().query::<Spawner>().map(|s| s.count).next(),
        Some(0)
    );

    world.insert_resource(MenuActive(false));
    churn(&mut spawn, &mut world, &mut backend);
    assert!(state.lock().saw(&Call::RetireDrawObject(DrawIndex(0))));
}

// A SpawnRequest or Spawner naming a template that does not resolve has
// nothing to copy.
#[test]
fn a_spawn_naming_an_unknown_template_is_skipped() {
    let (state, mut backend) = recording_backend();
    let mut world = world_with_queues(1);
    {
        let mut ctx = world.context();
        ctx.events_mut::<SpawnRequest>().send(SpawnRequest {
            template: AssetId(902),
            name: Some(AssetId(903)),
            transform: Transform::default(),
            lifetime_secs: None,
        });
        let spawner = ctx.components.spawn();
        ctx.insert(spawner, due_spawner(AssetId(904), 0.0));
    }
    churn(&mut SpawnSystem::new(), &mut world, &mut backend);

    assert_eq!(clones(&state.lock().calls), 0);
    assert_eq!(world.context().query::<RenderHandle>().count(), 1);
}

// A spawn naming a skinned template takes the instance-pool path rather than
// cloning a static slot; with no instance reserved it spawns nothing.
#[test]
fn a_spawn_naming_a_skinned_template_takes_the_instance_pool_path() {
    const HERO: AssetId = AssetId(853);
    let (state, mut backend) = recording_backend();
    let mut world = world_with_queues(1);
    {
        let mut ctx = world.context();
        let hero = ctx.components.spawn();
        ctx.insert(
            hero,
            SkeletonPose::new(
                SkinnedMeshHandle::new(0),
                SkinnedIndex(0),
                Skeleton::new(Vec::new()),
            ),
        );
        ctx.identify(hero, HERO);
        ctx.events_mut::<SpawnRequest>().send(SpawnRequest {
            template: HERO,
            name: Some(AssetId(905)),
            transform: Transform::default(),
            lifetime_secs: Some(2.0),
        });
        let spawner = ctx.components.spawn();
        ctx.insert(spawner, due_spawner(HERO, 2.0));
    }
    churn(&mut SpawnSystem::new(), &mut world, &mut backend);

    assert_eq!(clones(&state.lock().calls), 0);
    assert_eq!(world.context().query::<SkeletonPose>().count(), 1);
}
