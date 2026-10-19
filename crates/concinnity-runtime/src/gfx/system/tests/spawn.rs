use super::*;
use concinnity_core::render::queues::ActiveRenderQueues;

#[test]
fn spawn_request_clones_template_draw_slot() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    {
        let mut ctx = world.context();
        ctx.events_mut::<SpawnRequest>().send(SpawnRequest {
            template: PROP,
            name: Some(AssetId(900)),
            transform: Transform {
                position: [5.0, 0.0, 0.0],
                rotation_deg: [0.0; 3],
                scale: [1.0; 3],
            },
            lifetime_secs: None,
        });
    }
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);

    assert!(lock(&state).saw(&Call::CloneStaticDrawObject {
        src: DrawIndex(0),
        new_idx: DrawIndex(1)
    }));
    let ctx = world.context();
    let mut handles: Vec<Vec<DrawIndex>> = ctx
        .query::<RenderHandle>()
        .map(|h| h.draws.to_vec())
        .collect();
    handles.sort();
    assert_eq!(
        handles,
        vec![vec![DrawIndex(0)], vec![DrawIndex(1)]],
        "spawned copy owns slot 1"
    );
}

#[test]
fn visibility_request_switches_slots_and_hidden_tag() {
    use concinnity_core::components::{Hidden, VisibilityRequest};

    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    {
        let mut ctx = world.context();
        ctx.events_mut::<VisibilityRequest>()
            .send(VisibilityRequest {
                target: PROP.into(),
                visible: false,
            });
    }
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    assert!(lock(&state).saw(&Call::UpdateVisibility {
        draw_idx: DrawIndex(0),
        visible: false
    }));
    assert_eq!(
        world.context().query::<Hidden>().count(),
        1,
        "hide tags the entity"
    );

    {
        let mut ctx = world.context();
        ctx.events_mut::<VisibilityRequest>()
            .send(VisibilityRequest {
                target: PROP.into(),
                visible: true,
            });
    }
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    assert!(lock(&state).saw(&Call::UpdateVisibility {
        draw_idx: DrawIndex(0),
        visible: true
    }));
    assert_eq!(
        world.context().query::<Hidden>().count(),
        0,
        "show clears the tag"
    );
}

#[test]
fn despawn_request_retires_draw_slots() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    {
        let mut ctx = world.context();
        ctx.events_mut::<DespawnRequest>().send(DespawnRequest {
            target: PROP.into(),
        });
    }
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);

    assert!(lock(&state).saw(&Call::RetireDrawObject(DrawIndex(0))));
    let ctx = world.context();
    assert_eq!(
        ctx.query::<RenderHandle>().count(),
        0,
        "despawned entity gone"
    );
}

// Drive a persistent SpawnSystem the way the schedule does -- against the
// world's parked backend -- without the rest of the frame, so a test can seed
// the world clock's inputs (Lifetime / Spawner / MenuActive) and observe just
// the churn.
// One SpawnSystem step followed by a replay of its recorded ops onto the
// parked backend, so the mock's call log reflects what submission would apply.
fn spawn_step(spawn: &mut concinnity_core::spawn::SpawnSystem, world: &mut World) {
    use concinnity_core::ecs::System;
    {
        let mut ctx = world.context();
        spawn.step(&mut ctx);
    }
    replay_pending_ops(world);
}

// Replay the recorded op queue onto the parked backend (both restored after),
// standing in for the submit-path replay in tests that step one system.
fn replay_pending_ops(world: &mut World) {
    let Some(mut queues) = ActiveRenderQueues::take(world.context().resources) else {
        return;
    };
    if let Some(mut backend) =
        concinnity_core::ecs::ActiveRenderBackend::take(world.context().resources)
    {
        queues.ops.replay(backend.as_mut());
        concinnity_core::ecs::ActiveRenderBackend::put(world.context().resources, backend);
    }
    ActiveRenderQueues::put(world.context().resources, queues);
}

// The entity an asset id resolves to through `EntityById`.
fn entity_named(world: &mut World, name: AssetId) -> Entity {
    world
        .context()
        .entity_of(name)
        .expect("the id resolves to an entity")
}

// With no op queue / slot allocator (graphics never inited) there are no draw
// slots to retire or clone, so the churn waits rather than dropping the
// events on the floor.
#[test]
fn spawn_system_without_graphics_leaves_the_churn_pending() {
    let (_state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let _gs = init_graphics(&mut world, hooks);
    ActiveRenderQueues::take(world.context().resources)
        .expect("the recording surfaces were published");

    {
        let mut ctx = world.context();
        ctx.events_mut::<DespawnRequest>().send(DespawnRequest {
            target: PROP.into(),
        });
    }
    let mut spawn = concinnity_core::spawn::SpawnSystem::new();
    spawn_step(&mut spawn, &mut world);

    assert_eq!(
        world.context().query::<RenderHandle>().count(),
        1,
        "the despawn never ran without the slot allocator to retire into"
    );
}

// A ReparentRequest naming both ends re-points the child's Parent edge and
// recomposes its world matrix under the new parent.
#[test]
fn reparent_request_repoints_the_child_under_the_named_parent() {
    const OTHER: AssetId = AssetId(8);
    let (_state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push_textured_quad(AssetId(5), AssetId(6), AssetId(7), OTHER);
    let mut world = b.build();
    let _gs = init_graphics(&mut world, hooks);
    let child = entity_named(&mut world, OTHER);
    let parent = entity_named(&mut world, PROP);

    {
        let mut ctx = world.context();
        ctx.events_mut::<ReparentRequest>().send(ReparentRequest {
            child: OTHER.into(),
            parent: Some(PROP.into()),
        });
    }
    let mut spawn = concinnity_core::spawn::SpawnSystem::new();
    spawn_step(&mut spawn, &mut world);

    let ctx = world.context();
    assert_eq!(
        ctx.get::<Parent>(child).map(|p| p.0),
        Some(parent),
        "the child hangs off the named parent"
    );
    // Both quads are placed at (1, 2, 3), so the child's world position is now
    // the parent's translation applied twice.
    let global = ctx
        .get::<GlobalTransform>(child)
        .expect("child keeps a world matrix");
    assert_eq!(
        [global.0[3][0], global.0[3][1], global.0[3][2]],
        [2.0, 4.0, 6.0],
        "the child's world matrix recomposed under the parent chain"
    );
}

// Regression guard: a ReparentRequest naming a parent that does NOT resolve is
// skipped entirely. Silently falling through to `None` would detach the child to
// a world root, so a typo'd parent name would teleport the child instead of
// doing nothing.
#[test]
fn reparent_request_with_an_unresolved_parent_is_skipped() {
    const OTHER: AssetId = AssetId(8);
    const GHOST: AssetId = AssetId(900);
    let (_state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push_textured_quad(AssetId(5), AssetId(6), AssetId(7), OTHER);
    let mut world = b.build();
    let _gs = init_graphics(&mut world, hooks);
    let child = entity_named(&mut world, OTHER);
    let parent = entity_named(&mut world, PROP);

    // Park the child under a real parent first, so a wrongful detach shows.
    {
        let mut ctx = world.context();
        propagation::reparent(&mut ctx, child, Some(parent));
        ctx.events_mut::<ReparentRequest>().send(ReparentRequest {
            child: OTHER.into(),
            parent: Some(GHOST.into()),
        });
    }
    let mut spawn = concinnity_core::spawn::SpawnSystem::new();
    spawn_step(&mut spawn, &mut world);

    assert_eq!(
        world.context().get::<Parent>(child).map(|p| p.0),
        Some(parent),
        "an unresolved parent name never detaches the child to a root"
    );
}

// A ReparentRequest with no parent named at all IS a detach: the child becomes a
// root and its world matrix falls back to its own transform.
#[test]
fn reparent_request_without_a_parent_detaches_the_child() {
    const OTHER: AssetId = AssetId(8);
    let (_state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push_textured_quad(AssetId(5), AssetId(6), AssetId(7), OTHER);
    let mut world = b.build();
    let _gs = init_graphics(&mut world, hooks);
    let child = entity_named(&mut world, OTHER);
    let parent = entity_named(&mut world, PROP);

    {
        let mut ctx = world.context();
        propagation::reparent(&mut ctx, child, Some(parent));
        ctx.events_mut::<ReparentRequest>().send(ReparentRequest {
            child: OTHER.into(),
            parent: None,
        });
    }
    let mut spawn = concinnity_core::spawn::SpawnSystem::new();
    spawn_step(&mut spawn, &mut world);

    let ctx = world.context();
    assert!(
        ctx.get::<Parent>(child).is_none(),
        "an unnamed parent detaches the child"
    );
    let global = ctx.get::<GlobalTransform>(child).unwrap();
    assert_eq!(
        [global.0[3][0], global.0[3][1], global.0[3][2]],
        [1.0, 2.0, 3.0],
        "the detached child falls back to its own transform"
    );
}

// A Lifetime already at zero expires on the next tick (any dt >= 0), routing
// through the same cascade a DespawnRequest uses: the entity leaves the ECS and
// its draw slot is retired.
#[test]
fn expired_lifetime_despawns_the_entity_and_retires_its_slot() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let _gs = init_graphics(&mut world, hooks);
    let entity = entity_named(&mut world, PROP);
    world.context().insert(entity, Lifetime { remaining: 0.0 });

    let mut spawn = concinnity_core::spawn::SpawnSystem::new();
    spawn_step(&mut spawn, &mut world);

    assert!(lock(&state).saw(&Call::RetireDrawObject(DrawIndex(0))));
    assert_eq!(
        world.context().query::<RenderHandle>().count(),
        0,
        "the expired entity is gone"
    );
}

// A Spawner whose accumulator already covers its interval is due on the next
// tick (any dt >= 0): it clones its template's draw slot at the spawner's own
// transform, and the copy carries the spawner's Lifetime countdown.
#[test]
fn due_spawner_clones_its_template_at_its_own_transform() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let _gs = init_graphics(&mut world, hooks);
    {
        let mut ctx = world.context();
        let spawner = ctx.components.spawn();
        ctx.insert(
            spawner,
            Spawner {
                template: PROP,
                interval: 1.0,
                lifetime: 5.0,
                // Already at the interval, so the first tick is due at dt >= 0.
                elapsed: 1.0,
                count: 0,
            },
        );
        ctx.insert(
            spawner,
            Transform {
                position: [9.0, 0.0, 0.0],
                rotation_deg: [0.0; 3],
                scale: [1.0; 3],
            },
        );
    }

    let mut spawn = concinnity_core::spawn::SpawnSystem::new();
    spawn_step(&mut spawn, &mut world);

    assert!(lock(&state).saw(&Call::CloneStaticDrawObject {
        src: DrawIndex(0),
        new_idx: DrawIndex(1)
    }));
    let ctx = world.context();
    assert_eq!(
        ctx.query::<Lifetime>()
            .map(|l| l.remaining)
            .collect::<Vec<_>>(),
        vec![5.0],
        "the cadence copy carries the spawner's lifetime"
    );
    let spawner_count = ctx.query::<Spawner>().map(|s| s.count).next();
    assert_eq!(spawner_count, Some(1), "the spawner counted the copy");
}

// The world clock freezes behind an open menu: neither a due Lifetime nor a due
// Spawner advances while `MenuActive` is set, and both resume once it clears.
// Time-independent, so it holds whatever the real frame dt happens to be.
#[test]
fn menu_active_freezes_lifetimes_and_spawners() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let _gs = init_graphics(&mut world, hooks);
    let entity = entity_named(&mut world, PROP);
    {
        let mut ctx = world.context();
        ctx.insert(entity, Lifetime { remaining: 0.0 });
        let spawner = ctx.components.spawn();
        ctx.insert(
            spawner,
            Spawner {
                template: PROP,
                interval: 1.0,
                lifetime: 0.0,
                elapsed: 1.0,
                count: 0,
            },
        );
    }
    world.insert_resource(MenuActive(true));

    let mut spawn = concinnity_core::spawn::SpawnSystem::new();
    spawn_step(&mut spawn, &mut world);

    {
        let s = lock(&state);
        assert!(
            !s.saw(&Call::RetireDrawObject(DrawIndex(0))),
            "the expired Lifetime does not fire behind the menu"
        );
        assert!(
            !s.calls
                .iter()
                .any(|c| matches!(c, Call::CloneStaticDrawObject { .. })),
            "the due Spawner does not fire behind the menu"
        );
    }
    assert_eq!(
        world.context().query::<Spawner>().map(|s| s.count).next(),
        Some(0),
        "the spawner's clock never advanced"
    );

    // Closing the menu resumes the same world clock.
    world.insert_resource(MenuActive(false));
    spawn_step(&mut spawn, &mut world);
    assert!(
        lock(&state).saw(&Call::RetireDrawObject(DrawIndex(0))),
        "the expiry fires once the menu closes"
    );
}

// A ReparentRequest whose child name does not resolve finds nothing to move (a
// reparent naming an entity a despawn already removed is a no-op, not a panic).
#[test]
fn reparent_request_with_an_unresolved_child_is_skipped() {
    let (_state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let _gs = init_graphics(&mut world, hooks);

    {
        let mut ctx = world.context();
        ctx.events_mut::<ReparentRequest>().send(ReparentRequest {
            child: AssetId(901).into(),
            parent: Some(PROP.into()),
        });
    }
    let mut spawn = concinnity_core::spawn::SpawnSystem::new();
    spawn_step(&mut spawn, &mut world);

    let child = entity_named(&mut world, PROP);
    assert!(
        world
            .context()
            .get::<Children>(child)
            .is_none_or(|c| c.0.is_empty()),
        "nothing was parented under the named parent"
    );
}

// A SpawnRequest or Spawner naming a template that does not resolve is skipped:
// there is nothing to copy, so no draw slot is cloned.
#[test]
fn a_spawn_naming_an_unknown_template_is_skipped() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let _gs = init_graphics(&mut world, hooks);
    {
        let mut ctx = world.context();
        ctx.events_mut::<SpawnRequest>().send(SpawnRequest {
            template: AssetId(902),
            name: Some(AssetId(903)),
            transform: Transform::default(),
            lifetime_secs: None,
        });
        let spawner = ctx.components.spawn();
        ctx.insert(
            spawner,
            Spawner {
                template: AssetId(904),
                interval: 1.0,
                lifetime: 0.0,
                elapsed: 1.0,
                count: 0,
            },
        );
    }
    let mut spawn = concinnity_core::spawn::SpawnSystem::new();
    spawn_step(&mut spawn, &mut world);

    assert!(
        !lock(&state)
            .calls
            .iter()
            .any(|c| matches!(c, Call::CloneStaticDrawObject { .. })),
        "an unresolvable template clones nothing"
    );
    assert_eq!(
        world.context().query::<RenderHandle>().count(),
        1,
        "only the original"
    );
}

// A spawn naming a SKINNED template dispatches to the instance-pool path rather
// than cloning a static draw slot: a skinned copy claims one of the hidden
// bind-pose instances reserved at init. With every reserved slot taken (here:
// none reserved) the spawn yields nothing rather than growing a GPU buffer.
#[test]
fn a_spawn_naming_a_skinned_template_takes_the_instance_pool_path() {
    use concinnity_core::components::SkinnedMesh;

    const HERO: AssetId = AssetId(853);
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    push_skinned_mesh(
        &mut b,
        HERO,
        SkinnedMesh {
            // No instances reserved, so the pool has nothing to hand out.
            max_instances: 0,
            ..Default::default()
        },
        3,
    );
    let mut world = b.build();
    let _gs = init_graphics(&mut world, hooks);
    let poses_before = world.context().query::<SkeletonPose>().count();

    {
        let mut ctx = world.context();
        ctx.events_mut::<SpawnRequest>().send(SpawnRequest {
            template: HERO,
            name: Some(AssetId(905)),
            transform: Transform::default(),
            lifetime_secs: Some(2.0),
        });
        // A cadence spawn of the same skinned template takes the same path.
        let spawner = ctx.components.spawn();
        ctx.insert(
            spawner,
            Spawner {
                template: HERO,
                interval: 1.0,
                lifetime: 2.0,
                elapsed: 1.0,
                count: 0,
            },
        );
    }
    let mut spawn = concinnity_core::spawn::SpawnSystem::new();
    spawn_step(&mut spawn, &mut world);

    assert!(
        !lock(&state)
            .calls
            .iter()
            .any(|c| matches!(c, Call::CloneStaticDrawObject { .. })),
        "a skinned template never clones a static draw slot"
    );
    assert_eq!(
        world.context().query::<SkeletonPose>().count(),
        poses_before,
        "an exhausted instance pool spawns nothing rather than growing a buffer"
    );
}
