// BehaviorSystem unit tests: drive the tick against a hand-assembled
// PipelineContext with synthetic dt, so timers, delays, and per-entity state
// are deterministic.

use super::*;
use crate::assets::{
    Behavior, Expr, Literal, LocalDecl, Node, Prop, QueryDecl, Target, Transform, VarDecl,
};
use crate::blob::BlobData;
use crate::ecs::{Arena, ComponentStorage, EventCursor, FrameContext, Resources};
use crate::gfx::profile::FrameProfile;

struct TestWorld {
    components: ComponentStorage,
    blob: BlobData,
    profile: FrameProfile,
    resources: Resources,
    scratch: Arena,
    // Simulated time accumulated by the `tick` helper's explicit dts.
    elapsed: f32,
}

impl TestWorld {
    fn ctx(&mut self) -> PipelineContext<'_> {
        PipelineContext {
            components: &mut self.components,
            blob: &mut self.blob,
            profile: &mut self.profile,
            resources: &mut self.resources,
            frame: FrameContext::new(&self.scratch),
        }
    }
}

fn world_with(behaviors: Vec<Behavior>) -> TestWorld {
    let mut world = TestWorld {
        components: ComponentStorage::default(),
        blob: BlobData::empty(),
        profile: FrameProfile::default(),
        resources: Resources::default(),
        scratch: crate::ecs::Arena::with_capacity(64 * 1024),
        elapsed: 0.0,
    };
    for b in behaviors {
        world.components.push_typed(b);
    }
    world
}

fn system(world: &mut TestWorld) -> BehaviorSystem {
    let mut sys = BehaviorSystem::new();
    sys.init(&mut world.ctx());
    sys
}

// Resolve one tag set through the production fill-in-place path.
fn entities_matching(ctx: &PipelineContext, tags: &[u8]) -> Vec<Entity> {
    let mut scratch = Vec::new();
    let mut out = Vec::new();
    BehaviorSystem::entities_matching_into(ctx, tags, &mut scratch, &mut out);
    out
}

// Drive one tick with an explicit dt.
fn tick(sys: &mut BehaviorSystem, world: &mut TestWorld, dt: f32) {
    world.elapsed += dt;
    let elapsed = world.elapsed;
    sys.tick(&mut world.ctx(), dt, elapsed);
}

fn spawn_prop(world: &mut TestWorld, position: [f32; 3]) -> Entity {
    let entity = world.components.push_typed(Prop::default());
    world.components.insert_typed(
        entity,
        Transform {
            position,
            rotation_deg: [0.0; 3],
            scale: [1.0; 3],
        },
    );
    entity
}

fn set_var(var: &str, value: i32, add: bool) -> Node {
    Node::Set {
        var: var.to_string(),
        value: Expr::Int(value),
        add,
    }
}

// The integer reading of a world variable, for the many tests that count.
fn var(sys: &BehaviorSystem, name: &str) -> i32 {
    match var_val(sys, name) {
        Val::Int(i) => i,
        other => panic!("variable '{name}' is {other:?}, not an int"),
    }
}

fn var_val(sys: &BehaviorSystem, name: &str) -> Val {
    sys.var_table
        .slot_of(name)
        .and_then(|s| sys.vars.get(s as usize))
        .copied()
        .unwrap_or(Val::Int(0))
}

fn count<E: 'static>(world: &mut TestWorld, cursor: &mut EventCursor) -> usize {
    world
        .ctx()
        .events::<E>()
        .map(|e| e.read(cursor).count())
        .unwrap_or(0)
}

#[test]
fn start_fires_once_and_writes_a_variable() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Start,
        body: vec![set_var("visits", 1, true)],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.016);
    assert_eq!(var(&sys, "visits"), 1);
    tick(&mut sys, &mut world, 0.016);
    assert_eq!(var(&sys, "visits"), 1, "start fires exactly once");
}

#[test]
fn tick_source_fires_every_tick() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Tick,
        body: vec![set_var("n", 1, true)],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);

    for _ in 0..3 {
        tick(&mut sys, &mut world, 0.016);
    }
    assert_eq!(var(&sys, "n"), 3);
}

#[test]
fn a_scoped_behavior_runs_once_per_matching_entity() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Tick,
        scope: vec!["Prop".into()],
        body: vec![set_var("seen", 1, true)],
        ..Default::default()
    }]);
    spawn_prop(&mut world, [0.0; 3]);
    spawn_prop(&mut world, [1.0, 0.0, 0.0]);
    spawn_prop(&mut world, [2.0, 0.0, 0.0]);
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.016);
    assert_eq!(var(&sys, "seen"), 3, "one run per Prop");
}

#[test]
fn locals_are_per_entity() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Tick,
        scope: vec!["Prop".into()],
        locals: vec![LocalDecl {
            name: "count".into(),
            value: Literal::Int(0),
        }],
        body: vec![Node::SetLocal {
            local: "count".into(),
            value: Expr::Int(1),
            add: true,
        }],
        ..Default::default()
    }]);
    spawn_prop(&mut world, [0.0; 3]);
    spawn_prop(&mut world, [1.0, 0.0, 0.0]);
    let mut sys = system(&mut world);

    for _ in 0..4 {
        tick(&mut sys, &mut world, 0.016);
    }
    let locals: Vec<Val> = sys.instances[0].iter().map(|i| i.locals[0]).collect();
    assert_eq!(
        locals,
        vec![Val::Int(4), Val::Int(4)],
        "each entity counts independently"
    );
}

#[test]
fn self_moves_only_its_own_entity() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Tick,
        scope: vec!["Prop".into()],
        body: vec![Node::SetTransform {
            entity: Expr::SelfEntity,
            position: Some(Expr::Add(
                Box::new(Expr::Position(Box::new(Expr::SelfEntity))),
                Box::new(Expr::Vec3([1.0, 0.0, 0.0])),
            )),
            rotation_deg: None,
            scale: None,
        }],
        ..Default::default()
    }]);
    let a = spawn_prop(&mut world, [0.0; 3]);
    let b = spawn_prop(&mut world, [10.0, 0.0, 0.0]);
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.016);
    let ctx = world.ctx();
    assert_eq!(ctx.get::<Transform>(a).unwrap().position, [1.0, 0.0, 0.0]);
    assert_eq!(ctx.get::<Transform>(b).unwrap().position, [11.0, 0.0, 0.0]);
}

#[test]
fn a_query_counts_matching_entities() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Tick,
        queries: vec![QueryDecl {
            name: "props".into(),
            has: vec!["Prop".into()],
        }],
        body: vec![Node::Set {
            var: "n".into(),
            value: Expr::Count("props".into()),
            add: false,
        }],
        ..Default::default()
    }]);
    spawn_prop(&mut world, [0.0; 3]);
    spawn_prop(&mut world, [1.0, 0.0, 0.0]);
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.016);
    assert_eq!(var(&sys, "n"), 2);
}

#[test]
fn for_each_binds_every_queried_entity() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Tick,
        queries: vec![QueryDecl {
            name: "props".into(),
            has: vec!["Prop".into()],
        }],
        body: vec![Node::ForEach {
            query: "props".into(),
            bind: "e".into(),
            body: vec![Node::SetTransform {
                entity: Expr::Bind("e".into()),
                position: Some(Expr::Vec3([5.0, 0.0, 0.0])),
                rotation_deg: None,
                scale: None,
            }],
        }],
        ..Default::default()
    }]);
    let a = spawn_prop(&mut world, [0.0; 3]);
    let b = spawn_prop(&mut world, [1.0, 0.0, 0.0]);
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.016);
    let ctx = world.ctx();
    assert_eq!(ctx.get::<Transform>(a).unwrap().position, [5.0, 0.0, 0.0]);
    assert_eq!(ctx.get::<Transform>(b).unwrap().position, [5.0, 0.0, 0.0]);
}

#[test]
fn distance_gates_a_condition() {
    // Fires only while the two props are closer than 5 units.
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Tick,
        queries: vec![QueryDecl {
            name: "props".into(),
            has: vec!["Prop".into()],
        }],
        body: vec![
            Node::Let {
                name: "first".into(),
                value: Expr::First("props".into()),
            },
            Node::If {
                cond: Expr::Lt(
                    Box::new(Expr::Distance(
                        Box::new(Expr::Bind("first".into())),
                        Box::new(Expr::Named(Some(AssetId(7)))),
                    )),
                    Box::new(Expr::Float(5.0)),
                ),
                then: vec![set_var("near", 1, false)],
                otherwise: vec![set_var("near", 0, false)],
            },
        ],
        ..Default::default()
    }]);
    let near = spawn_prop(&mut world, [0.0; 3]);
    let far = spawn_prop(&mut world, [100.0, 0.0, 0.0]);
    let mut index = std::collections::BTreeMap::new();
    index.insert(AssetId(7), far);
    world
        .resources
        .insert(crate::ecs::decompose::EntityByName(index));
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.016);
    assert_eq!(var(&sys, "near"), 0, "100 units apart is not near");

    world
        .components
        .get_mut::<Transform>(near)
        .unwrap()
        .position = [98.0, 0.0, 0.0];
    tick(&mut sys, &mut world, 0.016);
    assert_eq!(var(&sys, "near"), 1);
}

#[test]
fn despawn_of_self_addresses_the_entity_not_a_name() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Tick,
        scope: vec!["Prop".into()],
        body: vec![Node::Despawn {
            target: Expr::SelfEntity,
        }],
        ..Default::default()
    }]);
    let entity = spawn_prop(&mut world, [0.0; 3]);
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.016);
    let mut cursor = EventCursor::default();
    let requests: Vec<Target> = world
        .ctx()
        .events::<DespawnRequest>()
        .map(|e| e.read(&mut cursor).map(|r| r.target).collect())
        .unwrap_or_default();
    assert_eq!(requests, vec![Target::Entity(entity)]);
}

#[test]
fn a_timer_fires_on_its_interval() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Timer {
            interval: 1.0,
            repeat: true,
        },
        body: vec![set_var("ticks", 1, true)],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.5);
    assert_eq!(var(&sys, "ticks"), 0);
    tick(&mut sys, &mut world, 0.6);
    assert_eq!(var(&sys, "ticks"), 1);
    tick(&mut sys, &mut world, 1.0);
    assert_eq!(var(&sys, "ticks"), 2);
}

#[test]
fn a_delay_postpones_the_body() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Start,
        delay: 1.0,
        body: vec![set_var("late", 1, false)],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.016);
    assert_eq!(var(&sys, "late"), 0, "the delay has not elapsed");
    tick(&mut sys, &mut world, 1.5);
    assert_eq!(var(&sys, "late"), 1);
}

#[test]
fn a_variable_source_advances_one_link_per_tick() {
    let mut world = world_with(vec![
        Behavior {
            on: BehaviorSource::Start,
            body: vec![set_var("a", 1, false)],
            ..Default::default()
        },
        Behavior {
            on: BehaviorSource::Variable("a".into()),
            body: vec![set_var("b", 1, false)],
            ..Default::default()
        },
    ]);
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.016);
    assert_eq!(var(&sys, "a"), 1);
    assert_eq!(var(&sys, "b"), 0, "the chain has not advanced yet");
    tick(&mut sys, &mut world, 0.016);
    assert_eq!(var(&sys, "b"), 1);
}

#[test]
fn spawn_emits_a_request_and_binds_nothing_this_tick() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Start,
        body: vec![
            Node::Spawn {
                template: Some(AssetId(3)),
                position: [0.0, 1.0, 0.0],
                rotation_deg: [0.0; 3],
                scale: [1.0; 3],
                lifetime: 2.0,
                bind: Some("made".into()),
            },
            // The entity does not exist yet, so this despawn is skipped
            // rather than acting on a stale handle.
            Node::Despawn {
                target: Expr::Bind("made".into()),
            },
        ],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.016);
    let mut spawn_cursor = EventCursor::default();
    assert_eq!(count::<SpawnRequest>(&mut world, &mut spawn_cursor), 1);
    let mut despawn_cursor = EventCursor::default();
    assert_eq!(
        count::<DespawnRequest>(&mut world, &mut despawn_cursor),
        0,
        "a spawn binding holds nothing until the request is applied"
    );
}

#[test]
fn instances_track_entities_appearing_and_disappearing() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Tick,
        scope: vec!["Prop".into()],
        body: vec![set_var("runs", 1, true)],
        ..Default::default()
    }]);
    let first = spawn_prop(&mut world, [0.0; 3]);
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.016);
    assert_eq!(var(&sys, "runs"), 1);

    spawn_prop(&mut world, [1.0, 0.0, 0.0]);
    tick(&mut sys, &mut world, 0.016);
    assert_eq!(var(&sys, "runs"), 3, "two entities now match");

    world.components.despawn(first);
    tick(&mut sys, &mut world, 0.016);
    assert_eq!(var(&sys, "runs"), 4, "one entity remains");
}

#[test]
fn spawned_does_not_fire_for_the_initial_population() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Spawned,
        scope: vec!["Prop".into()],
        body: vec![set_var("born", 1, true)],
        ..Default::default()
    }]);
    spawn_prop(&mut world, [0.0; 3]);
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.016);
    assert_eq!(var(&sys, "born"), 0, "authored entities are not spawned");

    spawn_prop(&mut world, [1.0, 0.0, 0.0]);
    tick(&mut sys, &mut world, 0.016);
    assert_eq!(var(&sys, "born"), 1, "an entity that appeared later is");
}

#[test]
fn query_order_is_stable_across_a_removal() {
    // Column order shifts on swap-remove; the resolved query must not.
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Tick,
        queries: vec![QueryDecl {
            name: "props".into(),
            has: vec!["Prop".into()],
        }],
        body: vec![],
        ..Default::default()
    }]);
    let a = spawn_prop(&mut world, [0.0; 3]);
    let b = spawn_prop(&mut world, [1.0, 0.0, 0.0]);
    let c = spawn_prop(&mut world, [2.0, 0.0, 0.0]);
    let sys = system(&mut world);

    let before = entities_matching(&world.ctx(), &sys.programs[0].queries[0]);
    assert_eq!(before, vec![a, b, c]);

    world.components.despawn(b);
    let after = entities_matching(&world.ctx(), &sys.programs[0].queries[0]);
    assert_eq!(after, vec![a, c], "the survivors keep their relative order");
}

#[test]
fn a_query_intersects_every_declared_component() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Tick,
        queries: vec![QueryDecl {
            name: "placed".into(),
            has: vec!["Prop".into(), "Transform".into()],
        }],
        body: vec![],
        ..Default::default()
    }]);
    let placed = spawn_prop(&mut world, [0.0; 3]);
    // A Prop with no Transform must not match.
    world.components.push_typed(Prop::default());
    let sys = system(&mut world);

    let matched = entities_matching(&world.ctx(), &sys.programs[0].queries[0]);
    assert_eq!(matched, vec![placed]);
}

#[test]
fn each_run_gets_back_exactly_the_effects_it_produced() {
    // Every run appends into one shared buffer and is handed back its own slice
    // by count, so runs that produce different numbers of effects are what
    // catches a mis-paired count: a slip would land one behavior's local write
    // on the other's instance.
    let mut world = world_with(vec![
        Behavior {
            asset_id: AssetId(1),
            on: BehaviorSource::Tick,
            scope: vec!["Prop".into()],
            locals: vec![LocalDecl {
                name: "one".into(),
                value: Literal::Int(0),
            }],
            body: vec![Node::SetLocal {
                local: "one".into(),
                value: Expr::Int(1),
                add: true,
            }],
            ..Default::default()
        },
        Behavior {
            asset_id: AssetId(2),
            on: BehaviorSource::Tick,
            scope: vec!["Prop".into()],
            locals: vec![LocalDecl {
                name: "many".into(),
                value: Literal::Int(0),
            }],
            body: vec![
                Node::SetLocal {
                    local: "many".into(),
                    value: Expr::Int(10),
                    add: true,
                },
                Node::SetLocal {
                    local: "many".into(),
                    value: Expr::Int(100),
                    add: true,
                },
            ],
            ..Default::default()
        },
    ]);
    spawn_prop(&mut world, [0.0; 3]);
    spawn_prop(&mut world, [1.0, 0.0, 0.0]);
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.016);

    let one: Vec<Val> = sys.instances[0].iter().map(|i| i.locals[0]).collect();
    let many: Vec<Val> = sys.instances[1].iter().map(|i| i.locals[0]).collect();
    assert_eq!(one, vec![Val::Int(1), Val::Int(1)], "one effect each");
    assert_eq!(
        many,
        vec![Val::Int(110), Val::Int(110)],
        "both effects each"
    );
}

#[test]
fn alive_follows_the_world_not_this_ticks_queries() {
    // A variable can hold an entity long after the query that produced it stops
    // matching, so `alive` has to ask the world whether the entity exists
    // rather than whether anything this tick happens to name it.
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Tick,
        queries: vec![QueryDecl {
            name: "props".into(),
            has: vec!["Prop".into()],
        }],
        body: vec![
            // An empty query yields nothing and skips this node, so the latched
            // handle survives every tick after the entity stops matching.
            Node::Set {
                var: "target".into(),
                value: Expr::First("props".into()),
                add: false,
            },
            Node::If {
                cond: Expr::Alive(Box::new(Expr::Var("target".into()))),
                then: vec![set_var("still_here", 1, false)],
                otherwise: vec![set_var("still_here", 0, false)],
            },
        ],
        ..Default::default()
    }]);
    let entity = spawn_prop(&mut world, [0.0; 3]);
    let mut sys = system(&mut world);

    // The first tick only latches: a `set` lands after the body that recorded
    // it, so `target` holds the entity from the second tick on.
    tick(&mut sys, &mut world, 0.016);
    tick(&mut sys, &mut world, 0.016);
    assert_eq!(var(&sys, "still_here"), 1, "a matched entity is alive");

    world.components.remove_typed::<Prop>(entity);
    world.components.remove_typed::<Transform>(entity);
    tick(&mut sys, &mut world, 0.016);
    assert_eq!(
        var(&sys, "still_here"),
        1,
        "leaving every query and losing every component is not death",
    );

    world.components.despawn(entity);
    tick(&mut sys, &mut world, 0.016);
    assert_eq!(var(&sys, "still_here"), 0, "a despawned entity is gone");
}

#[test]
fn a_body_reads_positions_from_before_this_ticks_writes() {
    // Effects land only once every body has run, so one behavior's move is
    // invisible to another's read in the same tick no matter which order the
    // two run in. Reads go straight to the world, and this is what makes that
    // safe.
    let mover = Behavior {
        asset_id: AssetId(1),
        on: BehaviorSource::Tick,
        queries: vec![QueryDecl {
            name: "props".into(),
            has: vec!["Prop".into()],
        }],
        body: vec![Node::SetTransform {
            entity: Expr::First("props".into()),
            position: Some(Expr::Vec3([9.0, 0.0, 0.0])),
            rotation_deg: None,
            scale: None,
        }],
        ..Default::default()
    };
    let reader = Behavior {
        asset_id: AssetId(2),
        on: BehaviorSource::Tick,
        queries: vec![QueryDecl {
            name: "props".into(),
            has: vec!["Prop".into()],
        }],
        body: vec![Node::Set {
            var: "seen_x".into(),
            value: Expr::Position(Box::new(Expr::First("props".into()))),
            add: false,
        }],
        ..Default::default()
    };
    let mut world = world_with(vec![mover, reader]);
    let entity = spawn_prop(&mut world, [1.0, 0.0, 0.0]);
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.016);
    assert_eq!(
        var_val(&sys, "seen_x"),
        Val::Vec3([1.0, 0.0, 0.0]),
        "the reader saw the position from before the mover wrote",
    );
    assert_eq!(
        world.ctx().get::<Transform>(entity).unwrap().position,
        [9.0, 0.0, 0.0],
        "the move still landed",
    );
}

#[test]
fn a_cooldown_rate_limits_firing() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Tick,
        cooldown: 1.0,
        body: vec![set_var("n", 1, true)],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.1);
    tick(&mut sys, &mut world, 0.1);
    assert_eq!(var(&sys, "n"), 1, "the second tick is inside the cooldown");
    tick(&mut sys, &mut world, 1.2);
    assert_eq!(var(&sys, "n"), 2);
}

// Source events, the menu freeze, persistence, and the system gate.

fn save_dir(test: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("cn-behavior-{}-{}", std::process::id(), test));
    std::fs::remove_dir_all(&dir).ok();
    dir
}

fn persisting_system(world: &mut TestWorld, dir: &std::path::Path) -> BehaviorSystem {
    let mut sys = BehaviorSystem::new();
    sys.save_dir = dir.to_path_buf();
    sys.init(&mut world.ctx());
    sys
}

fn counter_behavior() -> Behavior {
    Behavior {
        asset_id: AssetId(1),
        body: vec![set_var("visits", 1, true), Node::Save],
        once: true,
        ..Default::default()
    }
}

fn despawn_named(target: u32) -> Node {
    Node::Despawn {
        target: Expr::Named(Some(AssetId(target))),
    }
}

#[test]
fn a_oneshot_timer_fires_exactly_once() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Timer {
            interval: 1.0,
            repeat: false,
        },
        body: vec![set_var("n", 1, true)],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 1.5);
    tick(&mut sys, &mut world, 1.5);
    tick(&mut sys, &mut world, 1.5);
    assert_eq!(var(&sys, "n"), 1);
}

#[test]
fn once_limits_a_repeating_source() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Tick,
        once: true,
        body: vec![set_var("n", 1, true)],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);

    for _ in 0..5 {
        tick(&mut sys, &mut world, 0.016);
    }
    assert_eq!(var(&sys, "n"), 1);
}

#[test]
fn nodes_apply_in_body_order() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Start,
        body: vec![
            set_var("n", 5, false),
            set_var("n", 2, true),
            set_var("n", 1, true),
        ],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.016);
    assert_eq!(var(&sys, "n"), 8);
}

#[test]
fn enter_fires_on_matching_crossings_only() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Enter(Some(AssetId(5))),
        body: vec![despawn_named(7)],
        ..Default::default()
    }]);
    let mut index = std::collections::BTreeMap::new();
    let entity = world.components.push_typed(Prop::default());
    index.insert(AssetId(7), entity);
    world
        .resources
        .insert(crate::ecs::decompose::EntityByName(index));
    let mut sys = system(&mut world);
    let mut cursor = EventCursor::default();

    sys.step(&mut world.ctx());
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 0);

    world.ctx().events_mut::<VolumeEvent>().send(VolumeEvent {
        volume: AssetId(5),
        entered: true,
    });
    sys.step(&mut world.ctx());
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 1);

    // An exit of the same volume, or an enter of another, does not.
    world.ctx().events_mut::<VolumeEvent>().send(VolumeEvent {
        volume: AssetId(5),
        entered: false,
    });
    world.ctx().events_mut::<VolumeEvent>().send(VolumeEvent {
        volume: AssetId(6),
        entered: true,
    });
    sys.step(&mut world.ctx());
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 0);
}

#[test]
fn crossings_survive_a_menu_pause() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Enter(Some(AssetId(5))),
        body: vec![despawn_named(7)],
        ..Default::default()
    }]);
    let mut index = std::collections::BTreeMap::new();
    let entity = world.components.push_typed(Prop::default());
    index.insert(AssetId(7), entity);
    world
        .resources
        .insert(crate::ecs::decompose::EntityByName(index));
    let mut sys = system(&mut world);
    let mut cursor = EventCursor::default();

    // The crossing lands while the menu is open: the paused step drains it but
    // holds it, and the first unpaused step fires it.
    world.ctx().events_mut::<VolumeEvent>().send(VolumeEvent {
        volume: AssetId(5),
        entered: true,
    });
    world.ctx().insert_resource(crate::ecs::MenuActive(true));
    sys.step(&mut world.ctx());
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 0);

    world.ctx().insert_resource(crate::ecs::MenuActive(false));
    sys.step(&mut world.ctx());
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 1);
}

#[test]
fn interact_fires_on_matching_press_only() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Interact(Some(AssetId(4))),
        body: vec![despawn_named(7)],
        ..Default::default()
    }]);
    let mut index = std::collections::BTreeMap::new();
    let entity = world.components.push_typed(Prop::default());
    index.insert(AssetId(7), entity);
    world
        .resources
        .insert(crate::ecs::decompose::EntityByName(index));
    let mut sys = system(&mut world);
    let mut cursor = EventCursor::default();

    world
        .ctx()
        .events_mut::<InteractSignal>()
        .send(InteractSignal { target: AssetId(9) });
    sys.step(&mut world.ctx());
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 0);

    world
        .ctx()
        .events_mut::<InteractSignal>()
        .send(InteractSignal { target: AssetId(4) });
    sys.step(&mut world.ctx());
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 1);
}

#[test]
fn show_and_hide_send_visibility_requests() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Start,
        body: vec![
            Node::Hide {
                target: Expr::Named(Some(AssetId(3))),
            },
            Node::Show {
                target: Expr::Named(Some(AssetId(3))),
            },
        ],
        ..Default::default()
    }]);
    let mut index = std::collections::BTreeMap::new();
    let entity = world.components.push_typed(Prop::default());
    index.insert(AssetId(3), entity);
    world
        .resources
        .insert(crate::ecs::decompose::EntityByName(index));
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.016);
    let mut cursor = EventCursor::default();
    let visible: Vec<bool> = world
        .ctx()
        .events::<VisibilityRequest>()
        .map(|e| e.read(&mut cursor).map(|r| r.visible).collect())
        .unwrap_or_default();
    assert_eq!(visible, vec![false, true], "hide then show, in body order");
}

#[test]
fn story_sends_the_playback_command() {
    let mut world = world_with(vec![Behavior {
        on: BehaviorSource::Start,
        body: vec![Node::Story(StoryPlayback::Continue)],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.016);
    let mut cursor = EventCursor::default();
    assert_eq!(count::<StoryCommand>(&mut world, &mut cursor), 1);
}

#[test]
fn save_persists_vars_and_fired_state_across_runs() {
    let dir = save_dir("roundtrip");

    let mut world = world_with(vec![counter_behavior()]);
    let mut sys = persisting_system(&mut world, &dir);
    tick(&mut sys, &mut world, 0.016);
    assert_eq!(var(&sys, "visits"), 1);

    // A fresh run over the same world: the variable is restored and the fired
    // `once` behavior stays fired.
    let mut world2 = world_with(vec![counter_behavior()]);
    let mut sys2 = persisting_system(&mut world2, &dir);
    assert_eq!(var(&sys2, "visits"), 1, "variable restored at init");
    tick(&mut sys2, &mut world2, 0.016);
    assert_eq!(
        var(&sys2, "visits"),
        1,
        "the fired once behavior does not fire again"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_edited_behavior_loses_its_persisted_fired_flag() {
    let dir = save_dir("edited");

    let mut world = world_with(vec![counter_behavior()]);
    let mut sys = persisting_system(&mut world, &dir);
    tick(&mut sys, &mut world, 0.016);

    // Same asset id, different content: the flag no longer applies, so the
    // behavior fires once more.
    let mut edited = counter_behavior();
    edited.body[0] = set_var("visits", 5, true);
    let mut world2 = world_with(vec![edited]);
    let mut sys2 = persisting_system(&mut world2, &dir);
    tick(&mut sys2, &mut world2, 0.016);
    assert_eq!(var(&sys2, "visits"), 6, "restored 1 + refired add 5");
    std::fs::remove_dir_all(&dir).ok();
}

// Execution tracing: on only while a request stands, with checker-shaped
// node paths.

fn branching_behavior() -> Behavior {
    Behavior {
        asset_id: AssetId(7),
        body: vec![Node::If {
            cond: Expr::Bool(true),
            then: vec![set_var("n", 1, true)],
            otherwise: vec![set_var("n", 5, true)],
        }],
        ..Default::default()
    }
}

#[test]
fn without_a_request_no_trace_is_published() {
    let mut world = world_with(vec![branching_behavior()]);
    let mut sys = system(&mut world);
    tick(&mut sys, &mut world, 0.016);
    assert!(
        world
            .resources
            .get::<crate::ecs::ExecutionTrace>()
            .is_none()
    );
}

#[test]
fn tracing_publishes_events_paths_and_values() {
    use crate::ecs::{TraceEvent, TraceStep};

    let mut world = world_with(vec![branching_behavior()]);
    world.resources.insert(crate::ecs::TraceRequest::default());
    let mut sys = system(&mut world);
    tick(&mut sys, &mut world, 0.016);

    let trace = world
        .resources
        .get::<crate::ecs::ExecutionTrace>()
        .expect("published while requested");
    assert_eq!(trace.frame, 1);
    let ran = |node| {
        trace.events.contains(&TraceEvent {
            behavior: AssetId(7),
            node,
        })
    };
    assert!(ran(0), "the if node ran");
    assert!(ran(1), "the taken then arm ran");
    assert!(!ran(2), "the else arm did not");
    assert!(
        trace
            .vars
            .contains(&("n".to_string(), crate::ecs::TraceVal::Int(1))),
        "world variables carry this tick's values: {:?}",
        trace.vars
    );

    // The path table addresses nodes the way the world checker's faults do.
    let paths = world
        .resources
        .get::<crate::ecs::TracePaths>()
        .expect("published with the first trace");
    let of = |id: usize| &paths.0[0].1[id];
    assert_eq!(of(0), &vec![TraceStep::Field("do"), TraceStep::Index(0)]);
    assert_eq!(
        of(1),
        &vec![
            TraceStep::Field("do"),
            TraceStep::Index(0),
            TraceStep::Field("if"),
            TraceStep::Field("then"),
            TraceStep::Index(0),
        ]
    );
    assert_eq!(of(2)[3], TraceStep::Field("else"));
}

#[test]
fn a_breakpoint_reports_a_hit() {
    use crate::ecs::TraceEvent;

    let mut world = world_with(vec![branching_behavior()]);
    world.resources.insert(crate::ecs::TraceRequest {
        entity: None,
        breakpoints: vec![TraceEvent {
            behavior: AssetId(7),
            node: 1,
        }],
    });
    let mut sys = system(&mut world);
    tick(&mut sys, &mut world, 0.016);
    let trace = world.resources.get::<crate::ecs::ExecutionTrace>().unwrap();
    assert_eq!(
        trace.hit,
        Some(TraceEvent {
            behavior: AssetId(7),
            node: 1,
        })
    );
}

#[test]
fn tracing_surfaces_the_requested_entitys_locals() {
    let scoped = Behavior {
        asset_id: AssetId(9),
        on: BehaviorSource::Tick,
        scope: vec!["Prop".into()],
        locals: vec![LocalDecl {
            name: "count".into(),
            value: Literal::Int(0),
        }],
        body: vec![Node::SetLocal {
            local: "count".into(),
            value: Expr::Int(1),
            add: true,
        }],
        ..Default::default()
    };
    let mut world = world_with(vec![scoped]);
    let entity = spawn_prop(&mut world, [0.0; 3]);
    world.resources.insert(crate::ecs::TraceRequest {
        entity: Some(entity.to_bits()),
        breakpoints: Vec::new(),
    });
    let mut sys = system(&mut world);
    tick(&mut sys, &mut world, 0.016);
    tick(&mut sys, &mut world, 0.016);
    let trace = world.resources.get::<crate::ecs::ExecutionTrace>().unwrap();
    assert_eq!(
        trace.locals,
        vec![(
            AssetId(9),
            "count".to_string(),
            crate::ecs::TraceVal::Int(2)
        )]
    );
}

#[test]
fn transient_saves_neither_read_nor_write_state() {
    let dir = save_dir("transient");

    // A real run leaves a save behind.
    let mut world = world_with(vec![counter_behavior()]);
    let mut sys = persisting_system(&mut world, &dir);
    tick(&mut sys, &mut world, 0.016);
    assert!(save::state_file(&dir).exists());

    // A transient session over the same directory: the save is not restored,
    // and the session's own `save` node writes nothing.
    let mut world2 = world_with(vec![counter_behavior()]);
    world2.resources.insert(crate::ecs::TransientSaves(true));
    let mut sys2 = persisting_system(&mut world2, &dir);
    assert_eq!(var(&sys2, "visits"), 0, "nothing restored at init");
    std::fs::remove_dir_all(&dir).ok();
    tick(&mut sys2, &mut world2, 0.016);
    assert_eq!(var(&sys2, "visits"), 1, "the once behavior fired fresh");
    assert!(!save::state_file(&dir).exists(), "nothing written");
}

#[test]
fn worlds_without_a_save_node_never_read_state() {
    let dir = save_dir("optin");

    let mut world = world_with(vec![counter_behavior()]);
    let mut sys = persisting_system(&mut world, &dir);
    tick(&mut sys, &mut world, 0.016);

    // Same directory, but nothing saves: the world starts fresh.
    let mut world2 = world_with(vec![Behavior {
        body: vec![set_var("visits", 0, false)],
        ..Default::default()
    }]);
    let sys2 = persisting_system(&mut world2, &dir);
    assert_eq!(var(&sys2, "visits"), 0);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_restored_variable_is_not_an_edge_for_variable_sources() {
    let dir = save_dir("baseline");

    let mut world = world_with(vec![
        counter_behavior(),
        Behavior {
            asset_id: AssetId(2),
            on: BehaviorSource::Variable("visits".into()),
            body: vec![set_var("echo", 1, true)],
            ..Default::default()
        },
    ]);
    let mut sys = persisting_system(&mut world, &dir);
    tick(&mut sys, &mut world, 0.016);

    // A second run starts with `visits` already 1; that is not a change, so
    // the variable-sourced behavior must not fire on tick one.
    let mut world2 = world_with(vec![
        counter_behavior(),
        Behavior {
            asset_id: AssetId(2),
            on: BehaviorSource::Variable("visits".into()),
            body: vec![set_var("echo", 1, true)],
            ..Default::default()
        },
    ]);
    let mut sys2 = persisting_system(&mut world2, &dir);
    tick(&mut sys2, &mut world2, 0.016);
    assert_eq!(var(&sys2, "echo"), 0);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn behavior_gates_the_system_and_a_menu_freezes_it() {
    let mut world = crate::ecs::World::new_empty();
    // A `story` node emits unconditionally; `named` targets would need a live
    // entity and a name index this bare world has not built.
    world.add_component(Behavior {
        body: vec![Node::Story(StoryPlayback::Start)],
        ..Default::default()
    });
    world.start().unwrap();
    let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
    assert_eq!(names, ["BehaviorSystem"]);

    let mut cursor = EventCursor::default();
    world.insert_resource(crate::ecs::MenuActive(true));
    world.step();
    let fired = world
        .events::<StoryCommand>()
        .map(|e| e.read(&mut cursor).count())
        .unwrap_or(0);
    assert_eq!(fired, 0, "a paused world fires nothing");

    world.insert_resource(crate::ecs::MenuActive(false));
    world.step();
    let fired = world
        .events::<StoryCommand>()
        .expect("the behavior fired after unpause")
        .read(&mut cursor)
        .count();
    assert_eq!(fired, 1);
}

// Typed world variables, declared by the world's Variables asset.

fn world_with_vars(behaviors: Vec<Behavior>, vars: Vec<(&str, Literal)>) -> TestWorld {
    let mut world = world_with(behaviors);
    world.components.push_typed(Variables {
        vars: vars
            .into_iter()
            .map(|(name, value)| VarDecl {
                name: name.to_string(),
                value,
            })
            .collect(),
        ..Default::default()
    });
    world
}

#[test]
fn a_declared_variable_starts_at_its_declared_value() {
    let mut world = world_with_vars(
        vec![Behavior {
            on: BehaviorSource::Start,
            body: vec![],
            ..Default::default()
        }],
        vec![("health", Literal::Float(100.0))],
    );
    let sys = system(&mut world);
    assert_eq!(var_val(&sys, "health"), Val::Float(100.0));
}

#[test]
fn a_float_variable_holds_a_float() {
    let mut world = world_with_vars(
        vec![Behavior {
            on: BehaviorSource::Start,
            body: vec![Node::Set {
                var: "health".into(),
                value: Expr::Float(-2.5),
                add: true,
            }],
            ..Default::default()
        }],
        vec![("health", Literal::Float(100.0))],
    );
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.016);
    assert_eq!(
        var_val(&sys, "health"),
        Val::Float(97.5),
        "a float variable does not truncate"
    );
}

#[test]
fn a_vec3_variable_feeds_a_transform() {
    let mut world = world_with_vars(
        vec![Behavior {
            on: BehaviorSource::Tick,
            scope: vec!["Prop".into()],
            body: vec![Node::SetTransform {
                entity: Expr::SelfEntity,
                position: Some(Expr::Var("spawn".into())),
                rotation_deg: None,
                scale: None,
            }],
            ..Default::default()
        }],
        vec![("spawn", Literal::Vec3([1.0, 2.0, 3.0]))],
    );
    let entity = spawn_prop(&mut world, [0.0; 3]);
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.016);
    assert_eq!(
        world.ctx().get::<Transform>(entity).unwrap().position,
        [1.0, 2.0, 3.0]
    );
}

#[test]
fn an_undeclared_variable_is_still_an_integer() {
    let mut world = world_with_vars(
        vec![Behavior {
            on: BehaviorSource::Start,
            body: vec![set_var("loose", 3, true)],
            ..Default::default()
        }],
        vec![("health", Literal::Float(1.0))],
    );
    let mut sys = system(&mut world);

    tick(&mut sys, &mut world, 0.016);
    assert_eq!(var_val(&sys, "loose"), Val::Int(3));
}

#[test]
fn a_typed_variable_survives_a_save_and_restore() {
    let dir = save_dir("typed");
    let author = || Behavior {
        asset_id: AssetId(1),
        on: BehaviorSource::Start,
        body: vec![
            Node::Set {
                var: "health".into(),
                value: Expr::Float(-25.0),
                add: true,
            },
            Node::Save,
        ],
        ..Default::default()
    };

    let mut world = world_with_vars(vec![author()], vec![("health", Literal::Float(100.0))]);
    let mut sys = persisting_system(&mut world, &dir);
    tick(&mut sys, &mut world, 0.016);
    assert_eq!(var_val(&sys, "health"), Val::Float(75.0));

    let mut world2 = world_with_vars(vec![author()], vec![("health", Literal::Float(100.0))]);
    let sys2 = persisting_system(&mut world2, &dir);
    assert_eq!(
        var_val(&sys2, "health"),
        Val::Float(75.0),
        "the float restores as a float"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_retyped_variable_ignores_its_stale_save() {
    let dir = save_dir("retyped");
    let saver = Behavior {
        asset_id: AssetId(1),
        on: BehaviorSource::Start,
        body: vec![
            Node::Set {
                var: "v".into(),
                value: Expr::Float(7.5),
                add: false,
            },
            Node::Save,
        ],
        ..Default::default()
    };
    let mut world = world_with_vars(vec![saver], vec![("v", Literal::Float(0.0))]);
    let mut sys = persisting_system(&mut world, &dir);
    tick(&mut sys, &mut world, 0.016);

    // The world now declares `v` an int: the saved float no longer applies, so
    // the declared starting value stands.
    let mut world2 = world_with_vars(
        vec![Behavior {
            asset_id: AssetId(1),
            body: vec![Node::Save],
            ..Default::default()
        }],
        vec![("v", Literal::Int(3))],
    );
    let sys2 = persisting_system(&mut world2, &dir);
    assert_eq!(var_val(&sys2, "v"), Val::Int(3));
    std::fs::remove_dir_all(&dir).ok();
}

// The pooled evaluation path (ScheduleMode::Parallel, enough firing
// instances) must land exactly the state the serial path lands: same var
// values, same transforms, same effect order.
#[test]
fn parallel_eval_matches_serial_state() {
    fn run(parallel: bool) -> (Vec<i32>, Vec<[f32; 3]>) {
        let mover = Behavior {
            on: BehaviorSource::Tick,
            scope: vec!["Prop".into()],
            body: vec![
                set_var("total", 1, true),
                Node::SetTransform {
                    entity: Expr::SelfEntity,
                    position: Some(Expr::Add(
                        Box::new(Expr::Position(Box::new(Expr::SelfEntity))),
                        Box::new(Expr::Vec3([0.25, 0.0, 0.0])),
                    )),
                    rotation_deg: None,
                    scale: None,
                },
            ],
            ..Default::default()
        };
        let counter = Behavior {
            on: BehaviorSource::Tick,
            body: vec![Node::Set {
                var: "double".into(),
                value: Expr::Mul(Box::new(Expr::Var("total".into())), Box::new(Expr::Int(2))),
                add: false,
            }],
            ..Default::default()
        };
        let mut world = world_with(vec![mover, counter]);
        if parallel {
            world.resources.insert(crate::ecs::ScheduleMode::Parallel);
        }
        for i in 0..150 {
            spawn_prop(&mut world, [i as f32, 0.0, 0.0]);
        }
        let mut sys = system(&mut world);
        for _ in 0..5 {
            tick(&mut sys, &mut world, 0.016);
        }
        let vars = vec![var(&sys, "total"), var(&sys, "double")];
        let positions = world
            .ctx()
            .query::<Transform>()
            .map(|t| t.position)
            .collect();
        (vars, positions)
    }

    let serial = run(false);
    let parallel = run(true);
    assert_eq!(serial, parallel);
    // 150 movers over 5 ticks; the mover really fired every tick.
    assert_eq!(serial.0[0], 750);
}
