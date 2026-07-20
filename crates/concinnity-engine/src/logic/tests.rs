// ReactionSystem unit tests: drive the tick against a hand-assembled
// PipelineContext with synthetic dt, so timers, delays, and cooldowns are
// deterministic. Full-world gate and menu-freeze behavior use a real World.

use super::*;
use crate::assets::{
    CmpOp, Condition, DespawnRequest, InteractSignal, Reaction, ReactionAction, ReactionSource,
    SpawnRequest, StoryCommand, StoryPlayback, VisibilityRequest, VolumeEvent,
};
use crate::blob::BlobData;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{ComponentStorage, EventCursor, Resources, System};
use crate::gfx::profile::FrameProfile;

// Owns the storage a PipelineContext borrows from.
struct TestWorld {
    components: ComponentStorage,
    blob: BlobData,
    profile: FrameProfile,
    resources: Resources,
}

impl TestWorld {
    fn ctx(&mut self) -> PipelineContext<'_> {
        PipelineContext {
            components: &mut self.components,
            blob: &mut self.blob,
            profile: &mut self.profile,
            resources: &mut self.resources,
        }
    }
}

fn world_with(reactions: Vec<Reaction>) -> TestWorld {
    let mut world = TestWorld {
        components: ComponentStorage::default(),
        blob: BlobData::empty(),
        profile: FrameProfile::default(),
        resources: Resources::default(),
    };
    for r in reactions {
        world.components.push_typed(r);
    }
    world
}

// An initialized system over the world's reactions.
fn system(world: &mut TestWorld) -> ReactionSystem {
    let mut sys = ReactionSystem::new();
    sys.init(&mut world.ctx());
    sys
}

fn despawn(target: u32) -> ReactionAction {
    ReactionAction::Despawn {
        target: Some(AssetId(target)),
    }
}

fn set(name: &str, value: i32, add: bool) -> ReactionAction {
    ReactionAction::Set {
        name: name.to_string(),
        value,
        add,
    }
}

fn count<E: 'static>(world: &mut TestWorld, cursor: &mut EventCursor) -> usize {
    world
        .ctx()
        .events::<E>()
        .map(|e| e.read(cursor).len())
        .unwrap_or(0)
}

#[test]
fn start_fires_once() {
    let mut world = world_with(vec![Reaction {
        actions: vec![despawn(7)],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);
    let mut cursor = EventCursor::default();

    sys.tick(&mut world.ctx(), 0.0);
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 1);
    sys.tick(&mut world.ctx(), 1.0);
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 0);
}

#[test]
fn failing_condition_suppresses_firing() {
    let mut world = world_with(vec![Reaction {
        conditions: vec![Condition {
            name: "has_key".into(),
            op: CmpOp::Ne,
            value: 0,
        }],
        actions: vec![despawn(7)],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);
    let mut cursor = EventCursor::default();

    sys.tick(&mut world.ctx(), 0.0);
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 0);
}

#[test]
fn passing_condition_fires() {
    let mut world = world_with(vec![Reaction {
        conditions: vec![Condition {
            name: "score".into(),
            op: CmpOp::Ge,
            value: 3,
        }],
        actions: vec![despawn(7)],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);
    world
        .ctx()
        .resource_mut::<Variables>()
        .unwrap()
        .apply("score", 3, false);
    let mut cursor = EventCursor::default();

    sys.tick(&mut world.ctx(), 0.0);
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 1);
}

#[test]
fn repeat_timer_fires_on_cadence() {
    let mut world = world_with(vec![Reaction {
        on: ReactionSource::Timer {
            interval: 1.0,
            repeat: true,
        },
        actions: vec![despawn(7)],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);
    let mut cursor = EventCursor::default();

    sys.tick(&mut world.ctx(), 0.6);
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 0);
    sys.tick(&mut world.ctx(), 0.6);
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 1);
    sys.tick(&mut world.ctx(), 1.0);
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 1);
}

#[test]
fn oneshot_timer_fires_exactly_once() {
    let mut world = world_with(vec![Reaction {
        on: ReactionSource::Timer {
            interval: 1.0,
            repeat: false,
        },
        actions: vec![despawn(7)],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);
    let mut cursor = EventCursor::default();

    sys.tick(&mut world.ctx(), 1.5);
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 1);
    sys.tick(&mut world.ctx(), 5.0);
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 0);
}

#[test]
fn variable_source_fires_on_change_only() {
    let mut world = world_with(vec![Reaction {
        on: ReactionSource::Variable("v".into()),
        actions: vec![set("w", 1, true)],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);

    sys.tick(&mut world.ctx(), 0.0);
    world
        .ctx()
        .resource_mut::<Variables>()
        .unwrap()
        .apply("v", 1, false);
    sys.tick(&mut world.ctx(), 0.0);
    sys.tick(&mut world.ctx(), 0.0);
    let ctx = world.ctx();
    let vars = ctx.resource::<Variables>().unwrap();
    assert_eq!(vars.get("w"), 1, "one change, one firing");
}

#[test]
fn once_limits_a_repeat_source() {
    let mut world = world_with(vec![Reaction {
        on: ReactionSource::Timer {
            interval: 0.0,
            repeat: true,
        },
        actions: vec![despawn(7)],
        once: true,
        ..Default::default()
    }]);
    let mut sys = system(&mut world);
    let mut cursor = EventCursor::default();

    for _ in 0..3 {
        sys.tick(&mut world.ctx(), 0.1);
    }
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 1);
}

#[test]
fn cooldown_spaces_firings() {
    let mut world = world_with(vec![Reaction {
        on: ReactionSource::Timer {
            interval: 0.0,
            repeat: true,
        },
        actions: vec![despawn(7)],
        cooldown: 1.0,
        ..Default::default()
    }]);
    let mut sys = system(&mut world);
    let mut cursor = EventCursor::default();

    sys.tick(&mut world.ctx(), 0.0);
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 1);
    sys.tick(&mut world.ctx(), 0.5);
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 0);
    sys.tick(&mut world.ctx(), 0.6);
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 1);
}

#[test]
fn delay_defers_the_actions() {
    let mut world = world_with(vec![Reaction {
        actions: vec![despawn(7)],
        delay: 1.0,
        ..Default::default()
    }]);
    let mut sys = system(&mut world);
    let mut cursor = EventCursor::default();

    sys.tick(&mut world.ctx(), 0.0);
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 0);
    sys.tick(&mut world.ctx(), 0.6);
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 0);
    sys.tick(&mut world.ctx(), 0.5);
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 1);
}

#[test]
fn set_actions_apply_in_order() {
    let mut world = world_with(vec![Reaction {
        actions: vec![set("v", 2, false), set("v", 3, true)],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);

    sys.tick(&mut world.ctx(), 0.0);
    let mut ctx = world.ctx();
    assert_eq!(ctx.resource_mut::<Variables>().unwrap().get("v"), 5);
}

#[test]
fn variable_chain_advances_one_tick_per_link() {
    let mut world = world_with(vec![
        Reaction {
            actions: vec![set("v", 1, false)],
            ..Default::default()
        },
        Reaction {
            on: ReactionSource::Variable("v".into()),
            actions: vec![set("w", 9, false)],
            ..Default::default()
        },
    ]);
    let mut sys = system(&mut world);

    sys.tick(&mut world.ctx(), 0.0);
    {
        let ctx = world.ctx();
        let vars = ctx.resource::<Variables>().unwrap();
        assert_eq!(vars.get("v"), 1);
        assert_eq!(
            vars.get("w"),
            0,
            "the chained rule sees the change next tick"
        );
    }
    sys.tick(&mut world.ctx(), 0.0);
    let ctx = world.ctx();
    assert_eq!(ctx.resource::<Variables>().unwrap().get("w"), 9);
}

#[test]
fn spawn_action_sends_an_anonymous_request() {
    let mut world = world_with(vec![Reaction {
        actions: vec![ReactionAction::Spawn {
            template: Some(AssetId(3)),
            position: [1.0, 2.0, 3.0],
            rotation_deg: [0.0; 3],
            scale: [0.0; 3],
            lifetime: 2.5,
        }],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);

    sys.tick(&mut world.ctx(), 0.0);
    let ctx = world.ctx();
    let mut cursor = EventCursor::default();
    let events = ctx.events::<SpawnRequest>().unwrap();
    let reqs: Vec<&SpawnRequest> = events.read(&mut cursor).into_iter().collect();
    assert_eq!(reqs.len(), 1);
    assert_eq!(reqs[0].template, AssetId(3));
    assert_eq!(reqs[0].name, None, "reaction spawns are transient");
    assert_eq!(reqs[0].transform.position, [1.0, 2.0, 3.0]);
    assert_eq!(
        reqs[0].transform.scale, [1.0; 3],
        "zero scale reads as unit"
    );
    assert_eq!(reqs[0].lifetime_secs, Some(2.5));
}

#[test]
fn story_action_sends_the_playback_command() {
    let mut world = world_with(vec![Reaction {
        actions: vec![ReactionAction::Story(StoryPlayback::Start)],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);

    sys.tick(&mut world.ctx(), 0.0);
    let ctx = world.ctx();
    let mut cursor = EventCursor::default();
    let events = ctx.events::<StoryCommand>().unwrap();
    let cmds: Vec<&StoryCommand> = events.read(&mut cursor).into_iter().collect();
    assert_eq!(cmds.len(), 1);
    assert!(matches!(cmds[0], StoryCommand::Start));
}

#[test]
fn enter_source_fires_on_matching_crossings_only() {
    let mut world = world_with(vec![Reaction {
        on: ReactionSource::Enter(Some(AssetId(5))),
        actions: vec![despawn(7)],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);
    let mut cursor = EventCursor::default();

    // No crossing: nothing fires.
    sys.step(&mut world.ctx());
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 0);

    // A matching enter fires once.
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
fn exit_source_fires_on_the_way_out() {
    let mut world = world_with(vec![Reaction {
        on: ReactionSource::Exit(Some(AssetId(5))),
        actions: vec![despawn(7)],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);
    let mut cursor = EventCursor::default();

    world.ctx().events_mut::<VolumeEvent>().send(VolumeEvent {
        volume: AssetId(5),
        entered: false,
    });
    sys.step(&mut world.ctx());
    assert_eq!(count::<DespawnRequest>(&mut world, &mut cursor), 1);
}

#[test]
fn crossings_survive_a_menu_pause() {
    let mut world = world_with(vec![Reaction {
        on: ReactionSource::Enter(Some(AssetId(5))),
        actions: vec![despawn(7)],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);
    let mut cursor = EventCursor::default();

    // The crossing lands while the menu is open: the paused step drains it
    // but holds it, and the first unpaused step fires it.
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
fn interact_source_fires_on_matching_press_only() {
    let mut world = world_with(vec![Reaction {
        on: ReactionSource::Interact(Some(AssetId(4))),
        actions: vec![despawn(7)],
        ..Default::default()
    }]);
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
fn show_and_hide_actions_send_visibility_requests() {
    let mut world = world_with(vec![Reaction {
        actions: vec![
            ReactionAction::Hide {
                target: Some(AssetId(3)),
            },
            ReactionAction::Show {
                target: Some(AssetId(4)),
            },
        ],
        ..Default::default()
    }]);
    let mut sys = system(&mut world);

    sys.tick(&mut world.ctx(), 0.0);
    let ctx = world.ctx();
    let mut cursor = EventCursor::default();
    let events = ctx.events::<VisibilityRequest>().unwrap();
    let reqs: Vec<&VisibilityRequest> = events.read(&mut cursor).into_iter().collect();
    assert_eq!(reqs.len(), 2);
    assert_eq!(reqs[0].name, AssetId(3));
    assert!(!reqs[0].visible);
    assert_eq!(reqs[1].name, AssetId(4));
    assert!(reqs[1].visible);
}

// A unique per-test save directory, cleaned before use.
fn save_dir(test: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("cn-logic-{}-{}", std::process::id(), test));
    std::fs::remove_dir_all(&dir).ok();
    dir
}

// An initialized system persisting into `dir`.
fn persisting_system(world: &mut TestWorld, dir: &std::path::Path) -> ReactionSystem {
    let mut sys = ReactionSystem::new();
    sys.save_dir = dir.to_path_buf();
    sys.init(&mut world.ctx());
    sys
}

fn counter_rule() -> Reaction {
    Reaction {
        asset_id: AssetId(1),
        actions: vec![set("visits", 1, true), ReactionAction::Save],
        once: true,
        ..Default::default()
    }
}

#[test]
fn save_action_persists_vars_and_fired_state_across_runs() {
    let dir = save_dir("roundtrip");

    let mut world = world_with(vec![counter_rule()]);
    let mut sys = persisting_system(&mut world, &dir);
    sys.tick(&mut world.ctx(), 0.0);
    assert_eq!(
        world.ctx().resource::<Variables>().unwrap().get("visits"),
        1
    );

    // A fresh run over the same world: the variable is restored and the
    // fired `once` rule stays fired.
    let mut world2 = world_with(vec![counter_rule()]);
    let mut sys2 = persisting_system(&mut world2, &dir);
    assert_eq!(
        world2.ctx().resource::<Variables>().unwrap().get("visits"),
        1,
        "variable restored at init"
    );
    sys2.tick(&mut world2.ctx(), 0.0);
    assert_eq!(
        world2.ctx().resource::<Variables>().unwrap().get("visits"),
        1,
        "the fired once rule does not fire again"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn restored_variable_is_not_an_edge_for_variable_sources() {
    let dir = save_dir("baseline");

    let mut world = world_with(vec![counter_rule()]);
    let mut sys = persisting_system(&mut world, &dir);
    sys.tick(&mut world.ctx(), 0.0);

    // Second run adds a watcher on the restored variable: restoring is not a
    // change, so it must not fire.
    let watcher = Reaction {
        asset_id: AssetId(2),
        on: ReactionSource::Variable("visits".into()),
        actions: vec![despawn(7)],
        ..Default::default()
    };
    let mut world2 = world_with(vec![counter_rule(), watcher]);
    let mut sys2 = persisting_system(&mut world2, &dir);
    let mut cursor = EventCursor::default();
    sys2.tick(&mut world2.ctx(), 0.0);
    assert_eq!(count::<DespawnRequest>(&mut world2, &mut cursor), 0);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn edited_rule_loses_its_persisted_fired_flag() {
    let dir = save_dir("edited");

    let mut world = world_with(vec![counter_rule()]);
    let mut sys = persisting_system(&mut world, &dir);
    sys.tick(&mut world.ctx(), 0.0);

    // Same asset id, different content: the flag no longer applies, so the
    // rule fires once more.
    let mut edited = counter_rule();
    edited.actions[0] = set("visits", 5, true);
    let mut world2 = world_with(vec![edited]);
    let mut sys2 = persisting_system(&mut world2, &dir);
    sys2.tick(&mut world2.ctx(), 0.0);
    assert_eq!(
        world2.ctx().resource::<Variables>().unwrap().get("visits"),
        6,
        "restored 1 + refired add 5"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn worlds_without_a_save_action_never_read_state() {
    let dir = save_dir("optin");

    let mut world = world_with(vec![counter_rule()]);
    let mut sys = persisting_system(&mut world, &dir);
    sys.tick(&mut world.ctx(), 0.0);

    // Same directory, but no rule saves: the world starts fresh.
    let mut world2 = world_with(vec![Reaction {
        actions: vec![set("other", 1, false)],
        ..Default::default()
    }]);
    let _sys2 = persisting_system(&mut world2, &dir);
    assert_eq!(
        world2.ctx().resource::<Variables>().unwrap().get("visits"),
        0
    );
    std::fs::remove_dir_all(&dir).ok();
}

// A declared Reaction gates the internal system on, and the menu freeze
// holds every firing until the menu closes.
#[test]
fn reaction_gates_system_and_menu_freezes_it() {
    let mut world = crate::ecs::World::new_empty();
    world.add_component(Reaction {
        actions: vec![despawn(7)],
        ..Default::default()
    });
    world.start().unwrap();
    let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
    assert_eq!(names, ["ReactionSystem"]);

    let mut cursor = EventCursor::default();
    world.insert_resource(crate::ecs::MenuActive(true));
    world.step();
    let fired = world
        .events::<DespawnRequest>()
        .map(|e| e.read(&mut cursor).len())
        .unwrap_or(0);
    assert_eq!(fired, 0, "a paused world fires nothing");

    world.insert_resource(crate::ecs::MenuActive(false));
    world.step();
    let fired = world
        .events::<DespawnRequest>()
        .expect("the reaction fired after unpause")
        .read(&mut cursor)
        .len();
    assert_eq!(fired, 1);
}
