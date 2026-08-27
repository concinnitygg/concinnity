// What only a started world in this crate can show. The system's own tick
// semantics are covered where the system lives (concinnity-core); these cover
// the wiring: that a scope survives the load-time decomposition pass, and that
// the state a `save` node writes reaches the file store and comes back.

use crate::components::{Behavior, BehaviorExpr, BehaviorNode, BehaviorSource, Prop, PropInstance};
use crate::components::{Transform, Variables};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{MeshHandle, SYSTEMS, World};

// The core tests drive a bare component storage, where decomposition never
// runs, so they cannot show that a scope survives it. This goes through
// `World::start`, where the authored Prop column is drained, and covers both
// halves of the ModelRenderer-xor-MeshRenderer split a Prop decomposes into.
#[test]
fn a_prop_scoped_behavior_fires_once_started() {
    let mut world = World::new();
    world.add_component(Prop {
        asset_id: AssetId(1),
        mesh: Some(MeshHandle(10)),
        scale: [1.0; 3],
        ..Default::default()
    });
    world.add_component(Prop {
        asset_id: AssetId(2),
        model: Some(AssetId(20)),
        position: [10.0, 0.0, 0.0],
        scale: [1.0; 3],
        ..Default::default()
    });
    world.add_component(Behavior {
        on: BehaviorSource::Tick,
        scope: vec!["Prop".into()],
        body: vec![BehaviorNode::SetTransform {
            entity: BehaviorExpr::SelfEntity,
            position: Some(BehaviorExpr::Add(
                Box::new(BehaviorExpr::Position(Box::new(BehaviorExpr::SelfEntity))),
                Box::new(BehaviorExpr::Vec3([0.0, 1.0, 0.0])),
            )),
            rotation_deg: None,
            scale: None,
        }],
        ..Default::default()
    });

    world.start(SYSTEMS).unwrap();
    assert_eq!(
        world.query::<Prop>().count(),
        0,
        "decomposition drained the authored column",
    );
    world.step();

    let mut lifted: Vec<f32> = world
        .join2::<PropInstance, Transform>()
        .map(|(_, _, t)| t.position[1])
        .collect();
    lifted.sort_by(f32::total_cmp);
    assert_eq!(
        lifted,
        vec![1.0, 1.0],
        "both the mesh- and the model-backed prop ran the behavior",
    );
}

// The store this host attaches is reached by a `save` node and read back at the
// next start: the seam between the system and the file is live, not just
// constructible.
#[test]
fn a_saving_world_restores_its_variable_through_the_file_store() {
    use super::save::FileStore;
    use crate::components::{BehaviorLiteral, VariableDecl};
    use crate::ecs::System;
    use concinnity_core::behavior::{BehaviorStore, BehaviorSystem};

    let dir = std::env::temp_dir().join(format!("cn-behavior-wired-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();

    let saver = || Behavior {
        asset_id: AssetId(1),
        on: BehaviorSource::Start,
        body: vec![
            BehaviorNode::Set {
                var: "visits".into(),
                value: BehaviorExpr::Int(1),
                add: true,
            },
            BehaviorNode::Save,
        ],
        ..Default::default()
    };
    let declared = || Variables {
        vars: vec![VariableDecl {
            name: "visits".into(),
            value: BehaviorLiteral::Int(0),
        }],
        ..Default::default()
    };

    // One run: the `save` node's write reaches the file.
    let run = |expected: i32, what: &str| {
        let mut world = World::new();
        world.add_component(saver());
        world.add_component(declared());
        let mut system = BehaviorSystem::new().with_store(Box::new(FileStore::at(&dir)));
        system.init(&mut world.context());
        system.step(&mut world.context());

        let state = FileStore::at(&dir)
            .read()
            .expect("the save node wrote through the store");
        assert_eq!(
            state.vars.get("visits"),
            Some(&BehaviorLiteral::Int(expected)),
            "{what}"
        );
    };

    run(1, "the tick's value reached the file");
    run(2, "the second run started from what the first stored");
    std::fs::remove_dir_all(&dir).ok();
}
