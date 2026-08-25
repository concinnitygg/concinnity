// Build-time count of a world's physics content, shipped in the blob so the
// runtime reserves exactly what the world needs and refuses to exceed it.
//
// The counting rules mirror what the driver actually builds at init: a Prop
// only becomes a body if it carries a collider, a joint only wires if both its
// ends resolve to one, and only the first declared camera can own the player
// capsule. Where the two sides disagree the runtime's debug assert trips, so
// this module tracks the driver, not the authoring schema.

use std::collections::HashSet;

use concinnity_core::blob::PhysicsBudgetRecord;
use concinnity_physics::{PhysicsBudget, PhysicsCounts};
use concinnity_world::world::WorldJsonlAsset;

use crate::spawn_population::SpawnPopulation;

fn norm(asset_type: &str) -> String {
    asset_type.to_lowercase().replace('_', "")
}

// A named asset reference arg: a non-empty string naming another asset.
fn name_arg<'a>(asset: &'a WorldJsonlAsset, field: &str) -> Option<&'a str> {
    asset
        .args
        .get(field)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
}

// Whether the world runs physics at all, mirroring the schedule's gate: a
// PhysicsConfig, a RigidBody, a PropBody, a TriggerVolume, or a SkinnedMesh
// that declared a character capsule.
fn has_physics_content(assets: &[WorldJsonlAsset]) -> bool {
    assets.iter().any(|a| match norm(&a.asset_type).as_str() {
        "physicsconfig" | "rigidbody" | "propbody" | "triggervolume" => true,
        "skinnedmesh" => has_capsule(a),
        _ => false,
    })
}

fn has_capsule(asset: &WorldJsonlAsset) -> bool {
    asset.args.get("capsule").is_some_and(|c| !c.is_null())
}

// Names of the Props that carry a collider, i.e. the ones the load-time
// decomposition gives a `Collider` component and the driver builds a body for.
fn collider_props(assets: &[WorldJsonlAsset]) -> HashSet<&str> {
    assets
        .iter()
        .filter(|a| norm(&a.asset_type) == "prop")
        .filter(|a| a.args.get("collider").is_some_and(|c| !c.is_null()))
        .map(|a| a.name.as_str())
        .collect()
}

// Count the physics content of an expanded world. `None` when the world runs
// no physics at all.
pub(crate) fn count(assets: &[WorldJsonlAsset]) -> Option<PhysicsCounts> {
    if !has_physics_content(assets) {
        return None;
    }

    let colliders = collider_props(assets);

    // A PropBody makes its target simulate freely. Several bodies naming one
    // prop still make one dynamic body, and one naming a collider-less prop
    // makes none.
    let dynamic: HashSet<&str> = assets
        .iter()
        .filter(|a| norm(&a.asset_type) == "propbody")
        .filter_map(|a| name_arg(a, "prop_name"))
        .filter(|name| colliders.contains(name))
        .collect();

    let mut counts = PhysicsCounts {
        static_colliders: (colliders.len() - dynamic.len()) as u32,
        dynamic_colliders: dynamic.len() as u32,
        ..PhysicsCounts::default()
    };

    for asset in assets {
        match norm(&asset.asset_type).as_str() {
            "triggervolume" => counts.trigger_volumes += 1,
            "skinnedmesh" if has_capsule(asset) => counts.rig_capsules += 1,
            "physicsjoint" => {
                // The driver skips a joint whose ends do not both resolve to a
                // body; an unwirable one costs nothing.
                if !name_arg(asset, "body_a").is_some_and(|a| colliders.contains(a)) {
                    continue;
                }
                match name_arg(asset, "body_b") {
                    Some(b) if colliders.contains(b) => counts.joints += 1,
                    Some(_) => continue,
                    // No second body: the joint mints a hidden static anchor.
                    None => {
                        counts.joints += 1;
                        counts.world_anchored_joints += 1;
                    }
                }
            }
            _ => {}
        }
    }

    // Only the first declared camera can own the player capsule, and only when
    // it is not a third-person orbit around a followed character.
    let first_camera = assets.iter().find(|a| norm(&a.asset_type) == "camera3d");
    let follows = |camera: &WorldJsonlAsset| {
        camera
            .args
            .get("controller")
            .and_then(|c| c.get("follow"))
            .is_some_and(|f| !f.is_null())
    };
    if first_camera.is_some_and(|camera| !follows(camera)) {
        counts.player_capsules = 1;
    }

    Some(counts)
}

// The world's authored spawn headroom, from its PhysicsConfig.
fn spawn_headroom(assets: &[WorldJsonlAsset]) -> u32 {
    assets
        .iter()
        .find(|a| norm(&a.asset_type) == "physicsconfig")
        .and_then(|a| a.args.get("spawn_headroom"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
        .min(u64::from(u32::MAX)) as u32
}

/// Derive the world's physics reservation for the blob. `None` when the world
/// runs no physics.
pub(crate) fn compute(assets: &[WorldJsonlAsset]) -> Option<PhysicsBudgetRecord> {
    let counts = count(assets)?;
    // A floor, not an override: the authored value still wins when it is the
    // larger of the two, and the authored PhysicsConfig is never rewritten.
    let headroom = spawn_headroom(assets).max(spawn_reservation(assets).floor);
    Some(record(&PhysicsBudget::derive(&counts, headroom)))
}

// The shipped record for a derived budget. The runtime reads it back through
// the inverse conversion in the engine's physics driver.
fn record(budget: &PhysicsBudget) -> PhysicsBudgetRecord {
    PhysicsBudgetRecord {
        fixed: budget.fixed,
        dynamic: budget.dynamic,
        kinematic: budget.kinematic,
        sensors: budget.sensors,
        joints: budget.joints,
        anchors: budget.anchors,
        spawn_headroom: budget.spawn_headroom,
    }
}

// What a world's runtime spawn sources cost: the bodies the ones with a
// computable population need at once, and a line describing each source whose
// population cannot be bounded from the authored world.
#[derive(Debug, Default, PartialEq)]
struct SpawnReservation {
    floor: u32,
    counted: Vec<(String, u32)>,
    unbounded: Vec<String>,
}

fn spawn_reservation(assets: &[WorldJsonlAsset]) -> SpawnReservation {
    let sources = concinnity_world::check::physics::collider_spawn_sources(assets);
    let mut out = SpawnReservation::default();
    for spawner in &sources.spawners {
        match crate::spawn_population::population(spawner.interval, spawner.lifetime) {
            SpawnPopulation::Inert => {}
            SpawnPopulation::Bounded(bodies) => {
                out.floor = out.floor.saturating_add(bodies);
                out.counted.push((spawner.name.to_string(), bodies));
            }
            SpawnPopulation::Unbounded => out.unbounded.push(format!(
                "Spawner '{}' never removes its copies (lifetime {}), so its population is unbounded",
                spawner.name, spawner.lifetime
            )),
        }
    }
    for behavior in &sources.behaviors {
        out.unbounded.push(format!(
            "Behavior '{behavior}' spawns from a `spawn` node, which has no cadence to count"
        ));
    }
    out
}

/// Report what a world reserves for the props it creates while it runs: the
/// bodies derived from each spawner whose cadence bounds its population, and a
/// warning for every source left uncovered, whose spawns past the authored
/// bodies are refused a body at runtime.
///
/// The warnings are keyed on the authored headroom, not the effective one: a
/// derived floor is earmarked for the spawners it was counted from and covers
/// nothing else, so only a hand-written value says the author accounted for
/// the rest. A warning, not an error -- a world may deliberately spawn only
/// props it never simulates.
pub(crate) fn report_spawn_reservation(assets: &[WorldJsonlAsset]) {
    if !has_physics_content(assets) {
        return;
    }
    let reservation = spawn_reservation(assets);
    let authored = spawn_headroom(assets);
    if reservation.floor > authored {
        let per_spawner: Vec<String> = reservation
            .counted
            .iter()
            .map(|(name, bodies)| format!("{name} ({bodies})"))
            .collect();
        tracing::info!(
            "PhysicsConfig: reserving {} spawn bodies for {}; authored spawn_headroom is {}",
            reservation.floor,
            per_spawner.join(", "),
            authored
        );
    }
    if authored > 0 {
        return;
    }
    for source in &reservation.unbounded {
        tracing::warn!(
            "PhysicsConfig: {source}; nothing is reserved for it, so every spawn past the \
             authored bodies is refused a body. Set spawn_headroom to cover it."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(name: &str, asset_type: &str, args: serde_json::Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            name: name.to_string(),
            asset_type: asset_type.to_string(),
            args,
        }
    }

    fn prop(name: &str, collider: bool) -> WorldJsonlAsset {
        let args = if collider {
            serde_json::json!({"mesh": "m", "collider": {"shape": "ball", "radius": 0.5}})
        } else {
            serde_json::json!({"mesh": "m"})
        };
        asset(name, "Prop", args)
    }

    fn prop_body(name: &str, target: &str) -> WorldJsonlAsset {
        asset(name, "PropBody", serde_json::json!({"prop_name": target}))
    }

    fn physics_config(args: serde_json::Value) -> WorldJsonlAsset {
        asset("physics", "PhysicsConfig", args)
    }

    #[test]
    fn a_world_with_no_physics_reserves_nothing() {
        let assets = [
            prop("banner", false),
            asset("cam", "Camera3D", json_empty()),
        ];
        assert_eq!(count(&assets), None);
        assert_eq!(compute(&assets), None);
    }

    fn json_empty() -> serde_json::Value {
        serde_json::json!({})
    }

    #[test]
    fn a_prop_body_makes_its_collider_prop_dynamic() {
        let assets = [
            prop("wall", true),
            prop("crate_a", true),
            prop("banner", false),
            prop_body("crate_body", "crate_a"),
        ];
        let counts = count(&assets).expect("PropBody turns physics on");
        assert_eq!(counts.static_colliders, 1, "the wall");
        assert_eq!(counts.dynamic_colliders, 1, "the crate");
    }

    // A PropBody naming a Prop with no collider gives the runtime nothing to
    // build, so it must not add a body to the reservation either.
    #[test]
    fn a_prop_body_without_a_collider_counts_nowhere() {
        let assets = [prop("banner", false), prop_body("banner_body", "banner")];
        let counts = count(&assets).expect("PropBody turns physics on");
        assert_eq!(counts.static_colliders, 0);
        assert_eq!(counts.dynamic_colliders, 0);

        let record = compute(&assets).expect("a budget");
        assert_eq!(record.fixed, 1, "only the floor");
        assert_eq!(record.dynamic, 0);
    }

    #[test]
    fn trigger_volumes_and_rig_capsules_are_counted_per_asset() {
        let assets = [
            asset("gate", "TriggerVolume", json_empty()),
            asset("porch", "TriggerVolume", json_empty()),
            asset(
                "hero",
                "SkinnedMesh",
                serde_json::json!({"capsule": {"half_height": 0.9, "radius": 0.3}}),
            ),
            asset("banner_mesh", "SkinnedMesh", json_empty()),
        ];
        let counts = count(&assets).expect("trigger volumes turn physics on");
        assert_eq!(counts.trigger_volumes, 2);
        assert_eq!(counts.rig_capsules, 1, "only the mesh with a capsule");
    }

    #[test]
    fn joints_count_only_when_both_ends_resolve_to_a_body() {
        let assets = [
            physics_config(json_empty()),
            prop("post", true),
            prop("gate", true),
            prop("banner", false),
            asset(
                "hinge",
                "PhysicsJoint",
                serde_json::json!({"body_a": "post", "body_b": "gate"}),
            ),
            asset(
                "rope",
                "PhysicsJoint",
                serde_json::json!({"body_a": "post", "anchor_b": [0, 4, 0]}),
            ),
            asset(
                "loose",
                "PhysicsJoint",
                serde_json::json!({"body_a": "post", "body_b": "banner"}),
            ),
            asset(
                "unanchored",
                "PhysicsJoint",
                serde_json::json!({"body_b": "gate"}),
            ),
        ];
        let counts = count(&assets).expect("PhysicsConfig turns physics on");
        assert_eq!(counts.joints, 2, "the hinge and the world-anchored rope");
        assert_eq!(counts.world_anchored_joints, 1);

        let record = compute(&assets).expect("a budget");
        assert_eq!(record.joints, 2);
        assert_eq!(record.anchors, 1, "the rope mints a hidden static body");
    }

    // The driver looks at the first camera only, so a later first-person
    // camera behind a third-person one gets no capsule.
    #[test]
    fn only_the_first_camera_can_own_the_player_capsule() {
        let follow = serde_json::json!({"controller": {"follow": {"target": "hero"}}});
        let assets = [
            physics_config(json_empty()),
            asset("orbit", "Camera3D", follow),
            asset("spectator", "Camera3D", json_empty()),
        ];
        assert_eq!(count(&assets).expect("physics").player_capsules, 0);

        // First-person first: the capsule is built.
        let assets = [
            physics_config(json_empty()),
            asset("spectator", "Camera3D", json_empty()),
        ];
        assert_eq!(count(&assets).expect("physics").player_capsules, 1);

        // An uncontrolled camera is still collided as a flying capsule.
        let assets = [
            physics_config(json_empty()),
            asset(
                "cutscene",
                "Camera3D",
                serde_json::json!({"controller": null}),
            ),
        ];
        assert_eq!(count(&assets).expect("physics").player_capsules, 1);
    }

    // The spawner in `spawning`: one copy every 2s, each living 4s, so at most
    // ceil(4 / 2) + 1 = 3 are alive at once.
    const BOUNDED_FLOOR: u32 = 3;

    fn spawner(name: &str, template: &str, cadence: serde_json::Value) -> WorldJsonlAsset {
        let mut args = serde_json::json!({"template": template});
        let (serde_json::Value::Object(args_map), serde_json::Value::Object(cadence)) =
            (&mut args, cadence)
        else {
            unreachable!("both are objects");
        };
        args_map.extend(cadence);
        asset(name, "Spawner", args)
    }

    fn bounded_world(headroom: serde_json::Value) -> Vec<WorldJsonlAsset> {
        vec![
            physics_config(serde_json::json!({"spawn_headroom": headroom})),
            prop("crate_a", true),
            spawner(
                "drop",
                "crate_a",
                serde_json::json!({"interval": 2.0, "lifetime": 4.0}),
            ),
        ]
    }

    #[test]
    fn a_bounded_spawner_reserves_its_steady_state_population() {
        let assets = bounded_world(serde_json::json!(0));
        let reservation = spawn_reservation(&assets);
        assert_eq!(reservation.floor, BOUNDED_FLOOR);
        assert_eq!(
            reservation.counted,
            vec![("drop".to_string(), BOUNDED_FLOOR)]
        );
        assert!(reservation.unbounded.is_empty(), "the cadence bounds it");

        let record = compute(&assets).expect("a budget");
        assert_eq!(record.spawn_headroom, BOUNDED_FLOOR);
    }

    // The trap: `lifetime: 0` means the copies live forever, so the naive
    // lifetime / interval would silently reserve nothing for the spawner that
    // most needs a reservation.
    #[test]
    fn a_forever_spawner_reserves_nothing_and_warns() {
        let assets = [
            physics_config(json_empty()),
            prop("crate_a", true),
            spawner("drop", "crate_a", serde_json::json!({"lifetime": 0.0})),
        ];
        let reservation = spawn_reservation(&assets);
        assert_eq!(reservation.floor, 0, "an unbounded spawner has no floor");
        assert!(reservation.counted.is_empty(), "not counted as zero");
        assert_eq!(reservation.unbounded.len(), 1);
        assert!(
            reservation.unbounded[0].contains("never removes its copies"),
            "got: {}",
            reservation.unbounded[0]
        );

        // An unauthored cadence is the same case: the SpawnerArgs default
        // lifetime is 0.
        let bare = [
            physics_config(json_empty()),
            prop("crate_a", true),
            asset(
                "drop",
                "Spawner",
                serde_json::json!({"template": "crate_a"}),
            ),
        ];
        assert_eq!(spawn_reservation(&bare).unbounded.len(), 1);
    }

    #[test]
    fn a_spawner_that_can_never_fire_reserves_nothing_and_stays_quiet() {
        for interval in [serde_json::json!(0.0), serde_json::json!(-1.0)] {
            let assets = [
                physics_config(json_empty()),
                prop("crate_a", true),
                spawner(
                    "drop",
                    "crate_a",
                    serde_json::json!({"interval": interval, "lifetime": 4.0}),
                ),
            ];
            let reservation = spawn_reservation(&assets);
            assert_eq!(reservation, SpawnReservation::default(), "{interval}");
        }
    }

    #[test]
    fn a_spawner_over_a_collider_less_prop_reserves_nothing() {
        let assets = [
            physics_config(json_empty()),
            prop("banner", false),
            spawner(
                "drop",
                "banner",
                serde_json::json!({"interval": 2.0, "lifetime": 4.0}),
            ),
        ];
        assert_eq!(spawn_reservation(&assets), SpawnReservation::default());
        assert_eq!(compute(&assets).expect("a budget").spawn_headroom, 0);
    }

    #[test]
    fn several_bounded_spawners_sum() {
        let assets = [
            physics_config(json_empty()),
            prop("crate_a", true),
            prop("crate_b", true),
            spawner(
                "drop",
                "crate_a",
                serde_json::json!({"interval": 2.0, "lifetime": 4.0}),
            ),
            spawner(
                "sprinkle",
                "crate_b",
                serde_json::json!({"interval": 0.5, "lifetime": 1.0}),
            ),
        ];
        // 3 for the first, ceil(1 / 0.5) + 1 = 3 for the second.
        assert_eq!(spawn_reservation(&assets).floor, 6);
        assert_eq!(compute(&assets).expect("a budget").spawn_headroom, 6);
    }

    // A floor, not an override: a larger authored value wins, and the config
    // it came from is never rewritten.
    #[test]
    fn an_authored_headroom_above_the_floor_wins_and_the_config_is_untouched() {
        let assets = bounded_world(serde_json::json!(8));
        assert_eq!(compute(&assets).expect("a budget").spawn_headroom, 8);

        let before: Vec<serde_json::Value> = assets.iter().map(|a| a.args.clone()).collect();
        let _ = compute(&assets);
        report_spawn_reservation(&assets);
        let after: Vec<serde_json::Value> = assets.iter().map(|a| a.args.clone()).collect();
        assert_eq!(after, before, "the authored config is never rewritten");
        assert_eq!(
            assets[0].args["spawn_headroom"],
            serde_json::json!(8),
            "the author still reads back what they wrote"
        );
    }

    // Counts the warn-level events `f` emits. A live subscriber is required:
    // tracing skips the whole call when nothing is listening.
    fn count_warnings(f: impl FnOnce()) -> usize {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Counter(std::sync::Arc<AtomicUsize>);
        impl tracing::Subscriber for Counter {
            fn enabled(&self, meta: &tracing::Metadata<'_>) -> bool {
                *meta.level() == tracing::Level::WARN
            }
            fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
                tracing::span::Id::from_u64(1)
            }
            fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
            fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
            fn event(&self, _: &tracing::Event<'_>) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
            fn enter(&self, _: &tracing::span::Id) {}
            fn exit(&self, _: &tracing::span::Id) {}
        }
        let count = std::sync::Arc::new(AtomicUsize::new(0));
        tracing::subscriber::with_default(Counter(count.clone()), f);
        count.load(Ordering::Relaxed)
    }

    // A bounded spawner is reserved for, so it leaves nothing to warn about;
    // a behavior spawn node has no cadence to count and still warns.
    #[test]
    fn a_reserved_world_builds_clean_and_an_uncountable_source_still_warns() {
        let bounded = bounded_world(serde_json::json!(0));
        assert_eq!(count_warnings(|| report_spawn_reservation(&bounded)), 0);

        let mut with_behavior = bounded.clone();
        with_behavior.push(asset(
            "thrower",
            "Behavior",
            serde_json::json!({"on": "tick", "do": [{"spawn": {"template": "crate_a"}}]}),
        ));
        assert_eq!(
            count_warnings(|| report_spawn_reservation(&with_behavior)),
            1,
            "the behavior node is the only uncountable source"
        );

        // An authored headroom says the author accounted for the rest.
        let mut covered = bounded_world(serde_json::json!(8));
        covered.push(with_behavior.pop().expect("the behavior"));
        assert_eq!(count_warnings(|| report_spawn_reservation(&covered)), 0);
    }

    // The warning is about sources cook could not size, so a world whose only
    // spawn source is a bounded spawner has nothing left to warn about.
    #[test]
    fn only_uncomputable_sources_are_left_to_warn_about() {
        assert!(
            spawn_reservation(&bounded_world(serde_json::json!(0)))
                .unbounded
                .is_empty(),
            "a bounded spawner is handled, not warned about"
        );

        // A behavior spawn node has no cadence to count, so it still warns
        // even alongside a bounded spawner that was reserved for.
        let mut mixed = bounded_world(serde_json::json!(0));
        mixed.push(asset(
            "thrower",
            "Behavior",
            serde_json::json!({"on": "tick", "do": [{"spawn": {"template": "crate_a"}}]}),
        ));
        let reservation = spawn_reservation(&mixed);
        assert_eq!(
            reservation.floor, BOUNDED_FLOOR,
            "the spawner is still counted"
        );
        assert_eq!(reservation.unbounded.len(), 1);
        assert!(
            reservation.unbounded[0].contains("Behavior 'thrower'"),
            "got: {}",
            reservation.unbounded[0]
        );
    }

    #[test]
    fn the_record_carries_the_authored_headroom() {
        let assets = [
            physics_config(serde_json::json!({"spawn_headroom": 32})),
            prop("wall", true),
        ];
        let record = compute(&assets).expect("a budget");
        assert_eq!(record.spawn_headroom, 32);
        assert_eq!(record.fixed, 2, "the wall plus the floor");

        // The record is exactly what the shared derivation produces, which is
        // the property the runtime asserts against its own scan.
        let counts = count(&assets).expect("physics");
        assert_eq!(record, super::record(&PhysicsBudget::derive(&counts, 32)));
    }

    // Cook injects a PhysicsConfig into a world with physics content that
    // declares none, and the budget is computed from that same expanded list.
    // The injected config carries the engine defaults, so it must move nothing.
    #[test]
    fn the_injected_config_leaves_the_budget_and_the_warning_alone() {
        let world = concat!(
            r#"{"name":"box","type":"ProceduralMesh","args":{"generator":"box","half_extents":[1,1,1]}}"#,
            "\n",
            r#"{"name":"crate_a","type":"Prop","args":{"mesh":"box","collider":{"shape":"cuboid"}}}"#,
            "\n",
            r#"{"name":"crate_body","type":"PropBody","args":{"prop_name":"crate_a"}}"#,
            "\n",
            r#"{"name":"drop","type":"Spawner","args":{"template":"crate_a"}}"#,
            "\n",
        );
        let expanded = crate::world::prepare_world(world).expect("prepare").assets;
        assert!(
            expanded
                .iter()
                .any(|a| norm(&a.asset_type) == "physicsconfig"),
            "the expanded world carries the injected config"
        );

        // The same world minus the injected config: what the budget was
        // derived from before this default existed.
        let without: Vec<WorldJsonlAsset> = expanded
            .iter()
            .filter(|a| norm(&a.asset_type) != "physicsconfig")
            .cloned()
            .collect();
        assert_eq!(count(&expanded), count(&without));
        assert_eq!(compute(&expanded), compute(&without));
        assert_eq!(spawn_headroom(&expanded), 0);

        // The spawner leaves its copies forever, so nothing can be reserved
        // for it and a visible config does not make that go away.
        assert_eq!(spawn_reservation(&expanded).unbounded.len(), 1);
    }
}
